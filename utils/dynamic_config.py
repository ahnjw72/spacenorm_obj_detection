"""dynamic_config.py

Translates the vision_nodes API response (the "dynamic" sensors & CCTV
configuration method) into the same cctv_ID[key]-shaped dicts that the
"static" method builds in spacenorm_obj_detection.init_cctv_data(), so the
rest of the detection pipeline (detector_per_cam(), RTSP_Camera,
remove_outside_ROI(), report_via_mqtt()) is reused unchanged.
"""
import os
import json
import logging
import time
import urllib.parse

import cv2
import requests

from .vision_node_api import VisionNodeAPI
from .post_processing import draw_roi_overlay
from .metrics import dynamic_config_last_fetch_timestamp, dynamic_config_sensors_active

logger = logging.getLogger(__name__)

# Temporary hardcoded RTSP credentials (requirement 6): the API does not yet
# return per-CCTV ID/PASSWORD, so every dynamic-method CCTV uses the same
# credentials until a dedicated API is available.
DEFAULT_RTSP_USERNAME = 'space'
DEFAULT_RTSP_PASSWORD = 'spacenorm12!@#'

# cctv_ID fields that affect a running detector thread's behavior; a change
# in any of these (see diff_configs()) means the entry counts as "changed".
_CHANGE_TRACKED_FIELDS = ('ROI', 'min_obj_size_ratio', 'mqtt_broker_addr', 'mqtt_broker_port', 'uri')

# Of _CHANGE_TRACKED_FIELDS, only these are fixed at thread-start time (the
# RTSP connection / MQTT client are created once in detector_per_cam()), so
# only a change in one of these requires stopping and respawning the
# detector thread. ROI / min_obj_size_ratio changes are picked up for free
# since detector_per_cam() re-reads cctv_ID[key] every loop iteration.
_RESTART_REQUIRED_FIELDS = ('uri', 'mqtt_broker_addr', 'mqtt_broker_port')


def inject_rtsp_credentials(url, username=DEFAULT_RTSP_USERNAME, password=DEFAULT_RTSP_PASSWORD):
    """Embed hardcoded RTSP credentials into a credential-less private_url.

    Most RTSP backends (and the FFmpeg/GStreamer paths used by
    open_rtsp_universal() in utils/cctv_camera.py) only use embedded userinfo
    if the server issues an auth challenge, so this should be harmless for
    cameras that don't require authentication -- but that assumption should
    be confirmed against this project's actual camera fleet during testing.
    """
    parsed = urllib.parse.urlsplit(url)
    if parsed.username: # URL already carries credentials -- do not override
        return url

    netloc = parsed.hostname or ''
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    userinfo = f"{urllib.parse.quote(username, safe='')}:{urllib.parse.quote(password, safe='')}"
    netloc = f"{userinfo}@{netloc}"

    return urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def build_cctv_entry(company_name, sensor, cctv, access_token):
    """Build one cctv_ID[key]-shaped dict from a (sensor, cctv) pair.

    key is constructed as "<company_name>_<desc>" (requirement 9).
    """
    key = f"{company_name}_{cctv['desc']}"

    entry = {
        'device_id': sensor['device_id'],
        'company_name': company_name, # needed for the spacenorm_camera_info metric
        'uri': inject_rtsp_credentials(cctv['private_url']),
        'access_token': access_token,
        'refresh_token': None, # unused by Gateway.__init__ -- see utils/gateway_api.py
        'snapshot_url': cctv.get('snapshot_url'),
    }

    roi_vertices_raw = cctv.get('roi_vertices')
    if roi_vertices_raw:
        # API vertices are ratios in [0,1], already sorted counter-clockwise
        # (requirement 4), so img_w=img_h=1 makes remove_outside_ROI()'s
        # pixel-normalization a no-op and vertices_sorted=1 skips
        # make_counterclockwise().
        entry['ROI'] = {
            'img_w': 1,
            'img_h': 1,
            'vertices': json.loads(roi_vertices_raw),
            'vertices_sorted': 1,
        }

    min_obj_size_ratio = cctv.get('min_obj_size_ratio')
    if min_obj_size_ratio is not None:
        entry['min_obj_size_ratio'] = float(min_obj_size_ratio)

    mqtt_broker_addr = sensor.get('mqtt_broker_addr')
    mqtt_broker_port = sensor.get('mqtt_broker_port')
    if mqtt_broker_addr and mqtt_broker_port:
        entry['mqtt_broker_addr'] = mqtt_broker_addr
        entry['mqtt_broker_port'] = mqtt_broker_port

    return key, entry


def fetch_dynamic_config(cfg):
    """Fetch and parse the dynamic sensors & CCTV configuration for the site
    identified by cfg.access_token / cfg.vision_node_id.

    Returns a dict[key -> cctv_entry] on success, or None (logged) on any
    fetch/parse failure -- callers should treat None as "retry next period".
    """
    api = VisionNodeAPI(cfg.access_token)
    data = api.fetch_sync_data(cfg.vision_node_id)
    if data is None:
        return None

    try:
        company_name = data['company_name']
        sensors = data.get('sensors', [])
        cctvs_by_id = {cctv['id']: cctv for cctv in data.get('cctvs', [])}
        # security_regions is deliberately ignored (requirement 10).

        test_device_ids = getattr(cfg, 'test_device_ids', None)

        result = {}
        for sensor in sensors:
            if sensor.get('vision_node_id') != cfg.vision_node_id:
                continue # belongs to a different vision node in a multi-id response

            if test_device_ids and sensor['device_id'] not in test_device_ids:
                continue

            cctv = cctvs_by_id.get(sensor['cctv_id'])
            if cctv is None:
                logger.warning(f"sensor {sensor.get('sensor_id')} references unknown cctv_id {sensor.get('cctv_id')} -- skipping")
                continue

            key, entry = build_cctv_entry(company_name, sensor, cctv, cfg.access_token)
            result[key] = entry

        dynamic_config_last_fetch_timestamp.set(time.time())
        dynamic_config_sensors_active.set(len(result))
        return result
    except (KeyError, TypeError, ValueError) as e:
        logger.error(f"Failed to parse vision_nodes sync_data response: {e}")
        return None


def diff_configs(old, new):
    """Compare two {key: cctv_entry} maps.

    Returns (added_keys, removed_keys, changed_keys) where "changed" means
    one of the detection-affecting fields (_CHANGE_TRACKED_FIELDS) differs
    from the previous fetch.
    """
    old_keys = set(old.keys())
    new_keys = set(new.keys())

    added = new_keys - old_keys
    removed = old_keys - new_keys

    changed = set()
    for key in (old_keys & new_keys):
        old_entry = old[key]
        new_entry = new[key]
        if any(old_entry.get(field) != new_entry.get(field) for field in _CHANGE_TRACKED_FIELDS):
            changed.add(key)

    return added, removed, changed


def requires_thread_restart(old_entry, new_entry):
    """Whether moving from old_entry to new_entry requires stopping and
    respawning the detector thread (vs. an in-place cctv_ID[key] update)."""
    return any(old_entry.get(field) != new_entry.get(field) for field in _RESTART_REQUIRED_FIELDS)


def save_and_upload_snapshot(key, cam, cctv_entry, snapshot_dir):
    """Write a sample image (with the ROI drawn, if defined) for human ROI
    verification, and upload it to cctv_entry['snapshot_url'] if set.
    """
    img = cam.read()
    if img is None:
        logger.warning(f"[{key}] could not read a frame for ROI snapshot -- skipping")
        return

    roi = cctv_entry.get('ROI')
    if roi:
        img = draw_roi_overlay(img, roi)

    os.makedirs(snapshot_dir, exist_ok=True)
    snapshot_path = os.path.join(snapshot_dir, f"{key}.jpg")
    cv2.imwrite(snapshot_path, img)
    logger.info(f"[{key}] ROI snapshot written to {snapshot_path}")

    snapshot_url = cctv_entry.get('snapshot_url')
    if not snapshot_url:
        logger.warning(f"[{key}] snapshot_url is not set -- skipping snapshot upload")
        return

    try:
        with open(snapshot_path, 'rb') as f:
            r = requests.put(snapshot_url, data=f, timeout=10)
            r.raise_for_status()
        logger.info(f"[{key}] snapshot uploaded to {snapshot_url}")
    except requests.exceptions.RequestException as e:
        logger.error(f"[{key}] failed to upload snapshot to {snapshot_url}: {e}")
