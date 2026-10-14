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

# Temporary hardcoded RTSP credential candidates (requirement 6): the API
# does not yet return per-CCTV ID/PASSWORD, so a dynamic-method CCTV whose
# private_url carries no credentials is probed against this known
# fleet-wide set (2 usernames x 2 passwords) until a per-camera credential
# API is available. Order tries the previous single hardcoded default
# first, so already-working cameras resolve on the first probe.
RTSP_CREDENTIAL_CANDIDATES = [
    ('space', 'spacenorm12!@#'),
    ('admin', 'spacenorm12!@#'),
    ('space', 'Qwert12#'),
    ('admin', 'Qwert12#'),
]

# Bound on each individual probe connection attempt -- keeps a fully
# unreachable camera from stalling the whole dynamic-config refresh cycle
# for 4x the default (much longer) FFmpeg RTSP connect timeout.
RTSP_PROBE_TIMEOUT_MS = 5000

# key -> (username, password) resolved once per process lifetime. Avoids
# re-probing (and re-authenticating against the NVR) every refresh cycle
# once a working combination is found; a camera not yet in this cache is
# retried on the next refresh.
_resolved_rtsp_credentials = {}

# cctv_ID fields that affect a running detector thread's behavior; a change
# in any of these (see diff_configs()) means the entry counts as "changed".
_CHANGE_TRACKED_FIELDS = ('ROI', 'min_obj_size_ratio', 'mqtt_broker_addr', 'mqtt_broker_port', 'uri')

# Of _CHANGE_TRACKED_FIELDS, only these are fixed at thread-start time (the
# RTSP connection / MQTT client are created once in detector_per_cam()), so
# only a change in one of these requires stopping and respawning the
# detector thread. ROI / min_obj_size_ratio changes are picked up for free
# since detector_per_cam() re-reads cctv_ID[key] every loop iteration.
_RESTART_REQUIRED_FIELDS = ('uri', 'mqtt_broker_addr', 'mqtt_broker_port')


def inject_rtsp_credentials(url, username, password):
    """Embed the given RTSP credentials into a credential-less private_url.

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


def resolve_rtsp_uri(key, private_url, candidates=RTSP_CREDENTIAL_CANDIDATES, timeout_ms=RTSP_PROBE_TIMEOUT_MS):
    """Return a connect-ready RTSP URI for private_url.

    If private_url already carries its own credentials (the eventual,
    per-camera-correct API shape), it is returned unchanged -- no probing.
    Otherwise, probes `candidates` in order with a short-timeout
    cv2.VideoCapture open (the same FFmpeg backend open_rtsp_universal()
    already falls back to), caches the first one that authenticates against
    `key`, and reuses that cached combination on every later call instead of
    re-probing. If every candidate fails, returns None so the caller can
    decide how to handle an unresolved camera (callers should retry on the
    next refresh rather than caching a failure).
    """
    parsed = urllib.parse.urlsplit(private_url)
    if parsed.username:
        return private_url

    cached = _resolved_rtsp_credentials.get(key)
    if cached is not None:
        return inject_rtsp_credentials(private_url, *cached)

    for username, password in candidates:
        candidate_uri = inject_rtsp_credentials(private_url, username, password)
        cap = cv2.VideoCapture(candidate_uri, cv2.CAP_FFMPEG,
                                [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms,
                                 cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms])
        opened = cap.isOpened()
        cap.release()
        if opened:
            logger.info(f"[{key}] RTSP credential probe: resolved username='{username}'")
            _resolved_rtsp_credentials[key] = (username, password)
            return candidate_uri

    logger.warning(f"[{key}] RTSP credential probe: none of the {len(candidates)} "
                   "candidate credentials authenticated -- will retry next refresh cycle")
    return None


def build_cctv_entry(company_name, sensor, cctv, access_token, user_group_index):
    """Build one cctv_ID[key]-shaped dict from a (sensor, cctv) pair.

    key is constructed as "<company_name>_<desc>" (requirement 9), falling
    back to "<company_name>_<cctv_id>" when desc is empty -- some sites'
    cctvs all have desc="" (e.g. every 대영빌딩 camera observed), which
    otherwise collapses every one of that company's cameras onto the same
    key and silently drops all but the last one processed (result[key] =
    entry overwrites). cctv['id'] is guaranteed unique and present (it's
    the key fetch_dynamic_config() builds cctvs_by_id from), so it's a safe
    disambiguator that doesn't change the key for any camera with a real
    desc.

    Returns (key, None) when private_url has no embedded credentials and
    none of the RTSP_CREDENTIAL_CANDIDATES authenticated -- the caller
    should skip adding this camera this cycle and let it retry next refresh,
    rather than configuring a URI already known not to work.
    """
    key = f"{company_name}_{cctv['desc']}" if cctv['desc'] else f"{company_name}_{cctv['id']}"

    uri = resolve_rtsp_uri(key, cctv['private_url'])
    if uri is None:
        return key, None

    entry = {
        'device_id': sensor['device_id'],
        'company_name': company_name, # needed for the spacenorm_camera_info metric
        'uri': uri,
        'access_token': access_token,
        'refresh_token': None, # unused by Gateway.__init__ -- see utils/gateway_api.py
        'snapshot_url': cctv.get('snapshot_url'),
        # Hierarchy fields for the spacenorm_camera_info metric -- lets
        # Grafana show the user_groups -> vision_nodes -> CCTVs hierarchy
        # defined in the overrides json file, not just a flat camera list.
        'user_group_index': user_group_index,
        'vision_node_id': sensor.get('vision_node_id'),
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
    """Fetch and parse the dynamic sensors & CCTV configuration for the site,
    merging results across every {access_token, vision_node_id} user group
    in cfg.user_groups -- a site can serve multiple user groups, each its
    own account/token with its own set of vision nodes.

    A group that fails to fetch or parse is logged and skipped; the site
    continues with whichever groups succeeded rather than losing all
    coverage because one group's token or network had an issue. Returns
    None only when every group failed (nothing to merge) or cfg.user_groups
    is empty -- callers should treat None as "retry next period".
    """
    if not cfg.user_groups:
        logger.warning("cfg.user_groups is empty -- nothing to fetch in dynamic config method")
        return None

    test_device_ids = getattr(cfg, 'test_device_ids', None)
    result = {}
    any_group_succeeded = False

    for user_group_index, group in enumerate(cfg.user_groups):
        access_token = group['access_token']
        vision_node_id = group['vision_node_id']
        # vision_node_id may be a single id or a list of ids (see
        # VisionNodeAPI.fetch_sync_data); normalize to a list once so the
        # membership check below is the same either way.
        group_vision_node_ids = vision_node_id if isinstance(vision_node_id, (list, tuple)) else [vision_node_id]

        api = VisionNodeAPI(access_token)
        data = api.fetch_sync_data(vision_node_id)
        if data is None:
            logger.warning(f"user group with vision_node_id {vision_node_id} failed to fetch -- skipping this group")
            continue

        try:
            sensors = data.get('sensors', [])
            cctvs_by_id = {cctv['id']: cctv for cctv in data.get('cctvs', [])}
            # Each vision_node carries its own name, and a multi-id fetch can
            # span multiple companies (e.g. ids=[1,8,9] -> 3 different
            # companies) -- so every sensor's company_name must be looked up
            # by its OWN vision_node_id, not a single shared value. The
            # response's top-level data['company_name'] only ever reflects
            # one node and would mislabel every other node's sensors.
            company_name_by_node = {n['id']: n['name'] for n in data.get('vision_nodes', [])}
            # security_regions is deliberately ignored (requirement 10).

            for sensor in sensors:
                sensor_node_id = sensor.get('vision_node_id')
                if sensor_node_id not in group_vision_node_ids:
                    continue # belongs to a vision node not in this group's configured list

                if test_device_ids and sensor['device_id'] not in test_device_ids:
                    continue

                cctv = cctvs_by_id.get(sensor['cctv_id'])
                if cctv is None:
                    logger.warning(f"sensor {sensor.get('sensor_id')} references unknown cctv_id {sensor.get('cctv_id')} -- skipping")
                    continue

                if not cctv.get('private_url'):
                    logger.warning(f"sensor {sensor.get('sensor_id')} / cctv {cctv.get('id')} has no private_url -- skipping")
                    continue

                company_name = company_name_by_node.get(sensor_node_id)
                if company_name is None:
                    logger.warning(f"sensor {sensor.get('sensor_id')} references vision_node_id {sensor_node_id} not present in vision_nodes list -- skipping")
                    continue

                key, entry = build_cctv_entry(company_name, sensor, cctv, access_token, user_group_index)
                if entry is None:
                    continue # RTSP credential probe exhausted all candidates -- retry next refresh
                result[key] = entry

            any_group_succeeded = True
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Failed to parse vision_nodes sync_data response for group with vision_node_id {vision_node_id}: {e}")
            continue

    if not any_group_succeeded:
        return None

    dynamic_config_last_fetch_timestamp.set(time.time())
    dynamic_config_sensors_active.set(len(result))
    return result


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
