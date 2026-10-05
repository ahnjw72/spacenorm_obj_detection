"""metrics.py

Central definitions for every spacenorm_obj_detection Prometheus metric.
Kept in one module (rather than defined inline in spacenorm_obj_detection.py)
so utils/cctv_camera.py and utils/dynamic_config.py can update them too,
without a circular import on the main script. See
docs/prometheus_grafana_monitoring_plan.md for the full rationale per metric.
"""
import logging
import threading

from prometheus_client import Gauge, Counter, start_http_server

logger = logging.getLogger(__name__)

detector_threads_active = Gauge(
    'spacenorm_detector_threads_active', 'Live detector thread count')
detector_threads_expected = Gauge(
    'spacenorm_detector_threads_expected', 'Expected sensor count from config')

camera_connected = Gauge(
    'spacenorm_camera_connected', 'Per-camera RTSP connection state (0/1)', ['camera'])
camera_last_frame_timestamp = Gauge(
    'spacenorm_camera_last_frame_timestamp', 'Epoch of last successfully grabbed frame', ['camera'])
camera_reconnect_attempts_total = Counter(
    'spacenorm_camera_reconnect_attempts_total', 'Cumulative RTSP (re)connect attempts', ['camera'])
detections_total = Counter(
    'spacenorm_detections_total', 'Cumulative occupancy-report events', ['camera'])
mqtt_connected = Gauge(
    'spacenorm_mqtt_connected', 'Per-camera MQTT client connection state (0/1)', ['camera'])

dynamic_config_last_fetch_timestamp = Gauge(
    'spacenorm_dynamic_config_last_fetch_timestamp',
    'Epoch of last successful dynamic config fetch (dynamic method only)')
dynamic_config_sensors_active = Gauge(
    'spacenorm_dynamic_config_sensors_active',
    'Sensor count from the last successful dynamic config fetch (dynamic method only)')

# "info metric" pattern (same idea as kube_pod_info): value is always 1, the
# labels carry the data. Deliberately excludes access_token/refresh_token and
# the raw uri/private_url -- the dynamic-mode RTSP URI embeds hardcoded
# credentials, and a label is plaintext in both the scrape endpoint and
# Prometheus's own storage. See docs/prometheus_grafana_monitoring_plan.md §2.
camera_info = Gauge(
    'spacenorm_camera_info', 'CCTV configuration info (value always 1)',
    ['camera', 'device_id', 'company_name', 'config_method', 'monitor_id',
     'mqtt_broker_addr', 'mqtt_broker_port', 'min_obj_size_ratio', 'has_roi'])

# camera_info's label set can change over time for a given camera (dynamic
# mode: ROI added, mqtt broker changed, etc.). A Gauge has no "upsert
# replacing the old label combination" primitive, so a naive
# camera_info.labels(...).set(1) on every update would leave the previous
# label combination registered forever. This cache tracks the currently
# registered label tuple per camera so set_camera_info() can remove() the
# stale one before registering the new one.
_camera_info_label_cache = {}
_camera_info_lock = threading.Lock()


def start_metrics_server(port):
    """Start the Prometheus HTTP exposition server. Call once at startup."""
    start_http_server(port)
    logger.info(f"Prometheus metrics server started on :{port}")


def set_camera_info(camera, device_id, company_name, config_method, monitor_id,
                     mqtt_broker_addr, mqtt_broker_port, min_obj_size_ratio, has_roi):
    label_values = (
        camera,
        str(device_id or ''),
        str(company_name or ''),
        str(config_method or ''),
        str(monitor_id or ''),
        str(mqtt_broker_addr or ''),
        str(mqtt_broker_port or ''),
        str(min_obj_size_ratio or ''),
        str(bool(has_roi)),
    )
    with _camera_info_lock:
        previous = _camera_info_label_cache.get(camera)
        if previous is not None and previous != label_values:
            camera_info.remove(*previous)
        camera_info.labels(*label_values).set(1)
        _camera_info_label_cache[camera] = label_values


def remove_camera_info(camera):
    with _camera_info_lock:
        previous = _camera_info_label_cache.pop(camera, None)
        if previous is not None:
            camera_info.remove(*previous)


def remove_camera_metrics(camera):
    """Drop every per-camera label series for `camera`. Call when a sensor
    is removed in dynamic mode, so stale series don't linger forever."""
    remove_camera_info(camera)
    for metric in (camera_connected, camera_last_frame_timestamp,
                   camera_reconnect_attempts_total, detections_total, mqtt_connected):
        try:
            metric.remove(camera)
        except KeyError:
            pass
