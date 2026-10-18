"""Unit tests for utils/dynamic_config.py.

Run from the repo root with:
    python3 -m unittest discover -s tests -t .

No network, RTSP, or filesystem access happens in these tests -- every
external dependency (VisionNodeAPI, cv2.VideoCapture, requests.put,
draw_roi_overlay, os.makedirs, open()) is mocked.
"""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch

import requests

# Make the project's top-level `utils` package importable when this file is
# run directly or via `python -m unittest discover` from the repo root --
# same pattern already used by scripts/cctv/monitor_detection_result/.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils import dynamic_config as dc


def make_sensor(sensor_id=1, device_id='dev1', vision_node_id=10, cctv_id=100, **extra):
    sensor = {
        'sensor_id': sensor_id,
        'device_id': device_id,
        'vision_node_id': vision_node_id,
        'cctv_id': cctv_id,
    }
    sensor.update(extra)
    return sensor


def make_cctv(id=100, desc='lobby', private_url='rtsp://user:pass@host/1', **extra):
    cctv = {'id': id, 'desc': desc, 'private_url': private_url}
    cctv.update(extra)
    return cctv


class DynamicConfigTestCase(unittest.TestCase):
    """Clears dynamic_config's process-global caches before each test.

    resolve_rtsp_uri() caches resolved credentials in
    _resolved_rtsp_credentials, and the snapshot-upload helpers track state
    in _snapshot_uploaded_keys, both at module scope -- left uncleared, a
    result from one test would leak into and silently affect another.
    """

    def setUp(self):
        dc._resolved_rtsp_credentials.clear()
        dc._snapshot_uploaded_keys.clear()


class InjectRtspCredentialsTests(DynamicConfigTestCase):
    def test_injects_into_credential_less_url(self):
        result = dc.inject_rtsp_credentials('rtsp://host:554/stream', 'space', 'p@ss!')
        self.assertEqual(result, 'rtsp://space:p%40ss%21@host:554/stream')

    def test_leaves_url_with_existing_credentials_unchanged(self):
        result = dc.inject_rtsp_credentials('rtsp://already:there@host/stream', 'space', 'pass')
        self.assertEqual(result, 'rtsp://already:there@host/stream')

    def test_handles_source_url_with_no_port(self):
        result = dc.inject_rtsp_credentials('rtsp://host/stream', 'u', 'p')
        self.assertEqual(result, 'rtsp://u:p@host/stream')


class ResolveRtspUriTests(DynamicConfigTestCase):
    def test_returns_url_unchanged_when_already_credentialed(self):
        with patch.object(dc, 'cv2') as mock_cv2:
            result = dc.resolve_rtsp_uri('cam1', 'rtsp://u:p@host/1')
        self.assertEqual(result, 'rtsp://u:p@host/1')
        mock_cv2.VideoCapture.assert_not_called()

    def test_resolves_first_working_candidate_and_caches_it(self):
        fake_cap = MagicMock()
        fake_cap.isOpened.return_value = True
        with patch.object(dc, 'cv2') as mock_cv2:
            mock_cv2.VideoCapture.return_value = fake_cap
            result = dc.resolve_rtsp_uri(
                'cam1', 'rtsp://host/1', candidates=[('u1', 'p1'), ('u2', 'p2')])
        self.assertEqual(result, 'rtsp://u1:p1@host/1')
        self.assertEqual(dc._resolved_rtsp_credentials['cam1'], ('u1', 'p1'))
        # first candidate already worked -- the second must not be probed
        self.assertEqual(mock_cv2.VideoCapture.call_count, 1)
        fake_cap.release.assert_called_once()

    def test_falls_through_to_second_candidate_when_first_fails(self):
        failing_cap = MagicMock()
        failing_cap.isOpened.return_value = False
        working_cap = MagicMock()
        working_cap.isOpened.return_value = True
        with patch.object(dc, 'cv2') as mock_cv2:
            mock_cv2.VideoCapture.side_effect = [failing_cap, working_cap]
            result = dc.resolve_rtsp_uri(
                'cam1', 'rtsp://host/1', candidates=[('u1', 'p1'), ('u2', 'p2')])
        self.assertEqual(result, 'rtsp://u2:p2@host/1')
        self.assertEqual(dc._resolved_rtsp_credentials['cam1'], ('u2', 'p2'))

    def test_cached_credentials_reused_without_reprobing(self):
        dc._resolved_rtsp_credentials['cam1'] = ('u2', 'p2')
        with patch.object(dc, 'cv2') as mock_cv2:
            result = dc.resolve_rtsp_uri('cam1', 'rtsp://host/1')
        mock_cv2.VideoCapture.assert_not_called()
        self.assertEqual(result, 'rtsp://u2:p2@host/1')

    def test_returns_none_when_every_candidate_fails(self):
        fake_cap = MagicMock()
        fake_cap.isOpened.return_value = False
        with patch.object(dc, 'cv2') as mock_cv2:
            mock_cv2.VideoCapture.return_value = fake_cap
            result = dc.resolve_rtsp_uri(
                'cam1', 'rtsp://host/1', candidates=[('u1', 'p1'), ('u2', 'p2')])
        self.assertIsNone(result)
        self.assertEqual(mock_cv2.VideoCapture.call_count, 2)
        self.assertNotIn('cam1', dc._resolved_rtsp_credentials)


class BestEffortKeyTests(unittest.TestCase):
    def test_company_and_desc_known(self):
        key = dc._best_effort_key('acme', make_cctv(id=5, desc='lobby'), make_sensor())
        self.assertEqual(key, 'acme_lobby')

    def test_company_known_cctv_without_desc(self):
        key = dc._best_effort_key('acme', make_cctv(id=5, desc=''), make_sensor())
        self.assertEqual(key, 'acme_5')

    def test_company_unknown_cctv_known(self):
        key = dc._best_effort_key(
            None, make_cctv(id=5, desc='lobby'), make_sensor(vision_node_id=42))
        self.assertEqual(key, 'vn42_lobby')

    def test_company_known_cctv_unknown(self):
        key = dc._best_effort_key('acme', None, make_sensor(device_id='dev9'))
        self.assertEqual(key, 'acme_sensordev9')

    def test_company_and_cctv_both_unknown(self):
        key = dc._best_effort_key(
            None, None, make_sensor(vision_node_id=7, device_id='dev9'))
        self.assertEqual(key, 'vn7_sensordev9')

    def test_falls_back_to_sensor_id_when_device_id_missing(self):
        sensor = {'sensor_id': 55, 'vision_node_id': 7}
        key = dc._best_effort_key(None, None, sensor)
        self.assertEqual(key, 'vn7_sensor55')


class StatusInfoFieldsTests(unittest.TestCase):
    def test_fields_with_company_name(self):
        fields = dc._status_info_fields(
            make_sensor(device_id='d1', vision_node_id=3), 'acme', 2)
        self.assertEqual(fields, {
            'device_id': 'd1', 'company_name': 'acme',
            'user_group_index': 2, 'vision_node_id': 3,
        })

    def test_company_name_none_becomes_empty_string(self):
        fields = dc._status_info_fields(make_sensor(), None, 0)
        self.assertEqual(fields['company_name'], '')


class BuildCctvEntryTests(DynamicConfigTestCase):
    def test_basic_entry_fields(self):
        sensor = make_sensor(device_id='d1', vision_node_id=10)
        cctv = make_cctv(id=100, desc='lobby', private_url='rtsp://u:p@host/1')
        key, entry = dc.build_cctv_entry('acme', sensor, cctv, 'tok123', 0)

        self.assertEqual(key, 'acme_lobby')
        self.assertEqual(entry['device_id'], 'd1')
        self.assertEqual(entry['company_name'], 'acme')
        self.assertEqual(entry['uri'], 'rtsp://u:p@host/1')
        self.assertEqual(entry['access_token'], 'tok123')
        self.assertIsNone(entry['refresh_token'])
        self.assertEqual(entry['user_group_index'], 0)
        self.assertEqual(entry['vision_node_id'], 10)
        self.assertNotIn('ROI', entry)
        self.assertNotIn('min_obj_size_ratio', entry)
        self.assertNotIn('mqtt_broker_addr', entry)
        self.assertNotIn('mqtt_broker_port', entry)

    def test_key_falls_back_to_cctv_id_when_desc_empty(self):
        sensor = make_sensor()
        cctv = make_cctv(id=100, desc='', private_url='rtsp://u:p@host/1')
        key, _ = dc.build_cctv_entry('acme', sensor, cctv, 'tok', 0)
        self.assertEqual(key, 'acme_100')

    def test_roi_vertices_parsed(self):
        vertices = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9]]
        cctv = make_cctv(roi_vertices=json.dumps(vertices))
        _, entry = dc.build_cctv_entry('acme', make_sensor(), cctv, 'tok', 0)
        self.assertEqual(entry['ROI'], {
            'img_w': 1, 'img_h': 1,
            'vertices': vertices,
            'vertices_sorted': 1,
        })

    def test_min_obj_size_ratio_set_when_present(self):
        cctv = make_cctv(min_obj_size_ratio='0.05')
        _, entry = dc.build_cctv_entry('acme', make_sensor(), cctv, 'tok', 0)
        self.assertEqual(entry['min_obj_size_ratio'], 0.05)

    def test_mqtt_broker_set_only_when_both_present(self):
        sensor = make_sensor(mqtt_broker_addr='10.0.0.1', mqtt_broker_port=1883)
        _, entry = dc.build_cctv_entry('acme', sensor, make_cctv(), 'tok', 0)
        self.assertEqual(entry['mqtt_broker_addr'], '10.0.0.1')
        self.assertEqual(entry['mqtt_broker_port'], 1883)

    def test_mqtt_broker_omitted_when_only_addr_present(self):
        sensor = make_sensor(mqtt_broker_addr='10.0.0.1')
        _, entry = dc.build_cctv_entry('acme', sensor, make_cctv(), 'tok', 0)
        self.assertNotIn('mqtt_broker_addr', entry)
        self.assertNotIn('mqtt_broker_port', entry)

    def test_returns_none_entry_when_credentials_unresolved(self):
        cctv = make_cctv(private_url='rtsp://host/no-creds')
        with patch.object(dc, 'resolve_rtsp_uri', return_value=None):
            key, entry = dc.build_cctv_entry('acme', make_sensor(), cctv, 'tok', 0)
        self.assertIsNone(entry)
        self.assertEqual(key, 'acme_lobby')


class FetchDynamicConfigTests(DynamicConfigTestCase):
    def make_cfg(self, user_groups, test_device_ids=None):
        return SimpleNamespace(user_groups=user_groups, test_device_ids=test_device_ids)

    def test_empty_user_groups_returns_none(self):
        result, statuses = dc.fetch_dynamic_config(self.make_cfg([]))
        self.assertIsNone(result)
        self.assertEqual(statuses, {})

    def test_all_groups_fail_to_fetch_returns_none(self):
        cfg = self.make_cfg([{'access_token': 't', 'vision_node_id': 1}])
        with patch.object(dc, 'VisionNodeAPI') as mock_api_cls:
            mock_api_cls.return_value.fetch_sync_data.return_value = None
            result, statuses = dc.fetch_dynamic_config(cfg)
        self.assertIsNone(result)
        self.assertEqual(statuses, {})

    def test_sensor_outside_group_vision_node_ids_is_skipped_entirely(self):
        cfg = self.make_cfg([{'access_token': 't', 'vision_node_id': 10}])
        response = {
            'sensors': [make_sensor(vision_node_id=999, cctv_id=100)],
            'cctvs': [make_cctv(id=100)],
            'vision_nodes': [{'id': 10, 'name': 'acme'}],
        }
        with patch.object(dc, 'VisionNodeAPI') as mock_api_cls:
            mock_api_cls.return_value.fetch_sync_data.return_value = response
            result, statuses = dc.fetch_dynamic_config(cfg)
        self.assertEqual(result, {})
        self.assertEqual(statuses, {})

    def test_successful_sensor_lands_in_result_with_no_failure_status(self):
        cfg = self.make_cfg([{'access_token': 't', 'vision_node_id': 10}])
        response = {
            'sensors': [make_sensor(vision_node_id=10, cctv_id=100)],
            'cctvs': [make_cctv(id=100, desc='lobby', private_url='rtsp://u:p@host/1')],
            'vision_nodes': [{'id': 10, 'name': 'acme'}],
        }
        with patch.object(dc, 'VisionNodeAPI') as mock_api_cls:
            mock_api_cls.return_value.fetch_sync_data.return_value = response
            result, statuses = dc.fetch_dynamic_config(cfg)
        self.assertEqual(list(result.keys()), ['acme_lobby'])
        self.assertEqual(statuses, {})

    def test_test_device_ids_filters_to_excluded_status(self):
        cfg = self.make_cfg(
            [{'access_token': 't', 'vision_node_id': 10}], test_device_ids=['keep-me'])
        response = {
            'sensors': [
                make_sensor(sensor_id=1, device_id='keep-me', vision_node_id=10, cctv_id=100),
                make_sensor(sensor_id=2, device_id='skip-me', vision_node_id=10, cctv_id=101),
            ],
            'cctvs': [
                make_cctv(id=100, desc='a', private_url='rtsp://u:p@host/1'),
                make_cctv(id=101, desc='b', private_url='rtsp://u:p@host/2'),
            ],
            'vision_nodes': [{'id': 10, 'name': 'acme'}],
        }
        with patch.object(dc, 'VisionNodeAPI') as mock_api_cls:
            mock_api_cls.return_value.fetch_sync_data.return_value = response
            result, statuses = dc.fetch_dynamic_config(cfg)
        self.assertEqual(list(result.keys()), ['acme_a'])
        self.assertEqual(statuses['acme_b'][0], dc.STATUS_EXCLUDED_TEST_FILTER)
        self.assertEqual(statuses['acme_b'][1]['device_id'], 'skip-me')

    def test_unknown_cctv_id_status(self):
        cfg = self.make_cfg([{'access_token': 't', 'vision_node_id': 10}])
        response = {
            'sensors': [make_sensor(device_id='d2', vision_node_id=10, cctv_id=999)],
            'cctvs': [make_cctv(id=100)],
            'vision_nodes': [{'id': 10, 'name': 'acme'}],
        }
        with patch.object(dc, 'VisionNodeAPI') as mock_api_cls:
            mock_api_cls.return_value.fetch_sync_data.return_value = response
            result, statuses = dc.fetch_dynamic_config(cfg)
        self.assertEqual(result, {})
        [(status, fields)] = statuses.values()
        self.assertEqual(status, dc.STATUS_UNKNOWN_CCTV_ID)
        self.assertEqual(fields['company_name'], 'acme')

    def test_no_private_url_status(self):
        cfg = self.make_cfg([{'access_token': 't', 'vision_node_id': 10}])
        response = {
            'sensors': [make_sensor(vision_node_id=10, cctv_id=100)],
            'cctvs': [make_cctv(id=100, private_url=None)],
            'vision_nodes': [{'id': 10, 'name': 'acme'}],
        }
        with patch.object(dc, 'VisionNodeAPI') as mock_api_cls:
            mock_api_cls.return_value.fetch_sync_data.return_value = response
            result, statuses = dc.fetch_dynamic_config(cfg)
        self.assertEqual(result, {})
        [(status, _)] = statuses.values()
        self.assertEqual(status, dc.STATUS_NO_PRIVATE_URL)

    def test_vision_node_not_found_status(self):
        cfg = self.make_cfg([{'access_token': 't', 'vision_node_id': [10, 99]}])
        response = {
            'sensors': [make_sensor(vision_node_id=99, cctv_id=100)],
            'cctvs': [make_cctv(id=100)],
            # vision_node 99 is in this group's id list but absent from the
            # response's own vision_nodes array -- company_name lookup fails
            'vision_nodes': [{'id': 10, 'name': 'acme'}],
        }
        with patch.object(dc, 'VisionNodeAPI') as mock_api_cls:
            mock_api_cls.return_value.fetch_sync_data.return_value = response
            result, statuses = dc.fetch_dynamic_config(cfg)
        self.assertEqual(result, {})
        [(status, fields)] = statuses.values()
        self.assertEqual(status, dc.STATUS_VISION_NODE_NOT_FOUND)
        self.assertEqual(fields['company_name'], '')

    def test_credential_unresolved_status(self):
        cfg = self.make_cfg([{'access_token': 't', 'vision_node_id': 10}])
        response = {
            'sensors': [make_sensor(vision_node_id=10, cctv_id=100)],
            'cctvs': [make_cctv(id=100, desc='lobby', private_url='rtsp://host/no-creds')],
            'vision_nodes': [{'id': 10, 'name': 'acme'}],
        }
        with patch.object(dc, 'VisionNodeAPI') as mock_api_cls, \
             patch.object(dc, 'resolve_rtsp_uri', return_value=None):
            mock_api_cls.return_value.fetch_sync_data.return_value = response
            result, statuses = dc.fetch_dynamic_config(cfg)
        self.assertEqual(result, {})
        self.assertEqual(statuses['acme_lobby'][0], dc.STATUS_CREDENTIAL_UNRESOLVED)

    def test_multiple_user_groups_merge_results(self):
        cfg = self.make_cfg([
            {'access_token': 't1', 'vision_node_id': 10},
            {'access_token': 't2', 'vision_node_id': 20},
        ])
        response_a = {
            'sensors': [make_sensor(vision_node_id=10, cctv_id=100)],
            'cctvs': [make_cctv(id=100, desc='a', private_url='rtsp://u:p@host/1')],
            'vision_nodes': [{'id': 10, 'name': 'siteA'}],
        }
        response_b = {
            'sensors': [make_sensor(vision_node_id=20, cctv_id=200)],
            'cctvs': [make_cctv(id=200, desc='b', private_url='rtsp://u:p@host/2')],
            'vision_nodes': [{'id': 20, 'name': 'siteB'}],
        }
        with patch.object(dc, 'VisionNodeAPI') as mock_api_cls:
            mock_api_cls.return_value.fetch_sync_data.side_effect = [response_a, response_b]
            result, statuses = dc.fetch_dynamic_config(cfg)
        self.assertEqual(set(result.keys()), {'siteA_a', 'siteB_b'})
        self.assertEqual(statuses, {})

    def test_one_group_fetch_failure_does_not_lose_the_others(self):
        cfg = self.make_cfg([
            {'access_token': 't1', 'vision_node_id': 10},
            {'access_token': 't2', 'vision_node_id': 20},
        ])
        response_b = {
            'sensors': [make_sensor(vision_node_id=20, cctv_id=200)],
            'cctvs': [make_cctv(id=200, desc='b', private_url='rtsp://u:p@host/2')],
            'vision_nodes': [{'id': 20, 'name': 'siteB'}],
        }
        with patch.object(dc, 'VisionNodeAPI') as mock_api_cls:
            mock_api_cls.return_value.fetch_sync_data.side_effect = [None, response_b]
            result, statuses = dc.fetch_dynamic_config(cfg)
        self.assertEqual(list(result.keys()), ['siteB_b'])

    def test_malformed_response_in_one_group_does_not_abort_others(self):
        cfg = self.make_cfg([
            {'access_token': 't1', 'vision_node_id': 10},
            {'access_token': 't2', 'vision_node_id': 20},
        ])
        # Missing 'id' on the cctv entry raises KeyError while building
        # cctvs_by_id -- this group must be skipped, not fatal.
        malformed_response = {
            'sensors': [make_sensor(vision_node_id=10, cctv_id=100)],
            'cctvs': [{'desc': 'broken', 'private_url': 'rtsp://u:p@host/1'}],
            'vision_nodes': [{'id': 10, 'name': 'siteA'}],
        }
        good_response = {
            'sensors': [make_sensor(vision_node_id=20, cctv_id=200)],
            'cctvs': [make_cctv(id=200, desc='b', private_url='rtsp://u:p@host/2')],
            'vision_nodes': [{'id': 20, 'name': 'siteB'}],
        }
        with patch.object(dc, 'VisionNodeAPI') as mock_api_cls:
            mock_api_cls.return_value.fetch_sync_data.side_effect = [malformed_response, good_response]
            result, statuses = dc.fetch_dynamic_config(cfg)
        self.assertEqual(list(result.keys()), ['siteB_b'])

    def test_multi_id_vision_node_group(self):
        cfg = self.make_cfg([{'access_token': 't', 'vision_node_id': [10, 20]}])
        response = {
            'sensors': [
                make_sensor(sensor_id=1, vision_node_id=10, cctv_id=100),
                make_sensor(sensor_id=2, vision_node_id=20, cctv_id=200),
            ],
            'cctvs': [
                make_cctv(id=100, desc='a', private_url='rtsp://u:p@host/1'),
                make_cctv(id=200, desc='b', private_url='rtsp://u:p@host/2'),
            ],
            'vision_nodes': [{'id': 10, 'name': 'siteA'}, {'id': 20, 'name': 'siteB'}],
        }
        with patch.object(dc, 'VisionNodeAPI') as mock_api_cls:
            mock_api_cls.return_value.fetch_sync_data.return_value = response
            result, statuses = dc.fetch_dynamic_config(cfg)
        self.assertEqual(set(result.keys()), {'siteA_a', 'siteB_b'})


class DiffConfigsTests(unittest.TestCase):
    def test_added_and_removed(self):
        old = {'cam1': {'uri': 'a'}}
        new = {'cam2': {'uri': 'b'}}
        added, removed, changed = dc.diff_configs(old, new)
        self.assertEqual(added, {'cam2'})
        self.assertEqual(removed, {'cam1'})
        self.assertEqual(changed, set())

    def test_changed_when_tracked_field_differs(self):
        old = {'cam1': {'uri': 'a', 'min_obj_size_ratio': 0.1}}
        new = {'cam1': {'uri': 'a', 'min_obj_size_ratio': 0.2}}
        _, _, changed = dc.diff_configs(old, new)
        self.assertEqual(changed, {'cam1'})

    def test_unchanged_when_only_untracked_field_differs(self):
        old = {'cam1': {'uri': 'a', 'company_name': 'acme'}}
        new = {'cam1': {'uri': 'a', 'company_name': 'other-corp'}}
        _, _, changed = dc.diff_configs(old, new)
        self.assertEqual(changed, set())

    def test_unrelated_keys_are_neither_added_nor_removed_nor_changed(self):
        old = {'cam1': {'uri': 'a'}}
        new = {'cam1': {'uri': 'a'}}
        added, removed, changed = dc.diff_configs(old, new)
        self.assertEqual((added, removed, changed), (set(), set(), set()))


class RequiresThreadRestartTests(unittest.TestCase):
    def test_true_when_uri_changes(self):
        self.assertTrue(dc.requires_thread_restart({'uri': 'a'}, {'uri': 'b'}))

    def test_true_when_mqtt_broker_addr_changes(self):
        self.assertTrue(dc.requires_thread_restart(
            {'mqtt_broker_addr': 'a'}, {'mqtt_broker_addr': 'b'}))

    def test_true_when_mqtt_broker_port_changes(self):
        self.assertTrue(dc.requires_thread_restart(
            {'mqtt_broker_port': 1883}, {'mqtt_broker_port': 1884}))

    def test_false_when_only_roi_changes(self):
        self.assertFalse(dc.requires_thread_restart(
            {'uri': 'a', 'ROI': {}}, {'uri': 'a', 'ROI': {'x': 1}}))

    def test_false_when_only_min_obj_size_ratio_changes(self):
        self.assertFalse(dc.requires_thread_restart(
            {'uri': 'a', 'min_obj_size_ratio': 0.1}, {'uri': 'a', 'min_obj_size_ratio': 0.2}))


class SnapshotUploadStateTests(DynamicConfigTestCase):
    def test_uploads_once_per_contiguous_true_stretch(self):
        entry = {'snapshot_update_required': True}
        self.assertTrue(dc.should_upload_snapshot('cam1', entry))
        dc._snapshot_uploaded_keys.add('cam1')  # simulate a completed upload
        self.assertFalse(dc.should_upload_snapshot('cam1', entry))

    def test_false_when_flag_not_set_and_clears_tracking(self):
        dc._snapshot_uploaded_keys.add('cam1')
        entry = {'snapshot_update_required': False}
        self.assertFalse(dc.should_upload_snapshot('cam1', entry))
        self.assertNotIn('cam1', dc._snapshot_uploaded_keys)

    def test_true_again_after_a_false_then_true_transition(self):
        entry_true = {'snapshot_update_required': True}
        entry_false = {'snapshot_update_required': False}
        self.assertTrue(dc.should_upload_snapshot('cam1', entry_true))
        dc._snapshot_uploaded_keys.add('cam1')
        self.assertFalse(dc.should_upload_snapshot('cam1', entry_true))
        dc.should_upload_snapshot('cam1', entry_false)  # flips back, clears tracking
        self.assertTrue(dc.should_upload_snapshot('cam1', entry_true))

    def test_forget_snapshot_state_clears_tracking(self):
        dc._snapshot_uploaded_keys.add('cam1')
        dc.forget_snapshot_state('cam1')
        self.assertNotIn('cam1', dc._snapshot_uploaded_keys)


class SaveAndUploadSnapshotTests(DynamicConfigTestCase):
    def test_skips_entirely_when_frame_read_fails(self):
        cam = MagicMock()
        cam.read.return_value = None
        with patch.object(dc, 'cv2') as mock_cv2, patch.object(dc.os, 'makedirs') as mock_makedirs:
            dc.save_and_upload_snapshot('cam1', cam, {}, '/tmp/snaps')
        mock_cv2.imwrite.assert_not_called()
        mock_makedirs.assert_not_called()

    def test_writes_snapshot_and_skips_upload_when_not_due(self):
        cam = MagicMock()
        cam.read.return_value = 'fake-frame'
        entry = {'snapshot_update_required': False}
        with patch.object(dc, 'cv2') as mock_cv2, \
             patch.object(dc.os, 'makedirs') as mock_makedirs, \
             patch.object(dc, 'draw_roi_overlay') as mock_roi, \
             patch.object(dc.requests, 'put') as mock_put:
            dc.save_and_upload_snapshot('cam1', cam, entry, '/tmp/snaps')
        mock_makedirs.assert_called_once_with('/tmp/snaps', exist_ok=True)
        mock_cv2.imwrite.assert_called_once_with('/tmp/snaps/cam1.jpg', 'fake-frame')
        mock_roi.assert_not_called()
        mock_put.assert_not_called()

    def test_draws_roi_overlay_when_roi_is_set(self):
        cam = MagicMock()
        cam.read.return_value = 'frame'
        entry = {'snapshot_update_required': False, 'ROI': {'vertices': []}}
        with patch.object(dc, 'cv2') as mock_cv2, \
             patch.object(dc.os, 'makedirs'), \
             patch.object(dc, 'draw_roi_overlay', return_value='frame-with-roi') as mock_roi:
            dc.save_and_upload_snapshot('cam1', cam, entry, '/tmp/snaps')
        mock_roi.assert_called_once_with('frame', entry['ROI'])
        mock_cv2.imwrite.assert_called_once_with('/tmp/snaps/cam1.jpg', 'frame-with-roi')

    def test_uploads_when_due_and_marks_as_uploaded(self):
        cam = MagicMock()
        cam.read.return_value = 'frame'
        entry = {'snapshot_update_required': True, 'snapshot_url': 'https://api/snap/cam1'}
        fake_response = MagicMock()
        with patch.object(dc, 'cv2'), \
             patch.object(dc.os, 'makedirs'), \
             patch('builtins.open', mock_open()), \
             patch.object(dc.requests, 'put', return_value=fake_response) as mock_put:
            dc.save_and_upload_snapshot('cam1', cam, entry, '/tmp/snaps')
        mock_put.assert_called_once()
        self.assertEqual(mock_put.call_args.args[0], 'https://api/snap/cam1')
        fake_response.raise_for_status.assert_called_once()
        self.assertIn('cam1', dc._snapshot_uploaded_keys)

    def test_upload_failure_does_not_mark_as_uploaded(self):
        cam = MagicMock()
        cam.read.return_value = 'frame'
        entry = {'snapshot_update_required': True, 'snapshot_url': 'https://api/snap/cam1'}
        with patch.object(dc, 'cv2'), \
             patch.object(dc.os, 'makedirs'), \
             patch('builtins.open', mock_open()), \
             patch.object(dc.requests, 'put', side_effect=requests.exceptions.ConnectionError('boom')):
            dc.save_and_upload_snapshot('cam1', cam, entry, '/tmp/snaps')
        self.assertNotIn('cam1', dc._snapshot_uploaded_keys)

    def test_missing_snapshot_url_skips_upload(self):
        cam = MagicMock()
        cam.read.return_value = 'frame'
        entry = {'snapshot_update_required': True, 'snapshot_url': None}
        with patch.object(dc, 'cv2'), \
             patch.object(dc.os, 'makedirs'), \
             patch.object(dc.requests, 'put') as mock_put:
            dc.save_and_upload_snapshot('cam1', cam, entry, '/tmp/snaps')
        mock_put.assert_not_called()
        self.assertNotIn('cam1', dc._snapshot_uploaded_keys)


if __name__ == '__main__':
    unittest.main()
