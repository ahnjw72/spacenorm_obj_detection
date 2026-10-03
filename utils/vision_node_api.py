import logging
import requests
import inspect

logger = logging.getLogger(__name__)


class VisionNodeAPI:
    """Client for the "dynamic" sensors & CCTV configuration method.

    Fetches the sensors/cctvs configuration for a site from the Spacenorm
    vision_nodes API, as an alternative to the "static" json/.keys files
    under spacenorm_cfg/cctv/.
    """

    server_url_base = 'https://dev.contextmatter.com/api/v1'
    sync_data_url = server_url_base + '/vision_nodes/sync_data'

    def __init__(self, access_token):
        self.__header = {
            'Authorization': "Bearer " + access_token,
            'Accept': 'application/json',
        }
        self.requests_timeout = 10

    def print_requests_exception(self, function_name, e):
        logger.error(f"An exception occurred in {function_name}(): {e}")
        if isinstance(e, requests.exceptions.HTTPError):
            logger.error(f"HTTP error: {e.response.text}")

    def fetch_sync_data(self, vision_node_id):
        """Fetch sensors/cctvs configuration for the given vision_node_id.

        Returns the parsed JSON dict on success, or None (with the error
        logged) on any network/HTTP/JSON failure -- callers should treat a
        None return as "retry next period", not a fatal error.
        """
        params = {'ids': vision_node_id}

        try:
            r = requests.get(self.sync_data_url, headers=self.__header, params=params, timeout=self.requests_timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            function_name = inspect.currentframe().f_code.co_name
            self.print_requests_exception(function_name, e)
            return None
        except ValueError as e: # JSON decode error
            function_name = inspect.currentframe().f_code.co_name
            logger.error(f"[{function_name}] Failed to decode JSON response: {e}")
            return None
