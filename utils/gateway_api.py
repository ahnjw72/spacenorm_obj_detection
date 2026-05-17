import logging
import requests
import json
import time
import threading
import queue
import inspect
import urllib.parse

# logger = logging.getLogger('spacenorm_person_detect')
logger = logging.getLogger(__name__)

class Hearbeat:
    def __init__(self):
        self.__lock = threading.Lock()

class PresenceData:
    def __init__(self):
        self.__queue = queue.Queue()

    def put(self, device_id, sensor_data):
        data = {
            'id':device_id,
            'vs': [{
                'k': 'PS',
                'v': str(sensor_data),
                'ts': int(time.time())
            }]
        }
        self.__queue.put(data)

    def get(self):
        loop = True
        data_list = []

        while loop:
            try:
                data = self.__queue.get_nowait()
                data_list.append(data)
            except queue.Empty:
                loop = False
            else:
                self.__queue.task_done

        return data_list

class Gateway:
    server_url_base = 'https://dev.contextmatter.com/api/v1'
    # server_url = 'https://dev.contextmatter.com/api/v1/gateways'
    server_url = server_url_base+'/gateways'
    hb_url = server_url + '/hb'
    report_url = server_url + '/report'
    action_requests_url = 'https://dev.contextmatter.com/api/v1/action_requests'
    false_alarm_report_url = 'https://dev.contextmatter.com/api/v1/false_alarm_reports'

    def __init__(self, access_token, refresh_token):
        self.__header = {
            'Authorization': "Bearer " + access_token,
            'cache-control': "no-cache",
            'Content-Type': "application/json"
        }

        """
        self.__header_AR_comment_with_files = {
            'Authorization': "Bearer " + access_token,
            'cache-control': "no-cache",
            'MIME-Type': "image/jpeg"
        }
        """

        self.__refresh_token = refresh_token
        self.__presence_data = PresenceData()

        self.requests_timeout = 10
      

    def add_presence_data(self, device_id, data):
        self.__presence_data.put(device_id, data)

    def print_requests_response(self, function_name, r, key):
        if r.status_code != 201:
            logger.debug(f"[{key}] {function_name}(): r.status_code = {r.status_code} (not 201) --> do not print r.json()")
        else:
            logger.debug(f"[{key}] {function_name}(): response : {r.json()}")
        logger.debug(f"[{key}] {function_name}(): code : {r.status_code}")

    def print_requests_exception(self, function_name, e, key):
        logger.error(f"[{key}] An exception occurred in {function_name}(): {e}")
        if isinstance(e, requests.exceptions.HTTPError):
            logger.error(f"[{key}] HTTP error: {e.response.text}")    


    def heartbeat(self, device_ids, key):    # device_id : list
        payload = {}
        if device_ids:
            payload['sensors'] = device_ids

        try:
            r = requests.post(Gateway.hb_url, data=json.dumps(payload), headers=self.__header, timeout=self.requests_timeout)
            r.raise_for_status()
            request_message = r.request
   
        except requests.exceptions.RequestException as e:
            function_name = inspect.currentframe().f_code.co_name
            self.print_requests_exception(function_name, e, key)     
        else:        
            function_name = inspect.currentframe().f_code.co_name
            self.print_requests_response(function_name, r, key)            


    def report(self, sensor_type, device_id, sensor_data, key):
        payload = {
            'nodes':[{
                'id':device_id,
                'vs': [{

                    'k': sensor_type, 
                    # 'PS' : Presence Status (재실상태) 
                    # 'ED' : Fire Anomaly Detection (화재 관련 이상상황 발생)

                    'v': str(sensor_data),
                    'ts': int(time.time())
                }]
            }]
        }

        return self.__report(payload, key)


    def __report(self, payload, key):
        
        # payload = {
        #     'nodes':[{
        #         'id':device_id,
        #         'vs': [{
        #             'k': 'PS', # Presence Status (재실상태)
        #             'v': str(sensor_data),
        #             'ts': int(time.time())
        #         }]
        #     }]
        # } 
        
        logger.debug(f"[{key}] __report(): payload = {payload}")

        try:
            r = requests.post(Gateway.report_url, data=json.dumps(payload), headers=self.__header, timeout=self.requests_timeout)
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            function_name = inspect.currentframe().f_code.co_name
            self.print_requests_exception(function_name, e, key)     
        else:        
            function_name = inspect.currentframe().f_code.co_name
            self.print_requests_response(function_name, r, key)

            return r

    #def __del__(self):
    def release(self):
        logger.debug("    Gateway::release() is called..")
        self.__thread_running = False # stop    __work_thread()
        # self.__thread_handle.join()
        logger.debug("    Gateway::release() returns..")

    def AR_comment_with_files(self, r, image_file, image_filename, msg, key):
        data = r.json()
        assert(len(data['result'][0]['camera_snapshots']) == 1) # don't know how to process when there are more than one element            
        
        presigned_url = data['result'][0]['camera_snapshots'][0]['presigned_url']
        public_url = data['result'][0]['camera_snapshots'][0]['public_url']
        action_request_id = data['result'][0]['camera_snapshots'][0]['action_request_id']

        logger.debug(f"[{key}] Presigned URL for {image_filename}: {presigned_url}")

        # Upload an image file using PUT method
        try:
            #response = requests.put(presigned_url, files={image_filename:image_file}) <-- this method corrupts the content of original file
            headers = {'Content-type': 'image/jpeg', 'Slug': image_filename}
            response = requests.put(presigned_url, data=image_file, headers=headers, timeout=self.requests_timeout)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            function_name = inspect.currentframe().f_code.co_name
            self.print_requests_exception(function_name, e, key)    
        else:        
            logger.debug(f"[{key}] PUT in AR_comment_with_files() : {image_filename}")

        # Register public_url for the uploaded image using POST method
        payload = {'message':msg, 'url':public_url, 'mime_type':'image/jpeg', 'id':action_request_id}
        comment_url = self.action_requests_url + f'/{action_request_id}/comment_with_files'        
        try:
            response = requests.post(comment_url, data=json.dumps(payload), headers=self.__header, timeout=self.requests_timeout)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            function_name = inspect.currentframe().f_code.co_name
            self.print_requests_exception(function_name, e, key)    
        else:        
            logger.debug(f"[{key}] public_url : {public_url}")


    def AR_comment(self, r, msg, key):

        # data = r.json()

        try:             
            data = r.json()   
            public_url = data['result'][0]['camera_snapshots'][0]['public_url']        
            action_request_id = data['result'][0]['camera_snapshots'][0]['action_request_id']
        except Exception as e:
            # logger.error(f"Exception in AR_comment() : {e} (data = {data})")
            logger.error(f"Exception in AR_comment() : {e}")
            return

        # Register public_url for the uploaded image using POST method
        payload = {'message':msg, 'url':public_url, 'mime_type':'image/jpeg', 'id':action_request_id}
        comment_url = self.action_requests_url + f'/{action_request_id}/comment_with_files'        
        
        # payload = {'message':msg, 'id':action_request_id}
        # comment_url = self.action_requests_url + f'/{action_request_id}/comments'        
        
        try:
            response = requests.post(comment_url, data=json.dumps(payload), headers=self.__header, timeout=self.requests_timeout)
            
            # Just for testing ------------------------------
            request_object = response.request
            request_body = request_object.body
            # print("POST Request Content:", request_body)
            # -----------------------------------------------
            
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            function_name = inspect.currentframe().f_code.co_name
            self.print_requests_exception(function_name, e, key)
        else:        
            logger.debug(f"[{key}] public_url : {public_url}")

    def get_values_for_device(self, device_ID, sensor_type, start_ts, end_ts):        
        server_url_for_get_values = Gateway.server_url_base + f"/sensors/{device_ID}/values_for_device/{sensor_type}"
        params = (
                ('start_ts', start_ts),
                ('end_ts', end_ts),
                ('id_type', 'd')
        )
 
        try:
            response = requests.get(server_url_for_get_values, headers=self.__header, params=params, timeout=self.requests_timeout)
        except Exception as e:
            logger.error(f"Exception in requests.get() get_values_for_device() : {e}")
            return None
        
        try:
            data = json.loads(response.text)
        except Exception as e:
            logger.error(f"Exception in get_values_for_device() : {e}")
            return None
 
        # data['values'] is like the following list
        # [{'value': '0.0', 'time': 1636752020}, {'value': '1.0', 'time': 1636752027}, {'value': '0.0', 'time': 1636752030}]
        try:
            values = data['values']
            return values
        except Exception as e:
            logger.error(f"Exception in get_values_for_device() : {e}")
            return None


    def get_false_alarm_report(self, group_id, start_time, end_time):
        """
        Get false alarm report (notion page: https://www.notion.so/47c0e74527e44abaaf4d0ad0714b8bd0?v=9ecf307c6a81417c80b3eab079171777&p=02a60b06665842679f74c47481f3f0ba&pm=s)
        start_time, end_time : Timezone정보 포함하여 보내야 함(iso8601포멧으로 yyyy-mm-ddThh:mm:ss+09:00 예:2021-04-25T09:00:00+09:00)
        response는 다음과 같음:
        [
            {
                "id": 1,
                "category": "security",
                "url": "https://contextmatter-files.s3.ap-northeast-2.amazonaws.com/security_alerts/20230625/221432-200-S314.jpg",
                "reason": "test",
                "message": "1 person(s) detected",
                "alerted_at": "2023-06-25T22:14:32.000Z",
                "created_at": "2023-06-27T06:26:11.000Z",
                "action_request_comment_id": 305697,
                "sensor_id": 314,
                "user_group_id": 63
            },
            {
                "id": 2,
                "category": "security",
                "url": "https://contextmatter-files.s3.ap-northeast-2.amazonaws.com/security_alerts/20230625/221432-200-S314.jpg",
                "reason": "test2",
                "message": "1 person(s) detected",
                "alerted_at": "2023-06-25T22:14:32.000Z",
                "created_at": "2023-06-27T06:57:16.000Z",
                "action_request_comment_id": 305697,
                "sensor_id": 314,
                "user_group_id": 63
            }
        ]
        """
        params = (
            ('user_group_id', group_id),
            ('start_time', start_time),
            ('end_time', end_time),
        )

        response = requests.get(self.false_alarm_report_url, headers=self.__header, params=params, timeout=self.requests_timeout)

        return response

def AR_comment_with_files_test():
    """
    Test code for uploading image file with bounding boxes to the presigned url returned from spacenorm server.
    Flow is explained in this notion page: https://www.notion.so/AR-comment-ba54e3b28f674fb6ac8e11c77e8f65e4
    """
    image_filename = 'set80_1199.jpg' #'test.jpg'
    with open(image_filename, 'rb') as f:
        r = api.report('73706163656e6f726d5f63616d657276', 0)
        if len(r.json()['result'][0]['camera_snapshots']) > 0:
            api.AR_comment_with_files(r, f, image_filename, 'test..')
        time.sleep(2)            

        print('')
        r = api.report('73706163656e6f726d5f63616d657276', 1)
        if len(r.json()['result'][0]['camera_snapshots']) > 0:
            api.AR_comment_with_files(r, f, image_filename, 'test..')


def AR_comment_for_ED_sensor_test(access_token, refresh_token, api, virtual_ED_sensor_id):
    print("report 0")
    r = api.report('ED', virtual_ED_sensor_id, '0', None)
    print(f"--> response = {r.json()}")

    time.sleep(1)
   
    print("report 1")
    r = api.report('ED', virtual_ED_sensor_id, '1', None)
    print(f"Response: {r.json()}")
    print(f"action request id = {r.json()['result'][0]['camera_snapshots'][0]['action_request_id']}")
    
    # api.AR_comment(r, msg='테스트 중입니다..', key=None)
    if len(r.json()['result'][0]['camera_snapshots']) > 0:
        image_filename = 'logo.png'        
        f = open(image_filename, 'rb')
        urllib.parse.quote(image_filename)
        msg = "테스트 중입니다. 무시하세요."
        api.AR_comment_with_files(r, f, urllib.parse.quote(image_filename), msg, None)
        print(f"AR_comment_with_files() succeeded with msg = '{msg}'")
    else:
        print("no information for uploading AR comment with file")

def AR_comment_for_PS_sensor_test(access_token, refresh_token, api, device_ID):
    print("report 0")
    r = api.report('PS', device_ID, 0, 'test_korean')
    print(f"--> response = {r.json()}")

    time.sleep(1)

    print("report 1")
    r = api.report('PS', device_ID, 1, 'AR_comment_test')
    print(f"--> response = {r.json()}")

    if len(r.json()['result'][0]['camera_snapshots']) > 0:
        image_filename = 'test_car.jpg'        
        f = open(image_filename, 'rb')
        urllib.parse.quote(image_filename)
        msg = "!! 스냅샷 테스트 중입니다. 무시하십시오 !!"
        api.AR_comment_with_files(r, f, urllib.parse.quote(image_filename), msg, 'test_korean')
        print(f"AR_comment_with_files_succeeded with msg = '{msg}'")
    else:
        print("no information for uploading AR comment with file")

def main():
    
    # 정양산업(부산)
    access_token = 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE'
    refresh_token = '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'

    
    # 금호정공
    # access_token = 'h9ev3VYRmpEsWAHnLVLQSkpLIxDeGdz6IOjXG699Ivo'
    # refresh_token = 'NU3vAm8fblJnwmjQNV77NcMHEZhf944kyi4sh4JIy8c'
    
    # 유진판지
    # access_token = 'CB5Ggdt9G7mdhHXozcaGw1pVM86yOLF-keIjGJIcyRA'
    # refresh_token = 'uq0KRolQ3IM-BsIa88C3r0zffiPDK9aAtcs9PdQqYOo'

    # 아세아제지(안성)
    # access_token = 'INWC6TMFDeMNZR1PWwzbOz3CTBDuuFGTqRtUuZpUPSA'
    # refresh_token = 'ZP8LtFwI6s5VRa5slQUJi6SC0tNl21FV7oL6-Suzx3E'

    api = Gateway(access_token, refresh_token)

    # AR_comment_for_ED_sensor_test(access_token, refresh_token, api, virtual_ED_sensor_id='5356453130300')

    AR_comment_for_PS_sensor_test(access_token, refresh_token, api, device_ID='73706163656e6f726d5f63616d657273')

       
if __name__ == "__main__":
        main()
