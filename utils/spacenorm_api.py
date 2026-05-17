import requests
import json
import time

class Spacenorm_API:
    server_url_base = 'https://dev.contextmatter.com/api/v1'

    def __init__(self, access_token, refresh_token):
        self.__header = {
        'Authorization': "Bearer " + access_token,
        'cache-control': "no-cache",
        'Content-Type': "application/json"
        }
        self.__refresh_token = refresh_token

    def get_values_for_device(self, device_ID, sensor_type, start_ts, end_ts):
        #server_url = Spacenorm_API.server_url_base + f"/sensors/{device_ID}/values_for_device/PS"
        server_url = Spacenorm_API.server_url_base + f"/sensors/{device_ID}/values_for_device/{sensor_type}"
        params = (
            ('start_ts', start_ts),
            ('end_ts', end_ts),
            ('id_type', 'd')
        )
        """
        print("========= get_values_for_device() ===========")
        print(server_url)
        print(device_ID)
        print(params)
        print("=============================================")
        """
        response = requests.get(server_url, headers=self.__header, params=params)
        
        data = json.loads(response.text)
        #print(data['values'])

        # data['values'] is like the following list
        # [{'value': '0.0', 'time': 1636752020}, {'value': '1.0', 'time': 1636752027}, {'value': '0.0', 'time': 1636752030}]
        return data['values']

    def AR_comment_file_upload(response):
        print(response.text)

        """ parse response like this:
        [{
            "id": "device_id of sensor",
            "count": 1,
            "camera_snapshots": [
                {
                    "presigned_url":"https://contextmatter-files.s3.ap-northeast-2.amazonaws.com/security_alerts/20230110/203529-146-S113629430?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIASKJQW2PLPVEILYWG%2F20230110%2Fap-northeast-2%2Fs3%2Faws4_request&X-Amz-Date=20230110T113529Z&X-Amz-Expires=900&X-Amz-SignedHeaders=host&x-amz-acl=public-read&X-Amz-Signature=2a81d6fc3f72b4ada5ef684a6b2ac98581667537ad0e46ffaa4c0fe02bb96275", 
                    "public_url":"https://contextmatter-files.s3.ap-northeast-2.amazonaws.com/security_alerts/20230110/203529-146-S113629430",
                    "action_request_id": 12345
                }
            ]
        }]
        """

def main():
    access_token = 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE'
    refresh_token = '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'
    spacenorm = Spacenorm_API(access_token, refresh_token)

    """
    Test code for reading previously reported sensor values from the server.
    """
    sensor_type = 'DT'
    #value_list = spacenorm.get_values_for_device("73706163656e6f726d5f63616d657273", 'PS', "1636752000", "1636752360")
    # value_list: [{'value': '0.0', 'time': 1636752020}, {'value': '1.0', 'time': 1636752027}, {'value': '0.0', 'time': 1636752030}]

    end_timestamp = int(time.time())
    duration_sec = 7*24*60*60 
    start_timestamp = end_timestamp - duration_sec
    value_list = spacenorm.get_values_for_device('531135ad3d32', 'DT', start_timestamp, end_timestamp)
    for value in value_list:
        print(f"({value['value']}, {value['time']})")

    


if __name__ == "__main__":
    main()
