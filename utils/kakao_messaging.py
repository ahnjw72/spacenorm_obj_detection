import requests
import json
import threading
import time

ACCESS_TOKEN_REFRESH_INTERVAL_HOUR = 5 # access toekn for REST API is valid only for 6 hours
KAKAO_CODE_JSON_FILE = "kakao_code.json"

"""
https://devtalk.kakao.com/t/refresh-token/85191

access_token은 짧은 주기로 만료되고, 매번 로그인을 다시 하는것은 비효율적이기 때문에 긴 만료주기인 refresh_token으로 access_token을 새로 받습니다.
다만 refresh_token도 무제한이 아니기 때문에 비교적 긴 달 단위의 갱신이 필요합니다.
refresh_token의 갱신은 별도 API로 하는것이 아닌, access_token을 갱신 받는 과정에서, 시스템 판단으로 필요시 refresh_token까지 갱신해줍니다.
응답에 보통 access token 갱신일 경우 access_token만 내려주지만, refresh_token이 갱신이 필요할 경우, access_token+refresh_token 이렇게 2개를 내려줍니다.
refresh_token이 응답으로 내려올 경우 기존 refresh_token은 더이상 사용이 안되므로 파기 및 새로 받은 refresh_token으로 교체해 주셔야 합니다.

지금 정책으로 refresh_token은 발급시점부터 2달간 유효합니다.
1월1일 발급이라면, 2월말까지 유효할 것입니다.
다만 access_token 갱신을 2월에 한다면, 그 요청한 2월에 refresh_token이 같이 갱신됩니다. 예를들어,
1월 중순 요청: access_token만 갱신
2월 중순 요청: access_token + new refresh_token 갱신
new refresh_token은 이제 2월 중순~4월 중순까지 유효합니다.

1월 29일은 refresh_token이 갱신되지 않습니다. 2월 중순에 재갱신된다면 2월 중순부터 다시 2달의 유효시간을 갖습니다.
"""
def refreshToken(refresh_token) -> str:
    REST_API_KEY = "d34b990e983b8869a3ab3175fc0c43d7"
    REDIRECT_URI = "https://kauth.kakao.com/oauth/token"

    data = {
        "grant_type": "refresh_token", # 얘는 단순 String임. "refresh_token"
        "client_id": "d34b990e983b8869a3ab3175fc0c43d7",
        "refresh_token": refresh_token # 여기가 위에서 얻은 refresh_token 값
    }    
 
    resp = requests.post(REDIRECT_URI , data=data)
    print("resp = ", resp)
    new_token = resp.json()
    print("new_token = ", new_token)

    #return new_token['access_token']
    return new_token

def kakaoMsgSend(tokens, msg_str):
    url="https://kapi.kakao.com/v2/api/talk/memo/default/send"

    headers={
        "Authorization" : "Bearer " + tokens["access_token"]
    }

    data={
        "template_object": json.dumps({
            "object_type": "text",
            "text": msg_str,
            "link": {
                "web_url" : "text와 link 객체는 필수로 넣어야 함. button_title과 buttons는 안 넣어도 상관 없음.",
                "mobile_web_url" : "text와 link 객체는 필수로 넣어야 함. button_title과 buttons는 안 넣어도 상관 없음."
            },
            "button_title" : "헤헤"
        })
    }

    response = requests.post(url, headers=headers, data=data)
    response.status_code   


if __name__ == "__main__":

    while(1):
        with open(KAKAO_CODE_JSON_FILE,"r") as fp:
            tokens = json.load(fp)

        kakaoMsgSend(tokens, "spacenorm CCTV 이상 상황 발생")
        time.sleep(2)

        refresh_token = tokens['refresh_token']
        print("refresh_toekn = ", refresh_token)
        
        #new_access_token = refreshToken(refresh_token)
        new_tokens = refreshToken(refresh_token)

        assert('access_token' in new_tokens)
        new_access_token = new_tokens['access_token']
        tokens['access_token'] = new_access_token

        if ('refresh_token' in new_tokens):
            new_refresh_token = new_tokens['refresh_token']
            tokens['refresh_token'] = new_refresh_token
    
        #print("old_access_token = ", tokens['access_token']) 
        #print("new_access_token = ", new_access_token)

        with open(KAKAO_CODE_JSON_FILE,"w") as fp:
            json.dump(tokens, fp)

