# Script for making training set with false positive images from NVR server.

import os
import yaml
import glob
import cv2
import datetime
import subprocess
import json
import argparse
import pytz
import copy
from pymediainfo import MediaInfo
from gateway_api import Gateway

yaml_file = "set.yaml"
#yaml_file = "set_temp.yaml"

cctv_ID_deprecated = {     
    # device ID, uri, monitor ID, access token, refresh token for each CCTV IP Camera - ahnjw,2023.03.28
    'B2F_machine_room':('73706163656e6f726d5f63616d657260','rtsp://cym-gamcheon.iptime.org:11087/profile4/media.smp', 'Gdvu1tX6Eq', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    'B1F_food1':('73706163656e6f726d5f63616d657261','rtsp://cym-gamcheon.iptime.org:11082/profile4/media.smp', 'Gdvv1tX6Eq', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    'B1F_food2':('73706163656e6f726d5f63616d65727d','rtsp://cym-gamcheon.iptime.org:11137/profile4/media.smp', '5hjVTQ5Hpe', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    'B1F_food3':('73706163656e6f726d5f63616d65727e','rtsp://cym-gamcheon.iptime.org:11142/profile4/media.smp','XqGLqxsmfc', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    'B1F_B101':('73706163656e6f726d5f63616d657262','rtsp://cym-gamcheon.iptime.org:11052/profile4/media.smp', 'QjEwMQ', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    '1F_dock_roadside1':('73706163656e6f726d5f63616d657263','rtsp://cym-gamcheon.iptime.org:11077/profile4/media.smp', 'Gdvu1tX6Ep', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    '1F_dock_roadside2':('73706163656e6f726d5f63616d657272','rtsp://cym-gamcheon.iptime.org:11097/profile4/media.smp', '7ZWY7Jet7J6l64E66GcMg', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    '1F_dock_roadside3':('73706163656e6f726d5f63616d65727f','rtsp://cym-gamcheon.iptime.org:11147/profile4/media.smp','cuQrHcCSJj', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    '1F_dock_seaside1':('73706163656e6f726d5f63616d657264','rtsp://cym-gamcheon.iptime.org:11072/profile4/media.smp', 'Gevu1tX6Eq', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    '1F_dock_seaside2':('73706163656e6f726d5f63616d657273','rtsp://cym-gamcheon.iptime.org:11092/profile4/media.smp', '7ZWY7Jet7J6l67CU64ukMg', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    '1F_dock_seaside3':('73706163656e6f726d5f63616d657280','rtsp://cym-gamcheon.iptime.org:11152/profile4/media.smp','vu3SG8jJXO', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    '2F_office':('73706163656e6f726d5f63616d657265','rtsp://cym-gamcheon.iptime.org:11067/profile5/media.smp', 'Kdvu1tX6Eq', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    '2F_201':('73706163656e6f726d5f63616d657266','rtsp://cym-gamcheon.iptime.org:11022/profile4/media.smp', 'MjAx', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    '2F_202':('73706163656e6f726d5f63616d657267','rtsp://cym-gamcheon.iptime.org:11057/profile4/media.smp', 'MjAy', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    '3F_301':('73706163656e6f726d5f63616d657268','rtsp://cym-gamcheon.iptime.org:11017/profile4/media.smp', 'MzAx', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    '3F_302':('73706163656e6f726d5f63616d657269','rtsp://cym-gamcheon.iptime.org:11062/profile4/media.smp', 'MzAy', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    '4F_401':('73706163656e6f726d5f63616d65726a','rtsp://cym-gamcheon.iptime.org:11027/profile4/media.smp', 'NDAx', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    '4F_402':('73706163656e6f726d5f63616d65726b','rtsp://cym-gamcheon.iptime.org:11047/profile4/media.smp', 'NDAy', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    '5F_501':('73706163656e6f726d5f63616d65726c','rtsp://cym-gamcheon.iptime.org:11007/profile4/media.smp', 'NTAx', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    '5F_502':('73706163656e6f726d5f63616d65726d','rtsp://cym-gamcheon.iptime.org:11042/profile4/media.smp', 'NTAy', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    '6F_601':('73706163656e6f726d5f63616d65726e','rtsp://cym-gamcheon.iptime.org:11012/profile4/media.smp', 'NjAx', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    '6F_602':('73706163656e6f726d5f63616d65726f','rtsp://cym-gamcheon.iptime.org:11032/profile4/media.smp', 'NjAy', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    '7F_701':('73706163656e6f726d5f63616d657270','rtsp://cym-gamcheon.iptime.org:11002/profile4/media.smp', 'NzAx', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    '7F_702':('73706163656e6f726d5f63616d657271','rtsp://cym-gamcheon.iptime.org:11037/profile4/media.smp', 'NzAy', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    'ROOF_greenhouse':('73706163656e6f726d5f63616d657274','rtsp://cym-gamcheon.iptime.org:11102/profile4/media.smp', '7Jil7IOB7ZWY7Jqw7Iqk64K067aACg', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
 
    'openfield_1':('73706163656e6f726d5f63616d657275','rtsp://admin:Qwert12%23@openfield.iptime.org:11094/profile4/media.smp', '64yA7KCA64W47KeAXzE', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    'openfield_2':('73706163656e6f726d5f63616d657276','rtsp://admin:Qwert12%23@openfield.iptime.org:11097/profile4/media.smp', '64yA7KCA64W47KeAXzI', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    'openfield_3':('73706163656e6f726d5f63616d657281','rtsp://admin:Qwert12%23@openfield.iptime.org:11100/profile4/media.smp', 'WrcmrL1kiq', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    'openfield_4':('73706163656e6f726d5f63616d657282','rtsp://admin:Qwert12%23@openfield.iptime.org:11103/profile4/media.smp', 'hVwIRxQylW', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),

    'kiosk_roadside':('73706163656e6f726d5f63616d657277','rtsp://cym-gamcheon.iptime.org:11107/profile4/media.smp', 'UOHbV6ApMH', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    'kiosk_seaside':('73706163656e6f726d5f63616d657278','rtsp://cym-gamcheon.iptime.org:11112/profile4/media.smp', 'm9QgiP1miU', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),

    'B1F_food_warehouse1':('73706163656e6f726d5f63616d657279','rtsp://cym-gamcheon.iptime.org:11117/profile4/media.smp', '7x9XBfVQN9', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    'B1F_food_warehouse2':('73706163656e6f726d5f63616d65727a','rtsp://cym-gamcheon.iptime.org:11122/profile4/media.smp', 'sbGLiPa5wx', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    'B1F_food_warehouse3':('73706163656e6f726d5f63616d65727b','rtsp://cym-gamcheon.iptime.org:11127/profile4/media.smp', '6HFRgFsbuA', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),
    'B1F_food_warehouse4':('73706163656e6f726d5f63616d65727c','rtsp://cym-gamcheon.iptime.org:11132/profile4/media.smp', 'Oe7eTFZIjO', 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE', '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'),

    'Press_scrap_paper' : ('73706163656e6f726d5f63616d657283','rtsp://admin:Qwert12%23@119.207.239.126:11097/profile4/media.smp', 'RJvaOtwrHS', 'gMivuUBsxb6MG_nuAYWoDaYK-tc_YrJ_rIILc6noOVM', 'VkpzjaKgHJ8KHHONHXmBC1rpEESjf7Optgs-8vyGWcI'),

    'KUMHO_security1' : ('73706163656e6f726d5f63616d657284','rtsp://admin:Qwert12%23@221.152.97.59:11094/profile4/media.smp', 'cPkg5uqXpC', 'W60YEc6Uf460_QRGLLcuoXRmHD6bX9YrqLn5NeN1fSs', 'olgkltf3j9vFZeU_sQCjTzWuyVT0iNt1RuNGZBrM6RI'),
    'KUMHO_security2' : ('73706163656e6f726d5f63616d657285','rtsp://admin:Qwert12%23@221.152.97.59:11097/profile4/media.smp', 'fwPLiCI39W', 'W60YEc6Uf460_QRGLLcuoXRmHD6bX9YrqLn5NeN1fSs', 'olgkltf3j9vFZeU_sQCjTzWuyVT0iNt1RuNGZBrM6RI'),
    'KUMHO_warehouse1' : ('73706163656e6f726d5f63616d657286','rtsp://admin:Qwert12%23@221.152.97.59:11100/profile4/media.smp', 'hGfYcInSXs', 'W60YEc6Uf460_QRGLLcuoXRmHD6bX9YrqLn5NeN1fSs', 'olgkltf3j9vFZeU_sQCjTzWuyVT0iNt1RuNGZBrM6RI'),
    'KUMHO_warehouse2' : ('73706163656e6f726d5f63616d657287','rtsp://admin:Qwert12%23@221.152.97.59:11103/profile4/media.smp', 'MiaPxO0p0c', 'W60YEc6Uf460_QRGLLcuoXRmHD6bX9YrqLn5NeN1fSs', 'olgkltf3j9vFZeU_sQCjTzWuyVT0iNt1RuNGZBrM6RI'),

} # cctv_ID

cctv_names_deprecated = {
  314: '2F_office',
  333: '1F_dock_seaside2',
  357: 'ROOF_greenhouse',
  380: 'openfield_1',
  381: 'openfield_2',
  434: 'kiosk_roadside',
  435: 'kiosk_seaside',
  460: 'openfield_3',
  461: 'openfield_4',  
  484: 'KUMHO_security1',
  485: 'KUMHO_security2',
  737: '초량대영빌딩_1층현관',
}

group_ids = {
    '정양산업': 63,
    '금호정공': 281,
    '초량대영빌딩': 298,
    '유진판지': 249
}

# BASE_FOLDER = '/home/cym/Work/spacenorm_yolov7/data/cctv_train_data/inspection'

# fp_toi = {
#     'set94':[{'2F_office':'202304231018'}, {'5F_502':'202201281331'}],
#     'set95':[{'KUMHO_security1':'202304161326'}, {'KUMHO_security2':'202304291313'}],
#     'set96':[{'openfield_1':'202211061036'}, {'openfield_1':'202211051643'}, {'openfield_1':'202212141340'}, {'openfield_2':'202302021731'}, {'openfield_2':'202304201725'}, {'openfield_2':'202304250529'}, {'openfield_2':'202304251910'}, {'openfield_2':'202304281732'}, {'openfield_3':'202212200741'}, {'openfield_4':'202304061311'}, {'openfield_4':'202304061746'}, {'openfield_4':'202304140836'}, {'openfield_4':'202304141656'}, {'openfield_4':'202304241232'}, {'openfield_4':'202304282011'}, {'openfield_4':'202304301136'}],
# }

def get_duration_of_video(mp4_file):

    # https://stackoverflow.com/questions/3844430/how-to-get-the-duration-of-a-video-in-python
    media_info = MediaInfo.parse(mp4_file)
    #duration in milliseconds
    duration = media_info.tracks[0].duration
    if duration is not None:
        duration_in_sec = duration/1000
    else:
        duration_in_sec = -1

    return duration_in_sec


# Check if the when_str (eg: 20230701014758) is in time duration of mp4_file_name (eg: 2021-09-04T09-15-00)
def check_time(when_str, mp4_file):    

    # mp4_file : "/mnt/cctv_videos/NSTs5hzJDK/Gdvu1tX6Ep/2022-01-05T01-30-01.mp4"
    # mp4_file_name : "2022-01-05T01-30-01"
    mp4_file_name = mp4_file.split('/')[-1].split('.')[0]
    
    """
    duration_of_video = get_duration_of_video(mp4_file)
    print(f"{mp4_file} --> {duration_of_video}  secs")
    """
    
    if len(when_str) != len("20230701014758") or len(mp4_file_name) != len("2021-09-04T09-15-00") or mp4_file_name.count('-') != 4 or mp4_file_name.count('T') != 1:
        print(f"check_time returns False due to incorrectly formated {when_str} or {mp4_file_name}")
        return False

    if mp4_file_name[:4] == when_str[:4]: # check year
        assert(mp4_file_name[:4].isdecimal())
        if mp4_file_name[5:7] == when_str[4:6]: # check month
            assert(mp4_file_name[5:7].isdecimal())
            if mp4_file_name[8:10] == when_str[6:8]: # check day
                assert(mp4_file_name[8:10].isdecimal())
                # compare point in time in terms of minutes (i.e. 09h04min --> 544min)
                assert(mp4_file_name[11:13].isdecimal())
                assert(mp4_file_name[14:15].isdecimal())
                when_in_minutes = int(when_str[8:10])*60 + int(when_str[10:12])
                mp4_file_in_minutes = int(mp4_file_name[11:13])*60 + int(mp4_file_name[14:16])
                if mp4_file_in_minutes <= when_in_minutes and when_in_minutes <= mp4_file_in_minutes+15:
                    duration_of_video_in_sec = get_duration_of_video(mp4_file)                    

                    # check if this file surely contains 'when_in_minutes' since the duration of some mp4 file may not be 15 min.
                    if when_in_minutes < mp4_file_in_minutes + (duration_of_video_in_sec)/60.0: 
                        print(f"{when_str} is in {mp4_file_name}")
                        return True

    #print(f"check_time returns False since it cannot find suitable mp4 file for {when_str} ({mp4_file_name})")
    return False

def get_nvr_file_path(monitor_id, when_str, space_name):
        
    time_of_interest = when_str

    # if cctv_key in cctv_ID:
    #     nvr_file_base = os.path.join("/mnt/cctv_videos/NSTs5hzJDK", cctv_ID[cctv_key][2])
    #     #print("nvr_file_base = ", nvr_file_base)
    # else:
    #     print(f"Cannot get data of {cctv_key} from Shinobi NVR..")
    #     return None, f"No mp4 file for {when_str}"
    nvr_file_base = os.path.join("/mnt/cctv_videos/NSTs5hzJDK", monitor_id)
    
    # find most suitable mp4 file for 'when_str' (YYMMDDhhmm) -------

    # 1. Check the validity of when_str
    if when_str.isdecimal() is False:
        # logger.warning("when_str contains non-decimal character")
        time_of_interest = f"{when_str} is invalid"
        return None, time_of_interest

    # 2. Make a list of mp4 files in nvr_file_base    
    mp4_files = sorted(glob.glob(os.path.join(nvr_file_base, "*.mp4")))

    # 3. Find a suitable mp4 file in the list
    for mp4_file in mp4_files: # mp4_file : "/mnt/cctv_videos/NSTs5hzJDK/Gdvu1tX6Ep/2022-01-05T01-30-01.mp4"
        #print(f"mp4_file = {mp4_file}") 
        #mp4_file_name = mp4_file.split('/')[-1].split('.')[0]
        if check_time(when_str, mp4_file):
            return mp4_file, time_of_interest

    print(f"\nNo suitable mp4 file for {space_name} @ {when_str}")
    # logger.warning(f"There is no suitable mp4 file for {when_str}")
    
    time_of_interest = f"No mp4 file for {when_str}"
    return None, time_of_interest


# folder_path : "/home/cym/Work/yolov7/data/cctv_train_data/inspection/openfield_2@2023-07-05T00:06:23_UTC"
# spacen_name : "openfield_2" 
# toi : ['20230423101822', 30, 30]
def download_FP_training_data_Shinobi(folder_path, alert_info):
    
    space_name = folder_path.split('/')[-1].split('@')[0]
    
    assert alert_info['monitor_id'] !="", f"No Shinobi NVR's monitor_id for {space_name}.."
    
    # backward_duration_seconds = toi[1]
    # forward_duration_seconds = toi[2]
    backward_duration_seconds = alert_info['before']
    forward_duration_seconds = alert_info['after']
    
    alerted_at = alert_info['alerted_at']
    nvr_file_path, time_of_interest = get_nvr_file_path(alert_info['monitor_id'], alerted_at, space_name) # ex of alerted_at : '20230701014758'

    if (nvr_file_path): # nvr_file_path : /mnt/cctv_videos/NSTs5hzJDK/hVwIRxQylW/2023-04-24T12-30-00.mp4

        filename = nvr_file_path.split('/')[-1] # ex of filename: 2021-09-01T09-45-01.mp4
        print("\nfilename is: ",filename)
        start_year = int(filename.split('-')[0])
        start_month = int(filename.split('-')[1])
        start_day = int(filename.split('-')[2].split('T')[0])
        start_hour = int(filename.split('-')[2].split('T')[1])
        start_minute = int(filename.split('-')[3])
        start_second = int(filename.split('-')[4].split('.')[0])
        print(start_year, start_month, start_day, start_hour, start_minute, start_second)
        start_date = datetime.datetime(start_year, start_month, start_day, start_hour, start_minute, start_second)
        end_date = start_date + datetime.timedelta(minutes = 15) # FIXME: assumption: all mp4 files are 15 minutes long
        print(f"mp4 start_date: {start_date}")
        print(f"mp4 end_date  : {end_date}")

        toi_date = datetime.datetime(int(alerted_at[:4]), int(alerted_at[4:6]), int(alerted_at[6:8]), int(alerted_at[8:10]), int(alerted_at[10:12]), int(alerted_at[12:14]))
        timedelta_start_to_toi = toi_date - start_date # timedelta object has hour, minute, second

        cap = cv2.VideoCapture(nvr_file_path)
        frame_rate = (cv2.VideoCapture.get(cap, int(cv2.CAP_PROP_FPS)))
        time_per_frame = 1.0/frame_rate
                
        assert(backward_duration_seconds >= 0)
        print(f"timedelta_start_to_toi.seconds = {timedelta_start_to_toi.seconds}")
        print(f"backward_duration_seconds = {backward_duration_seconds}")
        if (timedelta_start_to_toi.seconds > backward_duration_seconds):
            if cap.set(cv2.CAP_PROP_POS_MSEC, (timedelta_start_to_toi.seconds-backward_duration_seconds)*1000):
                print("\nSuccessful cap.set()\n")
                start_date += datetime.timedelta(seconds = timedelta_start_to_toi.seconds-backward_duration_seconds)
                print(f"New start_date = {start_date}")
            else: 
                print("\nFail in cap.set()\n")
                exit()
        else:
            backward_duration_seconds = timedelta_start_to_toi.seconds
        
        num_frame = 0
        time_elapsed = 0
        while (cap.isOpened()):
            # print(f"==> Successfully opened {nvr_file_path}")
            ret, img = cap.read()
            if ret == True:
                # print(f"(frame {num_frame}) {nvr_file_path}")
                # frame_name = folder_path.split('/')[-1] + f"_{num_frame:04d}" # ex of frame_name : set94_0001
                frame_name = f"{space_name}_{alerted_at}_b{backward_duration_seconds}_f{forward_duration_seconds}_{num_frame:04d}" # ex of frame_name : openfield_3_202306031528_b10_f30_0001
                img_filename = f"{folder_path}/{frame_name}.jpg"
                annotation_filename = f"{folder_path}/{frame_name}.txt"
                cv2.imwrite(img_filename, img)
                subprocess.call(f"touch {annotation_filename}", shell=True)
                # print(f"Write {img_filename}")
                # print(f"Touch {annotation_filename}")
                time_elapsed += time_per_frame
                # check the case of 'early' break (i.e. stop before end-of-file)
                if forward_duration_seconds >= 0:
                    if time_elapsed > (backward_duration_seconds + forward_duration_seconds):                
                        print("Early break")
                        break
                num_frame += 1
            else:
                print(f"cap.read() returns {ret} --> end of file reached")
                break
        
        return True
    else:
        return False

# ex of set_folder_path : "/home/cym/Work/spacenorm_yolov7/data/cctv_train_data/inspection/openfield_4@2023-06-30T16:47:58_UTC"
# ex of toi_list : ["20230701014758",0,60]
def make_FP_inspection_data(set_folder_path, alert_info):
    
    # alerted_at_seoul = toi_info[0] # alerted_at_seoul : "20230701014758"
    # spacename = set_folder_path.split('/')[-1].split('@')[0] # spacename: "openfield_4"

    if 'monitor_id' in alert_info:
        result = download_FP_training_data_Shinobi(set_folder_path, alert_info)
    elif 'wisenet_channel_id' in alert_info:
        result = False
        print(f"ERROR: Download from Wisenet NVR is not yet implemented.. (refer to SUNAPI/hanwha_NVR_get_rtsp.py)")
    
    return result

def get_cctv_data(cctv_info, sensor_device_id):
    for cctv_name, value in cctv_info.items(): # value is also a dictionary
        if sensor_device_id == value['device_id']:
            return cctv_name, copy.deepcopy(value)

    return None, None
            
# format of start_time: '2023-07-20T00:00:00+09:00'
def get_FP_info_from_server(start_time, end_time, api, cctv_info):   

    data_out = {}
    
    for company_name in group_ids:
        # if company_name != '유진판지':
            # continue
        group_id = group_ids[company_name]
        print(f"{company_name} ::")
        r = api.get_false_alarm_report(group_id, start_time, end_time)
        # print(f"r = {r.json()}")
        for data in r.json():
            # print(f"data = {data}")
            sensor_id = data['sensor_id']
            sensor_device_id = data['sensor_device_id']
            
            # print(f"alerted_at = {data['alerted_at']}")
            yymmdd = data['alerted_at'].split('T')[0].split('-')
            hhmmss = data['alerted_at'].split('T')[1].split('.')[0].split(':')
            # print(f"yymmdd = {yymmdd}")
            # print(f"hhmmss = {hhmmss}")
            year = yymmdd[0]
            month = yymmdd[1]
            day = yymmdd[2]
            hour = hhmmss[0]
            minute = hhmmss[1]
            second = hhmmss[2]

            alerted_at_utc_datetime = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second), tzinfo=pytz.utc)

            # Convert it to a different time zone
            target_timezone = pytz.timezone('Asia/Seoul')
            alerted_at_seoul_datetime = alerted_at_utc_datetime.astimezone(target_timezone)

            # Format the datetime in ISO 8601 format
            iso8601_format = '%Y-%m-%dT%H:%M:%S%z'
            iso8601_string = alerted_at_seoul_datetime.strftime(iso8601_format) # "2023-07-05T09:06:23+0900"
            yymmdd_seoul = iso8601_string.split('T')[0].split('-')
            hhmmss_seoul = iso8601_string.split('T')[1].split(':')

            year = yymmdd_seoul[0]
            month = yymmdd_seoul[1]
            day = yymmdd_seoul[2]
            hour = hhmmss_seoul[0]
            minute = hhmmss_seoul[1]
            second = hhmmss_seoul[2].split('+')[0]

            alerted_at_seoul_string = f"{year}{month}{day}{hour}{minute}{second}" # "20230705090623"

            print(f"sensor_device_id = {sensor_device_id}")
            cctv_name, cctv_data = get_cctv_data(cctv_info, sensor_device_id)
            print(f"cctv_name = {cctv_name}")
            print(f"cctv_data = {cctv_data}")

            if (cctv_name == None) or (cctv_data == None):
                continue

            data_elem = {}

            if 'monitor_id' in cctv_data: # in case of shinobi NVR
                data_elem['monitor_id'] = cctv_data['monitor_id']
                if int(second) < 30:
                    before_sec = 30
                    after_sec = 30
                else:
                    before_sec = 0
                    after_sec = 60
            elif 'wisenet_channel_id' in cctv_data: # in case of Hanwha WiseNet NVR
                data_elem['wisenet_channel_id'] = cctv_data['wisenet_channel_id']
                before_sec = 30
                after_sec = 30
            else:
                assert(0, "Error: Unknown NVR identity")

            data_elem['alerted_at'] = alerted_at_seoul_string            
            data_elem['before'] = before_sec
            data_elem['after'] = after_sec

            key = cctv_name + '@' + data['alerted_at'].split('.')[0]+'_UTC'            
            data_out[key] = data_elem

    return data_out

def build_cctv_info(info_folder_path):
    cctv_info_dict = {}
    for info_file in sorted(glob.glob(info_folder_path + "/*.json")):
        # print(info_file)
        with open(info_file, "r", encoding="utf-8") as fp:
            info = json.load(fp)
            companies_info = info['companies']
            for company in companies_info:
                # print(f"company_name = {company['company_name']}")
                company_name = company['company_name']
                cctvs = company['CCTV']
                NVR_name = company['NVR'] # either 'shinobi' or 'wisenet'
                for space in cctvs: # ex. of space : 'B1F_food'
                    new_key = company_name + '_' + space
                    cctv_info_dict[new_key] = cctvs[space]
                    if NVR_name == 'shinobi':
                        assert('monitor_id' in cctv_info_dict[new_key])
                    elif NVR_name == 'wisenet':
                        assert('wisenet_channel_id' in cctv_info_dict[new_key])
    
    # for key, value in cctv_info_dict.items():
    #     print(f"{key}:")
    #     print(f"{value}\n")

    return cctv_info_dict


def parse_args():
    """Parse input arguments."""
    desc = ('Make False Positive training sets by JSON config file or'
            'accessing dev.contextmatter.com server to get the reported'
            'FP information')
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        '-s', '--start', type=str, default='2024-11-25T00:00:00+09:00',
        help=('start time of TOI like this: 2024-01-05T12:00:00+09:00'))
    parser.add_argument(
        '-e', '--end', type=str, default='2025-01-14T00:00:00+09:00',
        help=('end time of TOI like this: 2023-07-20T00:00:00+09:00'))
    parser.add_argument(
        '-c', '--config', type=str, default=None,
        help=('JSON configuration input file (if this is given, start-time and end-time input are ignored)'))
    parser.add_argument(
        '-o', '--out', type=str, default='out.json',
        help=('JSON configuration output file'))
    parser.add_argument(
        '-b', '--base', type=str, default='/home/cym/Work/spacenorm_yolov7/data/cctv_train_data/inspection',
        help=('base folder where FP data is recorded'))
    parser.add_argument(
        '-n', '--no-download', action="store_true",
        help='do not download FP files but only output json file')
    args = parser.parse_args()
    return args

def main():    
    
    args = parse_args()

    base_folder = args.base

    cctv_info = build_cctv_info('./company_cfg')
    # print(f"cctv_info = {cctv_info}")

    # 정양산업(부산)
    access_token = 'HcvuAx6ilImJ8MRCIHp5Hl-PID-PSDWjB1Syd4ZzedE'
    refresh_token = '1TGp6DUPtkOdaJXqBvGPBZ6MgmNUBnjg6j2ka0PFycc'
    api = Gateway(access_token, refresh_token)

    if args.config:
        json_config_file = args.config
        with open(json_config_file, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        assert(not args.no_download)
    else:
        if not args.start: 
            raise SystemExit('ERROR: start time must be designated (-s)')
        if not args.end:
            raise SystemExit('ERROR: end time must be designated (-e)')

        start_time = args.start
        end_time = args.end

        data = get_FP_info_from_server(start_time, end_time, api, cctv_info)
        # ex of data: 
        # {
        #     "정양산업_kiosk_roadside@2023-09-29T01:51:16_UTC": {
        #         "monitor_id": "UOHbV6ApMH",
        #         "alerted_at": "20230929105116",
        #         "before": 30,
        #         "after": 30
        #     },
        #     "정양산업_ROOF_greenhouse@2023-09-30T19:14:25_UTC": {
        #         "monitor_id": "7Jil7IOB7ZWY7Jqw7Iqk64K067aACg",
        #         "alerted_at": "20231001041425",
        #         "before": 30,
        #         "after": 30
        #     }
        # }

        #print(data)

        json_output_file = args.out
        with open(json_output_file, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=4, ensure_ascii=False)                
        print(f"\n{json_output_file} is written..")
    
    if not args.no_download:
        for space_time in data:

            set_folder_path = os.path.join(base_folder, space_time)
            # set_folder_path : "/home/cym/Work/spacenorm_yolov7/data/cctv_train_data/inspection/openfield_4@2023-06-30T16:47:58_UTC"
            
            if not os.path.exists(set_folder_path):
                os.makedirs(set_folder_path)
                alert_info = data[space_time]
                result = make_FP_inspection_data(set_folder_path, alert_info)
                if not result:
                    os.rmdir(set_folder_path)
                    print(f"remove empty directory : {set_folder_path}")
            
            else:
                print(f"Set folder already exists : {set_folder_path} --> do not make again")

                
if __name__=='__main__':
    
    main()   
