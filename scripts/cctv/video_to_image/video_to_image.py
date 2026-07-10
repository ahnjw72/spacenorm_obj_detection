#!/usr/bin/env python
# coding: utf-8

# In[28]:

'''
Script for converting mp4 files to jpg images (ahnjw,2020.06.14)
1. Assign set names as a list to the variable 'sets'
2. Copy video (mp4 or avi) files to specific folders under INPUT_VIDEO_BASE_DIR
3. Run this script w/o any argument
4. *.jpg from *.mp4 (or *.avi) will be generated under the same folders under OUPTUT_IMAGE_BASE_DIR
'''

import os
import cv2
import glob

INPUT_VIDEO_BASE_DIR="01.videos"
OUTPUT_IMAGE_BASE_DIR="02.images"

#sets = os.listdir(INPUT_VIDEO_BASE_DIR)
sets = ["yujin3"]
# sets = ["초량대영빌딩_1층현관@2023-10-05T23:04:49_UTC", "초량대영빌딩_1층현관@2023-10-06T04:09:33_UTC", 
#         "초량대영빌딩_1층현관@2023-10-05T16:09:33_UTC", "초량대영빌딩_1층현관@2023-10-06T16:57:04_UTC",
#         "초량대영빌딩_1층현관@2023-10-06T15:58:24_UTC", "초량대영빌딩_1층현관@2023-10-06T22:37:13_UTC",
#         "초량대영빌딩_1층현관@2023-10-06T23:59:12_UTC", "초량대영빌딩_1층현관@2023-10-07T12:47:32_UTC",
#         "초량대영빌딩_1층현관@2023-10-07T14:01:15_UTC", "초량대영빌딩_1층현관@2023-10-07T23:52:21_UTC"
#         ]
#video_type = "mp4" # "mp4" or "avi"
skip = 10 # skip=1 --> save all the frames

for video_set in sets:
    #input_videos = os.path.join(INPUT_VIDEO_BASE_DIR, video_set) + "/*.mp4"
    #input_videos = os.path.join(INPUT_VIDEO_BASE_DIR, video_set) + "/*.{}".format(video_type)
    input_videos = os.path.join(INPUT_VIDEO_BASE_DIR, video_set) + "/*"
    print(input_videos)
    
    try:
        output_image_dir = os.path.join(OUTPUT_IMAGE_BASE_DIR, video_set)
        if not os.path.exists(output_image_dir):
            print("making {}".format(output_image_dir))
            os.makedirs(output_image_dir)
                                        
    except OSError:
        print("Error: Creating folder {}".format(output_image_dir))
    
    for input_file_path in glob.glob(input_videos):
        cam = cv2.VideoCapture(input_file_path)
        print(input_file_path)
        if not cam.isOpened():
            print("--> Not Opened!!")
                
        currentframe = 0
        image_index = 0
        while(True):
            ret,frame = cam.read()

            if ret:
                if currentframe % skip == 0:
                    # if video is still left continue creating images 
                    name = input_file_path.split("/")[-1].split(".")[0] + "_" + "{:04d}".format(image_index) + '.jpg'
                    print ('Creating... ' + name) 
    
                    # writing the extracted images 
                    cv2.imwrite(os.path.join(output_image_dir, name), frame)
    
                    image_index += 1
            else: 
                print("ret = {}".format(ret))
                break

            # increasing counter so that it will 
            # show how many frames are created 
            currentframe += 1

            if (currentframe % 100 == 0):
                print("{} frames".format(currentframe))
            #assert currentframe < 10000
