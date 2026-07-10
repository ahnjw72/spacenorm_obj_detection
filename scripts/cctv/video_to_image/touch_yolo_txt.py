#!/usr/bin/env python
# coding: utf-8

# In[28]:

import os
import glob
import subprocess

IMAGE_BASE_DIR="02.images"

#sets = os.listdir(INPUT_VIDEO_BASE_DIR)
#sets = ["set0107", "set0108"]
sets = ["test"]

for image_set in sets:
    input_images = os.path.join(IMAGE_BASE_DIR, image_set) + "/*.jpg" 

    print(input_images)
    
    for input_image_path in glob.glob(input_images):
        #print(input_image_path)
        yolo_txt_filename = input_image_path.split("/")[-1].split(".")[0]+".txt"
        #print(yolo_txt_filename)
        yolo_txt_filepath = os.path.join(IMAGE_BASE_DIR, image_set, yolo_txt_filename)
        #print(yolo_txt_filepath)
        touch_command = "touch " + yolo_txt_filepath
        print(touch_command)
        subprocess.call(touch_command, shell=True)
        
