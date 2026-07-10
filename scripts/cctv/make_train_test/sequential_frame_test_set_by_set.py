# coding: utf-8

# In[2]:


# this script generates *.txt file conatining 'sequential' list of image file names
import csv
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from glob import glob

BASE_DIR = "/home/cym/Work/darknet/data/cctv"
#SET_LIST = ["set22", "set23"] # sequential data information will be created using these sets.

SET_LIST = []
for i in range(1,22):
    if i<10:
        set = f"set0{i}"
    else:    
        set = f"set{i}"
    SET_LIST.append(set)

#print(SET_LIST)

for set in SET_LIST:
    test_txt_file = f"seq_test_{set}.txt"
    print(test_txt_file)

    with open(test_txt_file, "w") as test_file:
        image_name_list = []    
        gt_files = sorted(glob(os.path.sep.join([BASE_DIR, set, "*.txt"]))) # gt_files : ["set01_0001.txt", "set01_0002.txt", ...]        
        print("gt_files: ", gt_files)
        for gt_file in gt_files:
            image_name = (gt_file.split('/')[-1]).split('.')[0]
            #print(image_name)
            image_name_list.append(image_name)
                
        for images in image_name_list:        
            #print(images)
            test_file.write("data/cctv/{}/{}".format(set, images) + ".jpg\n")
            #test_file.write("data/cctv/{}/{}".format(set, images) + ".jpg\n")    
            
print("Done.")


