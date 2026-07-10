import os
import shutil

# Define the source and destination directories
src_dir = './'  # Current directory
dest_dir = './temp/'  # Replace with your desired target directory
img_dir = '/home/cym/Work/spacenorm_yolov7/scripts/cctv/video_to_image/02.images/yujin_all'

# Create destination directory if it doesn't exist
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# Iterate over all files in the source directory
for file_name in os.listdir(src_dir):
    # Check if the file is a .txt file
    if file_name.endswith('.txt'):
        file_path = os.path.join(src_dir, file_name)
        # Check if the file has non-zero size
        if os.path.getsize(file_path) > 0:
            # Copy the file to the destination directory
            shutil.copy(file_path, dest_dir)
            print(f"Copied: {file_name}")
            src_img_file_name = file_name.split('.')[0] + '.jpg'
            src_img_path = os.path.join(img_dir, src_img_file_name)
            #print(src_img_path)
            shutil.copy(src_img_path, dest_dir)
            print(f"Copied: {src_img_file_name}")
        #else:
            #print(f"Skipped (zero size): {file_name}")

