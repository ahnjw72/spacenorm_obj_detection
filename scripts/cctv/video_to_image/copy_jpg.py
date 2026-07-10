import os, subprocess
import shutil

def copy_jpg_to_data(txt_file, working_dir):

    os.chdir(working_dir)

    # Get the filename without extension
    base_filename = os.path.splitext(txt_file)[0]

    # Find the corresponding jpg file
    jpg_file = f"{base_filename}.jpg"

    data_dir = 'data'

    # Check if jpg file exists and has non-zero size
    if os.path.exists(jpg_file):
        data_dir = 'data'
        os.makedirs(data_dir, exist_ok=True)

        # Copy the jpg file to 'data' directory
        shutil.copy(jpg_file, os.path.join(data_dir, jpg_file))
        print(f"Copied {jpg_file} to {data_dir}")

def main():

    # Save the current working directory
    original_working_directory = os.getcwd()

    # Specify the path to the new working directory
    new_working_directory = '/home/cym/Work/spacenorm_yolov7/runs/detect/test2/labels'
    os.chdir(new_working_directory)

    txt_files = [f for f in os.listdir() if f.endswith('.txt')]

    # Process each txt file
    for txt_file in txt_files:
        # Check if txt file has non-zero size
        if os.path.getsize(txt_file) > 0:
            #print(txt_file)
            copy_jpg_to_data(txt_file, original_working_directory)
            os.chdir(new_working_directory)
            shutil.copy(txt_file, os.path.join(original_working_directory, 'data', txt_file))

if __name__ == "__main__":
    main()

