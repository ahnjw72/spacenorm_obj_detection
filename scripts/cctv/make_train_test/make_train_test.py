import csv
import hashlib
import os
from PIL import Image
from glob import glob

TEST_FRACTION = 0.25

BASE_DIR = "/home/ahnjw/Work/spacenorm_obj_detection/data/cctv_train_data"
#SET_LIST = ["set01", "set02", "set03", "set04", "set05", "set06", "set07", "set08", "set09", "set10", "set11", "set12", "set13", "set14", "set15", "set16", "set17", "set18"] # train & test data will be created using these sets.
SET_LIST = []

SET_START_NUM = 1
SET_END_NUM = 151

def is_test(set_name, image_name):
    """Deterministic per-image train/test assignment.

    Keyed only on (set_name, image_name), so it's stable across reruns
    regardless of glob() directory-entry order, and stable across future
    reruns that add new sets/images -- unlike sklearn's train_test_split,
    which reassigns everything whenever the input list's order or length
    changes. That instability previously caused ~75% of images to silently
    swap between train/test when the split was regenerated for a wider set
    range, leaking most of the old test set into the new training data.
    """
    digest = hashlib.sha256(f"{set_name}/{image_name}".encode()).hexdigest()
    bucket = int(digest[:8], 16) % 10000
    return bucket < TEST_FRACTION * 10000


def image_ext(set_name, image_name):
    """.jpg for legacy sets, .png for sets staged by dataset_builder (lossless
    pass-1 cache, ALGORITHM.md 3) -- decided per file since old and new sets
    coexist under BASE_DIR."""
    if os.path.exists(os.path.join(BASE_DIR, set_name, image_name + ".png")):
        return ".png"
    return ".jpg"

assert(SET_START_NUM<=99) # just for simplicity..
if SET_END_NUM <= 99:
    for i in range(SET_START_NUM,SET_END_NUM+1):
        set_name = "set{:02d}".format(i)
        print(set_name)
        SET_LIST.append(set_name)
    train_txt_file = "train_{:02d}_to_{:02d}.txt".format(SET_START_NUM, SET_END_NUM)
    test_txt_file = "test_{:02d}_to_{:02d}.txt".format(SET_START_NUM, SET_END_NUM)

else:
    for i in range(SET_START_NUM,100):
        set_name = "set{:02d}".format(i)
        print(set_name)
        SET_LIST.append(set_name)
    for i in range(100,SET_END_NUM+1):
        set_name = "set{:04d}".format(i)
        print(set_name)
        SET_LIST.append(set_name)    
    train_txt_file = "train_{:02d}_to_{:04d}.txt".format(SET_START_NUM, SET_END_NUM)
    test_txt_file = "test_{:02d}_to_{:04d}.txt".format(SET_START_NUM, SET_END_NUM)

with open(train_txt_file, "w") as train_file, open(test_txt_file, "w") as test_file:
    for set in SET_LIST:
        print(f"SET = {set}")
        txt_files = glob(os.path.sep.join([BASE_DIR, set, "*.txt"]))
        image_name_list = sorted((txt_file.split('/')[-1]).split('.')[0] for txt_file in txt_files)

        image_names_train = [name for name in image_name_list if not is_test(set, name)]
        image_names_test = [name for name in image_name_list if is_test(set, name)]
        print(len(image_names_train))
        print(len(image_names_test))
        for images in image_names_train:
            train_file.write("data/cctv_train_data/{}/{}".format(set, images) + image_ext(set, images) + "\n")
        for images in image_names_test:
            test_file.write("data/cctv_train_data/{}/{}".format(set, images) + image_ext(set, images) + "\n")

print(train_txt_file)
print(test_txt_file)

