import numpy as np
import os

import shutil

ROOT = "../../../RawData/bird_origin_data/for_yexi/"

all_root = os.path.join(ROOT, "all")
train_root = os.path.join(ROOT, "train")
valid_root = os.path.join(ROOT, "validation")
map_path = os.path.join(all_root, "image_map.txt")

def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

def get_validation_path():
    file = open(map_path,"r")
    string_list = file.read()[:-1].split("\n")
    path_list = []
    for s in string_list:
        image_name = s.split(" : ")[1]
        word_id = image_name.split("_")[0]
        path = os.path.join(all_root, word_id, image_name)
        path_list.append(path)
    return path_list

def moving(path_list):
    for path in path_list:
        word_id, image_name = path.split("\\")[-2:]
        train_image_path = os.path.join(train_root, word_id, image_name)
        valid_image_path = os.path.join(valid_root, word_id, image_name)
        check_path(os.path.join(valid_root, word_id))
        shutil.copy(train_image_path, valid_image_path)
        os.remove(train_image_path)


if __name__ == "__main__":
    path_list = get_validation_path()
    moving(path_list)