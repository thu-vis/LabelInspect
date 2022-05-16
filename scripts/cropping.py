import os
import numpy as np
import shutil
from sklearn import manifold
import matplotlib.pyplot as plt
from PIL import Image

from scripts.configs import config
from scripts.backend_model import get_tsne_from_similarity_matrix
from scripts.crowd_data import CrowdData
from scripts.backend import decom_similarity_matrix

def cropping_process(cropping_info):
    print(cropping_info)
    dataname = cropping_info["dataname"]
    image_id = cropping_info["id"] + 1
    image_filename = os.path.join(config.data_root,
                                  dataname, config.image_data,
                                  str(image_id) + config.image_ext)
    img = Image.open(image_filename)
    height, width = img.size
    up_left_x = int(cropping_info["up_left"]["x"] * height)
    up_left_y = int(cropping_info["up_left"]["y"] * width)
    down_right_x = int(cropping_info["down_right"]["x"] * height)
    down_right_y = int(cropping_info["down_right"]["y"] * width)
    cropped_image_data = np.array(img)[up_left_y:down_right_y, up_left_x:down_right_x]
    cropped_image = Image.fromarray(cropped_image_data)
    cropped_image.save(image_filename)
    return {
        "feedback":"success"
    }

def border_process(x_min, x_max, y_min, y_max, image_x_max, image_y_max):
    border = 10
    x_min = (x_min - border) if (x_min - border) >= 0 else 0
    x_max = (x_max + border) if (x_max + border) <= image_x_max else image_x_max
    y_min = (y_min - border) if (y_min - border) >= 0 else 0
    y_max = (y_max + border) if (y_max + border) <= image_y_max else image_y_max
    return x_min, x_max, y_min, y_max

def get_border_according_to_mask(mask_data):
    mask_data = mask_data > 0
    mask_data_x = mask_data.sum(axis=1) > 0
    mask_data_y = mask_data.sum(axis=0) > 0
    mask_data_x_indicator = np.array(range(len(mask_data_x)))[mask_data_x]
    mask_data_y_indicator = np.array(range(len(mask_data_y)))[mask_data_y]
    x_min = mask_data_x_indicator.min()
    x_max = mask_data_x_indicator.max()
    y_min = mask_data_y_indicator.min()
    y_max = mask_data_y_indicator.max()
    return x_min, x_max, y_min, y_max


def cropping(origin_image_filename, mask_filename, target_image_filename):
    origin_image = Image.open(origin_image_filename)
    mask = Image.open(mask_filename)
    origin_image_data = np.array(origin_image)
    mask_data = np.array(mask)
    x_min, x_max, y_min, y_max = get_border_according_to_mask(mask_data)
    x_min, x_max, y_min, y_max = border_process(x_min, x_max, y_min, y_max, mask_data.shape[0], mask_data.shape[1])
    # mask_data[x_min, y_min:y_max] = 255
    # mask_data[x_max, y_min:y_max] = 255
    # mask_data[x_min:x_max, y_min] = 255
    # mask_data[x_min:x_max, y_max] = 255
    # mask = Image.fromarray(mask_data)
    # mask.show()
    target_image_data = origin_image_data[x_min:x_max, y_min:y_max, :]
    target_image = Image.fromarray(target_image_data)
    target_image.save(target_image_filename)


def cropping_according_to_mask(input_dir, output_dir):
    all_file_name = os.listdir(input_dir)
    print(len(all_file_name))

    #get image_name
    image_names = []
    for name in all_file_name:
        if name.count("_"):
            continue
        image_names.append(os.path.splitext(name)[0])

    # map image_name and mask_name
    mask_name_map = {}
    for name in all_file_name:
        if name.count("_"):
            image_name = name.split("_")[0]
            mask_name_map[image_name] = name

    #processing image according to mask
    for name in image_names:
        origin_image_filename = os.path.join(input_dir, name + ".jpg")
        mask_filename = os.path.join(input_dir, mask_name_map[name] )
        target_image_filename = os.path.join(output_dir, name + ".jpg")
        cropping(origin_image_filename, mask_filename, target_image_filename)

def cropping_image_list(image_list_filename, image_root, output_dir):
    image_list = open(image_list_filename, "r").read().split("\n")[:-1]
    image_list = [s.strip("/").split(".")[0] for s in image_list]
    for name in image_list:
    # for name in ["1133"]:
        if name in ["1034","1133","821"]:
            continue
        print(name)
        mask_name = name + "_ss.jpg"
        mask_filename = os.path.join(image_root, mask_name)
        origin_image_filename = os.path.join(image_root, name + ".jpg")
        target_image_filename = os.path.join(output_dir, name + ".jpg")
        cropping(origin_image_filename, mask_filename, target_image_filename)


if __name__ == "__main__":
    ROOT = "D:\CrowdSourcing2018\Codes\mingming\objectness"
    category = "image_all"
    image_category_name = category + "_result"
    image_output_name = category + "_cropped"
    # input_dir = os.path.join(ROOT, "bird_swan_mislabeled_filtered")
    # output_dir = os.path.join(ROOT, "bird_swan_mislabeled_cropped")
    # cropping_according_to_mask(input_dir, output_dir)
    cropping_image_list(os.path.join(ROOT, image_category_name, "image_list.txt"),
                        os.path.join(ROOT, image_category_name),
                        os.path.join(ROOT, image_output_name))