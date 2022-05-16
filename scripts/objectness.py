import os
import numpy as np
import scipy.io as sio
import shutil
from sklearn import manifold
import matplotlib.pyplot as plt
from PIL import Image

from scripts.configs import config
from scripts.backend_model import get_tsne_from_similarity_matrix
from scripts.crowd_data import CrowdData
from scripts.backend import decom_similarity_matrix

ROOT = "D:\CrowdSourcing2018\Codes\mingming\objectness"


def mat_map_2_image_map(image_root, image_name):
    mat_input = os.path.join(image_root, image_name + "_blob_0.mat")
    image_input = os.path.join(image_root, image_name + ".jpg")
    blend_image_filename = os.path.join(image_root, image_name + "_b.jpg")
    segmentation_filename = os.path.join(image_root, image_name + "_s.jpg")
    mat = sio.loadmat(mat_input)
    data = mat["data"]
    data = data.reshape(data.shape[:3])
    data = data.transpose(1, 0, 2)
    img = Image.open(image_input)
    img_data = np.array(img)
    img_row = img_data.shape[0]
    img_col = img_data.shape[1]
    data_row = data.shape[0]
    data_col = data.shape[1]
    row = img_row if img_row < data_row else data_row
    col = img_col if img_col < data_col else data_col
    data = data[:row, :col, :]
    if row < img_row:
        # img_data = np.array(img.resize((row, col)))
        img_data = img_data[img_row // 2 - row // 2: img_row // 2 + row // 2 + 1,
                   img_col // 2 - col // 2: img_col // 2 + 1 + col // 2, :]
    segmentation = data.argmax(axis=2) * 250
    segmentation = segmentation.astype(np.uint8)

    alpha = 0.5
    seg_indicator = segmentation > 0
    blend = np.array([0, 255, 0]).reshape(-1, 3). \
        repeat(axis=0, repeats=row * col).reshape(row, col, -1)
    blend_img = img_data.copy()
    blend_img[seg_indicator, :] = alpha * img_data[seg_indicator, :] + (1 - alpha) * blend[seg_indicator, :]
    # objectness = data[:,:,1]
    result = Image.fromarray(blend_img)
    result.save(blend_image_filename)
    result = Image.fromarray(segmentation)
    result.save(segmentation_filename)


def get_segmentation(image_list_filename, image_root):
    image_list = open(image_list_filename, "r").read().split("\n")[:-1]
    image_list = [s.strip("/").split(".")[0] for s in image_list]
    for name in image_list:
        mat_map_2_image_map(image_root, name)


if __name__ == "__main__":
    image_category_name = "image_all_result"
    # mat_map_2_image_map(os.path.join(ROOT, "images"),"n02129530_1632")
    get_segmentation(os.path.join(ROOT, image_category_name, "image_list.txt"),
                     os.path.join(ROOT, image_category_name))
