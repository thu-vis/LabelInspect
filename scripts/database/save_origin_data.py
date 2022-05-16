# coding: utf-8
# create by Changjian 11/1/2018

import numpy as np
import os
import scipy.io as sio
import json
import time
import shutil

from matplotlib import pyplot as plt
from scripts.backend import load_static_data, load_manifest_data, get_time_string
from scripts.configs import config
from scripts.crowd import mmcrowd
from scripts.database.extract_similarity import extract_similarity

def get_name_2_id_map():
    filename = os.path.join(config.data_root, config.bird_dataset_name,
                            config.origin_data, "image_map.txt")
    lines = open(filename,"r").read().split("\n")[:-1]
    name_2_id_map = {}
    for s in lines:
        id, image_name = s.split(" : ")
        name_2_id_map[image_name] = int(id)
    return name_2_id_map
def get_name_2_index_map():
    name_2_id_map = get_name_2_id_map()
    id = []
    for image_name in name_2_id_map:
        id.append(name_2_id_map[image_name])
    id = np.array(id)
    order = id.argsort()
    name_2_index = {}
    id_2_index = {}
    for i in range(id.shape[0]):
        id_2_index[id[order[i]]] = i
    for image_name in name_2_id_map:
        name_2_index[image_name] = id_2_index[name_2_id_map[image_name]]
    return name_2_index

class Database(object):
    def __init__(self, dataname, rectify=False):
        self.dataname = dataname
        self.root = os.path.join(config.data_root, dataname, config.origin_data)
        self.manifest_data_path = os.path.join(self.root, config.manifest_name)
        self.static_info_path = os.path.join(self.root, config.static_info_name)
        self.crowd_data_path = os.path.join(self.root, config.crowd_data_name)

    #load origin data from a old format
    def old_load_data(self):
        manifest_data = json.load(open(self.manifest_data_path,"r"))
        static_info = json.load(open(self.static_info_path,"r"))
        crowd_data = {}
        for info in manifest_data:
            crowd_data[info] = manifest_data[info]
        for info in static_info:
            crowd_data[info] = static_info[info]
        self.crowd_data = crowd_data

    def load_data(self, filename = None):
        if filename is None:
            self.crowd_data = json.load(open(self.crowd_data_path,"r"))
        else:
            self.crowd_data = json.load(open(filename,"r"))

    def old_key_rename(self):
        if not hasattr(self,"crowd_data"):
            self.old_load_data()
        if "L" in self.crowd_data:
            self.crowd_data["WorkerLabels"] = self.crowd_data["L"]
            del(self.crowd_data["L"])
        if "ItemTotalNum" in self.crowd_data:
            self.crowd_data["InstanceTotalNum"] = self.crowd_data["ItemTotalNum"]
            del(self.crowd_data["ItemTotalNum"])
        true_labels = np.array(self.crowd_data["true_labels"]).reshape(-1)
        # make sure category range from 0
        if true_labels.min() > 0:
            true_labels = true_labels - 1
        #TODO: make sure labels are shuffled

        self.crowd_data["true_labels"] = true_labels.tolist()

    def change_label(self):
        date = get_time_string()[:10]
        filename = os.path.join(config.data_root,
                                config.bird_dataset_name,
                                config.origin_data,
                                "change_" + date + ".txt")
        if not os.path.exists(filename):
            raise ValueError("you should manually the change filename"
                             " according to date today")
        lines = open(filename,"r").read().split("\n")[:-1]
        name_2_index = get_name_2_index_map()
        true_labels = self.crowd_data[config.true_labels_name]
        for s in lines:
            image_name, label = s.split(" ")
            label = int(label)
            if label > -1:
                true_labels[name_2_index[image_name]] = label
                print(name_2_index[image_name], label)
        self.crowd_data[config.true_labels_name] = true_labels

    def directly_change_label(self):
        true_labels = self.crowd_data[config.true_labels_name]
        if self.dataname == config.bird_dataset_name:
            true_labels[71-1] = 4
            true_labels[232-1] = 4
            true_labels[1438-1] = 4
            true_labels[1482-1] = 6
            true_labels[1588-1] = 4
            true_labels[1818-1] = 4
            true_labels[403-1] = 4
            true_labels[486-1] = 4
            true_labels[1941-1] = 4

    def dog_dataset_chen_labels(self):
        if self.dataname != config.dog_dataset_name:
            print("dog_dataset_chen_label only works in dog dataset!!")
            return
        chen_labels_filepath = os.path.join(config.row_data_root,
                                            config.dog_dataset_name,
                                            "chen_labels.npy")
        chen_labels = np.load(chen_labels_filepath)


    def save_crowd_data(self, commit):
        if not hasattr(self,"crowd_data"):
            self.old_load_data()
        self.old_key_rename()
        print("information that was saved:")

        # check if there exist crowd_data before
        if os.path.exists(self.crowd_data_path):
            # if crowd_data exists, back up it and writes a commit file
            t = "." + get_time_string()
            backup_crowd_data_path = self.crowd_data_path + t
            shutil.copy(self.crowd_data_path, backup_crowd_data_path)
            commit_path =os.path.join(self.root, "commit" + t)
            open(commit_path,"w").writelines(commit)

        # make fake similarity matrix
        # if self.crowd_data[config.similarity_matrix_name] == "None":
            # instances_num = self.crowd_data[config.instance_num_name]
            # element_num = instances_num * (instances_num-1) / 2
            # print(element_num)
            # simi = np.random.rand( int(element_num) )
            # self.crowd_data[config.similarity_matrix_name] = simi.tolist()
        simi = extract_similarity(self.dataname)
        self.crowd_data[config.similarity_matrix_name] = simi

        for info in self.crowd_data:
            print("-- " + info +" --")
        open(self.crowd_data_path,"w").write(json.dumps(self.crowd_data))
        print("information has been saved in %s" %(self.crowd_data_path))

if __name__ == "__main__":
    crowd_data = Database(config.dog_dataset_name)
    crowd_data.load_data()
    # crowd_data.change_label()
    # crowd_data.directly_change_label()

    crowd_data.old_key_rename()
    crowd_data.save_crowd_data("directly label change")