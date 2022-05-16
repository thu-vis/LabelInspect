import numpy as np
import os
import json
from PIL import Image
from matplotlib import pyplot as plt
from scipy.special import comb, perm
from time import time
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.io as sio

from scripts.crowd import CrowdsourcingModel, M3VModel
from scripts.backend import load_static_data, decom_similarity_matrix
from scripts.crowd_data import CrowdData
from scripts.configs import config
from scripts.crowd_worker import CrowdWorkers
from scripts.crowd_instances import CrowdInstances
from scripts.backend_model import BackendModel

def plot_relation(filename):
    json_file = open(filename, 'r')
    data = json.load(json_file)
    score_list = [0, 0, 0, 0, 0]
    score_list[0] = data["WorkerRealiability"]
    score_list[1] = data["WorkerAccuracy"]
    score_list[2] = data["SloppyScore"]
    score_list[3] = data["RandomScore"]
    f1 = plt.figure(1)
    plt.subplot(231)
    plt.scatter(score_list[0], score_list[1])
    ax = plt.gca()
    ax.set_xlabel("1-2")
    plt.subplot(232)
    plt.scatter(score_list[0], score_list[2])
    ax = plt.gca()
    ax.set_xlabel("1-3")
    plt.subplot(233)
    plt.scatter(score_list[0], score_list[3])
    ax = plt.gca()
    ax.set_xlabel("1-4")
    plt.subplot(234)
    plt.scatter(score_list[1], score_list[2])
    ax = plt.gca()
    ax.set_xlabel("2-3")
    plt.subplot(235)
    plt.scatter(score_list[1], score_list[3])
    ax = plt.gca()
    ax.set_xlabel("2-4")
    plt.subplot(236)
    plt.scatter(score_list[2], score_list[3])
    ax = plt.gca()
    ax.set_xlabel("3-4")
    plt.show()

def theory_error_rate(x):
    assert x >= 0
    assert x <= 1
    p = 0.75 * x
    q = 1 - 0.75 * x
    n = 10
    error = 0
    for i in range(n):
        if( i == n/2 ):
            error += 0.5 * comb(n,i) * pow(p,i) * pow(q,n-i)
        elif( i > n/2 ):
            error += comb(n,i) * pow(p,i) * pow(q, n-i)
    return error

def save_confusion_matrix(dataset_name):
    crowd_data = CrowdData(dataset_name, rectify=True)
    crowd_worker = CrowdWorkers(crowd_data)
    for i in range(crowd_data.get_attr(config.worker_num_name)):
    # for i in range(45,46):
        confusion_matrix = crowd_worker._get_ground_truth_confusion_matrix(i)
        df = pd.DataFrame(confusion_matrix)
        plt.figure(figsize=(24,20))
        sns.set(font_scale=2)
        sns.heatmap(df, annot=True, cmap="OrRd")
        plt.savefig(os.path.join(config.data_root, dataset_name,
                                 config.confusion_matrix_data_path, str(i) + ".jpg"),
                    )
        # plt.show()
        plt.close()

def save_estimated_confusion_matrix(dataset_name):
    backend_model = BackendModel(dataset_name)
    crowd_data = backend_model.crowd_data
    crowd_worker = backend_model.crowd_workers
    true_labels = crowd_data.get_attr(config.true_labels_name)
    true_labels = np.array(true_labels)
    posterior_labels = backend_model.model.get_posterior_labels()
    # for i in range(4):
    #     print(sum(true_labels==i), sum(posterior_labels==i))
    # exit()
    for i in range(crowd_data.get_attr(config.worker_num_name)):
    # for i in range(45,46):
        confusion_matrix = crowd_worker._get_posterior_confusion_matrix(i)
        df = pd.DataFrame(confusion_matrix)
        plt.figure(figsize=(24,20))
        sns.set(font_scale=2)
        sns.heatmap(df, annot=True, cmap="OrRd")
        plt.savefig(os.path.join(config.data_root, dataset_name,
                                 "estimated_confusion_count", str(i) + ".jpg"),
                    )
        # plt.show()
        plt.close()

if __name__ == "__main__":
    # plot_relation("../data/dog/info/dynamic_info.json")
    # save_confusion_matrix(config.bird_dataset_name)
    save_estimated_confusion_matrix(config.bird_dataset_name)
    # bm = BackendModel(config.dog_dataset_name)
    # simi = decom_similarity_matrix(bm.crowd_data.get_attr(config.similarity_matrix_name),
    #                                bm.crowd_data.get_attr(config.instance_num_name))
    # sio.savemat(os.path.join(config.data_root,config.dog_dataset_name, "simi.mat"),{"simi":simi})
    # x = []
    # e = []
    # for i in range(100):
    #     x.append(i * 0.01)
    #     e.append(1- theory_error_rate(i * 0.01))
    #
    # plt.plot(x,e)
    # plt.show()