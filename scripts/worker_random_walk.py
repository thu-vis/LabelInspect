import numpy as np
import os
import scipy.io as sio
from time import time
from matplotlib import pyplot as plt
from easydict import EasyDict as edict

from scripts.backend import load_static_data, load_manifest_data
from scripts.crowd_data import CrowdData
from scripts.configs import config
from scripts.backend import load_static_data, decom_similarity_matrix
# from scripts.test.crowd_spammer_propagation import mmcrowd
from scripts.backend_model import BackendModel
from concurrent.futures import ProcessPoolExecutor

class WorkerSimi(object):
    def __init__(self, dataname):
        self.model = BackendModel(dataname)
        self.model.load_model()

    def common_count_simi(self, type = 0):
        instance_num = self.get_attr(config.instance_num_name)
        worker_num = self.get_attr(config.worker_num_name)
        label_num = self.get_attr(config.label_num_name)
        worker_labels = np.array(self.get_attr(config.workers_labels_name))\
            .reshape(instance_num, worker_num)
        neib_worker = []
        for i in range(worker_num):
            neib_worker.append([])
        for i in range(instance_num):
            for j, label in enumerate(worker_labels[i,:]):
                if label != -1 and ( label == 2*type or label == (2 * type + 1) ):
                    neib_worker[j].append(i)
                else:
                    worker_labels[i,j] = -1

        similarity = np.zeros((worker_num, worker_num))
        for i in range(worker_num):
            for j in range(i+1, worker_num):
                intersection = set(neib_worker[i]).intersection(neib_worker[j])
                if i == 16 and j == 35:
                    a =1
                similarity[i][j] = len(intersection)
                similarity[j][i] = similarity[i][j]
        return similarity

    def adamic_simi(self, type = 0):
        instance_num = self.get_attr(config.instance_num_name)
        worker_num = self.get_attr(config.worker_num_name)
        label_num = self.get_attr(config.label_num_name)
        worker_labels = np.array(self.get_attr(config.workers_labels_name))\
            .reshape(instance_num, worker_num).copy()
        neib_worker = []
        for i in range(worker_num):
            neib_worker.append([])
        for i in range(instance_num):
            for j, label in enumerate(worker_labels[i,:]):
                if label != -1 and ( label == 2*type or label == (2 * type + 1) ):
                    neib_worker[j].append(i)
                else:
                    worker_labels[i,j] = -1

        similarity = np.zeros((worker_num, worker_num))
        for i in range(worker_num):
            for j in range(i+1, worker_num):
                common_item = 0
                intersection = set(neib_worker[i]).intersection(neib_worker[j])
                # for s in intersection:
                #     print(worker_labels[s,i],end=" ")
                # exit()
                if i == 16 and j == 35:
                    a =1
                for item in intersection:
                    if worker_labels[item][i] == worker_labels[item][j]:
                        common_item = common_item + 1
                        similarity[i][j] += 1.0 / np.log(
                            np.sum(worker_labels[item,:] == worker_labels[item][i])
                        )
                similarity[i][j] /= float(common_item+1)
                similarity[i][j] *= float(len(intersection))
                similarity[j][i] = similarity[i][j]
        return similarity

    def confusion_matrix_simi(self, type=0):
        instance_num = self.get_attr(config.instance_num_name)
        worker_num = self.get_attr(config.worker_num_name)
        label_num = self.get_attr(config.label_num_name)
        worker_labels = np.array(self.get_attr(config.workers_labels_name))\
            .reshape(instance_num, worker_num)
        posterior_labels = self.model.crowd_instances.get_posterior_labels()
        confusion_matrices = self.model.model.get_confusion_matrices()
        similarity = np.zeros((worker_num, worker_num))
        for i in range(worker_num):
            for j in range(i+1, worker_num):
                cm_i = confusion_matrices[:,:,i][type*2:type*2+2,:][:,type*2:type*2+2]
                cm_j = confusion_matrices[:,:,j][type*2:type*2+2,:][:,type*2:type*2+2]
                similarity[i,j] = (1 - ((cm_i - cm_j) ** 2).sum())**2
                similarity[j,i] = similarity[i,j]
        return similarity, confusion_matrices

    def random_walk_simi(self, type=0, d=0.3, validation = None):
        instance_num = self.get_attr(config.instance_num_name)
        worker_num = self.get_attr(config.worker_num_name)
        label_num = self.get_attr(config.label_num_name)
        worker_labels = np.array(self.get_attr(config.workers_labels_name))\
            .reshape(instance_num, worker_num)
        spammer_score = self.model.crowd_workers.\
            get_spammer_scores_according_to_selected_list([type*2, type*2+1])
        spammer_score = spammer_score / spammer_score.sum()
        print(spammer_score.argsort()[::-1][:40])
        print(spammer_score[spammer_score.argsort()[::-1][:40]])
        print(spammer_score[32])
        simi = self.adamic_simi(type)
        c_simi, _ = self.confusion_matrix_simi(type)
        # simi = c_simi * simi
        # print(spammer_score)
        state = np.ones(worker_num) / float(worker_num)
        # aggregate expert validation
        if validation is not None:
            valid_spammer_score = np.zeros(worker_num)
            valid_state = np.zeros(worker_num)
            average = 1.0 / len(validation)
            for valid_id in validation:
                valid_spammer_score[valid_id] = average
                valid_state[valid_id] = average
            spammer_score = 0.5 * spammer_score + 0.5 * valid_spammer_score
            state = 0.5 * state + 0.5 * valid_state
        for i in range(1000):
            state = (1-d) * spammer_score + d * np.dot(simi, state/(simi.sum(axis=1) + 1))
        return state

    def show_top(self,metric, worker_id, top=60):
        score = metric[worker_id,:]
        order = np.array(score).argsort()[::-1][:top]
        top_score = score[order]
        print(order)
        print(top_score)

    def get_attr(self,attr):
        return self.model.crowd_data.get_attr(attr)


def get_most_common_of_one_worker(worker_id, type = 0, dataname = "dog"):
    crowd_data = CrowdData(dataname)
    mm = mmcrowd()
    mm.from_crowd_data(crowd_data)
    simi_matrix = crowd_data.get_attr(config.similarity_matrix_name)
    worker_num = crowd_data.get_attr(config.worker_num_name)
    instance_num = crowd_data.get_attr(config.instance_num_name)
    label_num = crowd_data.get_attr(config.label_num_name)
    worker_labels = crowd_data.get_attr(config.workers_labels_name)
    worker_labels = np.array(worker_labels).reshape(instance_num, worker_num)
    simi_matrix = decom_similarity_matrix(simi_matrix, instance_num)

    self_labels = worker_labels[:,worker_id]
    common_label = np.zeros(worker_num)
    self_indicator = (self_labels > 0) * ((self_labels == (2 * type + 1)) + (self_labels == (2 * type + 2)))
    for i in range(worker_num):
        if i == worker_id:
            common_label[i] = 0
            continue
        labels = worker_labels[:, i]
        indicator = (labels > 0) * ( (labels == ( 2 * type + 1)) + ( labels == (2 * type + 2)) )
        indicator_common = (self_indicator * indicator)
        common_label[i] = sum(indicator)
    print(common_label)
    print(common_label.argsort()[::-1][:20])

def case_test():
    type = 2
    worker_id = 1
    w = WorkerSimi(config.bird_dataset_name)
    adamic_simi = w.adamic_simi(type=type)
    common_count_simi = w.common_count_simi(type=type)
    confusion_matrix_simi, confusion_matrices = w.confusion_matrix_simi(type=type)
    mix = adamic_simi * confusion_matrix_simi
    w.show_top(common_count_simi, worker_id)
    w.show_top(adamic_simi, worker_id)
    w.show_top(confusion_matrix_simi, worker_id)
    # w.show_top(mix, 35)
    random_walk_simi = w.random_walk_simi(type=type)
    order = random_walk_simi.argsort()[::-1][:60]
    print(order)
    print(random_walk_simi[order])
    a = 1

def bird_validatioin_list():
    validation = {

    }

def bird_spammer_test():
    dataname = config.bird_dataset_name
    w = WorkerSimi(dataname)
    crowd_data = CrowdData(dataname)
    mm = mmcrowd()
    mm.from_crowd_data(crowd_data)
    valid_info = [
        {"worker_id":[13,22,19,35], "number": 20},
        {"worker_id":[12,35], "number": 12},
        {"worker_id":None, "number": 10},
        {"worker_id":None, "number": 25},
        {"worker_id":None, "number": 25},
    ]
    total_expert_info = []
    for type in range(5):
        worker_id = valid_info[type]["worker_id"]
        number = valid_info[type]["number"]
        random_walk_simi = w.random_walk_simi(type=type, validation=worker_id)
        order = random_walk_simi.argsort()[::-1][:number]
        class_list = [type * 2 + 1, type * 2 + 2]
        for worker_id in order:
            total_expert_info.append({
                "worker_id" : worker_id,
                "class_list": class_list
            })
    for record in total_expert_info:
        print(record,end=",\n")
    # total_expert_info = [
    #     {'worker_id': 35, 'class_list': [1, 2]},
    #     {'worker_id': 19, 'class_list': [1, 2]},
    #     {'worker_id': 22, 'class_list': [1, 2]},
    #     {'worker_id': 13, 'class_list': [1, 2]},
    #     {'worker_id': 54, 'class_list': [1, 2]},
    #     {'worker_id': 4, 'class_list': [1, 2]},
    #     # {'worker_id': 53, 'class_list': [1, 2]},
    #     # {'worker_id': 55, 'class_list': [1, 2]},
    #     # {'worker_id': 11, 'class_list': [1, 2]},
    #     {'worker_id': 1, 'class_list': [1, 2]},
    #     # {'worker_id': 49, 'class_list': [1, 2]},
    #     # {'worker_id': 62, 'class_list': [1, 2]},
    #     # {'worker_id': 0, 'class_list': [1, 2]},
    #     {'worker_id': 52, 'class_list': [1, 2]},
    #     # {'worker_id': 10, 'class_list': [1, 2]},
    #     # {'worker_id': 8, 'class_list': [1, 2]},
    #     {'worker_id': 51, 'class_list': [1, 2]},
    #     # {'worker_id': 50, 'class_list': [1, 2]},
    #     # {'worker_id': 61, 'class_list': [1, 2]},
    #     {'worker_id': 64, 'class_list': [1, 2]},
    #     {'worker_id': 12, 'class_list': [3, 4]},
    #     {'worker_id': 35, 'class_list': [3, 4]},
    #     {'worker_id': 4, 'class_list': [3, 4]},
    #     {'worker_id': 50, 'class_list': [3, 4]},
    #     {'worker_id': 0, 'class_list': [3, 4]},
    #     {'worker_id': 11, 'class_list': [3, 4]},
    #     {'worker_id': 1, 'class_list': [3, 4]},
    #     {'worker_id': 54, 'class_list': [3, 4]},
    #     {'worker_id': 55, 'class_list': [3, 4]},
    #     {'worker_id': 39, 'class_list': [3, 4]},
    #     {'worker_id': 49, 'class_list': [3, 4]},
    #     {'worker_id': 64, 'class_list': [3, 4]},
    #     {'worker_id': 55, 'class_list': [5, 6]},
    #     {'worker_id': 54, 'class_list': [5, 6]},
    #     {'worker_id': 53, 'class_list': [5, 6]},
    #     {'worker_id': 11, 'class_list': [5, 6]},
    #     {'worker_id': 4, 'class_list': [5, 6]},
    #     # {'worker_id': 10, 'class_list': [5, 6]},
    #     {'worker_id': 49, 'class_list': [5, 6]},
    #     {'worker_id': 48, 'class_list': [5, 6]},
    #     {'worker_id': 6, 'class_list': [5, 6]},
    #     {'worker_id': 8, 'class_list': [5, 6]},
    #     {'worker_id': 1, 'class_list': [7, 8]},
    #     {'worker_id': 53, 'class_list': [7, 8]},
    #     {'worker_id': 54, 'class_list': [7, 8]},
    #     {'worker_id': 55, 'class_list': [7, 8]},
    #     {'worker_id': 63, 'class_list': [7, 8]},
    #     {'worker_id': 62, 'class_list': [7, 8]},
    #     {'worker_id': 4, 'class_list': [7, 8]},
    #     {'worker_id': 15, 'class_list': [7, 8]},
    #     # {'worker_id': 11, 'class_list': [7, 8]},
    #     {'worker_id': 64, 'class_list': [7, 8]},
    #     # {'worker_id': 28, 'class_list': [7, 8]},
    #     {'worker_id': 51, 'class_list': [7, 8]},
    #     {'worker_id': 38, 'class_list': [7, 8]},
    #     # {'worker_id': 29, 'class_list': [7, 8]},
    #     {'worker_id': 52, 'class_list': [7, 8]},
    #     {'worker_id': 45, 'class_list': [7, 8]},
    #     {'worker_id': 25, 'class_list': [7, 8]},
    #     {'worker_id': 39, 'class_list': [7, 8]},
    #     {'worker_id': 26, 'class_list': [7, 8]},
    #     {'worker_id': 35, 'class_list': [7, 8]},
    #     {'worker_id': 36, 'class_list': [7, 8]},
    #     {'worker_id': 9, 'class_list': [7, 8]},
    #     {'worker_id': 2, 'class_list': [7, 8]},
    #     # {'worker_id': 27, 'class_list': [7, 8]},
    #     # {'worker_id': 33, 'class_list': [7, 8]},
    #     {'worker_id': 11, 'class_list': [9, 10]},
    #     {'worker_id': 53, 'class_list': [9, 10]},
    #     {'worker_id': 55, 'class_list': [9, 10]},
    #     {'worker_id': 54, 'class_list': [9, 10]},
    #     # {'worker_id': 1, 'class_list': [9, 10]},
    #     {'worker_id': 62, 'class_list': [9, 10]},
    #     {'worker_id': 63, 'class_list': [9, 10]},
    #     {'worker_id': 64, 'class_list': [9, 10]},
    #     # {'worker_id': 20, 'class_list': [9, 10]},
    #     {'worker_id': 19, 'class_list': [9, 10]},
    #     {'worker_id': 2, 'class_list': [9, 10]},
    #     {'worker_id': 15, 'class_list': [9, 10]},
    #     {'worker_id': 37, 'class_list': [9, 10]},
    #     {'worker_id': 51, 'class_list': [9, 10]},
    #     {'worker_id': 26, 'class_list': [9, 10]},
    #     # {'worker_id': 4, 'class_list': [9, 10]},
    #     {'worker_id': 27, 'class_list': [9, 10]},
    #     {'worker_id': 52, 'class_list': [9, 10]},
    #     {'worker_id': 39, 'class_list': [9, 10]},
    #     {'worker_id': 30, 'class_list': [9, 10]},
    #     {'worker_id': 45, 'class_list': [9, 10]},
    #     {'worker_id': 29, 'class_list': [9, 10]},
    #     # {'worker_id': 57, 'class_list': [9, 10]},
    #     # {'worker_id': 10, 'class_list': [9, 10]},
    #     {'worker_id': 41, 'class_list': [9, 10]},
    # ]
    mm.expert_info_takein(total_expert_info, crowd_data.get_attr(config.label_num_name)//2)
    mm.train()

if __name__ == "__main__":
    # get_most_common_of_one_worker(21)
    # bird_spammer_test()
    case_test()