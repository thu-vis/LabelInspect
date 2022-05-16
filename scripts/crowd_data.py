
import numpy as np
import os
import scipy.io as sio
import json

from matplotlib import pyplot as plt
from scripts.backend import load_static_data, load_manifest_data
from scripts.configs import config
from scripts.crowd import mmcrowd

class CrowdData(object):
    def __init__(self, dataname, rectify=True):
        self.dataname = dataname
        self.root = os.path.join(config.data_root, dataname, config.origin_data)
        self.manifest_data_path = os.path.join(self.root, config.manifest_name)
        self.static_info_path = os.path.join(self.root, config.static_info_name)
        self.crowd_data_path = os.path.join(self.root, config.crowd_data_name)
        self.rectify = rectify

    def rectify_labels(self):
        expert_labels =[
            [90, 95, 162, 230, 236, 404, 556, 657, 717, 771, 795, 19, 282, 558, 639, 755],
            [7, 262, 286, 293, 307, 354, 413, 470, 508, 517, 603, 616, 636, 638, 669, 699, 769, 692],
            [190, 268, 309, 311, 388, 451, 453, 562, 789, 579],
            [25, 641, 697]
        ]
        true_labels = self.crowd_data[config.true_labels_name]
        if self.dataname == config.dog_dataset_name:
            print("data rectify!!!")
            for idx, indexes in enumerate(expert_labels):
                for index in indexes:
                    true_labels[index-1] = idx
            # chen_labels_filepath = os.path.join(config.row_data_root,
            #                                     config.dog_dataset_name,
            #                                     "chen_labels.npy")
            # chen_labels = np.load(chen_labels_filepath)
            # for idx, label in enumerate(chen_labels):
            #     if label > 0:
            #         assert label==2 or label==3
            #         assert true_labels[idx]==2 or true_labels[idx] == 3
            #         true_labels[idx] = label

            #process workers with very little labels
            worker_labels = self.crowd_data[config.workers_labels_name]
            instance_num = self.crowd_data[config.instance_num_name]
            worker_labels = np.array(worker_labels).reshape(instance_num, -1)
            worker_list = (worker_labels!= -1).sum(axis=0) > 40
            self.crowd_data[config.worker_num_name] = int(sum(worker_list))
            worker_labels = worker_labels[:,worker_list]
            print((worker_labels!= -1).sum())
            self.crowd_data[config.workers_labels_name] = worker_labels.reshape(-1).tolist()

            self.crowd_data[config.true_labels_name] = true_labels
    def load_data(self, filename = None):
        if filename is None:
            self.crowd_data = json.load(open(self.crowd_data_path,"r"))
        else:
            self.crowd_data = json.load(open(filename,"r"))
        if self.rectify:
            self.rectify_labels()

    def get_attr(self,attr):
        if not hasattr(self,"crowd_data"):
            self.load_data()
        return self.crowd_data[attr]

    def get_features(self):
        total_feature = np.load(os.path.join(config.row_data_root, config.bird_dataset_name,
                                             "feature/GMP_feature_2000.npy"))
        return total_feature


class CrowdSimulator(object):
    def __init__(self, name):
        self.name = name
        self.reliable_worker_ratio = 0.45
        self.sloppy_worker_ratio = 0.25
        self.uniform_spammer_ratio = 0.15
        self.random_spammer_ratio = 0.15

        self.worker_num = 103
        self.labels_per_instance = 10

        self.reliable_worker_accuracy = 0.8
        self.sloppy_worker_accuracy = 0.4


    def from_crowd_data(self, crowd_data):
        self.true_labels = np.array(crowd_data.get_attr("true_labels")).reshape(-1)
        self.instance_num = self.true_labels.shape[0]
        self.label_num = len(set(self.true_labels.tolist()))
        self.crowd_data = crowd_data

    def generate_simulate_data(self):
        np.random.seed(6)
        reliable_worker_num = int(self.worker_num * self.reliable_worker_accuracy)
        sloppy_worker_num = int(self.worker_num * self.sloppy_worker_ratio)
        uniform_spammer_num = int(self.worker_num * self.uniform_spammer_ratio)
        random_spammer_num = self.worker_num - reliable_worker_num\
                             - sloppy_worker_num - uniform_spammer_num

        # decide workers' type
        self.worker_type = np.zeros(self.worker_num)
        for i in range(self.worker_num):
            while True:
                type = int( np.random.rand() * 100 ) % 4 + 1
                if type == 1 and reliable_worker_num:
                    reliable_worker_num = reliable_worker_num - 1
                    break
                if type == 2 and sloppy_worker_num:
                    sloppy_worker_num = sloppy_worker_num - 1
                    break
                if type == 3 and uniform_spammer_num:
                    uniform_spammer_num = uniform_spammer_num - 1
                    break
                if type == 4 and random_spammer_num:
                    random_spammer_num = random_spammer_num - 1
                    break
            self.worker_type[i] = type

        workers_labels = np.ones((self.instance_num, self.worker_num)) * -1
        workers_labels = workers_labels.astype(int)
        assignment = np.zeros((self.instance_num, self.worker_num))
        for i in range(self.instance_num):
            index = np.array(range(self.worker_num))
            np.random.shuffle(index)
            for j in range(self.labels_per_instance):
                assignment[i,index[j]] = 1

        for i in range(self.instance_num):
            for j in range(self.worker_num):
                if assignment[i,j]:
                    workers_labels[i,j] = self.worker_to_instance(i,j)
        self.workers_labels = workers_labels
        print(workers_labels)

    def load_stimulated_data(self,filename):
        mat = sio.loadmat(filename)
        self.workers_labels = mat["L"].astype(int) - 1
        self.workers_num = self.workers_labels.shape[1]
        self.worker_type = mat["assigned_flag"][:,0]


    def save_data(self):
        filename = os.path.join(config.data_root, self.name, config.origin_data, "stimulated.json")
        crowd_data = {}
        crowd_data["InstanceTotalNum"] = self.instance_num
        crowd_data["WorkerTotalNum"] = self.worker_num
        crowd_data["LabelTotalNum"] = self.label_num
        crowd_data["LabelNames"] = self.crowd_data.crowd_data["LabelNames"]
        crowd_data["SimiGraph"] = self.crowd_data.crowd_data["SimiGraph"]
        crowd_data["WorkerLabels"] = self.workers_labels.tolist()
        crowd_data["true_labels"] = self.true_labels.tolist()
        crowd_data["WorkerType"] = self.worker_type.tolist()
        open(filename, "w").write(json.dumps(crowd_data))

    def worker_to_instance(self,instance_id, worker_id):
        type = self.worker_type[worker_id]
        true_label = self.true_labels[instance_id]
        if type == 1:
            reliability = self.reliable_worker_accuracy
            r = np.random.rand()
            if r < reliability:
                return true_label
            else:
                label = int(np.random.rand() * 100 ) % (self.label_num - 1)
                if label >= true_label:
                    return label + 1
                else:
                    return label
        elif type == 2:
            reliability = self.sloppy_worker_ratio
            r = np.random.rand()
            if r < reliability:
                return true_label
            else:
                label = int(np.random.rand() * 100000) % (self.label_num - 1)
                if label >= true_label:
                    return label + 1
                else:
                    return label
        elif type == 3:
            return worker_id % self.label_num
        else:
            return int(np.random.rand() * 100000 ) % (self.label_num - 1)

    def set_worker_labels(self, worker_labels):
        self.worker_labels = worker_labels
        self.worker_num = worker_labels.shape[1]
        self.instance_num = worker_labels.shape[0]

    def set_true_labels(self, true_labels):
        self.true_labels = np.array(true_labels) - 1

    def save_mat(self,filename):
        m = {}
        m["L"] = self.worker_labels + 1
        m["true_labels"] = self.true_labels + 1
        sio.savemat(filename, m)

    def get_reliability(self):
        if not ( hasattr(self,"worker_labels") and hasattr(self,"true_labels") ):
            raise ValueError("Miss worker_labels or true_labels or both")
        worker_labels = self.worker_labels.transpose()
        label_index = worker_labels > -1
        self.worker_labels_count= label_index.sum(axis=1)
        print(self.worker_labels_count)
        self.reliability = np.zeros(self.worker_num)
        for i in range(self.worker_num):
            self.reliability[i] = sum( worker_labels[i,label_index[i]] == self.true_labels[label_index[i]])
            self.reliability[i] = self.reliability[i] / float( self.worker_labels_count[i])
        print(self.reliability)
        return self.reliability

    def plot_worker_reliability(self):
        if not hasattr(self,"reliability"):
            self.get_reliability()
        fig = plt.figure(1)
        plt.subplot(211)
        ax = fig.add_subplot(211)
        ax.hist(self.worker_labels_count,bins=50)
        plt.title("worker labels count distribution")
        plt.xlabel("labels count")
        plt.ylabel("workers number")
        ax = fig.add_subplot(212)
        ax.hist(self.reliability, bins=10)
        plt.title("worker reliability distribution")
        plt.xlabel("reliability")
        plt.ylabel("workers number")
        plt.show()

if __name__ == "__main__":
    None
#     crowd_data = CrowdData("monkey")
#     crowd_data.load_data()
#     # open(os.path.join(crowd_data.root,"worker_type.json"), "w").write(json.dumps(crowd_data.get_attr("WorkerType")))
#     cs = CrowdSimulator("monkey")
#     cs.from_crowd_data(crowd_data)
#     # cs.generate_simulate_data()
#     cs.load_stimulated_data(os.path.join(config.data_root, crowd_data.dataname, "origin_data/synthetic_data.mat"))
#     cs.save_data()
#     #
#     crowd_data = CrowdData("monkey")
#     crowd_data.load_data(os.path.join(config.data_root, crowd_data.dataname, "origin_data/stimulated.json"))
#     crowd_data.save_crowd_data()
#     # mm = mmcrowd()
#     # mm.from_crowd_data(crowd_data)
#     # mm.train()
#     # # exit()