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
from scripts.test.crowd_spammer_propagation import mmcrowd
from scripts.backend_model import BackendModel
from concurrent.futures import ProcessPoolExecutor

class InstanceRandomWalk(object):
    def __init__(self, dataname):
        self.model = BackendModel(dataname)
        self.model.load_model()

    def random_walk(self, type=0, d=0.3, validation = None):
        instance_num = self.get_attr(config.instance_num_name)
        label_num = self.get_attr(config.label_num_name)
        worker_num = self.get_attr(config.worker_num_name)
        similarity_matrix = decom_similarity_matrix(self.get_attr(config.similarity_matrix_name), instance_num)
        # print(similarity_matrix.shape)
        distance_matrix = 1 - similarity_matrix
        uncertainty = self.model.crowd_instances.get_uncertainty()
        uncertainty = np.array(uncertainty)
        uncertainty = uncertainty / uncertainty.sum()
        state = np.ones(instance_num) / float(instance_num)
        if validation is not None:
            for valid_id in validation:
                state[valid_id] = 0
        state = state / state.sum()
        # random walk model iteration
        for i in range(100):
            state = (1 - d) * uncertainty + \
                    d * np.dot(similarity_matrix, state/(similarity_matrix.sum(axis=1) + 1))


    def get_attr(self, attr):
        return self.model.crowd_data.get_attr(attr)

if __name__ == "__main__":
    model = InstanceRandomWalk(config.dog_dataset_name)
    model.random_walk()

