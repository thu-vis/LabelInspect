from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
import json
from matplotlib import offsetbox
from sklearn import manifold, datasets, decomposition
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse import csc_matrix

from scripts.backend_model import get_tsne_from_similarity_matrix
from scripts.backend import load_static_data, decom_similarity_matrix
from scripts.configs import config
from scripts.crowd_data import CrowdData
from scripts.database.extract_similarity import extract_similarity, mixed_similarity

from sklearn.cluster.bicluster import SpectralBiclustering, SpectralCoclustering



def plot_similarity_matrix_aligned(dataname):
    filename = os.path.join("D:\CrowdSourcing2018\RawData", dataname, "info\static.info")
    true_labels_filename = os.path.join("D:\CrowdSourcing2018\RawData", dataname, "crowdModel\\true_labels.mat")
    dynamic_filename = os.path.join("D:\CrowdSourcing2018\Project\data", dataname, "info\\dynamic_info.json")
    dynamic_info = load_static_data(dynamic_filename)
    # posterior = dynamic_info["Posterior"]
    d = {}
    fi = open(filename, 'r')
    line = fi.read().split('\n')
    d["ItemTotalNum"] = int(line[0])
    # posterior = np.array(posterior).reshape(d["ItemTotalNum"], -1)
    # posterior = posterior.argmax(axis=1)
    complete_simi_graph = np.ones((d["ItemTotalNum"], d["ItemTotalNum"]))
    for i in range(5, 5 + d["ItemTotalNum"] - 1):
        s = i - 5
        m = map(eval, line[i].split(' '))
        for t, j in enumerate(m):
            complete_simi_graph[s, s + t + 1] = j
            complete_simi_graph[s + t + 1, s] = j
    true_labels = sio.loadmat(true_labels_filename)
    true_labels = true_labels["true_labels"].reshape(-1)

    # seriation = SpectralCoclustering(n_clusters=4)
    # seriation.fit(complete_simi_graph)
    # row_ind, col_ind = seriation.get_indices(0)
    # complete_simi_graph = complete_simi_graph[row_ind,:]
    # complete_simi_graph = complete_simi_graph[:,col_ind]
    # print(complete_simi_graph.shape)

    # bi_simi = (complete_simi_graph > 0.02)
    # sparse_simi = csc_matrix(bi_simi)
    # perm = reverse_cuthill_mckee(sparse_simi)
    # # complete_simi_graph = complete_simi_graph[perm, perm]
    # complete_simi_graph = complete_simi_graph[perm,:]
    # complete_simi_graph = complete_simi_graph[:,perm]
    # complete_simi_graph = bi_simi[perm,:]
    # complete_simi_graph = complete_simi_graph[:,perm]

    reorder_path = os.path.join(config.data_root, dataname,"reorder.json")
    mat = json.load(open(reorder_path,"r"))
    reorder = mat["reorder"]
    reorder = np.array(reorder).reshape(-1) - 1

    print(true_labels[reorder])

    # order = true_labels.argsort()
    order = reorder
    complete_simi_graph = complete_simi_graph[order,:]
    complete_simi_graph = complete_simi_graph[:,order]

    plt.matshow(complete_simi_graph, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.show()

def plot_similarity_matrix_from_crowd_data(dataname):
    crowd_data = CrowdData(dataname)
    simi_matrix = crowd_data.get_attr(config.similarity_matrix_name)
    true_labels = crowd_data.get_attr(config.true_labels_name)
    simi_matrix = decom_similarity_matrix(simi_matrix, crowd_data.get_attr(config.instance_num_name))
    true_labels = np.array(true_labels).reshape(-1)

    order = true_labels.argsort()
    simi_matrix = simi_matrix[order,:]
    simi_matrix = simi_matrix[:,order]
    plt.matshow(simi_matrix, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.show()

def plot_similarity_matrix_from_features_monkey():
    simi_matrix = extract_similarity(config.monkey_dataset_name)
    simi_matrix = decom_similarity_matrix(simi_matrix, 1000)

    simi_matrix = simi_matrix[:957,:957]
    plt.matshow(simi_matrix, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.show()

def plot_similarity_matrix_from_features_bird():
    crowd_data = CrowdData(config.bird_dataset_name)

    simi_matrix = extract_similarity(config.bird_dataset_name)

    simi_matrix = decom_similarity_matrix(simi_matrix, 2000)
    true_labels = crowd_data.get_attr(config.true_labels_name)
    true_labels = np.array(true_labels).reshape(-1)

    order = true_labels.argsort()
    simi_matrix = simi_matrix[order, :]
    simi_matrix = simi_matrix[:, order]
    plt.matshow(simi_matrix, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.show()

def plot_tsne_result_of_monkey():
    crowd_data = CrowdData(config.monkey_dataset_name)
    true_labels = crowd_data.get_attr(config.true_labels_name)
    simi_matrix = crowd_data.get_attr(config.similarity_matrix_name)
    simi_matrix = decom_similarity_matrix(simi_matrix, 957)
    tsne = get_tsne_from_similarity_matrix(simi_matrix)

    color_map = plt.get_cmap("tab10")(true_labels)
    print(color_map)
    plt.scatter(tsne[:,0], tsne[:,1],s=6,c=color_map)
    plt.axis("tight")
    plt.show()

    print(true_labels)

def plot_tsne_result_of_bird():
    crowd_data = CrowdData(config.bird_dataset_name)

    simi_matrix = extract_similarity(config.bird_dataset_name)

    simi_matrix = decom_similarity_matrix(simi_matrix, 2000)
    true_labels = crowd_data.get_attr(config.true_labels_name)
    true_labels = np.array(true_labels).reshape(-1)
    true_labels = (true_labels + 6) % 10

    tsne = get_tsne_from_similarity_matrix(simi_matrix)

    color_map = plt.get_cmap("tab10")(true_labels)
    print(color_map)
    plt.scatter(tsne[:,0], tsne[:,1],s=6, c=color_map)
    # for i in range(len(true_labels)):
    #     if true_labels[i] == 2 or true_labels[i] == 3:
    #         plt.text(tsne[i,0], tsne[i,1], str(i+1), family='serif', style='italic', ha='right', wrap=True)
    plt.axis("tight")
    plt.show()

if __name__ == "__main__":
    # plot_similarity_matrix_from_features_monkey()
    # plot_tsne_result_of_bird()
    plot_tsne_result_of_monkey()
    # plot_similarity_matrix_from_crowd_data(config.dog_dataset_name)
