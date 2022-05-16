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

from scripts.backend import load_static_data
from scripts.configs import config

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


if __name__ == "__main__":
    plot_similarity_matrix_aligned("dog")
