from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
from scipy.cluster.hierarchy import dendrogram, linkage
import json
from matplotlib import offsetbox
from sklearn import manifold, datasets, decomposition
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse import csc_matrix

import warnings
# warnings.filterwarnings("error")

# import community
# import networkx as nx


M1 = [[0.77, 0.26, 0.02, 0.01],
      [0.22, 0.72, 0.01, 0],
      [0.01, 0.01, 0.83, 0.26],
      [0.01, 0, 0.14, 0.72]]

M2 = [[0.69, 0.22, 0.03, 0.04, 0.03, 0.03, 0.02, 0.03, 0.03, 0.02],
      [0.21, 0.73, 0.02, 0.01, 0.00, 0.00, 0.01, 0.00, 0.01, 0.00],
      [0.02, 0.01, 0.79, 0.15, 0.01, 0.00, 0.01, 0.00, 0.00, 0.00],
      [0.02, 0.01, 0.14, 0.76, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
      [0.02, 0.01, 0.01, 0.01, 0.61, 0.29, 0.02, 0.02, 0.01, 0.01],
      [0.02, 0.01, 0.01, 0.01, 0.29, 0.62, 0.04, 0.05, 0.01, 0.00],
      [0.01, 0.01, 0.00, 0.00, 0.02, 0.02, 0.66, 0.15, 0.01, 0.01],
      [0.01, 0.00, 0.00, 0.01, 0.01, 0.02, 0.24, 0.74, 0.01, 0.01],
      [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.71, 0.26],
      [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.22, 0.68]]

# def segmentation(dataname):
#     crowd_data = CrowdData(dataname)
#     true_labels = np.array(crowd_data.get_attr(config.true_labels_name))
#     complete_simi_graph = crowd_data.get_attr(config.similarity_matrix_name)
#     complete_simi_graph = decom_similarity_matrix(
#         complete_simi_graph,
#         crowd_data.get_attr(config.instance_num_name))
#
#     reorder_path = os.path.join(config.data_root, dataname, "reorder.json")
#     mat = json.load(open(reorder_path, "r"))
#     reorder = mat["reorder"]
#     reorder = np.array(reorder).reshape(-1) - 1
#
#     print(true_labels[reorder])
#
#     order = true_labels.argsort()
#     # order = reorder
#     complete_simi_graph = complete_simi_graph[order, :]
#     complete_simi_graph = complete_simi_graph[:, order]
#
#     plt.matshow(complete_simi_graph, cmap=plt.cm.Blues)
#     plt.colorbar()
#     plt.show()

def community_test():
    # better with karate_graph() as defined in networkx example.
    # erdos renyi don't have true community structure
    G = nx.erdos_renyi_graph(30, 0.05)

    # first compute the best partition
    partition = community.best_partition(G)

    # drawing
    size = float(len(set(partition.values())))
    pos = nx.spring_layout(G)
    count = 0.
    for com in set(partition.values()):
        count = count + 1.
        list_nodes = [nodes for nodes in partition.keys()
                      if partition[nodes] == com]
        nx.draw_networkx_nodes(G, pos, list_nodes, node_size=20,
                               node_color=str(count / size))

    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()

def trans_matrix_2_tuple(M):
    nx, ny = M.shape
    t = {}
    for x in range(nx):
        for y in range(ny):
            if x < y:
                _x = x
                _y = y
            else:
                _x = y
                _y = x
            v = np.sqrt(M[x,y])
            if (_x,_y) not in t:
                t[(_x,_y)] = v
            else:
                t[(_x,_y)] = t[(_x,_y)] + v
    print(t)
    res = []
    for _t in t:
        res.append((_t[0], _t[1], {"weight":t[_t]}))
    print(res)
    return res

def partition():
    t = trans_matrix_2_tuple(np.array(M2))
    G = nx.Graph()
    G.add_edges_from(t)

    print(G.edges(data=True))
    part = community.best_partition(G)
    print(part)
    # plt.subplot(111)
    # nx.draw(G,with_labels=True)
    # plt.show()

def optimal_leaf_ordering(M2):
    M = np.array(M2)
    if(M.sum() < 1e-3):
        return np.array(range(M.shape[0])).tolist()
    M = (M + M.transpose()) / 2
    Z = linkage(M,"average", optimal_ordering=True)
    Order = []
    for i in range(M.shape[0]):
        Order.append([i])
    for cell in Z:
        L1, L2, _, _ = cell
        Order.append(Order[int(L1)]+Order[int(L2)])
    return Order[-1]


def segmentation(M2):
    M = np.array(M2)
    if(M.sum() < 1e-3):
        return [0]
    SM = np.zeros(M.shape)
    # transformed to a symmetrical matrix
    for i in range(0,M.shape[0]):
        for j in range(i, M.shape[1]):
            SM[i,j] = (M[i,j] + M[j,i]) / 2
            SM[j,i] = SM[i, j]
    for i in range(M.shape[0]):
        SM[i,i] = M[i,i]
    def cost(a,b):
        a = a - 1
        b = b - 1
        if a == (b):
            return 0
        sum = 0.0
        count = 0.0
        for i in range(a,b+1):
            for j in range(i, b+1):
                sum = sum + SM[i,j]
                count = count + 1
        return sum / count
    R = np.zeros((SM.shape[0]+1, SM.shape[0]+1))
    Bx = np.zeros((SM.shape[0]+1, SM.shape[0]+1)).astype(int)
    By = np.zeros((SM.shape[0]+1, SM.shape[0]+1)).astype(int)
    for i in range(1, SM.shape[0]+1):
        R[i,i] = 0
        Bx[i,i], By[i,i] = [i-1, i-1]
        R[i,1] = cost(1,i)
        Bx[i,1], By[i,1] = [1,-1]
    for j in range(2, SM.shape[0]+1):
        for i in range(j+1, SM.shape[0]+1):
            max_v = -1
            coor = []
            for k in range(j-1, i):
                if( R[k,j-1] + cost(k+1,i) > max_v):
                    max_v = R[k,j-1] + cost(k+1,i)
                    coor = [k,j-1]
            R[i,j] = max_v
            Bx[i,j], By[i,j] = coor
    k = R[SM.shape[0],:].argmax()
    seg_point = []
    x,y = [Bx[SM.shape[0],k], By[SM.shape[0],k]]
    print(R)
    while y != -1:
        seg_point.append(x)
        x, y = [Bx[x,y], By[x,y]]
    seg_point.append(0)
    seg_point.reverse()
    print(seg_point)
    return seg_point

if __name__ == "__main__":
    # segmentation(config.dog_dataset_name)
    # community_test()
    # partition()
    optimal_leaf_ordering(M2)
    segmentation(M2)