import numpy as np
from matplotlib import pyplot as plt

from sklearn.cluster import KMeans

from sklearn.cluster import SpectralBiclustering

from sklearn.datasets import make_checkerboard
from sklearn.datasets import samples_generator as sg
from sklearn.metrics import consensus_score

def instance_kmeans(X, k):
    model = KMeans(n_clusters=k)
    s = model.fit_predict(X)
    return s

def instance_worker_bicluster(data, nrow, ncol):
    # plt.matshow(data, cmap=plt.cm.Blues)
    # plt.title("Original dataset")

    model = SpectralBiclustering(n_clusters=(nrow, ncol), method='log',
                                 random_state=0)
    model.fit(data)

    # fit_data = data[np.argsort(model.row_labels_)]
    # fit_data = fit_data[:, np.argsort(model.column_labels_)]
    #
    # plt.matshow(fit_data, cmap=plt.cm.Blues)
    # plt.title("After biclustering; rearranged to show biclusters")
    #
    # plt.matshow(np.outer(np.sort(model.row_labels_) + 1,
    #                      np.sort(model.column_labels_) + 1),
    #             cmap=plt.cm.Blues)
    # plt.title("Checkerboard structure of rearranged data")
    #
    # plt.show()

    return model.row_labels_, model.column_labels_


def cluster_data_by_kemeans(spammer_score, worker_labels, worker_cluster_num, instance_cluster_num):
    worker_per_cluster_num = int(len(spammer_score) / worker_cluster_num)
    instance_label = instance_kmeans(worker_labels, instance_cluster_num)
    instance_cluster = []
    for i in range(0, instance_cluster_num):
        instance_cluster.append({
            "id": "parent" + str(i),
            "label": i,
            "level": 0,
            "children": []
        })
    for i in range(0, len(instance_label)):
        instance_cluster[instance_label[i]]['children'].append({
            "id": i,
            "label": i,
            "level": 1
        })
    worker_cluster = []

    spammer_score_with_id = []
    for i in range(0, len(spammer_score)):
        spammer_score_with_id.append({
            'id': i,
            'score': spammer_score[i]
        })
    spammer_score_with_id.sort(key=lambda x:-x['score'])

    for i in range(0, worker_cluster_num):
        worker_cluster.append({
            "id": "parent" + str(i),
            "label": i,
            "level": 0,
            "children": []
        })
        for j in range(i * worker_per_cluster_num, (i + 1) * worker_per_cluster_num):
            worker_cluster[i]['children'].append({
                "id": spammer_score_with_id[j]['id'],
                "label": spammer_score_with_id[j]['id'],
                "level": 1
            })
    for j in range(worker_per_cluster_num * worker_cluster_num, len(spammer_score)):
        worker_cluster[worker_cluster_num - 1]['children'].append({
            "id": spammer_score_with_id[j]['id'],
            "label": spammer_score_with_id[j]['id'],
            "level": 1
        })

    return worker_cluster, instance_cluster

def cluster_data_by_bicluster(worker_labels, worker_cluster_num, instance_cluster_num):
    instance_label, worker_label = instance_worker_bicluster(worker_labels, instance_cluster_num, worker_cluster_num)
    instance_cluster = []
    for i in range(0, instance_cluster_num):
        instance_cluster.append({
            "id": "parent" + str(i),
            "label": i,
            "level": 0,
            "children": []
        })
    for i in range(0, len(instance_label)):
        instance_cluster[instance_label[i]]['children'].append({
            "id": i,
            "label": i,
            "level": 1
        })
    worker_cluster = []
    for i in range(0, worker_cluster_num):
        worker_cluster.append({
            "id": "parent" + str(i),
            "label": i,
            "level": 0,
            "children": []
        })
    for i in range(0, len(worker_label)):
        worker_cluster[worker_label[i]]['children'].append({
            "id": i,
            "label": i,
            "level": 1
        })

    return worker_cluster, instance_cluster