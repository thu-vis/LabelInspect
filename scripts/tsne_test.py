from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
from matplotlib import offsetbox
from sklearn import manifold, datasets, decomposition
from scripts import load_static_data


def mnist_save(filename):
    fi = open(filename, 'w')
    digits = datasets.load_digits(n_class=6)
    X = np.array(digits.data)
    y = digits.target
    for i in range(X.shape[0]):
        s = str(y[i]) + " "
        for j in X[i, :]:
            s = s + str(j) + " "
        fi.writelines(s + '\n')

def mnist():
    digits = datasets.load_digits(n_class=6)
    X = np.array(digits.data)
    y = digits.target
    n_samples, n_features = X.shape
    tsne = manifold.TSNE(n_components=2, random_state=0)
    simi_graph = np.dot(X, X.transpose())
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            simi_graph[i, j] = (((X[i, :] - X[j, :]) ** 2).sum()) ** 0.5
    X_tsne = tsne.fit_transform(X)
    fig = plt.figure(figsize=(15, 8))
    ax = plt.subplot(111)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=(y.astype(float) + 1), edgecolors='face')
    plt.axis('tight')
    plt.show()

def plot_traversal_params(dataname):
    filename = os.path.join("D:\CrowdSourcing2018\RawData", dataname, "info\static.info")
    true_labels_filename = os.path.join("D:\CrowdSourcing2018\RawData", dataname, "crowdModel\\true_labels.mat")
    dynamic_filename = os.path.join("D:\CrowdSourcing2018\Project\data", dataname, "info\\dynamic_info.json")
    dynamic_info = load_static_data(dynamic_filename)
    posterior = dynamic_info["Posterior"]
    d = {}
    fi = open(filename, 'r')
    line = fi.read().split('\n')
    d["ItemTotalNum"] = int(line[0])
    posterior = np.array(posterior).reshape(d["ItemTotalNum"], -1)
    posterior = posterior.argmax(axis=1)
    complete_simi_graph = np.ones((d["ItemTotalNum"], d["ItemTotalNum"]))
    for i in range(5, 5 + d["ItemTotalNum"] - 1):
        s = i - 5
        m = map(eval, line[i].split(' '))
        for t, j in enumerate(m):
            complete_simi_graph[s, s + t + 1] = j
            complete_simi_graph[s + t + 1, s] = j
    complete_simi_graph = 1 - complete_simi_graph
    true_labels = sio.loadmat(true_labels_filename)
    true_labels = true_labels["true_labels"].reshape(-1)

    # manually adjust similarity graph according to ground truth
    # for i in range(complete_simi_graph.shape[0]):
    #     for j in range(complete_simi_graph.shape[1]):
    #         if( posterior[i] == posterior[j]):
    #             complete_simi_graph[i][j] = complete_simi_graph[i][j] / 2

    for perplexity in [30]:
        for early_exaggeration in [12.0]:
            for learning_rate in [200.0]:
                if 1:
                    tsne = manifold.TSNE(n_components=2, perplexity=perplexity, early_exaggeration=early_exaggeration,
                                         learning_rate=learning_rate, metric="precomputed")
                    X = tsne.fit_transform(complete_simi_graph)
                else:
                    X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(complete_simi_graph)

                # plot test
                fig = plt.figure(figsize=(15, 8))
                ax = plt.subplot(111)
                plt.scatter(X[:, 0], X[:, 1], c=(true_labels.astype(float) + 1), edgecolors='face')
                plt.axis('tight')
                plt_name = "perplexity_" + str(perplexity) + "_early_exaggeration_" + str(early_exaggeration) \
                           + "_learning_rate_" + str(learning_rate) + ".png"
                # plt.savefig(os.path.join("D:\CrowdSourcing2018\RawData", dataname , "tsne", plt_name))
                # plt.close()
                plt.show()

if __name__ == "__main__":
    # plot_traversal_params("monkey")
    # mnist()
    mnist_save("../../minst_data.txt")
