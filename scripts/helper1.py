from stat import S_ISREG, ST_CTIME, ST_MODE
import numpy as np
import scipy.io as sio
import os
from os import makedirs, listdir
from os.path import exists, join
import json
from PIL import Image
from sklearn import manifold
from sklearn.neighbors import NearestNeighbors, BallTree
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from scripts.configs import config
from scripts.tsne import guide_tsne

def uncertainty_aggregation(model, data, solution):
    return ( np.array(model) + np.array(data) + np.array(solution) ) / 3.0

def spammer_score_aggregation(reliability, spammer_score):
    return ( np.array(spammer_score) - np.array(reliability) + 1.0) / 2.0

def cos_simi(X):
    X_len = ( (X**2).sum(axis=1) )**0.5
    norm = np.dot(X_len.reshape(-1,1), X_len.reshape(1,-1))
    simi = np.dot(X, X.transpose())
    simi = simi / norm
    return simi

def str_2_matrix(lines):
    M = []
    for i in range(len(lines)):
        m = map(eval, lines[i].split(' '))
        for j in m:
            M.append(j)
    return M

def get_knn( n_neighbors, simi):
    nn = np.ones((simi.shape[0], n_neighbors)) * -1

    for i in range(simi.shape[0]):
        simi[i,i] = 1
        nn[i,:] = simi[i,:].argsort()[:n_neighbors]
    return nn

# TODO :  this is a fake function
def list_normalizing(data):
    # data = np.array(data)
    # data = data/data.max()
    return data


def load_static_info(filename, true_labels_filename, feature_filename):
    '''
    Load a static.info file defined by Liu Jiang.
    :param filename:
    :return: a dict
    '''
    d = {}
    fi = open(filename, 'r')
    line = fi.read().split('\n')
    d[config.instance_num_name] = int(line[0])
    print("InstanceTotalNum: %s" % (d[config.instance_num_name]))
    d["WorkerTotalNum"] = int(line[1])
    print("WorkerTotalNum: %s" % (d["WorkerTotalNum"]))
    d["LabelTotalNum"] = int(line[2])
    print("LabelTotalNum: %s" % (d["LabelTotalNum"]))
    d["ModelTotalNum"] = int(line[3])
    print("ModelTotalNum: %s" % (d["ModelTotalNum"]))
    d["LabelNames"] = line[4].split('\t')
    print("LabelNames: %s" % (d["LabelNames"]))
    simi_graph = []
    complete_simi_graph = np.ones((d[config.instance_num_name], d[config.instance_num_name]))
    for i in range(5, 5 + d[config.instance_num_name] - 1):
        s = i - 5
        m = map(eval, line[i].split(' '))
        for t, j in enumerate(m):
            simi_graph.append(j)
            complete_simi_graph[s, s + t + 1] = j
            complete_simi_graph[s + t + 1, s] = j
    complete_dist_graph = 1 - complete_simi_graph
    d["SimiGraph"] = simi_graph
    print("SimiGraph length:%s" % (len(simi_graph)))
    d["complete_simi_graph"] = complete_simi_graph
    knn = get_knn(20, complete_simi_graph).astype(int)
    d["KNN"] = knn.tolist()

    #t-SNE result
    tsne = manifold.TSNE(n_components=2, metric="precomputed")
    plat_coordinate = tsne.fit_transform(complete_dist_graph)
    d["TSNECoordinate"] = plat_coordinate.tolist()
    print("TSNECoordinate shape:%s" % (str(plat_coordinate.shape)))

    print("SimiGraph length:%s" % (len(simi_graph)))
    d["WorkerLabels"] = str_2_matrix(line[5 + d[config.instance_num_name] - 1: 5 + d[config.instance_num_name] * 2 - 1])
    print("WorkerLabels shape:%s" % (len(d["WorkerLabels"])))
    true_labels = sio.loadmat(true_labels_filename)
    true_labels = true_labels["true_labels"].reshape(-1)
    d["true_labels"] = true_labels.tolist()
    print("true_labels len:%s" % (len(true_labels)))
    #load feature
    mat = sio.loadmat(feature_filename)
    features = mat["sub_features"]
    simi = mat["simi"]
    simi_test = cos_simi(features)
    error = ( ( simi - simi_test )**2 ).sum()
    print("error: %s" %(error) )
    d["features"] = features.tolist()
    print("features shape:%s" %( str(features.shape) ))




    # plot test
    # fig = plt.figure(figsize=(15, 8))
    # ax = plt.subplot(111)
    # plt.scatter(plat_coordinate[:, 0], plat_coordinate[:, 1], c='r', edgecolors='face')
    # ax.xaxis.set_major_formatter(NullFormatter())
    # ax.yaxis.set_major_formatter(NullFormatter())
    # plt.axis('tight')
    # plt.show()

    return d


def load_dynamic_info(ItemTotalNum, filename):
    '''
    Load a dynamic.info file defined by Liu Jiang.
    :param filename:
    :return: a dict
    '''
    d = {}
    fi = open(filename, 'r')
    line = fi.read().split('\n')
    d["DisagreementDistribution"] = str_2_matrix(line[:ItemTotalNum])
    print("DisagreementDistribution shape: %s" % (len(d["DisagreementDistribution"])))
    d["PosteriorDistribution"] = str_2_matrix(line[ItemTotalNum: 2 * ItemTotalNum])
    print("PosteriorDistribution shape: %s" % (len(d["PosteriorDistribution"])))
    d["InferredLabel"] = str_2_matrix(line[2 * ItemTotalNum:2 * ItemTotalNum + 1])
    print("InferredLabel shape: %s" % (len(d["InferredLabel"])))
    d["ModelUncertainty"] = str_2_matrix(line[2 * ItemTotalNum + 1: 2 * ItemTotalNum + 2])
    d["ModelUncertainty"] = list_normalizing(d["ModelUncertainty"])
    print("ModelUncertainty shape: %s" % (len(d["ModelUncertainty"])))
    d["DataUncertainty"] = str_2_matrix(line[2 * ItemTotalNum + 2: 2 * ItemTotalNum + 3])
    d["DataUncertainty"] = list_normalizing(d["DataUncertainty"])
    print("DataUncertainty shape: %s" % (len(d["DataUncertainty"])))
    d["SolutionUncertainty"] = str_2_matrix(line[2 * ItemTotalNum + 3: 2 * ItemTotalNum + 4])
    d["SolutionUncertainty"] = list_normalizing(d["SolutionUncertainty"])
    print("SolutionUncertainty shape: %s" % (len(d["SolutionUncertainty"])))
    d["WorkerReliability"] = str_2_matrix(line[2 * ItemTotalNum + 4: 2 * ItemTotalNum + 5])
    d["WorkerReliability"] = list_normalizing(d["WorkerReliability"])
    print("WorkerReliability shape: %s" % (len(d["WorkerReliability"])))
    d["WorkerAccuracy"] = str_2_matrix(line[2 * ItemTotalNum + 5: 2 * ItemTotalNum + 6])
    d["WorkerAccuracy"] = list_normalizing(d["WorkerAccuracy"])
    print("WorkerAccuracy shape: %s" % (len(d["WorkerAccuracy"])))
    d["WorkerLabellingNum"] = str_2_matrix(line[2 * ItemTotalNum + 6: 2 * ItemTotalNum + 7])
    print("WorkerLabellingNum shape: %s" % (len(d["WorkerLabellingNum"])))
    d["SloppyScore"] = str_2_matrix(line[2 * ItemTotalNum + 7: 2 * ItemTotalNum + 8])
    d["SloppyScore"] = list_normalizing(d["SloppyScore"])
    print("SloppyScore shape: %s" % (len(d["SloppyScore"])))
    d["RandomScore"] = str_2_matrix(line[2 * ItemTotalNum + 8: 2 * ItemTotalNum + 9])
    d["RandomScore"] = list_normalizing(d["RandomScore"])
    print("RandomScore shape: %s" % (len(d["RandomScore"])))
    d["UniformScore"] = str_2_matrix(line[2 * ItemTotalNum + 9: 2 * ItemTotalNum + 10])
    d["UniformScore"] = list_normalizing(d["UniformScore"])
    print("UniformScore shape: %s" % (len(d["UniformScore"])))

    return d


def save_manifest_jsonfile(data, outfile):
    '''
    save manifest info to a jsonfile.
    :param data:
    :param outfile:
    :return: None
    '''
    d = {}
    d[config.instance_num_name] = data[config.instance_num_name]
    d["WorkerTotalNum"] = data["WorkerTotalNum"]
    d["LabelTotalNum"] = data["LabelTotalNum"]
    d["ModelTotalNum"] = data["ModelTotalNum"]
    d["LabelNames"] = data["LabelNames"]
    fo = open(outfile, 'w')
    fo.write(json.dumps(d))
    return None

def get_tsne_jsonfile(dynamic_data, static_data):
    plat_coordinate = np.array(static_data["TSNECoordinate"])
    print("plat_coordinate: " + str(plat_coordinate.shape))
    instance_uncertainty = dynamic_data["DataUncertainty"]
    instance_num = len(instance_uncertainty)
    posterior_distribution = dynamic_data["PosteriorDistribution"]
    complete_simi_graph = static_data["complete_simi_graph"]
    posterior_labels = np.array(posterior_distribution).reshape(instance_num,-1).argmax(axis=1)

    plat_max = plat_coordinate.max(axis=0).reshape(1, -1).repeat(repeats=plat_coordinate.shape[0], axis=0)
    plat_min = plat_coordinate.min(axis=0).reshape(1, -1).repeat(repeats=plat_coordinate.shape[0], axis=0)
    plat_coordinate = (plat_coordinate - plat_min) / (plat_max - plat_min)
    constraint = [[0.5, 0], [1, 0.5]]
    constraint_num = len(constraint)
    constraint_metric = np.zeros((constraint_num, instance_num))
    for i in range(instance_num):
        index = int(posterior_labels[i]/2)
        constraint_metric[index,i] = 1
    constraint_metric = constraint_metric.tolist()
    print(constraint_metric)
    guide_tsne(plat_coordinate, complete_simi_graph, constraint, constraint_metric)
    exit()


def save_static_jsonfile(data, outfile):
    '''
    save static info to a jsonfile
    :param data:
    :param outfile:
    :return: None
    '''
    d = {}
    d["SimiGraph"] = data["SimiGraph"]
    d["WorkerLabels"] = data["WorkerLabels"]
    d["TSNECoordinate"] = data["TSNECoordinate"]
    d["KNN"] = data["KNN"]
    d["true_labels"] = data["true_labels"]
    d["features"] = data["features"]
    fo = open(outfile, 'w')
    fo.write(json.dumps(d))
    return None


def save_dynamic_jsonfile(data, outfile):
    '''
    save dynamic info to a jsonfile. But It may not be used in future plan.
    :param data:
    :param outfile:
    :return:
    '''
    d = {}
    d["PosteriorDistribution"] = data["PosteriorDistribution"]
    d["Uncertainty"] = uncertainty_aggregation(data["ModelUncertainty"],
                                                data["DataUncertainty"],
                                                data["SolutionUncertainty"] ).tolist()
    d[config.worker_accuracy_name] = data["WorkerReliability"]
    # d["SpammerScore"] = spammer_score_aggregation( data["WorkerReliability"],
    #                                                data["UniformScore"]).tolist()
    d["SpammerScore"] = data["UniformScore"]
    fo = open(outfile, 'w')
    fo.write(json.dumps(d))
    return None


def square_image_converter(image_input_url, image_output_url):
    im = Image.open(image_input_url)
    width, height = im.size
    size_length = max(width, height)
    im = im.resize((size_length, size_length))
    im.save(image_output_url)
    return


def test_random_redundancy_score():
    score = 0
    for i in range(1000):
        idx1 = np.array(range(200))
        idx2 = np.array(range(200))
        np.random.shuffle(idx1)
        np.random.shuffle(idx2)
        score = score + common_count(idx1[:40], idx2[:40]) / float(40)
    print(score / 1000)


def common_count(list1, list2):
    count = 0
    for i in list1:
        for j in list2:
            if i == j:
                count = count + 1
                break
    return count


def redundancy_score(list1, list2, percent):
    idx1 = np.array(list1).argsort()[::-1]
    idx2 = np.array(list2).argsort()[::-1]
    percent_len = int(len(list1) * percent)
    return common_count(idx1[:percent_len], idx2[:percent_len]) / percent_len


def attribute_redundancy(filename):
    json_file = open(filename, 'r')
    data = json.load(json_file)
    score_list = [0, 0, 0, 0, 0]
    score_list[0] = data["WorkerRealiability"]
    score_list[0] = - np.array(score_list[0])
    score_list[1] = - np.array(data["WorkerAccuracy"])
    score_list[2] = data["SloppyScore"]
    score_list[3] = data["RandomScore"]
    # score_list[4] = data["UniformScore"]
    for i in range(4):
        for j in range(4):
            print(redundancy_score(score_list[i], score_list[j], 0.4), end=" ")
        print(".")


def listdir_sorted_by_date(dirpath):
    entries = [(join(dirpath, fn), fn) for fn in listdir(dirpath)]
    entries = ((os.stat(path), path, fn) for path, fn in entries)

    # leave only regular files, insert creation date
    entries = ((stat[ST_CTIME], path, fn)
               for stat, path, fn in entries if not S_ISREG(stat[ST_MODE]))
    # NOTE: on Windows `ST_CTIME` is a creation date
    #  but on Unix it could be something else
    # NOTE: use `ST_MTIME` to sort by a modification date

    return [fn for cdate, path, fn in sorted(entries, reverse=True)]


if __name__ == "__main__":
    dataname = "dog"
    s = load_static_info(os.path.join("../../RawData", dataname, "info/static.info"),
                         os.path.join("../../RawData", dataname, "crowdModel/true_labels.mat"),
                         os.path.join("../../RawData", dataname, "info/dog_MutInf_64.mat"))
    d = load_dynamic_info(s[config.instance_num_name], os.path.join( "../../RawData",
                                                           dataname,"info/0.dynamic.info"))
    get_tsne_jsonfile(d, s)
    # save_manifest_jsonfile(s, os.path.join("../data",dataname,"info/manifest.json"))
    # save_static_jsonfile(s, os.path.join("../data", dataname, "info/static_info.json"))
    # save_dynamic_jsonfile(d, os.path.join("../data",dataname, "info/dynamic_info.json"))
    # print(d)
    # for i in range(1,1003):
    #     square_image_converter( "D:/CrowdSourcing/RawData/age/image source/" + str(i) + ".jpg",
    #                             "../data/age/image/" + str(i) + ".jpg")
    # attribute_redundancy( "../data/news/info/dynamic_info.json")
    # test_random_redundancy_score()
