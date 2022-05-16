import os
import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt
from scripts.configs import config
from scripts.backend_model import get_tsne_from_similarity_matrix
from scripts.crowd_data import CrowdData
from scripts.backend import decom_similarity_matrix

def get_name_2_id_map():
    filename = os.path.join(config.row_data_root, config.bird_dataset_name,
                            "feature/image_map.txt")
    lines = open(filename,"r").read().split("\n")[:-1]
    name_2_id_map = {}
    for s in lines:
        id, image_name = s.split(" : ")
        name_2_id_map[image_name] = int(id)
    return name_2_id_map

def get_order():
    name_2_id_map = get_name_2_id_map()
    filename = os.path.join(config.row_data_root, config.bird_dataset_name,
                            "feature/all_image_order.txt")
    lines = open(filename,"r").read().split("\n")[:-1]
    id_list = []
    for s in lines:
        id_list.append(name_2_id_map[s])
    id_list = np.array(id_list)
    order = id_list.argsort()
    return order

def similarity_according_feature(features):
    norm = ((features ** 2).sum(axis=1)) ** 0.5
    norm = np.dot(norm.reshape(-1, 1), norm.reshape(1, -1))
    print("norm shape: %s" % (str(norm.shape)))
    simi = np.dot(features, features.transpose()) / norm
    simi = simi + (1 - simi.max())
    return simi

def tsne_plot(dataname):
    # simi = extract_similarity_via_npy(dataname)
    simi = mixed_similarity()
    tsne = get_tsne_from_similarity_matrix(simi)
    # tsne = np.random.rand(2000,2)
    y = []
    for i in range(2000):
        y.append( i // 200 )
    color_map = plt.get_cmap("tab10")(y)
    print(color_map)
    plt.scatter(tsne[:,0], tsne[:,1],s=3, c=color_map)
    plt.show()

def mixed_similarity():
    total_feature = np.load(os.path.join(config.row_data_root, config.bird_dataset_name,
                                         "feature/GMP_feature_2000_87.npy"))
    total_feature_2 = np.load(os.path.join(config.row_data_root, config.bird_dataset_name,
                                         "feature/GMP_feature_2000_10_87.npy"))
    swan_feature = np.load(os.path.join(config.row_data_root, config.bird_dataset_name,
                                         "feature/feature_swan_256.npy"))
    sandpiper_feature = np.load(os.path.join(config.row_data_root, config.bird_dataset_name,
                                         "feature/feature_sandpiper_256.npy"))
    sparrow_feature = np.load(os.path.join(config.row_data_root, config.bird_dataset_name,
                                         "feature/feature_sparrow_256.npy"))

    # concatenate version
    # total_feature_norm = ((total_feature ** 2).sum(axis=1)) ** 0.5
    # total_feature_norm = total_feature_norm.reshape(-1,1).repeat(axis=1, repeats=total_feature.shape[1])
    # total_feature = total_feature / total_feature_norm
    #
    # total_feature2_norm = ((total_feature_2 ** 2).sum(axis=1)) ** 0.5
    # total_feature2_norm = total_feature2_norm.reshape(-1, 1).repeat(axis=1, repeats=total_feature_2.shape[1])
    # total_feature_2 = total_feature_2 / total_feature2_norm
    # split_param = 0.1
    # feature = np.concatenate(( ((1-split_param)**0.5) * total_feature, ((split_param)**0.5) * total_feature_2), axis=1)
    # simi = similarity_according_feature(feature)
    # return simi

    total_simi = similarity_according_feature(total_feature)
    total_simi_2 = similarity_according_feature(total_feature_2)

    # total_simi = temp()
    swan_simi = similarity_according_feature(swan_feature)
    sandpiper_simi = similarity_according_feature(sandpiper_feature)
    sparrow_simi = similarity_according_feature(sparrow_feature)

    rho = 0.1

    # total_simi[:400,:400] = (1 - rho) * total_simi[:400,:400] +  rho * sparrow_simi
    # total_simi[800:1200,800:1200] = rho * sparrow_simi
    # total_simi[1200:1600,1200:1600] = rho * sandpiper_simi
    # total_simi[1600:,1600:] = rho * swan_simi

    split_param = 0.1
    total_simi = (1-split_param) * total_simi + split_param * total_simi_2
    return total_simi
    # plt.matshow(total_simi, cmap=plt.cm.Blues)
    # plt.colorbar()
    # plt.show()
    # exit()

    # order = get_order()
    # total_simi = total_simi[order,:]
    # total_simi = total_simi[:,order]

    instance_num = total_simi.shape[0]
    com_simi = []
    for i in range(instance_num):
        for j in range(i + 1, instance_num):
            com_simi.append(total_simi[i, j])
    print("theoretical length: %s and its actual lengh of com_simi %s"
          % ((instance_num) * (instance_num - 1) / 2, len(com_simi)))
    com_simi = np.array(com_simi).tolist()
    return com_simi


def extract_similarity_via_npy(dataname):
    filename = os.path.join(config.row_data_root, dataname,
                            "feature/feature_2000.npy")
    features = np.load(filename)
    # feature2 = os.path.join(config.row_data_root, dataname,
    #                         "feature/feature_sparrow_256.npy")
    # feature2 = np.load(feature2)
    # print(features.shape, feature2.shape)
    # features = np.concatenate((features, feature2), axis=1)
    print(features.shape)
    simi = similarity_according_feature(features)
    print("similarity matrix shape:%s" % (str(simi.shape)))
    return simi

def temp():
    crowd_data = CrowdData(config.bird_dataset_name)
    simi_matrix = extract_similarity(config.bird_dataset_name)
    simi_matrix = decom_similarity_matrix(simi_matrix, 2000)
    true_labels = crowd_data.get_attr(config.true_labels_name)
    true_labels = np.array(true_labels).reshape(-1)

    order = true_labels.argsort()
    simi_matrix = simi_matrix[order, :]
    simi_matrix = simi_matrix[:, order]
    return simi_matrix

def extract_similarity(dataname):
    if(dataname == config.bird_dataset_name):
        return mixed_similarity()
    filename = os.path.join(config.row_data_root, dataname,
                            "feature/feature.txt")
    file = open(filename)
    print("extract similarity matrix of %s dataset"%(dataname))
    s = file.read().strip("\n").split(" ")[:-1]
    features = np.array([float(i) for i in s]).reshape(-1,4096)
    print("feature shape: %s"%(str(features.shape)))
    norm = ( (features**2).sum(axis=1) ) ** 0.5
    norm = np.dot(norm.reshape(-1,1), norm.reshape(1,-1) )
    print("norm shape: %s"%(str(norm.shape)))
    simi = np.dot(features, features.transpose()) / norm
    simi = simi + ( 1 - simi.max() )
    print("similarity matrix shape:%s"%(str(simi.shape)))

    # just save independent part of similarity matrix because of its symmetry
    instance_num = simi.shape[0]
    com_simi = []
    for i in range(instance_num):
        for j in range(i+1,instance_num):
            com_simi.append(simi[i,j])
    print("theoretical length: %s and its actual lengh of com_simi %s"
          %((instance_num)*(instance_num-1)/2, len(com_simi)))

    # print(max(com_simi), min(com_simi))

    return com_simi

if __name__ == "__main__":
    # extract_similarity_via_npy(config.bird_dataset_name)
    tsne_plot(config.bird_dataset_name)
    # mixed_similarity()