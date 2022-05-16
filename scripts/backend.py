import numpy as np
import os
import json
import time
import scipy.io as sio

from sklearn.metrics.pairwise import euclidean_distances
from scripts.configs import config
from scripts.tsne import guide_tsne, incremental_tsne, guide_tsne_old


def get_time_string():
    return time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))


def decom_similarity_matrix(compressed_similarity_matrix, dim):
    uncom_similarity_matrix = np.ones((dim, dim))
    count = 0
    for i in range(dim):
        for j in range(i + 1, dim):
            uncom_similarity_matrix[i, j] = compressed_similarity_matrix[count]
            uncom_similarity_matrix[j, i] = compressed_similarity_matrix[count]
            count = count + 1
    return uncom_similarity_matrix


def inverse_coordinate_transformation(coordinate, constraints):
    constraints_point_num = len(constraints)
    constraints = np.array(constraints)
    max_coordinate = coordinate.max(axis=0).reshape(1, -1).repeat(repeats=constraints_point_num, axis=0)
    min_coordinate = coordinate.min(axis=0).reshape(1, -1).repeat(repeats=constraints_point_num, axis=0)
    constraints = constraints * (max_coordinate - min_coordinate) + min_coordinate
    return constraints


def norm_coordinate(coordinate, center_point=None):
    max_coordinate = coordinate.max(axis=0).reshape(1, -1).repeat(repeats=coordinate.shape[0], axis=0)
    min_coordinate = coordinate.min(axis=0).reshape(1, -1).repeat(repeats=coordinate.shape[0], axis=0)
    result = (coordinate - min_coordinate) / (max_coordinate - min_coordinate)
    return result


def shrink_norm_coordinate(coordinate, center_point=None):
    max_coordinate = coordinate.max(axis=0).reshape(1, -1).repeat(repeats=coordinate.shape[0], axis=0)
    min_coordinate = coordinate.min(axis=0).reshape(1, -1).repeat(repeats=coordinate.shape[0], axis=0)
    result = (coordinate - min_coordinate) / (max_coordinate - min_coordinate)
    result = (result - np.array([[0.5, 0.5]])) * 0.9 + np.array([[0.5, 0.5]])
    return result


def unit_coordinate_center_norm(coordinate, center_point):
    max_coordinate = coordinate.max(axis=0).reshape(1, -1).repeat(repeats=coordinate.shape[0], axis=0)
    min_coordinate = coordinate.min(axis=0).reshape(1, -1).repeat(repeats=coordinate.shape[0], axis=0)
    center_point = (max_coordinate + min_coordinate) / 2.0
    # centralize all points on original point
    centered_coordinate = coordinate - center_point
    # get max radius (squared)
    radius_squared = (centered_coordinate ** 2).sum(axis=1)
    max_radius_squared = radius_squared.max()
    # get transforming scale of all points
    transform_scale = (0.25 / max_radius_squared) ** 0.5
    print("transform_scale: %s" % (transform_scale))
    # performing transformation
    transformed_centered_coordinate = centered_coordinate * transform_scale
    return transformed_centered_coordinate + np.array([0.5, 0.5])

    # max_coordinate = coordinate.max(axis=0).reshape(1, -1).repeat(repeats=coordinate.shape[0], axis=0)
    # min_coordinate = coordinate.min(axis=0).reshape(1, -1).repeat(repeats=coordinate.shape[0], axis=0)
    # return (coordinate - min_coordinate) / (max_coordinate - min_coordinate)


def load_manifest_data(filename):
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
        return data

def load_static_data(filename):
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
        return data

def get_dog_feature():
    mat = sio.loadmat(os.path.join(config.data_root, config.dog_dataset_name,
                                   "dog_MutInf_64.mat"))
    features = mat["sub_features"]
    norm_feature = (features**2).sum(axis=1)** 0.5
    norm_feature = norm_feature.reshape(-1,1).repeat(axis=1, repeats=features.shape[1])
    features = features / norm_feature
    return features

def get_bird_feature():
    total_feature = np.load(os.path.join(config.data_root, config.bird_dataset_name,
                                         "feature/GMP_feature_2000_87.npy"))
    total_feature_2 = np.load(os.path.join(config.data_root, config.bird_dataset_name,
                                         "feature/GMP_feature_2000_10_87.npy"))
    total_feature_norm = ((total_feature ** 2).sum(axis=1)) ** 0.5
    total_feature_norm = total_feature_norm.reshape(-1, 1).repeat(axis=1, repeats=total_feature.shape[1])
    total_feature = total_feature / total_feature_norm
    total_feature2_norm = ((total_feature_2 ** 2).sum(axis=1)) ** 0.5
    total_feature2_norm = total_feature2_norm.reshape(-1, 1).repeat(axis=1, repeats=total_feature_2.shape[1])
    total_feature_2 = total_feature_2 / total_feature2_norm
    split_param = 0.1
    feature = np.concatenate((((1 - split_param) ** 0.5) * total_feature, ((split_param) ** 0.5) * total_feature_2),
                             axis=1)
    return feature

def get_feature(dataname):
    if dataname == config.dog_dataset_name:
        return get_dog_feature()
    elif dataname == config.bird_dataset_name:
        return get_bird_feature()
    else:
        raise ValueError("wrong dataname when get features")

def similarity_according_feature(features):
    norm = ((features ** 2).sum(axis=1)) ** 0.5
    norm = np.dot(norm.reshape(-1, 1), norm.reshape(1, -1))
    print("norm shape: %s" % (str(norm.shape)))
    simi = np.dot(features, features.transpose()) / norm
    simi = simi + (1 - simi.max())
    return simi

def get_high_dimension_distance_to_arc(dataname, selected_list, posterior_labels):
    total_feature = get_feature(dataname)
    selected_instance_indicator = [True if c in selected_list else False for c in posterior_labels]
    selected_instance_posterior_labels = posterior_labels[selected_instance_indicator]
    total_feature = total_feature[selected_instance_indicator,:]
    center_of_mass = total_feature.mean(axis=0)
    distance_to_center = (total_feature - center_of_mass.reshape(1,-1).
                            repeat(repeats=total_feature.shape[0], axis=0))**2
    distance_to_center = distance_to_center.sum(axis=1) ** 0.5
    distance_to_arc = np.zeros(len(selected_instance_posterior_labels))
    for i in selected_list:
        class_instance_indicator = [True if c in [i] else False for c in selected_instance_posterior_labels]
        class_feature = total_feature[class_instance_indicator]
        class_distance_to_center = distance_to_center[class_instance_indicator]
        class_max_distance = class_distance_to_center.max()
        class_distance_to_arc = class_max_distance - class_distance_to_center
        distance_to_arc[class_instance_indicator] = class_distance_to_arc

    # distance_to_arc = np.random.rand(len(distance_to_arc))

    distance_to_arc = distance_to_arc.reshape(1, -1).repeat(repeats=len(selected_list), axis=0)
    return distance_to_arc

def get_high_dimension_distance_to_center(dataname, selected_list, posterior_labels):
    total_feature = get_feature(dataname)
    selected_instance_indicator = [True if c in selected_list else False for c in posterior_labels]
    selected_instance_posterior_labels = posterior_labels[selected_instance_indicator]
    total_feature = total_feature[selected_instance_indicator, :]
    center_of_mass = total_feature.mean(axis=0)
    distance_to_center = (total_feature - center_of_mass.reshape(1, -1).
                          repeat(repeats=total_feature.shape[0], axis=0)) ** 2
    distance_to_center = distance_to_center.sum(axis=1) ** 0.5
    dis = euclidean_distances(total_feature, total_feature)
    similarity_to_arc = np.zeros((len(selected_instance_posterior_labels), len(selected_list)))
    for idx, i in enumerate(selected_list):
        class_instance_indicator = [True if c in [i] else False for c in selected_instance_posterior_labels]
        class_feature = total_feature[class_instance_indicator]
        class_center_of_mass = class_feature.mean(axis=0)
        dis[class_instance_indicator, class_instance_indicator] = 3
        class_dis = dis[class_instance_indicator,:]
        class_dis = class_dis[:, class_instance_indicator]
        class_distance_to_center = distance_to_center[class_instance_indicator]
        class_distance_to_class_center = (class_feature - center_of_mass.reshape(1, -1).
                          repeat(repeats=class_feature.shape[0], axis=0)) ** 2
        class_distance_to_class_center = class_distance_to_class_center.sum(axis=1) ** 0.5
        norm_center_of_mass = ((center_of_mass**2).sum()) ** 0.5
        norm_class_center_of_mass = (((class_center_of_mass**2)).sum()) ** 0.5
        center_distance = ( (class_center_of_mass - center_of_mass)**2 ).sum()
        center_distance = center_distance ** 0.5

    # similarity_to_arc = np.random.rand(len(selected_instance_posterior_labels), len(selected_list))
    return 1 - similarity_to_arc.transpose()

def get_new_tsne_result(dataname,
                        center_points,
                        current_tsne,
                        similarity_matrix,
                        worker_labels,
                        posterior_dist_labels,
                        uncertainty,
                        selected_list,
                        re_calculate_tsne):
    # get some constraints
    uncertainty = np.array(uncertainty).reshape(-1)
    instance_num = uncertainty.shape[0]
    worker_num = int(len(worker_labels) / instance_num)
    # get posterior labels from posterior label distributions
    posterior_labels = np.array(posterior_dist_labels).reshape(instance_num, -1).argmax(axis=1)
    # get selected instance indicator according to selected list
    selected_instance_indicator = [True if c in selected_list else False for c in posterior_labels]
    selected_posterior_labels = posterior_labels[selected_instance_indicator]
    selected_instance_index = np.array(range(instance_num))[selected_instance_indicator]

    #TODO: Get distance matrix
    feature = get_feature(dataname)
    distance_matrix = euclidean_distances(feature, feature) / 1.414
    distance_matrix = distance_matrix[selected_instance_indicator,:]
    distance_matrix = distance_matrix[:,selected_instance_indicator]

    # get index of instances with highest uncertainty
    unselected_instance_indicator = [not c for c in selected_instance_indicator]
    uncertainty[unselected_instance_indicator] = -1
    mixed_index = uncertainty.argsort()[::-1][:20]
    selected_ids = np.array(range(len(uncertainty)))[selected_instance_indicator]
    # get tsne result according to selected class
    selected_current_tsne = np.array(current_tsne)[selected_instance_indicator, :]

    selected_instance_num = sum(selected_instance_indicator)

    # constraint points
    constraints = []
    class_effect_of_constraints = []
    for pair in center_points:
        constraints.append([pair["x"], pair["y"]])
        class_effect_of_constraints.append(pair["class"])

    # constraints.append([0.5, 0.5])
    # class_effect_of_constraints.append(-1)

    # inverse coordinate transformation of constraint points
    selected_current_tsne = np.array(selected_current_tsne)
    constraints = inverse_coordinate_transformation(selected_current_tsne, constraints).tolist()
    center_point = constraints[-1]

    constraints_map = {}
    for i in selected_list:
        constraints_map[i] = []
    for i in range(len(constraints)):
        constraints_map[int(class_effect_of_constraints[i])].append(constraints[i])
    multi_constraints = []
    class_effect_of_constraints = []
    center_constraints = []
    for i in constraints_map:
        multi_constraints.append(constraints_map[i])
        class_effect_of_constraints.append(i)
        center_constraints.append(constraints_map[i][len(constraints_map[i]) // 2])


    # add constraint to each point
    constraint_num = len(multi_constraints)
    constraint_metric = np.zeros((constraint_num, selected_instance_num))

    distance_arc = get_high_dimension_distance_to_arc(dataname, selected_list, posterior_labels)#[selected_instance_indicator]
    get_high_dimension_distance_to_center(dataname, selected_list, posterior_labels)
    aug_distance_matrix = np.ones((constraint_num + selected_instance_num,
                                   constraint_num + selected_instance_num))

    for i in range(selected_instance_num):
        _class = int(selected_posterior_labels[i])
        if i in mixed_index:
            labels = list(set(worker_labels[i * worker_num:i * worker_num + worker_num]))
            for j in range(constraint_num):
                if class_effect_of_constraints[j] in labels:
                    constraint_metric[j, i] = 1
                    aug_distance_matrix[j,constraint_num + i] = distance_arc[j, i]
                    aug_distance_matrix[constraint_num + i,j] = distance_arc[j, i]
            # constraint_metric[-1, i] = 2
        else:
            # constraint_metric[index, i] = 1
            for j in range(constraint_num):
                if class_effect_of_constraints[j] == _class:
                    constraint_metric[j, i] = 1
                    aug_distance_matrix[j, constraint_num + i] = distance_arc[j, i]
                    aug_distance_matrix[constraint_num + i, j] = distance_arc[j, i]

    aug_distance_matrix[constraint_num:,constraint_num:] = distance_matrix
    distance_matrix = aug_distance_matrix
    print(distance_matrix.shape)

    return_constraint_metric = constraint_metric.copy()
    constraint_metric = constraint_metric.tolist()

    if selected_list == [4,5] and dataname == "bird":
        datatype = "bad_bird"
    else:
        datatype = dataname
    id = "".join([str(i) + "_" for i in selected_list])
    unnorm_guided_tsne_file_path = os.path.join(config.data_root,
                                         dataname,
                                         config.info_data,
                                         id + config.info_unnorm_guided_tsne_name)
    if re_calculate_tsne is True or (not os.path.exists(unnorm_guided_tsne_file_path)):
        # call guided tsne function
        print("re-calculate tsne result...")
        guide_tsne_result = guide_tsne(
            selected_current_tsne,
            distance_matrix,
            center_constraints,
            constraint_metric,
            datatype=datatype,
            cluster_num = len(selected_list),
            multi_constraint=multi_constraints)
        print("re-calculate tsne finished.")
        print(return_constraint_metric.tolist())
        result_dir = {
            "guide_tsne_result":guide_tsne_result.tolist(),
            "selected_indicator": selected_instance_indicator
        }
        open(unnorm_guided_tsne_file_path, "w")\
            .write(json.dumps(result_dir))
    else:
        print("loading pre-computed result...")
        result_dir = json.load(open(unnorm_guided_tsne_file_path,"r"))
        prev_guide_tsne_result = np.array(result_dir["guide_tsne_result"])
        prev_selected_instance_indicator = np.array(result_dir["selected_indicator"])
        print("begin incremental tsne")
        print("prev num:%s, present num:%s"%(sum(prev_selected_instance_indicator), sum(selected_instance_indicator)))
        fake_tsne_result = np.zeros((instance_num, 2))
        fake_list = np.zeros(instance_num).astype(bool)
        fake_tsne_result[selected_instance_indicator,:] = selected_current_tsne
        fake_list[selected_instance_indicator] = True
        fake_tsne_result[prev_selected_instance_indicator,:] = prev_guide_tsne_result
        fake_list[prev_selected_instance_indicator] = False
        selected_current_tsne = fake_tsne_result[selected_instance_indicator,:]
        incre_list = np.array(range(instance_num))[fake_list]
        if len(incre_list) < 1:
            guide_tsne_result = selected_current_tsne
        else:
            guide_tsne_result = incremental_tsne(
                selected_current_tsne.copy(),
                similarity_matrix,
                constraints,
                constraint_metric,
                incre_list)

        print("loading finished")

    # print("NOTICE: the result is not processed by tsne")
    # return unit_coordinate_center_norm(selected_current_tsne, center_point), mixed_index, return_constraint_metric.tolist()
    # return norm_coordinate(current_tsne), mixed_index, return_constraint_metric.tolist()

    return unit_coordinate_center_norm(guide_tsne_result, center_point), mixed_index, return_constraint_metric.tolist()
    # return guide_tsne_result, mixed_index, return_constraint_metric.tolist()


def generate_dynamic_data(expert_labels=None, is_spammer=None, worker_labels=None,
                          similarity_matrix=None, current_tsne=None,
                          filename=None, re_calculate_tsne=False):
    '''
    none
    :return:
    '''
    json_file = open(filename, "r")
    data = json.load(json_file)
    if re_calculate_tsne is False:
        return data
    else:
        raise ValueError("error")

def incre_guided_tsne_data():
    None

def generate_guided_tsne_data(dataname=None, center_points=None, similarity_matrix=None,
                              current_tsne=None, worker_labels=None, uncertainty=None,
                              posterior_dist_labels=None, selected_list=None,
                              re_calculate_tsne=False, buffer=True):
    """
    get guided tsne result
    :param center_point:
    :param similarity_matrix:
    :param current_tsne:
    :param uncertainty:
    :param PosteriorLabels:
    :param re_calculate_tsne:
    :return:
    """

    # id = "".join([ str(i) + " " for i in selected_list])
    # guided_tsne_file_path = os.path.join(config.server_data_root,
    #                                      dataname,
    #                                      config.info_data,
    #                                      id + config.info_guided_tsne_name)
    # if re_calculate_tsne is False and \
    #                 re_normalizing is False and os.path.exists(guided_tsne_file_path):
    #     json_file = open(guided_tsne_file_path, "r")
    #     data = json.load(json_file)
    #     return data
    guided_tsne_result, mixed_index, constraint_metric = \
        get_new_tsne_result(dataname=dataname,
                            center_points=center_points,
                            current_tsne=current_tsne,
                            similarity_matrix=similarity_matrix,
                            worker_labels=worker_labels,
                            posterior_dist_labels=posterior_dist_labels,
                            uncertainty=uncertainty,
                            selected_list=selected_list,
                            re_calculate_tsne=re_calculate_tsne)
    data = {}
    data[config.guided_tsne_name] = guided_tsne_result.tolist()
    data[config.mixed_index_name] = mixed_index.tolist()
    data[config.constraint_name] = constraint_metric
    data["center_points"] = center_points
    # if buffer:
    #     guided_tsne_file_path = os.path.join(config.server_data_root,
    #                                          dataname,
    #                                          config.info_data,
    #                                          config.info_guided_tsne_name)
    #     open(guided_tsne_file_path, "w").write(json.dumps(data))
    return data
