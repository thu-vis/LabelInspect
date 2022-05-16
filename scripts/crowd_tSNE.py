import numpy as np
import os
import scipy.io as sio
from scipy.stats import multivariate_normal, entropy
import json
from sklearn import manifold
import warnings
import pickle

from sklearn.metrics.pairwise import euclidean_distances
from scripts.backend import load_static_data, decom_similarity_matrix
from scripts.configs import config
from scripts.tsne import guide_tsne, incremental_tsne

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

class CrowdTSNE(object):
    def __init__(self, crowd_data):
        self.dataname = crowd_data.dataname

        self.model = None
        self.instance_num = crowd_data.get_attr(config.instance_num_name)
        self.worker_num = crowd_data.get_attr(config.worker_num_name)
        self.worker_labels = np.array(crowd_data.get_attr(config.workers_labels_name)).reshape(-1).tolist()
        self.similarity_matrix = decom_similarity_matrix(
            crowd_data.get_attr(config.similarity_matrix_name),
            self.instance_num)
        # self.original_tsne = crowd_data.get_attr(config.tsne_name)
        self.tsne_buffer = {}

    def connect_to_model(self, mm):
        self.model = mm

    def tsne_update(self, center_points,
                    dynamic_info, selected_list, TSNE,
                    last_round_list, changed_indicator,
                    re_calculate_tsne=False):
        worker_labels = self.worker_labels
        instance_num = self.instance_num
        worker_num = self.worker_num
        # posterior_labels = self.model.get_posterior_labels()
        posterior_labels = np.array(dynamic_info[config.posterior_labels_name])
        selected_instance_indicator = [True if c in selected_list else False for c in posterior_labels]
        unselected_instance_indicator = [False if c in selected_list else True for c in posterior_labels]
        print("selected num: %s, unselected num: %s" %(sum(selected_instance_indicator), sum(unselected_instance_indicator)))
        selected_posterior_labels = posterior_labels[selected_instance_indicator]
        self.original_tsne = TSNE

        # TODO: Get distance matrix
        feature = get_feature(self.dataname)
        distance_matrix = euclidean_distances(feature, feature) / 1.414
        distance_matrix = distance_matrix[selected_instance_indicator, :]
        distance_matrix = distance_matrix[:, selected_instance_indicator]

        mixed_index = dynamic_info[config.instance_list_name][:20]
        selected_current_tsne = np.array(self.original_tsne)[selected_instance_indicator, :]
        selected_instance_num = sum(selected_instance_indicator)

        # for i in range(5):
        #     center_points.append({
        #         "x": 0.5,
        #         "y": 0.5,
        #         "class": 10
        #     })

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
        for i in selected_list: # + [10]:
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

        distance_arc = get_high_dimension_distance_to_arc(self.dataname, selected_list,
                                                          posterior_labels)  # [selected_instance_indicator]
        aug_distance_matrix = np.ones((constraint_num + selected_instance_num,
                                       constraint_num + selected_instance_num))

        distance_to_center = 0.07

        for i in range(selected_instance_num):
            _class = int(selected_posterior_labels[i])
            if i in mixed_index:
                labels = list(set(worker_labels[i * worker_num:i * worker_num + worker_num])) #  + [10]
                for j in range(constraint_num):
                    if j != 10:
                        if class_effect_of_constraints[j] in labels:
                            constraint_metric[j, i] = 1
                            aug_distance_matrix[j,constraint_num + i] = distance_arc[j, i]
                            aug_distance_matrix[constraint_num + i,j] = distance_arc[j, i]
                        # constraint_metric[-1, i] = 2
                    else:
                         if class_effect_of_constraints[j] in labels:
                            constraint_metric[j,i] = 1
                            aug_distance_matrix[j, constraint_num + i] = distance_to_center
                            aug_distance_matrix[constraint_num + i, j] = distance_to_center
            else:
                # constraint_metric[index, i] = 1
                for j in range(constraint_num):
                    if class_effect_of_constraints[j] == _class:
                        constraint_metric[j, i] = 1
                        aug_distance_matrix[j, constraint_num + i] = distance_arc[j, i]
                        aug_distance_matrix[constraint_num + i, j] = distance_arc[j, i]

        aug_distance_matrix[constraint_num:, constraint_num:] = distance_matrix
        distance_matrix = aug_distance_matrix
        print(distance_matrix.shape)

        return_constraint_metric = constraint_metric.copy()
        constraint_metric = constraint_metric.tolist()

        if selected_list == [4, 5] and self.dataname == "bird":
            datatype = "bad_bird"
        else:
            datatype = self.dataname
        id = "".join([str(i) + "_" for i in selected_list])
        unnorm_guided_tsne_file_path = os.path.join(config.data_root,
                                                    self.dataname,
                                                    config.info_data,
                                                    id + config.info_unnorm_guided_tsne_name)

        if (id not in self.tsne_buffer) and  \
            (re_calculate_tsne is True or (not os.path.exists(unnorm_guided_tsne_file_path))):
            print("re-calculate tsne result....")
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
                "guide_tsne_result": guide_tsne_result.tolist(),
                "selected_indicator": selected_instance_indicator
            }
            open(unnorm_guided_tsne_file_path, "w") \
                .write(json.dumps(result_dir))
            self.tsne_buffer[id] = result_dir
        elif id not in self.tsne_buffer:
            print("loading pre-computed result...")
            result_dir = json.load(open(unnorm_guided_tsne_file_path,"r"))
            guide_tsne_result = np.array(result_dir["guide_tsne_result"])
            self.tsne_buffer[id] = result_dir
        else:
            result_dir = self.tsne_buffer[id]
            prev_guide_tsne_result = np.array(result_dir["guide_tsne_result"])
            prev_selected_instance_indicator = np.array(result_dir["selected_indicator"])
            print("begin incremental tsne")
            print("prev num:%s, present num:%s"%(sum(prev_selected_instance_indicator), sum(selected_instance_indicator)))
            fake_tsne_result = np.zeros((instance_num, 2))
            fake_list = np.zeros(instance_num).astype(bool)
            fake_tsne_result[selected_instance_indicator, :] = selected_current_tsne
            fake_list[selected_instance_indicator] = True
            fake_tsne_result[prev_selected_instance_indicator, :] = prev_guide_tsne_result
            fake_list[prev_selected_instance_indicator] = False
            selected_current_tsne = fake_tsne_result[selected_instance_indicator, :]
            incre_list = np.array(range(sum(selected_instance_indicator)))[fake_list[selected_instance_indicator]]
            changed_indicator[unselected_instance_indicator] = False
            label_changed_list = np.array(range(sum(selected_instance_indicator)))[changed_indicator[selected_instance_indicator]]
            label_changed_list = label_changed_list.tolist() + incre_list.tolist()
            label_changed_list = list(set(label_changed_list))
            fake_list = np.zeros(instance_num).astype(bool)
            fake_list[last_round_list] = True
            last_round_list = np.array(range(sum(selected_instance_indicator)))[fake_list[selected_instance_indicator]]
            print("label_changed_length: %s"%(len(label_changed_list)))
            print(np.array(range(instance_num))[changed_indicator], last_round_list)
            if 0:
                guide_tsne_result = selected_current_tsne
            else:
                guide_tsne_result = incremental_tsne(
                    selected_current_tsne.copy(),
                    distance_matrix,
                    center_constraints,
                    constraint_metric,
                    last_round_list,
                    label_changed_list)

        guide_tsne_result = unit_coordinate_center_norm(guide_tsne_result, center_point)
        data = {}
        data[config.guided_tsne_name] = guide_tsne_result.tolist()
        data[config.mixed_index_name] = mixed_index
        # data[config.constraint_name] = None
        data["center_points"] = center_points

        return data