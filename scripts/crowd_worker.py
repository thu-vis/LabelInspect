import numpy as np
import os
import scipy.io as sio
from numpy import linalg as la
from matplotlib import pyplot as plt
from scripts.crowd_data import CrowdData
from scripts.crowd import mmcrowd
from scripts.configs import config
from scripts.segmentation import optimal_leaf_ordering, segmentation
import pickle


class SingleCrowdWorker(object):
    def __init__(self, confusion_matrix=None, mask_matrix=None):
        self.confusion_matrix = confusion_matrix
        # mask = np.array([[1, 1, 0, 0],
        #                  [1, 1, 0, 0],
        #                  [0, 0, 1, 1],
        #                  [0, 0, 1, 1]])
        # mask_matrix = mask
        # self.mask_matrix = mask_matrix
        segmentation_length = [2,2,2,2,2]
        if segmentation_length is not None:
            self.matrices = []
            offset = 0
            for seg_len in segmentation_length:
                try:
                    self.matrices.append(
                        self.confusion_matrix[offset:offset+seg_len, offset:offset+seg_len])
                    offset = offset + seg_len
                except:
                    break

    def set_confusion_matrix(self, confusion_matrix):
        self.confusion_matrix = confusion_matrix

    def get_distance_from_spammer_confusion_matrix(self):
        if not hasattr(self, "confusion_matrix") or self.confusion_matrix is None:
            raise ValueError("confusion matrix has not be set!")
        u, s, v = la.svd(self.confusion_matrix)
        score = (s ** 2).sum()
        min_score = 100
        for matrix in self.matrices:
            u, s, v = la.svd(matrix)
            inner_score = (s**2).sum() - (s.max())**2 + 1
            score = score -  1.0/ inner_score * (s.max())**2
            if(inner_score < min_score ):
                min_score = inner_score
        if( score < 0 ):
            score = 0
        return score ** 0.5

    def get_instance_from_spammer_confusion_matrix_according_to_selected_list(self, selected_list):
        # compare confusion matrix with matrix with rank 1, 2, 3,...
        selected_list = np.array(selected_list)
        selected_matrix = self.confusion_matrix[selected_list, :]
        selected_matrix = selected_matrix[:, selected_list]
        order = np.array(optimal_leaf_ordering(selected_matrix))
        selected_matrix = selected_matrix[order, :]
        selected_matrix = selected_matrix[:, order]
        seg_point = segmentation(selected_matrix)
        k = len(seg_point)
        u, s, v = la.svd(selected_matrix)
        square_s = np.array(s**2)
        square_s.sort()
        square_s = square_s[::-1]
        spammer_score_list = np.ones(k) * square_s.sum()
        for i in range(k):
            spammer_score_list[i] = (spammer_score_list[i] - square_s[:i+1].sum()) ** 0.5
        return spammer_score_list.mean()


    def _get_instance_from_spammer_confusion_matrix_according_to_selected_list(self, selected_list):
        # using hung's method for each block
        selected_list = np.array(selected_list)
        selected_matrix = self.confusion_matrix[selected_list, :]
        selected_matrix = selected_matrix[:, selected_list]
        order = np.array(optimal_leaf_ordering(selected_matrix))
        selected_matrix = selected_matrix[order,:]
        selected_matrix = selected_matrix[:,order]
        seg_point = segmentation(selected_matrix)
        seg_point.append(selected_matrix.shape[0])
        score = 0
        for i in range(len(seg_point)-1):
            block = selected_matrix[seg_point[i]:seg_point[i+1],:]
            block = block[:,seg_point[i]:seg_point[i+1]]
            u,s,v = la.svd(block)
            _score = (s**2).sum() - (s.max())**2
            _score = _score if _score > 0 else 0
            score = score + float(block.shape[0])/selected_matrix.shape[0] * _score ** 0.5
            # score = score +  _score ** 0.5
        return score

    def _get_instance_from_spammer_confusion_matrix_according_to_selected_list(self,selected_list):
        selected_list = np.array(selected_list)
        selected_matrix = self.confusion_matrix[selected_list,:]
        selected_matrix = selected_matrix[:,selected_list]
        u,s,v = la.svd(selected_matrix)
        score = (s**2).sum() - (s.max())**2
        if ( score < 0 ):
            score = 0
        return score ** 0.5

    def get_distance_from_rank_one_matrix_hung(self):
        if not hasattr(self, "confusion_matrix") or self.confusion_matrix is None:
            raise ValueError("confusion matrix has not be set!")
        u, s, v = la.svd(self.confusion_matrix)
        self.u = u
        self.v = v
        self.s = s
        low_rank_matrix = np.zeros((s.shape[0], s.shape[0]))
        low_rank_matrix[0, 0] = s[0]
        expend_s = np.zeros((s.shape[0], s.shape[0]))
        for i in range(s.shape[0]):
            expend_s[i, i] = s[i]
        self.reconstructed_confusion_matrix = np.dot(np.dot(u, expend_s), v)
        self.low_rank_matrix = np.dot(np.dot(u, low_rank_matrix), v)
        return la.norm(self.confusion_matrix - self.low_rank_matrix, ord="fro")


class CrowdWorkers(object):
    def __init__(self, crowd_data):
        self.dataname = crowd_data.dataname

        self.model = None
        self.true_labels = None
        self.instance_num = None
        self.label_num = None
        self.worker_num = None
        self.workers_labels = None

        self._init_from_crowd_data(crowd_data)

    def _init_from_crowd_data(self, crowd_data):
        self.true_labels = np.array(crowd_data.get_attr(config.true_labels_name)).reshape(-1)
        self.instance_num = self.true_labels.shape[0]
        self.label_num = crowd_data.get_attr(config.label_num_name)
        self.worker_num = crowd_data.get_attr(config.worker_num_name)
        self.workers_labels = np.array(crowd_data.get_attr(config.workers_labels_name)). \
            reshape(self.true_labels.shape[0], -1)

    def connect_to_model(self, mm):
        self.model = mm

    def init_from_mmcrowd(self, mm):
        confusion_matrices = mm.get_confusion_matrices()
        posterior_dist = mm.get_posterior_label_dist()
        self.posterior_labels = posterior_dist.argmax(axis=1)
        class_posterior = posterior_dist.sum(axis=0)
        print(class_posterior.shape)
        self.class_posterior = class_posterior / class_posterior.sum()
        # print(self.class_posterior, self.label_num)
        workers = []
        for i in range(confusion_matrices.shape[2]):
            #TODO
            workers.append(SingleCrowdWorker(self.__get_posterior_confusion_matrix(i)))
            # workers.append(SingleCrowdWorker(confusion_matrices[:, :, i]))
        self.workers = workers

    # def init_from_file(self, filename):
    #     mat = sio.loadmat(filename)
    #     confusion_matrices = mat["confusion_matrices"]
    #     self.class_posterior = np.ones(self.label_num) / float(self.label_num)
    #     workers = []
    #     for i in range(confusion_matrices.shape[2]):
    #         workers.append(SingleCrowdWorker(confusion_matrices[:, :, i]))
    #     self.workers = workers

    def __get_estimate_confusion_matrix(self, id):
        return self.workers[id].confusion_matrix

    def _get_ground_truth_confusion_matrix(self, id):
        if not hasattr(self, "true_labels"):
            raise ValueError("please init from crowd_data")
        confusion_matrix = np.zeros((self.label_num, self.label_num)) + 0.0001
        for i in range(self.instance_num):
            if self.workers_labels[i, id] > -1:
                confusion_matrix[self.workers_labels[i, id], self.true_labels[i]] += 1
        print(confusion_matrix)
        confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=0). \
            reshape(1, -1).repeat(repeats=self.label_num, axis=0)

        return confusion_matrix

    def _get_posterior_confusion_matrix(self, id):
        if not hasattr(self, "true_labels"):
            raise ValueError("please init from crowd_data")
        confusion_matrix = np.zeros((self.label_num, self.label_num))
        for i in range(self.instance_num):
            if self.workers_labels[i, id] > -1:
                confusion_matrix[self.workers_labels[i, id], self.posterior_labels[i]] += 1

        # confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=0). \
        #     reshape(1, -1).repeat(repeats=self.label_num, axis=0)
        # for i in range(confusion_matrix.shape[0]):
        #     sum = confusion_matrix[:,i].sum() + 1e-6
        #     confusion_matrix[:,i] = confusion_matrix[:,i] / float(sum)


        return confusion_matrix

    def __get_ground_truth_accuracy(self):
        posterior_accuracy = np.zeros(self.worker_num)
        for i in range(self.worker_num):
            total_sum = 0
            right_sum = 0
            for j in range(self.instance_num):
                if self.workers_labels[j, i] > - 1:
                    total_sum += 1
                    if self.workers_labels[j, i] == self.true_labels[j]:
                        right_sum += 1
            posterior_accuracy[i] = float(right_sum) / float(total_sum)
        return posterior_accuracy

    def __get_estimate_accuracy(self):
        accuracy = np.zeros(self.worker_num)
        for i in range(self.worker_num):
            for j in range(self.label_num):
                accuracy[i] = accuracy[i] + \
                              self.class_posterior[j] * self.workers[i].confusion_matrix[j, j]
        return accuracy

    def __get_posterior_accuracy(self):
        if not hasattr(self, "posterior_labels"):
            raise ValueError("posterior labels do not exists!")
        posterior_accuracy = np.zeros(self.worker_num)
        for i in range(self.worker_num):
            total_sum = 0
            right_sum = 0
            for j in range(self.instance_num):
                if self.workers_labels[j, i] > - 1:
                    total_sum += 1
                    if self.workers_labels[j, i] == self.posterior_labels[j]:
                        right_sum += 1
            posterior_accuracy[i] = float(right_sum) / float(total_sum)
        return posterior_accuracy

    def __get_spammer_scores(self):
        spammer_score = np.zeros(self.worker_num)
        for i in range(self.worker_num):
            spammer_score[i] = self.workers[i].get_distance_from_spammer_confusion_matrix()
        spammer_score_max = spammer_score.max()
        spammer_score_min = spammer_score.min()
        spammer_score = (spammer_score_max - spammer_score) / (spammer_score_max - spammer_score_min)
        return spammer_score

    def get_estimated_global_confusion_matrix(self):
        confusion_matrix = np.zeros((self.label_num, self.label_num))
        posterior_labels = self.posterior_labels
        worker_num = self.worker_num
        instance_num = self.instance_num
        label_num = self.label_num
        workers_labels = self.workers_labels

        for id in range(worker_num):
            for i in range(instance_num):
                if self.workers_labels[i, id] > -1:
                    confusion_matrix[workers_labels[i, id], posterior_labels[i]] += 1
        confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=0). \
            reshape(1, -1).repeat(repeats=label_num, axis=0)
        confusion_matrix[-1,:] = 1 - confusion_matrix[:-1,:].sum(axis=0)
        # confusion_matrix = np.array([[0.0, 0.0, 0., 0.],
        #           [0.0, 0.0, 0., 0.],
        #           [1, 1, 1, 1],
        #           [0., 0., 0., 0.]])
        return confusion_matrix

    def get_posterior_accuracy(self):
        worker_num = self.worker_num
        instance_num = self.instance_num
        workers_labels = self.workers_labels
        posterior_labels = self.posterior_labels

        posterior_accuracy = np.zeros(worker_num)
        for i in range(worker_num):
            total_sum = 0
            right_sum = 0
            for j in range(instance_num):
                if workers_labels[j,i] > -1:
                    total_sum += 1
                    if workers_labels[j,i] == posterior_labels[j]:
                        right_sum +=1
            posterior_accuracy[i] = float(right_sum) / float(total_sum)
        return posterior_accuracy

    def get_posterior_accuracy_according_to_selected_list(self, selected_list):
        worker_num = self.worker_num
        instance_num = self.instance_num
        workers_labels = self.workers_labels.copy()
        posterior_labels = self.posterior_labels
        selected_instance_indicator = [True if c in selected_list else False for c in posterior_labels]
        unselected_instance_indicator = [False if c in selected_list else True for c in posterior_labels]
        workers_labels[unselected_instance_indicator,:] = -1
        posterior_accuracy = np.zeros(worker_num)
        for i in range(worker_num):
            total_sum = 0
            right_sum = 0
            for j in range(instance_num):
                if workers_labels[j,i] > -1:
                    total_sum += 1
                    if workers_labels[j,i] == posterior_labels[j]:
                        right_sum +=1
            posterior_accuracy[i] = float(right_sum) / float(total_sum + 1e-6)
        return posterior_accuracy

    def get_spammer_scores(self):
        worker_num = self.worker_num
        workers_labels = self.workers_labels
        label_num = self.label_num
        instance_num = self.instance_num
        posterior_labels = self.posterior_labels
        spammer_score = np.zeros(self.worker_num)

        for i in range(worker_num):
            confusion_matrix = np.zeros((label_num, label_num))
            for j in range(instance_num):
                if workers_labels[j, i] > -1:
                    confusion_matrix[workers_labels[j, i], posterior_labels[j]] += 1
            for k in range(confusion_matrix.shape[0]):
                sum = confusion_matrix[:, k].sum() + 1e-6
                confusion_matrix[:, k] = confusion_matrix[:, k] / float(sum)
            worker = SingleCrowdWorker(confusion_matrix)
            spammer_score[i] = worker.get_distance_from_spammer_confusion_matrix()
        spammer_score_max = spammer_score.max()
        spammer_score_min = spammer_score.min()
        spammer_score = (spammer_score_max - spammer_score) / (spammer_score_max - spammer_score_min)
        return spammer_score

    def get_spammer_scores_according_to_selected_list(self, selected_list):
        worker_num = self.worker_num
        workers_labels = self.workers_labels
        label_num = self.label_num
        instance_num = self.instance_num
        posterior_labels = self.posterior_labels
        spammer_score = np.zeros(self.worker_num)

        for i in range(worker_num):
            confusion_matrix = np.zeros((label_num, label_num))
            for j in range(instance_num):
                if workers_labels[j, i] > -1:
                    confusion_matrix[workers_labels[j, i], posterior_labels[j]] += 1
            for k in range(confusion_matrix.shape[0]):
                sum = confusion_matrix[:, k].sum() + 1e-6
                confusion_matrix[:, k] = confusion_matrix[:, k] / float(sum)
            worker = SingleCrowdWorker(confusion_matrix)
            spammer_score[i] = worker.get_instance_from_spammer_confusion_matrix_according_to_selected_list(selected_list)
        spammer_score_max = spammer_score.max()
        spammer_score_min = spammer_score.min()
        spammer_score = (spammer_score_max - spammer_score) / (spammer_score_max - spammer_score_min)
        return spammer_score

    def get_worker_changed_list_according_to_selected_list(self,
                                                           selected_list,
                                                           present_spammer_score,
                                                           present_posterior_acc):
        worker_num = self.worker_num
        instance_num = self.instance_num
        workers_labels = self.workers_labels.copy()
        label_num = self.label_num
        posterior_labels = self.pre_posterior_labels
        if(len(posterior_labels) < 1):
            return {}
        selected_instance_indicator = [True if c in selected_list else False for c in posterior_labels]
        unselected_instance_indicator = [False if c in selected_list else True for c in posterior_labels]
        workers_labels[unselected_instance_indicator, :] = -1
        posterior_accuracy = np.zeros(worker_num)
        for i in range(worker_num):
            total_sum = 0
            right_sum = 0
            for j in range(instance_num):
                if workers_labels[j, i] > -1:
                    total_sum += 1
                    if workers_labels[j, i] == posterior_labels[j]:
                        right_sum += 1
            posterior_accuracy[i] = float(right_sum) / float(total_sum + 1e-6)

        spammer_score = np.zeros(self.worker_num)
        for i in range(worker_num):
            confusion_matrix = np.zeros((label_num, label_num))
            for j in range(instance_num):
                if workers_labels[j, i] > -1:
                    confusion_matrix[workers_labels[j, i], posterior_labels[j]] += 1
            for k in range(confusion_matrix.shape[0]):
                sum = confusion_matrix[:, k].sum() + 1e-6
                confusion_matrix[:, k] = confusion_matrix[:, k] / float(sum)
            worker = SingleCrowdWorker(confusion_matrix)
            spammer_score[i] = worker.get_instance_from_spammer_confusion_matrix_according_to_selected_list(selected_list)
        spammer_score_max = spammer_score.max()
        spammer_score_min = spammer_score.min()
        spammer_score = (spammer_score_max - spammer_score) / (spammer_score_max - spammer_score_min)

        delta_spammer = present_spammer_score - spammer_score
        delta_acc = present_posterior_acc - present_posterior_acc
        d = delta_spammer**2 + delta_acc**2
        d = d ** 0.5
        res = {}
        order = d.argsort()[::-1]
        for i in order[:5]:
            if d[i] < 0.09:
                break
            res[str(i)] = {
                "dx": delta_spammer[i],
                "dy": delta_acc[i]
            }

        return res

    def similarity_AA(self, selected_list = 0):
        instance_num = self.instance_num
        worker_num = self.worker_num
        label_num = self.label_num
        worker_labels = np.array(self.workers_labels) \
            .reshape(instance_num, worker_num).copy()
        if selected_list is None:
            raise ValueError("selected list is empty!!")
        neib_worker = []
        for i in range(worker_num):
            neib_worker.append([])
        for i in range(instance_num):
            for j, label in enumerate(worker_labels[i, :]):
                if label != -1 and label in selected_list:
                    neib_worker[j].append(i)
                else:
                    worker_labels[i, j] = -1

        similarity = np.zeros((worker_num, worker_num))
        for i in range(worker_num):
            for j in range(i + 1, worker_num):
                common_item = 0
                intersection = set(neib_worker[i]).intersection(neib_worker[j])
                # for s in intersection:
                #     print(worker_labels[s,i],end=" ")
                # exit()
                for item in intersection:
                    if worker_labels[item][i] == worker_labels[item][j]:
                        common_item = common_item + 1
                        similarity[i][j] += 1.0 / np.log(
                            np.sum((worker_labels[item, :] == worker_labels[item][i]))
                        )
                similarity[i][j] /= float(common_item + 1)
                similarity[i][j] *= float(len(intersection))
                similarity[j][i] = similarity[i][j]
        return similarity

    def similarity(self, selected_list, Method_name):
        if Method_name == "RA":
            return self.similarity_RA(selected_list)
        elif Method_name == "SRW":
            return self.similarity_SRW(selected_list)
        elif Method_name == "SimRank":
            return self.similarity_SimRank(selected_list)
        else:
            raise ValueError("similarity method name error!!!")

    def similarity_RA(self, selected_list = 0):
        # if hasattr(self,"RA_simi"):
        #     return self.RA_simi.copy()
        buffer_name = "".join([str(i) for i in selected_list])
        buffer_file_name = os.path.join(config.data_root, self.dataname,
                                   config.worker_simi_buffer_path, buffer_name)
        if os.path.exists(buffer_file_name):
            fr = open(buffer_file_name,"rb")
            return pickle.load(fr)
        instance_num = self.instance_num
        worker_num = self.worker_num
        label_num = self.label_num
        worker_labels = np.array(self.workers_labels) \
            .reshape(instance_num, worker_num).copy()
        if selected_list is None:
            raise ValueError("selected list is empty!!")
        neib_worker = []
        for i in range(worker_num):
            neib_worker.append([])
        for i in range(instance_num):
            for j, label in enumerate(worker_labels[i, :]):
                if label != -1 and label in selected_list:
                    neib_worker[j].append(i)
                else:
                    worker_labels[i, j] = -1

        degree = np.zeros((instance_num, label_num))
        for i in range(instance_num):
            for j in range(worker_num):
                degree[i][worker_labels[i,j]] += 1

        similarity = np.zeros((worker_num, worker_num))
        for i in range(worker_num):
            for j in range(i + 1, worker_num):
                common_item = 0
                intersection = set(neib_worker[i]).intersection(neib_worker[j])
                # for s in intersection:
                #     print(worker_labels[s,i],end=" ")
                # exit()
                for item in intersection:
                    if worker_labels[item][i] == worker_labels[item][j]:
                        common_item = common_item + 1
                        similarity[i][j] += 1.0 / float(degree[item][worker_labels[item][i]])
                # similarity[i][j] /= float(common_item + 1)
                # similarity[i][j] *= float(len(intersection))
                similarity[j][i] = similarity[i][j]
        self.RA_simi = similarity
        return similarity.copy()

    def similarity_SRW(self, selected_list = 0):
        # if hasattr(self,"SRW_simi"):
        #     return self.SRW_simi.copy()
        instance_num = self.instance_num
        worker_num = self.worker_num
        label_num = self.label_num
        worker_labels = np.array(self.workers_labels) \
            .reshape(instance_num, worker_num).copy()
        instance_similarity = self.instance_similarity
        if selected_list is None:
            raise ValueError("selected list is empty!!")

        instance_degree = np.zeros((instance_num, label_num))
        worker_degree = np.zeros(worker_num)
        for i in range(instance_num):
            for j in range(worker_num):
                if worker_labels[i,j] > -1:
                    instance_degree[i][worker_labels[i, j]] += 1
                    worker_degree[j] = worker_degree[j] + 1

        P1 = np.zeros((instance_num * label_num, worker_num))
        P2 = np.zeros((worker_num, instance_num * label_num))
        for i in range(instance_num):
            for j in range(worker_num):
                if worker_labels[i][j] > -1:
                    P1[i * label_num + worker_labels[i][j], j] = 1.0 / instance_degree[i][worker_labels[i][j]]
                    P2[j, i * label_num + worker_labels[i][j]] = 1.0 / worker_degree[j]
        P = np.zeros((instance_num * label_num + worker_num, instance_num * label_num + worker_num))
        P[:instance_num * label_num, instance_num * label_num :] = P1
        P[instance_num * label_num:, :instance_num * label_num] = P2

        total_t = 10
        state = P.transpose()
        for t in range(total_t):
            state = np.dot(P.transpose(), state)

        worker_state = P[instance_num * label_num :, instance_num * label_num :]
        E = (worker_labels > -1).sum()
        similarity = np.zeros((worker_num, worker_num))
        for i in range(worker_num):
            for j in range(worker_num):
                similarity[i][j] = (worker_degree[i] * worker_state[i][j] +
                                    worker_degree[j] * worker_state[j][i]) / float(2 * E)
        self.SRW_simi = similarity

        return similarity.copy()

    def similarity_SimRank(self, selected_list = 0):
        # if hasattr(self,"SimRank_simi"):
        #     return self.SimRank_simi.copy()
        buffer_name = "".join([str(i) for i in selected_list])
        buffer_file_name = os.path.join(config.data_root, self.dataname,
                                   config.worker_simi_buffer_path, buffer_name)
        if os.path.exists(buffer_file_name):
            fr = open(buffer_file_name,"rb")
            return pickle.load(fr)
        print("similarity_SimRank", selected_list)
        instance_num = self.instance_num
        worker_num = self.worker_num
        label_num = self.label_num
        posterior_labels = self.model.get_posterior_labels()
        worker_labels = np.array(self.workers_labels) \
            .reshape(instance_num, worker_num).copy()
        # process according to selected_list
        for i in range(worker_labels.shape[0]):
            for j in range(worker_labels.shape[1]):
                if worker_labels[i,j] not in selected_list:
                    worker_labels[i,j] = -1
                if worker_labels[i,j] > -1 :
                    if worker_labels[i,j] == posterior_labels[i]:
                        worker_labels[i,j] = 1
                    else:
                        worker_labels[i,j] = 0

        # label_num = list(set(worker_labels.reshape(-1).tolist())) - 1
        label_num = 2


        instance_similarity = self.instance_similarity
        if selected_list is None:
            raise ValueError("selected list is empty!!")

        expand_worker_labels = np.ones((instance_num * label_num, worker_num)) * 0
        WorkerNeib = []
        for i in range(worker_num):
            WorkerNeib.append([])
        for i in range(instance_num):
            for j in range(worker_num):
                if worker_labels[i][j] > -1:
                    expand_worker_labels[i * label_num + worker_labels[i][j]][j] = 1
                # WorkerNeib[j].append([ i, worker_labels[i][j] ])
        for i in range(worker_num):
            expand_worker_labels[:,i] = expand_worker_labels[:,i] / expand_worker_labels[:,i].sum()
        expand_instance_similarity = np.zeros((instance_num * label_num, instance_num * label_num))
        for i in range(instance_num):
            for j in range(instance_num):
                for k in range(label_num):
                    expand_instance_similarity[i * label_num + k, j * label_num + k] = instance_similarity[i][j] * np.exp(float(-k * 1))
        print("expand_worker")
        similarity = np.zeros((worker_num, worker_num))
        similarity = np.dot(np.dot(expand_worker_labels.transpose(), expand_instance_similarity), expand_worker_labels)

        mask_workers = (worker_labels != -1).sum(axis=0) < 40
        similarity[mask_workers,:] = 0
        similarity[:,mask_workers] = 0

        # C = 1.0
        # for i in range(worker_num):
        #     for j in range(i + 1, worker_num):
        #         Oi = WorkerNeib[i]
        #         Oj = WorkerNeib[j]
        #         for x in Oi:
        #             for y in Oj:
        #                 if (x % label_num == y % label_num):
        #                     similarity[i][j] = similarity[i][j] + instance_similarity[x//label_num][y//label_num]
        #                 # if Oi[x] != -1 and Oi[x] == Oj[y]:
        #                 #     similarity[i][j] = similarity[i][j] + instance_similarity[x//label_num][y//label_num]
        #         similarity[i][j] = similarity[i][j] * C / len(Oi) / len(Oj)
        #         similarity[j][i] = similarity[i][j]

        self.SimRank_simi = similarity
        fw = open(buffer_file_name,"wb")
        pickle.dump(similarity, fw)
        return similarity.copy()

    def _empty_get_worker_acc_per_class(self):
        worker_num = self.worker_num
        instance_num = self.instance_num
        workers_labels = self.workers_labels
        posterior_labels = self.posterior_labels



    @property
    def posterior_labels(self):
        return self.model.get_posterior_labels()
    @property
    def pre_posterior_labels(self):
        return self.model.get_pre_posterior()

    @property
    def instance_similarity(self):
        return self.model.get_simi_matrix()


if __name__ == "__main__":
    a = np.array([[0.0, 0.0, 0., 0.],
                  [0.0, 0.0, 0., 0.],
                  [1, 1, 1, 1],
                  [0., 0., 0., 0.]])
    b = np.array([[0.7, 0.25, 0., 0.],
                  [0.3, 0.75, 0., 0.],
                  [0., 0., 0.7, 0.3],
                  [0., 0., 0.3, 0.7]])
    c = np.array([[0.731, 0.408, 0.016, 0.013],
                  [0.245, 0.531, 0.016, 0.010],
                  [0.011, 0.030, 0.664, 0.394],
                  [0.013, 0.031, 0.304, 0.579]])
    d = np.array([[0.716, 0.096, 0.025, 0.019],
                  [0.213, 0.856, 0.036, 0.032],
                  [0.031, 0.023, 0.915, 0.933],
                  [0.039, 0.023, 0.023, 0.015]])
    e = np.zeros((4,4)) * 0.25
    mask = np.array([[1, 1, 0, 0],
                     [1, 1, 0, 0],
                     [0, 0, 1, 1],
                     [0, 0, 1, 1]])
    sa = SingleCrowdWorker(a, mask)
    sb = SingleCrowdWorker(b, mask)
    sc = SingleCrowdWorker(c, mask)
    sd = SingleCrowdWorker(d, mask)
    se = SingleCrowdWorker(e, mask)
    print(sa.get_distance_from_spammer_confusion_matrix())
    print(sb.get_distance_from_spammer_confusion_matrix())
    print(sc.get_distance_from_spammer_confusion_matrix())
    print(sd.get_distance_from_spammer_confusion_matrix())
    print(se.get_distance_from_spammer_confusion_matrix())
    # print(sa.get_distance_from_rank_one_matrix_hung())
    # print(sb.get_distance_from_rank_one_matrix_hung())
    # print(sc.get_distance_from_rank_one_matrix_hung())
    # print(sd.get_distance_from_rank_one_matrix_hung())

    exit()
    crowd_data = CrowdData("monkey")
    mm = mmcrowd()
    mm.from_crowd_data(crowd_data)
    # mm.train()
    # mm.save_confusion_matrices(os.path.join(config.data_root, crowd_data.dataname,
    #                          config.origin_data, "confusion_matrices.mat"))
    # mm.save_posterior_labels(os.path.join(config.data_root, crowd_data.dataname,
    #                          config.origin_data, "posterior_labels.mat"))

    crowd_workers = CrowdWorkers(crowd_data.dataname)
    crowd_workers.init_from_crowd_data(crowd_data)
    crowd_workers.init_from_file(os.path.join(config.data_root, crowd_data.dataname,
                                              config.origin_data, "confusion_matrices.mat"))
    crowd_workers.get_estimate_accuracy()
    # print(crowd_workers.get_estimate_confusion_matrix(2))
    # print(crowd_workers.get_ground_truth_confusion_matrix(2))
