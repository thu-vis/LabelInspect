import numpy as np
import os
import scipy.io as sio
import json
from scripts.backend import load_static_data
from scripts.crowd_data import CrowdData
from scripts.crowd import mmcrowd
from scripts.configs import config


class CrowdInstances(object):
    def __init__(self, crowd_data):
        self.dataname = crowd_data.dataname

        self.model = None
        self.true_labels = None
        self.instance_num = None
        self.label_num = None
        self.workers_labels = None

        self.affinity = None

        self._init_from_crowd_data(crowd_data)

    def _init_from_crowd_data(self, crowd_data):
        self.true_labels = np.array(crowd_data.get_attr("true_labels")).reshape(-1)
        self.instance_num = self.true_labels.shape[0]
        self.label_num = crowd_data.get_attr("LabelTotalNum")
        self.workers_labels = np.array(crowd_data.get_attr("WorkerLabels")). \
            reshape(self.true_labels.shape[0], -1)

    def connect_to_model(self, mm):
        self.model = mm
        # self.posterior_labels = mm.get_posterior_labels()
        # self.majority_voting_result = mm.get_majority_voting_result()

    # def init_from_file(self, filename):
    #     mat = sio.loadmat(filename)
    #     self.posterior_labels = mat["posterior_labels"]
    #     print("posterior labels of CrowdInstances has been loaded!")

    def get_diversity(self, simi_matrix, present_expert_instance_validation):
        # simi_matrix = self.instance_similarity
        # present_expert_instance_validation = self.expert_instance_validation
        validation_indicator = (present_expert_instance_validation != -1)
        unvalidation_indicator = (present_expert_instance_validation == -1)
        if sum(validation_indicator) == 0:
            return np.zeros(simi_matrix.shape[0])
        # n_valided = sum(validation_indicator)
        # n_unvalided = sum(unvalidation_indicator)
        # K_unlabeled = simi_matrix[unvalidation_indicator,:]
        # coef_2 = float(n_valided + 1) / (n_valided + n_unvalided)
        # coef_3 = float(n_unvalided -1) / (n_valided + n_unvalided)
        # K1 = K_unlabeled[:,unvalidation_indicator]
        # K1_hat = K_unlabeled[:,validation_indicator]
        # k_1 = np.array([ K1[i,i] for i in range(K1.shape[0])]) * 0.5
        # k_2 = coef_2 * K1.sum(axis=1)
        # k_3 = coef_3 * K1_hat.sum(axis=1)
        # representativeness = np.zeros((n_valided + n_unvalided))
        # representativeness[unvalidation_indicator] = k_1 - k_2 + k_3
        # representativeness = representativeness.max() - representativeness
        # representativeness = representativeness / representativeness.max()
        # representativeness[validation_indicator] = -1
        rest_simi = simi_matrix[unvalidation_indicator,:][:,validation_indicator]
        rest_min_simi = rest_simi.max(axis=1)
        score = np.ones(simi_matrix.shape[0])
        score[unvalidation_indicator] = rest_min_simi
        diversity = 1 - score
        return diversity

    def get_score(self):
        uncertainty = np.array(self.get_uncertainty())
        instance_num = uncertainty.shape[0]
        simi_matrix = self.instance_similarity
        present_expert_instance_validation = self.expert_instance_validation.copy()
        self.r = 0
        r = self.r * np.exp(- sum(present_expert_instance_validation != -1) / float(instance_num))
        score = np.ones(simi_matrix.shape[0]) * -1
        for i in range(20):
            diversity = self.get_diversity(simi_matrix, present_expert_instance_validation)
            tmp_score = (1 - r) * uncertainty + r * diversity
            max_index = tmp_score.argmax()
            score[max_index] = 1 - 0.01 * float(i)
            present_expert_instance_validation[max_index] = 1
            uncertainty[max_index] = 0
        return score

    def random_walk_score(self, worker_similarity, spammer_score):
        uncertainty = np.array(self.get_uncertainty()).copy()
        # uncertainty[uncertainty < 0.3] = 0
        posterior_labels = self.get_posterior_labels()
        instance_num = uncertainty.shape[0]
        worker_labels = self.workers_labels
        true_labels = self.model.true_labels - 1
        label_indicator = (worker_labels!= -1).astype(int)
        instance_similarity = self.instance_similarity
        worker_num = worker_similarity.shape[0]
        instance_affinity_matrix = instance_similarity / instance_similarity.sum(axis=0)\
            .reshape(1,-1).repeat(axis=0,repeats=instance_num)
        worker_similarity = worker_similarity + 1e-6
        worker_affinity_matrix = worker_similarity / worker_similarity.sum(axis=0)\
            .reshape(1,-1).repeat(axis=0,repeats=worker_num)
        present_expert_instance_validation = self.expert_instance_validation.copy()
        # present_expert_worker_validation = self.
        validated_list = np.array(range(instance_num))[present_expert_instance_validation != -1]
        uncertainty[validated_list] = 2
        k = 5
        alpha = 0.005
        added_uncertainty = uncertainty / uncertainty.sum()
        added_list = []
        correspond_list = []
        for validated_id in validated_list:
            simi_vector = instance_similarity[validated_id, :]
            affected_ids = simi_vector.argsort()[::-1][:k]
            for id in affected_ids:
                if posterior_labels[id] != posterior_labels[validated_id]:
                    added_uncertainty[id] = added_uncertainty[id] + alpha * simi_vector[id]
                    added_list.append(id)
                    correspond_list.append(validated_id)
        modified_uncertainty = added_uncertainty / added_uncertainty.sum()
        d = 0.85
        alpha_II = 0.9
        alpha_WW = 0.9
        alpha_IW = 0.005
        alpha_WI = 2
        # print("parameter k:%s, alpha:%s, d:%s, alpha_II:%s, alpha_WW:%s, alpha_IW:%s, alpha_WI:%s"
        #       %(k, alpha, d, alpha_II, alpha_WW, alpha_IW, alpha_WI))
        affinity_matrix = np.zeros((instance_num + worker_num, instance_num + worker_num))
        prior_saliency = np.zeros(instance_num + worker_num)
        affinity_matrix[:instance_num, :instance_num] = alpha_II * instance_affinity_matrix
        affinity_matrix[instance_num:, :instance_num] = alpha_IW * label_indicator.transpose()
        affinity_matrix[:instance_num, instance_num:] = alpha_WI * label_indicator
        affinity_matrix[instance_num:, instance_num:] = alpha_WW * worker_affinity_matrix
        affinity_matrix = affinity_matrix / affinity_matrix.sum(axis=0) \
            .reshape(1,-1).repeat(axis=0, repeats=(instance_num + worker_num))
        prior_saliency[:instance_num] = modified_uncertainty / modified_uncertainty.sum()
        prior_saliency[instance_num:] = spammer_score / spammer_score.sum()
        prior_saliency = prior_saliency / prior_saliency.sum()

        self.affinity = affinity_matrix

        state = np.ones(instance_num + worker_num) / float(instance_num + worker_num)
        err = 100
        pre_state = np.zeros(instance_num + worker_num)
        for j in range(100):
            state = (1 - d) * prior_saliency + d * np.dot(affinity_matrix, state)
            assert abs(state.sum() - 1) < 1e-3
            if((pre_state - state)**2).sum() < 1e-66:
                print("state converge in %s iteration"%(j))
                break
            else:
                pre_state = state.copy()
            if j == 900:
                continue
        # if idx > 9:
        #     state[:instance_num] = np.random.rand(instance_num)
        state[:instance_num][present_expert_instance_validation!= -1] = 0

        most_uncertain_index_8 = uncertainty.argsort()[::-1][:8]
        most_uncertain_index_20 = uncertainty.argsort()[::-1][:20]
        uncertain_mis_num_8 = sum(true_labels[most_uncertain_index_8] != posterior_labels[most_uncertain_index_8])
        uncertain_mis_num_20 = sum(true_labels[most_uncertain_index_20] != posterior_labels[most_uncertain_index_20])
        most_random_state_index_8 = state[:instance_num].argsort()[::-1][:8]
        most_random_state_index_20 = state[:instance_num].argsort()[::-1][:20]
        random_state_mis_num_8 = sum(true_labels[most_random_state_index_8] != posterior_labels[most_random_state_index_8])
        random_state_mis_num_20 = sum(true_labels[most_random_state_index_20] != posterior_labels[most_random_state_index_20])
        # print("8---uncertain mis_ratio:%s, random walk mis_radio:%s"%(uncertain_mis_num_8, random_state_mis_num_8))
        # print("20---uncertain mis_ratio:%s, random walk mis_radio:%s"%(uncertain_mis_num_20, random_state_mis_num_20))

        return state[:instance_num], state[instance_num:]

    def pure_instance_random_walk_score(self):
        uncertainty = np.array(self.get_uncertainty())
        instance_num = uncertainty.shape[0]
        uncertainty = uncertainty / uncertainty.sum()
        similarity_matrix = self.instance_similarity
        present_expert_instance_validation = self.expert_instance_validation.copy()
        self.d = 0.85
        d = self.d
        validated_indicator = present_expert_instance_validation != -1
        similarity_matrix[validated_indicator,:] = 1e-6
        similarity_matrix[:,validated_indicator] = 1e-6
        affinity_matrix = similarity_matrix / similarity_matrix.sum(axis=0)\
            .reshape(1,-1).repeat(axis=0,repeats=instance_num)
        affinity_matrix[:,validated_indicator] = 0
        affinity_matrix[validated_indicator,:] = 0
        state = np.ones(instance_num) / float(instance_num)
        state[validated_indicator] = 0
        state = state / state.sum()
        err = 100
        pre_state = np.zeros(instance_num)
        for j in range(1000):
            state = (1 - d) * uncertainty + d * np.dot(affinity_matrix, state)
            assert abs(state.sum() - 1) < 1e-3
            if ((pre_state - state)**2).sum() < 1e-66:
                print("state converge in %s iteration"%(j))
                break
            else:
                pre_state = state.copy()
            if j == 999:
                continue
        # state[present_expert_instance_validation != -1] = 0
        assert (abs(state[validated_indicator].sum()) < 1e-3)
        return state

    def instance_random_walk_score(self):
        uncertainty = np.array(self.get_uncertainty())
        posterior_labels = self.get_posterior_labels()
        instance_num = uncertainty.shape[0]
        similarity_matrix = self.instance_similarity
        present_expert_instance_validation = self.expert_instance_validation.copy()
        # change prior saliency according to expert validation
        uncertainty = uncertainty / uncertainty.sum()
        validated_list = np.array(range(instance_num))[present_expert_instance_validation!= -1]
        k = 5
        alpha = 0.000
        for validated_id in validated_list:
            simi_vector = similarity_matrix[validated_id,:]
            affected_ids = simi_vector.argsort()[::-1][:k]
            for id in affected_ids:
                if posterior_labels[id] != posterior_labels[validated_id]:
                    uncertainty[id] = uncertainty[id] + alpha * simi_vector[id]
        uncertainty = uncertainty / uncertainty.sum()
        # change affinity matrix according to expert validation
        affinity_matrix = similarity_matrix / similarity_matrix.sum(axis=0) \
            .reshape(1, -1).repeat(axis=0, repeats=instance_num)
        self.d = 0.85
        d = self.d
        state = np.ones(instance_num) / float(instance_num)
        err = 100
        pre_state = np.zeros(instance_num)
        for j in range(1000):
            state = (1 - d) * uncertainty + d * np.dot(affinity_matrix, state)
            assert abs(state.sum() - 1) < 1e-3
            if ((pre_state - state) ** 2).sum() < 1e-66:
                print("state converge in %s iteration" % (j))
                break
            else:
                pre_state = state.copy()
            if j == 999:
                continue
        state[present_expert_instance_validation != -1] = 0
        print("k:%s, alpha:%s, d:%s" %(k, alpha, d))
        return state


    def get_uncertainty(self):
        data_uncertainty = self.get_data_uncertainty()
        model_uncertainty = self.get_model_uncertainty()
        solution_uncertainty = self.get_solution_uncertainty()
        # TODO: this combining method is proposed by Liu's IJCAI 2017
        # the parameters are approximately the same as those described in the paper
        uncertainty = 0.4 * np.exp(data_uncertainty) + 0.54 * np.exp(model_uncertainty) + \
            0.06 * np.exp(solution_uncertainty)
        uncertainty = uncertainty - 0.99
        # the uncertainty of instanced validated before should be zero
        present_expert_instance_validation = self.expert_instance_validation
        validation_indicator = (present_expert_instance_validation == -1)
        uncertainty = uncertainty * validation_indicator

        return uncertainty

    def get_affinity_matrix(self):
        return self.affinity.copy()

    def get_data_uncertainty(self):
        if self.workers_labels is None:
            raise ValueError("worker labels is missing!!!")
        if self.posterior_labels is None:
            raise ValueError("posterior_labels is missing!!!")

        posterior_labels = self.posterior_labels
        workers_labels = self.workers_labels

        assert posterior_labels.shape[0] == workers_labels.shape[0]
        entropy = np.zeros(posterior_labels.shape[0])
        for i in range(entropy.shape[0]):
            entropy[i] = self.get_entropy_by_id(i)
        return entropy

    # TODO: expert validation is not taken into account
    def get_model_uncertainty(self):
        if self.workers_labels is None:
            raise ValueError("worker labels is missing!!!")
        if self.posterior_labels is None:
            raise ValueError("posterior_labels is missing!!!")

        posterior_labels = self.posterior_labels.reshape(-1)
        majority_voting_labels = self.majority_voting_labels.reshape(-1)
        model_uncertainty = (posterior_labels != majority_voting_labels).astype(int)
        return model_uncertainty

    def get_solution_uncertainty(self):
        solution_uncertainty_path = os.path.join(config.data_root,
                                                 self.dataname,
                                                 config.info_data,
                                                 config.solution_uncertainty_name)
        if os.path.exists(solution_uncertainty_path):
            res = np.load(solution_uncertainty_path)
            return res
        n = 5
        print("getting solution uncertainty_path")
        posterior_labels_matrix = np.zeros((self.instance_num, n))
        for i in range(n):
            self.model.train(np.random.randint(1,100100,1)[0])
            posterior_labels_matrix[:,i] = self.model.get_posterior_labels()
        posterior_label_dist = np.zeros((self.instance_num, self.label_num))
        posterior_labels_matrix = posterior_labels_matrix.astype(int)
        for i in range(self.instance_num):
            for j in range(n):
                posterior_label_dist[i, posterior_labels_matrix[i,j]] += 1
        entropy = np.zeros(self.instance_num)
        for i in range(self.instance_num):
            prob = posterior_label_dist[i,:]
            prob = prob / prob.sum() + 1e-12
            en = (-prob * np.log(prob) / np.log(2)).sum()
            entropy[i] = en
        np.save(solution_uncertainty_path, entropy)
        return entropy


    # TODO: ground truth as expert validation now
    def add_expert_instance_validation_and_return_validation_list(self, new_validation_list):
        expert_instance_validation = self.expert_instance_validation
        true_labels = self.true_labels
        for i in new_validation_list:
            if expert_instance_validation[i] != -1:
                print("repeated validation!!!!!!!!!!!")
            expert_instance_validation[i] = true_labels[i]
        return expert_instance_validation

    def get_posterior_labels(self):
        return self.posterior_labels

    def get_entropy_by_id(self, i):
        prob = self.posterior_label_dist[i, :]
        prob = prob / prob.sum() + 1e-12
        en = (-prob * np.log(prob) / np.log(2)).sum()
        return round(en, 3)

    @property
    def posterior_label_dist(self):
        return self.model.get_posterior_label_dist()

    @property
    def posterior_labels(self):
        return self.model.get_posterior_labels()

    @property
    def majority_voting_labels(self):
        return self.model.get_majority_voting_result()

    @property
    def expert_instance_validation(self):
        return self.model.get_expert_instance_validation()

    @property
    def instance_similarity(self):
        return self.model.get_simi_matrix()

if __name__ == "__main__":

    exit()
    crowd_data = CrowdData("monkey")
    ci = CrowdInstances(crowd_data.dataname)
    ci.init_from_file(os.path.join(config.data_root, crowd_data.dataname,
                                   config.origin_data, "posterior_labels.mat"))
    ci.init_from_crowd_data(crowd_data)
    ent = ci.get_data_uncertainty()
    index = ent.argsort()[::-1]
    print(ent[index[130]], ci.posterior_labels[index[130], :])
