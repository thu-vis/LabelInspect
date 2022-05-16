import numpy as np
from numpy.ctypeslib import ndpointer
import os
import ctypes
import scipy.io as sio
from scipy.stats import multivariate_normal, entropy

from time import time
from matplotlib import pyplot as plt
from scripts.crowd import CrowdsourcingModel, M3VModel
from scripts.backend import load_static_data, decom_similarity_matrix
from scripts.crowd_data import CrowdData
from scripts.configs import config
from scripts.crowd_worker import CrowdWorkers
from scripts.crowd_instances import CrowdInstances

lib = ctypes.cdll.LoadLibrary(os.path.join(config.scripts_root, config.dll_name, "M3VLib.dll"))
incre_m3v_c = lib.train
incre_m3v_c.restype = ctypes.c_void_p
incre_m3v_c.argtypes = [ctypes.c_void_p,
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

init = lib.init
init.restype = ctypes.c_void_p
init.argtypes =[ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]



class loss_driven_m3v(M3VModel):
    def __init__(self, l=3, c=0.25, n=50, maxIter=50, burnIn=10,
                 v=1, alpha=1, TOL=1e-2, alpha2=35.649, beta2=6,
                 seed=None):
        super(loss_driven_m3v, self).__init__(l=l, c=c, n=n, maxIter=maxIter,
                                              burnIn=burnIn, v=v, alpha=alpha, TOL=TOL,
                                              seed=seed)
        self.alpha2 = alpha2
        self.beta2 = beta2

    def from_crowd_data(self, crowd_data):
        super(loss_driven_m3v, self).from_crowd_data(crowd_data)
        self.instances_similarity = decom_similarity_matrix(
            crowd_data.get_attr(config.similarity_matrix_name),
            crowd_data.get_attr(config.instance_num_name))

        # Notice: initial function is called here in case model call it multiple times in train function
        self.initial()

    def process_spammer_validation(self):
        expert_spammer_validation = self.expert_spammer_validation
        if not hasattr(self,"backend_L"):
            self.backend_L = self.L.copy()
        self.L = self.backend_L.copy()
        all_workers_labels_num = (self.L > 0).sum()
        for worker_id in expert_spammer_validation:
            for idx, label in enumerate(self.L[:, worker_id]):
                if label in expert_spammer_validation[worker_id]:
                    self.L[idx, worker_id] = 0
        print("labels before spammer process:%s and labels after process:%s"
              %( all_workers_labels_num, (self.L > 0).sum())
              )
        self.process_data()

    def initial(self):
        super(loss_driven_m3v, self).initial()
        self.expert_instance_validation = np.zeros(self.Ntask)
        self.max_simi_value_vector = np.zeros(self.Ntask)
        self.max_simi_index_vector = np.zeros(self.Ntask)

    def getsi(self, eta, i, k):
        mm = float('-inf')
        si = 0
        for kk in range(self.Ndom):
            # if kk != k
            if abs(kk - k) > self.eps:
                dx = self.X[i, kk, :]
                tmp = np.dot(eta, dx.T)
                if tmp > mm:
                    si = kk
                    mm = tmp
        return si + 1

    def _train(self, ilm, eta, phi, Y, expert_instance_validation, seed=1):

        K = self.Ndom
        L = self.L
        X = self.X
        c = self.c
        l = self.l
        S = self.S
        A0 = self.A0
        B0 = self.B0
        probd0 = self.probd0
        instances_similarity = self.instances_similarity
        alpha2 = self.alpha2
        beta2 = self.beta2

        # variables needed to be returned
        ans_soft_labels = np.zeros(self.ans_soft_labels.shape)
        phic = np.zeros(self.phic.shape)
        etak = np.zeros(self.etak.shape)
        etak_count = 0

        for i in range(self.Ntask):
            S[i] = self.getsi(eta, i, Y[i] - 1)

        start_t = time()
        for iter in range(self.maxIter):
            A = A0.copy()
            B = B0.copy()
            for i in range(self.Ntask):
                dx = X[i, Y[i] - 1, :] - X[i, S[i] - 1, :]
                dx = dx.reshape(1, -1)
                B = B + np.dot(dx.T, dx) * ilm[i] * c * c
                A = A + (l * ilm[i] + 1 / float(c)) * dx * c * c

            eta = np.random.multivariate_normal(np.array(np.dot(A, B.I)).reshape(-1), B.I).reshape(1, -1) \
                  + self.eps

            # -- copy is necessary because probd0 cannot be change --
            probd = probd0.copy()
            for i in range(self.Ntask):
                for j in self.NeibTask[i]:
                    probd[L[i, j] - 1, Y[i] - 1, j] = probd[L[i, j] - 1, Y[i] - 1, j] + 1

            for i in range(K):
                for j in range(self.Nwork):
                    phi[:, i, j] = np.random.dirichlet(probd[:, i, j], 1) + self.eps
                    # print( "phi", phi [:,i,j])
                    # print( "probd", probd[:,i,j])

            for i in range(self.Ntask):
                dx = X[i, Y[i] - 1, :] - X[i, S[i] - 1, :]
                aczetai = abs(c * (l - np.dot(eta, dx.T)) + self.eps)
                ilm[i] = np.random.wald(np.linalg.inv(aczetai), 1)

            randomIdx = np.array(range(self.Ntask))
            np.random.shuffle(randomIdx)
            for i in randomIdx:
                logprob = np.zeros((K, 1))
                for k in range(K):
                    # if ilm[i] != 0
                    if abs(ilm[i]) > self.eps:
                        dx = X[i, k, :] - X[i, self.getsi(eta, i, k) - 1, :]
                        logprob[k] = -0.5 * ilm[i] * (1.0 / ilm[i] + c * (l - np.dot(eta, dx.T))) ** 2
                    for j in self.NeibTask[i]:
                        # TODO:it is right?? ??
                        logprob[k] = logprob[k] + np.log(phi[L[i, j] - 1, k, j] + self.eps)

                prob = np.exp(logprob - logprob.max())
                prob = prob / prob.sum()
                prob_sample = np.random.multinomial(1, prob.reshape(-1))
                prob_nnz = np.nonzero(prob_sample > 0)
                class_k = int(prob_nnz[0]) + 1

                # instance propagation
                # if instance not be validated
                class_kind_num = len(set(expert_instance_validation[expert_instance_validation!=0]))
                if (expert_instance_validation[i] == 0):
                    labeled_idx = np.array(range(self.Ntask))[expert_instance_validation != 0]
                    if len(labeled_idx) > 0:
                        max_simi_index = instances_similarity[i, labeled_idx].argmax()
                        max_simi_index = labeled_idx[max_simi_index]
                        max_simi_value = instances_similarity[i, max_simi_index]
                        if max_simi_value < 0.8 and class_kind_num < K:
                            max_simi_value = 0
                        diff_vector = (np.array(range(self.Ndom)) + 1) != \
                                      np.array(expert_instance_validation[max_simi_index]).repeat(repeats=self.Ndom)
                        loss_vector = np.abs(diff_vector)
                        prob = prob.reshape(-1) * np.exp(-beta2 * max_simi_value * loss_vector)
                        prob = prob / prob.sum()
                        prob_sample = np.random.multinomial(1, prob.reshape(-1))
                        prob_nnz = np.nonzero(prob_sample > 0)
                        class_k = int(prob_nnz[0]) + 1
                else:
                    diff_vector = (np.array(range(self.Ndom)) + 1) != \
                                  np.array(expert_instance_validation[i]).repeat(repeats=self.Ndom)
                    loss_vector = np.abs(diff_vector)
                    prob = (prob + self.eps).reshape(-1) * np.exp(-alpha2 * loss_vector)
                    prob = prob / prob.sum()
                    prob_sample = np.random.multinomial(1, prob.reshape(-1))
                    prob_nnz = np.nonzero(prob_sample > 0)
                    class_k = int(prob_nnz[0]) + 1
                Y[i] = class_k
                S[i] = self.getsi(eta, i, Y[i] - 1)

            if iter > self.burnIn:
                for i in range(self.Ntask):
                    ans_soft_labels[i, Y[i] - 1] = ans_soft_labels[i, Y[i] - 1] + 1
                phic = phic + phi
                etak[etak_count, :] = eta[0, :]
                etak_count += 1

            if self.verbose > 0 and iter > self.burnIn:
                ans_soft_labelst = ans_soft_labels / (etak_count)
                error_rate, soft_error_rate, error_L1, erroe_L2 = self.cal_error_using_soft_label(ans_soft_labelst,
                                                                                                  self.true_labels)
                end_t = time()
                print("iter:%s, error_rate:%s, totaltime:%s" % (iter, error_rate, end_t - start_t))
                # print( self.ans_soft_labels )

        ans_soft_labelst = ans_soft_labels / (etak_count)
        error_rate, soft_error_rate, error_L1, error_L2 = \
            self.cal_error_using_soft_label(ans_soft_labelst, self.true_labels)

        end_t = time()
        print("Final: iter:%s, error_rate:%s, totaltime:%s" % (iter, error_rate, end_t - start_t))
        return phic, etak_count, ans_soft_labels, Y, error_rate

    def _train_c_buffer(self,ilm, eta, phi, Y, expert_instance_validation, worker_prior=None, simi=None, seed=None):
        # np.random.seed(24)
        if seed is None:
            seed = (int(np.random.rand()*100000))
            seed = 4491775  # 123
        K = self.Ndom
        L = self.L
        instances_similarity = self.instances_similarity
        phi = np.zeros((K, K, self.Nwork))
        eta = np.zeros((1))
        posterior_distribution = np.zeros((self.Ntask, self.Ndom))
        if worker_prior is None:
            worker_prior = np.zeros(self.Nwork)

        if simi is None:
            modified_simi = self.instances_similarity.copy()
        else:
            modified_simi = simi.copy()

        # modified_simi = np.zeros((self.instances_similarity.shape))
        # for i in range(modified_simi.shape[0]):
        #     simi_vect = self.instances_similarity[:,i]
        #     top5 = simi_vect.argsort()[::-1][:60]
        #     # modified_simi[top5,i] = self.instances_similarity[top5,i]
        #     modified_simi[i,top5] = self.instances_similarity[top5,i]
        # modified_simi = modified_simi.transpose()
        start_t = time()
        print("begin training with dll!, seed:%s"%(seed))
        self.seed = seed
        if 1:
        # if not hasattr(self,"_data_pointer"):
            print("data preprocessing")
            self._data_pointer = init(seed, self.Ntask, K, self.Nwork, 0, self.alpha2, self.beta2,
                                      L.astype(np.float64),
                                      self.true_labels.astype(np.float64),
                                      modified_simi.astype(np.float64))
        # incre_m3v_c(seed, self.Ntask, K, self.Nwork,
        #             L.astype(np.float64),
        #             self.true_labels.astype(np.float64),
        #             expert_instance_validation.astype(np.float64),
        #             instances_similarity.astype(np.float64),
        #             phi, eta, posterior_distribution)
        incre_m3v_c(self._data_pointer, expert_instance_validation.astype(np.float64),
                    worker_prior,
                    phi, eta, posterior_distribution)
        error_rate, soft_error_rate, error_L1, error_L2 = \
            self.cal_error_using_soft_label(posterior_distribution, self.true_labels)
        end_t = time()
        posterior_labels = posterior_distribution.argmax(axis=1)
        true_labels = np.array(self.true_labels - 1)
        s = ""
        label_list = list(set(true_labels.tolist()))
        for i in range(len(label_list)//2):
            selected_class_list = [i * 2, i * 2 + 1]
            class_indicator = [True if c in selected_class_list else False for c in true_labels]
            selected_posterior_labels = posterior_labels[class_indicator]
            selected_true_labels = true_labels[class_indicator]
            selected_sum = sum(class_indicator)
            selected_wrong = sum(selected_posterior_labels != selected_true_labels)
            s = s + str(selected_wrong) + "/" + str(selected_sum) + " "
        print(s)
        print("Final: iter:%s, error_rate:%s, acc_rate:%s, totaltime:%s" % (49, error_rate, 1-error_rate , end_t - start_t))
        return phi, eta, posterior_distribution, Y, error_rate

    def _train_c(self, ilm, eta, phi, Y, expert_instance_validation, seed=131):
        # seed = seed
        K = self.Ndom
        L = self.L
        instances_similarity = self.instances_similarity
        phi = np.zeros((K, K, self.Nwork))
        eta = np.zeros((1))
        posterior_distribution = np.zeros((self.Ntask, self.Ndom))

        start_t = time()
        print("begin training with dll!")
        incre_m3v_c(seed, self.Ntask, K, self.Nwork,
                    L.astype(np.float64),
                    self.true_labels.astype(np.float64),
                    expert_instance_validation.astype(np.float64),
                    instances_similarity.astype(np.float64),
                    phi, eta, posterior_distribution)
        error_rate, soft_error_rate, error_L1, error_L2 = \
            self.cal_error_using_soft_label(posterior_distribution, self.true_labels)

        end_t = time()

        print("Final: iter:%s, error_rate:%s, totaltime:%s" % (49, error_rate, end_t - start_t))
        return phi, eta, posterior_distribution, Y, error_rate


    def train(self, seed=None, processed_L=None, worker_prior=None, simi=None):
        # self.process_spammer_validation()
        if processed_L is None:
            try:
                self.process_spammer_validation()
            except:
                print("skip spammer validation process")
        else:
            self.L = processed_L
            self.process_data()
        self.clean_dynamic_variable()
        self.initByMajorityVoting()

        # get majority voting error rate
        correct_num = 0
        for i in range(self.Ntask):
            if self.Y[i] == self.true_labels[i]:
                correct_num = correct_num + 1
        mv_error_rate = 1 - float(correct_num) / float(self.Ntask)
        print("majority voting, error_rate:%s" % (mv_error_rate))

        # get validation percent
        instance_validation_num = sum(self.expert_instance_validation > 0)
        print("validated instances percent: %s"
              % (float(instance_validation_num) / self.Ntask))

        ilm = self.ilm
        eta = self.eta
        phi = self.phi
        Y = self.Y
        if worker_prior is None:
            worker_prior = np.zeros(self.Nwork)


        # additional variables comparing to original m3v
        expert_instance_validation = self.expert_instance_validation

        phic, etak_count, ans_soft_labels, Y, error_rate = \
            self._train_c_buffer(ilm=ilm, eta=eta, phi=phi, Y=Y,
                        expert_instance_validation=expert_instance_validation,
                        worker_prior=worker_prior, simi=simi,
                        seed=seed)
        self.phi = phic / float(etak_count)
        self.ans_soft_labelst = ans_soft_labels
        self.Y = Y
        self.error_rate = error_rate
        self.set_trained_flag()

    def set_trained_flag(self):
        self.trained = 1

    # def influence_given_one_label(self, instance_id, label):
    #     expert_instance_validation = self._get_expert_instance_validation()
    #     expert_instance_validation[instance_id] = label
    #
    #     self.initByMajorityVoting()
    #     ilm = np.ones(self.ilm.shape)
    #     eta = np.dot(self.A0, self.B0.I)
    #     phi = np.zeros(self.phi.shape)
    #     Y = self.Y
    #
    #     phic, etak_count, influence_ans_soft_labels, Y, error_rate = \
    #         self._train_c(ilm=ilm, eta=eta, phi=phi, Y=Y,
    #                     expert_instance_validation=expert_instance_validation,
    #                     )
    #     ans_soft_labels = self._get_posterior_label_dist()
    #
    #     # using KL divergence as influence value
    #     influence = np.zeros(influence_ans_soft_labels.shape[0])
    #     for i in range(influence_ans_soft_labels.shape[0]):
    #         influence[i] = entropy(pk=ans_soft_labels[i,:] + self.eps,
    #                                qk=influence_ans_soft_labels[i,:] + self.eps)
    #     return influence
    #
    # def top_influence_of_instance(self, instance_id):
    #     posterior_dist = self.ans_soft_labelst[instance_id]
    #     influence = None
    #     for i in range(posterior_dist.shape[0]):
    #         _influ = self.influence_given_one_label(instance_id,i+1)
    #         if influence is None:
    #             influence = _influ * posterior_dist[i]
    #         else:
    #             influence = influence + _influ * posterior_dist[i]
    #     order = influence.argsort()[::-1]
    #     res = {}
    #     for i in order[:20]:
    #         res[str(i)] = influence[i]
    #     return res
    #
    # def get_flow_map(self, instance_list):
    #     res = {}
    #     for instance_id in instance_list:
    #         res[str(instance_id)] = self.top_influence_of_instance(instance_id)
    #     return res

    def get_confusion_matrices(self):
        if self.trained == 0:
            print("training when trying to get confusion matrix")
            self.train()
        return self.phi

    def get_posterior_labels(self):
        if self.trained == 0:
            print("training when trying to get posterior labels")
            self.train()
        return self.ans_soft_labelst.argmax(axis=1)

    def _get_posterior_label_dist(self):
        if self.trained == 0:
            print("training when trying to get posterior label dists")
            self.train()
        return self.ans_soft_labelst

    def get_posterior_label_dist(self):
        return self._get_posterior_label_dist()

    def _get_expert_instance_validation(self):
        return self.expert_instance_validation.astype(int)

    def get_expert_instance_validation(self):
        return self.expert_instance_validation.astype(int) - 1

    def save_confusion_matrices(self, filename):
        if self.trained == 0:
            self.train()
        mat = {"confusion_matrices": self.phi}
        sio.savemat(filename, mat)
        print("confusion matrices have been saved in %s" % (filename))

    def save_posterior_labels(self, filename):
        if self.trained == 0:
            self.train()
        mat = {"posterior_labels": self.ans_soft_labelst}
        sio.savemat(filename, mat)
        print("posterior labels have been saved in %s" % (filename))

    def get_simi_matrix(self):
        return self.instances_similarity

class incre_m3v(loss_driven_m3v):
    def __init__(self, l=3, c=0.25, n=50, maxIter=50, burnIn=10,
                 v=1, alpha=1, TOL=1e-2, alpha2=35.649, beta2=6,
                 seed=None):
        super(incre_m3v, self).__init__(l=l, c=c, n=n, maxIter=maxIter,
                                        burnIn=burnIn, v=v, alpha=alpha, TOL=TOL,
                                        alpha2=alpha2, beta2=beta2,seed=seed)
        self._propagation_iter = -1
        self._pre_posterior_dist = None
        self._pre_phi = None
        self._pre_Y = None

        self._expert_instance_validation_dict = None
        self._expert_spammer_validation_dict = None

        self.flag_get_new_instance_validation = False
        self.flag_get_new_spammer_validatioin = False

    def get_influence(self):
        if self._propagation_iter < 1:
            print("invalid propagation!")
            return {}
        pre_posterior_dist = self._pre_posterior_dist
        present_posterior_dist = self.get_posterior_label_dist()
        instance_num = pre_posterior_dist.shape[0]
        validation_num = len(self._expert_instance_validation_dict)
        influence = np.zeros(instance_num)
        validation_influence = {}
        for instance_id in self._expert_instance_validation_dict:
            validation_influence[instance_id] = []
        labeled_idx = np.array(range(self.Ntask))[self.expert_instance_validation != 0]
        for i in range(instance_num):
            influence[i] = entropy(pk=pre_posterior_dist[i,:] + self.eps,
                                   qk=present_posterior_dist[i,:] + self.eps)
            max_simi_index = self.instances_similarity[i, labeled_idx].argmax()
            max_simi_index = str(labeled_idx[max_simi_index])
            if max_simi_index in validation_influence and influence[i] > self.eps:
                validation_influence[max_simi_index].append({
                    "id":i,
                    "weight": influence[i]
                })
        return validation_influence

    def train(self, seed=None, processed_L=None, worker_prior=None, simi=None):
        if self.flag_get_new_instance_validation or self.flag_get_new_spammer_validatioin:
            self._propagation_iter = self._propagation_iter + 1
            self._pre_posterior_dist = self.get_posterior_label_dist()
            self._pre_phi = self.get_confusion_matrices()
            self.flag_get_new_instance_validation = False
            self.flag_get_new_spammer_validatioin = False
        if self._propagation_iter < 0:
            self._propagation_iter = self._propagation_iter + 1

        # preprocessing spammer validation

        super(incre_m3v, self).train(seed=seed, processed_L=processed_L,
                                     worker_prior=worker_prior, simi=simi)
        print("now finish %s-th propatation" %(self._propagation_iter) )

    def update_validation_list(self, expert_instance_validation):
        self.expert_instance_validation = expert_instance_validation.astype(int) + 1

    def update_instance_validation(self, expert_instance_validation_dict, temp_buffer):
        flag_get_new_instance_validation = False
        if len(temp_buffer) > 0:
            flag_get_new_instance_validation = True
            self._expert_instance_validation_dict = temp_buffer
            print("incre-m3v get new instance validatioin dict!!!")
        self.flag_get_new_instance_validation = self.flag_get_new_instance_validation or\
                                                flag_get_new_instance_validation
        print(self.flag_get_new_instance_validation)
        expert_instance_validation = np.zeros(self.Ntask)
        for instance_id in expert_instance_validation_dict:
            # adding 1 here because category number start from 1 rather than 0
            expert_instance_validation[int(instance_id)] = \
                expert_instance_validation_dict[instance_id] + 1
        self.expert_instance_validation = expert_instance_validation

    def update_spammer_validation(self, expert_spammer_validation_dict):
        flag_get_new_spammer_validation = False
        if len(expert_spammer_validation_dict) > 0:
            self.flag_get_new_spammer_validatioin = True
            print("incre-m3v get new spammer validatioin dict!!!")
        expert_spammer_validation = {}
        for worker_id in expert_spammer_validation_dict:
            confusing_classes_list = expert_spammer_validation_dict[worker_id]
            expert_spammer_validation[int(worker_id)] = []
            for _class in confusing_classes_list:
                expert_spammer_validation[int(worker_id)].append(_class + 1)
        self.expert_spammer_validation = expert_spammer_validation

    def get_pre_posterior(self):
        if self._propagation_iter >= 1:
            pre_posterior = self._pre_posterior_dist.argmax(axis=1).tolist()
        else:
            pre_posterior = {}
        return pre_posterior



if __name__ == "__main__":
    None
