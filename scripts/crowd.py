import numpy as np
import os
import scipy.io as sio
from time import time
from matplotlib import pyplot as plt
from scripts.backend import load_static_data, load_manifest_data
from concurrent.futures import ProcessPoolExecutor

class CrowdsourcingModel(object):
    def __init__(self, l=3, c=0.25, n=50, maxIter=50, burnIn=10, v=1, alpha=1, TOL=1e-2, seed=None):
        self.l = l
        self.c = c
        self.n = n
        self.maxIter = maxIter
        self.burnIn = burnIn
        self.v = v
        self.alpha = alpha
        self.eps = 2.e-6
        self.verbose = 0
        self.trained = 0
        self.TOL = TOL
        self.seed = seed

    def from_crowd_data(self, crowd_data):
        instance_num = crowd_data.get_attr("InstanceTotalNum")
        self.L = crowd_data.get_attr("WorkerLabels")
        self.true_labels = crowd_data.get_attr("true_labels")
        self.L = np.array(self.L).reshape(instance_num, -1) + 1
        self.backend_L = self.L.copy()
        self.true_labels = np.array(self.true_labels).reshape(-1) + 1
        self.origin_labels_num = (self.L > 0).sum()
        self.process_data()

    def loadData(self, filename):
        mat = sio.loadmat(filename)
        self.L = mat['L']
        try:
            self.L = self.L.toarray()
        except:
            self.L = self.L
        self.L = self.L.astype(np.int)
        self.backend_L = self.L.copy()
        self.true_labels = mat['true_labels'].reshape(-1).astype(np.int)
        self.process_data()

    def eliminate_workers_without_any_labels(self):
        # pre-process L to eliminate workers without anylabel
        labels = self.L.sum(axis=0)
        self.L = self.L[:, labels > 0]
        self.process_data()

    def process_data(self):
        # get info from L for further convenience
        self.Ntask, self.Nwork = self.L.shape
        self.Ndom = len(set([i for i in self.true_labels.reshape(-1)]))
        self.LabelDomain = np.unique(self.L[self.L != 0])
        self.Ndom = len(self.LabelDomain)
        # print( self.LabelDomain )
        # exit(0)
        self.NeibTask = []
        for i in range(self.Ntask):
            tmp = [nt for nt in range(self.Nwork) if self.L[i, nt] > 0]
            self.NeibTask.append(tmp)
        self.NeibWork = []
        for j in range(self.Nwork):
            tmp = [nw for nw in range(self.Ntask) if self.L[nw, j] > 0]
            self.NeibWork.append(tmp)
        self.LabelTask = []
        for i in range(self.Ntask):
            tmp = [self.L[i, nt] for nt in self.NeibTask[i]]
            self.LabelTask.append(tmp)
        self.LabelWork = []
        for j in range(self.Nwork):
            tmp = [self.L[nw, j] for nw in self.NeibWork[j]]
            self.LabelWork.append(tmp)

    def cal_error_using_soft_label(self, mu, true_labels):
        '''
            to avoid ties, we take uniform probability over all calsses that
        maxmumize mu(classes, workers)
            1. average in case of ties
            2. Ignore when ture true_labels are NaN (missing)
        :param mu:
        :param true_labels:
        :return:
        '''

        index = (true_labels > 0)
        mu = mu[index, :]
        true_labels = true_labels[index]
        soft_label = mu / mu.sum(axis=1).reshape(-1, 1).repeat(axis=1, repeats=self.Ndom)
        mu = (mu.max(axis=1).reshape(-1, 1).repeat(axis=1, repeats=self.Ndom) == mu)
        mu = mu.astype(float)
        self.posterior_labels = mu.argmax(axis=1) + 1
        mu = mu / mu.sum(axis=1).reshape(-1, 1).repeat(axis=1, repeats=self.Ndom)
        tmp1 = np.array(range(1, 1 + self.Ndom)).reshape(1, -1).repeat(axis=0, repeats=true_labels.shape[0])
        tmpTrue = true_labels.reshape(-1, 1).repeat(axis=1, repeats=self.Ndom)
        error_rate = ((tmpTrue != tmp1) * mu).sum(axis=1).mean()
        soft_error_rate = ((tmpTrue != tmp1) * soft_label).sum(axis=1).mean()
        return error_rate, soft_error_rate, -1, -1


class M3VModel(CrowdsourcingModel):
    def initial(self):
        self.init_static_info()
        self.clean_dynamic_variable()

    def init_static_info(self):
        K = self.Ndom
        self.X = np.zeros((self.Ntask, K, self.Nwork))
        self.majority_voting_result = np.zeros(self.Ntask)
        for i in range(self.Ntask):
            for j in self.NeibTask[i]:
                self.X[i, self.L[i, j] - 1, j] = 1

    def clean_dynamic_variable(self):
        K = self.Ndom
        Ndom = K
        self.A0 = np.zeros((1, self.Nwork))
        self.B0 = np.mat(np.diag([1 / float(self.v) for i in range(self.Nwork)]))
        self.probd0 = np.ones((Ndom, K, self.Nwork))  # * 0.01
        self.eta = np.dot(self.A0, self.B0.I)
        self.phi = np.zeros((Ndom, K, self.Nwork))
        self.ilm = np.ones((self.Ntask, 1))
        self.ilmc = np.zeros((self.Ntask, 1))
        self.Y = np.zeros((self.Ntask, 1)).astype(int)
        self.S = np.zeros((self.Ntask, 1)).astype(int)
        self.ans_soft_labels = np.zeros((self.Ntask, K))
        self.phic = np.zeros((Ndom, K, self.Nwork))
        self.etak = np.zeros((self.maxIter - self.burnIn, self.Nwork))
        self.etak_count = 0

    def initByMajorityVoting(self):
        if not hasattr(self, 'Y'):
            self.Y = np.zeros((self.Ntask, 1))
        for i in range(self.Ntask):
            ct = np.zeros((self.Ndom, 1))
            for j in range(len(ct)):
                ct[j] = sum(self.L[i, :] == j + 1)
            self.Y[i] = ct.argmax() + 1
        self.majority_voting_result = self.Y.copy()
        None

    def get_majority_voting_result(self):
        return self.majority_voting_result - 1


class mmcrowd(M3VModel):
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

    def train(self):
        self.initial()
        self.initByMajorityVoting()
        correct_num = 0
        for i in range(self.Ntask):
            if self.Y[i] == self.true_labels[i]:
                correct_num = correct_num + 1
        mv_error_rate = 1 - float(correct_num) / float(self.Ntask)
        print("majority voting, error_rate:%s" % (mv_error_rate))

        K = self.Ndom
        Ndom = self.Ndom
        L = self.L
        ilm = self.ilm
        eta = self.eta
        phi = self.phi
        X = self.X
        Y = self.Y
        c = self.c
        l = self.l
        S = self.S
        for i in range(self.Ntask):
            S[i] = self.getsi(eta, i, Y[i] - 1)

        start_t = time()
        for iter in range(self.maxIter):
            A = self.A0.copy()
            B = self.B0.copy()
            for i in range(self.Ntask):
                dx = X[i, Y[i] - 1, :] - X[i, S[i] - 1, :]
                dx = dx.reshape(1, -1)
                B = B + np.dot(dx.T, dx) * ilm[i] * c * c
                A = A + (l * ilm[i] + 1 / float(c)) * dx * c * c

            eta = np.random.multivariate_normal(np.array(np.dot(A, B.I)).reshape(-1), B.I).reshape(1, -1) \
                  + self.eps

            # -- copy is necessary because probd0 cannot be change --
            probd = self.probd0.copy()
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
                Y[i] = int(prob_nnz[0]) + 1
                S[i] = self.getsi(eta, i, Y[i] - 1)

            if iter > self.burnIn:
                for i in range(self.Ntask):
                    self.ans_soft_labels[i, Y[i] - 1] = self.ans_soft_labels[i, Y[i] - 1] + 1
                self.phic = self.phic + phi
                self.etak[self.etak_count, :] = eta[0, :]
                self.etak_count += 1

            if self.verbose > 0 and iter > self.burnIn:
                ans_soft_labelst = self.ans_soft_labels / (self.etak_count)
                error_rate, soft_error_rate, error_L1, erroe_L2 = self.cal_error_using_soft_label(ans_soft_labelst,
                                                                                                  self.true_labels)
                end_t = time()
                print("iter:%s, error_rate:%s, totaltime:%s" % (iter, error_rate, end_t - start_t))
                # print( self.ans_soft_labels )

        ans_soft_labelst = self.ans_soft_labels / (self.etak_count)
        error_rate, soft_error_rate, error_L1, erroe_L2 = self.cal_error_using_soft_label(ans_soft_labelst,
                                                                                          self.true_labels)
        self.phi = self.phic / float(self.etak_count)
        self.ans_soft_labelst = self.ans_soft_labels / (self.etak_count)
        self.trained = 1

        end_t = time()
        print("Final: iter:%s, error_rate:%s, totaltime:%s" % (iter, error_rate, end_t - start_t))

    def get_confusion_matrices(self):
        if self.trained == 0:
            self.train()
        return self.phi

    def get_posterior_labels(self):
        if self.trained == 0:
            self.train()
        return self.ans_soft_labelst.argmax(axis=1)

    def get_posterior_label_dist(self):
        if self.trained == 0:
            self.train()
        return self.ans_soft_labelst

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


if __name__ == "__main__":
    None
    import scipy.stats
    scipy.stats.norm(0,1)