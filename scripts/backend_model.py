import numpy as np
import os
import scipy.io as sio
from scipy.stats import multivariate_normal, entropy
import json
from sklearn import manifold
import warnings
import pickle
from time import time


from scripts.backend import load_static_data, decom_similarity_matrix
from scripts.crowd_data import CrowdData
from scripts.crowd import mmcrowd
from scripts.configs import config
from scripts.crowd_worker import CrowdWorkers
from scripts.crowd_instances import CrowdInstances
from scripts.crowd_tSNE import CrowdTSNE
# from scripts.loss_driven_m3v import incre_m3v
from scripts.loss_driven_m3v_c import incre_m3v



def get_tsne_from_similarity_matrix(similarity_matrix):
    print("now begin get tsne result from similarity matrix.")
    tsne = manifold.TSNE(n_components=2, metric="precomputed",random_state=15)
    dist = 1 - np.array(similarity_matrix)
    if dist.min() > 1e-3:
        warnings.warn("the max similarity of the matrix is larger than 1+1e-3. "
                      "Please check your input if there are some bugs.")
    dist = dist - dist.min()
    plat_coordinate = tsne.fit_transform(dist)
    print("tsne result got!")
    return plat_coordinate


def get_knn(n_neighbors, simi):
    nn = np.ones((simi.shape[0], n_neighbors)) * -1

    for i in range(simi.shape[0]):
        simi[i, i] = 1
        nn[i, :] = simi[i, :].argsort()[:n_neighbors]
    return nn.astype(int)


class ModelMaintainer(object):
    def __init__(self, dataname, beta2=6.0, simi_threshold=0.8):
        if dataname is None:
            return
        self.dataname = dataname
        t1 = time()
        self.info_manifest_path = os.path.join(config.data_root,
                                               dataname,
                                               config.info_data,
                                               config.info_manifest_name)
        self.info_static_path = os.path.join(config.data_root,
                                             dataname,
                                             config.info_data,
                                             config.info_static_name)
        self.info_dynamic_path = os.path.join(config.data_root,
                                              dataname,
                                              config.info_data,
                                              config.info_dynamic_name)
        self.model_path = os.path.join(config.data_root,
                                       dataname,
                                       config.info_data,
                                       config.init_model_name)
        self.debug_model_path = os.path.join(config.data_root,
                                       self.dataname,
                                       config.info_data,
                                       config.debug_model_name)
        self.crowd_data = CrowdData(dataname, rectify=True)
        t_model = time()
        self.model = incre_m3v(seed=2018, beta2=beta2)
        # np.random.seed(2018)
        t_data = time()
        self.model.from_crowd_data(self.crowd_data)
        t_component = time()
        self.crowd_workers = CrowdWorkers(self.crowd_data)
        self.crowd_instances = CrowdInstances(self.crowd_data)
        self.crowd_tsne = CrowdTSNE(self.crowd_data)
        t_connect = time()
        self.crowd_instances.connect_to_model(self.model)
        self.crowd_workers.connect_to_model(self.model)
        self.crowd_tsne.connect_to_model(self.model)

        # variables storing previous status for influence
        self._pre_posterior_dist = None
        self._pre_propagation_iter = 0
        print("init model total time cost: %s" %(time() - t1))
        print("from model init time cost: %s"%(time() - t_model))
        print("from data init time cost: %s"%(time() - t_data))
        print("from component init time cost: %s"%(time() - t_component))
        print("from connect time cost: %s"%(time() - t_connect))

    def adopt_instance_validation(self,expert_instance_validation, temp_buffer):
        '''
        adopting instance validation but without updating the model immediately
        :param expert_instance_validation:
        :return: None
        '''
        self.model.update_instance_validation(expert_instance_validation, temp_buffer)

    def adopt_spammer_validation(self, expert_spammer_validation):
        '''
        adopting spammer validation but without updating the model immediately
        :param expert_spammer_validation:
        :return: None
        '''
        self.model.update_spammer_validation(expert_spammer_validation)

    def _propagation(self, seed=None, processed_L = None, worker_prior = None, simi=None):
        '''
        updating the model immediately
        :return: None
        '''
        self.model.train(seed, processed_L=processed_L, worker_prior=worker_prior, simi=simi)

    def get_flow_map(self, instance_list):
        affinity_matrix = self.crowd_instances.get_affinity_matrix()
        instance_num = self.crowd_data.get_attr(config.instance_num_name)
        worker_num = self.crowd_data.get_attr(config.worker_num_name)
        instance_affinity_matrix = affinity_matrix[:instance_num][:instance_num]
        worker_affinity_matrix = affinity_matrix[instance_num:][instance_num:]

        # TODO:

    def _get_instance_influence(self, selected_list, instance_list):
        spammer_score = self.crowd_workers.\
            get_spammer_scores_according_to_selected_list(selected_list)
        worker_similarity = self.crowd_workers.similarity(selected_list,"RA")
        instance_similarity = self.model.instances_similarity
        posterior_labels = self.model.get_posterior_labels()
        selected_instance_indicator = [True if c in selected_list else False for c in posterior_labels]
        unselected_instance_indicator = [False if c in selected_list else True for c in posterior_labels]
        instance_num = instance_similarity.shape[0]
        instance_ranking_score, spammer_ranking = self.crowd_instances.\
            random_walk_score(worker_similarity, spammer_score)
        instance_ranking_score[unselected_instance_indicator] = 0
        instance_ranking = instance_ranking_score.argsort()[::-1]
        validation_influence = {}
        influence_num = min(len(instance_list),1)
        print("instance_list",instance_list)
        for i in range(influence_num):
            id = instance_list[::-1][i]
            simi = instance_similarity[id,:].copy()
            simi[id] = 0
            simi[instance_ranking[:20]] = simi[instance_ranking[:20]] + 100
            # influence_list = np.array(range(instance_num))[simi > 0.8]
            influence_list = simi.argsort()[::-1][:5]
            validation_influence[str(id)] = []
            for j in influence_list:
                validation_influence[str(id)].append({
                    "id":int(j),
                    "weight": (simi[j]-100) * 10
                })
        return validation_influence

    def _get_worker_influence(self, selected_list, worker_list):
        K = 9
        spammer_score = self.crowd_workers.\
            get_spammer_scores_according_to_selected_list(selected_list)
        worker_similarity = self.crowd_workers.similarity(selected_list, "RA")
        instance_similarity = self.model.instances_similarity
        instance_num = instance_similarity.shape[0]
        for i in worker_list:
            spammer_score[i] = 10
            simi = worker_similarity[i,:]
            spammer_score[simi.argsort()[::-1][:2]] = 5
        instance_ranking_score, spammer_ranking = self.crowd_instances.\
            random_walk_score(worker_similarity, spammer_score)
        validation_influence = {}

        spammer_list = spammer_ranking.argsort()[::-1][:K].tolist()
        posterior_labels = self.model.get_posterior_labels()
        selected_instance_indicator = [True if c in selected_list else False for c in posterior_labels]
        unselected_instance_indicator = [False if c in selected_list else True for c in posterior_labels]
        instance_ranking_score[unselected_instance_indicator] = 0
        instance_list = instance_ranking_score.argsort()[::-1].tolist()

        influence_num = min(len(worker_list),1)
        # print("type %s and worker_list %s"%(selected_list[0]//2, str(worker_list)))
        for i in range(influence_num):
            id = worker_list[::-1][i]
            simi = worker_similarity[id,:]
            simi[id] = 0
            # influence_list = np.array(range(instance_num))[simi > 0.8]
            influence_list = simi.argsort()[::-1][:K]
            validation_influence[str(id)] = []
            for j in influence_list:
                validation_influence[str(id)].append({
                    "id":int(j),
                    "weight": simi[j] * 10
                })

        # post-processing
        res = []
        for id in validation_influence:
            single_influence = []
            for inf in validation_influence[id]:
                single_influence.append(inf)
            res.append({
                "id": id,
                "influence": single_influence
            })
        return res, spammer_list, instance_list


    def get_dynamic_info(self, selected_list):
        '''
        get necessary information from model result
        :return:
        '''
        K = 9
        mat = {}
        mat[config.posterior_distribution_name] = self.crowd_instances.posterior_label_dist.reshape(-1).tolist()
        mat[config.posterior_labels_name] = self.crowd_instances.posterior_labels.reshape(-1).tolist()
        mat[config.uncertainty_name] = self.crowd_instances.get_uncertainty().tolist()
        spammer_score = self.crowd_workers.\
            get_spammer_scores_according_to_selected_list(selected_list)
        worker_similarity = self.crowd_workers.similarity(selected_list,"RA")
        instance_ranking, spammer_ranking = self.crowd_instances.\
            random_walk_score(worker_similarity, spammer_score)
        mat[config.spammer_list_name] = spammer_ranking.argsort()[::-1][:K].tolist()
        print(spammer_ranking.argsort()[::-1][:50])
        posterior_labels = self.model.get_posterior_labels()
        selected_instance_indicator = [True if c in selected_list else False for c in posterior_labels]
        unselected_instance_indicator = [False if c in selected_list else True for c in posterior_labels]
        print("dynamic info - selected num: %s, unselected num: %s" % (
            sum(selected_instance_indicator), sum(unselected_instance_indicator)))
        instance_ranking[unselected_instance_indicator] = 0
        mat[config.instance_list_name] = instance_ranking.argsort()[::-1].tolist()
        mat[config.spammer_ranking_name] = spammer_ranking.tolist()
        mat[config.instance_ranking_name] = instance_ranking.tolist()
        mat[config.worker_accuracy_name] = self.crowd_workers\
            .get_posterior_accuracy_according_to_selected_list(selected_list).tolist()
        mat[config.worker_changed_list_name] = self.crowd_workers\
            .get_worker_changed_list_according_to_selected_list(selected_list,
                                                                spammer_score,
                                                                np.array(mat[config.worker_accuracy_name]))
        mat[config.spammer_score_name] = spammer_score.tolist()
        mat[config.worker_prediction_matrix_name] = \
            self.crowd_workers.get_estimated_global_confusion_matrix().tolist()
        present_instance_validation = self.model.get_expert_instance_validation()
        mat[config.labeled_list_name] = np.array(range(len(instance_ranking))) \
            [present_instance_validation!=-1].tolist()
        #TODO: get influence file (buffer solution)
        # influence_path = os.path.join(config.data_root,
        #                               self.dataname,
        #                               config.info_data,
        #                               config.info_influence_name)
        # influence = json.load(open(influence_path, "r"))
        # mat[config.influence_name] = self.model.get_influence()
        mat[config.pre_posterior_name] = self.model.get_pre_posterior()

        return mat

    def _get_spammer_score_according_to_selected_list(self,selected_list):
        return self.crowd_workers.\
            get_spammer_scores_according_to_selected_list(selected_list).tolist()

class ValidationMaintainer(object):
    def __init__(self, dataname):
        self.dataname = dataname

        self._expert_instance_validation = {}
        self._expert_spammer_validation = {}
        self._buffer_expert_instance_validation = {}
        self._buffer_expert_spammer_validation = {}
        self._instance_list = []
        self._filtered_list = []
        self._last_round_list = []
        # TODO: BUG
        self._worker_list = {}

        self._flag_instance_new_fusion = 0
        self._flag_spammer_new_fusion = 0

    def _clean(self):
        self._expert_instance_validation = {}
        self._expert_spammer_validation = {}
        self._buffer_expert_instance_validation = {}
        self._buffer_expert_spammer_validation = {}
        self._instance_list = []
        self._last_round_list = []
        # TODO: BUG
        self._worker_list = {}

        self._flag_instance_new_fusion = 0
        self._flag_spammer_new_fusion = 0

    def adopt_validation(self, mat, filtered = False):
        '''
        mat should be a dict with two key words: validated_instances and validated_spammers.
        each key word corresponds to a dict. validated_instances contains multiple
        {instances_id,expert_label} pairs and validated_spammers contains multiple
        {worker_id, confusing_classes_list} pairs.
        :param mat:
        :return:
        '''

        expert_instance_validation = self._expert_instance_validation
        expert_spammer_validation = self._expert_spammer_validation
        buffer_expert_instance_validation = self._buffer_expert_instance_validation
        buffer_expert_spammer_validation = self._buffer_expert_spammer_validation

        validated_instances = mat["validated_instances"]
        validated_spammers = mat["validated_spammers"]
        selected_list = mat["selected_list"]
        type = selected_list[0] // 2

        if type not in self._worker_list:
            self._worker_list[type] = []

        # process instances validation
        for instance_id in validated_instances:
            if instance_id in expert_instance_validation or instance_id in buffer_expert_instance_validation:
                # print("(DEBUG) instance validation multiple times")
                continue
            expert_label = validated_instances[instance_id]
            buffer_expert_instance_validation[instance_id] = expert_label
            self._instance_list.append(int(instance_id))
            if filtered is True:
                self._filtered_list.append(int(instance_id))

        # process spammer validation
        for worker_id in validated_spammers:
            confusing_classes_list = validated_spammers[worker_id]
            # check expert_spammer_validation
            if worker_id in expert_spammer_validation:
                for _class in confusing_classes_list:
                    if _class in expert_spammer_validation[worker_id]:
                        # print("(DEBUG) spammer class validation multiple times")
                        continue
            # check buffer
            if worker_id in buffer_expert_spammer_validation:
                for _class in confusing_classes_list:
                    if _class in buffer_expert_spammer_validation[worker_id]:
                        # print("(DEBUG) spammer class validation multiple times")
                        continue
                    else:
                        buffer_expert_spammer_validation[worker_id].append(_class)
            else:
                buffer_expert_spammer_validation[worker_id] = []
                for _class in confusing_classes_list:
                    buffer_expert_spammer_validation[worker_id].append(_class)
                    if int(worker_id) not in self._worker_list[type]:
                        self._worker_list[type].append(int(worker_id))

        # TODO: when should update
        if len(buffer_expert_instance_validation) > 100:
            self._flag_instance_new_fusion = 1
        if len(buffer_expert_spammer_validation) > 100:
            self._flag_spammer_new_fusion = 1



    def get_all_instance_validation_and_clean_buffer(self):
        self._flag_instance_new_fusion = 0
        expert_instance_validation = self._expert_instance_validation
        buffer_expert_instance_validation = self._buffer_expert_instance_validation
        temp_buffer = buffer_expert_instance_validation.copy()

        # flag_get_new_instance_validation = False
        # if len(buffer_expert_instance_validation) > 0:
        #     flag_get_new_instance_validation = True

        # get all instance validation
        for instance_id in buffer_expert_instance_validation:
            expert_instance_validation[instance_id] = \
                buffer_expert_instance_validation[instance_id]

        # clean buffer of instance validation
        buffer_expert_instance_validation = {}

        # updating number variables
        self._buffer_expert_instance_validation = buffer_expert_instance_validation
        self._expert_instance_validation = expert_instance_validation

        return expert_instance_validation, temp_buffer

    def get_all_spammer_validation_and_clean_buffer(self):
        self._flag_spammer_new_fusion = 0
        expert_spammer_validation = self._expert_spammer_validation
        buffer_expert_spammer_validation = self._buffer_expert_spammer_validation
        # get all spammer validation
        for worker_id in buffer_expert_spammer_validation:
            confusing_classes_list = buffer_expert_spammer_validation[worker_id]
            if worker_id not in expert_spammer_validation:
                expert_spammer_validation[worker_id] = []
            for _class in confusing_classes_list:
                expert_spammer_validation[worker_id].append(_class)
        # clean buffer of instance validation
        buffer_expert_spammer_validation = {}
        # updating number variables
        self._buffer_expert_spammer_validation = buffer_expert_spammer_validation
        self._expert_spammer_validation = expert_spammer_validation

        return expert_spammer_validation

    @property
    def need_updating_instance(self):
        return self._flag_instance_new_fusion

    @property
    def need_updating_spammer(self):
        return self._flag_spammer_new_fusion

    @property
    def instance_list(self):
        return self._instance_list
    @property
    def worker_list(self):
        return self._worker_list

    @property
    def last_round_list(self):
        return self._last_round_list

    def reset_instance_list(self):
        self._last_round_list = self._instance_list.copy()
        self._instance_list = []
    def reset_worker_list(self):
        self._worker_list = {}

class BackendModel(ModelMaintainer, ValidationMaintainer):
    def __init__(self, dataname=None, beta2=6, simi_threshold=0.8):
        # super(BackendModel, self).__init__(dataname=dataname)
        ModelMaintainer.__init__(self,dataname, beta2=beta2, simi_threshold=simi_threshold)
        ValidationMaintainer.__init__(self, dataname)
        self._selected_list = None
        # self.load_model()

    def update_dataname(self, dataname):
        self.__init__(dataname)
        # self.load_model()

    def update_selected_list(self,selected_list):
        print("model selected list is updated!!!")
        self._selected_list = selected_list

    def get_selected_list(self):
        return self._selected_list

    def get_instance_list(self):
        return self.instance_list

    def get_last_round_list(self):
        return self.last_round_list

    def get_worker_list(self):
        return self.worker_list

    def get_instance_influence(self, selected_list):
        return self._get_instance_influence(selected_list, self.get_instance_list())

    def get_worker_influence(self, selected_list):
        type = selected_list[0] // 2
        return self._get_worker_influence(selected_list, self.get_worker_list()[type])

    def model_adopt_validation(self):
        if 1:#self.need_updating_instance:
            instances_validation, temp_buffer = self.get_all_instance_validation_and_clean_buffer()
            self.adopt_instance_validation(instances_validation, temp_buffer)
        if 1:#self.need_updating_spammer:
            spammers_validation = self.get_all_spammer_validation_and_clean_buffer()
            self.adopt_spammer_validation(spammers_validation)


    def propagation(self,seed=None):

        self.model_adopt_validation()

        self._propagation(seed=seed)
        self.reset_instance_list()
        self.reset_worker_list()

        updating_info = self.get_dynamic_info(self._selected_list)
        return updating_info

    def tsne_update(self, center_points, selected_list, TSNE, re_calculate_tsne=False):
        posterior_labels = self.model.get_posterior_labels()
        prev_posterior_labels = self.model.get_pre_posterior()
        expert_instance_validation = self.model.get_expert_instance_validation()
        if len(prev_posterior_labels) < 1:
            changed_indicator = np.zeros(len(posterior_labels)).astype(bool)
        else:
            prev_posterior_labels = np.array(prev_posterior_labels)
            changed_indicator = (posterior_labels!=prev_posterior_labels)
        last_round_list = self.get_last_round_list()
        changed_indicator[expert_instance_validation!=-1] = False
        dynamic_info = self.get_dynamic_info(selected_list)
        data = self.crowd_tsne.tsne_update(center_points=center_points,
                                    dynamic_info=dynamic_info,
                                    selected_list=selected_list,
                                    TSNE = TSNE,
                                    re_calculate_tsne=re_calculate_tsne,
                                    last_round_list = last_round_list,
                                    changed_indicator=changed_indicator)

        data["selected_spammer_score"] = dynamic_info[config.spammer_score_name]
        data[config.worker_accuracy_name] = dynamic_info[config.worker_accuracy_name]
        data[config.spammer_ranking_name] = dynamic_info[config.spammer_ranking_name]
        data[config.spammer_list_name] = dynamic_info[config.spammer_list_name]
        data[config.instance_ranking_name] = dynamic_info[config.instance_ranking_name]
        data[config.instance_list_name] = dynamic_info[config.instance_list_name]

        return data

    def roll_back(self,d):
        self._clean()
        self.adopt_validation(d)

    def get_spammer_score_according_to_selected_list(self):
        return self._get_spammer_score_according_to_selected_list(self._selected_list)

    def save_info(self):
        self.save_manifest()
        self.save_static_info()
        self.save_dynamic_info()
        self.save_model()

    def save_manifest(self):
        mat = {}
        mat[config.instance_num_name] = self.crowd_data.get_attr(config.instance_num_name)
        mat[config.worker_num_name] = self.crowd_data.get_attr(config.worker_num_name)
        mat[config.label_num_name] = self.crowd_data.get_attr(config.label_num_name)
        mat[config.label_names_name] = self.crowd_data.get_attr(config.label_names_name)
        open(self.info_manifest_path, "w").write(json.dumps(mat))
        print("manifest file has been saved in %s" % (self.info_manifest_path))

    def save_static_info(self):
        mat = {}
        mat[config.workers_labels_name] = self.crowd_data.get_attr(config.workers_labels_name)
        if len(mat[config.workers_labels_name]) < \
                        2 * self.crowd_data.get_attr(config.instance_num_name):
            mat[config.workers_labels_name] = np.array(mat[config.workers_labels_name]).reshape(-1).tolist()
        mat[config.similarity_matrix_name] = self.crowd_data.get_attr(config.similarity_matrix_name)
        mat[config.true_labels_name] = self.crowd_data.get_attr(config.true_labels_name)
        # TODO: for bird dataset because there is no features of birds images
        simi = decom_similarity_matrix(mat[config.similarity_matrix_name],
                                       self.crowd_data.get_attr(config.instance_num_name))
        mat[config.tsne_name] = get_tsne_from_similarity_matrix(simi).tolist()
        mat[config.knn_name] = get_knn(20, simi).tolist()
        try:
            mat[config.worker_type_name] = self.crowd_data.get_attr(config.worker_type_name)
            print(config.workers_type_name + "has saved")
        except:
            None
        open(self.info_static_path, "w").write(json.dumps(mat))
        print("static info file has been saved in %s" % (self.info_static_path))

    def save_dynamic_info(self):
        mat = self.get_dynamic_info([0,1])

        open(self.info_dynamic_path, "w").write(json.dumps(mat))
        print("dynamic info file has been saved in %s" % (self.info_dynamic_path))

    def save_model(self, model_path = None):
        if model_path is None:
            model_path = self.model_path
        fw = open(model_path,"wb")
        pickle.dump(self.model, fw)
        print("model has been saved in %s" %(model_path))
        fw.close()

    def save_debug_model(self):
        fw = open(self.debug_model_path,"wb")
        pickle.dump(self.model, fw)
        fw.close()

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        fr = open(model_path,"rb")
        self.model = pickle.load(fr)
        self.model.trained = 1
        self.crowd_instances.connect_to_model(self.model)
        self.crowd_workers.connect_to_model(self.model)
        self.crowd_tsne.connect_to_model(self.model)
        print("successfully load %s"%(model_path) )
        fr.close()

    def load_debug_model(self):
        fr = open(self.debug_model_path,"rb")
        self.model = pickle.load(fr)
        self.model.trained = 1
        self.crowd_instances.connect_to_model(self.model)
        self.crowd_workers.connect_to_model(self.model)
        self.crowd_tsne.connect_to_model(self.model)
        print("successfully load debug model: %s" %(self.debug_model_path))
        fr.close()

if __name__ == "__main__":
    bm = BackendModel(config.bird_dataset_name)
    bm.save_info()