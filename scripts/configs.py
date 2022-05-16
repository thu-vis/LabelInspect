import sys
import os
SERVER_ROOT = os.path.dirname(sys.modules[__name__].__file__)

class Config(object):
    def __init__(self):
        #raw data root
        self.row_data_root = os.path.join(SERVER_ROOT,"../../RawData")
        self.cropping_root = os.path.join(SERVER_ROOT, "../../Codes/mingming/test_images")
        self.log_root = os.path.join(SERVER_ROOT,"../log/")

        # first-level directory
        self.data_root = os.path.join(SERVER_ROOT,"../data/")
        self.scripts_root = os.path.join(SERVER_ROOT, "../scripts/")
        self.server_data_root = "data"

        # second-level directory
        self.dog_dataset_name = "dog"
        self.monkey_dataset_name = "monkey"
        self.bird_dataset_name = "bird"
        self.dll_name = "dll"

        # third-level directory
        self.origin_data = "origin_data"
        self.info_data = "info/"
        self.image_data = "image"
        self.confusion_matrix_data_path = "confusion_matrix"
        self.estimated_confusion_matrix_data_path = "estimated_confusion_matrix"
        self.worker_simi_buffer_path = "worker_simi_buffer"

        # filename
        self.manifest_name = "manifest.json"
        self.static_info_name = "static_info.json"
        self.crowd_data_name = "crowd_data.json"
        self.info_manifest_name = "manifest.json"
        self.info_static_name = "static_info.json"
        self.info_dynamic_name = "dynamic_info.json"
        self.info_guided_tsne_name = "guided_tsne_info.json"
        self.info_unnorm_guided_tsne_name = "unnorm_guided_tsne_info.json"
        self.info_influence_name = "influence.json"
        self.init_model_name = "init_model.pkl"
        self.debug_model_name = "debug_model.pkl"
        self.solution_uncertainty_name = "solution_uncertainty.npy"

        # variable name
        self.instance_num_name = "InstanceTotalNum"
        self.worker_num_name = "WorkerTotalNum"
        self.label_num_name = "LabelTotalNum"
        self.label_names_name = "LabelNames"
        self.workers_labels_name = "WorkerLabels"
        self.similarity_matrix_name = "SimiGraph"
        self.true_labels_name = "true_labels"
        self.uncertainty_name = "Uncertainty"
        self.spammer_ranking_name = "SpammerRanking"
        self.spammer_list_name = "SpammerList"
        self.instance_ranking_name = "InstanceRanking"
        self.instance_list_name = "InstanceList"
        self.worker_accuracy_name = "WorkerAccuracy"
        self.posterior_distribution_name = "PosteriorDistribution"
        self.posterior_labels_name = "PosteriorLabels"
        self.spammer_score_name = "SpammerScore"
        self.worker_type_name = "WorkerType"
        self.tsne_name = "TSNECoordinate"
        self.knn_name = "KNeighbors"
        self.guided_tsne_name = "GuidedTSNE"
        self.mixed_index_name = "MixedIndex"
        self.constraint_name = "Constraint"
        self.influence_name = "Influence"
        self.worker_prediction_matrix_name = "WorkerPredictionMatrix"
        self.pre_posterior_name = "PrePosterior"
        self.labeled_list_name = "LabeledList"
        self.worker_changed_list_name = "WorkerChangedList"

        #extension
        self.image_ext = ".jpg"

config = Config()