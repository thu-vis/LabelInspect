from flask import Flask, jsonify, send_file, request, render_template
import numpy as np
import os
import sys
import json

from scripts.backend import load_static_data, load_manifest_data, generate_dynamic_data
from scripts.backend import generate_guided_tsne_data
from scripts.backend_model import BackendModel
from scripts.helper1 import listdir_sorted_by_date
from scripts.tsne import guide_tsne
from scripts.configs import config

def guided_tsne_debug():
    dataname = config.dog_dataset_name
    dataset_path = os.path.join(
         config.data_root,
         dataname,
         config.info_data
    )
    static_data = load_static_data(os.path.join(dataset_path, config.static_info_name))
    SimilarityMatrix = static_data[config.similarity_matrix_name]
    TSNE = static_data[config.tsne_name]
    dynamic_data = load_static_data(os.path.join(dataset_path, config.info_dynamic_name))
    Uncertainty = dynamic_data[config.uncertainty_name]
    PosteriorDistLabels = dynamic_data[config.posterior_distribution_name]
    WorkerLabels = static_data[config.workers_labels_name]
    guided_tsne_data = load_static_data(os.path.join(dataset_path, config.info_guided_tsne_name))
    center_points = guided_tsne_data["center_points"]

    generate_guided_tsne_data(dataname=dataname,
                              center_points=center_points,
                              selected_list=[0, 1],
                              similarity_matrix=SimilarityMatrix,
                              current_tsne=TSNE,
                              worker_labels=WorkerLabels,
                              uncertainty=Uncertainty,
                              posterior_dist_labels=PosteriorDistLabels,
                              re_calculate_tsne=True,
                              buffer=False)

    # print(dynamic_data)

def incre_tsne_debug():

    dataname = config.dog_dataset_name
    dataset_path = os.path.join(
         config.data_root,
         dataname,
         config.info_data
    )
    print(dataset_path)
    backend_model = BackendModel(dataname)
    backend_model.load_debug_model()
    static_data = load_static_data(os.path.join(dataset_path, config.static_info_name))
    SimilarityMatrix = static_data[config.similarity_matrix_name]
    TSNE = static_data[config.tsne_name]
    dynamic_data = load_static_data(os.path.join(dataset_path, config.info_dynamic_name))
    Uncertainty = dynamic_data[config.uncertainty_name]
    PosteriorDistLabels = backend_model.crowd_instances.posterior_label_dist
    WorkerLabels = static_data[config.workers_labels_name]
    guided_tsne_data = load_static_data(os.path.join(dataset_path, config.info_guided_tsne_name))
    center_points = guided_tsne_data["center_points"]
    print(guided_tsne_data)

    generate_guided_tsne_data(dataname=dataname,
                              center_points=center_points,
                              selected_list=[0,1],
                              similarity_matrix=SimilarityMatrix,
                              current_tsne=TSNE,
                              worker_labels=WorkerLabels,
                              uncertainty=Uncertainty,
                              posterior_dist_labels=PosteriorDistLabels,
                              re_calculate_tsne=False,
                              buffer=False)



if __name__ == "__main__":
    guided_tsne_debug()
    # incre_tsne_debug()
