from flask import Flask, jsonify, send_file, request, render_template
import numpy as np
import os
import sys
import json
from time import time
import logging


from scripts.backend import load_static_data, load_manifest_data, generate_dynamic_data, incre_guided_tsne_data
from scripts.backend import generate_guided_tsne_data
from scripts.helper1 import listdir_sorted_by_date
from scripts.tsne import guide_tsne
from scripts.configs import config
from scripts.backend_model import BackendModel
from scripts.cropping import cropping_process
from scripts.backend_model import decom_similarity_matrix
from scripts.Logger import Logger
from scripts.backend import get_time_string

from scripts.clustering import *

# get server root
SERVER_ROOT = os.path.dirname(sys.modules[__name__].__file__)

#Backend Model
backend_model = BackendModel()

# global variables
WorkerLabels = None
InstanceTotalNum = None
SimilarityMatrix = None
simi = None
ExpertLabel = None
SpammerScore = None
Uncertainty = None
TSNE = None
PosteriorDistLabels = None
seed = None
# app = Flask(__name__, static_url_path="/static")
app = Flask(__name__)


@app.route("/")
def index():
    id = request.remote_addr
    agent = request.user_agent
    print("usr id:%s, time:%s"%(id, get_time_string()))
    return render_template("index.html")

@app.route("/api/manifest", methods=["GET"])
def get_manifest():

    # extract info from request
    dataset_identifier = request.args["dataset"]
    dataset_path = os.path.join(SERVER_ROOT, "data", dataset_identifier, "info")
    # get according data
    d = load_manifest_data(os.path.join(dataset_path, config.info_manifest_name))


    # global InstanceTotalNum
    # global backend_model# update global variables
    #
    # InstanceTotalNum = d[config.instance_num_name]
    # t1 = time()
    # backend_model.update_dataname(dataset_identifier)
    # backend_model.load_model()
    # print("worker num",backend_model.crowd_data.get_attr(config.worker_num_name))
    # posterior_labels = backend_model.model.get_posterior_labels()
    # true_labels = backend_model.model.true_labels - 1
    # print("model initial accuracy", ( sum(posterior_labels == np.array(true_labels)) / len(true_labels)))
    # print(backend_model.model.trained)
    # print("total time:%s"%(time() - t1))

    return jsonify(d)

@app.route("/api/model-update", methods=["GET"])
def model_update():
    global InstanceTotalNum
    global backend_model# update global variables

    dataset_identifier = request.args["dataset"]
    dataset_path = os.path.join(SERVER_ROOT, "data", dataset_identifier, "info")
    # get according data
    d = load_manifest_data(os.path.join(dataset_path, config.info_manifest_name))

    InstanceTotalNum = d[config.instance_num_name]
    t1 = time()
    backend_model.update_dataname(dataset_identifier)
    backend_model.load_model()
    print("worker num",backend_model.crowd_data.get_attr(config.worker_num_name))
    posterior_labels = backend_model.model.get_posterior_labels()
    true_labels = backend_model.model.true_labels - 1
    print("model initial accuracy", ( sum(posterior_labels == np.array(true_labels)) / len(true_labels)))
    print(backend_model.model.trained)
    print("total time:%s"%(time() - t1))
    feedback = {
        "feedback": "success"
    }
    return jsonify(feedback)


@app.route("/api/static-info", methods=["GET"])
def get_static_info():
    global WorkerLabels
    global SimilarityMatrix
    global simi
    global InstanceTotalNum
    global TSNE

    t1 = time()

    # extract info from request
    dataset_identifier = request.args["dataset"]
    dataset_path = os.path.join(SERVER_ROOT, "data", dataset_identifier, "info")
    # get according data
    d = load_static_data(os.path.join(dataset_path, "static_info.json"))

    # update global variables
    WorkerLabels = d[config.workers_labels_name]

    SimilarityMatrix = d[config.similarity_matrix_name]
    simi = decom_similarity_matrix(SimilarityMatrix, InstanceTotalNum)
    TSNE = d[config.tsne_name]
    d[config.similarity_matrix_name] = "None" # SimilarityMatrix

    print("static info backend time cost: %s" %(time() - t1))

    return jsonify(d)

@app.route("/api/dynamic-info", methods=["GET"])
def get_dynamic_info():
    t1 = time()
    print("dynamic starting time: %s" %(get_time_string()))
    global SpammerScore
    global Uncertainty
    global PosteriorDistLabels

    # extract info from request
    dataset_identifier = request.args["dataset"]
    dataset_path = os.path.join(SERVER_ROOT, "data", dataset_identifier, "info")
    # get according data
    d = generate_dynamic_data(filename=os.path.join(dataset_path, "dynamic_info.json"))

    # update global variables
    SpammerScore = d[config.spammer_score_name]
    Uncertainty = d[config.uncertainty_name]
    PosteriorDistLabels = d[config.posterior_distribution_name]

    mat = d[config.worker_prediction_matrix_name]
    mat = np.array(mat).reshape(10,10)
    idx = np.array(range(10))
    np.random.shuffle(idx)
    mat = mat[idx,:]
    mat = mat[:,idx]
    d[config.worker_prediction_matrix_name] = mat.tolist()

    feedback = jsonify(d)

    print("dynamic info backend time cost: %s" %(time() - t1))

    return feedback

@app.route("/api/more-instances", methods=["POST"])
def more_instances():
    global backend_model

    data = json.loads(request.get_data())
    validated_one_label = -10
    for k in data["validated_one"]:
        if validated_one_label > -5:
            raise ValueError("multi-validation in more instances")
        validated_one_label = data["validated_one"][k]
    simi_list = data["simi_list"]
    validated_instances = {}
    posterior_labels = backend_model.model.get_posterior_labels()
    true_labels = np.array(backend_model.model.true_labels - 1)
    for i in simi_list:
        if true_labels[i] == validated_one_label:
            validated_instances[i] = validated_one_label
    expert_validation = {}
    expert_validation["validated_instances"] = validated_instances
    expert_validation["validated_spammers"] = {}
    expert_validation["selected_list"] = [-2]
    print(expert_validation)
    backend_model.adopt_validation(expert_validation, filtered=True)

    feedback = {
        "feedback":"success"
    }
    return jsonify(feedback)

@app.route("/api/expert-info", methods=["POST"])
def get_expert_info():
    global backend_model

    # get expert validation from frontend
    expert_validation = json.loads(request.get_data())
    # update expert validation
    print(expert_validation)
    backend_model.adopt_validation(expert_validation)
    backend_model.propagation()
    # try:
    #     seed = expert_validation["seed"]
    # except:
    #     seed = None
    # TODO: make sure feedback can be jsonified

    d = {
        "feedback": "1"
    }
    return jsonify(d)

@app.route("/api/instance-candidate-selection", methods=["POST"])
def instance_candidate_selection():
    global backend_model
    global simi

    # get expert validation from frontend
    expert_validation = json.loads(request.get_data())
    #update expert validation
    print(expert_validation)
    for i in expert_validation["validated_instances"]:
        validated_id = int(i)
    backend_model.adopt_validation(expert_validation)
    backend_model.model_adopt_validation()
    selected_list = expert_validation["selected_list"]

    #TODO: BUG
    dynamic_info = backend_model.get_dynamic_info(selected_list)
    instance_ranking = dynamic_info[config.instance_ranking_name]
    instance_ranking_index = np.array(instance_ranking).argsort()[::-1].tolist()
    influence = backend_model.get_instance_influence(selected_list)
    simi_vect = simi[validated_id,:]
    simi_vect[validated_id] = 0
    top_5 = simi_vect.argsort()[::-1][:5].tolist()

    d = {
        "InstanceRanking":instance_ranking_index,
        "influence": influence,
        "top": top_5
    }
    return jsonify(d)


@app.route("/api/worker-candidate-selection", methods=["POST"])
def worker_candidate_selection():
    global backend_model

    # get expert validation from frontend
    expert_validation = json.loads(request.get_data())
    backend_model.model_adopt_validation()
    #update expert validation
    print(expert_validation)
    backend_model.adopt_validation(expert_validation)
    selected_list = expert_validation["selected_list"]

    #TODO: BUG
    dynamic_info = backend_model.get_dynamic_info(selected_list)
    spammer_ranking = dynamic_info[config.spammer_ranking_name]
    influence, spammer_list, instance_list = backend_model.get_worker_influence(selected_list)

    d = {
        "SpammerRanking" : spammer_ranking,
        "influence": influence,
        "SpammerList": spammer_list,
        "InstanceList": instance_list
    }
    return jsonify(d)

@app.route("/api/cropping-info", methods=["POST"])
def cropping():
    global backend_model

    cropping_info = json.loads(request.get_data())
    feedback = cropping_process(cropping_info)

    return jsonify(feedback)

@app.route("/api/roll-back", methods=["POST"])
def roll_back():
    global backend_model

    data = json.loads(request.get_data())
    print(backend_model.model._propagation_iter)
    print(data)
    backend_model.roll_back(data)
    feedback = backend_model.propagation()
    feedback[config.pre_posterior_name] = feedback[config.posterior_labels_name]
    feedback["SelectedListThisRound"] = []
    json_feedback = jsonify(feedback)
    return json_feedback

@app.route("/api/propagation", methods=["GET"])
def propagation():
    global backend_model

    # TODO: For Debug
    if 0:
    # if os.path.exists(backend_model.debug_model_path):
        backend_model.load_debug_model()
        backend_model.model._propagation_iter = 2
        feedback = backend_model.get_dynamic_info([0,1])
        # for workers view debuging
        # feedback["WorkerAccuracy"][0] = 0
        # feedback["WorkerAccuracy"][1] = 0
        # feedback["WorkerPredictionMatrix"][0][0] = 0.55
        # feedback["WorkerPredictionMatrix"][2][1] = 0.3
    else:
        print(backend_model.model._propagation_iter)
        selected_list_this_round = backend_model.get_instance_list()
        selected_spammer_this_round = backend_model.get_worker_list()
        print("selected_list_this_round", selected_list_this_round)
        # TODO: big bug when workers selected
        if (len(selected_list_this_round) < 1 and len(selected_spammer_this_round) < 1):
        # if 0:
            feedback = {}
        else:
            feedback = backend_model.propagation(seed)
        # backend_model.save_debug_model()

        feedback["SelectedListThisRound"] = selected_list_this_round
        if(len(selected_list_this_round)< 1 and len(selected_spammer_this_round) > 0):
            feedback["SelectedListThisRound"] = [-1]
        feedback["seed"] = backend_model.model.seed

        # debug
        # posterior_labels = feedback[config.posterior_distribution_name]
        # posterior_labels = np.array(posterior_labels).reshape(800,4).argmax(axis=1)
        # prev_posterior_label = feedback[config.pre_posterior_name]
        # print("Label changed in this round!",sum(posterior_labels != prev_posterior_label))

    # backend_model.save_debug_model()
    print("successfully feedback!!")

    json_feedback = jsonify(feedback)
    # joblib.dump(json_feedback, os.path.join("./feedback.json"))
    # json_feedback = joblib.load("./feedback.json")
    # try:
    #     re_feedback = json.loads(json_feedback)
    # except:
    #     a = 1

    # return jsonify(feedback)
    return json_feedback

@app.route("/api/image", methods=["GET"])
def get_images():
    dataset_identifier = request.args["dataset"]
    filename = request.args["filename"]
    path = os.path.join(SERVER_ROOT, "data", dataset_identifier, "image", filename)
    if path and os.path.exists(path):
        return send_file(path)
    else:
        return jsonify({
            "message": "not exists",
            "status": "failure"
        })

@app.route("/api/tsne-update", methods=["POST"])
def tsne_update():
    data = json.loads(request.get_data())
    selected_list = data["selected_list"]
    backend_model.update_selected_list(selected_list)
    data = backend_model.tsne_update(center_points=data["center_points"],
                                     selected_list=selected_list,
                                     TSNE = TSNE,
                                     re_calculate_tsne=False)
    return jsonify(data)
# def tsne_update():
#     data = json.loads(request.get_data())
#     print(data)
#     selected_list = data["selected_list"]
#     backend_model.update_selected_list(data["selected_list"])
#     data = generate_guided_tsne_data(dataname=data["dataset"],
#                                      center_points=data["center_points"],
#                                      selected_list=data["selected_list"],
#                                      similarity_matrix=SimilarityMatrix,
#                                      current_tsne=TSNE,
#                                      worker_labels = WorkerLabels,
#                                      uncertainty=backend_model.crowd_instances.get_uncertainty(),
#                                      posterior_dist_labels=backend_model.crowd_instances.posterior_label_dist,
#                                      re_calculate_tsne=False)
#
#     dynamic_info = backend_model.get_dynamic_info(selected_list)
#     data["selected_spammer_score"] = dynamic_info[config.spammer_score_name]
#     data[config.worker_accuracy_name] = dynamic_info[config.worker_accuracy_name]
#     data[config.spammer_ranking_name] = dynamic_info[config.spammer_ranking_name]
#     data[config.spammer_list_name] = dynamic_info[config.spammer_list_name]
#     data[config.instance_ranking_name] = dynamic_info[config.instance_ranking_name]
#     data[config.instance_list_name] = dynamic_info[config.instance_list_name]
#     return jsonify(data)

@app.route("/api/incre-tsne-update", methods=["POST"])
def incre_tsne_update():
    data = json.loads(request.get_data())
    print(data)
    incre_guided_tsne_data()

@app.route('/api/query-dataset', methods=['GET'])
def query_tag_names():
    datasets = listdir_sorted_by_date(os.path.join(SERVER_ROOT, "data"))
    return jsonify(datasets)

@app.route("/api/save-trail", methods=["POST"])
def save_trail():
    data = json.loads(request.get_data())
    selected_list = data["SelectedList"]
    filename = "".join([str(i) + "_" for i in selected_list]) + "log.json"
    filename = os.path.join(config.data_root, backend_model.dataname,
                            "case_log", filename)
    open(filename, "w").write(json.dumps(data))
    feedback = {
        "feedback": "success"
    }
    return jsonify(feedback)

@app.route("/api/load-trail", methods=["POST"])
def load_trail():
    data = json.loads(request.get_data())
    selected_list = data["SelectedList"]
    filename = "".join([str(i) + "_" for i in selected_list]) + "log.json"
    filename = os.path.join(config.data_root, backend_model.dataname,
                            "case_log", filename)
    data = json.load(open(filename,"r"))

    return jsonify(data)

def start_server(port=8181):
    app.run(port=port, host="0.0.0.0", threaded=True)

if __name__ == "__main__":
    sys.stdout = Logger(os.path.join(SERVER_ROOT, "console.log"))
    start_server()
