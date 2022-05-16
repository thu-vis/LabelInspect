/*

*/

var manifest_handler = function (data) {
    ManifestData = data;
    InstanceTotalNum = data.InstanceTotalNum;
    WorkerTotalNum = data.WorkerTotalNum;
    LabelTotalNum = data.LabelTotalNum;
    ModelTotalNum = data.ModelTotalNum;
    LabelNames = data.LabelNames;
    UncertaintyMax = null;
    //TODO : need a more robust way to get url
    //get url
    ImageUrl = [];
    for (var i = 0; i < InstanceTotalNum; i++) {
        ImageUrl[i] = ImageInfoApi + "?dataset=" + DatasetName + "&filename=" + (i + 1) + ".jpg";
    }
    console.log("manifest data load finished");
};

var static_handler = function (data) {
    StaticInfoData = data;
    // similarity matrix from backend is in a compress form
    SimilarityMatrix = simi_matrix(data.SimiGraph, InstanceTotalNum);

    // WorkerLabels is a 2-d matrix. The first dimension is instance and the second one is worker.
    // -1 means worker did not label corresponding
    WorkerLabels = reform_workers_labels(data.WorkerLabels, WorkerTotalNum, InstanceTotalNum);

    // KNeighbors = data.KNeighbors;
    // TSNECoordinate = data.TSNECoordinate;

    TrueLabels = data.true_labels;
    //TODO: fake matrix here for dog dataset
    // WorkerPredictionMatrix = [[0.62,0.6,0.052,0.01],[0.38,0.35,0,0.021],[0,0.038,0.56,0.094],[0,0.013,0.38,0.87]];

    console.log("static data load finished");
};

var dynamic_handler = function (data) {
    DynamicInfoData = data;
    PosteriorDistribution = data.PosteriorDistribution;
    PosteriorLabels = get_posterior_labels(PosteriorDistribution, InstanceTotalNum, LabelTotalNum);
    // PosteriorLabels = data.PosteriorLabels;
    // NOT TODO : this is for true labels result
    // PosteriorLabels = TrueLabels;
    Uncertainty = uncertainty_normalization(data.Uncertainty);
    SpammerScore = data.SpammerScore;
    SpammerRanking = data.SpammerRanking;
    SpammerList = data.SpammerList;
    WorkerAccuracy = data.WorkerAccuracy;
    Influence = data.Influence;
    WorkerPredictionMatrix =
        worker_prediction_matrix_processor(data.WorkerPredictionMatrix, LabelTotalNum);
    console.log("dynamic data load finished");

    setup();
};

var roll_bake_handler = function(data){
    global_update(data);
    SpammerMenu.update(ValidatedSpammer);
};

var propagation_handler = function(data){
    console.log("get propagation result.");
    SelectedListThisRound = data.SelectedListThisRound;
    // if( SelectedListThisRound.length < 1){
    //     console.log("no instance update this round!!");
    //     return 0;
    // }
    global_update(data);
    // TrailLog.push(trail_log_record());
    return 1;
};

var global_update = function(data){
    PosteriorDistribution = data.PosteriorDistribution;
    PosteriorLabels = get_posterior_labels(PosteriorDistribution, InstanceTotalNum, LabelTotalNum);
    PrePosteriorLabels = data.PrePosterior;
    LabeledList = data.LabeledList;
    Uncertainty = uncertainty_normalization(data.Uncertainty);
    SpammerRanking = data.SpammerRanking;
    SpammerList = data.SpammerList;
    InstanceRanking = data.InstanceRanking;
    InstanceList = data.InstanceList;
    SpammerScore = data.SpammerScore;
    WorkerAccuracy = data.WorkerAccuracy;
    Influence = data.Influence;
    WorkerPredictionMatrix =
        worker_prediction_matrix_processor(data.WorkerPredictionMatrix, LabelTotalNum);
    Seed = data.seed;
    ChangedWorker = data.WorkerChangedList;
    if(!ChangedWorker){
        ChangedWorker = {};
    }
}

var guided_tsne_handler = function(data){
    GuidedTSNE = data.GuidedTSNE;
    MixedIndex = data.MixedIndex;
    Constraint = data.Constraint;
    SpammerScore = data.selected_spammer_score;
    WorkerAccuracy = data.WorkerAccuracy;
    SpammerRanking = data.SpammerRanking;
    SpammerList = data.SpammerList;
    InstanceRanking = data.InstanceRanking;
    InstanceList = data.InstanceList;
    guided_tsne_handler_draw();
};

function guided_tsne_handler_draw(){
    // InstancesView.data_update();
    InstancesView.draw(InstanceList);
    InstancesView.draw_flowmap(true);
    DetailView.draw();
    // DetailView.draw_flowmap([]);
    WorkerView.draw();
    WorkerView.remove_flowmap();
    TrailView.draw();
}