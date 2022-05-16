/*
    Notice: variables like AbcDefgHijk are globals ones
            variables like abc_defg_hijk are local ones
 */



var load_data = function (dataset) {
    /*
    * load data that need to be stored in global variables
    * */
    console.log("loading data...");
    DatasetName = dataset;
    var params = "?dataset=" + DatasetName;
    var manifest_node = new request_node(ManifestApi + params, manifest_handler, "json", "GET");
    var model_update_node = new request_node(ModelUpdate + params, function(){}, "json", "GET");
    var dynamic_node = new request_node(DynamicInfoApi + params, dynamic_handler, "json", "GET");
    dynamic_node.depend_on(manifest_node);
    var static_node = new request_node(StaticInfoApi + params, static_handler, "json", "GET");
    static_node.depend_on(manifest_node);
    manifest_node.notify();
    model_update_node.notify();
    console.log("loading finished.")
};

var remove_dom = function(){
    d3.select("#block-1-1").selectAll("svg").remove();
    d3.select("#block-1-2").selectAll("svg").remove();
    d3.select("#block-2-1").selectAll("svg").remove();
    // LabelMenu.destroy();
    // SpammerMenu.destroy();
};

var setup = function () {
    /*
    * instantiate all view objects*/
    // class_selection_data_update();
    InstancesView = new InstancesLayout(d3.select("#block-1-1"));
    // TODO: for debug
    // fetch_guided_tsne_result(SelectedGlobal["Constraints"]);

    WorkerView = new Worker(d3.select("#block-1-2"));
    TrailView = new HistoryTrail(d3.select("#revision-trail-container"));
    DetailView = new ImageDetail(d3.select("#detail-image-container"));

    InstanceMenu = new InstanceLabelMenu(".instance-filter",LabelNames);
    SpammerMenu = new WorkerSpammerMenu(".worker-node",LabelNames);
    SpammerMenu.createSingleMenu(".worker-dialog-matrix-label");

    d3.select("#preloader").style("display", "none");
    // ranking_view_setup();
};

$(document).ready(function () {

    var loading_button = d3.selectAll("#dataset");

    load_data("bird");
    loading_button.on("click",function(){
        var dataset = this.text;
        remove_dom();
        load_data(dataset);
    });

});

var resizeTimer = null;


$(window).resize(function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(resize, 50);

});

function resize() {
    WorkerView.resize();
    InstancesView.resize();
    TrailView.resize();
    DetailView.resize();
    //updateGraph();
}