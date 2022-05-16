function on_worker_selected(data) {
    if (data.length == 0) {
        on_worker_unselected();
    } else {
        var ids = [];
        for (var i = 0; i < data.length; i++){
            ids.push(data[i].id);
        }
        SelectedWorkers = data;
        WorkerView.set_selected(ids);
        if (data.length == 1) {
            var d = data[0];
            DetailView.draw_worker_detail(d.instances);
            //InstancesView.set_highlight_by_label(d.instances);
        }

        //on_glyph_unselected();
    }
}
function on_worker_unselected() {
    SelectedWorkers = [];
    WorkerView.set_selected([]);
    InstancesView.set_highlight_by_label([]);
}

function on_glyph_selected(data) {
    if (data.length == 0) {
        on_glyph_unselected();
    } else {
        var ids = [];
        for (var i = 0; i < data.length; i++){
            ids.push(data[i].id);
        }
        SelectedInstances = data;
        InstancesView.set_selected(ids);
        WorkerView.draw_selected(ids);
        DetailView.draw_instance_detail(ids);

        on_worker_unselected();
    }
}

function on_glyph_unselected() {
    SelectedInstances = [];
    InstancesView.set_selected([]);
    WorkerView.draw_selected([]);
    DetailView.set_selected([]);
}

function on_worker_clicked(d) {
    if (WorkerView.is_selected(d.id)){
        on_worker_unselected();
    }
    else {
        SelectedWorkers.push(d);
        on_worker_selected(SelectedWorkers);
        WorkerView.draw_dialog(d);
    }
}
function on_glyph_clicked(d) {
    if (InstancesView.is_selected(d.id)) {
        on_glyph_unselected();
    } else {
        on_glyph_selected([d]);
    }
}
function on_image_clicked(d) {
    if (DetailView.is_selected(d.id)){
        SelectedInstances = [];
        DetailView.set_selected([]);
        InstancesView.set_selected([]);
        WorkerView.set_highlight_by_label([]);
    } else {
        SelectedInstances = [d];
        DetailView.set_selected([d.id]);
        InstancesView.set_selected([d.id]);
        WorkerView.set_highlight_by_label(d.workers);
    }
}

function on_test_detail_clicked() {
    var ids = [];
    var token = $("#test_input").val().split(",");
    for (var i = 0; i < token.length; i++){
        ids.push(parseInt(token[i].trim()));
    }
    DetailView.draw_instance_detail(ids);
}

function on_plot_clicked(d) {
    DetailView.draw_instance_detail([d.id]);
    WorkerView.draw_selected([d.id]);
}


function on_worker_hover(d) {
    InstancesView.set_highlight_by_label(d.instances,d.id);
    WorkerView.highlight_flowmap([d.id]);
}

function on_worker_unhover(d) {
    InstancesView.set_highlight_by_label([]);
    WorkerView.highlight_flowmap([]);
}

function on_glyph_arc_hover(d) {
    var indexs = [];
    for (var i = 0; i < LabelNames.length; i++){
        indexs.push([]);
    }
    indexs[d.data.label] = d.data.workers;
    WorkerView.set_highlight_by_label(indexs);
    DetailView.set_highlight([d.data.parentid]);
}

function on_glyph_arc_unhover(d) {
    WorkerView.set_highlight_by_label([]);
    DetailView.set_highlight([]);
}
function on_glyph_center_hover(d) {
    WorkerView.set_highlight_by_label(d.workers);
    DetailView.set_highlight([d.id]);
    InstancesView.highlight_flowmap([d.id]);
}

function on_glyph_center_unhover(d) {
    WorkerView.set_highlight_by_label([]);
    DetailView.set_highlight([]);
    InstancesView.highlight_flowmap([]);
}

function on_worker_bar_hover(d) {
    InstancesView.set_highlight(d.instances);
    InstancesView.multi_highlight(d.instances, 1);
}
function on_worker_bar_unhover(d) {
    InstancesView.set_highlight([]);
    InstancesView.multi_highlight(d.instances,0);
}
function on_worker_bar_clicked(d) {
    DetailView.draw_instance_detail(d.instances);
}

function on_trail_clicked(d) {
    DetailView.draw_trail_detail(d);
}

function on_image_hover(d) {
    WorkerView.set_highlight_by_label(d.workers);
    InstancesView.single_highlight(d.id, 1);
    InstancesView.set_highlight([d.id]);
}

function on_image_unhover(d) {
    InstancesView.single_highlight(d.id,0);
    InstancesView.set_highlight([]);
    WorkerView.set_highlight_by_label([]);
}

function matrix_selected() {
    InstancesView.Matrix.narrow_matrix();
    on_classes_selected_end(InstancesView.Matrix.selected);
}

function on_matrix_clicked(d, i) {

}

function on_validated_dialog_update() {
    var data = d3.selectAll(".validated-dialog-img").data();
    var ids = [];
    for (var i = 0; i < data.length; i++){
        if($("#validated-dialog-checkbox" + data[i].id).is(':checked')){
            ids.push(data[i].id);
        }
    }
    return ids;
}

function change_bottom(flag) {
    d3.select(".select-button").style("display", function () {
        if (flag == 0 || flag == 2) {
            return "none";
        }
        return "block";
    })
    d3.select(".propagation-button").style("display", function () {
        if (flag == 0 || flag == 1) {
            return "none";
        }
        return "block";
    })
}

d = {
    "validated_instances":{},
    "validated_spammers" :{},
    "selected_list": [],
    "seed": 0
};

function validation_update(data, callback){
    validation_updating(data, callback);
}

function propagation_update(){
    d3.select("#preloader").style("display", "block");
    propagation(function(data){
        var state = propagation_handler(data);
        // if (state < 1){
        //     d3.select("#preloader").style("display", "none");
        //     return;
        // }
        on_glyph_unselected();
        console.log("class_selection_data_update!!!");
        class_selection_data_update(SelectedList);
        console.log("fetch guided tsne result!!!");
        fetch_guided_tsne_result(SelectedGlobal["Constraints"]);
        // incremental_tsne_result(SelectedGlobal["Constraints"]);
        console.log("update views!!!");
        InstancesView.Matrix.update_matrix(WorkerPredictionMatrix);
        console.log("propagation finished!!!");
        d3.select("#preloader").style("display", "none");
    });
}

function propagation_without_view_update(){

    d3.select("#preloader").style("display", "block");
    propagation(function(data) {
        var state = propagation_handler(data);
        if (state < 1) {
            d3.select("#preloader").style("display", "none");
            return;
        }
        on_glyph_unselected();
        console.log("class_selection_data_update!!!");
        class_selection_data_update(SelectedList);
        console.log("fetch guided tsne result!!!");
        InstancesView.Matrix.update_matrix(WorkerPredictionMatrix);
        console.log("propagation finished!!!");
        TrailView.draw();
        d3.select("#preloader").style("display", "none");
    });
}

// function trail_roll_back(data){
//     var d = {};
//     if(!data){
//         d["validated_instances"] = {}
//     }
// }

function instance_candidate_selection_update(data){
    var d = {};
    if(!data){
        d["validated_instances"] = {704:1};
    }
    else{
        d["validated_instances"] = data;
    }
    update_validated_labels(data);
    d["validated_spammers"] = {};
    var verified_data = data;
    instance_candidate_selection(d, function(data){
        var instance_ranking_index = data["InstanceRanking"];
        var influence = data["influence"];
        var top = data["top"];
        InstancesView.candidate_selection_draw(instance_ranking_index, influence);
        InstancesView.draw_flowmap();
        on_glyph_unselected();
        var feedback = {
            "validated_one":verified_data,
            "simi_list": top
        };
        // TODO: for case convenience
        more_instances(feedback);
        // $('.img-validated').modal('show');
        // DetailView.draw_validated_dialog(d3.select(".img-validated").select(".modal-body"), [1,2,3,4,5]);
    });
}

function worker_candidate_selection_update(data, id){
    var d = {};
    d["validated_instances"] = {};
    if(!data){
        d["validated_spammers"] = {1:[0,1]};
    }
    else{
        d["validated_spammers"] = data;
    }
    update_spammer_labels(data);
    worker_candidate_selection(d, function(data){
        // WorkerView.draw();
        InstancesView.draw(data["InstanceList"]);
        WorkerView.draw_flowmap(data["influence"]);
        WorkerView.update_dialog(id);
        on_worker_unselected();
    });
}



// /-------------------- boundary for Changjian -------------------------------/
function trail_roll_back(v_id){
    var new_traillog = [];
    for(var i = 0; i < v_id; i++){
        new_traillog.push(TrailLog[i]);
    }
    TrailLog = new_traillog;
    var validated_ids = {};
    for( var i = 0; i < TrailLog.length; i++ ){
        for(var j = 0; j < TrailLog[i][0].length; j++){
            var id = TrailLog[i][0][j]["id"];
            validated_ids[id] = true_label(id);
        }
    }
    var d = {};
    d["validated_instances"] = validated_ids;
    d["validated_spammers"] = TrailSpammer[v_id-1];
    d["selected_list"] = SelectedList;
    d["Seed"] = SeedLog[v_id-1];
    roll_back(d,function(data){
        roll_bake_handler(data);
        on_glyph_unselected();
        console.log("class_selection_data_update!!!");
        class_selection_data_update(SelectedList);
        console.log("fetch guided tsne result!!!");
        fetch_guided_tsne_result(SelectedGlobal["Constraints"]);
        // incremental_tsne_result(SelectedGlobal["Constraints"]);
        console.log("update views!!!");
        InstancesView.Matrix.update_matrix(WorkerPredictionMatrix);
        console.log("propagation finished!!!");
        d3.select("#preloader").style("display", "none");
    });
}

function set_trail(){
    TrailLog = {};
    TrailSpammer = {};
    SelectedList = [];
    Seed = 0;
}

function recover() {
    SeedLog[SeedLog.length-1] += 1;
    trail_roll_back(TrailLog.length);
}

function on_image_crop(info){
    cropping_info(info);
}


function on_classes_selected_end(selected_classes_list ){
    class_selection_data_update(selected_classes_list);
    d3.select("#preloader").style("display", "block");
    fetch_guided_tsne_result(function(){
        d3.select("#preloader").style("display", "none");
    });
    // DetailView.draw();
    // WorkerView.draw();
    // TODO : animation
    d3.select(".my-panel-heading")
        .text("Instance");
}

function on_classes_selected_start() {
    propagation_without_view_update();
    instance_redraw();
    DetailView.redraw();
    WorkerView.redraw();
    d3.select(".my-panel-heading")
        .text("Confusion");
    d3.select("#preloader").style("display", "none");
}

function instance_redraw(){
    InstancesView.redraw();
}

function on_circle_select(id){
    console.log("on circle selected");
    DetailView.draw_instance_detail([id]);
}


function error_uncertainty(){
    var static_list = [];
    var static_list_id = [];
    for(var i = 0; i < LabelTotalNum; i++){
        static_list.push([]);
        static_list_id.push([]);
    }
    for(var i = 0; i < TrueLabels.length; i++ ){
        if(PosteriorLabels[i] != TrueLabels[i]){
            static_list[TrueLabels[i]].push({
                "id": i,
                "Uncertainty": Uncertainty[i],
                "PosteriorLabel": PosteriorLabels[i]
            });
            static_list_id[TrueLabels[i]].push(i);
        }
    }
    for(var i = 0; i < LabelTotalNum; i++){
        static_list[i].sort(function (a, b) {
            return a.Uncertainty - b.Uncertainty;
        })
        static_list_id[i].sort(function (a, b) {
            return Uncertainty[a] - Uncertainty[b];
        })
    }
    return [static_list, static_list_id];
}