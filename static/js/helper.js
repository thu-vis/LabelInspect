/*
    some functions that are used multiple times in this project
*/


// /----------------- Deal with similarity matrix -------------------/
var get_start_point = function(instance_num){
    //get start point of simi matrix because the matrix store in a 1-dim list
    var start_point = [];
    start_point[0] = 0;
    for( var i = instance_num-1; i > 0; i-- ){
        start_point[ instance_num - i ] = start_point[instance_num - i - 1] + i;
    }
    return start_point;
};

function simi_matrix_single_element(data, start_point, i, j){
        if (i == j) {
            return 1;
        }
        else if (i < j) {
            return data[start_point[i] + j - i - 1];
        }
        else {

            return data[start_point[j] + i - j - 1];
        }
}

function simi_matrix(data, instance_num){
    var start_point = get_start_point(instance_num);
    var similarity_matrix = [];
    for(var i = 0; i < instance_num; i++ ){
        similarity_matrix[i] = [];
        for( var j = 0; j < instance_num; j++){
            similarity_matrix[i][j] = simi_matrix_single_element(data, start_point, i, j);
        }
    }
    return similarity_matrix;
}

// /------------------- get posterior_labels from posterior_distribution -------------/
function single_instance_posterior_label(posterior_distribution, label_num, id){
    var label = -1;
    var score = -1;
    for( var i = 0; i < label_num; i++ ){
        if( posterior_distribution[ id * label_num + i] > score ){
            label = i;
            score = posterior_distribution[ id * label_num + i];
        }
    }
    return label;
}

function get_posterior_labels(posterior_distribution, instance_num, label_num){
    var posterior_labels = [];
    for( var i = 0; i < instance_num; i++){
        posterior_labels[i] = single_instance_posterior_label(posterior_distribution, label_num, i);
    }
    return posterior_labels;
}

// /------------ worker prediction matrix ----------------/
function worker_prediction_matrix_processor(worker_prediction_matrix, label_num){
    console.log("worker prediction matrix updating");
    for( var i = 0; i < label_num; i++ ){
        for ( var j = 0; j < label_num; j++ ){
            worker_prediction_matrix[i][j] = worker_prediction_matrix[i][j].toFixed(2);
        }
    }
    return worker_prediction_matrix;
}

// /------------------ reform plated workers labels ---------------------------/
function reform_workers_labels(L, workers_num, instances_num){
    var workers_labels = [];
    for( var i = 0; i < instances_num; i++){
        workers_labels[i] = [];
        for( var j = 0; j < workers_num; j++){
            workers_labels[i][j] = L[ i * workers_num + j];
        }
    }
    return workers_labels;
}

// /------------------ uncertainty normalization -------------------------------/
function uncertainty_normalization(uncertainty){
    if(!UncertaintyMax){
        UncertaintyMax = Math.max.apply(null, uncertainty);
    }
    var res = [];
    for( var i = 0; i < uncertainty.length; i++ ){
        if( uncertainty[i] == 0){
            res[i] = 0;
        }
        else{
            res[i] = (uncertainty[i]) / ( UncertaintyMax );
        }
    }
    return res;
}
// /------------------ uncertainty normalization -------------------------------/

// /------------------ aggregation -------------------------------/
function uncertainty_aggregation(model, data, solution){
    var uncertainty = [];
    for( var i = 0; i < model.length; i++ ){
        uncertainty[i] = (model[i] + data[i] + solution[i])/ 3.0;
    }
    return uncertainty;
}

function spammer_score_aggregation(reliability, spammer_score){
    var spammer = [];
    for( var i = 0; i < reliability.length; i++ ){
        spammer[i] = ( spammer_score[i] - reliability[i] + 1)/2.0;
    }
    return spammer;
}
// /------------------ aggregation -------------------------------/

// /------------------ pie -----------------------------/
var pie = d3.pie().value(function(d){
    return d.value;
    })
    .sort(function (a, b) {
        return a.index - b.index;
    });
// /------------------ pie -----------------------------/

// /----------- previous posterior labels -------------/
var pre_posterior_labels = function(d){
    if(!PrePosteriorLabels || JSON.stringify(PrePosteriorLabels) == "{}"){
        return -1;
    }
    return PrePosteriorLabels[d];
};

var guided_tsne_coordinate_x = function(i){
    return GuidedTSNE[i][0];
};

var guided_tsne_coordinate_y = function(i){
    return GuidedTSNE[i][1];
};

// /----------- previous posterior labels -------------/

// /-----update global arc info and constraints in a unit coordinate according to selected info----/
function update_global_arc_info_and_constraints(selected_posterior_labels, selected_labels, selected_instance_num, selected_label_num){
    //get class percent for each class
    var sum = 0;
    var class_percent = [];
    for( var i = 0; i < selected_label_num; i++ ){
        class_percent.push(0);
    }
    for( var i = 0; i < selected_instance_num; i++ ){
        for(var j = 0; j < selected_label_num; j++ ){
            if( selected_posterior_labels[i] == selected_labels[j]){
                class_percent[j] += 1;
                sum += 1;
            }
        }
    }
    for( var i = 0; i < selected_label_num; i++ ){
        class_percent[i] /= sum;
    }
    for( var i = 0; i < class_percent.length; i++ ){ // for function pie API
        class_percent[i] = {
            "value": class_percent[i],
            "index": i
        };
    }

    // computing global arc information for each class
    var class_arc = pie(class_percent);
    var rotate = 0;
    for( var i = 0; i < class_arc.length; i++ ){
        if( class_arc[i].startAngle == 0){
            rotate = class_arc[i].endAngle / 2;
            break;
        }
    }
    for( var i = 0; i < class_arc.length; i++ ){
        class_arc[i].startAngle -= rotate;
        class_arc[i].endAngle -= rotate;
    }

    //get center point for each global arc
    var selected_center_point_in_unit_scale = [];
    for ( var i = 0; i < class_arc.length; i++ ){
        var center_arc = (class_arc[i].startAngle + class_arc[i].endAngle) / 2.0;
        var x = 0.5 * Math.sin(center_arc) + 0.5;
        // minus here for coordinate in frontend
        // of which the positive direction of y-axis is toward down
        var y = - 0.5 * Math.cos(center_arc) + 0.5;
        selected_center_point_in_unit_scale[i] = {
            "x": x,
            "y": y
        };
    }

    //TODO: adding constraint points according intuition
    var constraint_points_num_per_class = 7;
    var constraints = [];
    var shift_index = Math.floor(constraint_points_num_per_class/2);
    for ( var i = 0; i < class_arc.length; i++ ){
        var arc_per_part = ( class_arc[i].endAngle - class_arc[i].startAngle) /
             constraint_points_num_per_class;
        var center_arc = ( class_arc[i].endAngle + class_arc[i].startAngle) / 2.0;
        for( var j = 0; j < constraint_points_num_per_class; j++ ){
            if(constraint_points_num_per_class!=1 &&
                (j==0 || j==(constraint_points_num_per_class-1))){
                continue;
            }
            var arc = center_arc + (j - shift_index) * arc_per_part;
            var x = 0.5 * Math.sin(arc) + 0.5;
            var y = - 0.5 * Math.cos(arc) + 0.5;
            constraints.push({
                "x": x,
                "y": y,
                "class": selected_labels[i]
            })
        }
    }

    //NOTICE!!!! those codes will cover the result above
    // constraints = [];
    // constraints.push({
    //     "x":0.5156566647054978,
    //     "y": 0.17049094593711187 ,
    //     "class": 0
    // });
    // constraints.push({
    //     "x":0.8012028502320716,
    //     "y": 0.4047852520101981,
    //     "class": 1
    // });
    // constraints.push({
    //     "x":0.64953912531869 ,
    //     "y": 0.8576845133032621 ,
    //     "class": 2
    // });
    // constraints.push({
    //     "x":0.2782244884617899 ,
    //     "y": 0.690331437536772 ,
    //     "class": 3
    // });
    // constraints.push({
    //     "x":0.19977773419624764 ,
    //     "y": 0.39432568477479246 ,
    //     "class": 3
    // });
    return {
        "class_arc":class_arc,
        "selected_center_point_in_unit_scale":selected_center_point_in_unit_scale,
        "constraints": constraints
     }
}
// /-----update global arc info and constraints in a unit coordinate according to selected info----/

var trail_log_record = function(){

};


// /------------------  data update of class selection ----------------------------/
function class_selection_data_update(selected_list){
    if(!selected_list){
        selected_list = [4,5,6,7];
        // selected_list = [0,1,2,3,4,5,6,7,8,9];
    }
    selected_list.sort();
    console.log(selected_list);
    SelectedList = selected_list;
    SelectedGlobal = {};
    SelectedGlobal["LabelTotalNum"] = selected_list.length;
    SelectedGlobal["WorkerTotalNum"] = WorkerTotalNum;
    SelectedGlobal["LabelNames"] = [];
    SelectedGlobal["InstanceIndex"] = [];
    SelectedGlobal["PosteriorLabels"] = {};
    SelectedGlobal["Uncertainty"] = {};
    SelectedGlobal["WorkerLabels"] = {};
    SelectedGlobal["ImageUrl"] = {};
    SelectedGlobal["KNeighbors"] = {};
    SelectedGlobal["CategoryColor"] = [];


    for( var i = 0; i < selected_list.length; i++){
        SelectedGlobal["LabelNames"].push(LabelNames[selected_list[i]]);
        SelectedGlobal["CategoryColor"].push(CategoryColor[selected_list[i]]);
    }

    for( var i = 0; i < InstanceTotalNum; i++ ){
        for( var j = 0; j < selected_list.length; j++ ){
            if( PosteriorLabels[i] == selected_list[j]){
                SelectedGlobal["InstanceIndex"].push(i);
                SelectedGlobal["PosteriorLabels"][i] = PosteriorLabels[i];
                SelectedGlobal["Uncertainty"][i] = Uncertainty[i];
                SelectedGlobal["WorkerLabels"][i] = WorkerLabels[i];
                SelectedGlobal["ImageUrl"][i] = ImageUrl[i];
                // SelectedGlobal["KNeighbors"][i] = KNeighbors[i];


                // SelectedGlobal["InstanceIndex"].push(i);
                // SelectedGlobal["PosteriorLabels"].push(PosteriorLabels[i]);
                // SelectedGlobal["Uncertainty"].push(Uncertainty[i]);
                // SelectedGlobal["WorkerLabels"].push(WorkerLabels[i]);
                // SelectedGlobal["ImageUrl"].push(ImageUrl[i]);
                // SelectedGlobal["KNeighbors"].push(KNeighbors[i]);
            }
        }
    }

    // TODO
    if(SelectedList.length == LabelTotalNum){

    }
    else{
        SelectedGlobal["LabelNames"].push("Others");
        SelectedGlobal["CategoryColor"].push(Gray);

    }
    SelectedGlobal["InstanceTotalNum"] = SelectedGlobal["InstanceIndex"].length;

    var arc_info = update_global_arc_info_and_constraints(
        SelectedGlobal["PosteriorLabels"],
        SelectedList,
        SelectedGlobal["InstanceTotalNum"],
        SelectedGlobal["LabelTotalNum"]);
    SelectedGlobal["ClassArc"] = arc_info.class_arc;
    SelectedGlobal["SelectedCenterPointInUnitScale"] =
        arc_info.selected_center_point_in_unit_scale;
    SelectedGlobal["ConstraintsPoints"] = arc_info.constraints;

    if(SelectedListThisRound.length > 0){
        var new_validated_record = [];
        var new_changed_record = [];
        for( var i = 0; i < SelectedListThisRound.length; i++){
            new_validated_record.push({
                "id":SelectedListThisRound[i]
            });
        }
        for( var i = 0; i < PosteriorLabels.length; i++){
            if(PosteriorLabels[i] != pre_posterior_labels(i) &&
                !(SelectedListThisRound.includes(i))){
                new_changed_record.push({
                    "id": i
                });
            }
        }
        var seed = Seed;

        if( SelectedListThisRound.length == 1 && SelectedListThisRound[0] == -1){
            TrailLog.push([[], new_changed_record])
        }
        else{
            TrailLog.push([new_validated_record, new_changed_record]);
        }
        SeedLog.push(seed);
        SelectedListThisRound = [];

        var new_validated_spammer = {};
        for( var i in ValidatedSpammer){
            new_validated_spammer[i] = ValidatedSpammer[i];
        }
        TrailSpammer.push(new_validated_spammer);
        ValidatedSpammer = {};
    }

    SelectedGlobal["Trail"] = TrailLog;

    SelectedGlobal["ChangedWorker"] = ChangedWorker;
    // SelectedGlobal["ChangedWorker"] = {
    //     0:{
    //         "dx": 0.2,
    //         "dy": 0
    //     },
    //     7:{
    //         "dx": -0.3,
    //         "dy": -0
    //     }
    // }


    // SelectedGlobal["Trail"] = [
    //     [[{"id":1},{"id":2},{"id":3}], [{"id":5}, {"id":1},6]],
    //     [ [{"id":12}, {"id":13}, {"id":14}], [{"id":15}, {"id":16}, {"id":17}, {"id":18}, {"id":19}, {"id":10}] ]
    // ];

    // fetch_guided_tsne_result(arc_info.constraints);
}
// /------------------ data update of class selection ----------------------------/


function true_label(id){
    return TrueLabels[id];
}

function update_validated_labels(data){
    for(var i in data){
        ValidatedLabel[i] = data[i];
    }
}

function get_validated_labels(i){
    return ValidatedLabel[i];
}

function update_spammer_labels(data){
    for(var i in data){
        if(!(i in ValidatedSpammer) ){
            ValidatedSpammer[i] = [];
        }
        for (var j = 0; j < data[i].length; j++){
            if(!(ValidatedSpammer[i].includes(data[i][j]))){
                ValidatedSpammer[i].push(data[i][j]);
            }
        }
    }
}

function get_spammer_this_round(id){

}

function dog_group_acc(){
    var group_num = [0,0];
    var group_correct = [0,0];
    var k = 0;
    for( var i = 0; i < TrueLabels.length; i++){
        k = parseInt(TrueLabels[i] / 2);
        group_num[k] ++;
        if(TrueLabels[i] == PosteriorLabels[i]){
            group_correct[k] ++ ;
        }
    }
    console.log(group_num,group_correct);
}