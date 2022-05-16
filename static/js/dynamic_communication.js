//-------------------------- Fangxin -------------------------------------//

//-------------------------- Changjian ----------------------------------//

// /----- update backend model using validation gathered so far -----/
var propagation = function(callback){
    /*
    * update according variables when backend send new data.*/
    d = {"propagation": "true"};
    $.ajax({
        type: "GET",
        url: PropagationApi,
        data: JSON.stringify(d),
        contentType: "application.json; charset=UTF-8",
        dataType: "json",
        success: function(data){
            console.log("propagation success!!");
            if(callback) {
                callback(data);
            }
        },
        error: function(xhr, type){
            console.log("propagation handler error!!!!!!!!!!!!");
        }
    })
};

var more_instances = function(d,callback){
    $.ajax({
        type: "POST",
        url: "/api/more-instances",
        data: JSON.stringify(d),
        contentType: "application.json; charset=UTF-8",
        dataType: "json",
        success: function(data){
            if(callback) {
                callback(data);
            }
        },
        error: function(xhr, type){
            console.log("propagation handler error!!!!!!!!!!!!");
        }
    })
};

// var update_propagation = function(){
//     d = {"propagation": "true"};
//     $.ajax({
//         type: "GET",
//         url: PropagationApi,
//         data: JSON.stringify(d),
//         contentType: "application.json; charset=UTF-8",
//         dataType: "json",
//         success: function(data){
//             console.log("propagation success!!");
//             callback(data);
//         },
//         error: function(xhr, type){
//             console.log("propagation handler error!!!!!!!!!!!!");
//         }
//     })
// };

// /----- update backend model using validation gathered so far -----/

var instance_candidate_selection = function(d, callback){
    if( !d ){
    // if( !0 ){
        d = {};
        d["validated_instances"] = {259:1, 254:1};
        d["validated_spammers"] = {2:[0,1]};
    }
    d["selected_list"] = SelectedList;
    $.ajax({
        type: "POST",
        url: InstanceCandidateSelectionApi,
        data: JSON.stringify(d),
        contentType: "application.json; charset=UTF-8",
        dataType: "json",
        success: function(data){
            console.log("instance candidate selection update!!!");
            callback(data);
        },
        error: function(xhr, type){
            console.log("Error!!!!!!!!!: instance candidate selection!!!!!");
        }
    })
};

var worker_candidate_selection = function(d, callback){
    if( !d ){
    // if( !0 ){
        d = {};
        d["validated_instances"] = {259:1, 254:1};
        d["validated_spammers"] = {2:[0,1]};
    }
    d["selected_list"] = SelectedList;

    $.ajax({
        type: "POST",
        url: WorkerCandidateSelectionApi,
        data: JSON.stringify(d),
        contentType: "application.json; charset=UTF-8",
        dataType: "json",
        success: function(data){
            console.log("worker candidate selection update!!!");
            FeedBack = data;
            SpammerList = data["SpammerList"];
            InstanceList = data["InstanceList"];
            callback(data);
        },
        error: function(xhr, type){
            console.log("Error!!!!!!!!!!!!:worker candidate selection!!!!!");
        }
    })
};

var cropping_info = function(d, callback){
    if( !d ){
    // if( !0 ){
        d = {};
        d["id"] = 1;
        d["up_left"] = {
            "x":0.3,
            "y":0.3
        };
        d["down_right"] = {
            "x":0.7,
            "y":0.7
        };
    }
    d["dataname"] = DatasetName;
    $.ajax({
        type: "POST",
        url: CroppingApi,
        data: JSON.stringify(d),
        contentType: "application.json; charset=UTF-8",
        dataType: "json",
        success: function(data){
            console.log("cropping info!!!");
            FeedBack = data;
        },
        error: function(xhr, type){
            console.log("Error!!!!!!!!!!!!: cropping info!!!!!");
        }
    })
};

var roll_back = function(d, callback){
     if( !d ){
    // if( !0 ){
        d = {};
        d["validated_instances"] = {0:1, 1:0};
        d["validated_spammers"] = {};
        d["selected_list"] = SelectedList;
        d["seed"] = 0;
    }
    $.ajax({
        type: "POST",
        url: RollBackApi,
        data: JSON.stringify(d),
        contentType: "application.json; charset=UTF-8",
        dataType: "json",
        success: function(data){
            console.log("roll back successes!!!");
            // FeedBack = data;
            if(callback){
                callback(data);
            }
        },
        error: function(xhr, type){
            alert("roll back error!!");
        }
    })
};


// /----- send validation information to backend ---------------/
var validation_updating = function(d, callback){
    // Here is a demo data structure that will be sent to backend
    if( !d ){
    // if( !0 ){
        d = {};
        d["validated_instances"] = {0:1, 1:0};
        d["validated_spammers"] = {2:[0,1]};
        d["selected_list"] = SelectedList;
    }
    $.ajax({
        type: "POST",
        url: ExpertInfoApi,
        data: JSON.stringify(d),
        contentType: "application.json; charset=UTF-8",
        dataType: "json",
        success: function(data){
            console.log("validation successfully update!!!");
            // FeedBack = data;
            if(callback){
                callback();
            }
        },
        error: function(xhr, type){

        }
    })
};
// /----- send validation information to backend ---------------/

// for debug purpose
var validation_updating_2 = function(){
    // Here is a demo data structure that will be sent to backend
    var d = {};
    d["validated_instances"] = {7:1, 5:0};
    d["validated_spammers"] = {3:[2,3],2:[2,3]};

    $.ajax({
        type: "POST",
        url: ExpertInfoApi,
        data: JSON.stringify(d),
        contentType: "application.json; charset=UTF-8",
        dataType: "json",
        success: function(data){
            FeedBack = data;
        },
        error: function(xhr, type){

        }
    })
};
// for debug purpose
var validation_updating_3 = function(){
    // Here is a demo data structure that will be sent to backend
    var d = {};
    d["validated_instances"] = {123:1, 42:2};
    d["validated_spammers"] = {5:[0,1]};

    $.ajax({
        type: "POST",
        url: ExpertInfoApi,
        data: JSON.stringify(d),
        contentType: "application.json; charset=UTF-8",
        dataType: "json",
        success: function(data){
            FeedBack = data;
        },
        error: function(xhr, type){

        }
    })
};

// /------- send a request to backend for guided tsne result /
// /------- according to selected list ----------------------/
function fetch_guided_tsne_result(callback){
    var d = {};
    // d["constraints"] = constraints_processor(SelectedGlobal["InstanceIndex"],
    //     SelectedGlobal["PosteriorLabels"],
    //     SelectedGlobal["ConstraintsPoints"]);
    d["center_points"] = SelectedGlobal["ConstraintsPoints"];
    d["dataset"] = DatasetName;
    d["selected_list"] = SelectedList;
    $.ajax({
        type: "POST",
        url: TSNEUpdateApi,
        data: JSON.stringify(d),
        contentType: "application.json; charset=UTF-8",
        dataType: "json",
        success: function(data){
            guided_tsne_handler(data);
            // InstancesView.update_tsne(data);
            if(callback){
                callback();
            }
        },
        error: function(xhr, type){
            console.log("fetch guided tsne result handler error!!!!!!!!!!!!");
        }
    })
}

function save_trail(callback){
    var d = {};
    d["TrailLog"] = TrailLog;
    d["TrailSpammer"] = TrailSpammer;
    d["SelectedList"] = SelectedList;
    d["SeedLog"] = SeedLog;
    $.ajax({
        type: "POST",
        url: "/api/save-trail",
        data: JSON.stringify(d),
        contentType: "application.json; charset=UTF-8",
        dataType: "json",
        success: function(data){
            if(callback){
                callback();
            }
        },
        error: function(xhr, type){
            alert("save trail error");
        }
    })
}

function load_trail(callback){
    var d = {};
    d["SelectedList"] = SelectedList;
    $.ajax({
        type: "POST",
        url: "/api/load-trail",
        data: JSON.stringify(d),
        contentType: "application.json; charset=UTF-8",
        dataType: "json",
        success: function(data){
            TrailLog = data["TrailLog"];
            TrailSpammer = data["TrailSpammer"];
            // ValidatedSpammer = TrailSpammer[TrailSpammer.length-1];
            ValidatedSpammer = {};
            for( var i = 0; i < TrailSpammer.length; i++){
                for( var id in TrailSpammer[i]){
                    if(!(id in ValidatedSpammer)){
                        ValidatedSpammer[id] = TrailSpammer[i][id];
                    }
                    else{

                    }
                }
            }
            for( var k = 0; k < TrailSpammer.length; k++){
                for( var id in ValidatedSpammer){
                TrailSpammer[k][id] = ValidatedSpammer[id];
                }
            }


            SelectedList = data["SelectedList"];
            SeedLog = data["SeedLog"];
            if(callback){
                callback();
            }
        },
        error: function(xhr, type){
            alert("load trail error");
        }
    })
}

function incremental_tsne_result(){
    var d = {};
    // d["constraints"] = constraints_processor(SelectedGlobal["InstanceIndex"],
    //     SelectedGlobal["PosteriorLabels"],
    //     SelectedGlobal["ConstraintsPoints"]);
    d["center_points"] = SelectedGlobal["ConstraintsPoints"];
    d["dataset"] = DatasetName;
    d["selected_list"] = SelectedList;

    $.ajax({
        type: "POST",
        url: IncreTSNEUpdateApi,
        data: JSON.stringify(d),
        contentType: "application.json; charset=UTF-8",
        dataType: "json",
        success: function(data){
            FeedBack = data;
            // InstancesView.update_tsne(data);
        },
        error: function(xhr, type){
            console.log("fetch guided tsne result handler error!!!!!!!!!!!!");
        }
    })
}

function constraints_processor(selected_index, selected_posterior_labels, constraint_points){
    var constraints = {};
    //TODO: what if mixed instances
    for (var i = 0; i < selected_index.length; i++ ){
        var index = selected_index[i];
        constraints[index] = [];
        for ( var j = 0; j < constraint_points.length; j++ ){
            if( constraint_points[j].class == selected_posterior_labels[index] ){
                constraints[index].push(j);
            }
        }
    }
    return constraints;
}
// /------- send a request to backend for guided tsne result /
// /------- according to selected list ----------------------/


function instance_change_label_type(){
    if( TrueLabelOrPosteriorFlag%2 == 1){
        TrueLabelOrPosteriorFlag = TrueLabelOrPosteriorFlag + 1;
        InstancesView.draw();
        console.log("posterior");
    }
    else{
        TrueLabelOrPosteriorFlag = TrueLabelOrPosteriorFlag + 1;
        InstancesView.draw();
        console.log("true");
    }
}
