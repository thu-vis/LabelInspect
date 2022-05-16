//-------------------------- Fangxin -------------------------------------//


//-------------------------- Changjian ----------------------------------//

var update_data = function(){
    //TODO: can it be done that this function monitor some info sent from backend ?
    /*
    * update according variables when backend send new data.*/
};

var sent_data_to_backend = function(){
    //TODO
    var d = {};
    d["ExpertLabel"] = 100;
    d["isSpammer"] = "a";

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
