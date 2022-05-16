/*
    some functions that communicate with backend.
*/



var tsne_update = function(){
    var d = {};
    d["state"] = "test";
    // d = InstancesView.get_tsne_update_input();
    $.ajax({
        type: "POST",
        url: TSNEUpdateApi,
        data: JSON.stringify(d),
        contentType: "application.json; charset=UTF-8",
        dataType: "json",
        success: function(data){
            FeedBack = data;
            InstancesView.update_tsne(data);
        },
        error: function(xhr, type){

        }
    })
};
