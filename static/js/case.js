function bird_case_sparrow_step_1(){
    var selected_list = [633, 1779, 334, 900, 1952, 1874, 1349, 1897, 194, 1277, 1197];
    var spammer_list = [0, 7, 50];
    var validated_instances = {};
    // for( var i = 0; i < selected_list.length; i++){
    //     validated_instances[selected_list[i]] = TrueLabels[selected_list[i]];
    // }
    var validated_spammers = {};
    for( var i = 0; i < spammer_list.length; i++){
        validated_spammers[spammer_list[i]] = [4,5];
    }
    var d = {};
    d["validated_instances"] = validated_instances;
    d["validated_spammers"] = validated_spammers;
    d["selected_list"] = [4,5];
    validation_updating(d);
}
function fake_trail_log(){
    var validated_size = [11, 5, 4, 4, 3];
    var changed_size = [ 58, 27, 14, 8, 2];
    var traillog = [];
    for( var i = 0; i < validated_size.length; i++){
        var new_validated_record = [];
        var new_changed_record = [];
        for(var j = 0; j < validated_size[i]; j++ ){
            new_validated_record.push({
                "id":j
            });
        }
        for(var j = 0; j < changed_size[i]; j++ ){
            new_changed_record.push({
                "id": j
            });
        }
        TrailLog.push([new_validated_record, new_changed_record])
    }

    SelectedGlobal["TrailLog"] = traillog;
    TrailView.draw();
}