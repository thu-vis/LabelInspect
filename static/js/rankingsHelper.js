
function update_ranking_view() {
    RankingsView.draw();
}

function ranking_instances_processing(data) {
    var sum = 0, min = 1, max = 0;
    for (var i = 0; i < data.length; i++){
        if(data[i].children){
            var score = ranking_instances_processing(data[i].children);
            sum += score[0];
            if (min > score[0]) min = score[0];
            if (max < score[0]) max = score[0];
            data[i].weight = score[0];
            data[i].min = score[1];
            data[i].max = score[2];
            data[i].expand = false;
        } else {
            var score = Uncertainty[data[i].id];
            sum += score;
            if (min > score) min = score;
            if (max < score) max = score;
            data[i].weight = score;
        }
        data[i].type = "instance";
    }
    return [sum / data.length, min, max];
}

function ranking_workers_processing(data) {
    var sum = 0, min = 1, max = 0;
    for (var i = 0; i < data.length; i++){
        if(data[i].children){
            var score = ranking_workers_processing(data[i].children);
            sum += score[0];
            if (min > score[0]) min = score[0];
            if (max < score[0]) max = score[0];
            data[i].weight = score[0];
            data[i].min = score[1];
            data[i].max = score[2];
            data[i].expand = false;
        } else {
            var score = SpammerScore[data[i].id];
            sum += score;
            if (min > score) min = score;
            if (max < score) max = score;
            data[i].weight = score;
        }
        data[i].type = "worker";
    }
    return [sum / data.length, min, max];
}

function get_leaf_node(data) {
    var nodes = []
    if (data.children){
        for (var i = 0; i < data.children.length; i++) {
            nodes = nodes.concat(get_leaf_node(data.children[i]));
        }
    } else {
        nodes.push(data.id);
    }
    return nodes;
}

function ranking_workers_instances_processing(source_list, target_list) {
    var ranking_link = [];
    for (var i = 0; i < source_list.length; i++){
        for (var j = 0; j < target_list.length; j++){
            var source_node = get_leaf_node(source_list[i]);
            var target_node = get_leaf_node(target_list[j]);
            var type = [0, 0, 0, 0];
            for (var k = 0; k < source_node.length; k++){
                var source = source_node[k];
                for (var h = 0; h < target_node.length; h++){
                    var target = target_node[h];
                    if (WorkerLabels[target][source] != -1){
                        type[WorkerLabels[target][source]] += 1;
                    }
                }
            }
            var sum = 0;
            for (var m = 0; m < type.length; m++){
                sum += type[m];
            }
            if (sum > 0){
                ranking_link.push({
                    'source': source_list[i],
                    'target': target_list[j],
                    'weight': sum,
                    'type': type
                })
            }
        }
    }
    return ranking_link;
}

