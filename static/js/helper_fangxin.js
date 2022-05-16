/**
 * Created by 欧阳方昕 on 2017/11/24.
 */

function workers_processing() {
    var workers = [];
    var instancesIndex = SelectedGlobal.InstanceIndex;
    var colors = [];
    var labelNames = [];
    var selectedIndex = {};
    for (var i = 0; i < SelectedList.length; i++){
        colors.push(CategoryColor[SelectedList[[i]]]);
        labelNames.push(LabelNames[SelectedList[[i]]]);
        selectedIndex[SelectedList[i]] = i;
    }
    colors.push("#cccccc");
    labelNames.push("Others");
    selectedIndex["other"] = SelectedList.length;
    for (var i = 0; i < SpammerScore.length; i++){
        var instances = [[]];
        for (var j = 0; j < SelectedList.length; j++){
            instances.push([]);
        }
        for (var j = 0; j < WorkerLabels.length; j++){
            if (!instancesIndex.includes(j)){
                continue;
            }

            if (SelectedList.includes(WorkerLabels[j][i])){
                instances[selectedIndex[WorkerLabels[j][i]]].push(j);
            } else if (WorkerLabels[j][i] != -1){
                instances[selectedIndex.other].push(j);
            }
        }
        var total = 0;
        for (var j = 0; j < instances.length; j++){
            total += instances[j].length;
        }
        workers.push({
            'id': i,
            'spammer_score': SpammerScore[i],
            'reliability': WorkerAccuracy[i],
            'instances': instances,
            'weight': total
        })
    }
    return workers;
}

function instances_processing() {
    var instances = [];
    var instancesIndex = SelectedGlobal.InstanceIndex;
    var colors = [];
    var labelNames = [];
    var selectedIndex = {};
    for (var i = 0; i < SelectedList.length; i++){
        colors.push(CategoryColor[SelectedList[[i]]]);
        labelNames.push(LabelNames[SelectedList[[i]]]);
        selectedIndex[SelectedList[i]] = i;
    }
    colors.push("#cccccc");
    labelNames.push("Others");
    selectedIndex["other"] = SelectedList.length;
    for (var i = 0; i < InstanceTotalNum; i++){
        var worker = [[]];
        for (var j = 0; j < SelectedList.length; j++){
            worker.push([]);
        }
        for (var j = 0; j < WorkerTotalNum; j++){
            if (SelectedList.includes(WorkerLabels[i][j])){
                worker[selectedIndex[WorkerLabels[i][j]]].push(j);
            } else if (WorkerLabels[i][j] != -1){
                worker[selectedIndex.other].push(j);
            }
        }
        var total = 0;
        for (var j = 0; j < worker.length; j++){
            total += worker[j].length;
        }
        if (!instancesIndex.includes(i)){
            instances.push({
                'id': i,
                'url': ImageUrl[i],
                'trueLabel': PosteriorLabels[i],
                'label': selectedIndex.other,
                'uncertainty': Uncertainty[i],
                'workers': worker,
                'weight': total
            })
        } else {
            instances.push({
                'id': i,
                'url': ImageUrl[i],
                'trueLabel': PosteriorLabels[i],
                'label': selectedIndex[PosteriorLabels[i]],
                'uncertainty': Uncertainty[i],
                'workers': worker,
                'weight': total
            })
        }

    }
    return instances;
}

function getTopSimInstance(i, n) {
    var arr = SimilarityMatrix[i];
    var sort_arr = Array.from({length:arr.length},(item,index)=>index)
    sort_arr.sort(function (a, b) {
        return arr[b] - arr[a];
    })

    return sort_arr.slice(1, n+1);
}