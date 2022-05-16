/**
 * Created by 欧阳方昕 on 2018/1/15.
 */

var InstanceLabelMenu = function(filter, classNames) {
    var that = this;
    var classNames = classNames;
    this.label_list = {}

    this.create = function (classes) {
        var menu = new BootstrapMenu(filter,
        {
            fetchElementData: function ($rowElem) { //fetchElementData获取元数据
                var data = d3.select("#" + $rowElem.attr("id")).data()[0]; //获取表格数据
                return data; //return的目的是给下面的onClick传递参数
            },
            actions: that.createActionsByClassName(classes)
        });
    }

    this.createActionsByClassName = function(classes) {
        var actions = {};
        for (var i = 0; i < classes.length; i++){
            actions[classes[i] + "Row"] = {
                classId: i,
                name: '<font size=3 style="color:' + CategoryColor[i] + ';">' + classes[i] + '</font>',
                classNames: function (d, index) {
                    if (that.label_list.hasOwnProperty(d.id) && that.label_list[d.id] == this.classId)
                        return "menu-selected";
                    return "";
                },
                isShown: function(d, index) {
                    if (SelectedList.indexOf(this.classId) >= 0) {
                        return true;
                    }
                    return false;
                },
                onClick: function (d, index) {
                    if(that.label_list.hasOwnProperty(d.id) && that.label_list[d.id] == this.classId){
                        delete that.label_list[d.id];
                    } else {
                        that.label_list[d.id] = this.classId;
                    }
                    var label_item = {};
                    label_item[d.id] = this.classId;
                    instance_candidate_selection_update(label_item);
                }
            }
        }
        actions["OthersRow"] = {
            name: '<font size=3 style="color:' + Gray + ';">Others</font>',
            classNames: function (d) {
                if (that.label_list.hasOwnProperty(d.id) && that.label_list[d.id] == -1)
                    return "menu-selected";
                return "";
            },
            isShown: function(d) {
                if (SelectedList.length < LabelTotalNum) {
                    return true;
                }
                //TODO false
                return true;
            },
            onClick: function (d) {
                //TODO: fix others
                var truelabel = TrueLabels[d.id];
                if(that.label_list.hasOwnProperty(d.id) && that.label_list[d.id] == truelabel){
                    delete that.label_list[d.id];
                } else {
                    that.label_list[d.id] = truelabel;
                }
                var label_item = {};
                label_item[d.id] = truelabel;
                instance_candidate_selection_update(label_item);
            }
        }
        return actions;
    }
    this.destroy = function () {
        d3.selectAll(".bootstrapMenu").remove();
    }
    this.update = function (data) {
        that.label_list = data;
        //that.label_list = {};
    }
    this.init = function () {
        that.create(classNames);
    }.call();

}

var WorkerSpammerMenu = function(filter, classNames) {
    var that = this;
    var classNames = classNames;
    this.spammer_list = {}
    this.is_Labeled = false;
    this.worker_id = -1;
    this.create = function (classes) {
        var menu = new BootstrapMenu(filter,
        {
            fetchElementData: function ($rowElem) { //fetchElementData获取元数据
                var data = d3.select("#" + $rowElem.attr("id")).data()[0]; //获取表格数据
                return data; //return的目的是给下面的onClick传递参数
            },
            actions: that.createActionsByClassName(classes),
            closeEvent: function () {
                if (that.is_Labeled){
                    worker_candidate_selection_update(that.spammer_list, that.worker_id);
                    that.worker_id = -1;
                    that.is_Labeled = false;
                }

            }
        });
    }

    this.createActionsByClassName = function(classes) {
        var actions = {};
        for (var i = 0; i < classes.length; i++){
            actions[classes[i] + "Row"] = {
                classId: i,
                name: '<font size=3 style="color:' + CategoryColor[i] + ';">' + classes[i] + '</font>',
                iconClass: "workermenu" + i,
                classNames: function (d, index) {
                    if (that.spammer_list.hasOwnProperty(d.id) && that.spammer_list[d.id].indexOf(this.classId) >= 0){
                        d3.select(".workermenu" + this.classId).classed("glyphicon glyphicon-ok", true);
                    } else {
                        d3.select(".workermenu" + this.classId).classed("glyphicon glyphicon-ok", false);
                    }
                    return "";
                },
                isShown: function(d) {
                    if (SelectedList.indexOf(this.classId) >= 0) {
                        return true;
                    }
                    return false;
                },
                onClick: function (d) {
                    //event.preventDefault();
                    if (that.spammer_list.hasOwnProperty(d.id) && that.spammer_list[d.id].indexOf(this.classId) >= 0){
                        delete that.spammer_list[d.id].splice(that.spammer_list[d.id].indexOf(this.classId), 1);
                        d3.select(".workermenu" + this.classId).classed("glyphicon glyphicon-ok", false);
                    } else if(that.spammer_list.hasOwnProperty(d.id)){
                        d3.select(".workermenu" + this.classId).classed("glyphicon glyphicon-ok", true);
                        that.spammer_list[d.id].push(this.classId);
                    } else {
                        d3.select(".workermenu" + this.classId).classed("glyphicon glyphicon-ok", true);
                        that.spammer_list[d.id] = [this.classId];
                    }
                    that.is_Labeled = true;
                    that.worker_id = d.id;
                    // worker_candidate_selection_update(that.spammer_list);
                    return true;
                }
            }
        }
        // actions["OthersRow"] = {
        //     name: '<font size=3 style="color:' + Gray + ';">Others</font>',
        //     classNames: function (d) {
        //         if (that.spammer_list.hasOwnProperty(d.id) && that.spammer_list[d.id].indexOf(-1) >= 0){
        //             d3.select(".workermenuother").classed("glyphicon glyphicon-ok", true);
        //         } else {
        //             d3.select(".workermenuother").classed("glyphicon glyphicon-ok", false);
        //         }
        //         return "";
        //     },
        //     iconClass: "workermenuother",
        //     isShown: function(d) {
        //         if (SelectedList.length < LabelTotalNum) {
        //             return true;
        //         }
        //         //TODO false
        //         return true;
        //     },
        //     onClick: function (d) {
        //         if (that.spammer_list.hasOwnProperty(d.id) && that.spammer_list[d.id].indexOf(-1) >= 0){
        //             delete that.spammer_list[d.id].splice(that.spammer_list[d.id].indexOf(-1), 1);
        //             d3.select(".workermenuother").classed("glyphicon glyphicon-ok", false);
        //         } else if(that.spammer_list.hasOwnProperty(d.id)){
        //             d3.select(".workermenuother").classed("glyphicon glyphicon-ok", true);
        //             that.spammer_list[d.id].push(-1);
        //         } else {
        //             d3.select(".workermenuother").classed("glyphicon glyphicon-ok", true)
        //             that.spammer_list[d.id] = [-1];
        //         }
        //         that.is_Labeled = true;
        //         that.worker_id = d.id;
        //         // worker_candidate_selection_update(that.spammer_list);
        //         return true;
        //     }
        // }
        return actions;
    }
    this.destroy = function () {
        d3.selectAll(".bootstrapMenu").remove();
    }
    this.update = function (data) {
        that.spammer_list = data;
        //that.spammer_list = {};
    }

    this.createSingleMenu = function(singleFilter){
        var menu = new BootstrapMenu(singleFilter,
        {
            fetchElementData: function ($rowElem) { //fetchElementData获取元数据
                var token = $rowElem.attr("id").split("_");
                var data = {
                    "id": parseInt(token[1]),
                    "class": parseInt(token[2])
                };
                return data; //return的目的是给下面的onClick传递参数
            },
            actions: {
                "SpammerRow": {
                    name: '<font size=3 style="color:' + Gray + ';">Mark As Spammer</font>',
                    classNames: function (d) {
                        if (that.spammer_list.hasOwnProperty(d.id) && that.spammer_list[d.id].indexOf(d.class) >= 0){
                            d3.select(".workermenuspammer").classed("glyphicon glyphicon-ok", true);
                        } else {
                            d3.select(".workermenuspammer").classed("glyphicon glyphicon-ok", false);
                        }
                        return "";
                    },
                    iconClass: "workermenuspammer",
                    onClick: function (d) {
                        if (that.spammer_list.hasOwnProperty(d.id) && that.spammer_list[d.id].indexOf(d.class) >= 0){
                            delete that.spammer_list[d.id].splice(that.spammer_list[d.id].indexOf(d.class), 1);
                            d3.select(".workermenuspammer").classed("glyphicon glyphicon-ok", false);
                        } else if(that.spammer_list.hasOwnProperty(d.id)){
                            d3.select(".workermenuspammer").classed("glyphicon glyphicon-ok", true);
                            that.spammer_list[d.id].push(d.class);
                        } else {
                            d3.select(".workermenuspammer").classed("glyphicon glyphicon-ok", true)
                            that.spammer_list[d.id] = [d.class];
                        }
                        that.worker_id = d.id;
                        worker_candidate_selection_update(that.spammer_list, d.id);
                    }
                }
            }
        });
    }


    this.init = function () {
        that.create(classNames);
    }.call();
}

