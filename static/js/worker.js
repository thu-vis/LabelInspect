/**
 * Created by 欧阳方昕 on 2017/11/24.
 */

var Worker = function (container) {
    var that = this;
    var duration = AnimationDuration;
    var container = container;
    var outer_radius = 1;
    var flowscale = 600;
    var bbox = container.node().getBoundingClientRect();
    var width = 1300;
    //TODO : changjian
    var height = 350; //350; // 28 is title height
    var margin = {top: 20, right: 20, bottom: 30, left: 40};
    var plot_width = width - margin.left - margin.right;
    var plot_height = height - margin.top - margin.bottom;
    var base_data = null;
    var isLasso = false;
    var pie = d3.pie().value(function(d) { return d.instances.length; });
    var arrow = "m121.3,34.6c-1.6-1.6-4.2-1.6-5.8,0l-51,51.1-51.1-51.1c-1.6-1.6-4.2-1.6-5.8,0-1.6,1.6-1.6,4.2 0,5.8l53.9,53.9c0.8,0.8 1.8,1.2 2.9,1.2 1,0 2.1-0.4 2.9-1.2l53.9-53.9c1.7-1.6 1.7-4.2 0.1-5.8z"
    this.flowmapdata = [];


    var svg = container.append("svg")
        .attr("id", "worker-svg")
        .attr("width", bbox.width)
        .attr("height", bbox.height - 28)
        //TODO: changjian
        .attr("viewBox", "0, 0, 1300, 350")
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
    var defs = svg.append("defs")
        .append("clipPath")
        .attr("id", "clip")
        .append("rect")
        .attr("y", -margin.top)
        .attr("width", plot_width)
        .attr("height", height);
    var x = d3.scaleLinear()
        .domain([-0.1, 1.1])
        .range([0, plot_width]);

    //TODO : changjian
    var y = d3.scaleLinear()
        .domain([0.0, 1])
        .range([plot_height, 0]);

    var xAxis = d3.axisBottom().scale(x);
    var yAxis = d3.axisLeft().scale(y);

    var dotsize = d3.scaleLinear()
        .range([1, 20]);
    var threshold = 0;

    var axis_view = svg.append("g").attr("class", "worker-axis-view");
    var lasso_view = svg.append("g").attr("class", "lasso-view");
    var flow_view = svg.append("g").attr("class", "flow-view");
    var dot_view = svg.append("g").attr("class", "worker-node-view").style("clip-path", "url(#clip)");
    var dialog_view = svg.append("g").attr("class", "dialog-view");

    var dialogDraw = new WorkerDialog(dialog_view, plot_width, plot_height);

    this.resize = function () {
        bbox = container.node().getBoundingClientRect();
        container.select("#worker-svg")
            .attr("width", bbox.width)
            .attr("height", bbox.height - 28);
    }

    this.draw = function () {
        base_data = workers_processing();
        var max_worker_instance = 0;
        for (var i = 0; i < base_data.length; i++) {
            for (var j = 0; j < base_data[i].instances.length; j++){
                if (base_data[i].instances[j].length > max_worker_instance){
                    max_worker_instance = base_data[i].instances[j].length;
                }
            }
        }
        dialogDraw.max_worker_instance = max_worker_instance;
        that.create_dot();
        that.update_dot();
        that.remove_dot();
        // that.create_axis();
        // that.update_axis();
    }

    this.redraw = function () {
        that.draw_selected([]);
        on_worker_selected([]);
        that.remove_flowmap();
        dialogDraw.removeAll();
    }

    this.get_container = function () {
        return svg;
    }

    this.create_axis = function (){

        var zoom = d3.zoom().scaleExtent([1,10])
            .on("zoom", zoomed);

        svg.call(zoom)
            .call(zoom.transform, d3.zoomIdentity
                .scale(1));
        function zoomed() {
            var t = d3.event.transform, xt = t.rescaleX(x);
            svg.selectAll(".worker-node").attr("transform", function (d) {
                return "translate(" + xt(d.spammer_score) + "," + y(d.reliability) + ")"
            });
            flow_view.selectAll(".worker-flow-link").attr("d", function (d) {
                var path = d3.path();
                path.moveTo(xt(d[0].split(",")[0] / flowscale), y(d[0].split(",")[1] / flowscale));
                path.quadraticCurveTo(xt(d[1].split(",")[0] / flowscale), y(d[1].split(",")[1] / flowscale), xt(d[2].split(",")[0] / flowscale), y(d[2].split(",")[1] / flowscale));
                return path.toString();
            })
            svg.select(".axis.x").call(xAxis.scale(xt));
        }
        axis_view.style("opacity", 0)
        axis_view.append("rect")
            .attr("width",plot_width)
            .attr("height",plot_height)
            .style("opacity",0);
        axis_view.append("g")
            .attr("class", "x axis")
            .attr("transform", "scale(0) translate(0," + plot_height + ")")
            .append("text")
            .attr("class", "label")
            .attr("x", 0)
            .attr("y", -6)
            .attr("x", plot_width)
            .style("text-anchor", "end")
            .style("font-size", 12)
            .text("Spammer Score");

          axis_view.append("g")
              .attr("class", "y axis")
              .attr("transform", "scale(0)")
            .append("text")
              .attr("class", "label")
              .attr("transform", "rotate(-90)")
              .attr("y", 6)
              .attr("dy", ".71em")
              .style("text-anchor", "end")
              .style("font-size", 12)
              .text("Accuracy");
    }

    this.update_axis = function () {
         var animation = axis_view.transition().duration(duration);
         animation.style("opacity", 1)
         animation.select(".x.axis")
             .attr("transform", "scale(1) translate(0," + plot_height + ")")
             .call(xAxis)
             .select("text")
             .attr("x", plot_width);

        animation.select(".y.axis").attr("transform", "scale(1)").call(yAxis)
    }

    this.remove_axis = function () {
        axis_view.transition().duration(duration).style("opacity", 0).remove();
    }

    this.create_dot = function () {
        var spammer_list = [57, 28, 1, 2, 25, 30, 13, 55, 27, 34, 15, 21, 9, 36, 14];
        // var spammer_list = [4, 8, 11, 14, 19, 21, 23, 24, 28, 36, 44, 45, 51 ]; // dog 1_2
        // var spammer_list = [12, 24, 30, 41, 45, 47]; //dog 3_4
        that.workers = dot_view.selectAll(".worker-node")
            .data(base_data, function (d) { return "worker" + d.id; });

        var workers = that.workers.enter().append('g')
            .attr("id", function (d) { return "worker" + d.id; })
            .attr("class", "worker-node")
            .style("opacity", 0)
            .attr("transform", function (d) {
                    return "translate(" + x(d.spammer_score) + "," + y(d.reliability) + ")"
                });
        workers.append("circle")
            .style("opacity", 0)
            .attr("class", "dot")
            .style("fill", function (d) {
                return "#7f7f7f";
            })
            .attr("r", 0);
        workers.append("circle")
            .style("opacity", 0)
            .attr("class", "dot_total")
            .style("stroke", function (d) {
                return "#7f7f7f";
            })
            .attr("r", 0);

        workers.append("g")
            .attr("class", "dot_arrow")
            .style("opacity", 0)
            .style("fill","#7f7f7f")
            .append("path")
            .attr("d", arrow)
            .attr("transform","translate(-13, 0) scale(0.2)")

        workers.on("mouseover", on_worker_hover)
            .on("mouseout", on_worker_unhover)
            .on("click", on_worker_clicked);

        that.workers = dot_view.selectAll(".worker-node")
            .data(base_data, function (d) { return "worker" + d.id; });

        if (isLasso) {
            that.lasso.items(that.workers)
        }
    }

    this.update_dot = function () {
        dotsize.range([1, 20]).domain([1, SelectedGlobal.InstanceTotalNum]);
        var animation = that.workers.transition().duration(duration).style("opacity", 1).attr("transform", function (d) {
            return "translate(" + x(d.spammer_score) + "," + y(d.reliability) + ")"
        });
        animation.select(".dot").attr("r", function (d) {
            if (SpammerMenu.spammer_list.hasOwnProperty(d.id)) {
                var true_weight = d.weight;
                for (var i = 0; i < d.instances.length; i++){
                    if (SpammerMenu.spammer_list[d.id].indexOf(SelectedList[i]) >= 0){
                        true_weight -= d.instances[i].length;
                    }
                }
                return dotsize(true_weight);
            }
            return dotsize(d.weight);
        }).style("opacity", 1);
        animation.select(".dot_total").attr("r", function (d) { return dotsize(d.weight); }).style("opacity", 1);
        that.workers.select(".dot_arrow").style("opacity", function (d) {
            if (SelectedGlobal.ChangedWorker.hasOwnProperty(d.id)){
                return 1;
            } else {
                return 0;
            }
        })
        .attr("transform", function (d) {
            if (SelectedGlobal.ChangedWorker.hasOwnProperty(d.id)){
                var node = SelectedGlobal.ChangedWorker[d.id];
                var dist = 0, deg = 0, dx = node.dx / 1.2 * plot_width, dy = node.dy * plot_height;
                if (node.dx >= 0 && node.dy >= 0){
                    dist = dx*dx + dy*dy;
                    deg = Math.acos(dy/Math.sqrt(dist,0.5)) / Math.PI * 180 - 180;
                }
                else if (node.dx >= 0 && node.dy < 0){
                    dy = -dy;
                    dist = dx*dx + dy*dy;
                    deg = -Math.acos(dy/Math.sqrt(dist,0.5)) / Math.PI * 180;
                }
                else if (node.dx < 0 && node.dy >= 0){
                    dist = dx*dx + dy*dy;
                    deg = -Math.acos(dy/Math.sqrt(dist,0.5)) / Math.PI * 180 - 180;
                }
                else{
                    dy = -dy;
                    dist = dx*dx + dy*dy;
                    deg = Math.acos(dy/Math.sqrt(dist,0.5)) / Math.PI * 180 - 360;
                }
                return "rotate(" + deg + ")";
            } else {
                return "";
            }
        }).select("path").attr("transform", function (d) {
            return "translate(-13, " + (dotsize(d.weight) - 8) + ") scale(0.2)"
        });
        that.workers.attr("class", function (d) {
                if (SpammerList.indexOf(d.id) >= 0){
                    return "worker-node spammer_node";
                }
                return "worker-node"
            })

    }

    this.remove_dot = function () {
        that.workers.exit().transition().duration(duration).style("opacity", 0).remove();
    }

    this.draw_lasso = function () {
        // Lasso functions to execute while lassoing
        var lasso_start = function() {
            svg.selectAll(".selected").classed("selected", false);
            svg.selectAll(".unselected").classed("unselected", false);
          that.lasso.items().classed("selected", false).classed("unselected", true); // style as not possible
        };

        var lasso_draw = function() {
            that.lasso.possibleItems().classed("selected", function (d) {
                if (d.isShown) {
                    return true;
                }
                return false;
            }).classed("unselected", function (d) {
                if (d.isShown) {
                    return false;
                }
                return true;
            });
            that.lasso.notPossibleItems().classed("unselected", true).classed("selected", false);
        };

        var lasso_end = function() {
            var isselected = false;
            that.lasso.selectedItems().classed("selected", function (d) {
                if (d.isShown) {
                    isselected = true;
                    return true;
                }
                return false;
            }).classed("unselected", function (d) {
                if (d.isShown) {
                    return false;
                }
                return true;
            });
            that.lasso.notSelectedItems().classed("unselected", true).classed("selected", false);
            if (!isselected){
                that.lasso.notSelectedItems().classed("unselected", false).classed("selected", false);
            }
        };

        // Create the area where the lasso event can be triggered
        var lasso_area = svg.append("rect")
                              .attr("width",plot_width)
                              .attr("height",plot_height)
                              .style("opacity",0);

        // Define the lasso
        that.lasso = d3.lasso()
            .closePathSelect(true)
            .closePathDistance(100)
            .targetArea(lasso_area)
            .on("start",lasso_start)
            .on("draw",lasso_draw)
            .on("end",lasso_end);

        // Init the lasso on the svg:g that contains the dots
        svg.call(that.lasso);
    }

    this.draw_selected = function (ids) {
        var animation = dot_view.selectAll(".worker-node").transition().duration(duration);
        if (ids.length == 0){
            animation.selectAll(".dot").style('opacity', function (d) {
                    d.isShown = true;
                    return 1;
                })
                .on("start", function(){
                    d3.select(this).style("display", "block");
                })
            animation.selectAll(".dot_total").style('opacity', 1)
                .on("start", function(){
                    d3.select(this).style("display", "block");
                })
            animation.selectAll(".worker-pie").style("opacity", 0).remove();
            flow_view.selectAll("g").transition().duration(duration).style('opacity', 1);
        } else {
            dot_view.selectAll(".worker-node").selectAll(".worker-pie").style("opacity", 0).remove();
            dotsize.range([5, 20]).domain([1, ids.length]);
            var pies = dot_view.selectAll(".worker-node").selectAll(".worker-pie").data(function (d) {
                    var data = [];
                    var count = 0;
                    for (var i = 0; i < d.instances.length; i++) {
                        if (SpammerMenu.spammer_list.hasOwnProperty(d.id) && SpammerMenu.spammer_list[d.id].indexOf(SelectedList[i]) >= 0) {
                            continue;
                        }
                        var instances = d.instances[i].filter(v => ids.includes(v));
                        count += instances.length;
                        data.push({
                            "parentid": d.id,
                            "id": i,
                            "label": i,
                            "instances": instances,
                            "r": dotsize(d.weight)
                        });
                    }
                    for (var i = 0; i < data.length; i++){
                        data[i].r = dotsize(count);
                    }
                    if (count > 0) {
                        d.isShown = true;
                    }else {
                        d.isShown = false;
                    }
                    return pie(data);
                })
                .enter().append("g")
                .attr("class", "worker-pie")
                .attr("id", function (d) {
                    return "worker-pie" + d.data.parentid + "-" + d.data.id;
                })
                .style("opacity", 0)
                .style("fill", function(d) { return SelectedGlobal.CategoryColor[d.data.label]; });
            pies.append("path")
                .attr("d", function (d) {
                    return d3.arc()
                        .outerRadius(outer_radius * d.data.r)
                        .innerRadius(0)(d);
                });
            animation.selectAll(".dot").style('opacity', 0).on("end", function(){
                d3.select(this).style("display", "none");
            })
            animation.selectAll(".dot_total").style('opacity', 0).on("end", function(){
                d3.select(this).style("display", "none");
            })
            animation.selectAll(".worker-pie").style('opacity', 1)
                .on("start", function(d){
                    if (d.data.instances.length == 0) {
                        d3.select(this).style("display", "none");
                    }
                });
            flow_view.selectAll("g").transition().duration(duration).style('opacity', 0);
        }
    }
    this.set_selected = function (ids) {
        if (ids.length != 0) {
            svg.selectAll(".worker-node").classed("unselected", true);
        } else {
            svg.selectAll(".worker-node").classed("unselected", false);
        }
        svg.selectAll(".worker-node").classed("selected", false);
        for (var i = 0; i < ids.length; i++){
            svg.select("#worker" + ids[i]).classed("unselected", false);
            svg.select("#worker" + ids[i]).classed("selected", true);
        }
        //that.set_dehighlight(indexs);
    }
    this.set_highlight = function (indexs) {
        // svg.selectAll(".worker-node").classed("selected", false);
        // for (var i = 0; i < indexs.length; i++){
        //     svg.select("#worker" + indexs[i]).classed("selected", true);
        // }
        // TODO
        if (indexs.length > 1){
            dot_view.selectAll(".worker-node").sort(function (a, b) {
                if (indexs.indexOf(a.id) >= 0){
                    return 1;
                }
                return -1;
            })
        }

        that.set_dehighlight(indexs);
    }
    this.set_dehighlight = function (indexs) {
        if (indexs.length == 0){
            svg.selectAll(".dot").classed("dehighlight", false);
            svg.selectAll(".dot_arrow").classed("dehighlight", false);
            svg.selectAll(".worker-pie").classed("dehighlight", false);
        }
        else {
            svg.selectAll(".dot").classed("dehighlight", true);
            svg.selectAll(".dot_arrow").classed("dehighlight", true);
            svg.selectAll(".worker-pie").classed("dehighlight", true);
            for (var i = 0; i < indexs.length; i++) {
                svg.select("#worker" + indexs[i]).select(".dot").classed("dehighlight", false);
                svg.select("#worker" + indexs[i]).select(".dot_arrow").classed("dehighlight", false);
                svg.select("#worker" + indexs[i]).selectAll(".worker-pie").classed("dehighlight", false);
            }
        }
    }
    this.set_highlight_by_label = function (indexs) {
        if (!indexs){
            return;
        }
        var index_list = []
        svg.selectAll(".dot").style("fill", "");
        for (var i = 0; i < indexs.length; i++){
            for (var j = 0; j < indexs[i].length; j++){
                svg.select("#worker" + indexs[i][j]).select(".dot").style("fill", SelectedGlobal.CategoryColor[i]);
                //svg.select("#worker" + indexs[i][j]).select("#worker-pie" + indexs[i][j] + "-" + i).style("fill", CategoryColor[i]);
            }
            index_list = index_list.concat(indexs[i]);
        }
        //TODO
        dot_view.selectAll(".worker-node").sort(function (a, b) {
            if (index_list.indexOf(a.id) >= 0){
                return 1;
            }
            return -1;
        })
        that.set_dehighlight_by_label(indexs);
    }
    this.set_dehighlight_by_label = function (indexs) {
        if (indexs.length == 0){
            svg.selectAll(".dot").classed("dehighlight", false);
            svg.selectAll(".dot_arrow").classed("dehighlight", false);
            svg.selectAll(".worker-pie").classed("dehighlight", false);
        }
        else {
            svg.selectAll(".dot").classed("dehighlight", true);
            svg.selectAll(".worker-pie").classed("dehighlight", true);
            for (var i = 0; i < indexs.length; i++){
                for (var j = 0; j < indexs[i].length; j++){
                    svg.select("#worker" + indexs[i][j]).select(".dot").classed("dehighlight", false);
                    svg.select("#worker" + indexs[i][j]).select(".dot_arrow").classed("dehighlight", false);
                    svg.select("#worker" + indexs[i][j]).select("#worker-pie" + indexs[i][j] + "-" + i).classed("dehighlight", false);
                }
            }
        }

    }
    this.is_selected = function (id) {
        if (svg.select("#worker" + id).classed("selected")) {
            return true;
        }
        return false;
    }

    this.draw_dialog = function (d) {
        dialogDraw.create(d, x(d.spammer_score), y(d.reliability));
    }
    this.update_dialog = function (id) {
        dialogDraw.update(id);
    }
    this.draw_flowmap = function (data) {
        that.flowmapdata = data;
        that.remove_flowmap();
        that.update_dot();
        var flowmaps = {};
        for (var i = 0; i < data.length; i++){
            var node = data[i];
            var source = {"x": base_data[node.id].spammer_score * flowscale, "y": base_data[node.id].reliability * flowscale, "id": node.id};
            var targets = [];
            var influence = [];
            for (var j = 0; j < node.influence.length; j++){
                var node2 = node.influence[j];
                if (node2.weight > threshold) {
                    influence.push(node2.id);
                    var target = {"x": base_data[node2.id].spammer_score * flowscale, "y": base_data[node2.id].reliability * flowscale, "weight": node2.weight};
                    targets.push(target);
                }
            }
            if (targets.length > 0  ){
                flowmaps[i] = {
                    'source': source,
                    'target': targets
                };
            }
        }
        // http://visgroup.thss.tsinghua.edu.cn:30075/
        $.post("http://visgroup.thss.tsinghua.edu.cn:30075/api/flowmap/layout", flowmaps ,function(response){
            // var flowedge = [];
            // for (var i = 0; i < response.length; i++){
            //     flowedge = flowedge.concat(response[i]);
            // }
            if (response == ""){
                response = [];
            }
            var flows = flow_view.selectAll("g").data(response).enter().append("g")
            .attr("id", function (d, i) {
                return "worker-flow-link" + flowmaps[i].source.id;
            })
             flows.selectAll(".worker-flow-link").data(function (d) {
                 return d;
             }).enter().append("path")
            .attr("class", "worker-flow-link")
            .attr("d", function (d) {
                var path = d3.path();
                path.moveTo(x(d[0].split(",")[0] / flowscale), y(d[0].split(",")[1] / flowscale));
                path.quadraticCurveTo(x(d[1].split(",")[0] / flowscale), y(d[1].split(",")[1] / flowscale), x(d[2].split(",")[0] / flowscale), y(d[2].split(",")[1] / flowscale));
                return path.toString();
            }).style("stroke-width", function (d) {
                return d[3].split(",")[0];
            })
            .style("opacity", 0)
            .transition().duration(duration).style("opacity", 1);
        });
    }
    this.update_flowmap = function () {
        flow_view.selectAll(".worker-flow-link").attr("d", function (d) {
            var path = d3.path();
            path.moveTo(x(d[0].split(",")[0] / flowscale), y(d[0].split(",")[1] / flowscale));
            path.quadraticCurveTo(x(d[1].split(",")[0] / flowscale), y(d[1].split(",")[1] / flowscale), x(d[2].split(",")[0] / flowscale), y(d[2].split(",")[1] / flowscale));
            return path.toString();
        })
    }
    this.remove_flowmap = function () {
        flow_view.selectAll("g").remove();
    }
    this.highlight_flowmap = function (ids) {
        if (ids.length == 0){
            flow_view.selectAll(".worker-flow-link").style("stroke", "#ccc");
            that.set_highlight([]);
        }else {
            var flowids = [];
            for (var i = 0; i < that.flowmapdata.length; i++){
                flowids.push(that.flowmapdata[i].id);
            }
            flow_view.selectAll(".worker-flow-link").style("stroke", "#eee");
            var highlight_list = [];
            for (var i = 0; i < ids.length; i++){
                if (flowids.indexOf(ids[i].toString()) >= 0){
                    var node = that.flowmapdata[flowids.indexOf(ids[i].toString())];
                    highlight_list.push(ids[i]);
                    for (var j = 0; j < node.influence.length; j++){
                        highlight_list.push(node.influence[j].id);
                    }
                    flow_view.select("#worker-flow-link" + ids[i]).selectAll(".worker-flow-link").style("stroke", "#ccc");
                }
            }
            if (highlight_list.length > 0){
                that.set_highlight(highlight_list);
            } else {
                flow_view.selectAll(".worker-flow-link").style("stroke", "#ccc");
                that.set_highlight([]);
            }
        }
    }


    this.init = function () {
        if (isLasso) {
            that.draw_lasso();
        }
        that.create_axis();
        that.update_axis();
    }.call();
}
