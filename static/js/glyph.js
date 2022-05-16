/**
 * Created by 欧阳方昕 on 2017/11/28.
 */

var InstanceGlyph = function (container, size, flow_container, button_container) {
    var that = this;
    var duration = AnimationDuration;
    var container = container;
    var width = size.width;
    var height = size.height;
    var px = size.px;
    var py = size.py;
    var svg = container;
    var isDebug = false;
    var isLasso = true;
    var glyphscale = width / 1000;
    var imagewidth = 100 * glyphscale;
    var outer_radius = 23;
    var inner_radius = 17;
    var center_radius = 12;
    var base_data = null;
    var arrow = "M286.935,69.377c-3.614-3.617-7.898-5.424-12.848-5.424H18.274c-4.952,0-9.233,1.807-12.85,5.424   C1.807,72.998,0,77.279,0,82.228c0,4.948,1.807,9.229,5.424,12.847l127.907,127.907c3.621,3.617,7.902,5.428,12.85,5.428   s9.233-1.811,12.847-5.428L286.935,95.074c3.613-3.617,5.427-7.898,5.427-12.847C292.362,77.279,290.548,72.998,286.935,69.377z"
    var lasso_view = svg.append("g").attr("class", "lasso-view");
    var flow_view = flow_container.append("g").attr("class", "flow-view");
    var glyph_view = svg.append("g").attr("class", "glyph-node-view");
    var clickTimer = null;
    var button1 = button_container.append("g").attr("class", "propagation-button")
            .style("display", "none")
            .attr("transform", "translate(" + (width - 180) + ", " + (height - 10) + ")")
            .on("click", function () {
                clearTimeout(clickTimer);
                clickTimer = setTimeout(propagation_update, 200);
            });
        button1.append("rect")
            .attr("x", -10)
            .attr("y", -24)
            .attr("width", 180)
            .attr("height", 30);
        button1.append("text").attr("class", "header-button")
            .attr("x", 15)
            .text("Propagation");

        var button2 = button_container.append("g").attr("class", "select-button")
            .style("display", "none")
            .attr("transform", "translate(" + (width - 180) + ", " + (height - 10) + ")")
            .on("click", function () {
                clearTimeout(clickTimer);
                clickTimer = setTimeout(matrix_selected, 200);
            });
        button2.append("rect")
            .attr("x", -10)
            .attr("y", -24)
            .attr("width", 180)
            .attr("height", 30);
        button2.append("text").attr("class", "header-button")
            .text("Show Instance");

    svg.append("defs").append("pattern")
        .attr("id", "diagonalHatch")
        .attr("patternUnits","userSpaceOnUse")
        .attr("width", 10)
        .attr("height", 10)
        .append("path")
        .attr("d","M0,10 l10,-10 M-2, 2l4, -4 M8,12l4,-4")
        .style("stroke", "dimgray");
    var pie = d3.pie().value(function(d) { return d.workers.length; }).sort(function (a, b) {return a.label - b.label;});
    var link = d3.linkVertical();
    var threshold = 0;

    var gain_list_dog = [] //[245, 251, 322, 393, 180, 796, 323, 591, 641, 201, 733, 255, 236, 697, 479, 306, 347,  33,  477,  458];
    var gain_list_bird = [] //[1925, 534, 976, 596, 263, 1156, 207, 356, 1935, 1332, 444, 1108, 1354, 1113, 1347, 258, 1236, 1558, 384, 1218, 1364, 1768, 27, 1226, 1961, 889, 1894, 107, 668, 361, 853, 1319, 1349, 1874, 1480, 1070, 984, 1619, 1640, 1536, 1105, 480, 235, 675, 1603, 926, 1354, 201, 1332, 984];
    var isVisDebug = false;
    this.glyph_dict = {};
    this.flowmapdata = [];
    this.flowmapdata = [];
    if (isDebug) {
        var constraint = [];
        var classlabel = 0;
        var plot_height = height;
        var rect = container.append("rect").attr("width", plot_height).attr("height", plot_height).style("fill-opacity", 0).attr("transform", "translate(0,0)");
        var bottom = container.append("text").attr("y", 20).text("Add Constraint " + classlabel);
        var clear = container.append("text").attr("y", 40).text("clear");
        clear.on("click", function () {
            constraint = [];
        })
        bottom.on("click", function () {
            if (classlabel < 3){
                classlabel += 1;
                bottom.text("Add Constraint " + classlabel);
            } else {
                fetch_guided_tsne_result(constraint);
                classlabel = 0;
                bottom.text("Add Constraint " + classlabel);
            }
        })
        var rect_center_x = plot_height / 2;
        var rect_center_y = plot_height / 2;
        rect.on("click", function () {
            var x = d3.event.offsetX - px;
            var y = d3.event.offsetY - py;
            var normal_x = x / plot_height;
            var normal_y = y / plot_height;
            constraint.push({
                "x": normal_x,
                "y": normal_y,
                "class": classlabel
            })
            // console.log(x, y, normal_x, normal_y, classlabel);
        })
    }


    this.get_radius = function () {
        return outer_radius;
    }

    this.set_selected = function (ids) {
        if (ids.length != 0) {
            svg.selectAll(".glyph-node").classed("unselected", true);
        } else {
            svg.selectAll(".glyph-node").classed("unselected", false);
        }
        svg.selectAll(".glyph-node").classed("selected", false);
        for (var i = 0; i < ids.length; i++){
            svg.select("#glyph" + ids[i]).classed("unselected", false);
            svg.select("#glyph" + ids[i]).classed("selected", true);
        }
    }

    this.set_highlight = function (ids) {
        // svg.selectAll(".glyph-arc").style("fill", function(d) { return SelectedGlobal.CategoryColor[d.data.label]; });
        // svg.selectAll(".glyph-circle").style("fill", function(d) { return SelectedGlobal.CategoryColor[d.label]; });
        // for (var i = 0; i < ids.length; i++){
        //     svg.select("#glyph" + ids[i]).selectAll(".glyph-arc").style("fill", function(d) { return d3.rgb(SelectedGlobal.CategoryColor[d.data.label]).darker(); });
        //     svg.select("#glyph" + ids[i]).select(".glyph-circle").style("fill", function(d) { return d3.rgb(SelectedGlobal.CategoryColor[d.label]).darker(); })
        // }
        that.set_dehighlight(ids);
    }
    this.set_dehighlight = function (ids) {
        if (ids.length == 0){
            svg.selectAll(".glyph-arc").classed("dehighlight", false);
            svg.selectAll(".glyph-circle").classed("dehighlight", false);
            svg.selectAll(".glyph-arrow").classed("dehighlight", false);
        } else {
            svg.selectAll(".glyph-arc").classed("dehighlight", true);
            svg.selectAll(".glyph-circle").classed("dehighlight", true);
            svg.selectAll(".glyph-arrow").classed("dehighlight", true);
            for (var i = 0; i < ids.length; i++){
                svg.select("#glyph" + ids[i]).selectAll(".glyph-arc").classed("dehighlight", false);
                svg.select("#glyph" + ids[i]).selectAll(".glyph-arrow").classed("dehighlight", false);
                svg.select("#glyph" + ids[i]).select(".glyph-circle").classed("dehighlight", false);
            }
        }
    }
    this.set_highlight_by_label = function (ids) {
        // svg.selectAll(".glyph-arc").style("fill", function(d) { return SelectedGlobal.CategoryColor[d.data.label]; });
        // svg.selectAll(".glyph-circle").style("fill", function(d) { return SelectedGlobal.CategoryColor[d.label]; });
        // for (var i = 0; i < ids.length; i++){
        //     for (var j = 0; j < ids[i].length; j++){
        //         svg.select("#glyph-arc" + ids[i][j] + "-" + i ).style("fill", function(d) { return d3.rgb(SelectedGlobal.CategoryColor[d.data.label]).darker(); });
        //         svg.select("#glyph" + ids[i][j]).select(".glyph-circle").style("fill", function(d) { return d3.rgb(SelectedGlobal.CategoryColor[d.label]).darker(); })
        //     }
        // }
        that.set_dehighlight_by_label(ids)
    }
    this.set_dehighlight_by_label = function (ids) {
        if (ids.length == 0){
            svg.selectAll(".glyph-arc").classed("dehighlight", false);
            svg.selectAll(".glyph-circle").classed("dehighlight", false);
            svg.selectAll(".glyph-arrow").classed("dehighlight", false);
        } else {
            svg.selectAll(".glyph-arc").classed("dehighlight", true);
            svg.selectAll(".glyph-circle").classed("dehighlight", true);
            svg.selectAll(".glyph-arrow").classed("dehighlight", true);
            for (var i = 0; i < ids.length; i++){
                for (var j = 0; j < ids[i].length; j++){
                    svg.select("#glyph-arc" + ids[i][j] + "-" + i ).classed("dehighlight", false);
                    svg.select("#glyph-arrow" + ids[i][j] + "-" + i ).classed("dehighlight", false);
                    svg.select("#glyph" + ids[i][j]).select(".glyph-circle").classed("dehighlight", false);
                }
            }
        }
    }
    this.is_selected = function (id) {
        if (svg.select("#glyph" + id).classed("selected")) {
            return true;
        }
        return false;
    }

    this.draw = function (data) {
        that.init();
        that.cur_data = [];
        for (var i = 0; i < data.length; i++){
            that.cur_data.push(base_data[data[i].id]);
            base_data[data[i].id].x = data[i].x;
            base_data[data[i].id].y = data[i].y;
            base_data[data[i].id].r = (data[i].uncertainty * 0.4 + 0.3) * glyphscale;
            base_data[data[i].id].arc_list = data[i].arc_list;
            base_data[data[i].id].last_round_label = data[i].last_round_label;
            base_data[data[i].id].is_labeled = data[i].is_labeled;
            if (that.glyph_dict.hasOwnProperty(data[i].id)) {
                base_data[data[i].id].imageX = that.glyph_dict[data[i].id].imageX;
                base_data[data[i].id].imageY = that.glyph_dict[data[i].id].imageX;
            } else {
                base_data[data[i].id].imageX = 0;
                base_data[data[i].id].imageY = 0;
            }
        }
        that.glyph_dict = {};
        for (var i = 0; i < data.length; i++) {
            that.glyph_dict[data[i].id] = base_data[data[i].id];
        }

        that.create();
        that.update();
        that.remove();
        if (!isDebug && isLasso){
            that.lasso.items(glyph_view.selectAll(".glyph-node"));
        }

    }
    this.single_draw = function (data, isdraw) {
        for (var i = 0; i < data.length; i++) {
            var node = data[i];
            if (isdraw){
                that.cur_data.push(base_data[node.id]);
                base_data[node.id].x = node.x;
                base_data[node.id].y = node.y;
                base_data[node.id].r = (node.uncertainty * 0.4 + 0.3) * glyphscale;
                base_data[node.id].arc_list = node.arc_list;
                base_data[node.id].is_labeled = node.is_labeled;
                that.glyph_dict[node.id] = base_data[node.id];
                // console.log("show" + node.id);
            } else {
                var index = that.cur_data.indexOf(that.glyph_dict[node.id]);
                that.cur_data.splice(index, 1);
                delete that.glyph_dict[node.id];
                // console.log("remove" + node.id);
            }
        }
        that.create();
        that.update(500, true);
        that.remove(500);
    }

    this.create = function () {
        that.glyphs = glyph_view.selectAll(".glyph-node").data(that.cur_data, function (d) {
            return "glyph" + d.id;
        });
        var glyphs = that.glyphs.enter().append("g").attr("id", function (d) {return "glyph" + d.id;})
            .attr("class", "instance-filter glyph-node")
            .style("opacity", 0)
            .attr("transform", function (d) {
                return "translate(" + d.x + "," + d.y + ") scale(" + d.r + ")";
            });
        that.glyphs = glyph_view.selectAll(".glyph-node").data(that.cur_data, function (d) {
            return "glyph" + d.id;
        });

        glyphs.on("click", on_glyph_clicked);

        glyphs.append("image")
            .style("opacity", function (d) {
                d.isShowImage = false;
                return 0;
            })
            .attr("width", 0)
            .attr("height", 0)
            .attr("x", 0)
            .attr("y", 0)
            .attr("class", "glyph-image")
            .attr("xlink:href", function (d) {
                return d.url;
            })
            .on("mouseover", on_glyph_center_hover)
            .on("mouseout", on_glyph_center_unhover);

        glyphs.append("circle").attr("class", "glyph-circle")
            .attr("r", function (d) { return center_radius; })
            .style("fill", function(d) { return SelectedGlobal.CategoryColor[d.label]; })
            .style("stroke", function(d) {
                if (d.last_round_label == -1){
                    return 'none';
                }
                return SelectedGlobal.CategoryColor[d.last_round_label];
            });


        glyphs.append("circle").attr("class", "glyph-center")
            .attr("r", function (d) { return inner_radius; })
            .on("mouseover", on_glyph_center_hover)
            .on("mouseout", on_glyph_center_unhover);

        var arcs = glyphs.append("g").attr("class", "glyph-arcview")
            .attr("id", function (d) {
                return "glyph-arcview" + d.id;
            }).selectAll(".glyph-arc").data(function (d) {
                var data = [];
                var maxindex = 0;
                var max = 10000;
                for (var i = 0; i < d.workers.length; i++){
                    if (d.workers[i].length < max && d.workers[i].length != 0 && i < d.arc_list.length){
                        maxindex = i;
                        max = d.workers[i].length;
                    }
                    data.push({
                        "parentid": d.id,
                        "id": i,
                        "label": i,
                        "workers": d.workers[i],
                        "r": d.r,
                    });
                }
                var piedata = pie(data);
                d.angle = d.arc_list[maxindex] - (piedata[maxindex].startAngle + piedata[maxindex].endAngle) / 2   / Math.PI * 180;
                return piedata;
            }, function (d) {
                return "glyph-arc" + d.parentid + "-" + d.id
            })
            .enter().append("g")
            .attr("class", "glyph-arc")
            .attr("id", function (d) {
                return "glyph-arc" + d.data.parentid + "-" + d.data.id;
            })
            .style("fill", function(d) { return SelectedGlobal.CategoryColor[d.data.label]; });


        glyphs.selectAll(".glyph-arcview").attr("transform", function (d) {
                return "rotate(" + d.angle + ")";
            })

        arcs.append("path")
            .attr("d", function (d) {
                return d3.arc()
                    .outerRadius(outer_radius)
                    .innerRadius(inner_radius)
                    .padAngle(0.03)(d);
            });


        arcs.on("mouseover", on_glyph_arc_hover)
            .on("mouseout", on_glyph_arc_unhover);

        glyphs.append("g").attr("class", "glyph-arrowview")
            .attr("id", function (d) {
                return "glyph-arrowview" + d.id;
            }) .selectAll(".glyph-arrow").data(function (d) {
                var data = [];
                for (var i = 0; i < d.arc_list.length; i++){
                    var isShown = true;
                    if (d.workers[i].length == 0){
                        isShown = false;
                    }
                    data.push({
                        "parentid": d.id,
                        "id": i,
                        "label": i,
                        "deg": d.arc_list[i]-180,
                        "isShown" : isShown
                    });
                }
                return data;
            })
            .enter().append("g")
            .attr("class", "glyph-arrow")
            .attr("id", function (d) {
                return "glyph-arrow" + d.parentid + "-" + d.id;
            })
            .attr("transform", function (d) {
                return "rotate(" + d.deg + ")";
            })
            .style("fill", function(d) { return SelectedGlobal.CategoryColor[d.label]; })
            .style("display", function (d) {
                if (d.isShown){
                    return "block";
                }
                return "none";
            })
            .append("path")
            .attr("class", "glyph-arrow-path")
            .attr("d", arrow);
        glyphs.append("circle")
            .attr("class", "glyph-labeled")
            .attr("r", outer_radius)
            .on("mouseover", on_glyph_center_hover)
            .on("mouseout", on_glyph_center_unhover);



        if (isVisDebug){
            glyphs.append("text")
                .attr("class", "glyph-id-text")
                .style("font-size", 30)
                .text(function (d) {
                    return d.id;
                })
            var labels = glyphs.append("g")
            .attr("class", "glyph-text-label");
            labels.append("rect");
            labels.append("text")
            .attr("dy", "1em")
            .text(function (d) {
                return d.uncertainty.toFixed(4);
            })
        }

    }
    this.update = function (t, nottick) {
        var glyphs = that.glyphs.enter().merge(that.glyphs);
        if (!nottick) {
            var simulation = d3.forceSimulation(glyphs.data())
                .alphaDecay(0.05)
                .alphaMin(0.05)
                .force("collide", d3.forceCollide().radius(function(d) { return (d.r + 0.3)*outer_radius; }))
                .on("end", ticked);
            function ticked() {//.transition().duration(100)
                var animation = glyphs.transition()
                    .duration(t ? t : duration)
                    .style("opacity", 1)
                    .style("stroke", function(d) {
                        var gain_list = gain_list_bird;
                        if (DatasetName == "dog"){
                            gain_list = gain_list_dog;
                        }
                        if (gain_list.indexOf(d.id) >= 0){
                            return 'black';
                        }
                        return "none";
                    })
                    .attr("transform", function (d) {
                        return "translate(" + d.x + "," + d.y + ") scale(" + d.r + ")";
                    });
                animation.select(".glyph-circle")
                .style("fill", function(d) { return SelectedGlobal.CategoryColor[d.label]; })
                .style("stroke", function(d) {
                    if (d.last_round_label == -1 || d.last_round_label == SelectedList[d.label]){
                        return 'none';
                    }
                    return CategoryColor[d.last_round_label];
                });

                animation.select(".glyph-labeled").style("opacity", 1).on("start", function (d) {
                    if (d.is_labeled)
                        d3.select(this).style("display", "block");
                    else
                        d3.select(this).style("display", "none");
                })
                that.draw_flowmap(that.flowmapdata);
                that.draw_image(t);

            }
        } else {
            glyphs.transition()
                .duration(t ? t : duration)
                .style("opacity", 1)
                .attr("transform", function (d) {
                    return "translate(" + d.x + "," + d.y + ") scale(" + d.r + ")";
                });
        }


    }
    this.remove = function (t) {
        var glyphs = that.glyphs.exit().transition()
            .duration(t ? t :duration)
            .style("opacity", 0).remove();
    }



    this.draw_lasso = function () {
        lasso_view.selectAll("rect").remove();
        lasso_view.selectAll("g").remove();
        // Lasso functions to execute while lassoing
        var lasso_start = function() {
            svg.selectAll(".selected").classed("selected", false);
            svg.selectAll(".unselected").classed("unselected", false);
          that.lasso.items().classed("selected", false).classed("unselected", true); // style as not possible
        };

        var lasso_draw = function() {
            that.lasso.possibleItems().classed("selected", true).classed("unselected", false);
            that.lasso.notPossibleItems().classed("unselected", true).classed("selected", false);
        };

        var lasso_end = function() {
            var isselected = false;
            that.lasso.selectedItems().classed("selected", function (d) {
                isselected = true;
                return true;
            }).classed("unselected", false);
            that.lasso.notSelectedItems().classed("unselected", true).classed("selected", false);
            if (!isselected){
                that.lasso.notSelectedItems().classed("unselected", false).classed("selected", false);
            }
            var data = svg.selectAll(".selected").data();
            on_glyph_selected(data);
        };

        // Create the area where the lasso event can be triggered
        var lasso_area = lasso_view.append("rect")
                              .attr("width",width)
                              .attr("height",height)
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
        lasso_view.call(that.lasso);
    }
    this.set_flowmap_data = function (data) {
        that.flowmapdata = data;
    }
    this.draw_flowmap = function (data) {
        that.remove_flowmap();
        var flowmaps = {};
        for (var i = 0; i < data.length; i++){
            var node = data[i];
            var source = {"x": node.x, "y": node.y, "id": node.id};
            var targets = [];
            if (that.glyph_dict.hasOwnProperty(node.id)){
                source.x = that.glyph_dict[node.id].x;
                source.y = that.glyph_dict[node.id].y;
            }
            var influence = [];
            for (var j = 0; j < node.influence.length; j++){
                var node2 = node.influence[j];
                if (node2.weight > threshold) {
                    influence.push(node2.id);
                    var target = {"x": node2.x, "y": node2.y, "weight": node2.weight};
                    if (that.glyph_dict.hasOwnProperty(node2.id)){
                        target.x = that.glyph_dict[node2.id].x;
                        target.y = that.glyph_dict[node2.id].y;
                    }
                    targets.push(target);
                    // glyph_view.select("#glyph" + node2.id).append("text").style("font-size", 24).text(SimilarityMatrix[node.id][node2.id])
                }
            }
            for(var glyphkey in that.glyph_dict){
                if (glyphkey != node.id && influence.indexOf(glyphkey) == -1){
                    targets.push({"x": that.glyph_dict[glyphkey].x, "y": that.glyph_dict[glyphkey].y, "weight": 0})
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
                return "glyph-flow-link" + flowmaps[i].source.id;
            })
             flows.selectAll(".glyph-flow-link").data(function (d) {
                 return d;
             }).enter().append("path")
            .attr("class", "glyph-flow-link")
            .attr("d", function (d) {
                var path = d3.path();
                path.moveTo(d[0].split(",")[0], d[0].split(",")[1]);
                path.quadraticCurveTo(d[1].split(",")[0], d[1].split(",")[1], d[2].split(",")[0], d[2].split(",")[1]);
                return path.toString();
            }).style("stroke-width", function (d) {
                return d[3].split(",")[0];
            })
            .style("opacity", 0)
            .transition().duration(duration).style("opacity", 1);
        });
    }
    this.remove_flowmap = function () {
        flow_view.selectAll("g").remove();
    }
    this.highlight_flowmap = function (ids) {
        if (ids.length == 0){
            flow_view.selectAll(".glyph-flow-link").style("stroke", "#ccc");
        }else {
            flow_view.selectAll(".glyph-flow-link").style("stroke", "#eee");
            for (var i = 0; i < ids.length; i++){
                flow_view.select("#glyph-flow-link" + ids[i]).selectAll(".glyph-flow-link").style("stroke", "#ccc");
            }
        }
    }

    this.draw_image = function (t) {
        var nodelist = [];
        var nodes = [];
        for (var i = 0; i < that.cur_data.length; i++){
            var node = that.cur_data[i];
            nodes.push([node.x, node.y, node.r, imagewidth * node.r, imagewidth * node.r, node.uncertainty])
        }
        var labellayout = {"width": width, "height": height, "data": nodes};
        //http://visgroup.thss.tsinghua.edu.cn:30075/
        $.post("http://visgroup.thss.tsinghua.edu.cn:30075/api/flowmap/labellayout", labellayout ,function(response){
            for (var i = 0; i < response.length; i++){
                var image = response[i];
                var truewidth = imagewidth * that.cur_data[i].r;
                var ix = image.Label.Left + truewidth / 2 - width / 2;
                var iy = image.Label.Top + truewidth / 2 - height / 2;
                if(image.Label.IsVisible && ix*ix+iy*iy < (width / 2 - 0.75 * truewidth)*(width/2 - 0.75 * truewidth)){
                    that.cur_data[i].imageX = (image.Label.Left - that.cur_data[i].x) / that.cur_data[i].r;
                    that.cur_data[i].imageY = (image.Label.Top - that.cur_data[i].y) / that.cur_data[i].r;
                    that.cur_data[i].isShowImage = true;
                } else {
                    that.cur_data[i].isShowImage = false;
                }
            }
            that.glyphs.enter().merge(that.glyphs).select(".glyph-image").transition()
                .duration(t ? t :duration)
                .attr("width", function (d) {
                    if (d.isShowImage){
                        return imagewidth;
                    }
                    return 0;
                })
                .attr("height", function (d) {
                    if (d.isShowImage){
                        return imagewidth;
                    }
                    return 0;
                })
                .attr("x",function (d) {
                    if (d.isShowImage){
                        return d.imageX;
                    }
                    return 0;
                })
                .attr("y",function (d) {
                    if (d.isShowImage){
                        return d.imageY;
                    }
                    return 0;
                })
                .style("opacity", function (d) {
                    if (d.isShowImage){
                        return 1;
                    }
                    return 0;
                });
        });
    }

    this.redraw = function () {
        lasso_view.transition().duration(duration).style("opacity", 0).remove();
        flow_view.transition().duration(duration).style("opacity", 0).remove();
        glyph_view.transition().duration(duration).style("opacity", 0).remove();
        lasso_view = svg.append("g").attr("class", "lasso-view");
        flow_view = svg.append("g").attr("class", "flow-view");
        glyph_view = svg.append("g").attr("class", "glyph-node-view");
        on_glyph_selected([]);
    }

    this.init = function () {
        base_data = instances_processing();
        if (!isDebug && isLasso) {
            that.draw_lasso();
        }
    }
}