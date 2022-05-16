function simi_graph(image_group, param) {
    var that = this;

    var radius = param.radius;
    var border_width = 3;
    var center_x = param.center_x;
    var center_y = param.center_y;

    var layout_info_saved = [];
    var layout_info = [];
    var layout_info_count = 0;
    var layer_image_size = [150, 100, 50];

    var current_focus;

    this.draw = function () {
        that.update();
    };


    this.get_current_focus = function () {
        return current_focus;
    };

    this.update = function (id) {
        if (!id) {
            id = 0;
        }
        current_focus = id;
        var diagonal = d3.svg.diagonal()
            .projection(function (d) {
                return [d.x, d.y];
            });

        var layer_info = get_layer2_and_layer3(id);
        var layer2 = layer_info[0];
        var layer3 = layer_info[1];
        layout_info = get_layout_info(id, layer2, layer3);

        /*
        * begin add DOM element
        * */
        //delete old DOM
        image_group.selectAll("g.node-image")
            .remove();
        //draw new DOM

        image_group.selectAll("g.node-image")
            .data(layout_info)
            .enter()
            .append("g")
            .attr("class", "node-image");
        image_group.selectAll("g.node-image")
            .data(layout_info)
            .style("opacity", 0)
            .transition()
            .duration(500)
            .style("opacity", 1)
            .each(function (d, i) {
                var _this = d3.select(this);
                _this
                    .attr("transform", function () {
                        return "translate("
                            + (d.position.x)
                            + ","
                            + (d.position.y)
                            + ")";
                    });

                _this.selectAll("rect.frame")
                    .data([[d.width, d.height]])
                    .enter()
                    .append("rect")
                    .attr("class", "frame");
                _this.selectAll("rect.frame")
                    .data([[d.width, d.height]])
                    .attr("width", d.width + 2 * border_width)
                    .attr("height", d.height + 2 * border_width)
                    .attr("x", -d.width / 2 - border_width)
                    .attr("y", -d.height / 2 - border_width)
                    .attr("rx", border_width)
                    .attr("ry", border_width)
                    .style("fill", CategoryColor[PosteriorLabels[d.id]]);
                _this.selectAll("rect.frame")
                    .data([[d.width, d.height]])
                    .exit()
                    .remove();

                _this.selectAll("image")
                    .data([[d.width, d.height]])
                    .enter()
                    .append("image")
                    .attr("class", "c-" + i);
                _this.selectAll("image")
                    .data([[d.width, d.height]])
                    .attr("width", d.width)
                    .attr("height", d.height)
                    .attr("x", -d.width / 2)
                    .attr("y", -d.height / 2)
                    .attr("xlink:href", ImageUrl[d.id])
                    .on("mousedown", function () {

                    });
                _this.selectAll("image")
                    .data([[d.width, d.height]])
                    .exit()
                    .remove();

            });
        image_group.selectAll("g.node-image")
            .data(layout_info)
            .exit()
            .remove();

    };

    that.add_layout = function () {
        layout_info_saved[layout_info_count++] = layout_info;
    };

    that.save_layout = function () {
        // console.log(layout_info_saved);
        var obj = JSON.stringify(layout_info_saved);
        localStorage.setItem("temp", obj);
    };

    var map_ang_to_position = function (ang, layer) {
        var dis = null;
        if (layer == 2) {
            dis = 0.50 * ( 0.9 + Math.random() * 0.25 );
            // dis = 0.48;
        }
        else if (layer == 3) {
            // dis = 0.85 * ( 0.87 + Math.random() * 0.25 );
            dis = 0.90;
        }
        else {
            alert("map_ang_to_position function wrong");
            // this line aim to terminate the program
            dis[1] = 1;
        }

        var x = dis * Math.cos(ang) * radius;
        var y = dis * Math.sin(ang) * radius;
        return {
            "x": x,
            "y": y
        };
    };

    var clean_repeated_image = function (id, layer2, layer3, n) {
        var visited = [];
        var cleaned_layer2 = [];
        var cleaned_layer3 = {};
        for (var i = 0; i < InstanceTotalNum; i++) {
            visited[i] = 0;
        }
        visited[id] = 1;
        var cleaned_layer2_count = 0;
        for (var i = 0; i < layer2.length; i++) {
            if (visited[layer2[i]]) {
                continue;
            }
            cleaned_layer2[cleaned_layer2_count++] = layer2[i];
            visited[layer2[i]] = 1;
            if (cleaned_layer2_count >= n) {
                break;
            }
        }
        for (var i = 0; i < cleaned_layer2.length; i++) {
            var cleaned_layer3_count = 0;
            var l3 = layer3[cleaned_layer2[i]];
            cleaned_layer3[cleaned_layer2[i]] = [];
            for (var j = 0; j < l3.length; j++) {
                if (visited[l3[j]]) {
                    continue;
                }
                cleaned_layer3[cleaned_layer2[i]][cleaned_layer3_count++] = l3[j];
                visited[l3[j]] = 1;
                if (cleaned_layer3_count >= n) {
                    break;
                }
            }
        }
        return [cleaned_layer2, cleaned_layer3]
    };

    var get_layer2_and_layer3 = function (id) {
        var layer2 = [];
        var layer3 = {};
        // TODO: need more rubost decision
        var num_image_per_layer = 5;
        var max_image_num = 30;
        layer2 = get_top_n_simi_id(max_image_num, id);
        for (var i = 0; i < layer2.length; i++) {
            layer3[layer2[i]] = get_top_n_simi_id(max_image_num, layer2[i]);
        }
        return clean_repeated_image(id, layer2, layer3, num_image_per_layer);
    };

    var get_top_n_simi_id = function (n, id) {
        var simi_list = [];
        for (var i = 0; i < InstanceTotalNum; i++) {
            simi_list[i] = [simi_matrix(i, id), i];
        }
        simi_list[id] = [0, id];
        simi_list.sort(function (a, b) {
            return ( b[0] - a[0] );
        });
        var top_n = [];
        for (var i = 0; i < n; i++) {
            top_n[i] = simi_list[i][1];
        }
        return top_n;
    };

    var get_arc_info = function (center, center_width, point, point_width, count) {
        var position_x = point.x - center.x;
        var position_y = point.y - center.y;
        var start = {};
        var end = {};
        if (Math.abs(position_y) < Math.abs(position_x)) {
            start["x"] = ( (position_x > 0) ? center_width / 2.0 : -center_width / 2.0);
            start["y"] = 0;
            if (( position_y / position_x) > 0.5) {
                end["x"] = 0;
                end["y"] = ( (position_x > 0) ? -point_width / 2.0 : point_width / 2.0);
            }
            else if (( position_y / position_x) < -0.5) {
                end["x"] = 0;
                end["y"] = ( (position_x > 0) ? point_width / 2.0 : -point_width / 2.0);
            }
            else {
                end["x"] = ( (position_x > 0) ? -point_width / 2.0 : point_width / 2.0);
                end["y"] = 0;
            }
        }
        else {
            start["y"] = ( (position_y > 0) ? center_width / 2.0 : -center_width / 2.0);
            start["x"] = 0;
            if (( position_x / position_y) > 0.5) {
                end["x"] = ( (position_y > 0) ? -point_width / 2.0 : point_width / 2.0);
                end["y"] = 0;
            }
            else if (( position_x / position_y) < -0.5) {
                end["x"] = ( (position_y > 0) ? point_width / 2.0 : -point_width / 2.0);
                end["y"] = 0;
            }
            else {
                end["x"] = 0;
                end["y"] = ( (position_y > 0) ? -point_width / 2.0 : point_width / 2.0);
            }
        }

        if (count == 17 || count == 27 || count == 14 || count == 30) {
            start["x"] = 0;
            start["y"] = ( (position_y > 0) ? center_width / 2.0 : -center_width / 2.0);
        }
        return {
            "source": {
                "x": end.x,
                "y": end.y
            },
            "target": {
                "x": start.x + center.x - point.x,
                "y": start.y + center.y - point.y
            }
        }
    };


    var get_layout_info = function (id, layer2, layer3) {
        var info = [];
        // the info element contains start_x, start_y, width, height, id, end_x, end_y
        var info_element = {};
        info_element["position"] = {"x": 0, "y": 0};
        info_element["width"] = layer_image_size[0];
        info_element["height"] = layer_image_size[0];
        info_element["id"] = id;
        info_element["simi"] = 0;
        info_element["link"] = {
            "source": {
                "x": 0,
                "y": 0
            },
            "target": {
                "x": 0,
                "y": 0
            }
        };
        info.push(info_element);
        // var layer2_start_arc = StartAngle[( Count++ )% StartAngle.length ] * Math.PI * 2;
        var layer2_start_arc = StartAngle[0] * Math.PI * 2;
        var layer3_zoom = Math.PI * 2 / layer2.length;
        var count = 0;
        for (var i = 0; i < layer2.length; i++) {
            count++;
            var layer2_width = layer_image_size[1];
            var layer2_height = layer_image_size[1];
            var layer2_arc = layer2_start_arc + Math.PI * 2 / layer2.length * i;
            var layer2_position = map_ang_to_position(layer2_arc, 2);
            var info_element = {};
            info_element["position"] = layer2_position;
            info_element["width"] = layer2_width;
            info_element["height"] = layer2_height;
            info_element["id"] = layer2[i];
            info_element["simi"] = simi_matrix(layer2[i], id);
            info_element["link"] = get_arc_info({"x": 0, "y": 0}, layer_image_size[0],
                layer2_position, layer2_width, count);
            info.push(info_element);

            var l3 = layer3[layer2[i]];
            var lower_bound = layer2_arc - layer3_zoom / 2;
            for (var j = 0; j < l3.length; j++) {
                count++;
                var layer3_arc = lower_bound + layer3_zoom / l3.length * j + layer3_zoom / l3.length / 2;
                var layer3_width = layer_image_size[2];
                var layer3_height = layer_image_size[2];
                var info_element = {};
                var layer3_position = map_ang_to_position(layer3_arc, 3);
                info_element["position"] = layer3_position;
                info_element["width"] = layer3_width;
                info_element["height"] = layer3_height;
                info_element["id"] = l3[j];
                info_element["simi"] = simi_matrix(l3[j], layer2[i]);
                info_element["link"] = get_arc_info(layer2_position, layer2_width,
                    layer3_position, layer3_width, count);
                info.push(info_element);
            }
        }
        return info;
    };
}
var StartAngle = [ 0.15, 0.1, 0.3, 0.5, 0.9 ];