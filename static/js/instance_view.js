var InstancesLayout = function (container) {
    var that = this;
    that.container = container;

    var bbox = that.container.node().getBoundingClientRect();
    var width = 1200;
    var height = 1200; // 28 is title height
    var plot_width = width - 50;
    var plot_height = height - 50;
    var plot_radius = plot_height - 50;
    var padAngle = 0.05;
    var plot_x_shift = 0;
    var plot_y_shift = 0;
    var legend_x_shift = 10;
    var legend_y_shift = height - 150;
    var legend_height = 20;
    var legend_width = 40;
    var legend_padding = 10;
    var text_height = legend_height;
    var text_x_shift = legend_width + 10;
    var text_y_shift = legend_y_shift;
    var text_padding = legend_padding;
    var text_px = 12;
    var matrix_px = 0;
    var matrix_py = 0;
    var center_x = plot_width / 2 + 30;
    var center_y = plot_height / 2 + 30;
    var point_scale = plot_width / 1500.0;
    var solid_point_radius = point_scale * 3;
    var influence_threshold = 1;
    var change_flag = true;

    var force_layout = null;
    var ghost_point_id = null;
    var center_point_x = plot_width / 2;
    var center_point_y = plot_height / 2;
    var corner_point_id = [];

    var image_layout = null;

    var svg = that.container.append("svg");
    var flow_map_group = svg.append("g").attr("id","flow-map-group");
    var plot_group = svg.append("g").attr("id", "plot-group");
    var glyph_group = svg.append("g").attr("id", "glyph-group");
    var legend_group = svg.append("g").attr("id", "legend-group");
    var matrix_group = svg.append("g").attr("id", "matrix-group");
    var arc_group = svg.append("g").attr("id","arc-group");
    var button_group = svg.append("g").attr("id", "button-group");

    var class_arc = null;

    //data
    var selected_influence = null;
    var class_percent = null;
    var selected_labels = null;
    var selected_instance_index = null;
    var selected_label_num = null;
    var selected_worker_num = null;
    var selected_instance_num = null;
    var selected_label_names = null;
    var selected_posterior_labels = null;
    var selected_uncertainty = null;
    var selected_worker_labels = null;
    var selected_image_url = null;
    var selected_class_color = null;
    var selected_center_points = null;
    var selected_center_point_in_unit_scale = null;
    var selected_nodes = null;
    var selected_significant_nodes = null;
    var selected_index_2_order_map = null;

    var source_id = null;


    this._init = function () {
        // set up svg's width and height
        svg.attr("width", bbox.width)
            .attr("height", bbox.height - 28)
            .attr("viewBox", "0, 0, 1200, 1200")
            .attr("id", "instance-svg");

        //set up groups positions
        plot_group.attr("transform", "translate(" + (center_x - plot_radius/2 ) + "," + (center_y - plot_radius/2)+ ")");
        glyph_group.attr("transform", "translate(" + (center_x - plot_radius/2 ) + "," + (center_y - plot_radius/2)+ ")");
        legend_group.attr("transform", "translate(" + legend_x_shift + "," + legend_y_shift + ")");
        arc_group.attr("transform", "translate(" + center_x + "," + center_y + ")");
        flow_map_group.attr("transform", "translate(" + (center_x - plot_radius/2 ) + "," + (center_y - plot_radius/2)+ ")");
        button_group.attr("transform", "translate(" + (center_x - plot_radius/2 ) + "," + (center_y - plot_radius/2)+ ")");
        // matrix_group.attr("transform","translate(" + ( matrix_px ) + "," + ( matrix_py ) + ")");

    };

    this.init = function(){
        that._init();
        that.instanceGlyph = new InstanceGlyph(glyph_group,
            {"width": plot_radius, "height": plot_radius, "px" : center_x - plot_radius/2, "py" : center_y - plot_radius/2},
            flow_map_group, button_group);
        //TODO: for debug
        that.Matrix = new ConfusionMatrix(matrix_group, {"width": width, "height": height});
        that.Matrix.draw(WorkerPredictionMatrix, LabelNames);
        d3.selectAll(".tooltip")
            .transition()
            .duration(AnimationDuration)
            .style("opacity", 0.7);
    }.call();

    this.resize = function () {
        bbox = container.node().getBoundingClientRect();
        svg.attr("width", bbox.width)
            .attr("height", bbox.height - 28);
    }


    this.data_update = function(instance_ranking_index){
        selected_labels = SelectedList;
        selected_instance_index = SelectedGlobal["InstanceIndex"];
        selected_label_num = SelectedGlobal["LabelTotalNum"];
        selected_worker_num = SelectedGlobal["WorkerTotalNum"];
        selected_instance_num = SelectedGlobal["InstanceTotalNum"];
        selected_label_names= SelectedGlobal["LabelNames"];
        selected_posterior_labels = SelectedGlobal["PosteriorLabels"];
        selected_uncertainty = SelectedGlobal["Uncertainty"];
        selected_worker_labels = SelectedGlobal["WorkerLabels"];
        selected_class_color = SelectedGlobal["CategoryColor"];

        class_arc = SelectedGlobal["ClassArc"];
        selected_center_point_in_unit_scale =
            SelectedGlobal["SelectedCenterPointInUnitScale"];

        selected_index_2_order_map = {};
        for (var i = 0; i < selected_instance_index.length; i++){
            var index = selected_instance_index[i];
            selected_index_2_order_map[index] = i;
        }

        this._arc_data_update();
        this._instance_data_update(instance_ranking_index);
    };

    this.ranking_data_update = function(instance_ranking_index, influence){
        selected_nodes = [];
        selected_significant_nodes = [];
        var selected_significant_nodes_id = [];
        var selected_significant_node_type = [];
        var map = {};
        var mix_index = [];
        for (var i = 0; i < instance_ranking_index.length ; i++){
            if( selected_instance_index.includes(instance_ranking_index[i])) {
                mix_index.push(instance_ranking_index[i]);
                selected_significant_nodes_id.push(instance_ranking_index[i]);
                if(PosteriorLabels[instance_ranking_index[i]] != pre_posterior_labels(instance_ranking_index[i])
                    && pre_posterior_labels(instance_ranking_index[i])!=-1){
                    selected_significant_node_type.push(2);
                }
                else{
                    selected_significant_node_type.push(0);
                }
            }
            if(selected_significant_nodes_id.length >= 20 ){
                break;
            }
        }
        source_id = [];
        for(var str_i in influence){
            var i = parseInt(str_i);
            source_id.push(parseInt(i));
        }
        for (var i = 0; i < source_id.length; i++){
            selected_significant_nodes_id.push(source_id[i]);
            if(PosteriorLabels[source_id[i]] != pre_posterior_labels(source_id[i])
                && pre_posterior_labels(instance_ranking_index[i])!=-1){
                selected_significant_node_type.push(2);
                }
            else{
                selected_significant_node_type.push(1);
            }
        }

        // update influence
        selected_influence = [];
        for(var str_i in influence){
            var i = parseInt(str_i);
            var individual_influence = [];
            var source_x = guided_tsne_coordinate_x(selected_index_2_order_map[i]) * plot_radius;
            var source_y = guided_tsne_coordinate_y(selected_index_2_order_map[i]) * plot_radius;
            for(var j = 0; j < influence[i].length; j++ ){
                var tmp = influence[i][j];
                var index = tmp.id;
                if( selected_instance_index.includes(index)
                    && (!source_id.includes(index))
                    && selected_significant_nodes_id.includes(index) ){
                    var x = guided_tsne_coordinate_x(selected_index_2_order_map[index]) * plot_radius;
                    var y = guided_tsne_coordinate_y(selected_index_2_order_map[index]) * plot_radius;
                    tmp["x"] = x;
                    tmp["y"] = y;
                    individual_influence.push(tmp);
                }
            }
            var index = parseInt(i);
            selected_influence.push({
                "id":index,
                "influence": individual_influence,
                "x":source_x,
                "y":source_y
            });
        }
        selected_nodes = [];
        selected_significant_nodes = [];
        for( var i = 0; i < selected_instance_index.length; i++){
            var index = selected_instance_index[i];
            if( !selected_significant_nodes_id.includes(index)){
                selected_nodes.push(that.node_format(index));
            }
            else{
                for( var k = 0; k < selected_significant_nodes_id.length; k++) {
                    if( selected_significant_nodes_id[k] == index){
                        selected_significant_nodes.push(that.node_format(index, selected_significant_node_type[k]));
                    }
                }
            }
        }
    };

    this.node_format = function(index, type){
        var x = guided_tsne_coordinate_x(selected_index_2_order_map[index]) * plot_radius;
        var y = guided_tsne_coordinate_y(selected_index_2_order_map[index]) * plot_radius;
        var node =  {
            "x":x,
            "y":y,
            "label": selected_posterior_labels[index],
            "id": index,
            "arc_list": that._computing_arc_of_instance_to_arc_centering(x,y),
            "uncertainty":Uncertainty[index]
        };
        if(type == 0){
        }
        else if( type == 1){
            node["is_labeled"] = true;
        }
        else if( type == 2){
            // node["is_last_round"] = true;
            node["last_round_label"] = pre_posterior_labels(index);
        }
        return node;
    };

    this._arc_data_update = function(){
        // class_arc = pie(class_percent);
        // var rotate = 0;
        // for( var i = 0; i < class_arc.length; i++ ){
        //     if( class_arc[i].startAngle == 0){
        //         rotate = class_arc[i].endAngle / 2;
        //         break;
        //     }
        // }
        // for( var i = 0; i < class_arc.length; i++ ){
        //     class_arc[i].startAngle -= rotate;
        //     class_arc[i].endAngle -= rotate;
        // }
        selected_center_points = [];
        for( var i = 0; i < class_arc.length; i++ ){
            // var center_arc = (class_arc[i].startAngle + class_arc[i].endAngle) / 2.0;
            var x = plot_radius * selected_center_point_in_unit_scale[i].x;
            var y = plot_radius * selected_center_point_in_unit_scale[i].y;
            selected_center_points[i] = {
                "x": x,
                "y": y
            };
        }
        console.log(selected_center_points);
    };
    this._instance_data_update = function(instance_ranking_index){
        selected_nodes = [];
        selected_significant_nodes = [];
        var selected_significant_nodes_id = [];
        var selected_significant_node_type = [];
        var mix_index = [];
        for (var i = 0; i < instance_ranking_index.length ; i++){
            var ranking_index = instance_ranking_index[i];
            if( selected_instance_index.includes(ranking_index)) {
                mix_index.push(ranking_index);
                selected_significant_nodes_id.push(ranking_index);
                if(PosteriorLabels[ranking_index] != pre_posterior_labels(ranking_index)
                    && pre_posterior_labels(ranking_index)!=-1){
                    selected_significant_node_type.push(2);
                }
                else{
                    selected_significant_node_type.push(0);
                }
            }
            if(selected_significant_nodes_id.length >= 20 ){
                break;
            }
        }

        // test
        var test_sum = 0;
        var test_list = [];
        var selected_test_sum = 0;
        var selected_test_list = [];
        for( var i = 0; i < PosteriorLabels.length; i++){
            if(PosteriorLabels[i] != pre_posterior_labels(i)
                && pre_posterior_labels(i)!=-1){
                test_sum = test_sum + 1;
                test_list.push(i);
                if( SelectedList.includes(PosteriorLabels[i])){
                    selected_test_sum += 1;
                    selected_test_list.push(i);
                }
            }
        }
        console.log("num of changed : " + (test_sum));
        console.log(test_list);
        console.log("num of selected changed:" + selected_test_sum);
        console.log(selected_test_list);

        for (var i = 0; i < instance_ranking_index.length; i++){
            var index = instance_ranking_index[i];
            if(test_list.includes(index)){
                var test_a = 1;
            }
            if(selected_instance_index.includes(index)
                && (!selected_significant_nodes_id.includes(index))
                && (PosteriorLabels[index] != pre_posterior_labels(index))
                && (pre_posterior_labels(index) != -1)
                && (!LabeledList.includes(index))
                ){
                selected_significant_nodes_id.push(index);
                selected_significant_node_type.push(2);
            }
            if(selected_significant_nodes_id.length >= 25 ){
                break;
            }
        }
        selected_nodes = [];
        selected_significant_nodes = [];
        for( var i = 0; i < selected_instance_index.length; i++){
            var index = selected_instance_index[i];
            if( !selected_significant_nodes_id.includes(index)){
                if((PosteriorLabels[index] != pre_posterior_labels(index))){
                    selected_nodes.push(that.node_format(index, 2));
                }
                else{
                    selected_nodes.push(that.node_format(index, 2));
                }
            }
            else{
                for( var k = 0; k < selected_significant_nodes_id.length; k++) {
                    if( selected_significant_nodes_id[k] == index){
                        selected_significant_nodes.push(that.node_format(index, selected_significant_node_type[k]));
                    }
                }
            }
        }
    };
    this._computing_arc_of_instance_to_arc_centering = function(x,y){
        var arc_list = [];
        for( var i = 0; i < selected_center_points.length; i++ ){
            var dx = selected_center_points[i].x - x;
            var dy = selected_center_points[i].y - y;
            var z = Math.sqrt( Math.pow(dx,2) + Math.pow(dy,2) );
            var sin = dx / z;
            var radina = Math.asin(sin);
            if ( dy > 0 ){
                if ( dx > 0 ){
                    arc_list[i] = 180 - 180 * radina / Math.PI;
                }
                else{
                    arc_list[i] = 180 - 180 * radina / Math.PI;
                }
            }
            else{
                if ( dx > 0 ){
                    arc_list[i] = 180 * radina / Math.PI;
                }
                else{
                    arc_list[i] = 360 + 180 * radina / Math.PI;
                }
            }
            // arc_list[i] = 180 * radina / Math.PI;
        }
        return arc_list;
    };

    this.draw = function (instance_ranking_index) {
        that.data_update(instance_ranking_index);
        that._draw();
    };

    this.candidate_selection_draw = function(instance_ranking_index, influence){
        that.ranking_data_update(instance_ranking_index, influence);
        that._draw();
    };

    this._draw = function(seed){
        that.create();
        that.update();
        that.remove();
        that.instanceGlyph.draw(selected_significant_nodes);
    };

    this.redraw = function(){
        // plot_group.transition().duration(AnimationDuration).style("opacity",0).remove();
        // plot_group.selectAll().transition().duration(AnimationDuration).style("opacity",0);
        // legend_group.transition().duration(AnimationDuration).style("opacity",0).remove();
        // arc_group.transition().duration(AnimationDuration).style("opacity",0).remove();
        // plot_group = svg.append("g").attr("id", "plot-group");
        // legend_group = svg.append("g").attr("id", "legend-group");
        // arc_group = svg.append("g").attr("id","arc-group");
        that.delete();
        this.instanceGlyph.redraw();
        that._init();

    };
    // global
    this.create = function(){
        // this.create_legend();
        this.create_arc();
        this.create_instance();
    };
    this.update = function(){
        // this.update_legend();
        this.update_arc();
        this.update_instance();
    };
    this.remove = function(){
        // this.remove_legend();
        this.remove_arc();
        this.remove_instance();
    };
    this.delete = function(){
        this.delete_legend();
        this.delete_arc();
        this.delete_instance();
    };
    // legend
    this.create_legend = function(){
        legend_group.selectAll("rect.legend")
            .data(selected_class_color)
            .enter()
            .append("rect")
            .attr("class", "legend");
        legend_group.selectAll("rect.legend")
            .data(selected_class_color)
            .attr("width", legend_width)
            .attr("height", legend_height)
            .attr("x", 0)
            .attr("y", function (d, i) {
                return i * ( legend_height + legend_padding);
            })
            .style("fill", function(d,i){
                return d;
            })
            .style("opacity",0);


        legend_group.selectAll("text.legend")
            .data(selected_label_names)
            .enter()
            .append("text")
            .attr("class", "legend");
        legend_group.selectAll("text.legend")
            .data(selected_label_names)
            .attr("x", text_x_shift)
            .attr("y", function (d, i) {
                return i * ( text_height + text_padding ) + text_height / 2 + text_px / 2;
            })
            .attr("font-size", text_px + "px")
            .attr("text-anchor", "start")
            .text(function (d, i) {
                return d;
            })
            .style("opacity",0);
    };
    this.update_legend = function(){
        legend_group.selectAll("rect.legend")
            .data(selected_label_names)
            .transition()
            .duration(AnimationDuration)
            .style("opacity",1);

        legend_group.selectAll("text.legend")
            .data(selected_label_names)
            .transition()
            .duration(AnimationDuration)
            .style("opacity",1);
    };
    this.remove_legend = function(){
        legend_group.selectAll("rect.legend")
            .data(selected_class_color)
            .exit()
            .remove();
        legend_group.selectAll("text.legend")
            .data(selected_label_names)
            .exit()
            .remove();
    };
    this.delete_legend = function(){
        legend_group.selectAll("rect.legend")
            .transition()
            .duration(AnimationDuration)
            .style("opacity",0)
            .remove();
        legend_group.selectAll("text.legend")
            .transition()
            .duration(AnimationDuration)
            .style("opacity",0)
            .remove();
    };

    // arc
    this.create_arc = function(){
        var arc_path = arc_group.selectAll("path.overall")
            .data(class_arc)
            .enter()
            .append("path")
            .attr("class","overall");
        arc_path.attr("d", function(d){
                var centerAngle = (d.startAngle + d.endAngle) / 2;
                if (centerAngle > 1.57 && centerAngle < 4.71) {
                    var temp = d.startAngle;
                    d.startAngle = d.endAngle;
                    d.endAngle = temp;
                }
                return d3.arc()
                    .outerRadius(plot_radius/2 + 5)
                    .innerRadius(plot_radius/2 ).padAngle(padAngle)(d);
            })
            .attr("id", function(d,i){
                return "arc-" + i;
            })
            .style("fill", function(d,i){
                return selected_class_color[i];
            })
            .style("opacity",0)
            .transition()
            .duration(AnimationDuration)
            .style("opacity",1);

        var arc_text = arc_group.selectAll("text.arc-text")
            .data(class_arc)
            .enter()
            .append("text")
            .attr("class","arc-text");
        arc_text.attr("x", function(d,i){
                var centerAngle = Math.abs(d.endAngle - d.startAngle) / 2;
                if (centerAngle > 1.57 && centerAngle < 4.71) {
                    centerAngle += padAngle / 2;
                }else {
                    centerAngle -= padAngle / 2;
                }
                return (plot_radius/2) * centerAngle;//Math.PI * d.value;
            })
            .attr("dy", function (d) {
                var centerAngle = (d.startAngle + d.endAngle) / 2;
                if (centerAngle > 1.57 && centerAngle < 4.71) {
                    return 20;
                }
                return -5;
            })
            .style("font-size", "18px")
            .attr("text-anchor", "middle")
            .append("textPath")
            .attr("xmlns:xlink", "http://www.w3.org/1999/xlink")
            .attr("xlink:href", function(d,i){
                return "#arc-" + i;
            })
            .attr("fill", function(d,i){
                return selected_class_color[i];
            })
            .text(function(d,i){
                return selected_label_names[i];
            });
    };
    this.update_arc = function(){
    };
    this.remove_arc = function(){
        arc_group.selectAll("path.overall")
            .data(class_arc)
            .exit()
            .remove();
        arc_group.selectAll("text.arc-text")
            .data(class_arc)
            .exit()
            .remove();
    };
    this.delete_arc = function(){
        arc_group.selectAll("path.overall")
            .transition()
            .duration(AnimationDuration)
            .style("opacity",0)
            .remove();
        arc_group.selectAll("text.arc-text")
            .transition()
            .duration(AnimationDuration)
            .style("opacity", 0)
            .remove();
    };

    // instance
    this.create_instance = function(){
        plot_group.selectAll("circle.item")
            .data(selected_nodes)
            .enter()
            .append("circle")
            .attr("class", "item");
        plot_group.selectAll("circle.item")
            .data(selected_nodes)
            .attr("id", function (d, i) {
                return "ID-" + d.id;
            })
            .attr("cx", function (d, i) {
                return d.x;
            })
            .attr("cy", function (d, i) {
                return d.y;
            })
            .attr("r", solid_point_radius)
            .style("fill", function (d, i) {
                return CategoryColor[d.label];
            })
            .style("opacity",0)
            .on("click",on_plot_clicked);
        // plot_group.selectAll("")
    };
    this.update_instance = function(){
        plot_group.selectAll("circle.item")
            .data(selected_nodes)
            .transition()
            .duration(AnimationDuration)
            .style("opacity",1);
    };
    this.remove_instance = function(){
        plot_group.selectAll("circle.item")
            .data(selected_nodes)
            .exit()
            .remove();
    };
    this.delete_instance = function(){
        plot_group.selectAll("circle.item")
            .transition()
            .duration(AnimationDuration)
            .style("opacity",0)
            .remove();
    };
    this.set_all_instance_dehighlight = function(){
        plot_group.selectAll("circle.item")
            .classed("dehighlight", true);
    };

    this._set_instance_highlight = function(ids, worker_id){
        if( ids.length == 0){
            plot_group.selectAll("circle.item")
                .style("opacity",1);
        }
        var id_list = [];
        var class_map = {};
        for(var i = 0; i < ids.length; i++){
            for(var j = 0; j < ids[i].length; j++){
                id_list.push(ids[i][j]);
                class_map[ids[i][j]] = SelectedList[i];
            }
        }
        plot_group.selectAll("circle.item")
            .style("opacity",function(d){
                if( id_list.includes(d.id) ){
                    return 1;
                }
                else{
                    return 0.3;
                }
            })
            .attr("r", function(d){
                if( id_list.includes(d.id) ){
                    return 1.5 * solid_point_radius;
                }
                else{
                    return 1 * solid_point_radius;
                }
            })
            .style("fill", function(d){
                if( id_list.includes(d.id) ){
                    return CategoryColor[WorkerLabels[d.id][worker_id]];
                }
                else{
                    return CategoryColor[d.label];
                }
            });


    };

    this.set_instance_highlight = function(ids){
        if( ids.length == 0 ){
            plot_group.selectAll("circle.item")
                .classed("dehighlight", false);
            return;
        }
        var id_list = [];
        for(var i = 0; i < ids.length; i++){
            for(var j = 0; j < ids[i].length; j++){
                id_list.push(ids[i][j]);
            }
        }
        plot_group.selectAll("circle.item")
            .classed("dehighlight",function(d){
                if( id_list.includes(d.id) ){
                    return 1.5;
                }
                else{
                    return 1;
                }
            });
    };


    this.set_selected = function(ids){
        that.instanceGlyph.set_selected(ids);
    };
    this.single_highlight = function(id, isdraw){
        var data = null;
        for( var i = 0; i < selected_nodes.length; i++ ){
            if( selected_nodes[i].id == id ){
                data = selected_nodes[i];
                break;
            }
        }
        console.log("data", data);
        if(!data){
            return;
        }
        if(!isdraw){
            console.log(data);
        }
        that.instanceGlyph.single_draw([data], isdraw);
    };

    this._multi_highlight = function(ids, isdraw){
        var data = [];
        for (var i = 0; i < selected_nodes.length; i++){
            if(ids.includes(selected_nodes[i].id)){
                data.push(selected_nodes[i]);
            }
        }
        if( data.length < 1){
            return;
        }
        if(!isdraw){
            console.log(data);
        }
        that.instanceGlyph.single_draw(data, isdraw);
    };

    this.multi_highlight = function(ids, isdraw){
        if(isdraw){
        plot_group.selectAll("circle.item")
            .style("opacity",function(d){
                if( ids.includes(d.id) ){
                    return 1;
                }
                else{
                    return 0.3;
                }
            })
            .attr("r", function(d){
                if( ids.includes(d.id) ){
                    return 2.5 * solid_point_radius;
                }
                else{
                    return 1 * solid_point_radius;
                }
            })
            .style("fill", function(d){
                if( ids.includes(d.id) ){
                    return CategoryColor[WorkerLabels[d.id][worker_id]];
                }
                else{
                    return CategoryColor[d.label];
                }
            });
        }
        else{
            plot_group.selectAll("circle.item")
                .style("opacity",function(d){
                    return 1;
                })
                .attr("r", function(d){
                    return solid_point_radius;
                })
                .style("fill", function(d){
                    return CategoryColor[d.label];
                });
        }
    };

    this.set_highlight = function(ids){
        // that.set_instance_highlight(ids);
        that.instanceGlyph.set_highlight(ids);
    };
    this.set_highlight_by_label = function(ids, worker_id){
        if(ids == -1){
            that.set_all_instance_dehighlight();
            return;
        }
        that.set_instance_highlight(ids, worker_id);
        that.instanceGlyph.set_highlight_by_label(ids);
    };
    this.is_selected = function (id) {
        return that.instanceGlyph.is_selected(id);
    };


    this.draw_flowmap = function (empty) {
        if(!selected_influence){
            selected_influence = [];
        }
        if( empty == true){
            selected_influence = [];
        }
        that.instanceGlyph.set_flowmap_data(selected_influence);
    }
    this.remove_flowmap = function () {
        that.set_highlight([]);
        that.instanceGlyph.remove_flowmap();
    }
    this.highlight_flowmap = function (ids) {
        if(ids.length < 1 || !source_id || !source_id.includes(ids[0])) {
            that.instanceGlyph.highlight_flowmap([]);
            that.set_highlight([]);
        }
        else{
            that.instanceGlyph.highlight_flowmap(ids);
            that.set_highlight(ids);
        }
        // that.instanceGlyph.highlight_flowmap(ids);
        // that.set_highlight(ids);
    }

    function euclidean_distance(a, b){
        return Math.sqrt( (a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y) );
    }

};