/**
 * Created by 欧阳方昕 on 2017/11/26.
 */

var ImageDetail = function (container) {
    var that = this;
    var duration = AnimationDuration;
    var container = container;
    var bbox = container.node().getBoundingClientRect();
    var width = bbox.width;
    var height = bbox.height - 28; // 28 is title height
    var image_width = 200;
    var image_height = 200;
    var image_width_small = 40;
    var image_height_small = 40;
    var image_height_large = height * 0.9;
    var image_width_large = height * 0.9;
    var group_title_height = 40;
    var margin = 10;
    var margin_small = 5;
    var stroke = 8;
    var stroke_small = 3;
    var base_data = null;
    var mode = null;
    var clickTimer = null;
    that.svgs = [];

    var isShowTrueLabel = false;
    var isShowSimImage = false;

    function layout_large(data, svg, id) {
        var col_num = Math.floor(width / image_width);
        var addlarge = false, largey = 0;
        for (var i = 0; i < data.length; i++){
            data[i].x = i % col_num * image_width;
            data[i].y = Math.floor(i / col_num) * image_height;
            data[i].height = image_height;
            data[i].width = image_width;
            data[i].size_mode = "large";
            if(id != null && data[i].id == id){
                addlarge = true;
                largey = data[i].y;
            }
            if (addlarge && data[i].y > largey){
                data[i].y += image_height_large;
            }
        }
        if (addlarge) {
            svg.transition().duration(duration).attr("height", Math.ceil(data.length / col_num) * image_height + group_title_height + image_height_large);
        }else {
            svg.transition().duration(duration).attr("height", Math.ceil(data.length / col_num) * image_height + group_title_height);
        }
    }

    function layout_small(data, svg, id) {
        var col_num = Math.floor(width / image_width_small);
        var addlarge = false, largey = 0;
        for (var i = 0; i < data.length; i++){
            data[i].x = i % col_num * image_width_small;
            data[i].y = Math.floor(i / col_num) * image_height_small;
            data[i].height = image_height_small;
            data[i].width = image_width_small;
            data[i].size_mode = "small";
            if(id != null && data[i].id == id){
                addlarge = true;
                largey = data[i].y;
            }
            if (addlarge && data[i].y > largey){
                data[i].y += image_height_large;
            }
        }
        if (addlarge) {
            svg.transition().duration(duration).attr("height", Math.ceil(data.length / col_num) * image_height_small + group_title_height + image_height_large);
        }else {
            svg.transition().duration(duration).attr("height", Math.ceil(data.length / col_num) * image_height_small + group_title_height);
        }
    }

    this.resize = function () {
        bbox = container.node().getBoundingClientRect();
        width = bbox.width;
        height = bbox.height - 28;
        image_height_large = height * 0.9;
        image_width_large = height * 0.9;

        if (mode == "instance"){
            layout_large(that.cur_data, that.svgs[0]);
            that.update_detail(0, "large")
        }else {
            for (var i = 0; i < that.cur_data.length; i++) {
                layout_small(that.cur_data[i], that.svgs[i]);
                that.update_detail(i, "small")
            }
        }
    }

    this.set_selected = function (indexs) {
        container.selectAll(".image-node").classed("selected", false);
        for (var i = 0; i < indexs.length; i++){
            container.select("#image" + indexs[i]).classed("selected", true);
        }
    }

    this.set_highlight = function (indexs) {
        that.set_dehighlight(indexs);
    }

    this.set_dehighlight = function(indexs) {
        if (indexs.length == 0){
            container.selectAll(".image-node").classed("dehighlight", false);
        } else {
            container.selectAll(".image-node").classed("dehighlight", true);
            for (var i = 0; i < indexs.length; i++){
                container.select("#image" + indexs[i]).classed("dehighlight", false);
            }
        }

    }

    this.is_selected = function (id) {
        if (container.select("#image" + id).classed("selected")) {
            return true;
        }
        return false;
    }

    this.draw = function () {
        base_data = instances_processing();
    }
    this.draw_worker_detail = function (indexs, group_mode) {
        var data = [];
        var temp = [];
        that.images = [];
        if (group_mode == "M3V"){
            for (var i = 0; i < indexs.length-1; i++){
                data.push([]);
                temp.push([]);
            }
            for (var i = 0; i < indexs.length; i++){
                for (var j = 0; j < indexs[i].length; j++){
                    if (base_data[indexs[i][j]].label != i){
                        data[base_data[indexs[i][j]].label].push(base_data[indexs[i][j]]);
                    }else {
                        temp[base_data[indexs[i][j]].label].push(base_data[indexs[i][j]]);
                    }
                    base_data[indexs[i][j]].cur_label = i;
                }
            }
            for (var i = 0; i < indexs.length-1; i++){
                data[i].sort(function (a, b) {
                    return b.uncertainty - a.uncertainty;
                })
                temp[i].sort(function (a, b) {
                    return b.uncertainty - a.uncertainty;
                })
                data[i] = data[i].concat(temp[i]);
            }
        } else {
            for (var i = 0; i < indexs.length; i++){
                var sub_data = []
                for (var j = 0; j < indexs[i].length; j++){
                    sub_data.push(base_data[indexs[i][j]]);
                    base_data[indexs[i][j]].cur_label = base_data[indexs[i][j]].label;
                    if (isShowTrueLabel){
                        base_data[indexs[i][j]].cur_label = TrueLabels[base_data[indexs[i][j]].id];
                    }
                }
                sub_data.sort(function (a, b) {
                    return b.uncertainty - a.uncertainty;
                })
                data.push(sub_data);
            }
        }

        if (mode != "worker"){
            that.draw_group(data);
            for (var i = 0; i < data.length; i++){
                that.images.push(that.svgs[i].select(".group-image").selectAll(".image-node").data(data[i], function (d) { return "image" + d.id;}))
            }
            mode = "worker";
        }else {
            for (var i = 0; i < data.length; i++){
                that.images.push(that.svgs[i].select(".group-image").selectAll(".image-node").data(data[i], function (d) { return "image" + d.id;}))
            }
        }
        that.cur_data = data;
        for (var i = 0; i < data.length; i++) {
            layout_small(data[i], that.svgs[i]);
            that.create_detail(i)
            that.update_detail(i, "small")
            that.remove_detail(i)
        }
    }
    this.draw_group = function (data) {
        that.redraw();
        that.svgs = [];
        for (var i = 0; i < data.length; i++){
            var group_view = container.append("div").attr("class", "group-view").attr("id", "group" + i).append("svg").attr("width", width);
            that.svgs.push(group_view);
            var title = group_view.append("g").attr("class", "group-title");
            title.data([{
                "size_mode": "small",
                "group": i
            }]);
            title.append("text")
                .attr("x", margin_small)
                .attr("y", group_title_height / 2 + 6)
                .style("fill", SelectedGlobal.CategoryColor[i])
                .text(SelectedGlobal.LabelNames[i]);
            title.on("click", function (d) {
                if (d.size_mode == "small"){
                    layout_large(that.cur_data[d.group], that.svgs[d.group]);
                    d.size_mode = "large";
                } else {
                    layout_small(that.cur_data[d.group], that.svgs[d.group]);
                    d.size_mode = "small";
                }
                that.update_detail(d.group, d.size_mode)
            })
            group_view.append("g").attr("class", "group-image").attr("transform", "translate(0," + group_title_height + ")");
        }
    }
    this.draw_instance_detail = function (index) {
        var data = [];
        for (var i = 0; i < index.length; i++){
            data.push(base_data[index[i]]);
            base_data[index[i]].cur_label = base_data[index[i]].label;
            if (isShowTrueLabel){
                base_data[index[i]].cur_label = TrueLabels[base_data[index[i]].id];
            }
            if (isShowSimImage){
                // TODO: top 5
                var top_index = getTopSimInstance(index[i], 1);
                for (var j = 0; j < top_index.length; j++){
                    data.push(base_data[top_index[j]]);
                    base_data[top_index[j]].cur_label = base_data[top_index[j]].label;
                    base_data[top_index[j]].target_id = index[i];
                    if (isShowTrueLabel){
                        base_data[top_index[j]].cur_label = TrueLabels[base_data[top_index[j]].id];
                    }
                }
            }
        }
        if (!isShowSimImage){
            data.sort(function (a, b) {
                return b.uncertainty - a.uncertainty;
            })
        }
        if (mode != "instance"){
            that.draw_single(data);
            mode = "instance";
        }else {
            that.images = [];
            that.images.push(that.svgs[0].selectAll(".image-node").data(data, function (d) { return "image" + d.id;}))
        }
        that.cur_data = data;
        layout_large(data, that.svgs[0]);
        that.create_detail(0);
        that.update_detail(0, "large")
        that.remove_detail(0)

    }
    this.draw_single = function (data) {
        that.redraw();
        that.svgs = [];
        var svg = container.append("svg").attr("width", width);
        that.svgs.push(svg);
        that.images = [];
        that.images.push(svg.selectAll(".image-node").data(data, function (d) { return "image" + d.id;}))
    }

    this.draw_trail_detail = function (indexs) {
        var data = [];
        var temp = [];
        that.images = [];
        for (var i = 0; i < indexs.length; i++){
            var sub_data = [];
            var temp = [];
            for (var j = 0; j < indexs[i].length; j++){
                base_data[indexs[i][j].id].cur_label = base_data[indexs[i][j].id].label;
                if (isShowTrueLabel){
                    base_data[indexs[i][j].id].cur_label = TrueLabels[base_data[indexs[i][j].id].id];
                }
                if (base_data[indexs[i][j].id].label >= SelectedList.length){
                    temp.push(base_data[indexs[i][j].id]);
                }else {
                    sub_data.push(base_data[indexs[i][j].id]);
                }

            }
            sub_data.sort(function (a, b) {
                return b.uncertainty - a.uncertainty;
            })
            temp.sort(function (a, b) {
                return b.uncertainty - a.uncertainty;
            })
            sub_data = sub_data.concat(temp);
            data.push(sub_data);
        }

        if (mode != "trail"){
            that.draw_trail_group(data);
            for (var i = 0; i < data.length; i++){
                that.images.push(that.svgs[i].select(".group-image").selectAll(".image-node").data(data[i], function (d) { return "image" + d.id;}))
            }
            mode = "trail";
        }else {
            for (var i = 0; i < data.length; i++){
                that.images.push(that.svgs[i].select(".group-image").selectAll(".image-node").data(data[i], function (d) { return "image" + d.id;}))
            }
        }
        that.cur_data = data;
        for (var i = 0; i < data.length; i++) {
            layout_small(data[i], that.svgs[i]);
            that.create_detail(i)
            that.update_detail(i, "small")
            that.remove_detail(i)
        }
    }
    this.draw_trail_group = function (data) {
        that.redraw();
        that.svgs = [];
        var titles = ["Labeled Instances", "Influenced Instances"]
        for (var i = 0; i < data.length; i++){
            var group_view = container.append("div").attr("class", "group-view").attr("id", "group" + i).append("svg").attr("width", width);
            that.svgs.push(group_view);
            var title = group_view.append("g").attr("class", "group-title");
            title.data([{
                "size_mode": "small",
                "group": i
            }]);
            title.append("text")
                .attr("x", margin_small)
                .attr("y", group_title_height / 2 + 6)
                .style("fill", "gray")
                .text(titles[i]);
            title.on("click", function (d) {
                if (d.size_mode == "small"){
                    layout_large(that.cur_data[d.group], that.svgs[d.group]);
                    d.size_mode = "large";
                } else {
                    layout_small(that.cur_data[d.group], that.svgs[d.group]);
                    d.size_mode = "small";
                }
                that.update_detail(d.group, d.size_mode)
            })
            group_view.append("g").attr("class", "group-image").attr("transform", "translate(0," + group_title_height + ")");
        }
    }

    this.create_detail = function (index) {
        var images = that.images[index].enter().append('g')
            .attr("id", function (d) { return "image" + d.id; })
            .attr("class", "instance-filter image-node")
            .style("opacity", 0);
        function image_clicked (d) {
            var timerFunc = function () {
                that.expand_image(d);
                //TODO
                // on_image_clicked(d);
            }
            clearTimeout(clickTimer);
            clickTimer = setTimeout(timerFunc, 200);
        }

        images.append("image")
            .attr("class", "image")
            .attr("xlink:href", function (d) {return d.url;})
            .on("click",image_clicked);
        images.append("rect")
            .attr("class", "image-rect")
            .on("click", image_clicked);

        images.append("image")
            .attr("class", "image-mark")
            .attr("xlink:href", "/static/img/check-mark-button.svg");
        // images.append("image")
        //     .attr("class", "image-mask")
        //     .attr("xlink:href", "/static/img/ad_bg.svg")
        //     .style("opacity", 0);
        images.append("rect")
            .attr("class", "image-mask")
            .style("opacity", 0)
            .on("click", image_clicked);

        images.on("mouseover", on_image_hover)
            .on("mouseout", on_image_unhover);
        if(isShowSimImage){
            images.append("text")
                .attr("class", "sim_label")
                .text(function (d) {
                    if (d.target_id != null){
                        return SimilarityMatrix[d.target_id][d.id];
                    }
                })
        }

    }
    this.update_detail = function (index, size_mode) {
        var size_width = size_mode == "small" ? image_width_small : image_width;
        var size_height = size_mode == "small" ? image_height_small : image_height;
        var size_margin = size_mode == "small" ? margin_small : margin;
        var size_stroke = size_mode == "small" ? stroke_small : stroke;
        var animation = that.svgs[index].selectAll(".image-node").transition().duration(duration).style("opacity", 1);
        var images = that.svgs[index].selectAll(".image-node");
        animation.selectAll(".image")
            .attr("width", size_width - size_margin * 2)
            .attr("height", size_height - size_margin * 2)
            .attr("x", function (d) { return d.x + size_margin; });
        animation.selectAll(".image-rect")
            .attr("width", size_width - size_margin * 2)
            .attr("height", size_height - size_margin * 2)
            .attr("x", function (d) { return d.x + size_margin; })
            .style("stroke-width", size_stroke)
            .style("stroke", function (d) {
                if (isShowTrueLabel){
                    return CategoryColor[d.cur_label];
                }
                return SelectedGlobal.CategoryColor[d.cur_label];
            });
        animation.selectAll(".image-mark")
            .attr("width", (size_width - margin * 2) / 3)
            .attr("height", (size_height - margin * 2) / 3)
            .attr("x", function (d) { return d.x + margin; });
        animation.selectAll(".image-mask")
            .attr("width", size_width - size_margin * 2)
            .attr("height", size_height - size_margin * 2)
            .attr("x", function (d) { return d.x + size_margin; })
            .style("opacity", function (d) {
                if (MixedIndex.indexOf(d.id) >= 0){
                    return 0;
                }
                return 0.4;
            })
        animation.selectAll(".image")
            .attr("y", function (d) { return d.y + size_margin; });
        animation.selectAll(".image-rect")
            .attr("y", function (d) { return d.y + size_margin; });
        animation.selectAll(".image-mark")
            .attr("y", function (d) { return d.y + margin; });
        animation.selectAll(".image-mask")
            .attr("y", function (d) { return d.y + size_margin; });
        animation.selectAll(".sim_label")
            .attr("x", function (d) { return d.x + size_margin; })
            .attr("y", function (d) { return d.y + size_margin; });
    }
    this.remove_detail = function (index) {
        var images = that.images[index].exit().transition()
            .duration(duration)
            .style("opacity", 0);
        images.remove();
    }

    this.redraw = function () {
        for (var i = 0; i < that.svgs.length; i++){
            that.svgs[i].remove();//.transition().duration(duration).style("opacity", 0)
            container.selectAll(".group-view").remove(); //.transition().duration(duration).style("opacity", 0)
        }
        mode = null;
    }

    this.expand_image = function (d) {
        var nodes = container.selectAll(".image-detail-view");
        if (!d.isShowLarge){

            var node = container.select("#image" + d.id);
            var image_detail_view = node.append("g").attr("class", "image-detail-view").attr("transform", "translate(0," + (d.y + d.height + margin) + ")");
            image_detail_view.append("image").attr("class","image-arrow")
                .attr("x", d.x + d.width / 2 - 10)
                .attr("xlink:href", "/static/img/sort-down.svg");
            image_detail_view.append("rect").attr("width", width).attr("height", 0);
            image_detail_view.append("image").attr("class","image-detail")
                .attr("id", "image-detail")
                .attr("x", (width-image_width_large)/2 + margin)
                .attr("y", margin)
                .attr("width", image_width_large - margin * 2)
                .attr("height", 0)
                .attr("xlink:href", d.url);
            image_detail_view.append("text")
                .style("fill", "white")
                .attr("x", (width+image_width_large)/2 + margin * 2)
                .attr("y", image_height_large-margin*3)
                .style("font-size", 20)
                .text("Crop Image").on("click", function () {
                    on_image_crop(that.crop);
                })
            that.crop = {"id": d.id};
            var brush = d3.brush()
                .extent([[0, 0], [image_width_large - margin*2, image_height_large - margin*2]])
                .on("end", function () {
                    if (d3.event.selection){
                        var crop = d3.event.selection;
                        var my_width = image_width_large - margin*2;
                        that.crop["up_left"] = {
                            "x": crop[0][0] / my_width,
                            "y": crop[0][1] / my_width
                        };
                        that.crop["down_right"] = {
                            "x": crop[1][0] / my_width,
                            "y": crop[1][1] / my_width
                        };
                    }
                });
            node.append("g").attr("transform", "translate(" + ((width-image_width_large)/2 + margin) + "," + (d.y + d.height + margin * 2) + ")").call(brush)
        }
        if (mode != "instance"){
            for (var i = 0; i < that.cur_data.length; i++){
                if (i == d.label && !d.isShowLarge){
                    if (d.size_mode == "small"){
                        layout_small(that.cur_data[i], that.svgs[i], d.id);
                    }else {
                        layout_large(that.cur_data[i], that.svgs[i], d.id)
                    }
                    that.update_detail(i, d.size_mode);
                } else {
                    if (that.cur_data[i][0].size_mode == "small"){
                        layout_small(that.cur_data[i], that.svgs[i]);
                    }else {
                        layout_large(that.cur_data[i], that.svgs[i])
                    }
                    that.update_detail(i, that.cur_data[i][0].size_mode);
                }

            }
        } else{
            if (d.isShowLarge){
                if (d.size_mode == "small"){
                    layout_small(that.cur_data, that.svgs[0]);
                }else {
                    layout_large(that.cur_data, that.svgs[0])
                }
            }else{
                if (d.size_mode == "small"){
                    layout_small(that.cur_data, that.svgs[0], d.id);
                }else {
                    layout_large(that.cur_data, that.svgs[0], d.id)
                }
            }
            that.update_detail(0, d.size_mode)
        }
        //that.remove_image(nodes);
        if (d.isShowLarge){
            that.remove_image(nodes);
            d.isShowLarge = false;
        } else {
            that.remove_image(nodes, d.y + d.height + margin);
            var animation = image_detail_view.transition().duration(duration).attr("transform", "translate(0," + (d.y + d.height + margin) + ")");
            animation.select("rect").attr("height", image_height_large-margin);
            animation.select(".image-detail").attr("height", image_height_large-margin*3);
            d.isShowLarge = true;
        }

    }
    this.remove_image = function (nodes, cur_y) {
        var old_y = 0;
        var animation = nodes.transition().duration(duration).attr("transform", function (d) {
            d.isShowLarge = false;
            old_y = d.y + d.height + margin;
            return "translate(0," + (d.y + d.height + margin) + ")";
        });
        animation.select(".image-arrow").style("opacity", 0);
        if (cur_y && old_y != cur_y){
            animation.select("rect").attr("height", 0);
        }
        animation.select(".image-detail").attr("height", 0);
        animation.remove();
    }

    this.draw_validated_dialog = function (container, index) {
        var data = [];
        for (var i = 0; i < index.length; i++){
            data.push(base_data[index[i]]);
            base_data[index[i]].cur_label = base_data[index[i]].label;
            if (isShowTrueLabel){
                base_data[index[i]].cur_label = TrueLabels[base_data[index[i]].id];
            }
        }
        container.selectAll("div").remove();
        var items = container.selectAll("div").data(data).enter()
            .append("div")
            .attr("class", "validated-dialog-img")
            .attr("id", function (d) {
                return "validated-dialog-img" + d.id;
            })
        items.append("input")
            .attr("type", "checkbox")
            .attr("checked", true)
            .attr("id", function (d) {
                return "validated-dialog-checkbox" + d.id;
            })
        items.append("img")
            .attr("width", 100)
            .attr("height", 100)
            .attr("src", function (d) {
                return d.url;
            })
            .on("click", function (d) {
                if($("#validated-dialog-checkbox" + d.id).is(':checked')){
                    $("#validated-dialog-checkbox" + d.id).prop("checked", false);
                } else {
                    $("#validated-dialog-checkbox" + d.id).prop("checked", true);
                }
            })
    }
}