/**
 * Created by 欧阳方昕 on 2018/3/1.
 */

var WorkerDialog = function (container, rangewidth, rangeheight) {
    var that = this;
    var duration = AnimationDuration;
    var svg = container;
    var diagwidth = 380;
    var diagheight = 240;
    var barwidth = 190;
    var barheight = 190;
    var legend_width = 15;
    var margin = {top: 15, right: 10, bottom: 20, left: 15};
    var label_width = 100;
    this.max_worker_instance = 100;
    var rangewidth = rangewidth;
    var rangeheight = rangeheight;
    var start_color = '#ffffff';
    var end_color = "#808080";//'#e67e22';
    // svg.append("defs").append("pattern")
    //     .attr("id", "diagonalHatch")
    //     .attr("patternUnits","userSpaceOnUse")
    //     .attr("width", 10)
    //     .attr("height", 10)
    //     .append("path")
    //     .attr("d","M0,10 l10,-10 M-2, 2l4, -4 M8,12l4,-4")
    //     .style("stroke", "808080");

    var drag = d3.drag().on("drag", dragged)
    function dragged(d) {
        d.x0 += d3.event.dx;
        d.y0 += d3.event.dy;
        d3.select(this).attr("transform", "translate(" + d.x0 + "," + d.y0 + ")");
    }

    var colorMap = d3.scaleLinear()
            .domain([0,1])
            .range([start_color, end_color]);
    var legendMap = d3.scaleLinear()
            .domain([0,1])
            .range([barheight, 0]);
    var yAxis = d3.axisRight().scale(legendMap);

    this.resize = function (width, height) {
        rangewidth = width;
        rangeheight = height;
    }

    this.create = function (data, x, y) {
        if(document.getElementById("worker-dialog" + data.id)) {
            return;
        }
        if (x + diagwidth > rangewidth) x -= diagwidth;
        if (y + barheight > rangeheight) y -= diagheight;
        var diag = svg.selectAll("#worker-dialog" + data.id).data([{"x0": x, "y0": y}]).enter().append("g")
            .attr("class", "worker-dialog-view")
            .attr("id", "worker-dialog" + data.id)
            .attr("transform", "translate(" + x + "," + y + ")")
            .call(drag);
        diag.append("rect")
            .attr("class", "worker-dialog-bg")
            .attr("width", diagwidth)
            .attr("height", diagheight);
        var close = diag.append("g").attr("class", "worker-dialog-close")
            .on("click", function () {
                that.remove(data);
            });
        close.append("rect")
            .attr("class", "worker-dialog-close-bg")
        close.append("image")
            .attr("class", "worker-dialog-close-icon")
            .attr("xlink:href", "/static/img/close.svg")
            .style("opacity", 1);

        // var bar = diag.append("g").attr("class", "worker-dialog-barchart")
        //     .attr("transform", "translate(" + 30 + "," + 10 + ")");
        // that.draw_barchart(bar, data.instances);

        var matrix = diag.append("g").attr("class", "worker-dialog-matrix")
            .attr("transform", "translate(" + (margin.left + label_width) + "," + margin.top + ")");
        that.draw_matrix(matrix, data.instances, data.id);
        that.update_matrix(data.id);
    }

    this.draw_barchart = function (view, data) {

        var x = d3.scaleBand().rangeRound([0, barwidth]).padding(0.2),
        y = d3.scaleLinear().rangeRound([barheight, 0]);


        x.domain(SelectedGlobal.LabelNames);
        y.domain([0, that.max_worker_instance]);

        view.append("g")
            .attr("class", "axis axis--x")
            .attr("transform", "translate(0," + barheight + ")")
            .call(d3.axisBottom(x));

        view.append("g")
            .attr("class", "axis axis--y")
            .call(d3.axisLeft(y))
            .append("text")
            .attr("transform", "rotate(-90)")
            .attr("y", 6)
            .attr("dy", "0.71em")
            .attr("text-anchor", "end")
            .text("");

        view.selectAll(".axis--x").selectAll("text").attr("transform", "translate(" + x.bandwidth() / 2 + ", 5) rotate(15)")

        var bardata = [];
        var selectedIndex = {};
        for (var i = 0; i < SelectedList.length; i++){
            selectedIndex[SelectedList[i]] = i;
        }
        selectedIndex["other"] = SelectedList.length;
        for (var i = 0; i < data.length; i++) {
            var count = {};
            for (var j = 0; j < data.length; j++){
                count[j] = [];
            }
            for (var j = 0; j < data[i].length; j++){
                var label = SelectedGlobal.PosteriorLabels[data[i][j]];
                if (SelectedList.includes(label)){
                    count[selectedIndex[label]].push(data[i][j]);
                } else{
                    count[selectedIndex.other].push(data[i][j]);
                }
            }
            var temp = [{"state": i, "classlabel": i, "color": SelectedGlobal.CategoryColor[i], "start": 0, "end":count[i].length, "instances": count[i]}];
            var start = count[i].length;
            for (var j = 0; j < data.length; j++){
                if (i != j){
                    temp.push({"state": i, "classlabel": j, "color": SelectedGlobal.CategoryColor[j], "start":start, "end": start + count[j].length,  "instances": count[j]});
                    start += count[j].length;
                }
            }
            bardata.push(temp);
        }
        var false_bar = [];
        view.selectAll(".worker-dialog-bar").data(bardata).enter().append("g")
            .attr("class", "worker-dialog-bar")
            .selectAll(".worker-dialog-bar-rect")
            .data(function(d) { return d; })
            .enter().append("rect")
            .attr("class", "worker-dialog-bar-rect")
            .attr("x", function(d) { return x(SelectedGlobal.LabelNames[d.state]); })
            .attr("y", function(d) { return y(d.end); })
            .attr("width", x.bandwidth())
            .attr("height", function(d) { return y(d.start) - y(d.end); })
            .style("fill", function(d) { return d.color; })
            .style("fill-opacity", function(d) {
                if (d.state == d.classlabel) return 1;
                false_bar.push(d);
                return 0.5;
            })
            .on("mouseover", on_worker_bar_hover)
            .on("mouseout", on_worker_bar_unhover)
            .on("click", on_worker_bar_clicked);
        view.selectAll(".worker-dialog-bar-false-rect")
            .data(false_bar)
            .enter().append("rect")
            .attr("class", "worker-dialog-bar-false-rect")
            .attr("x", function(d) { return x(SelectedGlobal.LabelNames[d.state]); })
            .attr("y", function(d) { return y(d.end); })
            .attr("width", x.bandwidth())
            .attr("height", function(d) { return y(d.start) - y(d.end); })
            .style("fill", "url(#diagonalHatch)")
            .on("mouseover", on_worker_bar_hover)
            .on("mouseout", on_worker_bar_unhover)
            .on("click", on_worker_bar_clicked);
    }

    this.draw_matrix = function (view, data, worker_id) {

        var x = d3.scaleBand()
            .range([0, barwidth]);

        var y = d3.scaleBand()
            .range([0, barheight]);

        var classNames = SelectedGlobal.LabelNames;
        if (SelectedGlobal.LabelNames[SelectedGlobal.LabelNames.length-1] == "Others"){
            classNames = SelectedGlobal.LabelNames.slice(0, -1);
        }
        x.domain(classNames);
        y.domain(classNames);

        view.append("g")
            .attr("class", "axis axis--x")
            .attr("transform", "translate(0," + barheight + ")")
            .call(d3.axisBottom(x))
            .selectAll("text").attr("transform", "translate(0, 6) rotate(15)")
            .attr("class", "worker-dialog-matrix-label")
            .style("fill", function (d, i) {
                return SelectedGlobal.CategoryColor[i];
            });
            // .on("mouseover", function (d, i) {
            //     on_worker_bar_hover({"instances": data[i]});
            // })
            // .on("mouseout", function (d, i) {
            //     on_worker_bar_unhover({"instances": data[i]});
            // })
            // .on("click", function (d, i) {
            //     on_worker_bar_clicked({"instances": data[i]});
            // });

        view.append("g")
            .attr("class", "axis axis--y")
            .call(d3.axisLeft(y))
            .selectAll("text")
            .attr("class", "worker-dialog-matrix-label")
            .attr("id", function (d, i) {
                return "worker-dialog-matrix-label_" + worker_id + "_" + SelectedList[i];
            })
            .style("fill", function (d, i) {
                return SelectedGlobal.CategoryColor[i];
            })
            .on("mouseover", function (d, i) {
                on_worker_bar_hover({"instances": data[i]});
            })
            .on("mouseout", function (d, i) {
                on_worker_bar_unhover({"instances": data[i]});
            })
            .on("click", function (d, i) {
                on_worker_bar_clicked({"instances": data[i]});
            });

        view.selectAll(".domain").remove();


        var bardata = [];
        var selectedIndex = {};
        for (var i = 0; i < SelectedList.length; i++){
            selectedIndex[SelectedList[i]] = i;
        }
        selectedIndex["other"] = SelectedList.length;
        for (var i = 0; i < classNames.length; i++) {
            var count = {};
            for (var j = 0; j < classNames.length; j++){
                count[j] = [];
            }
            for (var j = 0; j < data[i].length; j++){
                var label = SelectedGlobal.PosteriorLabels[data[i][j]];
                if (SelectedList.includes(label)){
                    count[selectedIndex[label]].push(data[i][j]);
                }
                // else{
                //     count[selectedIndex.other].push(data[i][j]);
                // }
            }
            var temp = [];
            for (var j = 0; j < classNames.length; j++){
                temp.push({"color": SelectedGlobal.CategoryColor[j], "instances": count[j], "value": 1});
            }
            bardata.push(temp);
        }

        // var custom_matrix = [[0.001, 0.001, 0.001, 0.01],
        //                     [0.001, 0.001, 0.001, 0.001],
        //                     [1.001, 0.98, 1.001, 0.99],
        //                     [0.001, 0.02, 0.001, 0.001]];

        // var custom_matrix = [[0.95, 0.04, 0.001, 0.001],
        //                     [0.05, 0.93, 0.001, 0.001],
        //                     [0.001, 0.001, 0.031, 0.001],
        //                     [0.001, 0.031, 0.97, 1.001]];

        for (var i = 0; i < bardata[0].length; i++) {
            var sum = 0;
            for (var j = 0; j < bardata.length; j++) {
                sum += bardata[j][i].instances.length;
            }
            for (var j = 0; j < bardata.length; j++) {
                bardata[j][i].value = sum == 0? 0 : (bardata[j][i].instances.length / sum).toFixed(2);
                // bardata[j][i].value = custom_matrix[j][i].toFixed(2);
            }
        }

        var background = view.append("rect")
            .attr("class", "worker-dialog-matrix-background")
            .style("stroke", "black")
            .style("stroke-width", "1px")
            .style("fill", "none")
            .attr("width", barwidth)
            .attr("height", barheight);

        var row = view.selectAll(".worker-dialog-matrix-row")
            .data(bardata)
            .enter().append("g")
            .attr("class", "worker-dialog-matrix-row")
            .attr("transform", function(d, i) { return "translate(0," + y(classNames[i]) + ")"; });

        var cell = row.selectAll(".worker-dialog-matrix-cell")
            .data(function(d) { return d; })
                .enter().append("g")
            .attr("class", "worker-dialog-matrix-cell")
            .style("fill", function (d) {
                return colorMap(d.value);
            })
            .attr("transform", function(d, i) { return "translate(" + x(classNames[i]) + ", 0)"; })
            .on("mouseover", on_worker_bar_hover)
            .on("mouseout", on_worker_bar_unhover)
            .on("click", on_worker_bar_clicked);

        cell.append('rect')
            .attr("width", x.bandwidth())
            .attr("height", y.bandwidth())
            .style("stroke-width", 0);


        cell.append("text")
            .attr("class", "worker-dialog-matrix-text-value")
            .attr("dy", ".32em")
            .attr("x", x.bandwidth() / 2)
            .attr("y", y.bandwidth() / 2)
            .style("font-size", 16-classNames.length)
            .attr("text-anchor", "middle")
            .style("fill", "black")
            // attr("opacity",0)
            .text(function(d) { return d.value; });

        view.append("rect").attr("class", "worker-dialog-matrix_legend")
            .attr("width", legend_width)
            .attr("height", barheight)
            .attr("transform", "translate(" + (barwidth + margin.right) + ", 0)")
            .style("fill", "url(#gradient)");

        // view.append("text").attr("class", "worker-dialog-matrix_legend_label")
        //     .attr("transform", "translate(" + (barwidth + margin.right - 15) + ", -10)")
        //     .text("Confusing Degree");

        view.append("g")
        .attr("class", "y axis")
        .attr("transform", "translate("+ (barwidth + margin.right + legend_width) + ", 0)").call(yAxis);

        view.append("text").text(worker_id);

    }

    this.update_matrix = function (id) {
        var animation = svg.select("#worker-dialog" + id).transition().duration(duration);
        animation.selectAll(".worker-dialog-matrix-row")
            .style("opacity", function (d, i) {
                if (SpammerMenu.spammer_list.hasOwnProperty(id)){
                    if (SpammerMenu.spammer_list[id].indexOf(SelectedList[i]) >= 0){
                        return 0.3;
                    }
                }
                return 1;
            });
        animation.select(".axis--y").selectAll(".worker-dialog-matrix-label")
            .style("opacity", function (d, i) {
                if (SpammerMenu.spammer_list.hasOwnProperty(id)){
                    if (SpammerMenu.spammer_list[id].indexOf(SelectedList[i]) >= 0){
                        return 0.3;
                    }
                }
                return 1;
            });
        animation.select(".axis--x").selectAll(".worker-dialog-matrix-label")
            .style("opacity", function (d, i) {
                if (SpammerMenu.spammer_list.hasOwnProperty(id)){
                    if (SpammerMenu.spammer_list[id].indexOf(SelectedList[i]) >= 0){
                        return 0.3;
                    }
                }
                return 1;
            });
    }

    this.update = function (id) {
        that.update_matrix(id);
    }

    this.remove = function (data) {
        svg.select("#worker-dialog" + data.id).remove();
        var index = SelectedWorkers.indexOf(data);
        if (index >= 0){
            SelectedWorkers.splice(index, 1);
        }
        on_worker_selected(SelectedWorkers);
    }
    this.removeAll = function () {
        svg.select(".worker-dialog-view").remove();
    }

    this.__init = function () {

    }.call();
}