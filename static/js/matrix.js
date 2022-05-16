  /**
 * Created by 欧阳方昕 on 2018/1/9.
 */

var ConfusionMatrix = function (container, size) {
    var that = this;

    var duration = AnimationDuration;
    var width = size.width;
    var height = size.height;
    var margin = {top: 30, right: 30, bottom: 60, left: 20};
    var label_width = 170;
    var label_height = 120;
    var legend_width = 70;
    var plot_width = width - margin.left - margin.right * 2 - legend_width - label_width;
    var plot_height = height - margin.top - margin.bottom - label_height;
    var max_rect_width = 130;
    var max_plot_width = 0;
    var minplot = 300;
    var mode = 1;

    var min_selected_list = 1;

    var start_color = '#ffffff';


    var end_color = "#808080";//'#e67e22';
    var svg = container.attr("transform", "translate(" + (margin.left + label_width) + "," + margin.top + ")");

    this.selected = [];

    var x = d3.scaleBand()
            .range([0, plot_width]);

    var y = d3.scaleBand()
        .range([0, plot_height]);
    var colorMap = d3.scaleLinear()
            .domain([0,1])
            .range([start_color, end_color]);
    var legendMap = d3.scaleLinear()
            .domain([0,1])
            .range([plot_height, 0]);
    var yAxis = d3.axisRight().scale(legendMap);
    
    this.resize = function (size) {
        width = size.width;
        height = size.height;
        if (mode = 1){
            legend_width = 70;
            plot_width = width - margin.left - margin.right * 2 - legend_width - label_width;
            plot_height = height - margin.top - margin.bottom - label_height;
            var marginleft = 0, margintop = 0;
            if (plot_width > max_plot_width){
                marginleft = (plot_width - max_plot_width) / 2;
                plot_width = max_plot_width;
            }
            if (plot_height > max_plot_width){
                margintop = (plot_height - max_plot_width) / 2;
                plot_height = max_plot_width;
            }

            x.range([0, plot_width]);
            y.range([0, plot_height]);
            legendMap.range([plot_height, 0]);

            svg.transition().duration(duration).attr("transform", "translate(" + (margin.left + label_width + marginleft) + "," + (margin.top + margintop) + ")");
            that.update_matrix()
            that.update_labels();
            that.update_legend();
        }
    }

    this.draw = function (confusionMatrix, classNames) {
        change_bottom(0);
        max_plot_width = max_rect_width * confusionMatrix.length;
        var marginleft = 0, margintop = 0;
        if (plot_width > max_plot_width){
            marginleft = (plot_width - max_plot_width) / 2;
            plot_width = max_plot_width;
        }
        if (plot_height > max_plot_width){
            margintop = (plot_height - max_plot_width) / 2;
            plot_height = max_plot_width;
        }
        x.range([0, plot_width]);
        y.range([0, plot_height]);
        legendMap.range([plot_height, 0]);
        svg.attr("transform", "translate(" + (margin.left + label_width + marginleft) + "," + (margin.top + margintop) + ")");

        that.create_matrix(confusionMatrix);
        that.update_matrix();
        that.create_labels(classNames);
        that.update_labels()
        that.create_legend();
        that.update_legend();
    }

    this.expand_matrix = function () {
        change_bottom(1);
        // propagation();
        mode = 1;
        legend_width = 70;
        plot_width = width - margin.left - margin.right * 2 - legend_width - label_width;
        plot_height = height - margin.top - margin.bottom - label_height;
        var marginleft = 0, margintop = 0;
        if (plot_width > max_plot_width){
            marginleft = (plot_width - max_plot_width) / 2;
            plot_width = max_plot_width;
        }
        if (plot_height > max_plot_width){
            margintop = (plot_height - max_plot_width) / 2;
            plot_height = max_plot_width;
        }

        x.range([0, plot_width]);
        y.range([0, plot_height]);
        legendMap.range([plot_height, 0]);

        svg.transition().duration(duration).attr("transform", "translate(" + (margin.left + label_width + marginleft) + "," + (margin.top + margintop) + ")");
        that.update_matrix()
        that.update_labels();
        that.update_legend();
    }
    this.narrow_matrix = function (size) {
        size = {"width": 150, "height": 120};
        change_bottom(2);
        mode = 2;
        legend_width = size.width * 0.1;
        plot_width = size.width * 0.9;
        plot_height = size.height;

        x.range([0, plot_width]);
        y.range([0, plot_height]);
        legendMap.range([plot_height, 0]);

        var background = svg.append("rect")
            .attr("class", "narrow-matrix")
            .style("opacity", 0)
            .attr("width", plot_width)
            .attr("height", plot_height)
            .on("click", function () {
                on_classes_selected_start();
                that.expand_matrix();
                this.remove();
            })

        svg.transition().duration(duration).attr("transform", "translate(" + (margin.left) + "," + (margin.top) + ")");
        that.update_matrix()
        that.update_labels();
        that.update_legend();
        //setTimeout(that.expand_matrix, 3000);
    }

    this.create_matrix = function (data){
        svg.append("rect")
            .attr("class", "matrix-model")
            .style("opacity", 0)
            .attr("width", 0)
            .attr("height", 0);
        var background = svg.append("rect")
            .attr("class", "matrix-background")
            .style("stroke", "black")
            .style("stroke-width", "2px")
            .style("opacity", 0)
            .attr("width", 0)
            .attr("height", 0);

        var classnum = data.length;

        x.domain(d3.range(classnum));
        y.domain(d3.range(classnum));

        var row = svg.selectAll(".matrix-row")
            .data(data)
            .enter().append("g")
            .attr("class", "matrix-row");

        var cell = row.selectAll(".matrix-cell")
            .data(function(d, i) {
                var cell_data = []
                for (var j = 0; j < d.length; j++){
                    cell_data.push({"value": d[j], "row": i, "cell": j});
                }
                return cell_data;
            }).enter().append("g")
            .attr("class", "matrix-cell")
            .style("fill", function (d) {
                return colorMap(Math.pow(d.value > 0.03? d.value:0, 0.2));
            });

        cell.append('rect')
            .attr("width", 0)
            .attr("height", 0)
            .style("stroke-width", 0);


        cell.append("text")
            .attr("class", "matrix-text-value")
            .attr("dy", ".32em")
            .attr("x", 0)
            .attr("y", 0)
            .attr("text-anchor", "middle")
            .style("opacity", 0)
            .style("fill", function(d, i) { return 'black'; })
            .text(function(d, i) { return d.value; });

        cell.append("rect")
            .attr("class", "matrix-row-bg")
            .attr("width", 0)
            .attr("height", 0)
            .on("click", function (d) {
                if (that.selected.indexOf(d.row) < 0){
                    that.selected.push(d.row);
                    svg.selectAll(".matrix-label" + d.row).classed("selected", true);
                }
                if (that.selected.indexOf(d.cell) < 0){
                    that.selected.push(d.cell);
                    svg.selectAll(".matrix-label" + d.cell).classed("selected", true);
                }
                if (that.selected.length > min_selected_list){
                    change_bottom(1);
                } else {
                    change_bottom(0);
                }
            });

    }
    this.update_matrix = function(data){
        svg.selectAll(".matrix-model").transition().duration(duration)
            .attr("width", plot_width + margin.left + label_width)
            .attr("height", plot_height + margin.top + label_height)
            .attr("transform", "translate(" + (-margin.left - label_width) + "," + (-margin.top) + ")");
        var background = svg.selectAll(".matrix-background").transition().duration(duration)
            .style("opacity", 1)
            .attr("width", plot_width)
            .attr("height", plot_height);

        var row = svg.selectAll(".matrix-row").transition().duration(duration)
            .attr("transform", function(d, i) { return "translate(0," + y(i) + ")"; });

        var cell = row.selectAll(".matrix-cell")
            .attr("transform", function(d, i) { return "translate(" + x(i) + ", 0)"; });

        cell.selectAll('rect')
            .attr("width", x.bandwidth())
            .attr("height", y.bandwidth());

        cell.selectAll("text")
            .attr("x", x.bandwidth() / 2)
            .attr("y", y.bandwidth() / 2);
        cell.select(".matrix-row-bg")
            .attr("width", x.bandwidth())
            .attr("height", y.bandwidth());
        if (plot_width > minplot && plot_height > minplot){
            cell.selectAll("text").style("opacity", 1);
        }
        else {
            cell.selectAll("text").style("opacity", 0);
        }
        if (data){
            var row1 = svg.selectAll(".matrix-row").data(data);

            var cell1 = row1.selectAll(".matrix-cell").data(function(d, i) {
                var cell_data = []
                for (var j = 0; j < d.length; j++){
                    cell_data.push({"value": d[j], "row": i, "cell": j});
                }
                return cell_data;
            }).transition().duration(duration)
                .style("fill", function (d) {
                    return colorMap(d.value);
                }).select("text").text(function(d, i) { return d.value; });
        }
    }

    this.create_labels = function (data) {
        var labels = svg.append('g')
            .attr('class', "matrix-labels")
            .style("opacity", 0);

        d3.select(".select-buttom")
            .on("click", function () {
                that.narrow_matrix();
                on_classes_selected_end(that.selected);
            });

        var columnLabels = labels.selectAll(".column-label")
            .data(data)
            .enter().append("g")
            .attr("class", "column-label");

        columnLabels.append("line")
            .style("stroke", "black")
            .style("stroke-width", "1px")
            .attr("x1", 0)
            .attr("x2", 0)
            .attr("y1", 0)
            .attr("y2", 5);

        columnLabels.append("text")
            .attr("class", function (d, i) {
                return "matrix-label" + i;
            })
            .attr("x", 0)
            .attr("y", 0)
            .attr("dy", ".82em")
            .attr("text-anchor", "start")
            .attr("transform", "translate(0,8) rotate(30)")
            .style("fill", function (d, i) { return CategoryColor[i];})
            .text(function(d, i) { return d; })
            .on("click", function (d, i) {
                if (that.selected.indexOf(i) < 0){
                    that.selected.push(i);
                    svg.selectAll(".matrix-label" + i).classed("selected", true);
                } else {
                    that.selected.splice(that.selected.indexOf(i), 1)
                    svg.selectAll(".matrix-label" + i).classed("selected", false);
                }
                if (that.selected.length > min_selected_list){
                    change_bottom(1);
                } else {
                    change_bottom(0);
                }
            });

        var rowLabels = labels.selectAll(".row-label")
            .data(data)
            .enter().append("g")
            .attr("class", "row-label");

        rowLabels.append("line")
            .style("stroke", "black")
            .style("stroke-width", "1px")
            .attr("x1", 0)
            .attr("x2", -5)
            .attr("y1", 0)
            .attr("y2", 0);

        rowLabels.append("text")
            .attr("class", function (d, i) {
                return "matrix-label" + i;
            })
            .attr("x", -18)
            .attr("y", 0)
            .attr("dy", ".32em")
            .attr("text-anchor", "end")
            .style("fill", function (d, i) { return CategoryColor[i];})
            .text(function(d, i) { return d; })
            .on("click", function (d, i) {
                if (that.selected.indexOf(i) < 0){
                    that.selected.push(i);
                    svg.selectAll(".matrix-label" + i).classed("selected", true);
                } else {
                    that.selected.splice(that.selected.indexOf(i), 1)
                    svg.selectAll(".matrix-label" + i).classed("selected", false);
                }
                if (that.selected.length > min_selected_list){
                    change_bottom(1);
                } else {
                    change_bottom(0);
                }
            });

        rowLabels.append("image")
            .attr("class", function (d, i) {
                return "matrix-label" + i;
            })
            .attr("x", -18)
            .attr("y", y.bandwidth() / 2 - 8)
            .attr("xlink:href", "/static/img/close.svg")
            .style("opacity", 1)
            .on("click", function (d, i) {
                if (that.selected.indexOf(i) >= 0){
                    that.selected.splice(that.selected.indexOf(i), 1)
                    svg.selectAll(".matrix-label" + i).classed("selected", false);
                }
                if (that.selected.length > min_selected_list){
                    change_bottom(1);
                } else {
                    change_bottom(0);
                }
            });


    }
    this.update_labels = function () {
        var labels = svg.selectAll('.matrix-labels').transition().duration(duration);
        if (plot_width > minplot && plot_height > minplot){
            labels.style("opacity", 1);
        }
        else {
            labels.style("opacity", 0);
        }

        var columnLabels = labels.selectAll(".column-label")
            .attr("transform", function(d, i) { return "translate(" + x(i) + "," + plot_height + ")"; });

        columnLabels.selectAll("line")
            .attr("x1", x.bandwidth() / 2)
            .attr("x2", x.bandwidth() / 2);

        columnLabels.selectAll("text")
            .attr("transform", "translate(" + (x.bandwidth() / 2) + ",8) rotate(30)");

        var rowLabels = labels.selectAll(".row-label")
            .attr("transform", function(d, i) { return "translate(" + 0 + "," + y(i) + ")"; });

        rowLabels.selectAll("line")
            .attr("y1", y.bandwidth() / 2)
            .attr("y2", y.bandwidth() / 2);

        rowLabels.selectAll("text")
            .attr("y", y.bandwidth() / 2);
    }

    this.create_legend = function () {
        var legend = svg.append("defs").append("svg:linearGradient")
            .attr("id", "gradient")
            .attr("x1", "100%")
            .attr("y1", "0%")
            .attr("x2", "100%")
            .attr("y2", "100%")
            .attr("spreadMethod", "pad");

        legend
            .append("stop")
            .attr("offset", "0%")
            .attr("stop-color", end_color)
            .attr("stop-opacity", 1);

        legend
            .append("stop")
            .attr("offset", "100%")
            .attr("stop-color", start_color)
            .attr("stop-opacity", 1);

        svg.append("rect").attr("class", "matrix_legend")
            .attr("width", 0)
            .attr("height", 0)
            .style("fill", "url(#gradient)");

        svg.append("text").attr("class", "matrix_legend_label")
            .text("Confusing Degree");

        svg.append("g")
        .attr("class", "y axis")
        .attr("transform", "scale(0)");

    }
    this.update_legend = function () {
        var legend = svg.selectAll(".matrix_legend").transition().duration(duration)
            .attr("width", legend_width)
            .attr("height", plot_height)
            .attr("transform", "translate(" + (plot_width + margin.right) + ", 0)");
        var label = svg.selectAll(".matrix_legend_label").transition().duration(duration).attr("transform", "translate(" + (plot_width + margin.right - 15) + ", -10)");
        var axis = svg.selectAll(".axis").transition().duration(duration)
        .attr("transform", "scale(1) translate("+ (plot_width + margin.right + legend_width) + ", 0)").call(yAxis);

        if (plot_width > minplot && plot_height > minplot){
            legend.style("opacity", 1);
            label.style("opacity", 1);
            axis.style("opacity", 1);
        }
        else {
            legend.style("opacity", 0);
            axis.style("opacity", 0);
            label.style("opacity", 0);
        }
    }
}
