/**
 * Created by 欧阳方昕 on 2018/3/13.
 */


var HistoryTrail = function (container) {
    var that = this;
    var duration = AnimationDuration;
    var container = container;
    var bbox = container.node().getBoundingClientRect();
    var margin_bottom = 20;
    var width = 1300;
    var height = 150 - margin_bottom; // 28 is title height
    var margin = 10;
    var bar_width = 10;
    var trail_width = bar_width * 2 + margin * 2;
    var svg = container.append("svg")
        .attr("id", "trail-svg")
        .attr("width", bbox.width)
        .attr("height", bbox.height - 28)
        .attr("viewBox", "0, 0, 1300, 150")
        .append("g");
    var zoom = d3.zoom().scaleExtent([1,1])
            .on("zoom", zoomed);

    //TODO: 30
    var y = d3.scaleLinear()
            .domain([0,5])
            .range([0, height]);

    var trailx = 0;
    var selected = -1;

    svg.call(zoom)
        .call(zoom.transform, d3.zoomIdentity
            .scale(1));
    function zoomed() {
        var t = d3.event.transform;
        trailx = t.x;
        svg.transition().duration(duration).attr("transform", function (d) {
            return "translate(" + trailx + ",0)"
        });
    }

    this.resize = function () {
        bbox = container.node().getBoundingClientRect();
        container.select("#trail-svg")
            .attr("width", bbox.width)
            .attr("height", bbox.height - 28);
    }

    this.draw = function () {
        that.create();
        if (SelectedGlobal.Trail.length * trail_width + trailx > width){
            trailx = width - SelectedGlobal.Trail.length * trail_width;
            svg.transition().duration(duration).attr("transform", function (d) {
                return "translate(" + trailx + ",0)"
            });
        }
        svg.selectAll(".trail-node").data(SelectedGlobal.Trail).exit().remove();
    }
    this.create = function () {
        that.trails = svg.selectAll(".trail-node").data(SelectedGlobal.Trail).enter()
            .append("g").attr("class", "trail-node")
            .attr("id", function (d, i) {
                return "trail-node" + i;
            })
            .attr("transform", function (d, i) {
                return "translate(" + i * trail_width + ",0)"
            })
            .on("click", function (d, i) {
                svg.selectAll(".trail-node").classed("selected", false);
                svg.select("#trail-node" + i).classed("selected", true);
                svg.selectAll(".trail-redo").remove();
                svg.append("text")
                    .attr("class", "trail-redo")
                    .attr("dy", "1em")
                    .attr("y", height)
                    .attr("x", trail_width / 2 + i * trail_width)
                    .attr("text-anchor", "middle")
                    .text("redo")
                    .on("click", function () {
                        trail_roll_back(selected);
                    })
                selected = i;
                on_trail_clicked(d);
            });

        that.trails.append("rect")
            .attr("class", "trail-bg")
            .attr("height", height)
            .attr("width", trail_width);


        that.trails.append("rect")
            .attr("width", bar_width)
            .attr("height", function (d) {
                return y(Math.sqrt(d[0].length,0.5));
            })
            .attr("x", margin)
            .attr("y", function (d) {
                return height - y(Math.sqrt(d[0].length,0.5));
            });
        that.trails.append("rect")
            .attr("width", bar_width)
            .attr("height", function (d) {
                return y(Math.sqrt(d[1].length,0.5));
            })
            .attr("x", margin + bar_width)
            .attr("y", function (d) {
                return height - y(Math.sqrt(d[1].length,0.5));
            });

    }

    this.remove = function () {
        svg.selectAll(".trail-node").transition().duration(duration).style("opacity", 0).remove();
    }

}