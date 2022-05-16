var Rankings = function (container, data) {
    var that = this;

    var container = container;
    var bbox = container.node().getBoundingClientRect();
    var width = bbox.width;
    var height = bbox.height - 28; // 28 is title height

    var svg = container.append("svg");

    this.init = function () {
        svg.attr("width", width)
            .attr("height", height);
        var node_config = {
            "node_width": 200,
            "node_height": 20
        };
        var link_config = {
            "node_height": 20,
            "link_width" : 300
        }

        var bi_clustering_data = data;
        that.workers = bi_clustering_data.worker_cluster;
        var workers_container = svg.append('g').attr('id', "workers_container").attr("transform", "translate(" + 100 + "," + 100 + ")");
        workers_container.append("text")
            .attr('class', "title")
            .attr("x", 50)
            .text("worker");
        that.workers_view = new RankingsNode(workers_container, that.workers, node_config);
        that.workers_view.init();

        that.instances = bi_clustering_data.instance_cluster;;
        var instances_container = svg.append('g').attr('id', "instances_container").attr("transform", "translate(" + 600 + "," + 100 + ")");
        instances_container.append("text")
            .attr('class', "title")
            .attr("x", 50)
            .text("instance");
        that.instances_view = new RankingsNode(instances_container, that.instances, node_config);
        that.instances_view.init();

        var links_container = svg.append('g').attr('id', "links_container").attr("transform", "translate(" + 300 + "," + 100 + ")");
        that.links_view = new RankingsLink(links_container, that.workers_view, that.instances_view, link_config);
    }.call();

    this.draw = function () {
        that.workers_view.draw();
        that.instances_view.draw();
        that.links_view.draw();
    }
}

var RankingsNode  = function (container, base_data, config) {
    var that = this;
    this.detail_data = [];

    var container = container;
    var base_data = base_data;

    var node_height = config.node_height;
    var node_width = config.node_width;

    var get_detail_data = function (data) {
        for (var i = 0; i < data.length; i++){
            var node = data[i];
            if (node.expand && node.children){
                get_detail_data(node.children);
            } else {
                that.detail_data.push(node);
            }
        }
    }

    var ranking = function (data) {
        data.sort(function (a, b) {
            return b.weight - a.weight;
        });
        for (var i = 0; i < data.length; i++){
            var node = data[i];
            if (node.children){
                ranking(node.children)
            }
        }
    }

    var layout = function () {
        var start = 10;
        for (var i = 0; i < that.detail_data.length; i++){
            that.detail_data[i].x = 10 * that.detail_data[i].level;
            that.detail_data[i].y = start;
            start += node_height + 5;
        }
    }

    var create_box_plot = function(nodes){
        nodes.append("rect")
            .style("fill", "steelblue")
            .attr("x", function (d) {
                return d.x;
            })
            .attr("y", 0)
            .attr("height", node_height)
            .attr("width", function (d) {
                return node_width * d.weight;
            })

        nodes.append("line")
            .attr("class", "start-line")
            .style("opacity", function (d) {
                if (d.children){
                    return 1;
                }
                return 0;
            })
            .attr("x1", function (d) {
                if (d.min){
                    return node_width * d.min + d.x;
                }
                return 0;
            })
            .attr("x2", function (d) {
                if (d.min){
                    return node_width * d.min + d.x;
                }
                return 0;
            })
            .attr("y1",0)
            .attr("y2", node_height);
        nodes.append("line")
            .attr("class", "end-line")
            .style("opacity", function (d) {
                if (d.children){
                    return 1;
                }
                return 0;
            })
            .attr("x1", function (d) {
                if (d.max){
                    return node_width * d.max + d.x;
                }
                return 0;
            })
            .attr("x2", function (d) {
                if (d.max){
                    return node_width * d.max + d.x;
                }
                return 0;
            })
            .attr("y1",0)
            .attr("y2", node_height);
        nodes.append("line")
            .attr("class", "center-line")
            .style("opacity", function (d) {
                if (d.children){
                    return 1;
                }
                return 0;
            })
            .attr("x1", function (d) {
                if (d.min){
                    return node_width * d.min + d.x;
                }
                return 0;
            })
            .attr("x2", function (d) {
                if (d.max){
                    return node_width * d.max + d.x;
                }
                return 0;
            })
            .attr("y1",function (d) {
                return node_height / 2;
            })
            .attr("y2", function (d) {
                return node_height / 2;
            });
    }

    this.init = function () {
        ranking(base_data);
    }

    this.draw = function () {
        that.detail_data = [];
        get_detail_data(base_data);
        layout();
        that.create();
        that.update();
        that.remove();
    }
    this.create = function () {
        that.nodes = container.selectAll("g").data(that.detail_data, function (d) {
            return d.type + d.id;
        })
        var nodes = that.nodes.enter().append("g")
            .style("opacity", 0)
            .attr("id", function (d) {
                return "ranking-" + d.type + "-" + d.id;
            })
            .on("click", function (d) {
                var cur = base_data;
                for (var i = 0; i <= d.level; i++){
                    for (var j = 0; j < cur.length; j++){
                        if (cur[j].expand){
                            if (i == d.level){
                                cur[j].expand = false;
                            } else {
                                cur = cur[j].children;
                                break;
                            }
                        }
                    }
                }
                d.expand = true;
                update_ranking_view();
            })
            .on("mouseover", function (d) {
                d3.selectAll(".link").style("display", "none");
                d3.selectAll("." + d.type + d.id).style("display", "block");
            })
            .on("mouseout", function (d) {
                d3.selectAll(".link").style("display", "none");
            });
        create_box_plot(nodes);
    }

    this.update = function() {
        var nodes = that.nodes.transition()
            .duration(1000)
            .style("opacity", function(d) {
                return 1;
            });
        nodes.selectAll("rect").attr("y", function (d) {
            return d.y;
        });
        nodes.selectAll(".start-line")
            .attr("y1", function (d) {
                return d.y;
            })
            .attr("y2", function (d) {
                return d.y + node_height;
            });
        nodes.selectAll(".end-line")
            .attr("y1", function (d) {
                return d.y;
            })
            .attr("y2", function (d) {
                return d.y + node_height;
            });
        nodes.selectAll(".center-line")
            .attr("y1", function (d) {
                return d.y + node_height / 2;
            })
            .attr("y2", function (d) {
                return d.y + node_height / 2;
            });
    }
    
    this.remove = function () {
        var nodes = that.nodes.exit().transition()
            .duration(1000)
            .style("opacity", 0);
        nodes.remove();
    }
}

var RankingsLink = function (container, source_view, target_view, config) {
    var that = this;
    var container = container;
    var source_view = source_view;
    var target_view = target_view;
    var base_data = null;
    var link_width = config.link_width;
    var node_height = config.node_height;

    var line = d3.svg.line()
        .x(function (d) { return d.x; })
        .y(function (d) { return d.y; })
        .interpolate("linear");
    var diagonal = d3.svg.diagonal()
        .projection(function(d) { return [d.y, d.x]; });
    this.draw = function () {
        base_data = ranking_workers_instances_processing(source_view.detail_data, target_view.detail_data);
        that.create();
        that.update();
    }
    this.create = function () {
        that.links = container.selectAll(".link").data(base_data, function (d) {
            return d.source.type + d.source.id + d.target.type + d.target.id;
        })
        var links = that.links.enter().append("g")
            //.style("display", "none")
            .style("opacity", 0)
            .attr("id", function (d) {
                return "ranking-" + d.source.type + d.source.id + d.target.type + d.target.id;
            })
            .attr("class", function (d) {
               return "link " + d.source.type + d.source.id + " " + d.target.type + d.target.id;
            });
        links.append("path")
            .attr("class", "links")
            .style("stroke-width", function (d) {
                return 1;//Math.log(d.weight) + 1;//CategoryColor[d.type - 1];
            })
            .style("stroke", function (d) {
                if (!d.source.children && !d.target.children){
                    for (var i = 0; i < d.type.length; i++){
                        if (d.type[i] != 0){
                            return CategoryColor[i];
                        }
                    }
                }
                return "gray";
            })
            .attr("d",
                function(d) {
                    var o = {
                        x: 0,
                        y: d.source.y + node_height / 2
                    };
                    return line([o, o]);
                });
    }

    this.update = function() {
        var links = that.links.transition()
            .duration(1000)
            .style("opacity", function(d) {
                return 1;
            });
        links.selectAll("path").attr("d",
                function(d) {
                    var o = {
                        y: 0,
                        x: d.source.y + node_height / 2
                    };
                    var t = {
                        y: link_width,
                        x: d.target.y + node_height / 2
                    };
                    return diagonal({source: o, target: t});
                });
    }
}