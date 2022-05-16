var debug = function (container) {
    var that = this;
    that.container = container;

    var bbox = that.container.node().getBoundingClientRect();
    var width = bbox.width;
    var height = bbox.height - 28; // 28 is title height
    var plot_width = 800;
    var plot_height = 800;
    var plot_x_shift = 20;
    var plot_y_shift = 20;

    var svg = that.container.append("svg");
    var plot_group = svg.append("g").attr("id", "image-group");

    this.init = function () {
        // set up svg's width and height
        svg.attr("width", width)
            .attr("height", height);

        //set up groups positions
        plot_group.attr("transform", "translate(" + plot_x_shift + "," + plot_y_shift + ")");
    }.call();

    this.draw = function () {
        that.draw_background();
        that.update();
    };

    this.draw_background = function () {

    };

    this.update = function (id) {
        if (!id) {
            id = 0;
        }
        plot_group.selectAll("image.debug")
            .data([id])
            .enter()
            .append("image")
            .attr("class", "debug");
        plot_group.selectAll("image.debug")
            .data([id])
            .attr("id", "ID-" + id)
            .attr("width", plot_width)
            .attr("height", plot_height)
            .attr("x", plot_x_shift)
            .attr("y", plot_y_shift)
            .attr("xlink:href", ImageUrl[id]);
        plot_group.selectAll("image.debug")
            .data([id])
            .exit()
            .remove();
    };
};