var bindsim = {
    // Bindsim internal globals
    i: {
        endpoint: "api", // Simulator backend root endpoint
        plot_id: "plot", // ID of plot container div
        axis_isotherm: "axis-isotherm",
        axis_molefrac: "axis-molefrac",
        forms: "#params-nmr-1to1, #params-nmr-1to2",
        button: "#button-plot",
        control: "#control",
        selector: "#selector",
        $forms: {},
        $button: {}
    },
         
    on_ready: function() {
        /**
         * Initialisation function, called on document ready
         */
        console.log("bindsim.on_ready called");

        // Initialise internal jQuery objects
        // TODO: is this ok to do, or should the globals just be string selectors?
        bindsim.i.$forms = $(bindsim.i.forms); // All available param forms
        bindsim.i.$button = $(bindsim.i.button); // Plot button

        // Initialise default simulator plot
        bindsim[$(bindsim.i.selector).val()].init();

        // Trigger replot on Enter in any form
        $(bindsim.i.control+" :input").on("keyup", function(event) {
            //if(event.keyCode == 13) {
                bindsim.i.$button.click();
            //}
        });
        
        // Handle simulator selector dropdown
        $(bindsim.i.selector).on("change", bindsim.handle_selector);
    },

    handle_selector: function() {
        console.log("bindsim.handle_selector called");
        
        // On selection change, run selected simulator's init function
        // (Dropdown selection value is set to the name of the simulator object)
        selection = $(this).val();
        bindsim[selection].init();
    },

    bindsim_call: function(request, endpoint, on_done, on_fail) {
        /**
         * Call bindsim backend and execute specified callback on completion
         *
         * @param {Object} request - Appropriately structured request object 
         *  for bindsim backend API
         * @param {callback} on_done - Callback to execute on call success, 
         *  passed backend response and request object
         * @param {callback} on_faile - Callback to execute on call failure, 
         *  passed backend response and request object
         */
        
        var jqxhr = $.ajax({
            url: endpoint,
            type: "POST",
            data: JSON.stringify(request),
            contentType: "application/json; charset=utf-8",
            dataType: "json"
        });

        jqxhr.done(function(data) {
            on_done(data, request);
        });

        jqxhr.fail(function(data) {
            on_fail(data, request);
        });
    },

    params_to_json: function($form, on_fail) {
        /**
         * Parses control form into a JS Object for passing to bindsim backend
         *
         * @param {callback} on_fail - Callback to execute on form parsing failure
         *
         * @returns {Object} - Parsed form contents in format expected by bindsim backend
         */

        // Don't serialize empty fields
        // TODO why does $form.filter(":input") not work here?
        var $params = $($form.selector+" :input").filter(
                function(index, element) {
                    return $(element).val() != "";
                });

        var params = {};
        $params.serializeArray().map(function(x){params[x.name] = x.value;}); 
        
        return params;
    },



    // 
    // Failure functions
    //
    parse_fail: function() {
        // Form parsing failure
        console.log("ERROR: Failed to parse form data, check your input")
    },

    backend_fail: function(data, request) {
        // Backend failure
        console.log("ERROR: Backend error")
    },



    //
    // Highcharts
    // 
    plot_setup: function(s) {
        /**
         * Initialise a Highcharts chart in container #bindsim.i.plot_id
         *
         * @param {string} s - Series options (see Highcharts docs for format)
         *
         * @returns {Object} - Highcharts chart object
         */

        var chart = new Highcharts.Chart({
            chart: {
                renderTo: bindsim.i.plot_id,
                backgroundColor:null,
                style: {'font-family': 'Lato, Helvetica, Arial, Verdana', 'text-transform': 'none'}
            },
            title: {
                text: "",
            },
            subtitle: {
                text: "",
            },
            xAxis: {
                title: {
                    text: "Equivalent total [G]\u2080/[H]\u2080"
                },
                labels: {
                    format: "{value}"
                }
            },
            yAxis: [{ // Primary y axis
                id: bindsim.i.axis_isotherm,
                title: {
                    text: "\u0394\u03B4 (ppm or Hz)",
                    style: {'text-transform': 'none'}
                },
                labels: {
                    format: "{value}"
                },
                minPadding: 0,
                maxPadding: 0,
                startOnTick: false,
                endOnTick: false
            }, { // Secondary y axis
                id: bindsim.i.axis_molefrac,
                title: {
                    text: "Molefraction",
                    style: {'text-transform': 'none'}
                },
                labels: {
                    format: "{value}"
                },
                opposite: true,
                min: 0,
                max: 1,
                minPadding: 0,
                maxPadding: 0,
                startOnTick: false,
                endOnTick: false
            }],
            tooltip: {
                shared: true
            },
            legend: {
                layout: 'vertical',
                floating: true,
                align: 'left',
                verticalAlign: 'top',
                x: 70,
                borderWidth: 0
            },
            series: s
        });

        return chart;
    }

};
 


bindsim.sim_nmr_1to1 = {
    // Internal constants
    i: {
        endpoint: "/nmr/1to1",
        // Response json object names
        json_mf_h: "mf_h",
        json_mf_hg: "mf_hg",
        json_dd: "dd",
        // Colors for plotting
        color_mf_h: "rgba(7, 52, 115, 0.4)",
        color_mf_hg: "rgba(18, 89, 187, 0.4)",
        color_dd: "rgba(255, 100, 3, 1)",
        // jQuery selector for 1to1 form
        form: "#params-nmr-1to1",
        $form: {}
    },

    init: function() {
        console.log("bindsim.sim_nmr_1to1.init called");

        // Initialise internal jQuery globals
        bindsim.sim_nmr_1to1.i.$form = $(bindsim.sim_nmr_1to1.i.form);

        // Show only selected simulator's form
        bindsim.i.$forms.hide();
        bindsim.sim_nmr_1to1.i.$form.show();

        // Init new plot, remove old (if it exists)
        if (typeof bindsim.chart != 'undefined') {
            bindsim.chart.destroy();
        }
        bindsim.chart = bindsim.sim_nmr_1to1.plot_setup();

        // Bind/rebind click on plot button to 1:1 plot function
        bindsim.i.$button.unbind("click");
        bindsim.i.$button.on("click", bindsim.sim_nmr_1to1.plot);

        // Populate plot
        bindsim.i.$button.click();
    },

    plot: function() {
        /**
         * Parses control form input, passes to backend and plots result. 
         * Called on plot button click.
         */

        // Top-level function, do everything!
        console.log("bindsim.sim_nmr_1to1.plot called");

        // Parse form into request json for bindsim api
        request = bindsim.params_to_json(bindsim.sim_nmr_1to1.i.$form, bindsim.parse_fail);
        console.log("Parsed request json:");
        console.log(request);

        // Call bindsim with parsed request
        var endpoint = bindsim.i.endpoint+bindsim.sim_nmr_1to1.i.endpoint;
        bindsim.bindsim_call(request, 
                             endpoint,
                             bindsim.sim_nmr_1to1.plot_update, 
                             bindsim.backend_fail);


    },

    plot_update: function(points, request) {
        /**
         * Redraws plot with new data
         * 
         * @param {Object} points - New x, y points to plot
         * @param {array} points.dd - Simulated points, format: [[x,y],..n]
         * @param {array} points.mf_h - As above
         * @param {array} points.mf_hg - As above
         */
        console.log("bindsim.sim_nmr_1to1.plot_update called");
        console.log("plot_update: Received returned points");
        console.log(points);
        
        // Set appropriate extremes
        shift_min = parseFloat($(bindsim.sim_nmr_1to1.i.form+"-dh").val());
        shift_max = parseFloat($(bindsim.sim_nmr_1to1.i.form+"-dhg").val());
        bindsim.chart.get(bindsim.i.axis_isotherm).setExtremes(shift_min, shift_max);

        // Set ticks (workaround for setExtremes bug)
        // TODO: this breaks everything, why?
/*
        bindsim.chart.get(bindsim.i.axis_isotherm).update({
            tickPositioner: function() {
                var positions = [];
                var tick = this.dataMin;
                var increment = (this.dataMax - this.dataMin)/6;

                for (tick; tick - increment <= this.dataMax; tick += increment) {
                    positions.push(tick);
                }
                return positions;
            }
        });
*/

        console.log("Isotherm y axis extremes (after set):");
        console.log(bindsim.chart.yAxis[0].getExtremes());

        // Plot new data
        bindsim.chart.get("series-molefrac-h")
            .setData(points[bindsim.sim_nmr_1to1.i.json_mf_h], false);
        bindsim.chart.get("series-molefrac-hg")
            .setData(points[bindsim.sim_nmr_1to1.i.json_mf_hg], false);
        bindsim.chart.get("series-isotherm")
            .setData(points[bindsim.sim_nmr_1to1.i.json_dd], false);
        bindsim.chart.redraw();
    },

    //
    // Highcharts
    // 
    plot_setup: function() {
        /**
         * Initialise a Highcharts chart set up for the 1:1 simulator
         *
         * @returns {Object} - Highcharts chart object
         */

        var series = [{
            id: "series-molefrac-h",
            name: "H molefraction",
            type: "area",
            yAxis: bindsim.i.axis_molefrac,
            color: bindsim.sim_nmr_1to1.i.color_mf_h
            }, {
            id: "series-molefrac-hg",
            name: "HG molefraction",
            type: "area",
            yAxis: bindsim.i.axis_molefrac,
            color: bindsim.sim_nmr_1to1.i.color_mf_hg
            }, {
            id: "series-isotherm",
            name: "Isotherm",
            type: "line",
            yAxis: bindsim.i.axis_isotherm,
            tooltip: {
                valueSuffix: " (ppm or Hz)"
            },
            lineWidth: 5,
            color: bindsim.sim_nmr_1to1.i.color_dd
        }];
        
        // Call bindsim's global plot_setup with series parameters
        return bindsim.plot_setup(series);
    }
};



bindsim.sim_nmr_1to2 = {
    // Internal constants
    i: {
        endpoint: "/nmr/1to2",
        // Response json object names
        json_mf_h: "mf_h",
        json_mf_hg: "mf_hg",
        json_mf_hg2: "mf_hg2",
        json_dd: "dd",
        // Colors for plotting
        color_mf_h: "rgba(7, 52, 115, 0.4)",
        color_mf_hg: "rgba(90, 138, 205, 0.4)",
        color_mf_hg2: "rgba(18, 89, 187, 0.4)",
        color_dd: "rgba(255, 100, 3, 1)",
        // jQuery selector for 1to1 form
        form: "#params-nmr-1to2",
        $form: {}
    },

    init: function() {
        console.log("bindsim.sim_nmr_1to2.init called");

        // Initialise internal jQuery globals
        bindsim.sim_nmr_1to2.i.$form = $(bindsim.sim_nmr_1to2.i.form);

        // Show only selected simulator's form
        bindsim.i.$forms.hide();
        bindsim.sim_nmr_1to2.i.$form.show();

        // Init new plot, remove old (if it exists)
        if (typeof bindsim.chart != 'undefined') {
            bindsim.chart.destroy();
        }
        bindsim.chart = bindsim.sim_nmr_1to2.plot_setup();

        // Bind/rebind click on plot button to 1:1 plot function
        bindsim.i.$button.unbind("click");
        bindsim.i.$button.on("click", bindsim.sim_nmr_1to2.plot);

        // Populate plot
        bindsim.i.$button.click();
    },

    plot: function() {
        /**
         * Parses control form input, passes to backend and plots result. 
         * Called on plot button click.
         */

        // Top-level function, do everything!
        console.log("bindsim.sim_nmr_1to2.plot called");

        // Parse form into request json for bindsim api
        request = bindsim.params_to_json(bindsim.sim_nmr_1to2.i.$form, bindsim.parse_fail);
        console.log("Parsed request json:");
        console.log(request);

        // Call bindsim with parsed request
        var endpoint = bindsim.i.endpoint+bindsim.sim_nmr_1to2.i.endpoint;
        bindsim.bindsim_call(request, 
                             endpoint,
                             bindsim.sim_nmr_1to2.plot_update, 
                             bindsim.backend_fail);


    },

    plot_update: function(points, request) {
        /**
         * Redraws plot with new data
         * 
         * @param {Object} points - New x, y points to plot
         * @param {array} points.dd - Simulated points, format: [[x,y],..n]
         * @param {array} points.mfh - As above
         * @param {array} points.mfhg - As above
         */
        console.log("bindsim.sim_nmr_1to2.plot_update called");
        console.log("plot_update: Received returned points");
        console.log(points);
        
        // Set appropriate extremes
        var form = bindsim.sim_nmr_1to2.i.form;
        shifts = [parseFloat($(form+"-dh").val()), parseFloat($(form+"-dhg").val()), parseFloat($(form+"-dhg2").val())];
        shift_min = Math.min.apply(Math, shifts);
        shift_max = Math.max.apply(Math, shifts);
        bindsim.chart.get(bindsim.i.axis_isotherm).setExtremes(shift_min, shift_max);

        // Plot new data
        bindsim.chart.get("series-molefrac-h")
            .setData(points[bindsim.sim_nmr_1to2.i.json_mf_h], false);
        bindsim.chart.get("series-molefrac-hg")
            .setData(points[bindsim.sim_nmr_1to2.i.json_mf_hg], false);
        bindsim.chart.get("series-molefrac-hg2")
            .setData(points[bindsim.sim_nmr_1to2.i.json_mf_hg2], false);
        bindsim.chart.get("series-isotherm")
            .setData(points[bindsim.sim_nmr_1to2.i.json_dd], false);
        bindsim.chart.redraw();
    },

    //
    // Highcharts
    // 
    plot_setup: function() {
        /**
         * Initialise a Highcharts chart set up for the 1:1 simulator
         *
         * @returns {Object} - Highcharts chart object
         */

        var series = [{
            id: "series-molefrac-h",
            name: "H molefraction",
            type: "area",
            yAxis: bindsim.i.axis_molefrac,
            color: bindsim.sim_nmr_1to2.i.color_mf_h
            }, {
            id: "series-molefrac-hg",
            name: "HG molefraction",
            type: "area",
            yAxis: bindsim.i.axis_molefrac,
            color: bindsim.sim_nmr_1to2.i.color_mf_hg
            }, {
            id: "series-molefrac-hg2",
            name: "HG2 molefraction",
            type: "area",
            yAxis: bindsim.i.axis_molefrac,
            color: bindsim.sim_nmr_1to2.i.color_mf_hg2
            }, {
            id: "series-isotherm",
            name: "Isotherm",
            type: "line",
            yAxis: bindsim.i.axis_isotherm,
            tooltip: {
                valueSuffix: " (ppm or Hz)"
            },
            lineWidth: 5,
            color: bindsim.sim_nmr_1to2.i.color_dd
        }];
        
        // Call bindsim's global plot_setup with series parameters
        return bindsim.plot_setup(series);
    }
};



$(document).ready(bindsim.on_ready);
