Highcharts.theme = {
    colors: ["#79BCB8", "#0B4F6C", "#197BBD", "#033860", "#47A8BD", "#1E3888", "#EE6C4D", "#FA8334"],
    chart: {
        marginTop: 50,
        marginLeft: 100, // For chart stacking consistency w/
                         // differing axis label lengths
        backgroundColor: null,
        style: {'font-family': 'Lato, Helvetica, Arial, Verdana', 'text-transform': 'none'}
    },
    title: {
        text: "",
        style: {
            fontSize: '16px',
            fontWeight: 'bold',
        }
    },
    subtitle: {
        text: ""
    },
    tooltip: {
        shared: true,
        crosshairs: [true, false],
        borderWidth: 0,
        backgroundColor: 'rgba(219,219,216,0.8)',
        shadow: false,

        useHTML: true,
        headerFormat: '<span style="font-size: 10px">x: {point.key:.4f}</span><br/><table>',
        pointFormat: '<tr>'+
            '<td style="color: {point.color}">\u25CF {series.name}</td>'+
            '<td style="text-align: right"><b>{point.y} {point.yUnits}</b></td>'+
            '</tr>',
        footerFormat: '</table>',
        valueDecimals: 4
    },
    legend: {
        layout: 'horizontal',
        floating: true,
        align: 'center',
        verticalAlign: 'top',
        borderWidth: 0,
        itemStyle: {
            fontWeight: 'bold',
            fontSize: '14px'
        }
    },
    xAxis: {
        gridLineWidth: 1,
        minorTickInterval: null,
        labels: {
            style: {
                fontSize: '12px'
            }
        }
    },
    yAxis: {
        gridLineWidth: 1,
        minorTickInterval: null,
        title: {
            style: {
            }
        },
        labels: {
            style: {
                fontSize: '12px'
            }
        }
        // opposite: true,
    },
    plotOptions: {
        candlestick: {
            lineColor: '#404048'
        }
    }
};

// Apply the theme
var highchartsOptions = Highcharts.setOptions(Highcharts.theme);
