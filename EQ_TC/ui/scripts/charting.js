
var randomColorGenerator = function () {
    return '#' + (Math.random().toString(16) + '0000000').slice(2, 8);
};

var myChart;
var colors = {};
function drawChart(columns, resultData){

    var canvas = document.createElement("canvas");
    canvas.id = "myChart";
    canvas.style = "padding:20px 50px;  width:1200px; height:700px"
    $('.serp__web').append(canvas);
    var ctx = document.getElementById("myChart").getContext("2d");
    var labels = [];
    var data = [];
    var graphType ='line';

    if (resultData.length > 10){
        graphType ='line';
    }

    for (var i = 0; i < resultData.length; i++){
        var obj = resultData[i][0];
        console.log(obj);
        if (obj[0] == null) continue;

        labels.push(obj[0]);
        for (var j = 1; j < obj.length; j++){
            if (!(columns[j] in data)) {
                data[columns[j]] = [];
            }
            if (!(columns[j] in colors)){
                colors[columns[j]] = randomColor({luminosity: 'light'});
            }

            data[columns[j]].push(obj[j]);
        }
    }

    datasets = [];
    for (var key in data){
        datasets.push({
            label: key,
            backgroundColor: colors[key],
            borderWidth: 4,
//            strokeColor: "rgba(151,187,205,1)",
//            pointColor: "rgba(151,187,205,1)",
//            pointStrokeColor: "#fff",
//            pointHighlightFill: "#fff",
//            pointHighlightStroke: "rgba(151,187,205,1)",
            data: data[key]
        })
    }

    var graphData = {
        labels: labels,
        datasets: datasets
    };
    graphData = convertData(graphData);
    console.log(graphData);

//    if (myChart) myChart.destroy();
     myChart = new Chart(ctx, {

        type: graphType,
        data: graphData,
        options: {
            animation: {
                duration: 0
            },
            elements: {
                line: {
                    tension: 0
                }
            },
          responsive: false,
          maintainAspectRatio: true,
          scales: {
            yAxes: [{
                ticks: {
                    beginAtZero: true,
                    fontSize: 20
                },
                scaleLabel: {
                    display: true,
                    labelString: "Value",
                    fontSize: 22,
                    fontColor: '#b82e8a'

                }
            }],
            xAxes: [{
                ticks: {
                    beginAtZero: false,
                    fontSize: 20
                  
                },
                scaleLabel: {
                    display: true,
                    labelString: columns[0],
                    fontSize: 22,
                    fontColor: '#b82e8a'
                },
                type: 'linear'
            }]
          },
          legend: {
            display: true,
            labels: {
                fontColor: '#b82e8a',
                fontSize: 22
            }
          }
        }});


}

function convertData(data) {
  var newData = {};
  newData.datasets = data.datasets.map((dataset) => {
    return {
      order : 1,
      label: dataset.label,
      backgroundColor: dataset.backgroundColor,
      borderWidth: 4,
      data: dataset.data.map((d, i) => {
        return {
          x: data.labels[i],
          y: d
        }
      })
    };
  });
  return newData;
}

function drawAccuracy(accData){

    for (var accDataKey in accData){

        $('.serp__wiki').append(`<h4>${accDataKey}<h4>`);
        var canvas = document.createElement("canvas");
        canvas.id = "accuracyChart"+accDataKey;
        canvas.style = "width:320px; height:220px"


        $('.serp__wiki').append(canvas);
        var ctx = document.getElementById("accuracyChart"+accDataKey).getContext("2d");
        var labels = [];
        var data = [];
        var graphType ='line';
        var resultData = accData[accDataKey];
        var columns = ['Epoch', 'Precision', 'Recall', "F1-Measure"]

        for (var i = 0; i < resultData.length; i++){
            var obj = resultData[i];
            console.log(obj);
            if (obj[0] == null) continue;

            labels.push(obj[0]);
            for (var j = 1; j < obj.length; j++){
                if (!(columns[j] in data)) {
                    data[columns[j]] = [];
                }
                if (!(columns[j] in colors)){
                    colors[columns[j]] = randomColor({luminosity: 'light'});
                }

                data[columns[j]].push(obj[j]);
            }
        }
        console.log(data);

        datasets = [];
        for (var key in data){
            datasets.push({
                label: key,
                borderColor	: colors[key],
                borderWidth: 2,
    //            strokeColor: "rgba(151,187,205,1)",
    //            pointColor: "rgba(151,187,205,1)",
    //            pointStrokeColor: "#fff",
    //            pointHighlightFill: "#fff",
    //            pointHighlightStroke: "rgba(151,187,205,1)",
                data: data[key]
            })
        }

        console.log(datasets);
        var graphData = {
            labels: labels,
            datasets: datasets
        };

        console.log(graphData);

    //    if (myChart) myChart.destroy();
         myChart = new Chart(ctx, {

            type: graphType,
            data: graphData,
            options: {
                animation: {
                    duration: 0
                },
              responsive: false,
              maintainAspectRatio: true,
              scales: {
                yAxes: [{
                    ticks: {
                        beginAtZero: true,
                        fontSize: 8
                    },
                    scaleLabel: {
                        display: false
                    }
                }],
                xAxes: [{
                    ticks: {
                        beginAtZero: true,
                        fontSize: 8
                    },
                    scaleLabel: {
                        display: true,
                        labelString: "Epoch",
                        fontSize: 8,
                        fontColor: '#b82e8a'
                    }
                }]
              },
              legend: {
                display: true,
                labels: {
                    fontColor: '#b82e8a',
                    fontSize: 8
                }
              },
                elements: {
                    line: {
                            fill: false
                    }
                }
            }});
    }

}
