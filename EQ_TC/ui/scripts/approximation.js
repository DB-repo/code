var AP_EXECUTE = "/approx_execute";
var AP_RELOAD = "/approx_reload";
var AP_PLAN = "/approx_plan"
var AP_STOP = "/approx_stop"
var AP_EXPLAIN = "/approx_explain"


function executeApproximateQuery() {

    var query = document.getElementById("query-box").value;
    var delay = document.getElementById("delay").value;
    var epochs = document.getElementById("epochs").value;
    var group = document.getElementById("group").checked;

    if (group == false) {
        group = 0;
    } else {
        group = 1;
    }


    console.log(query);
    console.log(delay);
    console.log(epochs);
    console.log(group);

    var json = {
        'query': query,
        'delay': delay,
        'epochs': epochs,
        'group': group
    };
    console.log(json);
    console.log(JSON.stringify(json));

    var saveData = $.ajax({
        headers: {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        crossDomain: true,
        type: 'POST',
        url: ENRICH_DB + AP_EXECUTE,
        data: JSON.stringify(json),
        dataType: "json",
        success: function(resultData) {
            console.log(resultData);
            $('.serp__web').html('<h3><font color="green">Query submitted successfully. Click reload to get results</font></h3>')
        },
        error: function(response) {
            $('.serp__web').html('<h3><font color="red">Query unsuccessful. Check your query and/or stop previous query</font></h3>')
            console.log(response);
        }
    });
}



function stopApproximateQuery() {

    clearInterval(interval);
        var saveData = $.ajax({
			headers: {
				'Content-Type': 'application/json',
				'Access-Control-Allow-Origin': '*'
			},
			crossDomain: true,
			type: 'POST',
			url: ENRICH_DB + AP_STOP,
			success: function(resultData) {
				console.log(resultData);
			},
			error: function(response) {
				console.log(response);
				console.log(response.status);
			}
		});
}

function explainApproximateQuery() {

    var query = document.getElementById("query-box").value;
    console.log(query);

    var json = {
        'query': query
    };
    console.log(json);
    console.log(JSON.stringify(json));
    document.getElementById('qplan_li').className = 'serp__active';
    document.getElementById('results_li').className = '';
    document.getElementById('schema_li').className = '';
    document.getElementById('perf_li').className = '';

    var saveData = $.ajax({
        headers: {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        crossDomain: true,
        type: 'POST',
        url: ENRICH_DB + AP_EXPLAIN,
        data: JSON.stringify(json),
        dataType: "json",
        success: function(resultData) {
            console.log(resultData);
            $('.serp__web').empty();
            $('.serp__web').append('<textarea id="explainArea">'+resultData['explain']+'</textarea>');
            var editor = CodeMirror.fromTextArea(document.getElementById('explainArea'), {
                lineNumbers: false,
                mode: "text/x-sql"
            });

        },
        error: function(response) {
            $('.serp__web').html('<h3><font color="red">Query Error. Check your query </font></h3>')
            console.log(response);
        }
    });

}

function reloadApproximateQuery() {

    document.getElementById('perf_li').className = '';
        document.getElementById('schema_li').className = '';
        document.getElementById('qplan_li').className = '';
        document.getElementById('results_li').className = 'serp__active';

        var group = document.getElementById("group").checked;
        var dataType = document.getElementById("dataType").value;
        var approx = document.getElementById("approx").checked;

        if (group == false) {
            group = 0;
        } else {
            group = 1;
        }

        console.log(group);
        console.log(dataType);

        var json = {
            'group': group
        };
        console.log(json);
        console.log(JSON.stringify(json));

        var saveData = $.ajax({
			headers: {
				'Content-Type': 'application/json',
				'Access-Control-Allow-Origin': '*'
			},
			crossDomain: true,
			type: 'POST',
			url: ENRICH_DB + AP_RELOAD,
			data: JSON.stringify(json),
			dataType: "json",
			success: function(data) {

			    columns = data[0];
			    resultData = data[1];
			    $('.serp__web').empty();

				var noOfFiles = resultData.length;
                // console.log(resultData);
                var countNew=0; var countDel=0; var countOld=0;

                for(var i=0; i < noOfFiles; i++){
                    if (resultData[i][1] == 0) {
                        countOld += 1;
                    } else if (resultData[i][1] == 1) {
                        countNew += 1;
                    } else {
                        countDel += 1;
                    }
                }
                $('.serp__web').append('<h3><font color="green"><b>'+countNew+'</b></font> Results Added, <font color="red"><b>'+countDel+'</b></font> Results Deleted, <font color="blue"><b>'+countOld+'</b></font> No Change</h3><br/>');

                if(dataType == "standard") {

                    var result = drawTable(columns, resultData);

                } else if(dataType == "chart") {

                    var result = drawChart(columns, resultData);

                } else if(dataType == "location") {

                    var result = showLocation(columns, resultData);

                } else {

                    for(var i=0; i < noOfFiles; i++){
                        if (dataType == "images") {
                            if (resultData[i][1] == 0) {
                                imageIsLoaded('MultiPieDataset/' + resultData[i][0][0] + ".png", resultData[i][0], resultData[i][0][0]);
                            } else if (resultData[i][1] == 1) {
                                imageIsAdded('MultiPieDataset/' + resultData[i][0][0] + ".png", resultData[i][0], resultData[i][0][0]);
                            } else {
                                imageIsDeleted('MultiPieDataset/' + resultData[i][0][0] + ".png", resultData[i][0], resultData[i][0][0]);
                            }
                        } else if(dataType == "tweets") {
                            if (resultData[i][1] == 0) {
                                tweetIsLoaded(resultData[i][0][1], resultData[i][0][2], resultData[i][0][3], resultData[i][0], resultData[i][0][0]);
                            } else if (resultData[i][1] == 1) {
                                tweetIsAdded(resultData[i][0][1], resultData[i][0][2], resultData[i][0][3], resultData[i][0], resultData[i][0][0]);
                            } else {
                                tweetIsDeleted(resultData[i][0][1], resultData[i][0][2], resultData[i][0][3], resultData[i][0], resultData[i][0][0]);
                            }
                        }
                    }

                }

			},
			error: function(response) {
				console.log(response);
				console.log(response.status);
			}
		});

        var planData = $.ajax({
			headers: {
				'Content-Type': 'application/json',
				'Access-Control-Allow-Origin': '*'
			},
			crossDomain: true,
			type: 'POST',
			url: ENRICH_DB + AP_PLAN,
			data: JSON.stringify(json),
			dataType: "json",
			success: function(resultData) {

                console.log(resultData);
                epoch_txt = '<a href="javascript: document.body.scrollIntoView(false);"><h2> No. Splits ' +  Object.keys(resultData).length + "</h2></a><br/>\n";

                th_header = "<p><b>Current Epochs</b></p>\n";
                th_str = "";
                for(var i in resultData){
                    th_str += '<p style="color:blue;font-size:18px;">'+ i + "    -->    " + resultData[i] +"</p>\n";
                }
                th_str += "<br/>\n"

                str = epoch_txt + th_header + th_str;
                $('.serp__wiki').html(str);

			},
			error: function(response) {
				console.log(response);
				console.log(response.status);
			}
		});


}