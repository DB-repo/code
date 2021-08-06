/*
Script containing functions to fetch data from the backend
*/

var ENRICH_DB = "http://localhost:5000"
var EXECUTE = "/execute"
var FETCH = "/reload"
var STOP = "/stop"
var PLAN = "/plan"
var EXPLAIN = "/explain"
var STATE = "/state"
var SCHEMA = "/schema"
var PERF = "/perf"

var interval = null;
$("#execute-button").click(function(){
        var query = document.getElementById("query-box").value;
        var delay = document.getElementById("delay").value;
        var epochs = document.getElementById("epochs").value;
        var group = document.getElementById("group").checked;
        var algo = document.getElementById("algo").value;
        var token = document.getElementById("token").value;
        var approx = document.getElementById("approx").checked;

        if (algo != "pro1") {
            processBaselines(query, algo);
            return ;
        }

        if (group == false) {
            group = 0;
        } else {
            group = 1;
        }

        if (approx == true) {
            return executeApproximateQuery();
        }

        var json = {
            'query': query,
            'delay': delay,
            'epochs': epochs,
            'group': group,
            'token': token
        };

        var saveData = $.ajax({
			headers: {
				'Content-Type': 'application/json',
				'Access-Control-Allow-Origin': '*'
			},
			crossDomain: true,
			type: 'POST',
			url: ENRICH_DB + EXECUTE,
			data: JSON.stringify(json),
			dataType: "json",
			success: function(resultData) {
				$('.serp__web').html('<h3><font color="green">Query submitted successfully. Click reload to get results</font></h3>')
			},
			error: function(response) {
                $('.serp__web').html('<h3><font color="red">Query unsuccessful. Check your query and/or stop previous query</font></h3>')
				console.log(response);
			}
		});
});

$("#autoload").change(function(){
        var autoload = document.getElementById("autoload").checked;
        if (autoload == false) {
            if (interval != null) {
                clearInterval(interval);
            }
            interval = null;
        } else {
            interval = window.setInterval(function(){
                    $("#reload").click();
                }, 3500);
        };
});



$("#stop").click(function(){

        var token = document.getElementById("token").value;
        var json = {
            'token': token
        };
        var approx = document.getElementById("approx").checked;
        if (approx == true) {
            return stopApproximateQuery();
        }

        clearInterval(interval);
        var saveData = $.ajax({
			headers: {
				'Content-Type': 'application/json',
				'Access-Control-Allow-Origin': '*'
			},
			crossDomain: true,
			type: 'POST',
			data: JSON.stringify(json),
			dataType: "json",
            url: ENRICH_DB + STOP,
			success: function(resultData) {
				$('.serp__web').html('<h3><font color="green">Query Stopped Successfully</font></h3>')
			},
			error: function(response) {
			    $('.serp__web').html('<h3><font color="blue">Unable to stop query OR query is already stopped</font></h3>')
				console.log(response);
				console.log(response.status);
			}
		});
});


$("#explain").click(function(){

        var query = document.getElementById("query-box").value;
        var algo = document.getElementById("algo").value;
        var approx = document.getElementById("approx").checked;

        if (approx == true) {
            return explainApproximateQuery();
        }

        var json = {
            'query': query,
            'algo': algo

        };

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
			url: ENRICH_DB + EXPLAIN,
			data: JSON.stringify(json),
			dataType: "json",
			success: function(resultData) {
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
});

$("#reload").click(function(){

        document.getElementById('perf_li').className = '';
        document.getElementById('schema_li').className = '';
        document.getElementById('qplan_li').className = '';
        document.getElementById('results_li').className = 'serp__active';

        var group = document.getElementById("group").checked;
        var dataType = document.getElementById("dataType").value;
        var token = document.getElementById("token").value;
        var approx = document.getElementById("approx").checked;

        if (group == false) {
            group = 0;
        } else {
            group = 1;
        }

        if (approx == true) {
            return reloadApproximateQuery();
        }

        var json = {
            'group': group,
            'token': token
        };

        var saveData = $.ajax({
			headers: {
				'Content-Type': 'application/json',
				'Access-Control-Allow-Origin': '*'
			},
			crossDomain: true,
			type: 'POST',
			url: ENRICH_DB + FETCH,
			data: JSON.stringify(json),
			dataType: "json",
			success: function(data) {

			    columns = data[0];
			    resultData = data[1];
			    $('.serp__web').empty();

				var noOfFiles = resultData.length;
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
			url: ENRICH_DB + PLAN,
			data: JSON.stringify(json),
			dataType: "json",
			success: function(resultData) {

                epoch_txt = '<a href="javascript: document.body.scrollIntoView(false);"><h2> Epoch ' +  resultData[0] + "</h2></a><br/>\n";

                th_header = "<p><b>Current Thresholds</b></p>\n";
                th_str = "";
                for(var i=0; i < resultData[1].length; i++){
                    th_str += "<p>["+ resultData[1][i][0] + ", " + resultData[1][i][1] + "] --> " + resultData[1][i][2].toFixed(2) + "</p>\n";
                }
                th_str += "<br/>\n"

                enrich_header = "<p><b>Planned Enrichments</b></p><br/>\n";
                enrich_str = "";
                for(var i=0; i < resultData[2].length; i++){
                    enrich_str += "<p>["+ resultData[2][i][0] + ", " + resultData[2][i][1] + ", function-" + resultData[2][i][2] + "] --> " + resultData[2][i][3] + "</p>\n";
                }
                str = epoch_txt + th_header + th_str + enrich_header + enrich_str;

                $('.serp__wiki').html(str);
                drawAccuracy(resultData[3])

			},
			error: function(response) {
				console.log(response);
				console.log(response.status);
			}
		});
});

$('#results_pane').click(function(){

    document.getElementById('perf_li').className = '';
    document.getElementById('schema_li').className = '';
    document.getElementById('qplan_li').className = '';
    document.getElementById('results_li').className = 'serp__active';

    $("#reload").click();
})


$(function () {
    $(":file").change(function () {
        /*if (this.files && this.files[0]) {
            var reader = new FileReader();
            reader.onload = imageIsLoaded;
            reader.readAsDataURL(this.files[0]);
        }*/
            var noOfFiles = this.files.length;
            for(var i=0; i < noOfFiles; i++){
        	  var reader = new FileReader();
            reader.onload = imageIsLoaded;
            reader.readAsDataURL(this.files[i]);
        }
    });
});


function drawTable(columns, data) {

    var header = `
        <div class="limiter">
            <div class="container-table100">
                <div class="wrap-table100">
                    <div class="table100 ver1 m-b-110">
                    	<div class="table100-body js-pscroll">
                            <table>
        `;

    var tail = `
                        </table>
					</div>
				</div>
			</div>
		</div>
	</div>`

    var col_header = `
			<thead>
				<tr class="row100 head">
    `
    var col_tail = `
        </tr>
          </thead>
		<tbody>
    `

    var col_row = "";

    for (var j=0; j < columns.length; j++){
        col_row += '<th class="cell100 column1">' + columns[j] + '</th>\n'
    }

    var body = "";
    for (var i=0; i < data.length; i++ ){
        var row = "";
        for (var j=0; j < data[i][0].length; j++){
            row += '<td class="cell100 column1" >' + data[i][0][j] + '</td>\n'
        }
        if (data[i][1] == 0) {
            body += '<tr class="row100 body" bgcolor="#f8f6ff">' + row + '</tr>\n';
        } else if (data[i][1] == 1) {
            body += '<tr bgcolor="#C2FFC4" class="row100 body" >' + row + '</tr>\n';
        } else {
            body += '<tr class="row100 body" bgcolor="#FFC0C0">' + row + '</tr>\n';
        }
    }

    body = '<tbody>' + body + '</tbody>';

    var table = header + col_header + col_row +col_tail+ body + tail;
    $('.serp__web').append(table);

    return table;

}


// Function to add images
function imageIsAdded(e, hover, id) {
    var imgTmpl = `<img onclick="dialogOpen('images', ${id})" title=${hover} src=${e} style="width:250px;height:200px;margin-left: 5px; margin-bottom: 5px; border: 10px solid rgb(0, 201, 0);" >`;
    $('.serp__web').append(imgTmpl);
};

function imageIsDeleted(e, hover, id) {
    var imgTmpl = `<img onclick="dialogOpen('images', ${id})" title=${hover} src=${e} style="width:250px;height:200px;margin-left: 5px; margin-bottom: 5px; border: 10px solid rgb(201, 0, 0);" >`;
    $('.serp__web').append(imgTmpl);
};

function imageIsLoaded(e, hover, id) {
    var imgTmpl = `<img onclick="dialogOpen('images', ${id})" title=${hover} src=${e} style="width:250px;height:200px;padding-left: 20px; padding-bottom: 20px;">`;
    $('.serp__web').append(imgTmpl);
};



// Functions to add tweets
function tweetIsAdded(e, titles, url, hover, id) {
    var tweetTmpl = `<div onclick="dialogOpen('tweets', ${id})" title="${hover}" class="serp__result serp__tweet__added" style="padding-left: 15px; margin-bottom: 15px;">
                        <a href="##" target="_blank">
                        <span class="serp__url">${url}</span></a><br>
                        <span class="serp__description">${e}</span>
                    </div>`;
    $('.serp__web').append(tweetTmpl);
};

function tweetIsDeleted(e, titles, url, hover, id) {
    var tweetTmpl = `<div onclick="dialogOpen('tweets', ${id})" title="${hover}" class="serp__result serp__tweet__deleted" style="padding-left: 15px; margin-bottom: 15px;">
                        <a href="##" target="_blank">
                        <span class="serp__url">${url}</span></a><br>
                        <span class="serp__description">${e}</span>
                    </div>`;
    $('.serp__web').append(tweetTmpl);
};

function tweetIsLoaded(e, titles, url, hover, id) {
    var tweetTmpl = `<div onclick="dialogOpen('tweets', ${id})" title="${hover}" class="serp__result serp__tweet__loaded" style="padding-left: 15px; margin-bottom: 15px;">
                        <a href="#" target="_blank">
                        <span class="serp__url">${url}</span></a><br>
                        <span class="serp__description">${e}</span>
                    </div>`;
    $('.serp__web').append(tweetTmpl);
};


// Functions to plot Groups
function plotSingle(e, hover) {
    var imgTmpl = '<img title="'+hover+ '" src='+e+' style="width:250px;height:200px;margin-left: 5px; margin-bottom: 5px; border: 10px solid rgb(0, 201, 0);" >';
    $('.serp__web').append(imgTmpl);
};

function tweetMultiple(e, hover) {
    var imgTmpl = '<img title="'+hover+ '" src='+e+' style="width:250px;height:200px;margin-left: 5px; margin-bottom: 5px; border: 10px solid rgb(201, 0, 0);" >';
    $('.serp__web').append(imgTmpl);
};


$(document).ready(function() {
        var dialog = $("#dialog");

        dialog.dialog({
            title: "Object State",
            modal: true,
            draggable: true,
            resizable: true,
            autoOpen: false,
            width: 700,
            height: 300,
            dialogClass: "search__dialog"
        });

//        dialog.hide();
    });

function dialogOpen(type_, id) {


        var json = {
            'type': type_,
            'id': id
        };

        var saveData = $.ajax({
			headers: {
				'Content-Type': 'application/json',
				'Access-Control-Allow-Origin': '*'
			},
			crossDomain: true,
			type: 'POST',
			url: ENRICH_DB + STATE,
			data: JSON.stringify(json),
			dataType: "json",
			success: function(resultData) {

                data = ""
				for (var key of Object.keys(resultData)) {
				    data += '<h4><font color="red">'+key+":</font></h4>" + "<h5>"+resultData[key] +"</h5>"
				}
                $('#dialog').html(data);
                $('#dialog').dialog('open')
			},
			error: function(response) {
				$('#dialog').html('<h3><font color="red">Error Fetching State </font></h3>');
                $('#dialog').dialog('open')
			}
		});

};


$("#schema").click(function(){


        document.getElementById('schema_li').className = 'serp__active';
		document.getElementById('qplan_li').className = '';
        document.getElementById('results_li').className = '';
        document.getElementById('perf_li').className = '';


        var saveData = $.ajax({
			headers: {
				'Content-Type': 'application/json',
				'Access-Control-Allow-Origin': '*'
			},
			crossDomain: true,
			type: 'POST',
			url: ENRICH_DB + SCHEMA,
			dataType: "json",
			success: function(resultData) {
                showSchema(resultData);
			},
			error: function(response) {
                $('.serp__web').html('<h3><font color="red">Error Occurred While Fetching Schema</font></h3>')
				console.log(response);
			}
		});
});


$("#performance").click(function(){

        document.getElementById('perf_li').className = 'serp__active';
        document.getElementById('schema_li').className = '';
		document.getElementById('qplan_li').className = '';
        document.getElementById('results_li').className = '';
        var token = document.getElementById("token").value;
        var json = {
            'token': token
        };

        var saveDataInner = $.ajax({
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            crossDomain: true,
            type: 'POST',
            data: JSON.stringify(json),
			dataType: "json",
            url: ENRICH_DB + PERF,
            success: function(resultData) {
                $('.serp__web').html(drawPerformanceTable(resultData));

            },
            error: function(response) {
                $('.serp__web').html('<h3><font color="red">Query Error. Check your query </font></h3>')
                console.log(response);
            }
        });
});
