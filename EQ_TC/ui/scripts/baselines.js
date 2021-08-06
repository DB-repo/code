var NP_EXECUTE = "/npexecute";
var PRO_BL_EXECUTE = "/baseline_execute";

function processBaselines(query, algo) {

    if (algo == "ba1") {
        processBaseline1(query);
    } else if(algo == "ba2") {
        processProgressiveBaselines(2);
    } else if(algo == "ba3") {
        processProgressiveBaselines(3)
    }

}

function processProgressiveBaselines(baseline){
        var query = document.getElementById("query-box").value;
        var delay = document.getElementById("delay").value;
        var epochs = document.getElementById("epochs").value;
        var group = document.getElementById("group").checked;
        var token = document.getElementById("token").value;

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
            'group': group,
            'baseline': baseline,
            'token': token
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
			url: ENRICH_DB + PRO_BL_EXECUTE,
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


function processBaseline1(query) {

    $('.serp__web').empty();
    $('.serp__web').append('<h3><font color="blue">Query submitted/waiting</font></h3>')

    var json = {
         'query': query
    };
    console.log(json);
    console.log(JSON.stringify(json));

    var dataType = document.getElementById("dataType").value;

    var saveData = $.ajax({
        headers: {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        crossDomain: true,
        type: 'POST',
        url: ENRICH_DB + NP_EXECUTE,
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
                    resultData[i] = [resultData[i], 1];
            }

            console.log(resultData);
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
            $('.serp__web').html('<h3><font color="red">Query unsuccessful. Check your query and/or stop previous query</font></h3>')
            console.log(response);
        }
    });

}
