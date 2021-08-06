var PAUSE = '/pause'

$("#pause").click(function(){

        var token = document.getElementById("token").value;
        var json = {
            'token': token
        };

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
            url: ENRICH_DB + PAUSE,
			success: function(resultData) {
				$('.serp__web').html('<h3><font color="green">Query Paused successfully.</font></h3>')
			},
			error: function(response) {
				console.log(response);
				console.log(response.status);
			}
		});
});


$("#restart").click(function(){
           $("#execute-button").click();
});