function showSchema(data) {

    console.log(data);
	$('.serp__web').empty();
    $('.serp__web').append("<h2> Tables And Attributes</h2>");
    
    var header = `
        <div class="limiter">
		<div class="container-table100"   style="min-height: 5vh;">
			<div class="wrap-table100">
				<div  class="table100 ver1 m-b-110"   ;>
					<div class="table100-body js-pscroll">
						<table>
							<thead>
								<tr class="row100 head">
									<th class="cell100 column1">Table</th>
									<th class="cell100 column1">Attribute</th>
									<th class="cell100 column1">Type</th>
									<th class="cell100 column1"># Labels</th>
								</tr>
							</thead>
        `;

    var tail = `
                        </table>
					</div>
				</div>
			</div>
		</div>
	</div>`

    var body = "";
    for (var i=0; i < data['attributes'].length; i++ ){
        var row = "";
        for (var j=0; j < data['attributes'][i].length; j++){
            row += '<td class="cell100 column1">' + data['attributes'][i][j] + '</td>\n'
        }

        body += '<tr class="row100 body" bgcolor="#ccffff">' + row + '</tr>\n';

    }
    body = '<tbody>' + body + '</tbody>';

    var attrTable = header + body + tail;
    $('.serp__web').append(attrTable);

    $('.serp__web').append("<h2> Function Classes</h2>");
    header = `
        <div class="limiter">
		<div class="container-table100"   style="min-height: 5vh;">
			<div class="wrap-table100">
				<div  class="table100 ver1 m-b-110"   ;>
					<div class="table100-body js-pscroll">
						<table>
							<thead>
								<tr class="row100 head">
									<th class="cell100 column1">ID</th>
									<th class="cell100 column1">Name</th>
								</tr>
							</thead>
        `;

    body = "";
    for (var i=0; i < data['classes'].length; i++ ){
        var row = "";
        for (var j=0; j < data['classes'][i].length; j++){
            row += '<td class="cell100 column1">' + data['classes'][i][j] + '</td>\n'
        }

        body += '<tr class="row100 body" bgcolor="#ccffff">' + row + '</tr>\n';

    }
    body = '<tbody>' + body + '</tbody>';
    var classTable = header + body + tail;
    $('.serp__web').append(classTable);

    $('.serp__web').append("<h2> Function Table</h2>");
    header = `
        <div class="limiter">
		<div class="container-table100"   style="min-height: 5vh;">
			<div class="wrap-table100">
				<div  class="table100 ver1 m-b-110"   ;>
					<div class="table100-body js-pscroll">
						<table>
							<thead>
								<tr class="row100 head">
									<th class="cell100 column1">ID</th>
									<th class="cell100 column1">Function Class ID</th>
									<th class="cell100 column1">Table</th>
									<th class="cell100 column1">Attribute</th>
									<th class="cell100 column1">Index</th>
									<th class="cell100 column1">Cost</th>
									<th class="cell100 column1">Quality</th>
								</tr>
							</thead>
        `;

    body = "";
    for (var i=0; i < data['functions'].length; i++ ){
        var row = "";
        for (var j=0; j < data['functions'][i].length; j++){
            row += '<td class="cell100 column1">' + data['functions'][i][j] + '</td>\n'
        }

        body += '<tr class="row100 body" bgcolor="#ccffff">' + row + '</tr>\n';

    }
    body = '<tbody>' + body + '</tbody>';
    var funcTable = header + body + tail;
    $('.serp__web').append(funcTable);

    $('.serp__web').append("<h2> Decision Table</h2>");
    header = `
        <div class="limiter">
		<div class="container-table100"   style="min-height: 5vh;">
			<div class="wrap-table100">
				<div  class="table100 ver1 m-b-110"   ;>
					<div class="table100-body js-pscroll">
						<table>
							<thead>
								<tr class="row100 head">
									<th class="cell100 column1">Table</th>
									<th class="cell100 column1">Attribute</th>
									<th class="cell100 column1">State</th>
									<th class="cell100 column1">Uncertainty Ranges</th>
									<th class="cell100 column1">Next Function ID</th>
									<th class="cell100 column1">Delta Uncertainty</th>
								</tr>
							</thead>
        `;
    body = "";
    for (var i=0; i < data['decisions'].length; i++ ){
        var row = "";
        for (var j=0; j < data['decisions'][i].length; j++){
            row += '<td class="cell100 column1">' + data['decisions'][i][j] + '</td>\n'
        }

        body += '<tr class="row100 body" bgcolor="#ccffff">' + row + '</tr>\n';

    }
    body = '<tbody>' + body + '</tbody>';
    var decisionTable = header + body + tail;
    $('.serp__web').append(decisionTable);

    return attrTable;

}

function drawPerformanceTable(data) {

console.log(data);

    var header = `
        <div class="limiter">
		<div class="container-table100"   style="min-height: 5vh;">
			<div class="wrap-table100">
				<div  class="table100 ver1 m-b-110"   ;>
					<div class="table100-body js-pscroll">
						<table>
							<thead>
								<tr class="row100 head">
									<th class="cell100 column1">Epoch</th>
									<th class="cell100 column1">Setup</th>
									<th class="cell100 column1">Threshold Calc.</th>
									<th class="cell100 column1">Benefit Calc.</th>
									<th class="cell100 column1">Plan Gen.</th>
									<th class="cell100 column1">Enrichment</th>
									<th class="cell100 column1">Update State</th>
									<th class="cell100 column1">Delta Calc.</th>
									<th class="cell100 column1">Planned Enrichments</th>
									<th class="cell100 column1">Exec. Enrichments</th>
									<th class="cell100 column1">Success. Enrichments</th>
								</tr>
							</thead>
        `;

    var tail = `
                        </table>
					</div>
				</div>
			</div>
		</div>
	</div>`

    var body = "";
    for (var i=0; i < data.length; i++ ){
        var row = "";
        for (var j=0; j < data[i].length; j++){
            if (j !=0 && j < 8) {
                row += '<td class="cell100 column1">' + data[i][j].toFixed(3) + '</td>\n'
            } else {
                row += '<td class="cell100 column1">' + data[i][j] + '</td>\n'
            }
        }

        body += '<tr class="row100 body" bgcolor="#ccffff">' + row + '</tr>\n';

    }
    body = '<tbody>' + body + '</tbody>';

    return header + body + tail;

}
