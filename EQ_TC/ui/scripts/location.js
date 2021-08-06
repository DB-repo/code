function showLocation(columns, data) {
//    $('.serp__web').append('<h2 style="display:inline-block;">Floor 1</h2>');
    $('.serp__web').empty();

    var imgTmpl = `<canvas id="c1" style="background: url('DBH-FloorMap/CS3-FLR01.jpg'); background-size: 100% 100%; width:600px;height:450px;margin-left: 40px; margin-bottom: 40px; " >`;
    $('.serp__web').append(imgTmpl);

    imgTmpl = `<h2 style="display:inline-block;"><canvas id="c2"  style=" background: url('DBH-FloorMap/CS3-FLR02.jpg'); background-size: 100% 100%; width:600px;height:450px;margin-left: 40px; margin-bottom: 40px; " >Floor 2</h2>`;
    $('.serp__web').append(imgTmpl);

    imgTmpl = `<h2 style="display:inline-block;"><canvas id="c3"  style="background: url('DBH-FloorMap/CS3-FLR03.jpg'); background-size: 100% 100%; width:600px;height:450px;margin-left: 40px; margin-bottom: 40px; " >Floor 3</h2>`;
    $('.serp__web').append(imgTmpl);

    imgTmpl = `<h2 style="display:inline-block;"><canvas id="c4"  style="background: url('DBH-FloorMap/CS3-FLR04.jpg'); background-size: 100% 100%; width:600px;height:450px;margin-left: 40px; margin-bottom: 40px; " >Floor 4</h2>`;
    $('.serp__web').append(imgTmpl);

    imgTmpl = `<h2 style="display:inline-block;"><canvas id="c5"  style="background: url('DBH-FloorMap/CS3-FLR05.jpg'); background-size: 100% 100%; width:600px;height:450px;margin-left: 40px; margin-bottom: 40px; " >Floor 5</h2>`;
    $('.serp__web').append(imgTmpl);

    imgTmpl = `<h2 style="display:inline-block;"><canvas id="c6" style="background: url('DBH-FloorMap/CS3-FLR06.jpg'); background-size: 100% 100%; width:600px;height:450px;margin-left: 40px; margin-bottom: 40px; " >Floor 6</h2>`;
    $('.serp__web').append(imgTmpl);

    var contexts = [];
    var canvases = [];
    for (var i=0; i<6; i++) {
        canvases.push(document.getElementById('c'+(i+1).toString()));
        contexts.push(canvases[i].getContext("2d"));
    }
    // Map sprite
    var mapSprite = new Image();
    mapSprite.src = "img/person2.png";

//    context.drawImage(mapSprite, 100, 100, 10, 10);

    console.log(resultData.length);
    for(var i=0; i < resultData.length; i++){
        var floor = resultData[i][0][0];
                contexts[floor-1].drawImage(mapSprite, resultData[i][0][1]+95, 140-resultData[i][0][2]-30 , 10, 10);

//        contexts[floor-1].drawImage(mapSprite, resultData[i][0][1]+95+Math.floor(Math.random() * 5), 140-resultData[i][0][2]-30+Math.floor(Math.random() * 5) , 5, 5);
    }


};