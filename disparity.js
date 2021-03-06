//slider from: https://www.w3schools.com/howto/howto_js_rangeslider.asp
var textsize = 200
var draw

// line intercept math by Paul Bourke http://paulbourke.net/geometry/pointlineplane/ ---> not used

//source: http://jsfiddle.net/justin_c_rounds/Gd2S2/light/
function checkLineIntersection(line1StartX, line1StartY, line1EndX, line1EndY, line2StartX, line2StartY, line2EndX, line2EndY) {
    // if the lines intersect, the result contains the x and y of the intersection (treating the lines as infinite) and booleans for whether line segment 1 or line segment 2 contain the point
    var denominator, a, b, numerator1, numerator2, result = {
        x: null,
        y: null,
        onLine1: false,
        onLine2: false
    };
    denominator = ((line2EndY - line2StartY) * (line1EndX - line1StartX)) - ((line2EndX - line2StartX) * (line1EndY - line1StartY));
    if (denominator == 0) {
        return result;
    }
    a = line1StartY - line2StartY;
    b = line1StartX - line2StartX;
    numerator1 = ((line2EndX - line2StartX) * a) - ((line2EndY - line2StartY) * b);
    numerator2 = ((line1EndX - line1StartX) * a) - ((line1EndY - line1StartY) * b);
    a = numerator1 / denominator;
    b = numerator2 / denominator;

    // if we cast these lines infinitely in both directions, they intersect here:
    result.x = line1StartX + (a * (line1EndX - line1StartX));
    result.y = line1StartY + (a * (line1EndY - line1StartY));
    /*
            // it is worth noting that this should be the same as:
            x = line2StartX + (b * (line2EndX - line2StartX));
            y = line2StartX + (b * (line2EndY - line2StartY));
            */
    // if line1 is a segment and line2 is infinite, they intersect if:
    if (a > 0 && a < 1) {
        result.onLine1 = true;
    }
    // if line2 is a segment and line1 is infinite, they intersect if:
    if (b > 0 && b < 1) {
        result.onLine2 = true;
    }
    // if line1 and line2 are segments, they intersect if both of the above are true
    return result;
};


class Camera {
    //https://www.markhansen.co.nz/javascript-optional-parameters/

    constructor(nested_svg, options) {
        this._camrotation = options.camrotation || 0;
        this._fov = options.fov || 50;
        this._pixels = options.pixels || 20
        this._svg = nested_svg
        this.createSvg()
    }

    get camrotation(){
        return this._camrotation
    }

    set camrotation(rot){
        this._camrotation = rot;
        this._svg.clear()
        this.createSvg()
    }

    get pixels() {
        return this._pixels;
    }

    set pixels(pixels) {
        this._pixels = pixels
        this._svg.clear()
        this.createSvg()
    }

    set fov(fov) {
        this._fov = fov
        this._svg.clear()
        //TODO: just change rotation angle.Faster?
        this.createSvg()
    }

    get fov() {
        return this._fov;
    }

    get nested_svg() {
        return this._svg
    }

    createSvg() {
        var cam1 = this._svg.circle(20).move(-10, -10).fill('#F00').id('cam')
        this.createNestedLines();
    }

    createNestedLines() {
        var strokelength = window.innerWidth
        var lines = this._svg.group()
        for (let angle = this.fov / -2; angle <= this.fov / 2; angle += this.fov / this._pixels) {
            lines.line(0, 0, strokelength, 0).addClass('pixelline').rotate(angle + this._camrotation, 0, 0)
        }
    }

    highlight(angle) {
        // helper variable to only calculate from 0-fov degrees
        var angle_positive = -this._camrotation + angle + this._fov / 2
        //remove existing highlights
        var lines_list = this._svg.find('.pixelline_highlight')
        lines_list.removeClass('pixelline_highlight')
        if (angle_positive > this._fov || angle_positive < 0)
            //outside field of view
            return;
        else {

            var index = Math.floor(angle_positive / this.fov * this.pixels)
            //console.log(index);
            //TODO: rewrite to find an id'd item
            var line1 = this.nested_svg.children()[1].children()[index]
            var line2 = this.nested_svg.children()[1].children()[index + 1]
            line1.addClass('pixelline_highlight')
            line2.addClass('pixelline_highlight')
            var l1angle = this._camrotation + (-this._fov/2) + (index/this.pixels)*(this.fov)
            var l2angle = this._camrotation + (-this._fov/2) + ((index+1)/this.pixels)*(this.fov)
            return ([l1angle,l2angle])

        }

    }

}

SVG.on(document, 'DOMContentLoaded', function () {
    // var draw = SVG().addTo('body')

    var slider = document.getElementById("myRange");
    var output = document.getElementById("demo");
    var pixelinput = document.getElementById("pixels")

    var FoVslider = document.getElementById("myFoV");
    var FoVoutput = document.getElementById("FoVdemo");

    var CamRotationslider = document.getElementById("cam_rotation");
    var CamRotationoutput = document.getElementById("cam_rotation_text");

    FoVoutput.innerHTML = FoVslider.value
    CamRotationoutput.innerHTML = CamRotationslider.value
    var x_offset = 30
    var no_of_pixels = 320


    draw = SVG().addTo('body').size(window.innerWidth, window.innerHeight).id('canvas');

    var cam1 = new Camera(draw.nested(), { fov: FoVslider.value, pixels: no_of_pixels , camrotation: Number(CamRotationslider.value) })
    var cam2 = new Camera(draw.nested(), { fov: FoVslider.value, pixels: no_of_pixels, camrotation:  -Number(CamRotationslider.value) })
    var intersections = draw.nested();

    cam1.nested_svg.move(x_offset, 100)
    cam2.nested_svg.move(x_offset, 100)


    var measure_line1 = draw.line(x_offset, 0, 100, 100).addClass('measureline').move(cam1.nested_svg.x(), cam1.nested_svg.y()).id('ml1')
    var measure_line2 = draw.line(x_offset, 0, 100, 100).addClass('measureline').move(cam2.nested_svg.x(), cam2.nested_svg.y()).id('ml2')
    updateScreen(SVG(draw), cam1, cam2);

    var canvas = document.getElementById("canvas")
    canvas.onmousemove = function (event) {
        var x = event.x;
        var y = event.y;
        measure_line1.plot(cam1.nested_svg.x(), cam1.nested_svg.y(), x, y)
        measure_line2.plot(cam2.nested_svg.x(), cam2.nested_svg.y(), x, y)
        angle_deg_cam1 = 90 - (Math.atan2(x - cam1.nested_svg.x(), y - cam1.nested_svg.y()) * 180 / Math.PI)
        angle_deg_cam2 = 90 - (Math.atan2(x - cam2.nested_svg.x(), y - cam2.nested_svg.y()) * 180 / Math.PI)
        angles1 = cam1.highlight(angle_deg_cam1)
        //console.log(coords1)
        angles2 = cam2.highlight(angle_deg_cam2)
        if(angles1 != undefined && angles2 != undefined){
            calculate_intersections(intersections,angles1,angles2,slider.value)
            intersections.move(x_offset,window.innerHeight/2)
        }
        else
            intersections.clear(); //remove dots

        //document.getElementById("angle").innerHTML = angle_deg_cam1.toFixed(2)

    }

    function calculate_intersections(nested_svg,angles1,angles2,distance)
    {
        var l = 200 //linelength
        var circlesize = 5;
        var polylist = []
        distance=Number(distance)
        nested_svg.clear();
        line1StartX = 0;
        line1StartY = -distance;
        line2StartX=0
        line2StartY=distance
        for(let a1_index = 0 ; a1_index < 2 ; a1_index++){
            for(let a2_index = 0 ; a2_index < 2 ; a2_index++){
            line1EndX = (l*Math.cos(angles1[a1_index]*Math.PI/180))
            line1EndY = (l*Math.sin(Math.PI*angles1[a1_index]/180))-distance

            line2EndX = (l*Math.cos(Math.PI*angles2[a2_index]/180))
            line2EndY = (l*Math.sin(Math.PI*angles2[a2_index]/180))+distance
            results = checkLineIntersection(line1StartX, line1StartY, line1EndX, line1EndY, line2StartX, line2StartY, line2EndX, line2EndY)
            if(a1_index != a2_index)
                nested_svg.circle(circlesize).move(results.x-circlesize/2,results.y-circlesize/2).fill('#550');
            polylist.push([results.x,results.y])
            }
        }
        //swap elements to get a good polygon
        nested_svg.polygon([polylist[0],polylist[2],polylist[3],polylist[1]]).fill("#3A3")
        //TODO: this needs to get set from another function. Maybe pass back to calling function?
        document.getElementById('depth_uncertainty').innerHTML = Math.sqrt((polylist[1][0]-polylist[2][0])**2+(polylist[1][1]-polylist[2][1])**2).toFixed(2)
        document.getElementById('depth').innerHTML = Math.sqrt((polylist[0][0])**2+(polylist[0][1]**2)).toFixed(2)
        //nested_svg.line([line1StartX,line1StartY,line1EndX,line1EndY]).stroke({color:"#FF0"})
        //nested_svg.line([line2StartX,line2StartY,line2EndX,line2EndY]).stroke({color:"#F0F"})
        //nested_svg.circle(circlesize).move(line1EndX,line1EndY).fill('#000')

    }

    slider.oninput = function () {

        updateScreen(SVG(draw), cam1, cam2);
    }


    pixelinput.onchange = function () {
        no_of_pixels = pixelinput.value

        cam1.pixels = no_of_pixels
        cam2.pixels = no_of_pixels
    }

    FoVslider.oninput = function () {
        cam1.fov = FoVslider.value
        cam2.fov = FoVslider.value
        updateScreen(SVG(draw), cam1, cam2);
        FoVoutput.innerHTML = FoVslider.value
    }

    CamRotationslider.oninput = function(){
        cam1.camrotation = Number(CamRotationslider.value)*-1
        cam2.camrotation = Number(CamRotationslider.value)
        CamRotationoutput.innerHTML = CamRotationslider.value
    }

})


function updateScreen(drawsvg, cam1, cam2) {

    var output = document.getElementById("demo");
    var slider = document.getElementById("myRange");
    var slidevalue = slider.value;
    //because value is used for up and down, displayed value should be twice as big
    output.innerHTML = slider.value * 2
    cam1.nested_svg.y((window.innerHeight / 2) - slidevalue * 1)
    cam2.nested_svg.y((window.innerHeight / 2) + slidevalue * 1)

    drawsvg.find('line#ml1').y(cam1.nested_svg.y())
    drawsvg.find('line#ml2').y(cam2.nested_svg.y())

    //document.getElementById('cam1').cx(slider.value)
    //draw.element('cam2').cy(slider.value)
    // screenupdater = setTimeout(UpdateScreen(),100)
}
