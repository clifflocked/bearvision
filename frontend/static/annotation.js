const canvas = document.getElementById("annotation");
const ctx = canvas.getContext("2d");

let img = new Image;
let framenum = 0;

img.onload = function() {
    ctx.drawImage(img, 0, 0);
}
img.src = '/frame?ts=' + new Date().getTime();

fetch('/framenum').then(resp => resp.json()).then(data => {
    framenum = data;
});

colors = ["red", "orange", "yellow", "green", "blue", "violet"];

function Dot(i) {
    this.x = i * 32 + 400;
    this.y = 20;
    this.r = 6;
    this.color = colors[i];
    this.dragging = false;
}

let dots = [];
for (var i = 0; i < 6; i++) {
    dots.push(new Dot(i));
    console.log(dots[i]);
}

let dragging = false;

function draw() {
    ctx.drawImage(img, 0, 0);

    for (var i = 0; i < dots.length; i++) {
        ctx.beginPath();
        ctx.arc(dots[i].x, dots[i].y, dots[i].r, 0, Math.PI * 2);
        ctx.fillStyle = dots[i].color;
        ctx.fill();
        ctx.closePath();
    }
}

function nextframe() {
    const goodframe = document.getElementById("goodframe").checked;
    const teams = document.getElementById("teams").value;

    let data = {
        goodframe: `${goodframe}`,
        teams: `${teams}`,
        dots: [],
        framenum: `${framenum}`
    };

    for (var i = 0; i < dots.length; i++) {
        data.dots.push({ x: dots[i].x, y: dots[i].y });
    }

    console.log(data);

    fetch("/data", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        console.log(result);
    })
    .catch(error => {
        console.log(error);
    })

    img.src = '/frame?ts=' + new Date().getTime();
    fetch('/framenum').then(resp => resp.json()).then(data => {
        framenum = data;
    });

    console.log(framenum);

    draw();
}

function mouseOnDot(dotx, doty, dotr, x, y) {
    let dx = x - dotx;
    let dy = y - doty;
    let dist2 = dx * dx + dy * dy;
    return dist2 <= dotr * dotr;
}

canvas.addEventListener("mousedown", e => {
    dragging = true;
});

canvas.addEventListener("mousemove", e => {
    if (dragging) {
        const rect = canvas.getBoundingClientRect();
        let x = e.clientX - rect.left;
        let y = e.clientY - rect.top;

        for (var i = 0; i < dots.length; i++) {
            if (mouseOnDot(dots[i].x, dots[i].y, dots[i].r, x, y) || dots[i].dragging) {
                dots[i].x = Math.max(dots[i].r, Math.min(canvas.width - dots[i].r, x));
                dots[i].y = Math.max(dots[i].r, Math.min(canvas.height - dots[i].r, y));
                dots[i].dragging = true;
            }
        }

        draw();
    } else {
        for (var i = 0; i < dots.length; i++) {
            dots[i].dragging = false;
        }
    }
});

canvas.addEventListener("mouseup", () => {
    dragging = false;
});