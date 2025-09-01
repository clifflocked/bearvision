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

let dot = { x: 200, y: 150, r: 4 };
let dragging = false;

function draw() {
    ctx.drawImage(img, 0, 0);
    ctx.beginPath();
    ctx.arc(dot.x, dot.y, dot.r, 0, Math.PI * 2);
    ctx.fillStyle = "red";
    ctx.fill();
    ctx.closePath();
}

function nextframe() {
    const goodframe = document.getElementById("goodframe").checked;
    const teams = document.getElementById("teams").value;

    const data = {
        goodframe: `${goodframe}`,
        teams: `${teams}`,
        dot: {
            x: `${dot.x}`,
            y: `${dot.y}`
        },
        framenum: `${framenum}`
    };

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

canvas.addEventListener("mousedown", e => {
    dragging = true;
});

canvas.addEventListener("mousemove", e => {
    if (dragging) {
        const rect = canvas.getBoundingClientRect();
        dot.x = e.clientX - rect.left;
        dot.y = e.clientY - rect.top;

        dot.x = Math.max(dot.r, Math.min(canvas.width - dot.r, dot.x));
        dot.y = Math.max(dot.r, Math.min(canvas.height - dot.r, dot.y));

        draw();
    }
});

canvas.addEventListener("mouseup", () => {
    dragging = false;
});


