const defaultColors = ['255,0,0', '0,255,0', '0,0,255', '255,0,255', '255,255, 0', '0,255,255'];

var userData = [];

const userDataSet = {
    data: userData,
    showLine: false,
    label: 'user data',
    backgroundColor: '#000',
    borderColor: '#000',
    pointRadius: 5
};

var datasets = [userDataSet];

const chartCanvas = document.getElementById('chart');
const chart = new Chart(
    chartCanvas,
    {
        type: 'line',
        data: {
            datasets: datasets
        },
        options: {
            events: ['click'],
            animation: false,
            responsive: false,
            scales: {
                x: {
                    type: 'linear',
                    min: -10,
                    max: 10,
                    ticks: {
                        stepSize: 1.0
                    }
                },
                y: {
                    type: 'linear',
                    min: -chartCanvas.clientHeight / chartCanvas.clientWidth * 10,
                    max: chartCanvas.clientHeight / chartCanvas.clientWidth * 10,
                    ticks: {
                        stepSize: 1.0
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'left'
                },
                tooltip: {
                    enabled: false
                }
            },
            onClick: (e) => {
                const canvasPosition = Chart.helpers.getRelativePosition(e, chart);
                const x = chart.scales.x.getValueForPixel(canvasPosition.x);
                const y = chart.scales.y.getValueForPixel(canvasPosition.y);
                userData.push({ x: x, y: y });
                fetchPrediciton(userData);
            }
        }
    }
);

function fetchPrediciton(content) {
    fetch('http://' + window.location.host + '/technical', {
        method: 'GET',
        body: JSON.stringify(content)
    })
        .then(response => response.json())
        .then(models => {
            chart.data.datasets.length = 1;

            let minError = Math.min.apply(Math, models.map(function (x) { return x.error; }));

            for (i = 0; i < models.length; i++) {
                let color = models[i].error == minError ? 'rgb(' + defaultColors[i] + ')' : 'rgba(' + defaultColors[i] + ', 0.3)';

                chart.data.datasets.push({
                    data: models[i].prediction,
                    showLine: true,
                    fill: false,
                    label: models[i].name + " | error: " + models[i].error.toFixed(4),
                    backgroundColor: color,
                    borderColor: color,
                    pointRadius: 0
                });
            }
            chart.update();
        })
        .catch(e => chart.update());
}

document.getElementById('btnReset').addEventListener('click', () => {
    userData.length = 0;
    chart.data.datasets.length = 1;
    chart.update();
});

document.getElementById('btnModel1').addEventListener('click', () => {
    userData.length = 0;
    chart.data.datasets.length = 1;

    // random 1st degree polynomial
    let a = Math.random() * 2 - 1;
    let b = Math.random() * 6 - 3;
    for (i = -30; i <= 30; i++) {
        let noise = Math.random();
        x = i / 3;
        userData.push({ x: x, y: a * x + b + noise });
    }

    fetchPrediciton(userData);
});

document.getElementById('btnModel2').addEventListener('click', () => {
    userData.length = 0;
    chart.data.datasets.length = 1;

    // random 2nd degree polynomial
    let a = Math.random() * 2 - 1;
    let b = Math.random() * 2 - 1;
    let c = Math.random() * 6 - 3;
    for (i = -30; i <= 30; i++) {
        let noise = Math.random();
        x = i / 3;
        userData.push({ x: x, y: a * x * x + b * x + c + noise });
    }

    fetchPrediciton(userData);
});

document.getElementById('btnModel3').addEventListener('click', () => {
    userData.length = 0;
    chart.data.datasets.length = 1;

    // random 2nd degree polynomial
    let a = Math.random() * 2 - 1;
    let b = Math.random() * 2 - 1;
    let c = Math.random() * 2 - 1;
    let d = Math.random() * 6 - 3;
    for (i = -30; i <= 30; i++) {
        let noise = Math.random();
        x = i / 3;
        userData.push({ x: x, y: a * x * x * x + b * x * x + c * x + d + noise });
    }

    fetchPrediciton(userData);
});

document.getElementById('btnModel4').addEventListener('click', () => {
    userData.length = 0;
    chart.data.datasets.length = 1;

    // random circle
    let cx = Math.random() * 6 - 3;
    let cy = Math.random() * 6 - 3;
    let r = Math.random() * 2 + 1;
    for (i = 0; i <= 60; i++) {
        let noise = Math.random() / 2;
        a = i * 6 * Math.PI / 180;
        x = cx + Math.cos(a) * (r + noise);
        y = cy + Math.sin(a) * (r + noise);
        userData.push({ x: x, y: y });
    }

    fetchPrediciton(userData);
});
