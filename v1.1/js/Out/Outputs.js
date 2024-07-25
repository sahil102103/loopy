var chart;

function convertNumToColor(color) {
    switch (color) {
        case 0:
            return "Red" 
        case 1:
            return "Orange" 
        case 2:
            return "Yellow" 
        case 3:
            return "Green" 
        case 4:
            return "Blue" 
        case 5:
            return "Purple" 
        default:
            return;
    }
}



function drawTimeSeriesChart() {
    let timeSeriesNodeData = []
    if (selectedNodes.length == 0) {
        console.log(selectedNodes)
        loopy.model.nodes.forEach(node => {
            timeSeriesNodeData.push({
            label: node.label,
            data: [],
            borderColor: `${convertNumToColor(node.hue)}`,
            borderWidth: 1,
            fill: false
            })
        })
    } else {
        selectedNodes.forEach(node => {
            console.log(node)
            timeSeriesNodeData.push({
            label: node.label,
            data: [],
            borderColor: `${convertNumToColor(node.hue)}`,
            borderWidth: 1,
            fill: false
            })
        })
    }

    const ctx = document.getElementById('timeSeriesChart').getContext('2d');
    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['0'],
            datasets: timeSeriesNodeData
        },
        options: {
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Time'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Current Node Value'
                    },
                    beginAtZero: true
                }
            }
        }
    });

    for (node in selectedNodes) {
        console.log(node)
    }

}

// Function to execute when the button is clicked
function handleClick() {
    chart.destroy()
    drawTimeSeriesChart();
}

// Get the button element and add an event listener
const button = document.getElementById('rebuildTimeSeriesChartButton');
button.addEventListener('click', handleClick);

function openPage(pageName) {
    let tabcontent = document.getElementsByClassName('tabcontent');
    for (let i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = 'none';
    }

    let tablinks = document.getElementsByClassName('tablink');
    for (let i = 0; i < tablinks.length; i++) {
        tablinks[i].style.backgroundColor = '';
    }

    document.getElementById(pageName).style.display = 'block';
    document.querySelector(`[onclick="openPage('${pageName}')"]`).style.backgroundColor = '#ccc';

    document.getElementById('destoryTimeSeriesChart')

    if (pageName === 'TimeSeries') {
        drawTimeSeriesChart();
    }
}

function updateTimeSeriesChart(currentAmount, iter) {
    console.log(chart.data)
    chart.data.datasets[iter].data.push(currentAmount);
    chart.update();
}