document.getElementById('openPopupBtn').addEventListener('click', openPopup);
document.getElementById('closePopupBtn').addEventListener('click', closePopup);

function openPopup() {
    document.getElementById('popupContainer').style.display = 'block';
}

function closePopup() {
    document.getElementById('popupContainer').style.display = 'none';
}

// Initialize nodeData with a check to ensure model and model.nodes are defined



// loopy.model.nodes.forEach(function(node) {
//     var nodeData = `
//         <div>
//             <h3>Node: ${node.label}</h3>
//             <p>Hue: ${convertNumToColor(node.hue)}</p>
//             <p>Initial Amount: ${node.init * 6} (on a 1-6 scale)</p>
//             <p>Current Amount: ${node.value * 6} (on a 1-6 scale)</p>
//             <hr>
//         </div>
//     `;
//     content.innerHTML += nodeData;
// })



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
    console.log('a')
    if (selectedNodes.length == 0) {
        console.log('b')
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
        console.log('c')
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

    if (pageName === 'TimeSeries') {
        drawTimeSeriesChart();
    }
}

function updateTimeSeriesChart(tick, currentAmount, iter) {
    console.log(chart.data)
    chart.data.datasets[iter].data.push(currentAmount);
    chart.update();
}
