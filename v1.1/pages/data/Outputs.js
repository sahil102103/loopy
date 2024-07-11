// document.getElementById('openPopupBtn').addEventListener('click', function() {
//     document.getElementById('popupContainer').style.display = 'block';
// });

// document.getElementById('closePopupBtn').addEventListener('click', function() {
//     document.getElementById('popupContainer').style.display = 'none';
// });

// function openPage(pageName) {
//     var i, tabcontent, tablinks;

//     tabcontent = document.getElementsByClassName('tabcontent');
//     for (i = 0; i < tabcontent.length; i++) {
//         tabcontent[i].style.display = 'none';
//     }

//     tablinks = document.getElementsByClassName('tablink');
//     for (i = 0; i < tablinks.length; i++) {
//         tablinks[i].style.backgroundColor = '';
//     }

//     document.getElementById(pageName).style.display = 'block';
//     document.querySelector(`[onclick="openPage('${pageName}')"]`).style.backgroundColor = '#ccc';
// }

{/* <script src="https://cdn.jsdelivr.net/npm/chart.js"></script> */}

// // add CDN to head of DOM
// var chartScript = document.createElement('chart');  
// chartScript.setAttribute('src','https://ajax.googleapis.com/ajax/libs/jquery/3.2.1/jquery.min.js');
// document.head.appendChild(chartScript);

document.getElementById('openPopupBtn').addEventListener('click', function() {
    document.getElementById('popupContainer').style.display = 'block';
});

document.getElementById('closePopupBtn').addEventListener('click', function() {
    document.getElementById('popupContainer').style.display = 'none';
});

function openPage(pageName) {
    var i, tabcontent, tablinks;

    tabcontent = document.getElementsByClassName('tabcontent');
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = 'none';
    }

    tablinks = document.getElementsByClassName('tablink');
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].style.backgroundColor = '';
    }

    document.getElementById(pageName).style.display = 'block';
    document.querySelector(`[onclick="openPage('${pageName}')"]`).style.backgroundColor = '#ccc';
    
    if(pageName === 'TimeSeries') {
        drawTimeSeriesChart();
    }
}


// TODO: Make a function to iterate through all the nodes to create the time series
// TODO: I can store the time series information in a hashmap, with key as time elapsed and value as y as (currentVal/initalVal)
function drawTimeSeriesChart() {
    const ctx = document.getElementById('timeSeriesChart');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['January', 'February', 'March', 'April', 'May', 'June', 'July'], // Replace with your actual time data
            datasets: [{
                label: 'Initial Amount',
                data: [3, 2, 2, 5, 4, 6, 7], // Replace with your actual data
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1,
                fill: false
            },
            {
                label: 'Current Amount',
                data: [1, 3, 4, 2, 5, 3, 6], // Replace with your actual data
                borderColor: 'rgba(153, 102, 255, 1)',
                borderWidth: 1,
                fill: false
            }]
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
                        text: 'Current Node Value / Inital Node Value'
                    },
                    beginAtZero: true
                }
            }
        }
    });
}

