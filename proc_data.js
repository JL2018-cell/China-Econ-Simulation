
/* Utility functions */
function getRandomInt(max) {
  return Math.floor(Math.random() * max);
}
function sumOfArray(arr) {
    let sum = 0;
    for (let i = 0; i < arr.length; i++) {
        sum += arr[i];
    }
    return sum;
}


/* Draw bar chart, showing resource allocated to industries */
function drawIndustryDevpl() {

    function data_average(agent_num) {
        trials_num = Object.keys(mydata).length;
        data_array = [];
        for (let i=0; i<trials_num; i++) {
            data_array.push(Object.keys(mydata[i]['states']).map(function (ep) {return sumOfArray(Object.values(mydata[i]['states'][ep][agent_num.toString()]['inventory']))}));
        }
        data_avg = [];
        data_len = data_array[0].length;
        for (let j=1; j<data_len; j++) {
            let sum = 0;
            for (let i=0; i<trials_num; i++) {
                sum = sum + data_array[i][j];
            }
            data_avg.push(sum/trials_num);
            sum = 0;
        }
        return data_avg;
    }

    var myTableDiv = document.getElementById("bigCanvas");
    var canvas = document.createElement("canvas");
    var canvas_id = "drawIndustryDevplCanvas";
    canvas.id = canvas_id;
    canvas.width = 1000;
    canvas.height = 500;

    labels = [];
    for (let i=1; i<ep_num; i++) {
        labels.push("episode " + i.toString());
    }
    datasets = [];
    for (let i=0; i<agent_num; i++) {
        color_val1 = getRandomInt(255).toString();
        color_val2 = getRandomInt(255).toString();
        color_val3 = getRandomInt(255).toString();
        datasets.push({label: agent_names[i],
                       data: data_average(i),
                       backgroundColor: agent_clrs[i],
                       borderWidth: 1
                      });
    }
    //Object.keys(mydata[0]['states']).map(function (ep) {return "rgba(" + color_val1 + ", " + color_val2 + ", " + color_val3 + ", 1)"}),
    const data = {
      labels: labels,
      datasets: datasets
    };
    const config = {
      type: 'bar',
      data: data,
      options: {
        scales: {
          y: {
            beginAtZero: true,
            display: true,
            title: {
              display: true,
              text: '10 thousand people',
              font: {
                size: 20
              }
            }
          }
        }
      },
    };

    var mydiv = document.createElement("div");
    var mytitle = document.createElement("h3");
    mytitle.setAttribute("style", "text-align: center;");
    var title = document.createTextNode("Industry Development"); 
    mytitle.appendChild(title);
    mydiv.appendChild(mytitle);
    mydiv.appendChild(canvas);
    myTableDiv.appendChild(mydiv);
    var graphArea = document.getElementById(canvas_id).getContext("2d");
    new Chart(graphArea, config);
}

function drawIndustryComposition() {

    function data_average(episode_num, agent_num) {
        trials_num = Object.keys(mydata).length;
        data_array = [];
        for (let i=0; i<trials_num; i++) {
            data_array.push(Object.values(mydata[i]['states'][episode_num][(agent_num).toString()]['inventory']));
        }
        data_avg = [];
        data_len = data_array[0].length;
        for (let j=0; j<data_len; j++) {
            let sum = 0;
            for (let i=0; i<trials_num; i++) {
                sum = sum + data_array[i][j];
            }
            data_avg.push(sum/trials_num);
            sum = 0;
        }
        return data_avg;
    }

    var table = document.createElement('TABLE');
    table.border='1';
    
    var tableBody = document.createElement('TBODY');
    table.appendChild(tableBody);
      
    for (let i=1; i<ep_num+1; i++){
       var tr = document.createElement('TR');
       tableBody.appendChild(tr);
       
       for (let j=0; j<agent_num+1+1; j++){
           if (i==1 && j==0) {
               var td = document.createElement('TD');
               td.width='75';
               td.class='background-image: linear-gradient(to bottom right,  transparent calc(50% - 1px), red, transparent calc(50% + 1px));';
               tr.appendChild(td);
           }
           else if (i==1 && j==1) {
               var td = document.createElement('TD');
               td.appendChild(document.createTextNode("Total"));
               td.width='75';
               tr.appendChild(td);
           }
           else if (i==1 && j>1) {
               var td = document.createElement('TD');
               td.appendChild(document.createTextNode(mydata[0]['states'][0][(j-1-1).toString()]['name']));
               td.width='75';
               tr.appendChild(td);
           }
           else if (j==0) {
               var td = document.createElement('TD');
               td.appendChild(document.createTextNode("Episode " + (i - 1)));
               td.width='75';
               tr.appendChild(td);
           }
           else {
               var canvas = document.createElement('canvas');
               var canvas_id = "cursorLayer" + i.toString() + ", " + j.toString();
               canvas.id = canvas_id;
               canvas.width = 300;
               canvas.height = 300;
               var td = document.createElement('TD');
               td.width='75';
               td.appendChild(canvas);
               tr.appendChild(td);
           }
       }
    }
    var mydiv = document.createElement("div");
    var mytitle = document.createElement("h3");
    mytitle.setAttribute("style", "text-align: center;");
    var title = document.createTextNode("Industry Composition"); 
    mytitle.appendChild(title);
    mydiv.appendChild(mytitle);
    mydiv.appendChild(table);
    var myTableDiv = document.getElementById("bigCanvas");
    myTableDiv.appendChild(mydiv);
    
    var sum = (r, a) => r.map((b, i) => a[i] + b);

    myCharts = [];
    for (let j=0; j<agent_num+1+1; j++){
       clrs = Object.values(mydata[0]['states'][0]['0']['inventory']).map(function(i) {return "rgba(" + getRandomInt(255).toString() + ", " + getRandomInt(255).toString() + ", " + getRandomInt(255).toString() + ", " + getRandomInt(255).toString() + ")"});
       clrs.push("rgba(" + getRandomInt(255).toString() + ", " + getRandomInt(255).toString() + ", " + getRandomInt(255).toString() + ", " + getRandomInt(255).toString() + ")");
       for (let i=1; i<ep_num+1; i++){
           if (i>1 && j==1) {
               everyAgents = Object.keys(mydata[0]['states'][0]).slice(0, -1).map(function(agent) {return data_average(i - 1, agent)})
               var graphArea = document.getElementById("cursorLayer" + i.toString() + ", " + j.toString()).getContext("2d");
               const data2 = {
                 labels: Object.keys(mydata[0]['states'][i - 1][(j - 1).toString()]['inventory']),
                 datasets: [
                   {
                     label: 'Dataset 1',
                     data: everyAgents.reduce(sum),
                     backgroundColor: industries_clrs
                   }
                 ]
               };
               const config2 = {
                 type: 'pie',
                 data: data2,
                 options: {
                   responsive: true,
                   plugins: {
                     legend: {
                       position: 'top',
                     },
                     title: {
                       display: false,
                       text: 'Chart.js Pie Chart'
                     }
                   }
                 },
               };
               myCharts.push(new Chart(graphArea, config2));
           }
           if (i>1 && j>1) {
               var graphArea = document.getElementById("cursorLayer" + i.toString() + ", " + j.toString()).getContext("2d");
               const data2 = {
                 labels: Object.keys(mydata[0]['states'][i - 1][(j - 1 - 1).toString()]['inventory']),
                 datasets: [
                   {
                     label: 'Dataset 1',
                     data: data_average(i - 1, j - 1 - 1),
                     backgroundColor: industries_clrs
                   }
                 ]
               };
               const config2 = {
                 type: 'pie',
                 data: data2,
                 options: {
                   responsive: true,
                   plugins: {
                     legend: {
                       position: 'top',
                     },
                     title: {
                       display: false,
                       text: 'Chart.js Pie Chart'
                     }
                   }
                 },
               };
               myCharts.push(new Chart(graphArea, config2));
               //myChart.destroy();
           }
       }
    }
}    

function drawGDPDevpl() {

    function data_average(agent_num) {
        //data: Object.keys(mydata[0]['states']).map(function (ep) {return mydata[0]['states'][ep][i.toString()]['endogenous']['GDP']}), // For all agents
        trials_num = Object.keys(mydata).length;
        data_array = [];
        for (let i=0; i<trials_num; i++) {
            data_array.push(Object.keys(mydata[i]['states']).map(function (ep) {return mydata[i]['states'][ep][agent_num.toString()]['endogenous']['GDP']}));
        }
        data_avg = [];
        data_len = data_array[0].length;
        for (let j=1; j<data_len; j++) {
            let sum = 0;
            for (let i=0; i<trials_num; i++) {
                sum = sum + data_array[i][j];
            }
            data_avg.push(sum/trials_num);
            sum = 0;
        }
        return data_avg;
    }

    var myTableDiv = document.getElementById("bigCanvas");
    var canvas = document.createElement("canvas");
    var canvas_id = "drawGDPDevplCanvas";
    canvas.id = canvas_id;
    canvas.width = 1000;
    canvas.height = 500;

    labels = [];
    for (let i=1; i<ep_num; i++) {
        labels.push("episode " + i.toString());
    }
    datasets = [];
    for (let i=0; i<agent_num; i++) {
        color_val1 = getRandomInt(255).toString();
        color_val2 = getRandomInt(255).toString();
        color_val3 = getRandomInt(255).toString();
        datasets.push({label: agent_names[i],
                       data: data_average(i),
                       backgroundColor: agent_clrs[i],
                       borderWidth: 1
                      });
    }
    //Object.keys(mydata[0]['states']).map(function (ep) {return "rgba(" + color_val1 + ", " + color_val2 + ", " + color_val3 + ", 1)"}),
    //data: Object.keys(mydata[0]['states']).map(function (ep) {return mydata[0]['states'][ep][i.toString()]['endogenous']['GDP']}), // For all agents
    const data = {
      labels: labels,
      datasets: datasets
    };
    const config = {
      type: 'bar',
      data: data,
      options: {
        scales: {
          x: {
            stacked: true,
          },
          y: {
            stacked: true,
            display: true,
            title: {
              display: true,
              text: '0.1 billion RMB',
              font: {
                size: 20
              }
            }
          }
        }
      }
    };

    var mydiv = document.createElement("div");
    var mytitle = document.createElement("h3");
    mytitle.setAttribute("style", "text-align: center;");
    var title = document.createTextNode("GDP Development"); 
    mytitle.appendChild(title);
    mydiv.appendChild(mytitle);
    mydiv.appendChild(canvas);
    myTableDiv.appendChild(mydiv);
    var graphArea = document.getElementById(canvas_id).getContext("2d");
    new Chart(graphArea, config);
}


function drawCarbonEmission() {

    function data_average(agent_num) {
        //Object.keys(mydata[0]['states']).map(function (ep) {return mydata[0]['states'][ep][i.toString()]['endogenous']['CO2']}), // For all agents
        trials_num = Object.keys(mydata).length;
        data_array = [];
        for (let i=0; i<trials_num; i++) {
            data_array.push(Object.keys(mydata[i]['states']).map(function (ep) {return mydata[i]['states'][ep][agent_num.toString()]['endogenous']['CO2']}));
        }
        data_avg = [];
        data_len = data_array[0].length;
        for (let j=1; j<data_len; j++) {
            let sum = 0;
            for (let i=0; i<trials_num; i++) {
                sum = sum + data_array[i][j];
            }
            data_avg.push(sum/trials_num);
            sum = 0;
        }
        return data_avg;
    }

    var myTableDiv = document.getElementById("bigCanvas");
    var canvas = document.createElement("canvas");
    var canvas_id = "drawCO2EmissionCanvas";
    canvas.id = canvas_id;
    canvas.width = 1000;
    canvas.height = 500;

    labels = [];
    for (let i=1; i<ep_num; i++) {
        labels.push("episode " + i.toString());
    }
    datasets = [];
    for (let i=0; i<agent_num; i++) {
        color_val1 = getRandomInt(255).toString();
        color_val2 = getRandomInt(255).toString();
        color_val3 = getRandomInt(255).toString();
        datasets.push({label: agent_names[i],
                       data: data_average(i),
                       backgroundColor: agent_clrs[i],
                       borderWidth: 1
                      });
    }
    //Object.keys(mydata[0]['states']).map(function (ep) {return "rgba(" + color_val1 + ", " + color_val2 + ", " + color_val3 + ", 1)"}),
    //data: Object.keys(mydata[0]['states']).map(function (ep) {return mydata[0]['states'][ep][i.toString()]['endogenous']['CO2']}), // For all agents

    const data = {
      labels: labels,
      datasets: datasets
    };
    const config = {
      type: 'bar',
      data: data,
      options: {
        scales: {
          x: {
            stacked: true,
          },
          y: {
            stacked: true,
            display: true,
            title: {
              display: true,
              text: '10 thousand tons',
              font: {
                size: 20
              }
            }
          }
        }
      },
    };

    var mydiv = document.createElement("div");
    var mytitle = document.createElement("h3");
    mytitle.setAttribute("style", "text-align: center;");
    var title = document.createTextNode("CO2 Emission"); 
    mytitle.appendChild(title);
    mydiv.appendChild(mytitle);
    mydiv.appendChild(canvas);
    myTableDiv.appendChild(mydiv);
    var graphArea = document.getElementById(canvas_id).getContext("2d");
    new Chart(graphArea, config);
}


function drawRewards() {

    function data_average(episode_num) {
        trials_num = Object.keys(mydata).length;
        data_array = [];
        for (let i=0; i<trials_num; i++) {
            data_row = Object.values(mydata[i]['rewards'][episode_num]);
            data_array.push(data_row.slice(0, data_row.length - 1));
        }
        data_avg = [];
        data_len = data_array[0].length;
        for (let j=0; j<data_len; j++) {
            let sum = 0;
            for (let i=0; i<trials_num; i++) {
                sum = sum + data_array[i][j];
            }
            data_avg.push(sum/trials_num);
            sum = 0;
        }
        return data_avg;
    }

    var table = document.createElement('TABLE');
    table.setAttribute("style", "margin-left: auto; margin-right: auto;");
    table.border='1';
    
    var tableBody = document.createElement('TBODY');
    table.appendChild(tableBody);

    rewards_ep_num = ep_num - 1;

    for (let i=0; i<rewards_ep_num+1; i++){
       var tr = document.createElement('TR');
       tableBody.appendChild(tr);
       for (let j=0; j<2; j++){
           if (i==0 && j==0) {
               var td = document.createElement('TD');
               td.width='25';
               td.class='background-image: linear-gradient(to bottom right,  transparent calc(50% - 1px), red, transparent calc(50% + 1px));';
               tr.appendChild(td);
           }
           else if (i==0 && j==1) {
               var td = document.createElement('TD');
               td.appendChild(document.createTextNode("Rewards"));
               td.width='500';
               tr.appendChild(td);
           }
           else if (j==0) {
               var td = document.createElement('TD');
               td.appendChild(document.createTextNode("Episode " + i));
               td.width='25';
               td.class='background-image: linear-gradient(to bottom right,  transparent calc(50% - 1px), red, transparent calc(50% + 1px));';
               tr.appendChild(td);
           }
           else {
               var canvas = document.createElement('canvas');
               var canvas_id = "rewardLayer" + i.toString() + ", " + j.toString();
               canvas.id = canvas_id;
               canvas.width = 500;
               canvas.height = 500;
               var td = document.createElement('TD');
               td.width='500';
               td.appendChild(canvas);
               tr.appendChild(td);
           }
       }
    }

    var mydiv = document.createElement("div");
    var mytitle = document.createElement("h3");
    mytitle.setAttribute("style", "text-align: center;");
    var title = document.createTextNode("Rewards of Each Agent"); 
    mytitle.appendChild(title);
    mydiv.appendChild(mytitle);
    mydiv.appendChild(table);
    var myTableDiv = document.getElementById("bigCanvas");
    myTableDiv.appendChild(mydiv);

    //Assign colours for each agent.
    clrs = Object.values(mydata[0]['rewards'][0]).map(function(i) {return "rgba(" + getRandomInt(255).toString() + ", " + getRandomInt(255).toString() + ", " + getRandomInt(255).toString() + ", " + getRandomInt(255).toString() + ")"});
    for (let i=0; i<rewards_ep_num+1; i++){
       for (let j=0; j<2; j++){
           if (i>0 && j>0) {
               var graphArea = document.getElementById("rewardLayer" + i.toString() + ", " + j.toString()).getContext("2d");
               const DATA_COUNT = 5;
               const NUMBER_CFG = {count: DATA_COUNT, min: 0, max: 100};
               
               labels = Object.keys(mydata[0]['rewards'][0]).map(function(agent) {return mydata[0]['states'][0][agent]['name']});
               labels = labels.slice(0, labels.length - 1)
               const data = {
                 labels: labels,
                 datasets: [
                   {
                     label: 'Dataset 1',
                     data: data_average(i - 1),
                     backgroundColor: agent_clrs
                   }
                 ]
               };
               const config = {
                 type: 'polarArea',
                 data: data,
                 options: {
                   responsive: true,
                   plugins: {
                     legend: {
                       position: 'top',
                     },
                     title: {
                       display: true,
                       text: '(Use log scale)'
                     },
                     scales: {
                       y: {
                            type: 'logarithmic'
                          },
                     }
                   }
                 },
               };

               myCharts.push(new Chart(graphArea, config));
           }
       }
    }
}

function drawResourcesPt() {

    function data_average(agent_num) {
        trials_num = Object.keys(mydata).length;
        data_array = [];
        for (let i=0; i<trials_num; i++) {
            data_array.push(Object.keys(mydata[i]['states']).map(function (ep) {return mydata[i]['states'][ep][agent_num.toString()]['resource_points']}));
        }
        data_avg = [];
        data_len = data_array[0].length;
        for (let j=1; j<data_len; j++) {
            let sum = 0;
            for (let i=0; i<trials_num; i++) {
                sum = sum + data_array[i][j];
            }
            data_avg.push(sum/trials_num);
            sum = 0;
        }
        return data_avg;
    }

    var myTableDiv = document.getElementById("bigCanvas");
    var canvas = document.createElement("canvas");
    var canvas_id = "drawResourcePointsCanvas";
    canvas.id = canvas_id;
    canvas.width = 1000;
    canvas.height = 500;

    labels = [];
    for (let i=1; i<ep_num; i++) {
        labels.push("episode " + i.toString());
    }
    datasets = [];
    for (let i=0; i<agent_num; i++) {
        color_val1 = getRandomInt(255).toString();
        color_val2 = getRandomInt(255).toString();
        color_val3 = getRandomInt(255).toString();
        datasets.push({label: agent_names[i],
                       data: data_average(i),
                       backgroundColor: agent_clrs[i],
                       borderWidth: 1
                      });
    }
    //Object.keys(mydata[0]['states']).map(function (ep) {return "rgba(" + color_val1 + ", " + color_val2 + ", " + color_val3 + ", 1)"}),
    //data: Object.keys(mydata[0]['states']).map(function (ep) {return sumOfArray(Object.values(mydata[0]['states'][ep][i.toString()]['inventory']))}), // For all agents
    const data = {
      labels: labels,
      datasets: datasets
    };
    const config = {
      type: 'bar',
      data: data,
      options: {
        scales: {
          y: {
            beginAtZero: true
          }
        }
      },
    };

    var mydiv = document.createElement("div");
    var mytitle = document.createElement("h3");
    mytitle.setAttribute("style", "text-align: center;");
    var title = document.createTextNode("Resource Points"); 
    mytitle.appendChild(title);
    mydiv.appendChild(mytitle);
    mydiv.appendChild(canvas);
    myTableDiv.appendChild(mydiv);
    var graphArea = document.getElementById(canvas_id).getContext("2d");
    new Chart(graphArea, config);
}


function drawActionsTaken() {

    function data_average(episode_num, agent_num) {
        trials_num = Object.keys(mydata).length;
        data_array = [];
        for (let i=0; i<trials_num; i++) {
            data_array.push(Object.values(mydata[i]['actions'][episode_num][(agent_num).toString()]));
        }
        data_avg = [];
        data_len = data_array[0].length;
        for (let j=0; j<data_len; j++) {
            let sum = 0;
            for (let i=0; i<trials_num; i++) {
                sum = sum + data_array[i][j];
            }
            data_avg.push(sum/trials_num);
            sum = 0;
        }
        return data_avg;
    }

    var table = document.createElement('TABLE');
    table.border='1';
    
    var tableBody = document.createElement('TBODY');
    table.appendChild(tableBody);
      
    for (let i=0; i<ep_num; i++){
       var tr = document.createElement('TR');
       tableBody.appendChild(tr);
       for (let j=0; j<agent_num+1+1; j++){
           if (i==0 && j==0) {
               var td = document.createElement('TD');
               td.width='75';
               td.class='background-image: linear-gradient(to bottom right,  transparent calc(50% - 1px), red, transparent calc(50% + 1px));';
               tr.appendChild(td);
           }
           else if (i==0 && j==1) {
               var td = document.createElement('TD');
               td.appendChild(document.createTextNode("Total"));
               td.width='75';
               tr.appendChild(td);
           }
           else if (i==0 && j>1) {
               var td = document.createElement('TD');
               //td.appendChild(document.createTextNode("Agent " + j));
               td.appendChild(document.createTextNode(mydata[0]['states'][0][(j-1-1).toString()]['name']));
               td.width='75';
               tr.appendChild(td);
           }
           else if (j==0) {
               var td = document.createElement('TD');
               td.appendChild(document.createTextNode("Episode " + i));
               td.width='75';
               tr.appendChild(td);
           }
           else {
               var canvas = document.createElement('canvas');
               var canvas_id = "actionsLayer" + i.toString() + ", " + j.toString();
               canvas.id = canvas_id;
               canvas.width = 300;
               canvas.height = 300;
               var td = document.createElement('TD');
               td.width='75';
               td.appendChild(canvas);
               tr.appendChild(td);
           }
       }
    }

    var mydiv = document.createElement("div");
    var mytitle = document.createElement("h3");
    mytitle.setAttribute("style", "text-align: center;");
    var title = document.createTextNode("Actions Taken"); 
    mytitle.appendChild(title);
    mydiv.appendChild(mytitle);
    mydiv.appendChild(table);
    var myTableDiv = document.getElementById("bigCanvas");
    myTableDiv.appendChild(mydiv);

    var sum = (r, a) => r.map((b, i) => a[i] + b);

    myCharts = [];
    for (let j=0; j<agent_num+1+1; j++) {
       clrs = [];
       clrs.push("rgba(" + getRandomInt(255).toString() + ", " + getRandomInt(255).toString() + ", " + getRandomInt(255).toString() + ", " + getRandomInt(255).toString() + ")");
       clrs = clrs.concat(agent_clrs);
       for (let i=0; i<ep_num; i++){
           if (i>0 && j==1) {
               everyAgents = Object.keys(mydata[0]['actions'][0]).slice(0, -1).map(function(agent) {return data_average(i - 1, agent)})
               var graphArea = document.getElementById("actionsLayer" + i.toString() + ", " + j.toString()).getContext("2d");
               const data2 = {
                 labels: Object.keys(mydata[0]['actions'][i - 1][(j - 1).toString()]).map(function(a) {return a}),
                 datasets: [
                   {
                     label: 'Actions',
                     data: everyAgents.reduce(sum),
                     backgroundColor: clrs[j - 1]
                   }
                 ]
               };
               const config2 = {
                 type: 'radar',
                 data: data2,
                 options: {
                   responsive: true,
                   plugins: {
                     legend: {
                       position: 'top',
                     },
                     title: {
                       display: false,
                       text: 'Chart.js Pie Chart'
                     }
                   }
                 },
               };
               myCharts.push(new Chart(graphArea, config2));
               //myChart.destroy();

           }
           if (i>0 && j>1) {
               var graphArea = document.getElementById("actionsLayer" + i.toString() + ", " + j.toString()).getContext("2d");
               const data2 = {
                 labels: Object.keys(mydata[0]['actions'][i - 1][(j - 1 - 1).toString()]).map(function(a) {return a}),
                 datasets: [
                   {
                     label: 'Actions',
                     data: data_average(i - 1, j - 1 - 1),
                     backgroundColor: clrs[j - 1]
                   }
                 ]
               };
               const config2 = {
                 type: 'radar',
                 data: data2,
                 options: {
                   responsive: true,
                   plugins: {
                     legend: {
                       position: 'top',
                     },
                     title: {
                       display: false,
                       text: 'Chart.js Pie Chart'
                     }
                   }
                 },
               };
               myCharts.push(new Chart(graphArea, config2));
               //myChart.destroy();
           }
       }
    }

}

function refreshPage() {
    dataIdx = 0;
    var counter = 0;
    for (let i=0; i<Object.keys(dataLoc).length; i++) {
        dataIdx = dataIdx + dataLoc[Object.keys(dataLoc)[i]] * (2**counter);
        counter = counter + 1;
    }
    loadData(dataIdx);
    //location.reload();
    console.log(dataIdx);

    //Clear canvas
    document.getElementById("bigCanvas").innerHTML = "";

    drawAgentColours();
    if (Object.keys(diffdata).length == 3) {
        drawDiffGDP();
        drawDiffCarbon();
    }
    parametersTable();
    drawIndustriesColours();
    drawIndustryComposition();
    drawIndustryDevpl();
    drawGDPDevpl();
    drawCarbonEmission();
    drawRewards();
    drawResourcesPt();
    drawActionsTaken();
}


function setSrc(k, v, j) {
    dataLoc[k] = j;
    var thisCell = document.getElementById(k);
    thisCell.innerHTML = "";
    thisCell.width='200';
    thisCell.height='75';
    thisCell.setAttribute('style', "text-align: center; font-size:25px;");
    thisCell.setAttribute('id', k);
    thisCell.appendChild(document.createTextNode(v));

}

function parametersTable() {

    var table = document.createElement('TABLE');
    var title = document.createElement('h3');
    title.appendChild(document.createTextNode("Parameters Table"));
    title.setAttribute("style", "text-align: center;");
    table.border='1';
    table.setAttribute("style", "margin-left: auto; margin-right: auto;")
    var tableBody = document.createElement('TBODY');
    table.appendChild(tableBody);

    parameterNames = {"Depreciation Rate of Industries (per year)": ["0%", "30%"], 
                      "GDP contribution from each industry (per year)": ["50%", "100%"], 
                      "CO2 contribution from each industry (per year)": ["70%", "100%"], 
                      "Reward Function": ["IRL", "GDP - CO2"]};

    for (let i=0; i<Object.keys(parameterNames).length; i++) {
        k = Object.keys(parameterNames)[i];
        dataLoc[k] = 0;
    }

    for (let i=0; i<Object.keys(parameterNames).length; i++){
       var tr = document.createElement('TR');
       tableBody.appendChild(tr);
       for (let j=0; j<parameterNames[Object.keys(parameterNames)[i]].length + 1 + 1; j++){
           if (j==0) {
               var parmName = document.createTextNode(Object.keys(parameterNames)[i]);
               var td = document.createElement('TD');
               td.width='75';
               td.height='75';
               td.setAttribute('style', "text-align: center; font-size:25px;");
               td.appendChild(parmName);
               tr.appendChild(td);
           }
           else if (j==parameterNames[Object.keys(parameterNames)[i]].length + 1) {
               var td = document.createElement('TD');
               td.width='200';
               td.height='75';
               td.setAttribute('style', "text-align: center; font-size:25px;");
               k = Object.keys(parameterNames)[i];
               td.setAttribute('id', k);
               td.appendChild(document.createTextNode(parameterNames[k][dataLoc[k]]));
               tr.appendChild(td);
           }
           else {
               var mybutton = document.createElement('button');
               mybutton.setAttribute("onclick", "setSrc('" + Object.keys(parameterNames)[i] + "', '" + parameterNames[Object.keys(parameterNames)[i]][j-1] + "', '" + (j-1).toString() + "')");
               var td = document.createElement('TD');
               td.width='75';
               td.height='75';
               mybutton.appendChild(document.createTextNode(Object.values(parameterNames)[i][j - 1]));
               td.setAttribute('style', "text-align: center; font-size:20px;");
               td.appendChild(mybutton);
               tr.appendChild(td);
           }
       }
    }

    var td = document.createElement('TD');
    td.setAttribute('colspan', "4");
    td.width='600';
    td.height='75';
    var tr = document.createElement('TR');
    var mybutton = document.createElement('button');
    mybutton.setAttribute("onclick", "refreshPage()");
    mybutton.appendChild(document.createTextNode("Refresh"));
    mybutton.setAttribute("style", "padding: 32px 300px; text-align: center; font-size:35px;");
    td.appendChild(mybutton);
    tr.appendChild(td);
    tableBody.appendChild(tr);

    var parametersDiv = document.createElement('div');
    parametersDiv.appendChild(title);
    parametersDiv.appendChild(table);
    var myTableDiv = document.getElementById("bigCanvas");
    myTableDiv.appendChild(parametersDiv);

}


function drawDiffGDP() {
    const DATA_COUNT = 7;
    const NUMBER_CFG = {count: DATA_COUNT, min: -100, max: 100};
    
    datasets = [];
    for (let i=0; i<Object.keys(diffdata["GDP"]).length; i++) {
        agent_name = Object.keys(diffdata["GDP"])[i];
        datasets.push({label: agent_name,
                       data: diffdata["GDP"][agent_name][1],
                       backgroundColor: agent_clrs[i]
                       });
        labels = diffdata["GDP"][agent_name][0];
    }
    //                   backgroundColor: "rgba(" + getRandomInt(255).toString() + ", " + getRandomInt(255).toString() + ", " + getRandomInt(255).toString() + ", 1)"

    const data = {
      labels: labels,
      datasets: datasets,
    };
    
    const config = {
      type: 'bar',
      data: data,
      options: {
        responsive: true,
        scales: {
          y: {
            display: true,
            title: {
              display: true,
              text: '0.1 Billion RMB',
              font: {
                size: 20
              }
            }
          }
        },
        plugins: {
          legend: {
            position: 'top',
          },
          title: {
            display: false,
            text: 'Chart.js Bar Chart'
          }
        }
      },
    };

    var myTableDiv = document.getElementById("bigCanvas");
    var canvas = document.createElement("canvas");
    var canvas_id = "drawDiffGDPCanvas";
    canvas.id = canvas_id;
    //canvas.width = 800;
    //canvas.height = 400;

    var mydiv = document.createElement("div");
    //mydiv.style.height = "450px";
    //mydiv.style.width = "800px";
    var mytitle = document.createElement("h3");
    mytitle.setAttribute("style", "text-align: center;");
    var title = document.createTextNode("Simulated GDP Net of Historical GDP"); 
    mytitle.appendChild(title);
    mydiv.appendChild(mytitle);
    mydiv.appendChild(canvas);
    myTableDiv.appendChild(mydiv);

    var graphArea = document.getElementById(canvas_id).getContext("2d");
    new Chart(graphArea, config);

}

function drawDiffCarbon() {
    const DATA_COUNT = 7;
    const NUMBER_CFG = {count: DATA_COUNT, min: -100, max: 100};
    
    datasets = [];
    for (let i=0; i<Object.keys(diffdata["CO2"]).length; i++) {
        agent_name = Object.keys(diffdata["CO2"])[i];
        datasets.push({label: agent_name,
                       data: diffdata["CO2"][agent_name][1],
                       backgroundColor: agent_clrs[i]
                       });
        labels = diffdata["CO2"][agent_name][0];
    }
    //                   backgroundColor: "rgba(" + getRandomInt(255).toString() + ", " + getRandomInt(255).toString() + ", " + getRandomInt(255).toString() + ", 1)"

    const data = {
      labels: labels,
      datasets: datasets,
    };
    
    const config = {
      type: 'bar',
      data: data,
      options: {
        responsive: true,
        scales: {
          y: {
            display: true,
            title: {
              display: true,
              text: '10 thousand tons',
              font: {
                size: 20
              }
            }
          }
        },
        plugins: {
          legend: {
            position: 'top',
          },
          title: {
            display: false,
            text: 'Chart.js Bar Chart'
          }
        }
      },
    };

    var myTableDiv = document.getElementById("bigCanvas");
    var canvas = document.createElement("canvas");
    var canvas_id = "drawDiffCarbonCanvas";
    canvas.id = canvas_id;
    //canvas.width = 800;
    //canvas.height = 400;

    var mydiv = document.createElement("div");
    //mydiv.style.height = "450px";
    //mydiv.style.width = "800px";
    var mytitle = document.createElement("h3");
    mytitle.setAttribute("style", "text-align: center;");
    var title = document.createTextNode("Simulated CO2 Net of Historical CO2"); 
    mytitle.appendChild(title);
    mydiv.appendChild(mytitle);
    mydiv.appendChild(canvas);
    myTableDiv.appendChild(mydiv);

    var graphArea = document.getElementById(canvas_id).getContext("2d");
    new Chart(graphArea, config);

}

function drawAgentColours() {

    var table = document.createElement('TABLE');
    table.setAttribute("style", "margin-left: auto; margin-right: auto;");
    table.border='1';
    
    var tableBody = document.createElement('TBODY');
    table.appendChild(tableBody);

    for (let i=0; i<AGENT_COUNT; i++){
       var tr = document.createElement('TR');
       tableBody.appendChild(tr);
       for (let j=0; j<2; j++) {
           var td = document.createElement('TD');
           if (j==0) {
               td.appendChild(document.createTextNode(mydata[0]['states'][0][i.toString()]['name']));
           }
           else {
               td.setAttribute("style", "background-color:" + agent_clrs[i] + ";");
           }
           td.width='75';
           tr.appendChild(td);
       }
    }

    var mydiv = document.createElement("div");
    var mytitle = document.createElement("h3");
    mytitle.setAttribute("style", "text-align: center;");
    var title = document.createTextNode("Colours Assigned to Agents"); 
    mytitle.appendChild(title);
    mydiv.appendChild(mytitle);
    mydiv.appendChild(table);
    var myTableDiv = document.getElementById("bigCanvas");
    myTableDiv.appendChild(mydiv);

}

function drawIndustriesColours() {

    var table = document.createElement('TABLE');
    table.setAttribute("style", "margin-left: auto; margin-right: auto;");
    table.border='1';
    
    var tableBody = document.createElement('TBODY');
    table.appendChild(tableBody);

    for (let i=0; i<INDUSTRIES_COUNT; i++){
       var tr = document.createElement('TR');
       tableBody.appendChild(tr);
       for (let j=0; j<2; j++) {
           var td = document.createElement('TD');
           if (j==0) {
               td.appendChild(document.createTextNode(Object.keys(mydata[0]['states'][0]['0']['inventory'])[i]));
           }
           else {
               td.setAttribute("style", "background-color:" + industries_clrs[i] + ";");
           }
           td.width='75';
           tr.appendChild(td);
       }
    }

    var mydiv = document.createElement("div");
    var mytitle = document.createElement("h3");
    mytitle.setAttribute("style", "text-align: center;");
    var title = document.createTextNode("Colours Assigned to Industries"); 
    mytitle.appendChild(title);
    mydiv.appendChild(mytitle);
    mydiv.appendChild(table);
    var myTableDiv = document.getElementById("bigCanvas");
    myTableDiv.appendChild(mydiv);

}

function loadData(idx) {
    mydata = JSON.parse(jsonDATA[idx]);
    diffdata = JSON.parse(diffDATA[idx]);
}


/* Main program */
var dataIdx = 0;
var dataLoc = {};
var mydata = {};
var diffdata = {};
loadData(dataIdx);
//total, agent 1, agent 2, ..., agent n.
const AGENT_COUNT = Object.keys(mydata[0]['states'][0]).length - 1;
const INDUSTRIES_COUNT = Object.keys(mydata[0]['states'][0]['0']['inventory']).length;

template_colours = ['rgba(97, 78, 196, 49)', 'rgba(97, 18, 121, 108)', 'rgba(86, 140, 196, 52)', 'rgba(27, 166, 7, 247)', 'rgba(78, 16, 40, 139)', 'rgba(164, 219, 128, 47)', 'rgba(14, 136, 87, 153)', 'rgba(162, 183, 88, 130)', 'rgba(83, 229, 191, 11)', 'rgba(234, 224, 251, 39)', 'rgba(29, 239, 149, 178)', 'rgba(112, 199, 152, 74)', 'rgba(131, 93, 193, 86)', 'rgba(127, 182, 67, 187)', 'rgba(56, 168, 121, 9)', 'rgba(61, 226, 3, 71)', 'rgba(131, 189, 3, 11)', 'rgba(224, 148, 233, 152)', 'rgba(124, 223, 113, 196)', 'rgba(208, 245, 122, 90)'];
template_colours2 = ['rgba(43, 214, 54, 236)', 'rgba(248, 3, 147, 97)', 'rgba(95, 53, 204, 45)', 'rgba(146, 40, 225, 41)', 'rgba(193, 203, 216, 239)', 'rgba(234, 115, 137, 123)', 'rgba(109, 240, 103, 219)', 'rgba(60, 24, 87, 145)', 'rgba(47, 89, 138, 193)', 'rgba(72, 125, 5, 206)', 'rgba(22, 101, 115, 233)', 'rgba(28, 123, 39, 245)', 'rgba(30, 75, 148, 157)', 'rgba(35, 36, 135, 46)', 'rgba(242, 95, 36, 94)', 'rgba(73, 7, 87, 234)', 'rgba(137, 212, 87, 29)', 'rgba(45, 48, 11, 47)', 'rgba(138, 132, 97, 120)', 'rgba(209, 159, 137, 173)'];

industries_clrs = [];
for(let i=0; i<INDUSTRIES_COUNT; i++) {
    industries_clrs.push(template_colours2[i]);
}

agent_clrs = [];
for(let i=0; i<AGENT_COUNT; i++) {
    agent_clrs.push(template_colours[i]);
}

/* global variables */
agent_num = Object.keys(mydata[0]['states'][0]).length - 1;
ep_num = Object.keys(mydata[0]['states']).length;
agent_names = Object.keys(mydata[0]['states'][0]).map(function (agent) {return mydata[0]['states'][0][agent]['name']}).slice(0, agent_num);

drawAgentColours();
if (Object.keys(diffdata).length == 3) {
    drawDiffGDP();
    drawDiffCarbon();
}
parametersTable();
drawIndustriesColours();
drawIndustryComposition();
drawIndustryDevpl();
drawGDPDevpl();
drawCarbonEmission();
drawRewards();
drawResourcesPt();
drawActionsTaken();
