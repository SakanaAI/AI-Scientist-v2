const bgCol = "#FFFFFF";
const accentCol = "#1a439e";

hljs.initHighlightingOnLoad();

// Function to update background color globally
function updateBackgroundColor(color) {
  // Update the JS variable
  window.bgColCurrent = color;

  // Update body background
  document.body.style.backgroundColor = color;

  // Update canvas container background
  const canvasContainer = document.getElementById('canvas-container');
  if (canvasContainer) {
    canvasContainer.style.backgroundColor = color;
  }
}

// Store tree data for each stage
const stageData = {
  Stage_1: null,
  Stage_2: null,
  Stage_3: null,
  Stage_4: null
};

// Keep track of current selected stage
let currentStage = null;
let currentSketch = null;
let availableStages = [];

// Class definitions for nodes and edges
class Node {
  constructor(x, y, id, isRoot = false) {
    this.x = x;
    this.y = y;
    this.id = id;
    this.visible = isRoot; // Only root nodes are visible initially
    this.appearProgress = 0;
    this.popEffect = 0;
    this.selected = false;
    this.isRootNode = isRoot;
  }

  update() {
    if (this.visible) {
      // Handle the main appearance animation
      if (this.appearProgress < 1) {
        this.appearProgress += 0.06;

        // When we reach full size, trigger the pop effect
        if (this.appearProgress >= 1) {
          this.appearProgress = 1; // Cap at 1
          this.popEffect = 1; // Start the pop effect
        }
      }

      // Handle the pop effect animation
      if (this.popEffect > 0) {
        this.popEffect -= 0.15; // Control how quickly it shrinks back
        if (this.popEffect < 0) this.popEffect = 0; // Don't go negative
      }
    }
  }

  startAnimation() {
    this.visible = true;
  }

  color() {
    if (this.selected) {
      return accentCol; // Use the global accent color variable for selected node
    }
    return '#4263eb'; // Default blue color
  }

  render(p5) {
    if (this.visible) {
      const popBonus = this.popEffect * 0.1;
      const nodeScale = p5.map(this.appearProgress, 0, 1, 0, 1) + popBonus;
      const alpha = p5.map(this.appearProgress, 0, 1, 0, 255);

      p5.push();
      p5.translate(this.x, this.y);

      // Shadow effect
      p5.noStroke();
      p5.rectMode(p5.CENTER);

      for (let i = 1; i <= 4; i++) {
        p5.fill(0, 0, 0, alpha * 0.06);
        p5.rect(i, i, 30 * nodeScale, 30 * nodeScale, 10);
      }

      // Main square - use node's color with alpha
      let nodeColor = p5.color(this.color());
      nodeColor.setAlpha(alpha);
      p5.fill(nodeColor);
      p5.rect(0, 0, 30 * nodeScale, 30 * nodeScale, 10);

      // Draw checkmark icon if the node is selected
      if (this.selected && this.appearProgress >= 1) {
        p5.stroke(255);
        p5.strokeWeight(2 * nodeScale);
        p5.noFill();
        // Draw checkmark
        p5.beginShape();
        p5.vertex(-8, 0);
        p5.vertex(-3, 5);
        p5.vertex(8, -6);
        p5.endShape();
      }

      p5.pop();
    }
  }

  isMouseOver(p5) {
    return this.visible &&
           p5.mouseX > this.x - 15 &&
           p5.mouseX < this.x + 15 &&
           p5.mouseY > this.y - 15 &&
           p5.mouseY < this.y + 15;
  }

  // Connect this node to a child node
  child(childNode) {
    // Create an edge from this node to the child
    let isLeft = childNode.x < this.x;
    let isRight = childNode.x > this.x;
    let edge = new Edge(this, childNode, isLeft, isRight);
    return edge;
  }
}

class Edge {
  constructor(parent, child, isLeft, isRight) {
    this.parent = parent;
    this.child = child;
    this.isLeft = isLeft;
    this.isRight = isRight;
    this.progress = 0;

    // Calculate the midpoint where branching occurs
    this.midY = parent.y + (child.y - parent.y) * 0.6;

    // Use the actual child x-coordinate
    // This ensures the edge will connect directly to the child node
    this.branchX = child.x;
  }

  update() {
    if (this.parent.visible && this.progress < 1) {
      this.progress += 0.01; // Adjust animation speed
    }
    if (this.progress >= 1) {
      this.child.visible = true;
    }
  }

  color() {
    return this.child.color();
  }

  render(p5) {
    if (!this.parent.visible) return;

    // Calculate path lengths
    const verticalDist1 = this.midY - this.parent.y;
    const horizontalDist = Math.abs(this.branchX - this.parent.x);
    const verticalDist2 = this.child.y - this.midY;
    const totalLength = verticalDist1 + horizontalDist + verticalDist2;

    // Calculate how much of each segment to draw
    const currentLength = totalLength * this.progress;

    p5.stroke(180, 190, 205);
    p5.strokeWeight(1.5);
    p5.noFill();

    // Always draw the first vertical segment from parent
    if (currentLength > 0) {
      const firstSegmentLength = Math.min(currentLength, verticalDist1);
      const currentMidY = p5.lerp(this.parent.y, this.midY, firstSegmentLength / verticalDist1);
      p5.line(this.parent.x, this.parent.y, this.parent.x, currentMidY);
    }

    if (currentLength > verticalDist1) {
      // Draw second segment (horizontal)
      const secondSegmentLength = Math.min(currentLength - verticalDist1, horizontalDist);
      const currentBranchX = p5.lerp(this.parent.x, this.branchX, secondSegmentLength / horizontalDist);
      p5.line(this.parent.x, this.midY, currentBranchX, this.midY);

      if (currentLength > verticalDist1 + horizontalDist) {
        // Draw third segment (vertical to child)
        const thirdSegmentLength = currentLength - verticalDist1 - horizontalDist;
        const currentChildY = p5.lerp(this.midY, this.child.y, thirdSegmentLength / verticalDist2);
        p5.line(this.branchX, this.midY, this.branchX, currentChildY);
      }
    }
  }
}

// Create a modified sketch for each stage
function createTreeSketch(stageId) {
  return function(p5) {
    let nodes = [];
    let edges = [];
    let treeData = stageData[stageId];

    p5.setup = function() {
      const canvas = p5.createCanvas(p5.windowWidth * 0.4, p5.windowHeight);
      canvas.parent('canvas-container');
      p5.smooth();
      p5.frameRate(60);

      if (treeData) {
        createTreeFromData(treeData);
      }
    };

    p5.windowResized = function() {
      p5.resizeCanvas(p5.windowWidth * 0.4, p5.windowHeight);
    };

    function createTreeFromData(data) {
      // Clear existing nodes and edges
      nodes = [];
      edges = [];

      // Add defensive checks to prevent errors
      if (!data || !data.layout || !Array.isArray(data.layout) || !data.edges || !Array.isArray(data.edges)) {
        console.error("Invalid tree data format:", data);
        return; // Exit if data structure is invalid
      }

      // Find all parent nodes in edges
      const parentNodes = new Set();
      for (const [parentId, childId] of data.edges) {
        parentNodes.add(parentId);
      }

      // Create nodes
      for (let i = 0; i < data.layout.length; i++) {
        const [nx, ny] = data.layout[i];
        // A node is a root if it's a parent and not a child in any edge
        const isRoot = parentNodes.has(i) && data.edges.every(edge => edge[1] !== i);

        const node = new Node(
          nx * p5.width * 0.8 + p5.width * 0.1,
          ny * p5.height * 0.8 + p5.height * 0.1,
          i,
          isRoot
        );
        nodes.push(node);
      }

      // If no root was found, make the first parent node visible
      if (!nodes.some(node => node.visible) && parentNodes.size > 0) {
        // Get the first parent node
        const firstParentId = [...parentNodes][0];
        if (nodes[firstParentId]) {
          nodes[firstParentId].visible = true;
        }
      }

      // Create edges
      for (const [parentId, childId] of data.edges) {
        const parent = nodes[parentId];
        const child = nodes[childId];
        if (parent && child) { // Verify both nodes exist
          const isLeft = child.x < parent.x;
          const isRight = child.x > parent.x;
          edges.push(new Edge(parent, child, isLeft, isRight));
        }
      }

      // Select the first node by default
      if (nodes.length > 0) {
        nodes[0].selected = true;
        updateNodeInfo(0);
      }
    }

    p5.draw = function() {
      // Use the global background color if available, otherwise use the default bgCol
      const currentBgColor = window.bgColCurrent || bgCol;
      p5.background(currentBgColor);

      // Update and render edges
      for (const edge of edges) {
        edge.update();
        edge.render(p5);
      }

      // Update and render nodes
      for (const node of nodes) {
        node.update();
        node.render(p5);
      }

      // Handle mouse hover
      p5.cursor(p5.ARROW);
      for (const node of nodes) {
        if (node.isMouseOver(p5)) {
          p5.cursor(p5.HAND);
        }
      }
    };

    p5.mousePressed = function() {
      // Check if any node was clicked
      for (let i = 0; i < nodes.length; i++) {
        if (nodes[i].visible && nodes[i].isMouseOver(p5)) {
          // Deselect all nodes
          nodes.forEach(n => n.selected = false);
          // Select the clicked node
          nodes[i].selected = true;
          // Update the right panel with node info
          updateNodeInfo(i);
          break;
        }
      }
    };

    function updateNodeInfo(nodeIndex) {
      if (treeData) {
        setNodeInfo(
          treeData.code[nodeIndex],
          treeData.plan[nodeIndex],
          treeData.plot_code?.[nodeIndex],
          treeData.plot_plan?.[nodeIndex],
          treeData.metrics?.[nodeIndex],
          treeData.exc_type?.[nodeIndex] || '',
          treeData.exc_info?.[nodeIndex]?.args?.[0] || '',
          treeData.exc_stack?.[nodeIndex] || [],
          treeData.plots?.[nodeIndex] || [],
          treeData.plot_analyses?.[nodeIndex] || [],
          treeData.vlm_feedback_summary?.[nodeIndex] || '',
          treeData.datasets_successfully_tested?.[nodeIndex] || [],
          treeData.exec_time_feedback?.[nodeIndex] || '',
          treeData.exec_time?.[nodeIndex] || ''
        );
      }
    }
  };
}

// Start a new p5 sketch for the given stage
function startSketch(stageId) {
  if (currentSketch) {
    currentSketch.remove();
  }

  if (stageData[stageId]) {
    currentSketch = new p5(createTreeSketch(stageId));

    // Update stage info
    const stageNumber = stageId.split('_')[1];
    let stageDesc = '';
    switch(stageId) {
      case 'Stage_1': stageDesc = 'Preliminary Investigation'; break;
      case 'Stage_2': stageDesc = 'Baseline Tuning'; break;
      case 'Stage_3': stageDesc = 'Research Agenda Execution'; break;
      case 'Stage_4': stageDesc = 'Ablation Studies'; break;
    }

    document.getElementById('stage-info').innerHTML =
      `<strong>Current Stage: ${stageNumber} - ${stageDesc}</strong>`;
  }
}

// Handle tab selection
function selectStage(stageId) {
  if (!stageData[stageId] || !availableStages.includes(stageId)) {
    return; // Don't allow selection of unavailable stages
  }

  // Update active tab styles
  document.querySelectorAll('.tab').forEach(tab => {
    tab.classList.remove('active');
  });
  document.querySelector(`.tab[data-stage="${stageId}"]`).classList.add('active');

  // Start the new sketch
  currentStage = stageId;
  startSketch(stageId);
}

// Function to load the tree data for all stages
async function loadAllStageData(baseTreeData) {
  console.log("Loading stage data with base data:", baseTreeData);

  // The base tree data is for the current stage
  const currentStageId = baseTreeData.current_stage || 'Stage_1';

  // Ensure base tree data is valid and has required properties
  if (baseTreeData && baseTreeData.layout && baseTreeData.edges) {
    stageData[currentStageId] = baseTreeData;
    availableStages.push(currentStageId);
    console.log(`Added current stage ${currentStageId} to available stages`);
  } else {
    console.warn(`Current stage ${currentStageId} data is invalid:`, baseTreeData);
  }

  // Use relative path to load other stage trees
  const logDirPath = baseTreeData.log_dir_path || '.';
  console.log("Log directory path:", logDirPath);

  // Load data for each stage if available
  const stageNames = ['Stage_1', 'Stage_2', 'Stage_3', 'Stage_4'];
  const stageNames2actualNames = {
    'Stage_1': 'stage_1_initial_implementation_1_preliminary',
    'Stage_2': 'stage_2_baseline_tuning_1_first_attempt',
    'Stage_3': 'stage_3_creative_research_1_first_attempt',
    'Stage_4': 'stage_4_ablation_studies_1_first_attempt'
    }

  for (const stage of stageNames) {

    if (baseTreeData.completed_stages && baseTreeData.completed_stages.includes(stage)) {
      try {
        console.log(`Attempting to load data for ${stage} from ${logDirPath}/${stageNames2actualNames[stage]}/tree_data.json`);
        const response = await fetch(`${logDirPath}/${stageNames2actualNames[stage]}/tree_data.json`);

        if (response.ok) {
          const data = await response.json();

          // Validate the loaded data
          if (data && data.layout && data.edges) {
            stageData[stage] = data;
            availableStages.push(stage);
            console.log(`Successfully loaded and validated data for ${stage}`);
          } else {
            console.warn(`Loaded data for ${stage} is invalid:`, data);
          }
        } else {
          console.warn(`Failed to load data for ${stage} - HTTP status ${response.status}`);
        }
      } catch (error) {
        console.error(`Error loading data for ${stage}:`, error);
      }
    } else {
      console.log(`Skipping stage ${stage} - not in completed stages list:`, baseTreeData.completed_stages);
    }
  }

  // Update tab visibility based on available stages
  updateTabVisibility();

  // Start with the first available stage
  if (availableStages.length > 0) {
    selectStage(availableStages[0]);
  } else {
    console.warn("No stages available to display");
    // Display a message in the canvas area
    document.getElementById('canvas-container').innerHTML =
      '<div style="padding: 20px; color: #333; text-align: center;"><h3>No valid tree data available to display</h3></div>';
  }
}

// Update tab visibility based on available stages
function updateTabVisibility() {
  const tabs = document.querySelectorAll('.tab');
  tabs.forEach(tab => {
    const stageId = tab.getAttribute('data-stage');
    if (availableStages.includes(stageId)) {
      tab.classList.remove('disabled');
    } else {
      tab.classList.add('disabled');
    }
  });
}

// Utility function to set the node info in the right panel
const setNodeInfo = (code, plan, plot_code, plot_plan, metrics = null, exc_type = '', exc_info = '',
    exc_stack = [], plots = [], plot_analyses = [], vlm_feedback_summary = '',
    datasets_successfully_tested = [], exec_time_feedback = '', exec_time = '') => {
  const codeElm = document.getElementById("code");
  if (codeElm) {
    if (code) {
      codeElm.innerHTML = hljs.highlight(code, { language: "python" }).value;
    } else {
      codeElm.innerHTML = '<p>No code available</p>';
    }
  }

  const planElm = document.getElementById("plan");
  if (planElm) {
    if (plan) {
      planElm.innerHTML = hljs.highlight(plan, { language: "plaintext" }).value;
    } else {
      planElm.innerHTML = '<p>No plan available</p>';
    }
  }

  const plot_codeElm = document.getElementById("plot_code");
  if (plot_codeElm) {
    if (plot_code) {
      plot_codeElm.innerHTML = hljs.highlight(plot_code, { language: "python" }).value;
    } else {
      plot_codeElm.innerHTML = '<p>No plot code available</p>';
    }
  }

  const plot_planElm = document.getElementById("plot_plan");
  if (plot_planElm) {
    if (plot_plan) {
      plot_planElm.innerHTML = hljs.highlight(plot_plan, { language: "plaintext" }).value;
    } else {
      plot_planElm.innerHTML = '<p>No plot plan available</p>';
    }
  }

  const metricsElm = document.getElementById("metrics");
  if (metricsElm) {
      let metricsContent = `<h3>Metrics:</h3>`;
      if (metrics && metrics.metric_names) {
          for (const metric of metrics.metric_names) {
              metricsContent += `<div class="metric-group">`;
              metricsContent += `<h4>${metric.metric_name}</h4>`;
              metricsContent += `<p><strong>Description:</strong> ${metric.description || 'N/A'}</p>`;
              metricsContent += `<p><strong>Optimization:</strong> ${metric.lower_is_better ? 'Minimize' : 'Maximize'}</p>`;

              // Create table for dataset values
              metricsContent += `<table class="metric-table">
                  <tr>
                      <th>Dataset</th>
                      <th>Final Value</th>
                      <th>Best Value</th>
                  </tr>`;

              for (const dataPoint of metric.data) {
                  metricsContent += `<tr>
                      <td>${dataPoint.dataset_name}</td>
                      <td>${dataPoint.final_value?.toFixed(4) || 'N/A'}</td>
                      <td>${dataPoint.best_value?.toFixed(4) || 'N/A'}</td>
                  </tr>`;
              }

              metricsContent += `</table></div>`;
          }
      } else if (metrics === null) {
          metricsContent += `<p>No metrics available</p>`;
      }
      metricsElm.innerHTML = metricsContent;
  }

  // Add plots display
  const plotsElm = document.getElementById("plots");
  if (plotsElm) {
      if (plots && plots.length > 0) {
          let plotsContent = '';
          plots.forEach(plotPath => {
              plotsContent += `
                  <div class="plot-item">
                      <img src="${plotPath}" alt="Experiment Plot" onerror="console.error('Failed to load plot:', this.src)"/>
                  </div>`;
          });
          plotsElm.innerHTML = plotsContent;
      } else {
          plotsElm.innerHTML = '';
      }
  }

  // Add error info display
  const errorElm = document.getElementById("exc_info");
  if (errorElm) {
    if (exc_type) {
      let errorContent = `<h3 style="color: #ff5555">Exception Information:</h3>
                          <p><strong>Type:</strong> ${exc_type}</p>`;

      if (exc_info) {
        errorContent += `<p><strong>Details:</strong> <pre>${JSON.stringify(exc_info, null, 2)}</pre></p>`;
      }

      if (exc_stack) {
        errorContent += `<p><strong>Stack Trace:</strong> <pre>${exc_stack.join('\n')}</pre></p>`;
      }

      errorElm.innerHTML = errorContent;
    } else {
      errorElm.innerHTML = "No exception info available";
    }
  }

  const exec_timeElm = document.getElementById("exec_time");
  if (exec_timeElm) {
    let exec_timeContent = '<div id="exec_time"><h3>Execution Time (in seconds):</h3><p>' + exec_time + '</p></div>';
    exec_timeElm.innerHTML = exec_timeContent;
  }

  const exec_time_feedbackElm = document.getElementById("exec_time_feedback");
  if (exec_time_feedbackElm) {
    let exec_time_feedbackContent = '<div id="exec_time_feedback_content">'
    exec_time_feedbackContent += '<h3>Execution Time Feedback:</h3>'
    exec_time_feedbackContent += '<p>' + exec_time_feedback + '</p>'
    exec_time_feedbackContent += '</div>';
    exec_time_feedbackElm.innerHTML = exec_time_feedbackContent;
  }

  const vlm_feedbackElm = document.getElementById("vlm_feedback");
  if (vlm_feedbackElm) {
      let vlm_feedbackContent = '';

      if (plot_analyses && plot_analyses.length > 0) {
          vlm_feedbackContent += `<h3>Plot Analysis:</h3>`;
          plot_analyses.forEach(analysis => {
              if (analysis && analysis.plot_path) {  // Add null check
                  vlm_feedbackContent += `
                      <div class="plot-analysis">
                          <h4>Analysis for ${analysis.plot_path.split('/').pop()}</h4>
                          <p>${analysis.analysis || 'No analysis available'}</p>
                          <ul class="key-findings">
                              ${(analysis.key_findings || []).map(finding => `<li>${finding}</li>`).join('')}
                          </ul>
                      </div>`;
              } else {
                  console.warn('Received invalid plot analysis:', analysis);
                  vlm_feedbackContent += `
                      <div class="plot-analysis">
                          <p>Invalid plot analysis data received</p>
                      </div>`;
              }
          });
      }

      // Add actionable insights if available
      if (vlm_feedback_summary && typeof vlm_feedback_summary === 'string') {
          vlm_feedbackContent += `
              <div class="vlm_feedback">
                  <h3>VLM Feedback Summary:</h3>
                  <p>${vlm_feedback_summary}</p>
              </div>`;
      }

      console.log("Datasets successfully tested:", datasets_successfully_tested);
      if (datasets_successfully_tested && datasets_successfully_tested.length > 0) {
          vlm_feedbackContent += `
              <div id="datasets_successfully_tested">
                  <h3>Datasets Successfully Tested:</h3>
                  <p>${datasets_successfully_tested.join(', ')}</p>
              </div>`;
      }

      if (!vlm_feedbackContent) {
          vlm_feedbackContent = '<p>No insights available for this experiment.</p>';
      }

      vlm_feedbackElm.innerHTML = vlm_feedbackContent;
  }

  const datasets_successfully_testedElm = document.getElementById("datasets_successfully_tested");
  if (datasets_successfully_testedElm) {
      let datasets_successfully_testedContent = '';
      if (datasets_successfully_tested && datasets_successfully_tested.length > 0) {
          datasets_successfully_testedContent = `<h3>Datasets Successfully Tested:</h3><ul>`;
          datasets_successfully_tested.forEach(dataset => {
              datasets_successfully_testedContent += `<li>${dataset}</li>`;
          });
          datasets_successfully_testedContent += `</ul>`;
      } else {
          datasets_successfully_testedContent = '<p>No datasets tested yet</p>';
      }
      datasets_successfully_testedElm.innerHTML = datasets_successfully_testedContent;
  }
};

// Initialize with the provided tree data
const treeStructData = "PLACEHOLDER_TREE_DATA";

// Add log directory path and stage info to the tree data
treeStructData.log_dir_path = window.location.pathname.split('/').slice(0, -1).join('/');
treeStructData.current_stage = window.location.pathname.includes('stage_')
  ? window.location.pathname.split('stage_')[1].split('/')[0]
  : 'Stage_1';

// Initialize background color
window.bgColCurrent = bgCol;

// Function to set background color that can be called from the console
function setBackgroundColor(color) {
  // Update the global color
  updateBackgroundColor(color);

  // Refresh the current sketch to apply the new background color
  if (currentStage) {
    startSketch(currentStage);
  }
}

// Load all stage data and initialize the visualization
loadAllStageData(treeStructData);
