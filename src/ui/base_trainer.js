/**
 * Base Trainer - Common functionality for all training interfaces
 */
function BaseTrainer() {
    this.canvas = null;
    this.ctx = null;
    this.simulation = null;
    this.isTraining = false;
    this.frameCount = 0;
    this.speedMultiplier = 1.0;
    this.displayEnabled = true; // New property to control rendering
    this.autoRestartThreshold = -1; // Auto-restart when boid count drops below this (-1 = disabled)
    this.showVisionRange = true; // New property to control vision range visualization
    
    // Neural network monitor properties
    this.neuralMonitorEnabled = true;
    this.neuralMonitorUpdateInterval = 10; // Update every 10 frames
    this.neuralMonitorFrameCount = 0;
    
    // Loss chart properties
    this.lossChart = null;
    this.lossChartCtx = null;
    this.lossHistory = [];
    this.maxDisplayPoints = 200; // Maximum points to display for performance
    
    // Training components - updated for new architecture
    this.neuralNetwork = new NeuralNetwork(204, 64, 32, 2);
    this.inputProcessor = new InputProcessor();
    this.actionProcessor = new ActionProcessor();
    
    // Load pre-trained weights by default and handle result
    this.modelLoadResult = this.neuralNetwork.loadParameters();
}

BaseTrainer.prototype.initialize = function() {
    this.canvas = document.getElementById('boids-canvas');
    this.ctx = this.canvas.getContext('2d');
    
    if (!this.canvas || !this.ctx) {
        return;
    }
    
    // Make canvas responsive
    this.resizeCanvas();
    this.setupResizeHandler();
    
    // Initialize loss chart
    this.initializeLossChart();
    
    this.initializeSimulation();
    this.setupEventListeners();
    this.startAnimationLoop();
    this.updateDisplay();
    
    // Apply initial simulation controls to match HTML defaults
    this.applySimulationControls();
    
    // Initialize neural monitor display state
    this.toggleNeuralMonitorDisplay();
    
    // Show model loading warning if needed
    this.showModelLoadingStatus();
};

BaseTrainer.prototype.initializeLossChart = function() {
    this.lossChart = document.getElementById('loss-chart');
    if (this.lossChart) {
        this.lossChartCtx = this.lossChart.getContext('2d');
        // Set canvas size to match CSS dimensions
        var rect = this.lossChart.getBoundingClientRect();
        this.lossChart.width = rect.width;
        this.lossChart.height = rect.height;
        this.drawLossChart();
    }
};

BaseTrainer.prototype.addLossDataPoint = function(loss) {
    // Only add valid loss values
    if (typeof loss === 'number' && !isNaN(loss) && isFinite(loss)) {
        this.lossHistory.push(loss);
        this.drawLossChart();
    }
};

BaseTrainer.prototype.drawLossChart = function() {
    if (!this.lossChartCtx || !this.lossChart) return;
    
    var ctx = this.lossChartCtx;
    var width = this.lossChart.width;
    var height = this.lossChart.height;
    
    // Don't draw if canvas has no size
    if (width === 0 || height === 0) return;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Draw background
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);
    
    // Draw border
    ctx.strokeStyle = '#e9ecef';
    ctx.lineWidth = 1;
    ctx.strokeRect(0, 0, width, height);
    
    if (this.lossHistory.length === 0) {
        // Draw empty state message
        ctx.fillStyle = '#6c757d';
        ctx.font = '12px Source Code Pro, monospace';
        ctx.textAlign = 'center';
        ctx.fillText('No data yet', width / 2, height / 2);
        return;
    }
    
    if (this.lossHistory.length < 2) return;
    
    // Sample data for display if we have too many points
    var displayData = this.sampleDataForDisplay(this.lossHistory);
    
    // Calculate chart bounds
    var padding = 20;
    var chartWidth = width - 2 * padding;
    var chartHeight = height - 2 * padding;
    
    // Find min/max values
    var minLoss = Math.min.apply(Math, displayData);
    var maxLoss = Math.max.apply(Math, displayData);
    
    // Add some padding to the range
    var range = maxLoss - minLoss;
    if (range === 0) range = 1;
    var paddedMin = minLoss - range * 0.1;
    var paddedMax = maxLoss + range * 0.1;
    
    // Draw grid lines
    ctx.strokeStyle = '#f1f3f5';
    ctx.lineWidth = 1;
    
    // Horizontal grid lines
    for (var i = 0; i <= 4; i++) {
        var y = padding + (i * chartHeight / 4);
        ctx.beginPath();
        ctx.moveTo(padding, y);
        ctx.lineTo(width - padding, y);
        ctx.stroke();
    }
    
    // Vertical grid lines
    for (var i = 0; i <= 4; i++) {
        var x = padding + (i * chartWidth / 4);
        ctx.beginPath();
        ctx.moveTo(x, padding);
        ctx.lineTo(x, height - padding);
        ctx.stroke();
    }
    
    // Draw loss line
    ctx.strokeStyle = '#28a745';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    for (var i = 0; i < displayData.length; i++) {
        var x = padding + (i / (displayData.length - 1)) * chartWidth;
        var normalizedLoss = (displayData[i] - paddedMin) / (paddedMax - paddedMin);
        var y = padding + (1 - normalizedLoss) * chartHeight;
        
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    
    ctx.stroke();
    
    // Draw gradient fill under the line
    ctx.save();
    ctx.globalAlpha = 0.3;
    ctx.fillStyle = '#28a745';
    ctx.beginPath();
    
    // Start from bottom-left of chart area
    ctx.moveTo(padding, padding + chartHeight);
    
    // Draw along the loss line
    for (var i = 0; i < displayData.length; i++) {
        var x = padding + (i / (displayData.length - 1)) * chartWidth;
        var normalizedLoss = (displayData[i] - paddedMin) / (paddedMax - paddedMin);
        var y = padding + (1 - normalizedLoss) * chartHeight;
        ctx.lineTo(x, y);
    }
    
    // Close the path to bottom-right
    ctx.lineTo(padding + chartWidth, padding + chartHeight);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
    
    // Draw data points (only every 10th point to avoid clutter)
    ctx.fillStyle = '#28a745';
    var pointStep = Math.max(1, Math.floor(displayData.length / 20)); // Show max 20 points
    for (var i = 0; i < displayData.length; i += pointStep) {
        var x = padding + (i / (displayData.length - 1)) * chartWidth;
        var normalizedLoss = (displayData[i] - paddedMin) / (paddedMax - paddedMin);
        var y = padding + (1 - normalizedLoss) * chartHeight;
        
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, 2 * Math.PI);
        ctx.fill();
    }
    
    // Always draw the last point
    if (displayData.length > 1) {
        var lastIndex = displayData.length - 1;
        var x = padding + chartWidth;
        var normalizedLoss = (displayData[lastIndex] - paddedMin) / (paddedMax - paddedMin);
        var y = padding + (1 - normalizedLoss) * chartHeight;
        
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, 2 * Math.PI);
        ctx.fill();
    }
    
    // Draw axis labels
    ctx.fillStyle = '#6c757d';
    ctx.font = '10px Source Code Pro, monospace';
    ctx.textAlign = 'right';
    
    // Y-axis labels
    for (var i = 0; i <= 4; i++) {
        var value = paddedMax - (i * (paddedMax - paddedMin) / 4);
        var y = padding + (i * chartHeight / 4);
        ctx.fillText(value.toFixed(3), padding - 5, y + 3);
    }
    
    // X-axis label and data info
    ctx.textAlign = 'center';
    ctx.fillText('Training Progress →', width / 2, height - 5);
    
    // Show data count in top-left corner
    ctx.textAlign = 'left';
    ctx.fillStyle = '#6c757d';
    ctx.font = '9px Source Code Pro, monospace';
    ctx.fillText(this.lossHistory.length + ' data points', padding + 5, padding + 15);
};

BaseTrainer.prototype.sampleDataForDisplay = function(data) {
    if (data.length <= this.maxDisplayPoints) {
        return data.slice(); // Return copy of all data
    }
    
    // Downsample data to maxDisplayPoints while preserving trend
    var sampledData = [];
    var step = data.length / this.maxDisplayPoints;
    
    for (var i = 0; i < this.maxDisplayPoints; i++) {
        var index = Math.floor(i * step);
        sampledData.push(data[index]);
    }
    
    // Always include the last data point
    if (sampledData[sampledData.length - 1] !== data[data.length - 1]) {
        sampledData[sampledData.length - 1] = data[data.length - 1];
    }
    
    return sampledData;
};

BaseTrainer.prototype.resetLossChart = function() {
    this.lossHistory = [];
    this.drawLossChart();
};

BaseTrainer.prototype.exportLossData = function() {
    if (this.lossHistory.length === 0) {
        alert('No loss data to export');
        return;
    }
    
    var csvContent = 'Step,Loss\n';
    for (var i = 0; i < this.lossHistory.length; i++) {
        csvContent += (i + 1) + ',' + this.lossHistory[i] + '\n';
    }
    
    var blob = new Blob([csvContent], { type: 'text/csv' });
    var url = URL.createObjectURL(blob);
    var link = document.createElement('a');
    link.href = url;
    link.download = 'loss_data.csv';
    link.click();
    URL.revokeObjectURL(url);
};

BaseTrainer.prototype.resizeCanvas = function() {
    var simulationPanel = document.getElementById('simulation-panel');
    var newWidth = simulationPanel.clientWidth;
    var newHeight = simulationPanel.clientHeight;
    
    this.canvas.width = newWidth;
    this.canvas.height = newHeight;
    
    if (this.simulation) {
        this.simulation.canvasWidth = newWidth;
        this.simulation.canvasHeight = newHeight;
    }
};

BaseTrainer.prototype.setupResizeHandler = function() {
    var self = this;
    window.addEventListener('resize', function() {
        self.resizeCanvas();
        self.resizeLossChart();
    });
};

BaseTrainer.prototype.resizeLossChart = function() {
    if (this.lossChart) {
        var rect = this.lossChart.getBoundingClientRect();
        this.lossChart.width = rect.width;
        this.lossChart.height = rect.height;
        this.drawLossChart();
    }
};

BaseTrainer.prototype.initializeSimulation = function() {
    this.simulation = new Simulation('boids-canvas');
    this.simulation.ctx = this.ctx;
    this.simulation.canvasWidth = this.canvas.width;
    this.simulation.canvasHeight = this.canvas.height;
    this.simulation.initialize(false); // Don't skip predator
    
    // Replace the pre-trained predator with our training predator
    if (this.simulation.predator) {
        this.simulation.predator.neuralNetwork = this.neuralNetwork;
    }
    
    // Note: Simulation controls will be applied when user clicks Apply button
};

BaseTrainer.prototype.setupEventListeners = function() {
    var self = this;
    
    document.getElementById('start-training').addEventListener('click', function() {
        self.startTraining();
    });
    
    document.getElementById('stop-training').addEventListener('click', function() {
        self.stopTraining();
    });
    
    document.getElementById('reset-training').addEventListener('click', function() {
        self.resetNetwork();
    });
    
    document.getElementById('load-network').addEventListener('click', function() {
        self.loadNetwork();
    });
    
    document.getElementById('export-network').addEventListener('click', function() {
        self.exportNetwork();
    });
    
    document.getElementById('restart-simulation').addEventListener('click', function() {
        self.initializeSimulation();
    });
    
    // Simulation controls - apply button
    document.getElementById('apply-simulation').addEventListener('click', function() {
        self.applySimulationControls();
    });
    
    // Display toggle checkbox
    var displayToggle = document.getElementById('display-toggle');
    if (displayToggle) {
        displayToggle.addEventListener('change', function() {
            self.displayEnabled = displayToggle.checked;
        });
    }
    
    // Vision range toggle checkbox
    var visionRangeToggle = document.getElementById('vision-range-toggle');
    if (visionRangeToggle) {
        visionRangeToggle.addEventListener('change', function() {
            self.showVisionRange = visionRangeToggle.checked;
        });
    }
    
    // Neural network monitor controls
    var neuralMonitorToggle = document.getElementById('neural-monitor-toggle');
    if (neuralMonitorToggle) {
        neuralMonitorToggle.addEventListener('change', function() {
            self.neuralMonitorEnabled = neuralMonitorToggle.checked;
            self.toggleNeuralMonitorDisplay();
        });
    }
    
    var neuralUpdateInterval = document.getElementById('neural-update-interval');
    if (neuralUpdateInterval) {
        neuralUpdateInterval.addEventListener('input', function() {
            self.neuralMonitorUpdateInterval = parseInt(neuralUpdateInterval.value);
            self.neuralMonitorFrameCount = 0; // Reset counter
        });
    }
    
    // Export loss data button
    var exportLossButton = document.getElementById('export-loss-data');
    if (exportLossButton) {
        exportLossButton.addEventListener('click', function() {
            self.exportLossData();
        });
    }
};

BaseTrainer.prototype.startTraining = function() {
    this.isTraining = true;
    document.getElementById('start-training').disabled = true;
    document.getElementById('stop-training').disabled = false;
    document.getElementById('status-text').textContent = 'Training';
};

BaseTrainer.prototype.stopTraining = function() {
    this.isTraining = false;
    document.getElementById('start-training').disabled = false;
    document.getElementById('stop-training').disabled = true;
    document.getElementById('status-text').textContent = 'Stopped';
};

BaseTrainer.prototype.resetNetwork = function() {
    // Randomize all neural network parameters
    this.neuralNetwork.reset();
    
    // Update model loading status (will show that we're using random initialization)
    this.modelLoadResult = { 
        success: false, 
        message: "Using random initialization", 
        fallbackReason: "Network was manually reset" 
    };
    this.showModelLoadingStatus();
    
    this.onNetworkReset(); // Allow subclasses to handle specific reset logic
    this.updateDisplay();
};

BaseTrainer.prototype.loadNetwork = function() {
    // Load pre-trained weights from model.js and handle result
    this.modelLoadResult = this.neuralNetwork.loadParameters();
    this.showModelLoadingStatus(); // Show warning if loading failed
    this.onNetworkLoad(); // Allow subclasses to handle specific load logic
    this.updateDisplay();
};

BaseTrainer.prototype.exportNetwork = function() {
    // Generate model.js content with current weights
    var content = this.generateModelJSContent();
    
    // Create a popup window to display the content
    var popup = window.open('', 'Export Network', 'width=800,height=600,scrollbars=yes');
    popup.document.write('<html><head><title>Export Network - model.js</title></head><body>');
    popup.document.write('<h2>Copy this content to model.js:</h2>');
    popup.document.write('<textarea style="width:100%;height:500px;font-family:monospace;font-size:12px;">' + content + '</textarea>');
    popup.document.write('<br><button onclick="navigator.clipboard.writeText(document.querySelector(\'textarea\').value).then(function(){alert(\'Copied to clipboard!\');})">Copy to Clipboard</button>');
    popup.document.write('</body></html>');
    popup.document.close();
};

BaseTrainer.prototype.generateModelJSContent = function() {
    var nn = this.neuralNetwork;
    var content = '';
    
    content += '/**\n';
    content += ' * Neural Network Model - Pre-trained weights and architecture\n';
    content += ' * \n';
    content += ' * Network Architecture: ' + nn.inputSize + ' inputs → ' + nn.hidden1Size + ' hidden1 → ' + nn.hidden2Size + ' hidden2 → ' + nn.outputSize + ' outputs\n';
    content += ' */\n\n';
    content += 'window.NEURAL_PARAMS = {\n';
    content += '    // Network architecture\n';
    content += '    inputSize: ' + nn.inputSize + ',\n';
    content += '    hidden1Size: ' + nn.hidden1Size + ',\n';
    content += '    hidden2Size: ' + nn.hidden2Size + ',\n';
    content += '    outputSize: ' + nn.outputSize + ',\n';
    content += '    \n';
    content += '    // Input to Hidden1 layer weights (' + nn.hidden1Size + 'x' + nn.inputSize + ' matrix)\n';
    content += '    weightsIH1: [\n';
    
    for (var h1 = 0; h1 < nn.hidden1Size; h1++) {
        content += '        // Hidden1 neuron ' + (h1 + 1) + '\n';
        content += '        [';
        for (var i = 0; i < nn.inputSize; i++) {
            content += nn.weightsIH1[h1][i].toFixed(3);
            if (i < nn.inputSize - 1) content += ', ';
        }
        content += ']';
        if (h1 < nn.hidden1Size - 1) content += ',';
        content += '\n';
    }
    
    content += '    ],\n';
    content += '    \n';
    content += '    // Hidden1 to Hidden2 layer weights (' + nn.hidden2Size + 'x' + nn.hidden1Size + ' matrix)\n';
    content += '    weightsH1H2: [\n';
    
    for (var h2 = 0; h2 < nn.hidden2Size; h2++) {
        content += '        // Hidden2 neuron ' + (h2 + 1) + '\n';
        content += '        [';
        for (var h1 = 0; h1 < nn.hidden1Size; h1++) {
            content += nn.weightsH1H2[h2][h1].toFixed(3);
            if (h1 < nn.hidden1Size - 1) content += ', ';
        }
        content += ']';
        if (h2 < nn.hidden2Size - 1) content += ',';
        content += '\n';
    }
    
    content += '    ],\n';
    content += '    \n';
    content += '    // Hidden2 to Output layer weights (' + nn.outputSize + 'x' + nn.hidden2Size + ' matrix)\n';
    content += '    weightsH2O: [\n';
    
    var outputLabels = ['X steering force output', 'Y steering force output'];
    for (var o = 0; o < nn.outputSize; o++) {
        content += '        // ' + outputLabels[o] + '\n';
        content += '        [';
        for (var h2 = 0; h2 < nn.hidden2Size; h2++) {
            content += nn.weightsH2O[o][h2].toFixed(3);
            if (h2 < nn.hidden2Size - 1) content += ', ';
        }
        content += ']';
        if (o < nn.outputSize - 1) content += ',';
        content += '\n';
    }
    
    content += '    ],\n';
    content += '    \n';
    content += '    // Hidden1 layer biases (' + nn.hidden1Size + ' values)\n';
    content += '    biasH1: [';
    for (var h1 = 0; h1 < nn.hidden1Size; h1++) {
        content += nn.biasH1[h1].toFixed(3);
        if (h1 < nn.hidden1Size - 1) content += ', ';
    }
    content += '],\n';
    content += '    \n';
    content += '    // Hidden2 layer biases (' + nn.hidden2Size + ' values)\n';
    content += '    biasH2: [';
    for (var h2 = 0; h2 < nn.hidden2Size; h2++) {
        content += nn.biasH2[h2].toFixed(3);
        if (h2 < nn.hidden2Size - 1) content += ', ';
    }
    content += '],\n';
    content += '    \n';
    content += '    // Output layer biases (' + nn.outputSize + ' values)\n';
    content += '    biasO: [';
    for (var o = 0; o < nn.outputSize; o++) {
        content += nn.biasO[o].toFixed(3);
        if (o < nn.outputSize - 1) content += ', ';
    }
    content += ']\n';
    content += '}; \n';
    
    return content;
};

BaseTrainer.prototype.startAnimationLoop = function() {
    var self = this;
    
    function loop() {
        self.update();
        requestAnimationFrame(loop);
    }
    
    loop();
};

BaseTrainer.prototype.update = function() {
    if (!this.simulation || !this.simulation.predator) return;
    
    // Apply speed multiplier by running multiple simulation steps
    var steps = Math.max(1, Math.floor(this.speedMultiplier));
    var partialStep = this.speedMultiplier - steps;
    
    for (var step = 0; step < steps; step++) {
        this.updateTrainingStep(); // Subclass-specific training logic
        this.updateSimulationStep();
        this.frameCount++;
    }
    
    // Handle partial step for smooth speed control
    if (partialStep > 0 && Math.random() < partialStep) {
        this.updateTrainingStep();
        this.updateSimulationStep();
    }
    
    // Only render if display is enabled
    if (this.displayEnabled) {
        this.renderSimulation();
    }
};

BaseTrainer.prototype.updateSimulationStep = function() {
    // Update boids
    for (var i = 0; i < this.simulation.boids.length; i++) {
        this.simulation.boids[i].flock(this.simulation.boids);
        this.simulation.boids[i].update();
    }
    
    // Update predator
    if (this.simulation.predator) {
        this.simulation.predator.update(this.simulation.boids);
        
        // Capture neural network data for monitoring
        this.captureNeuralNetworkData();
        
        // Check for caught boids
        var caughtBoids = this.simulation.predator.checkForPrey(this.simulation.boids);
        for (var i = caughtBoids.length - 1; i >= 0; i--) {
            this.simulation.boids.splice(caughtBoids[i], 1);
        }
        
        // Check if we need to auto-restart simulation
        if (this.autoRestartThreshold > -1 && this.simulation.boids.length <= this.autoRestartThreshold) {
            this.initializeSimulation();
        }
    }
};

BaseTrainer.prototype.captureNeuralNetworkData = function() {
    if (!this.neuralMonitorEnabled || !this.simulation.predator || !this.simulation.predator.neuralNetwork) return;
    
    // Only update at specified intervals
    this.neuralMonitorFrameCount++;
    if (this.neuralMonitorFrameCount < this.neuralMonitorUpdateInterval) {
        return;
    }
    this.neuralMonitorFrameCount = 0;
    
    var nn = this.simulation.predator.neuralNetwork;
    
    // Store the current neural network state
    this.neuralNetworkData = {
        inputs: nn.lastInputs ? nn.lastInputs.slice() : null,
        hidden1: nn.lastHidden1 ? nn.lastHidden1.slice() : null,
        hidden2: nn.lastHidden2 ? nn.lastHidden2.slice() : null,
        outputs: nn.lastOutput ? nn.lastOutput.slice() : null
    };
    
    // Update the display
    this.updateNeuralNetworkDisplay();
};

BaseTrainer.prototype.updateNeuralNetworkDisplay = function() {
    if (!this.neuralNetworkData) return;
    
    var data = this.neuralNetworkData;
    
    // Update inputs display
    if (data.inputs) {
        this.updateInputsDisplay(data.inputs);
    }
    
    // Update hidden layers display
    if (data.hidden1) {
        this.updateHidden1Display(data.hidden1);
    }
    if (data.hidden2) {
        this.updateHidden2Display(data.hidden2);
    }
    
    // Update outputs display
    if (data.outputs) {
        this.updateOutputsDisplay(data.outputs);
    }
};

BaseTrainer.prototype.toggleNeuralMonitorDisplay = function() {
    var neuralMonitorSection = document.getElementById('neural-monitor-section');
    if (neuralMonitorSection) {
        neuralMonitorSection.style.display = this.neuralMonitorEnabled ? 'block' : 'none';
    }
};

BaseTrainer.prototype.updateInputsDisplay = function(inputs) {
    // Show first 5 boids for compatibility with existing HTML
    for (var i = 0; i < 5; i++) {
        var baseIndex = i * 4;
        var isVisible = Math.abs(inputs[baseIndex]) > 0.001 || Math.abs(inputs[baseIndex + 1]) > 0.001;
        
        // Update table cells
        var posXCell = document.getElementById('boid-pos-x-' + i);
        var posYCell = document.getElementById('boid-pos-y-' + i);
        var velXCell = document.getElementById('boid-vel-x-' + i);
        var velYCell = document.getElementById('boid-vel-y-' + i);
        
        if (posXCell) posXCell.textContent = inputs[baseIndex].toFixed(3);
        if (posYCell) posYCell.textContent = inputs[baseIndex + 1].toFixed(3);
        if (velXCell) velXCell.textContent = inputs[baseIndex + 2].toFixed(3);
        if (velYCell) velYCell.textContent = inputs[baseIndex + 3].toFixed(3);
        
        // Style visibility
        var row = document.getElementById('boid-row-' + i);
        if (row) {
            row.style.opacity = isVisible ? '1.0' : '0.5';
        }
    }
    
    // Update predator data - new format: [canvas_width_norm, canvas_height_norm, vel_x, vel_y]
    var predVelXCell = document.getElementById('pred-vel-x');
    var predVelYCell = document.getElementById('pred-vel-y');
    var predPosXCell = document.getElementById('pred-pos-x');  // Repurpose for canvas width
    var predPosYCell = document.getElementById('pred-pos-y');  // Repurpose for canvas height
    
    if (predVelXCell) predVelXCell.textContent = inputs[202].toFixed(3);  // Predator vel_x (position 2)
    if (predVelYCell) predVelYCell.textContent = inputs[203].toFixed(3);  // Predator vel_y (position 3)
    if (predPosXCell) predPosXCell.textContent = inputs[200].toFixed(3);  // Canvas width norm (position 0)
    if (predPosYCell) predPosYCell.textContent = inputs[201].toFixed(3);  // Canvas height norm (position 1)
};

BaseTrainer.prototype.calculateLayerStatistics = function(layerValues) {
    var sum = 0;
    var max = -Infinity;
    var min = Infinity;
    var activeCount = 0;
    var positiveCount = 0;
    
    for (var i = 0; i < layerValues.length; i++) {
        var value = layerValues[i];
        sum += value;
        max = Math.max(max, value);
        min = Math.min(min, value);
        
        if (Math.abs(value) > 0.1) { // Consider > 0.1 as "active"
            activeCount++;
        }
        if (value > 0) {
            positiveCount++;
        }
    }
    
    var avg = sum / layerValues.length;
    var positivePct = (positiveCount / layerValues.length) * 100;
    
    // Calculate standard deviation
    var sumSquaredDiffs = 0;
    for (var i = 0; i < layerValues.length; i++) {
        var diff = layerValues[i] - avg;
        sumSquaredDiffs += diff * diff;
    }
    var std = Math.sqrt(sumSquaredDiffs / layerValues.length);
    
    return {
        avg: avg,
        max: max,
        min: min,
        std: std,
        activeCount: activeCount,
        positivePct: positivePct
    };
};

BaseTrainer.prototype.updateHidden1Display = function(hidden1) {
    var stats = this.calculateLayerStatistics(hidden1);
    
    var avgCell = document.getElementById('hidden1-avg');
    var maxCell = document.getElementById('hidden1-max');
    var minCell = document.getElementById('hidden1-min');
    var stdCell = document.getElementById('hidden1-std');
    var activeCell = document.getElementById('hidden1-active');
    var posPctCell = document.getElementById('hidden1-pos-pct');
    
    if (avgCell) avgCell.textContent = stats.avg.toFixed(3);
    if (maxCell) maxCell.textContent = stats.max.toFixed(3);
    if (minCell) minCell.textContent = stats.min.toFixed(3);
    if (stdCell) stdCell.textContent = stats.std.toFixed(3);
    if (activeCell) activeCell.textContent = stats.activeCount;
    if (posPctCell) posPctCell.textContent = Math.round(stats.positivePct) + '%';
};

BaseTrainer.prototype.updateHidden2Display = function(hidden2) {
    var stats = this.calculateLayerStatistics(hidden2);
    
    var avgCell = document.getElementById('hidden2-avg');
    var maxCell = document.getElementById('hidden2-max');
    var minCell = document.getElementById('hidden2-min');
    var stdCell = document.getElementById('hidden2-std');
    var activeCell = document.getElementById('hidden2-active');
    var posPctCell = document.getElementById('hidden2-pos-pct');
    
    if (avgCell) avgCell.textContent = stats.avg.toFixed(3);
    if (maxCell) maxCell.textContent = stats.max.toFixed(3);
    if (minCell) minCell.textContent = stats.min.toFixed(3);
    if (stdCell) stdCell.textContent = stats.std.toFixed(3);
    if (activeCell) activeCell.textContent = stats.activeCount;
    if (posPctCell) posPctCell.textContent = Math.round(stats.positivePct) + '%';
};

BaseTrainer.prototype.updateOutputsDisplay = function(outputs) {
    var forceXCell = document.getElementById('output-force-x');
    var forceYCell = document.getElementById('output-force-y');
    var magnitudeCell = document.getElementById('output-magnitude');
    
    if (forceXCell) forceXCell.textContent = outputs[0].toFixed(3);
    if (forceYCell) forceYCell.textContent = outputs[1].toFixed(3);
    if (magnitudeCell) {
        var magnitude = Math.sqrt(outputs[0] * outputs[0] + outputs[1] * outputs[1]);
        magnitudeCell.textContent = magnitude.toFixed(3);
    }
};

BaseTrainer.prototype.renderSimulation = function() {
    this.simulation.ctx.clearRect(0, 0, this.simulation.canvasWidth, this.simulation.canvasHeight);
    
    // Note: Vision range drawing removed since new system doesn't use vision limitations
    
    // Draw boids
    for (var i = 0; i < this.simulation.boids.length; i++) {
        this.simulation.boids[i].render();
    }
    
    // Draw predator last (on top)
    if (this.simulation.predator) {
        this.simulation.predator.render();
    }
};

BaseTrainer.prototype.updateBoidCount = function(count) {
    if (!this.simulation) return;
    
    var currentCount = this.simulation.boids.length;
    var difference = count - currentCount;
    
    if (difference > 0) {
        // Add boids
        for (var i = 0; i < difference; i++) {
            var x = Math.random() * this.simulation.canvasWidth;
            var y = Math.random() * this.simulation.canvasHeight;
            var boid = new Boid(x, y, this.simulation);
            this.simulation.boids.push(boid);
        }
    } else if (difference < 0) {
        // Remove boids
        this.simulation.boids.splice(0, -difference);
    }
};

BaseTrainer.prototype.updateSimSpeed = function(speed) {
    this.speedMultiplier = speed;
};

BaseTrainer.prototype.applySimulationControls = function() {
    // Get values from HTML inputs
    var boidCountInput = document.getElementById('boid-count');
    var speedInput = document.getElementById('sim-speed');
    var autoRestartInput = document.getElementById('auto-restart-threshold');
    
    if (boidCountInput) {
        var boidCount = parseInt(boidCountInput.value);
        this.updateBoidCount(boidCount);
    }
    
    if (speedInput) {
        var speed = parseFloat(speedInput.value);
        this.updateSimSpeed(speed);
    }
    
    if (autoRestartInput) {
        var threshold = parseInt(autoRestartInput.value);
        this.autoRestartThreshold = threshold;
    }
};

BaseTrainer.prototype.showModelLoadingStatus = function() {
    var warningElement = document.getElementById('model-warning');
    
    if (this.modelLoadResult && !this.modelLoadResult.success) {
        if (warningElement) {
            warningElement.style.display = 'block';
            warningElement.innerHTML = '<strong>⚠️ Model Loading Warning:</strong> ' + 
                this.modelLoadResult.fallbackReason + 
                '<br>Using randomly initialized weights. Train the network to improve performance.';
        } else {
            // Create warning element if it doesn't exist
            this.createModelWarningElement();
        }
    } else if (warningElement) {
        warningElement.style.display = 'none';
    }
};

BaseTrainer.prototype.createModelWarningElement = function() {
    var warningHtml = '<div id="model-warning" style="' +
        'background: rgba(255, 193, 7, 0.1); ' +
        'border: 1px solid #ffc107; ' +
        'border-radius: 4px; ' +
        'padding: 12px; ' +
        'margin: 15px 0; ' +
        'color: #856404; ' +
        'font-size: 11px; ' +
        'line-height: 1.4;' +
        '"><strong>⚠️ Model Loading Warning:</strong> ' + 
        this.modelLoadResult.fallbackReason + 
        '<br>Using randomly initialized weights. Train the network to improve performance.</div>';
        
    // Insert after the status indicator
    var controlHeader = document.querySelector('.control-header');
    if (controlHeader) {
        controlHeader.insertAdjacentHTML('afterend', warningHtml);
    }
};

// Abstract methods that subclasses must implement
BaseTrainer.prototype.updateTrainingStep = function() {
    // Override in subclasses
};

BaseTrainer.prototype.updateDisplay = function() {
    // Override in subclasses
};

BaseTrainer.prototype.onNetworkReset = function() {
    // Override in subclasses if needed
};

BaseTrainer.prototype.onNetworkLoad = function() {
    // Override in subclasses if needed
}; 