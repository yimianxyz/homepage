// Neural Network Visualization System
// Real-time visualization of the predator's neural network activations
// 
// This system creates a beautiful background visualization that shows:
// - Input layer: Boid positions and predator velocity (12 nodes)
// - Hidden layer: Processing neurons that detect patterns (8 nodes)  
// - Output layer: Steering commands for hunting behavior (2 nodes)
// - Connections: Weighted synapses that pulse with neural activity
// - Learning stats: Real-time display of adaptation progress
//
// The visualization uses a subtle, translucent overlay that doesn't interfere
// with the main UI while providing fascinating insights into AI decision-making.

class NeuralVisualization {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) {
            console.warn('Neural visualization canvas not found');
            return;
        }
        this.ctx = this.canvas.getContext('2d');
        this.predator = null;
        
        // Network architecture
        this.inputCount = 12;
        this.hiddenCount = 8;
        this.outputCount = 2;
        
        // Animation state
        this.opacity = 0;
        this.targetOpacity = 0.15;
        this.fadeSpeed = 0.02;
        
        // Visual layout (responsive)
        this.isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) || window.innerWidth <= 768;
        this.nodeRadius = this.isMobile ? 6 : 8;
        this.layerSpacing = this.isMobile ? 150 : 200;
        this.nodeSpacing = this.isMobile ? 30 : 40;
        
        // Colors
        this.nodeColor = 'rgba(200, 200, 200, 0.8)';
        this.activeNodeColor = 'rgba(100, 150, 255, 0.9)';
        this.connectionColor = 'rgba(150, 150, 150, 0.3)';
        this.activeConnectionColor = 'rgba(100, 150, 255, 0.5)';
        
        // Animation
        this.lastActivations = {
            input: new Array(this.inputCount).fill(0),
            hidden: new Array(this.hiddenCount).fill(0),
            output: new Array(this.outputCount).fill(0)
        };
        
        this.smoothedActivations = {
            input: new Array(this.inputCount).fill(0),
            hidden: new Array(this.hiddenCount).fill(0),
            output: new Array(this.outputCount).fill(0)
        };
        
        // Layout calculations
        this.calculateLayout();
        
        // Start animation
        this.animate();
    }
    
    calculateLayout() {
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;
        
        // Scale layer spacing based on canvas size
        const scaledLayerSpacing = Math.min(this.layerSpacing, this.canvas.width / 4);
        
        // Position layers
        this.layers = {
            input: {
                x: centerX - scaledLayerSpacing,
                y: centerY,
                count: this.inputCount,
                nodes: []
            },
            hidden: {
                x: centerX,
                y: centerY,
                count: this.hiddenCount,
                nodes: []
            },
            output: {
                x: centerX + scaledLayerSpacing,
                y: centerY,
                count: this.outputCount,
                nodes: []
            }
        };
        
        // Calculate node positions for each layer
        for (const [layerName, layer] of Object.entries(this.layers)) {
            // Clear existing nodes
            layer.nodes = [];
            
            const totalHeight = (layer.count - 1) * this.nodeSpacing;
            const startY = layer.y - totalHeight / 2;
            
            for (let i = 0; i < layer.count; i++) {
                layer.nodes.push({
                    x: layer.x,
                    y: startY + i * this.nodeSpacing,
                    activation: 0,
                    targetActivation: 0
                });
            }
        }
    }
    
    setPredator(predator) {
        this.predator = predator;
        // Start fade-in animation when predator is connected
        this.targetOpacity = 0.15;
    }
    
    updateActivations() {
        if (!this.predator) return;
        
        // Get current activations from predator
        const currentActivations = {
            input: this.predator.lastInput || new Array(this.inputCount).fill(0),
            hidden: this.predator.hiddenActivations || new Array(this.hiddenCount).fill(0),
            output: this.predator.lastOutput || new Array(this.outputCount).fill(0)
        };
        
        // Smooth activations for visual appeal
        const smoothing = 0.1;
        
        for (let i = 0; i < this.inputCount; i++) {
            this.smoothedActivations.input[i] = 
                this.smoothedActivations.input[i] * (1 - smoothing) + 
                Math.abs(currentActivations.input[i] || 0) * smoothing;
        }
        
        for (let i = 0; i < this.hiddenCount; i++) {
            this.smoothedActivations.hidden[i] = 
                this.smoothedActivations.hidden[i] * (1 - smoothing) + 
                Math.abs(currentActivations.hidden[i] || 0) * smoothing;
        }
        
        for (let i = 0; i < this.outputCount; i++) {
            this.smoothedActivations.output[i] = 
                this.smoothedActivations.output[i] * (1 - smoothing) + 
                Math.abs(currentActivations.output[i] || 0) * smoothing;
        }
        
        // Update node target activations
        this.layers.input.nodes.forEach((node, i) => {
            node.targetActivation = this.smoothedActivations.input[i];
        });
        
        this.layers.hidden.nodes.forEach((node, i) => {
            node.targetActivation = this.smoothedActivations.hidden[i];
        });
        
        this.layers.output.nodes.forEach((node, i) => {
            node.targetActivation = this.smoothedActivations.output[i];
        });
    }
    
    drawConnections() {
        if (!this.predator) return;
        
        // Draw input to hidden connections
        for (let i = 0; i < this.inputCount; i++) {
            for (let j = 0; j < this.hiddenCount; j++) {
                const inputNode = this.layers.input.nodes[i];
                const hiddenNode = this.layers.hidden.nodes[j];
                
                const weight = this.predator.weightsIH[j][i] || 0;
                const connectionStrength = Math.abs(weight) * 0.5;
                const activation = Math.min(inputNode.activation * hiddenNode.activation, 1);
                
                this.ctx.strokeStyle = `rgba(150, 150, 150, ${Math.max(0.1, connectionStrength * 0.5 + activation * 0.3)})`;
                this.ctx.lineWidth = Math.max(0.5, connectionStrength * 2);
                
                this.ctx.beginPath();
                this.ctx.moveTo(inputNode.x, inputNode.y);
                this.ctx.lineTo(hiddenNode.x, hiddenNode.y);
                this.ctx.stroke();
            }
        }
        
        // Draw hidden to output connections
        for (let i = 0; i < this.hiddenCount; i++) {
            for (let j = 0; j < this.outputCount; j++) {
                const hiddenNode = this.layers.hidden.nodes[i];
                const outputNode = this.layers.output.nodes[j];
                
                const weight = this.predator.weightsHO[j][i] || 0;
                const connectionStrength = Math.abs(weight) * 0.5;
                const activation = Math.min(hiddenNode.activation * outputNode.activation, 1);
                
                this.ctx.strokeStyle = `rgba(150, 150, 255, ${Math.max(0.1, connectionStrength * 0.5 + activation * 0.5)})`;
                this.ctx.lineWidth = Math.max(0.5, connectionStrength * 2);
                
                this.ctx.beginPath();
                this.ctx.moveTo(hiddenNode.x, hiddenNode.y);
                this.ctx.lineTo(outputNode.x, outputNode.y);
                this.ctx.stroke();
            }
        }
    }
    
    drawNodes() {
        // Draw all layers
        for (const [layerName, layer] of Object.entries(this.layers)) {
            layer.nodes.forEach((node, i) => {
                // Smooth node activation for animation
                const smoothing = 0.15;
                node.activation = node.activation * (1 - smoothing) + node.targetActivation * smoothing;
                
                // Node color based on activation
                const activation = Math.min(node.activation, 1);
                const nodeAlpha = 0.3 + activation * 0.7;
                
                // Different colors for different layers
                let color;
                if (layerName === 'input') {
                    color = `rgba(255, 200, 100, ${nodeAlpha})`;
                } else if (layerName === 'hidden') {
                    color = `rgba(100, 255, 150, ${nodeAlpha})`;
                } else {
                    color = `rgba(255, 100, 150, ${nodeAlpha})`;
                }
                
                // Draw node
                this.ctx.fillStyle = color;
                this.ctx.beginPath();
                this.ctx.arc(node.x, node.y, this.nodeRadius * (0.8 + activation * 0.4), 0, 2 * Math.PI);
                this.ctx.fill();
                
                // Draw subtle border
                this.ctx.strokeStyle = `rgba(255, 255, 255, ${nodeAlpha * 0.5})`;
                this.ctx.lineWidth = 1;
                this.ctx.stroke();
            });
        }
    }
    
    drawLabels() {
        // Main title
        this.ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
        this.ctx.font = this.isMobile ? '10px monospace' : '12px monospace';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('NEURAL PREDATOR BRAIN', this.canvas.width / 2, 30);
        
        // Layer labels
        this.ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
        this.ctx.font = this.isMobile ? '10px monospace' : '12px monospace';
        
        const labelOffset = this.isMobile ? 50 : 60;
        this.ctx.fillText('INPUT', this.layers.input.x, this.layers.input.y + labelOffset);
        this.ctx.fillText('HIDDEN', this.layers.hidden.x, this.layers.hidden.y + labelOffset);
        this.ctx.fillText('OUTPUT', this.layers.output.x, this.layers.output.y + labelOffset);
        
        // Show neural network statistics
        if (this.predator && this.predator.getLearningStats) {
            const stats = this.predator.getLearningStats();
            this.ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
            this.ctx.font = this.isMobile ? '8px monospace' : '10px monospace';
            this.ctx.textAlign = 'left';
            
            const statsY = this.canvas.height - 80;
            this.ctx.fillText(`Learning: ${(stats.learningIntensity * 100).toFixed(1)}%`, 20, statsY);
            this.ctx.fillText(`Reward: ${stats.avgReward.toFixed(2)}`, 20, statsY + 15);
            this.ctx.fillText(`Size: ${stats.currentSize.toFixed(1)}px`, 20, statsY + 30);
        }
        
        // Input descriptions (very subtle, only on desktop)
        if (!this.isMobile) {
            this.ctx.font = '8px monospace';
            this.ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
            this.ctx.textAlign = 'left';
            
            const inputLabels = [
                'Boid 1 X', 'Boid 1 Y', 'Boid 2 X', 'Boid 2 Y',
                'Boid 3 X', 'Boid 3 Y', 'Boid 4 X', 'Boid 4 Y',
                'Boid 5 X', 'Boid 5 Y', 'Pred Vx', 'Pred Vy'
            ];
            
            this.layers.input.nodes.forEach((node, i) => {
                if (i < inputLabels.length) {
                    this.ctx.fillText(inputLabels[i], node.x - 80, node.y + 3);
                }
            });
            
            // Output descriptions
            this.ctx.textAlign = 'right';
            const outputLabels = ['Steer X', 'Steer Y'];
            this.layers.output.nodes.forEach((node, i) => {
                if (i < outputLabels.length) {
                    this.ctx.fillText(outputLabels[i], node.x + 80, node.y + 3);
                }
            });
        }
    }
    
    render() {
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Update fade-in animation
        if (this.predator && this.opacity < this.targetOpacity) {
            this.opacity = Math.min(this.targetOpacity, this.opacity + this.fadeSpeed);
        } else if (!this.predator && this.opacity > 0) {
            this.opacity = Math.max(0, this.opacity - this.fadeSpeed);
        }
        
        // Skip rendering if not visible
        if (this.opacity <= 0) return;
        
        // Save context and apply global opacity
        this.ctx.save();
        this.ctx.globalAlpha = this.opacity / this.targetOpacity;
        
        // Update activations from predator
        this.updateActivations();
        
        // Draw neural network
        this.drawConnections();
        this.drawNodes();
        this.drawLabels();
        
        // Restore context
        this.ctx.restore();
    }
    
    animate() {
        if (!this.canvas) return;
        this.render();
        requestAnimationFrame(() => this.animate());
    }
    
    // Handle canvas resize
    resize() {
        this.calculateLayout();
    }
}

// Global neural visualization instance
let neuralViz = null;

// Initialize neural visualization when page loads
document.addEventListener('DOMContentLoaded', () => {
    try {
        neuralViz = new NeuralVisualization('neural-viz');
        if (!neuralViz.canvas) {
            console.warn('Neural visualization failed to initialize - canvas not found');
            neuralViz = null;
        }
    } catch (error) {
        console.error('Neural visualization initialization error:', error);
        neuralViz = null;
    }
});

// Connect to predator when simulation starts
function connectNeuralViz(predator) {
    if (neuralViz && neuralViz.setPredator) {
        neuralViz.setPredator(predator);
        console.log('Neural network visualization connected to predator');
    } else {
        console.warn('Neural visualization not available or not initialized');
    }
}

// Handle resize
window.addEventListener('resize', () => {
    if (neuralViz && neuralViz.resize) {
        neuralViz.resize();
    }
}); 