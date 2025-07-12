/**
 * Homepage Initialization - Neural predator simulation
 */

document.addEventListener('DOMContentLoaded', function() {
    if (!isCanvasSupported()) {
        return;
    }
    
    // Set canvas to screen size
    resizeCanvas();
    
    var simulation = new Simulation('boids1');
    simulation.ctx = document.getElementById('boids1').getContext('2d');
    simulation.canvasWidth = window.innerWidth;
    simulation.canvasHeight = window.innerHeight;
    
    simulation.initialize(false);
    
    function animate() {
        simulation.render();
        requestAnimationFrame(animate);
    }
    
    animate();
    
    window.addEventListener('resize', function() {
        resizeCanvas();
        if (simulation) {
            simulation.canvasWidth = window.innerWidth;
            simulation.canvasHeight = window.innerHeight;
        }
    });
}); 