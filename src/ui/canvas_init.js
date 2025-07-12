function isCanvasSupported() {
	var elem = document.createElement('canvas');
	return !!(elem.getContext && elem.getContext('2d'));
}

function resizeCanvas() {
	var canvases = document.querySelectorAll('canvas');
	var newWidth = window.innerWidth;
	var newHeight = window.innerHeight;
	
	for (var i = 0; i < canvases.length; i++) {
		canvases[i].width = newWidth;
		canvases[i].height = newHeight;
	}
}; 