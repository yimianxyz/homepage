function isCanvasSupported() {
	var elem = document.createElement('canvas');
	return ! !(elem.getContext && elem.getContext('2d'));
}

function checkForCanvasSupport() {
	if (!isCanvasSupported()) {
		$('div#container').hide();

		var canvasNotice$ = jQuery('<div id="canvas_notice">Please update your browser to view this experiment.</div>');
		canvasNotice$.insertAfter($('div#header_wrapper'));

		return false;
	} else {
		return true;
	}
}
function getWidth() {
	return Math.max(
		document.documentElement.scrollWidth,
		document.documentElement.offsetWidth,
		document.documentElement.clientWidth,
		window.innerWidth
	);
}

function getHeight() {
	return Math.max(
		document.documentElement.scrollHeight,
		document.documentElement.offsetHeight,
		document.documentElement.clientHeight,
		window.innerHeight
	);
}


function resizeCanvas(width, height) {
	var canvases = document.querySelectorAll('canvas');
	var newWidth = getWidth();
	var newHeight = getHeight();
	
	for (var i = 0; i < canvases.length; i++) {
		canvases[i].width = newWidth;
		canvases[i].height = newHeight;
	}
	
	// Also notify the neural visualization of the resize
	if (typeof neuralViz !== 'undefined' && neuralViz && neuralViz.resize) {
		neuralViz.resize();
	}
}; 