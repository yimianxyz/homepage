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
// Visible viewport dimensions. On iOS Safari the layout viewport (100vh,
// innerHeight) includes the area covered by the URL bar and bottom toolbar,
// so content drawn at the bottom of the canvas is composited behind chrome.
// window.visualViewport returns the actually-visible region, which is what
// we want for sizing the canvas. Fall back to innerWidth/innerHeight on
// browsers without visualViewport support.
function getWidth() {
	if (window.visualViewport && window.visualViewport.width > 0) {
		return Math.round(window.visualViewport.width);
	}
	return window.innerWidth || document.documentElement.clientWidth;
}

function getHeight() {
	if (window.visualViewport && window.visualViewport.height > 0) {
		return Math.round(window.visualViewport.height);
	}
	return window.innerHeight || document.documentElement.clientHeight;
}


function resizeCanvas(width, height) {
	var canvas = document.querySelector('canvas');
	if (canvas) {
		canvas.width = getWidth();
		canvas.height = getHeight();
	}
};