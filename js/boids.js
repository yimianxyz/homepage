document.addEventListener('DOMContentLoaded', function() {
	if (!isCanvasSupported()) {
		return;
	}

	resizeCanvas();

	var initialize_canvas_simulation = function (name, use_obstacle) {
		var simulation = new Simulation(name);
		simulation.initialize(use_obstacle);
		simulation.run();
		return simulation;
	};

	// The predator's policy is a trained neural network; wait for its
	// weights to load before starting the simulation, so the NN drives
	// the predator from frame 1.
	window.__predatorReady.then(function () {
		// Expose on window so the interactive layer (below) can reach it.
		window.__sim = initialize_canvas_simulation('boids1', false);
	});

	// --- "What am I looking at?" info panel --------------------------------
	// The activation viz draws a small "?" badge next to its title and
	// publishes the badge's circular hit-region on window.__vizInfo (canvas
	// units == CSS px == clientX/Y here). Tapping it toggles a centered DOM
	// card describing the rules; hovering brightens the badge (the viz reads
	// window.__vizInfoHover) and shows a pointer cursor.
	var infoCard = document.getElementById('viz-info-card');
	var infoBackdrop = document.getElementById('viz-info-backdrop');
	var infoClose = infoCard && infoCard.querySelector('.viz-info-close');

	function infoIsOpen() { return infoCard && !infoCard.hidden; }
	function showInfo() {
		if (!infoCard) return;
		infoCard.hidden = false;
		if (infoBackdrop) infoBackdrop.hidden = false;
	}
	function hideInfo() {
		if (!infoCard) return;
		infoCard.hidden = true;
		if (infoBackdrop) infoBackdrop.hidden = true;
	}
	function overInfoBadge(x, y) {
		var i = window.__vizInfo;
		if (!i) return false;
		var dx = x - i.cx, dy = y - i.cy;
		return dx * dx + dy * dy <= i.r * i.r;
	}

	if (infoClose) infoClose.addEventListener('click', hideInfo);
	// Tap-outside dismissal is handled in the pointerup handler below — NOT by a
	// backdrop 'click' listener. On touch, the synthetic click the browser fires
	// after the *opening* tap lands on the now-visible backdrop and would close
	// the panel the instant it opened (desktop dodges this because the synthetic
	// click resolves to <body>, not the backdrop). Routing dismissal through the
	// pointer stream keeps open/close symmetric across mouse, touch, and pen.
	document.addEventListener('keydown', function (e) {
		if (e.key === 'Escape' && infoIsOpen()) hideInfo();
	});
	document.addEventListener('mousemove', function (e) {
		var over = overInfoBadge(e.clientX, e.clientY);
		window.__vizInfoHover = over;
		document.body.style.cursor = over ? 'pointer' : '';
	});

	// Tap / click anywhere on the page spawns a new boid at the pointer
	// position. We use pointer events instead of `click` because iOS Safari
	// only fires `click` on elements it considers "clickable" (cursor:pointer,
	// onclick handler, or interactive descendants) — a tap on the bare body
	// background of this page simply never bubbles a click event up to the
	// document. `pointerdown` + `pointerup` fire reliably on any element,
	// any input type, on every browser since 2019.
	//
	// Click-vs-drag detection is rolled by hand: we remember where the
	// pointer went down, and only spawn if it came up within an 8px radius
	// — far enough that desktop drag-to-select and touch-swipe-scroll
	// gestures naturally skip the spawn, close enough that any reasonable
	// tap registers. The text-selection guard catches the remaining edge
	// case where the user drags within a single paragraph (down and up on
	// the same element, browser fires click anyway, but selection is now
	// non-empty).
	var pdX = 0, pdY = 0, pdId = null;
	document.addEventListener('pointerdown', function (e) {
		if (e.button !== undefined && e.button !== 0) { pdId = null; return; }
		pdX = e.clientX;
		pdY = e.clientY;
		pdId = e.pointerId;
	});
	document.addEventListener('pointerup', function (e) {
		if (pdId !== e.pointerId) return;
		pdId = null;
		var dx = e.clientX - pdX;
		var dy = e.clientY - pdY;
		var isTap = (dx * dx + dy * dy <= 64);          // <=8px = tap, not drag
		// The "?" badge toggles the panel and never spawns.
		if (isTap && overInfoBadge(e.clientX, e.clientY)) {
			if (infoIsOpen()) hideInfo(); else showInfo();
			return;
		}
		// While the panel is open, a tap outside the card dismisses it; a tap
		// inside it does nothing. Either way, no boid is spawned.
		if (infoIsOpen()) {
			if (!(e.target && e.target.closest && e.target.closest('.viz-info-card'))) hideInfo();
			return;
		}
		if (e.target && e.target.closest && e.target.closest('a')) return;
		if (!isTap) return;                             // drag, not a tap
		var sel = window.getSelection && window.getSelection();
		if (sel && sel.toString().length > 0) return;   // user selected text
		if (window.__sim) window.__sim.spawnBoid(e.clientX, e.clientY);
	});
	document.addEventListener('pointercancel', function () { pdId = null; });
});