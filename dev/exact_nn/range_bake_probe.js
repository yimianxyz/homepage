// Faithful FRESH-PAGE load: index.html's exact separate-<script> order, MOBILE
// UA + mobile innerWidth, NOTHING forced. Reads PREDATOR_RANGE as prod bakes it.
'use strict';
const fs = require('fs'), path = require('path'), vm = require('vm');
const noop = () => {};
const cctx = {}; ['beginPath','moveTo','lineTo','stroke','fill','arc','clearRect','fillRect','closePath','save','restore','translate','rotate','scale','fillText','setLineDash','ellipse'].forEach(m=>cctx[m]=noop);
cctx.createLinearGradient = () => ({ addColorStop: noop });
global.window = { innerWidth: 390, innerHeight: 844, matchMedia: () => ({ matches:false, addEventListener:noop }), addEventListener: noop };
global.self = global;
global.navigator = { userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1' };
global.document = { getElementById: () => ({ getContext: () => cctx, width: 390, height: 844 }), addEventListener: noop };
global.renderActivationViz = noop;
global.fetch = (url) => Promise.resolve({ ok:true, status:200, json: () => Promise.resolve(JSON.parse(fs.readFileSync(path.join('js', path.basename(url)),'utf8'))) });
// EXACT index.html order (DOM-only files skipped, same as real load for these globals):
const ORDER = ['rng.js','vector.js','boid.js','predator.js','cheap_planner.js','predator_cheap.js','simulation.js'];
for (const f of ORDER) {
  vm.runInThisContext(fs.readFileSync(path.join('js', f),'utf8'), { filename: f });
  if (f === 'boid.js') console.log('after boid.js  : PREDATOR_RANGE =', global.PREDATOR_RANGE, '| typeof isMobileDevice =', typeof global.isMobileDevice, '| typeof PREDATOR_DESKTOP_RANGE =', typeof global.PREDATOR_DESKTOP_RANGE);
  if (f === 'simulation.js') {
    console.log('after sim.js   : PREDATOR_RANGE =', global.PREDATOR_RANGE, '| isMobileDevice() now =', global.isMobileDevice(), '| NUM_BOIDS =', global.NUM_BOIDS);
    console.log('what getBoidPredatorRange() would return IF re-run now:', global.getBoidPredatorRange());
  }
}
console.log('FINAL prod PREDATOR_RANGE (what boids/predator actually read):', global.PREDATOR_RANGE);
