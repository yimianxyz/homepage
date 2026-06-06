// Compute JS computeEvolvedTarget on the boids in a dumped state file.
const fs = require('fs'), vm = require('vm');
const ctx = { Math: Math, console: console };
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(__dirname + '/../js/predator.js', 'utf8'), ctx);
const I = JSON.parse(fs.readFileSync(process.argv[2], 'utf8')).init;
const boids = I.bx.map((x, i) => ({ position: { x: I.bx[i], y: I.by[i] }, velocity: { x: I.bvx[i], y: I.bvy[i] } }));
const t = ctx.computeEvolvedTarget({ x: I.px, y: I.py }, boids, ctx.EVOLVED_PATROL, null);
console.log('JS computeEvolvedTarget:', t.x.toFixed(4), t.y.toFixed(4));
