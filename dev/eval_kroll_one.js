// Single-seed cheap-policy episode with a configurable K_roll (window.__KROLL),
// for parallel JS validation of the sweet-spot configs. Loads a js dir whose
// predator_cheap.js honors window.__KROLL (see /tmp/pc_kroll.js) + the strict net.
// Prints: SEED <s> KROLL <k> CATCHES <n> MS <wallms>
//   node eval_kroll_one.js --js /tmp/js_kroll --seed 200000 --kroll 2 --frames 1500
const fs = require('fs'), path = require('path'), vm = require('vm');
function pa(a){const o={js:'/tmp/js_kroll',seed:200000,kroll:1,hs:0,frames:1500,width:1680,height:1680,numBoids:120};
  for(let i=2;i<a.length;i++){const k=a[i];if(k==='--js')o.js=a[++i];else if(k==='--seed')o.seed=+a[++i];else if(k==='--kroll')o.kroll=+a[++i];else if(k==='--hs')o.hs=+a[++i];else if(k==='--frames')o.frames=+a[++i];}return o;}
function stub(){const n=()=>{};return new Proxy({},{get:()=>n});}
const FILES=['rng.js','vector.js','boid.js','predator.js','simulation.js','cheap_planner.js','predator_cheap.js'];
function build(o){const win={innerWidth:o.width,innerHeight:o.height,matchMedia:()=>({matches:false,addEventListener:()=>{}}),addEventListener:()=>{}};
  const sb={navigator:{userAgent:'Node'},window:win,document:{getElementById:()=>({getContext:()=>stub(),width:o.width,height:o.height}),addEventListener:()=>{}},
    fetch:u=>Promise.resolve({ok:true,status:200,json:()=>Promise.resolve(JSON.parse(fs.readFileSync(path.join(o.js,path.basename(u)),'utf8')))}),renderActivationViz:()=>{},Math,Date,console,setTimeout,setImmediate,Promise};
  sb.self=sb;sb.global=sb;const ctx=vm.createContext(sb);
  for(const f of FILES)vm.runInContext(fs.readFileSync(path.join(o.js,f),'utf8'),ctx,{filename:f});return{sb,win};}
async function ready(win){for(let i=0;i<100;i++){let d=false;win.__predatorReady.then(()=>d=true);await new Promise(r=>setImmediate(r));if(d)return;}throw new Error('nr');}
async function main(){
  const o=pa(process.argv);const{sb,win}=build(o);await ready(win);
  sb.window.__KROLL=o.kroll; if(o.hs>0)sb.window.__HS=o.hs;
  sb.setSimSeed(o.seed,12);sb.NUM_BOIDS=o.numBoids;
  const sim=new sb.Simulation('boids1');sim.canvasWidth=o.width;sim.canvasHeight=o.height;sim.initialize(false);sb.setFrameMs(12);
  const t0=Date.now();
  for(let f=0;f<o.frames;f++){sb.simTick();sim.tick();sim.render();}
  console.log(`SEED ${o.seed} KROLL ${o.kroll} CATCHES ${sim.boidsEaten} MS ${Date.now()-t0}`);
}
main().catch(e=>{console.error('ERR',e&&e.message);process.exit(1);});
