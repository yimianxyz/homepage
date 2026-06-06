// Root-cause test: is the browser's lookahead (rolloutFlatState) a FAITHFUL
// predictor of the browser's actual game (boid.js+predator.js+simulation.js)?
// Run the real game to a mid-game state, freeze a target T, then compare:
//   A = rolloutFlatState(snapshot, T, H)         -- the lookahead's prediction
//   B = the SAME live game continuing H frames    -- ground truth
// both with the IDENTICAL fixed-target predator (chase-nearest-POLICY_R else seek T).
// If A != B, the lookahead mispredicts its own game => bad target choices => 2x gap.
// Per-frame log of predator pos / alive-count / catches localizes the first divergence.
//   node diff_rollout_vs_game.js --seed 200000 --warm 200 --H 60
const fs = require('fs'), path = require('path'), vm = require('vm');
function pa(a){const o={js:path.join(__dirname,'..','js'),pc:'/tmp/pc_test.js',seed:200000,warm:200,H:60,width:1680,height:1680,numBoids:120};
  for(let i=2;i<a.length;i++){const k=a[i];if(k==='--js')o.js=a[++i];else if(k==='--pc')o.pc=a[++i];else if(k==='--seed')o.seed=+a[++i];else if(k==='--warm')o.warm=+a[++i];else if(k==='--H')o.H=+a[++i];}return o;}
function stub(){const n=()=>{};return new Proxy({},{get:()=>n});}
function build(o){const win={innerWidth:o.width,innerHeight:o.height,matchMedia:()=>({matches:false,addEventListener:()=>{}}),addEventListener:()=>{}};
  const sb={navigator:{userAgent:'Node'},window:win,document:{getElementById:()=>({getContext:()=>stub(),width:o.width,height:o.height}),addEventListener:()=>{}},
    fetch:u=>Promise.resolve({ok:true,status:200,json:()=>Promise.resolve(JSON.parse(fs.readFileSync(path.join(o.js,path.basename(u)),'utf8')))}),renderActivationViz:()=>{},Math,Date,console,setTimeout,setImmediate,Promise};
  sb.self=sb;sb.global=sb;const ctx=vm.createContext(sb);
  // load real js, but swap predator_cheap.js for the instrumented /tmp copy
  const order=['rng.js','vector.js','boid.js','predator.js','simulation.js','cheap_planner.js'];
  for(const f of order)vm.runInContext(fs.readFileSync(path.join(o.js,f),'utf8'),ctx,{filename:f});
  vm.runInContext(fs.readFileSync(o.pc,'utf8'),ctx,{filename:'pc_test.js'});
  return{sb,win};}
async function ready(win){for(let i=0;i<100;i++){let d=false;win.__predatorReady.then(()=>d=true);await new Promise(r=>setImmediate(r));if(d)return;}throw new Error('not ready');}
async function main(){
  const o=pa(process.argv);const{sb,win}=build(o);await ready(win);
  const V=sb.Vector,MS=sb.PREDATOR_MAX_SPEED,MF=sb.PREDATOR_MAX_FORCE,POLICY_R=80;
  const ce=sb.computeEvolvedTarget,EP=sb.EVOLVED_PATROL;
  sb.setSimSeed(o.seed,12);sb.NUM_BOIDS=o.numBoids;
  const sim=new sb.Simulation('boids1');sim.canvasWidth=o.width;sim.canvasHeight=o.height;sim.initialize(false);sb.setFrameMs(12);
  // warm up with the NORMAL cheap policy to reach a realistic mid-game state
  for(let f=0;f<o.warm;f++){sb.simTick();sim.tick();sim.render();}
  const pred=sim.predator,boids=sim.boids;
  // optional: force a catch-heavy regime to exercise cooldown/growth/decay/catch.
  // Warp the predator onto the nearest boid + max size + cleared cooldown, so it
  // catches immediately and then chases nearest within POLICY_R (sustained catches).
  if(process.argv.includes('--bigsize')){
    pred.currentSize=pred.maxSize;pred.lastFeedTime=sb.simNow()-100000;
    let bd=Infinity,bi=0;for(let i=0;i<boids.length;i++){const dx=boids[i].position.x-pred.position.x,dy=boids[i].position.y-pred.position.y,d2=dx*dx+dy*dy;if(d2<bd){bd=d2;bi=i;}}
    pred.position.x=boids[bi].position.x;pred.position.y=boids[bi].position.y;pred.velocity.x=0;pred.velocity.y=0;
  }
  // freeze a target T. Default = flock centroid (drives the predator INTO the
  // flock so catches happen, exercising the cooldown/growth/decay path).
  const useE3D=process.argv.includes('--e3d');
  let T;
  if(useE3D){const lite=boids.map(b=>({position:{x:b.position.x,y:b.position.y},velocity:{x:b.velocity.x,y:b.velocity.y}}));
    T=ce({x:pred.position.x,y:pred.position.y},lite,EP,null)||{x:pred.position.x,y:pred.position.y};}
  else{let cx=0,cy=0;for(const b of boids){cx+=b.position.x;cy+=b.position.y;}T={x:cx/boids.length,y:cy/boids.length};}
  // ---- Branch A: the lookahead's prediction from the snapshot ----
  sb.window.__cheap._configure(sim);
  const snap=sb.window.__cheap._snapshot(pred,boids);
  const rr=sb.window.__cheap._rollout(snap,T.x,T.y,o.H);
  const trajA=sb.window.__rollTraj.slice();
  const catchesA=rr.catches;
  // ---- Branch B: continue the SAME live game H frames, fixed-target predator ----
  const eaten0=sim.boidsEaten;
  sb.window.__cheap.force=function(p,bs){
    if(bs.length===0)return new V(0,0);
    const px=p.position.x,py=p.position.y;let bestD2=Infinity,nx=0,ny=0;
    for(let i=0;i<bs.length;i++){const dx=bs[i].position.x-px,dy=bs[i].position.y-py,d2=dx*dx+dy*dy;if(d2<bestD2){bestD2=d2;nx=dx;ny=dy;}}
    let des;if(bestD2<POLICY_R*POLICY_R)des=new V(nx,ny);else des=new V(T.x-px,T.y-py);
    des.iFastSetMagnitude(MS);const s=des.subtract(p.velocity);s.iFastLimit(MF);return s;
  };
  const trajB=[];
  for(let f=0;f<o.H;f++){sb.simTick();sim.tick();sim.render();
    trajB.push({px:pred.position.x,py:pred.position.y,sz:pred.currentSize,c:sim.boidsEaten-eaten0,n:sim.boids.length});}
  const catchesB=sim.boidsEaten-eaten0;
  // optional: dump the exact branch-B init state + fixed target + trajectory to
  // JSON, so the sim_torch arm can be injected with the IDENTICAL state.
  const dumpIdx=process.argv.indexOf('--dump');
  if(dumpIdx>=0){
    const fp=process.argv[dumpIdx+1];
    const init={bx:snap.bx,by:snap.by,bvx:snap.bvx,bvy:snap.bvy,px:snap.px,py:snap.py,
      pvx:snap.pvx,pvy:snap.pvy,psize:snap.psize,lastFeed:snap.lastFeed,nowMs:snap.nowMs};
    const planDbg=sb.window.__cheap._planDebug(sb.window.__cheap._snapshot(pred,boids));
    fs.writeFileSync(fp,JSON.stringify({seed:o.seed,warm:o.warm,H:o.H,T:T,init:init,
      trajB:trajB,catchesB:catchesB,planDbg:planDbg}));
  }
  // ---- compare ----
  let firstPosDiv=-1,firstNDiv=-1;
  for(let f=0;f<o.H;f++){
    const a=trajA[f],b=trajB[f];
    const dp=Math.abs(a.px-b.px)+Math.abs(a.py-b.py);
    const nA=o.numBoids-a.c, nB=b.n; // alive in A = total - catches (A starts from snapshot of `warm` survivors though)
    if(firstPosDiv<0&&dp>1e-6)firstPosDiv=f;
    if(firstNDiv<0&&a.c!==b.c)firstNDiv=f;
  }
  // print a compact per-frame table around the first divergence
  const out={seed:o.seed,warm:o.warm,H:o.H,T:{x:+T.x.toFixed(2),y:+T.y.toFixed(2)},
    catchesA,catchesB,firstPosDiv,firstCatchDiv:firstNDiv,
    survivorsAtWarm:boids.length};
  console.log(JSON.stringify(out));
  const lo=Math.max(0,(firstPosDiv<0?o.H-6:firstPosDiv-2)),hi=Math.min(o.H,lo+10);
  console.log('frame |  A.px    A.py   A.c |  B.px    B.py   B.c B.n | dpos');
  for(let f=lo;f<hi;f++){const a=trajA[f],b=trajB[f];const dp=(Math.abs(a.px-b.px)+Math.abs(a.py-b.py));
    console.log(`${String(f).padStart(5)} | ${a.px.toFixed(2).padStart(7)} ${a.py.toFixed(2).padStart(7)} ${String(a.c).padStart(3)} | ${b.px.toFixed(2).padStart(7)} ${b.py.toFixed(2).padStart(7)} ${String(b.c).padStart(3)} ${String(b.n).padStart(3)} | ${dp.toExponential(2)}`);}
}
main().catch(e=>{console.error(e);process.exit(1);});
