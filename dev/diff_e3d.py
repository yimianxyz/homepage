"""Directly compare GPU _e3d_target vs JS computeEvolvedTarget on the IDENTICAL
injected boids, and verify the boids match. Pinpoints the E3D implementation
difference (params are confirmed identical)."""
import sys, json
import torch, subprocess
import sim_torch as st
from sim_torch import Sim
import planner_probe as pp

dev='cpu'; pp.WEIGHTS=st.load_weights('../js/predator_weights.json', device=dev)
J=json.load(open(sys.argv[1])); init=J['init']; nlive=len(init['bx'])
sim=Sim(seeds=[J['seed']],weights=pp.WEIGHTS,device=dev,auto_target='evolved',auto_target_opts=dict(pp.E3D),two_pass=True)
sim.boid_alive[0,:]=False; sim.boid_alive[0,:nlive]=True
sim.boid_pos[0,:nlive,0]=torch.tensor(init['bx'],dtype=torch.float64)
sim.boid_pos[0,:nlive,1]=torch.tensor(init['by'],dtype=torch.float64)
sim.boid_vel[0,:nlive,0]=torch.tensor(init['bvx'],dtype=torch.float64)
sim.boid_vel[0,:nlive,1]=torch.tensor(init['bvy'],dtype=torch.float64)
sim.pred_pos[0,0]=init['px']; sim.pred_pos[0,1]=init['py']
sim.always_recompute_target=True
e=pp._e3d_target(sim)
print('GPU _e3d_target:', round(float(e[0,0]),4), round(float(e[0,1]),4))
# JS computeEvolvedTarget on the same boids
js='''
const fs=require('fs');const vm=require('vm');const ctx={Math,console};vm.createContext(ctx);
vm.runInContext(fs.readFileSync('../js/predator.js','utf8'),ctx);
const I=JSON.parse(fs.readFileSync(process.argv[1],'utf8')).init;
const boids=I.bx.map((x,i)=>({position:{x:I.bx[i],y:I.by[i]},velocity:{x:I.bvx[i],y:I.bvy[i]}}));
const t=ctx.computeEvolvedTarget({x:I.px,y:I.py},boids,ctx.EVOLVED_PATROL,null);
console.log('JS computeEvolvedTarget:', t.x.toFixed(4), t.y.toFixed(4));
'''
open('/tmp/_e3djs.js','w').write(js)
print(subprocess.run(['node','/tmp/_e3djs.js',sys.argv[1]],capture_output=True,text=True,cwd='.').stdout.strip())
