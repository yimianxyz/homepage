'use strict';
const path = require('path');
const { runGame } = require('../diff_harness.js');
const { forkRun } = require('./endgame_fork.js');
const POLICY_DIR = path.join(__dirname, '..', '..', '..', 'js');
const CAND = path.join(__dirname, '..', 'candidates', 'split.js');
const MF = 60000, FORK_N = 13;
const CASES = [[390,844,true,0],[390,844,false,120]]; // mobile natural(60) + cross small forced120 (both fast)
const SEEDS = [270000, 270001];
const TS = [5, 8, 12];
async function fullGame(W,H,ua,N0,seed,T){
  process.env.EXACTNN_SPLIT_RULE='count'; process.env.EXACTNN_SPLIT_T=String(T);
  const r = await runGame({policyDir:POLICY_DIR,W,H,startBoids:N0,scatter:false,uaMobile:ua,maxFrames:MF,postExtinct:0,decisions:false,fastRender:true,mismatchLimit:0,mode:'fork',resync:false},seed,CAND);
  delete process.env.EXACTNN_SPLIT_T; delete process.env.EXACTNN_SPLIT_RULE;
  return {frames:r.frames,eaten:r.eaten,cleared:r.cleared,trajDigest:r.trajDigest};
}
(async()=>{
  let pass=0,fail=0; const fails=[];
  for(const [W,H,ua,N0] of CASES) for(const seed of SEEDS){
    const fk = await forkRun({W,H,uaMobile:ua,N0,seed,Ts:TS,rule:'count',forkN:FORK_N,maxFrames:MF,digest:true});
    const byT={}; for(const r of fk.results) byT[r.T]=r;
    for(const T of TS){
      const full=await fullGame(W,H,ua,N0,seed,T); const f=byT[T];
      const ok = f.frames===full.frames && f.eaten===full.eaten && f.cleared===full.cleared && f.trajDigest===full.trajDigest;
      if(ok)pass++; else {fail++; fails.push({W,H,ua,N0,seed,T,fork:{frames:f.frames,eaten:f.eaten,d:f.trajDigest},full:{frames:full.frames,eaten:full.eaten,d:full.trajDigest}});}
    }
    process.stderr.write(`[${W}x${H} ua=${ua} N0=${N0} seed=${seed}] prefix=${fk.prefixFrames}f forkN=${fk.forkNcount} pass=${pass} fail=${fail}\n`);
  }
  console.log('\nSMOKE '+(fail===0?'PASS':'FAIL')+' pass='+pass+' fail='+fail);
  if(fail) console.log(JSON.stringify(fails,null,2));
})();
