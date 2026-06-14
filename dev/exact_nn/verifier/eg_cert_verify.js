// eg_cert_verify.js — independent soundness stress-test of side-a's L1e zero-risk
// egBoid certificate (endgame/eg_bound.js). The claim: certify(k)==true ⇒ k is
// prod's TRUE argmin scan-t egBoid (eg_scan.js egPick, the exact 1400-step scan).
// SOUND ⇒ a false certification is impossible. We hunt one over a large sample of
// arbitrary LEGAL endgame states (random + adversarial near-tie geometries),
// since the only residual risk is float rounding in the integer bounds.
'use strict';
const path = require('path');
const D = path.join(__dirname, '..', 'endgame');
const { certify } = require(path.join(D, 'eg_bound.js'));
const { egPick, scanT } = require(path.join(D, 'eg_scan.js'));

function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;var t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}

const CELLS = [[390,844],[820,1180],[1024,768],[1512,982],[1680,1050],[2560,1440]];
function main(){
  const N = +(process.argv[2]||500000);
  const r = mulberry32(+(process.argv[3]||12345));
  let states=0, commits=0, certified=0, falseCerts=0, certWrongDetail=[];
  let reachableCommits=0, certifiedReachable=0;
  for(let it=0; it<N; it++){
    const [W,H]=CELLS[(r()*CELLS.length)|0];
    const PX=W+20, PY=H+20;
    const nb = 1 + ((r()*5)|0);                    // 1..5 boids (endgame)
    const px=r()*PX-10, py=r()*PY-10;
    const boids=[];
    // mix: mostly random; sometimes adversarial near-tie (two boids similar geometry)
    const adv = r()<0.25;
    for(let i=0;i<nb;i++){
      let bx,by;
      if(adv && i>0){ bx=boids[0].x+(r()*2-1)*8; by=boids[0].y+(r()*2-1)*8; }
      else { bx=r()*W; by=r()*H; }
      const ang=r()*2*Math.PI, sp=r()*6;           // |v|<=6 legal
      boids.push({x:bx,y:by,vx:Math.cos(ang)*sp,vy:Math.sin(ang)*sp});
    }
    states++;
    const gt = egPick(px,py,boids,W,H);            // exact prod egBoid
    commits++;
    if(!gt.nearestFallback) reachableCommits++;
    // check certify for EVERY index k (a false cert for any wrong k breaks soundness)
    for(let k=0;k<nb;k++){
      if(certify(px,py,boids,W,H,k)){
        if(k===gt.egIdx){ certified++; if(!gt.nearestFallback) certifiedReachable++; }
        else { falseCerts++; if(certWrongDetail.length<10) certWrongDetail.push({W,H,px,py,boids,k,trueEg:gt.egIdx,ts:gt.ts}); }
      }
    }
  }
  console.log(JSON.stringify({
    states, commits,
    falseCertifications: falseCerts,
    certified, certifiedFrac:+(certified/commits).toFixed(4),
    certifiedFrac_ofReachable:+(certifiedReachable/Math.max(1,reachableCommits)).toFixed(4),
    reachableCommits,
    verdict: falseCerts===0 ? 'SOUND — 0 false certifications over '+commits+' commits' : 'UNSOUND — '+falseCerts+' false certs!',
    falseCertExamples: certWrongDetail.slice(0,3),
  },null,1));
  process.exit(falseCerts>0?2:0);
}
main();
