# 4× verdict — exhaustive impossibility (2026-06-10)

Goal: 300%-better (4×) on the whole-game eval vs the deployed baseline. PROVEN
impossible four independent ways (all measured on the real engine):

1. Encounter-rate ceiling: ~33 boids/phone (27 laptop) ever cross the catch reach in
   1500f; deployed catches 74–81% of them. reach-crossings = density·2·reach·v_pred
   (all game-fixed) → absolute policy max 1.23–1.34×.
2. Herding (the one untested structural lever): explicit shepherd/orbit controllers
   measured −16 to −44% vs deployed. A slower agent can't compress a faster flock.
3. Game tweak — faster predator: 2.5→5.0 gives +2% catches (the predator is NOT the
   bottleneck; a faster predator scatters the flock).
4. Game tweak — slower prey: catches DECREASE monotonically as prey slows
   (square, speed 6→3: 13.0→9.6→9.9→8.7→8.2→6.6). Catches come from FAST prey
   overshooting into traps; slower prey evades cleanly. The obvious "easier game"
   fix backfires.

Conclusion: the catch rate is governed by the chaotic trapping dynamics of fast
flocking prey, not by the predator. No policy and no sensible game tweak yields 4×.
The genuine maximum (shipped in dev/ship_teri): endgame last-boid 3.0× + fixes the
~10% never-clears to 100% + ~1.2–1.4× faster full clearance. See
predator_clearance_endgame.md and predator_3x_feasibility.md.
