# Hybrid planner-reproduction plan — engineered features + tiny NN + minimal search

User pivot (2026-06-03): drop the pure end-to-end NN requirement. Find the
**simplest** architecture = engineered features + small NN + the *minimal*
deploy-time search that reproduces **≥99.9%** of the planner's catch rate, cheap
enough to run every frame in vanilla-JS browser. Inspiration: AlphaZero (keep
search, learn to make it shallow+narrow) and AlphaEvolve (evolve the config
against the catch harness). Do not stop until 99.9% is met.

## Governing theory (from paper research, see refs)
**Bertsekas limited-lookahead bound:** for `J = Σ_{t<H} γ^t r_t + γ^H V_T(s_H)`,
the H-step greedy policy's suboptimality ≤ ~`2 γ^H ε /(1−γ)` where ε=‖V_T−V*‖∞.
⇒ **a good learned terminal value V lets the horizon H collapse toward 1.** The
rollout horizon only needs to cover the part of the future V predicts poorly.
This is the AAlphaZero/TD-MPC insight: **don't kill the rollout — collapse it.**

Three independent cost levers (each maps to a paper family):
- **shallow** — cut rollout H, bridge with learned terminal value V (TD-MPC H=5
  + value; Bertsekas bound).
- **narrow** — policy-prior head ranks the K candidates so we only forward-sim
  the top 1–2 (AlphaZero PUCT prior).
- **cheap dynamics** — surrogate rollout need only predict catch-count correctly,
  NOT reproduce flock motion (MuZero: latent model trained for value-accuracy,
  not state reconstruction). In the limit V replaces the rollout entirely.

**Why prior reactive distillation failed (E1–E3, ≤34≈E3D, value-reg=21.58):**
those nets scored candidates from PER-FRAME features with H=0 (no rollout) and
were trained ONLY on planner-visited states. The deciding signal is the H-future,
absent from per-frame features ⇒ caps at E3D. Two fixes the literature prescribes:
(1) keep a short rollout + learned terminal value (put the future back in), and
(2) **DAgger** — relabel states the CHEAP policy visits (the plateau is partly a
distribution-shift artifact; a V trained only on planner states is wrong exactly
where the cheap policy wanders). DAgger is the single highest-leverage step.

## Training data already exists
`planner_probe.run_planner_log_cand` logs per decision: `(obs features, cand_rel
(K,2), gain (K), catches)` — gain = true H=120 catch count per candidate. That is
exactly the value-regression target `V(state, candidate) → catches` and the
argmax label for the policy-prior head. Supervised (MSE/Huber + CE), not RL,
because the env is deterministic & fully observed.

## Engineered per-candidate features (pursuit-guidance theory)
For each candidate target, compute (cheap, vanilla JS):
- range `R=|p_prey−p_pred|`, **closing velocity** `−Ṙ=−(relpos·relvel)/R`
- **time-to-go** `t_go≈R/|closing|`
- **LOS rate** `λ̇` (≈0 ⇒ collision course ⇒ high value — the PN signal)
- **lead/intercept point** (quadratic for predator-speed intercept of const-vel
  prey) → lead angle + **miss distance**
- **collision-cone / velocity-obstacle** membership (bool + margin)
- flock-structure: local prey density, dist to centroid, #prey within capture
  radius, flock-heading alignment vs LOS
Plus the existing `planner_obs` global features (pred vel, E3D target rel, M
nearest boids rel pos/vel, frac_alive, centroid rel).

## RESULT — load-bearing-horizon curve (single-pass, n=128, paired, 2026-06-03)
| H (rollout) | catches | retention | paired Δ vs H=120 (95% CI) |
|------------:|--------:|----------:|---------------------------:|
| 120 | 71.55 | 1.000 | — |
|  40 | 44.37 | 0.620 | [−28.9, −25.4] |
|  20 | 33.91 | 0.474 | [−39.5, −35.8] (≈E3D 34) |
|   8 | 33.21 | 0.464 | [−40.1, −36.6] (≈E3D) |
frac_non_e3d: H40=4.0%, H20=0.2%, H8=0.0%.

**Decisive finding.** The planner's edge is almost entirely a LONG-horizon effect.
Below H≈40 the planner collapses to the E3D baseline and stops deviating (catch
signal is too sparse over a short rollout → all-ties → default E3D). ⇒ a cheap
short-rollout student CANNOT work; the **learned terminal value V is mandatory**
and must carry ~27+ catches of beyond-horizon value. This is exactly the
Bertsekas regime where a good V collapses H — but here V is doing the heavy
lifting, so the value-net quality (features!) is the whole game. Confirms the
v2 feature idea: a per-candidate **2-body ballistic intercept** (predator +
targeted boid, const-vel, full horizon, O(1) no flock) is likely the single most
predictive cheap feature, since instantaneous pursuit geometry must stand in for
the long rollout. (H=60/30/12/5 from VM1 will fill the knee.)

## Execution ladder (cheapest first — each step a commit with its catch score)
1. **[RUNNING] Load-bearing-horizon frontier.** Sweep planner catches vs rollout
   H ∈ {120,90,60,40,30,20,12,8,5}, K=16 D=8, single-pass n=128 (VM1: 120/60/30/
   12/5; VM3: 90/40/20/8). This is the ε=0 (V_T≡0) curve: how short can the FULL
   rollout get before catches drop below 99.9%? Sets the horizon a learned V must
   bridge. Also sweep K and D next.
2. **Value-terminal short rollout.** Train tiny MLP `V(cand features)→remaining
   catches`; deploy = argmax over K of `(catches in short H') + V`. Find smallest
   (H', net width) hitting 99.9%. H'=0 (pure value net, no rollout) is the cheapest
   target; add H'∈{5,15,30} insurance if needed.
3. **DAgger 3–5 rounds** on states the cheap policy visits; relabel with true
   planner (run_planner) + true 120-frame catch counts; retrain. Closes dist-shift.
4. **Narrow K**: policy-prior head → forward-sim only top 1–2 candidates.
5. **AlphaEvolve config search** over {features, widths, K, H', D}, scored on a
   FIXED seed bank with a HELD-OUT test bank (never shown to search), MAP-elites
   binned on artifact size / deploy cost. Penalize per-frame compute to prevent
   reward-hacking by ballooning K/H'.

## Verification discipline
Every candidate selected by the two-pass JS catch metric (chaos ⇒ mean over many
seeds, paired Δ vs planner). Finalist JS-verified in real browser. Held-out seed
bank for the evolutionary search. Target: student ≥99.9% of zero-staleness
planner catches at n≥256, smallest artifact.

## Refs
TD-MPC (Hansen 2022, arXiv:2203.04955); Bootstrapped MPC (arXiv:2503.18871);
Value of Planning for Infinite-Horizon MPC (arXiv:2104.02863); Goal-Conditioned
Terminal Value for MPC (arXiv:2410.04929); MuZero (Schrittwieser); What model
does MuZero learn (arXiv:2306.00840); AlphaEvolve (DeepMind 2025); FunSearch
(Romera-Paredes, Nature 2024); DAgger (Ross 2011); Ideal PN guidance (AIAA);
Bertsekas, RL & Optimal Control (2019).
