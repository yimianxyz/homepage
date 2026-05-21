# GPU plan — what to do once ml-forecast-1 is accessible

## Phase 1 — bring up & verify (10 min)
1. `gcloud compute instances start ml-forecast-1 --csek-key-file=...` (now permitted).
2. SSH in via IAP. Check Python, NumPy, PyTorch versions (already DLVM image).
3. `git clone https://github.com/yimianxyz/homepage.git` into the SA's home dir.
4. Smoke-test `python3 dev/sim_np.py js/predator_weights.json 16 5000` — should hit ~25 catches, same as local.

## Phase 2 — port sim to GPU (1–2 hr)
Migrate `dev/sim_np.py` from NumPy to PyTorch with `device='cuda'`. The
sim is already batched — every state tensor is `(B, ...)`. Just need to
swap NumPy ops for torch equivalents and move arrays to GPU.

Expected speedup: B=16 currently 173s (NumPy CPU). On L4 GPU with
B=256, expect ~10–30s — that's 50–500x over CPU when amortized per
seed.

Validation: run B=16 on GPU and check mean catches matches CPU
NumPy's 25.44. If matches, port is correct.

## Phase 3 — big seed search (1 hr)
Train K=100 NNs (different inits, same dataset_v3 / hyperparams as
shipped). For each, eval at B=128 seeds × 5000 frames on GPU.
Distribution of mean catches tells us how rare the +2.25 shipped luck
really is, and the top-K might have a +3 to +5 lift.

## Phase 4 — RL training (2–4 hr)
With sim on GPU, can do REINFORCE/PPO end-to-end. Tiny NN (~150 params,
or grow to H=8/16). Batched rollouts give clean policy gradients. Skip
the rule entirely; reward = catches per frame.

Risk: policy gradient may not converge for this chaotic env without
careful reward shaping. Falls back to a pure seed search if so.

## Phase 5 — verify in JS (10 min)
Take the top GPU-found policy, run it through `dev/eval_tte.js` (JS
sim, ground truth). The NumPy/PyTorch port is an approximation — the
final number that ships must come from the JS sim.

## Coexistence notes (shared resource)
- Only start ml-forecast-1 (not all 3) unless I genuinely need 3-VM
  parallelism. Stop it when done so other agents have headroom.
- Use a unique work directory: `~/predator-rl-$(date +%Y%m%d-%H%M%S)/`
  so two agents don't clobber each other.
- Don't kill processes I don't own.
- ml-forecast-3 may have other agents on it — don't disturb.
