# autoresearch-at-home: M5Max Session Plan

## Status
- Agent: M5Max, registered and connected to swarm
- Branch: autoresearch/mar13-M5Max
- Fork: vnnkl/autoresearch-mlx (origin), upstream: ElixirLabsUK/autoresearch-mlx
- PR open: https://github.com/ElixirLabsUK/autoresearch-mlx/pull/1 (collab mode for MLX)
- Hardware: Apple M5 Max, 128GB unified memory, xl tier

## Run & publish flow (every experiment)
```python
from coordinator import Coordinator
coord = Coordinator()
coord.agent_id = "M5Max"

# 1. THINK — check swarm state
coord.analyze_swarm()
coord.get_unclaimed_hypotheses()

# 2. CLAIM
exp_key = coord.claim_experiment("description of experiment")

# 3. RUN — edit train.py, then:
#    uv run train.py > run.log 2>&1
#    grep "^val_bpb:\|^num_steps:\|^total_tokens_M:" run.log

# 4. COMMIT + PUSH before publishing (so commit URL matches the code)
#    git add train.py && git commit -m "desc" && git push origin autoresearch/mar13-M5Max

# 5. PUBLISH (all three, every time)
coord.publish_result(exp_key, val_bpb, memory_gb, status, description, open("train.py").read(),
    extra_metrics={"num_steps": N, "total_tokens_M": T, "num_params_M": P})
coord.post_insight("what we learned and why", evidence_keys=["results/..."])
coord.publish_hypothesis(title="next idea", hypothesis="reasoning",
    suggested_config={...}, evidence_keys=["results/..."], priority=4)
```

## CRITICAL BUG (FIXED): optimizer API
MLX `optimizer.apply_gradients(grads, model)` silently does nothing.
Correct: `optimizer.update(model, grads)` + `mx.eval(model.parameters(), optimizer.state)`.
Now using custom AdamW class with per-param-group LR (no built-in MLX optimizer).

## Current best: val_bpb = 1.285 (xl tier best, tuned LRs)

## Experiment results (this session)
| Experiment | val_bpb | Steps | Params | Status |
|---|---|---|---|---|
| Baseline (optimizer fix) | 1.686 | 2390 | 8.1M | keep |
| + VE, sqReLU, zero-init, logit cap, resid lambdas | 1.614 | 2311 | 10.7M | keep |
| + Per-param-group LR (emb=1.5, unemb=0.004, matrix=0.04) | **1.366** | 2236 | 10.7M | **BEST** |
| Wider model (N_EMBD=384, 15.6M params) | 1.438 | 1538 | 15.6M | discard |
| + SSSL sliding window (1024 short, 2048 long) | 1.376 | 2298 | 10.7M | discard (worse) |
| + EMBEDDING_LR=2.0 (up from 1.5) | **1.362** | 2355 | 10.7M | **NEW BEST** |
| + EMBEDDING_LR=2.5 | 1.362 | 2282 | 10.7M | plateau, keep 2.0 |
| + WARMDOWN_RATIO=0.3 (up from 0.2) | 1.355 | 2351 | 10.7M | keep |
| + WARMDOWN_RATIO=0.4 | **1.351** | 2356 | 10.7M | **NEW BEST** |
| + WARMDOWN_RATIO=0.5 | 1.359 | 2341 | 10.7M | discard (0.4 is optimum) |
| DEPTH=6, N_EMBD=192 | 1.401 | 1618 | 8.2M | discard (too few steps) |
| MATRIX_LR=0.05 (up from 0.04) | 1.370 | 2346 | 10.7M | discard (0.04 is better) |
| WARMDOWN=1.0, FINAL_LR_FRAC=0.01 (from helios) | 1.364 | 2326 | 10.7M | discard (doesn't transfer) |
| Optimizer state reset at 60% | 1.355 | 2349 | 10.7M | discard (small regression) |
| WD=0.15 with linear decay (from helios) | 1.342 | 2309 | 10.7M | keep |
| WD=0.25 with linear decay | **1.341** | 2355 | 10.7M | **NEW BEST** |
| WD=0.4 with linear decay | 1.348 | 2366 | 10.7M | discard (0.25 is optimum) |
| gc.freeze()+gc.disable() | 1.351 | 2373 | 10.7M | discard (no help/worse) |
| WARMDOWN=1.0, FINAL_LR=0.01 (helios) | 1.364 | 2326 | 10.7M | discard |
| MATRIX_LR=0.05 | 1.370 | 2346 | 10.7M | discard |
| Alternating VE (layers 1,3 only) | 1.368 | 2275 | 8.7M | discard (all-layer VE better) |
| --- AdamW-only experiments (baseline=1.341) --- | | | | |
| ADAM_BETAS=(0.8, 0.95) [AdamW] | 1.355 | 2170 | 10.7M | discard |
| 4x MLP hidden [AdamW] | 1.396 | 2147 | 11.3M | discard |
| Cautious weight decay [AdamW] | 1.343 | 2260 | 10.7M | neutral |
| EVAL_BATCH_SIZE=128 [AdamW] | 1.342 | 2211 | 10.7M | keep (measurement) |
| --- Muon optimizer (initial sweep, old config: betas=0.65/0.9, WD=0.25, emb_lr=2.0) --- | | | | |
| Muon LR=0.04 | 1.348 | 2071 | 10.7M | discard |
| Muon LR=0.02 | 1.327 | 2210 | 10.7M | keep |
| Muon LR=0.01 | **1.321** | 2192 | 10.7M | best at old config |
| Muon LR=0.005 | 1.333 | 2188 | 10.7M | discard |
| Muon LR=0.015 | 1.327 | 2192 | 10.7M | discard |
| --- Muon tuning (cumulative, base=Muon LR=0.01) --- | | | | |
| + WARMUP=0.05 | 1.317 | 2162 | 10.7M | keep |
| + WARMUP=0.1 | 1.312 | 2209 | 10.7M | keep |
| + WARMUP=0.2 | **1.311** | 2190 | 10.7M | keep |
| + WARMDOWN=0.3 (vs 0.4) | 1.313 | 2189 | 10.7M | discard |
| + WD=0.1 (from 0.25) | **1.303** | 2201 | 10.7M | keep |
| + WD=0.05 | 1.306 | 2180 | 10.7M | discard |
| + WD=0.15 | 1.306 | 2186 | 10.7M | discard |
| + WD=0.0 | 1.303 | 2180 | 10.7M | discard |
| + EMBEDDING_LR=0.6 (from 2.0) | **1.302** | 2192 | 10.7M | keep |
| + ADAM_BETAS=(0.8,0.95) for non-matrix | **1.297** | 2184 | 10.7M | keep |
| + NorMuon variance normalization | 1.297 | 2204 | 10.7M | neutral |
| --- Muon LR re-sweep (new config: betas=0.8/0.95, WD=0.1, emb_lr=0.6) --- | | | | |
| Muon LR=0.015 [new config] | 1.296 | 2072 | 10.7M | keep |
| Muon LR=0.02 [new config] | 1.292 | 2123 | 10.7M | keep |
| Muon LR=0.04 [new config] | **1.291** | 2202 | 10.7M | **NEW BEST** |
| Muon LR=0.08 [new config] | 1.303 | 2158 | 10.7M | discard (too high) |
| Muon LR=0.06 [new config] | 1.295 | 2261 | 10.7M | discard (0.04 confirmed optimal) |
| --- LR re-tuning round (cumulative, base=Muon 0.04, emb_lr=0.6) --- | | | | |
| EMBEDDING_LR=0.3 | 1.289 | 2148 | 10.7M | keep |
| EMBEDDING_LR=0.45 | 1.289 | 2243 | 10.7M | tied, keep 0.3 |
| EMBEDDING_LR=0.15 | 1.294 | 2209 | 10.7M | discard |
| EMBEDDING_LR=0.2 | 1.291 | 2235 | 10.7M | discard |
| EMBEDDING_LR=1.0 | 1.292 | 2227 | 10.7M | discard |
| WARMUP=0.1 (emb=0.3) | 1.292 | 2214 | 10.7M | discard |
| WARMUP=0.3 (emb=0.3) | 1.289 | 2234 | 10.7M | neutral |
| WARMUP=0.4 (emb=0.3) | 1.293 | 2249 | 10.7M | discard |
| Muon LR=0.03 (emb=0.3) | **1.286** | 2271 | 10.7M | **keep** |
| Muon LR=0.025 | 1.287 | 2239 | 10.7M | discard |
| Muon LR=0.035 | 1.287 | 2218 | 10.7M | discard |
| UNEMBEDDING_LR=0.002 | 1.287 | 2236 | 10.7M | discard |
| UNEMBEDDING_LR=0.003 | 1.287 | 2233 | 10.7M | discard |
| UNEMBEDDING_LR=0.008 | 1.291 | 2240 | 10.7M | discard |
| Cosine WD decay | 1.288 | 2242 | 10.7M | discard |
| Constant WD (no decay) | 1.302 | 2230 | 10.7M | discard |
| DEPTH=6 (Muon) | 1.288 | 1640 | 14.0M | discard (too few steps) |
| SCALAR_LR=1.0 | 1.289 | 2242 | 10.7M | discard |
| **SCALAR_LR=0.25** | **1.285** | 2237 | 10.7M | **NEW BEST** |
| SCALAR_LR=0.125 | 1.287 | 2215 | 10.7M | discard |

## Key learnings
1. Per-param-group LR is the BIGGEST single improvement (1.614 -> 1.366, 15.3%)
2. Value Embeddings helped (~4% from 1.686 -> 1.614)
3. Wider model HURTS on MLX — fewer steps outweighs capacity gain
4. SSSL sliding window hurts with only 4 layers
5. **Muon optimizer is a game changer**: 1.342 (AdamW best) -> 1.285 (Muon best), 4.2%
6. Muon needs full config re-tune: lower WD (0.1 vs 0.25), lower emb_lr (0.3 vs 2.0), higher betas (0.8/0.95 vs 0.65/0.9), 20% warmup
7. **Hyperparams interact**: Muon LR optimal shifted from 0.01→0.04→0.03 as other params changed. Always re-sweep after big changes
10. SCALAR_LR for resid/x0 lambdas: 0.25 beats 0.5 (lower LR for scalar params helps)
8. NorMuon variance normalization is neutral, cautious WD is neutral
9. dmodel_lr_scale = (n_embd/768)^-0.5 applied to AdamW LRs

## Current config (in train.py)
- DEPTH=4, N_EMBD=256, N_HEAD=2, N_KV_HEAD=1, BS=8
- Muon for 2D block params: MATRIX_LR=0.03, momentum ramp 0.85->0.95
- AdamW for rest: EMBEDDING_LR=0.3, UNEMBEDDING_LR=0.004, SCALAR_LR=0.25
- ADAM_BETAS=(0.8, 0.95), WD=0.1 (linear decay), WARMUP=0.2, WARMDOWN=0.4
- Architecture: VE (all layers), squared ReLU, zero-init c_proj, logit cap 15, resid+x0 lambdas
- EVAL_BATCH_SIZE=128
- ~10.7M params, ~2200 steps/5min, ~130k tok/sec

## Solo run reference
- Repo: /Users/constantin/Code/autoresearch (val_bpb=1.337)
- We now beat solo run best: 1.285 vs 1.337 (3.9% better!)

## Tuning approach
Binary search, not incremental. Jump wide to bracket (e.g. 0.4 → 0.8), then bisect.

## Next experiments to try
1. SCALAR_LR bisect: try 0.375 (between 0.25 and 0.5)
2. WD sweep with new LRs (try 0.05, 0.15 — currently 0.1)
3. WARMDOWN_RATIO re-sweep (try 0.3, 0.5 — currently 0.4, may shift with new LRs)
4. Muon momentum end: try 0.9, 0.99 (currently 0.95)
5. Muon ramp steps: try 150, 600 (currently 300)
6. Logit cap: try 10, 20 (currently 15)
7. Re-tune ADAM_BETAS with new config (try 0.85/0.95, 0.8/0.99)

## Swarm state
- Global best: 0.961639 by helios (CUDA+Muon)
- xl tier best: 1.285 by M5Max (us!)
- Medium tier best: 1.094 by cipher

## Files
- train.py — current best config (Muon + AdamW, all arch improvements)
- coordinator.py — Ensue integration
- collab.md — collaborative protocol
- run.log — latest training output
