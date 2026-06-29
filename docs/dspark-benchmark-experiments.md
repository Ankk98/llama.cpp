# DSpark benchmark experiments

Living log for DSpark speculative decoding performance on llama.cpp.
**Fair methodology only** - numbers below use vanilla-first + 3s cooldown unless noted.

## Environment

| Item | Value |
|------|-------|
| Machine | Strix Halo APU (Vulkan) |
| Target | `gemma-4-12B-it-QAT-Q4_0.gguf` (~7 GB) |
| Draft | `/tmp/dspark_gemma4_12b_q4pure.gguf` (~1.9 GB) |
| Backend | Vulkan (`-ngl 99 -ngld 99`) |
| Harness | `build/bin/compare_vanilla_speculative` |
| Eval prompts | `/tmp/dspark_eval/*.json` |

### Fair benchmark protocol

1. **Vanilla first** on a cool GPU (baseline not thermally penalized).
2. **3s cooldown** + KV reset before speculative (`DSPARK_BENCH_NO_COOLDOWN=1` to skip).
3. **Fixed config** per run (`DSPARK_NO_ADAPTIVE_NMAX=1` for sweeps).
4. Report **pp** and **tgp** tok/s for both paths plus speedup.

**Do not use spec-first ordering** for headline numbers. Earlier spec-first runs
inflated speedup by ~10-15% (e.g. 1.77x reported vs **1.57-1.61x fair**).

### Standard command

```bash
DSPARK_NO_ADAPTIVE_NMAX=1 ./build/bin/compare_vanilla_speculative \
  -m "$TARGET" -md "$DRAFT" \
  --input-ids /tmp/dspark_eval/code_500l.json \
  --spec-type draft-dspark \
  -c 1024 -ngl 99 -ngld 99 --temp 0 --seed 42 \
  --spec-draft-n-max 4 -n 400
```

### Profiling env vars

| Variable | Effect |
|----------|--------|
| `DSPARK_BENCH_NO_COOLDOWN=1` | Skip 3s pause between vanilla and spec |
| `DSPARK_NO_ADAPTIVE_NMAX=1` | Fixed n_max (required for fair sweeps) |
| `DSPARK_PROF=1` | Draft forward vs sampling split |
| `DSPARK_SPLIT_VERIFY=1` | Dual ctx verify (structurally slower; see below) |
| `DSPARK_GPU_GREEDY=1` | Opt-in GPU argmax accept (experimental; default off) |
| `DSPARK_NO_GPU_GREEDY=1` | Force CPU greedy accept (default path) |
| `DSPARK_NO_DEFER_LAYER_INP=1` | Full layer D2H on every verify row (slower) |

### Reading pp vs tgp

| Metric | Vanilla | Speculative |
|--------|---------|-------------|
| **pp** | Target prefill only | Prefill + first-token sample + `process()` setup |
| **tgp** | Target decode loop | Draft + verify + process (**use this for speedup**) |

Spec pp is not comparable to vanilla pp (extra setup). **tgp speedup is the headline metric.**

## Critical assessment (2026-06-29)

### What is real (structural)

| Change | Fair impact | Notes |
|--------|-------------|-------|
| Batched verify + greedy accept fast-path | Large | Core architecture |
| Shorter draft decode (`n_draft+1`) + per-length `block_gpu` | Moderate | Real GPU savings on draft |
| **Deferred partial layer D2H** | **~1-2%** | Correctness fix + skips rejected-row D2H |
| Verify-sized graph warmup (`n_max+1`) | Noise | Pre-alloc at init |
| `n_max=4` sweet spot | Best speed + **token match YES** | Config, not thermal |
| GPU greedy verify (opt-in) | **broken / slow** | Per-step graph alloc; default off |
| Adaptive n_max | Variance | 1.51x vs 1.62x fixed on code_500l |

### What is NOT real

| Approach | Fair result | Why |
|----------|-------------|-----|
| Spec-first bench order | +10-15% fake speedup | Vanilla runs hot |
| Split verify (dual ctx) | **0.79-1.04x** | 2nd target forward >> layer-tap savings |
| flash-attn on/off | **neutral** | ~1.5-1.7x within variance |
| ubatch 256-2048 | **neutral** | Within variance |
| `-c 512` vs `1024` | **neutral** | Within variance |
| `n_max=7` on coding | **0.96x, match NO** | Quality drift |

### Bottleneck (honest, measured 2026-06-29)

Per verify step (n_max=4, fair, defer layer on):

| Phase | ms/step | Notes |
|-------|---------|-------|
| decode submit (async return) | ~3 | `llama_decode` returns before GPU done |
| **GPU forward + logits fence** | **~60** | Dominates verify; labeled "accept" in harness |
| layer commit (partial D2H) | ~0.2 | Deferred path only |
| process (encode + inject) | ~1.3 | Not the limiter |
| draft step | ~18 | Separate from verify |

**The ~60 ms is 12B Q4 batched forward on 5 tokens**, not CPU argmax or layer D2H.
Vanilla 1-token forward is ~38 ms. Batching 5 tokens costs ~1.6x not 5x (good).

**2-3x coding target math (from ~1.62x fair):**

| Target | Required | Realistic path |
|--------|----------|----------------|
| **2.0x** | +23% tgp | ~3.1 accept/step at same verify ms, OR verify 60->46 ms (-23%) |
| **3.0x** | +85% tgp | Both: verify ~40 ms AND accept ~3.5+/step, OR draft model quality leap |

Config tuning alone will not reach 2x. Need **target batched-decode kernel/graph work**
and/or **higher acceptance without quality loss**.

## Fair throughput results

All: `-c 1024`, `temp=0`, `seed=42`, vanilla-first + 3s cooldown, `DSPARK_NO_ADAPTIVE_NMAX=1`.

### Coding (`code_500l`, n_predict=400)

| Config | tgp van | tgp spec | **Speedup** | Accept/step | Verify ms | Match |
|--------|---------|----------|-------------|-------------|-----------|-------|
| **defer layer (default)** | ~26.1 | **~42.2** | **1.62x** | 2.50 | ~64 | **YES** |
| no defer layer | ~26.0 | ~26.5 | 1.02x | 2.50 | ~201 | YES |
| adaptive n_max | ~26.1 | ~39.4 | 1.51x | 2.50 | ~69 | YES |
| GPU greedy (broken) | ~26.0 | ~35.5 | 1.37x | 3.89* | ~236 | NO |

\* GPU greedy reported fake 97% hit rate on wrong tokens.

**Sweet spot: `n_max=4`, defer layer on, CPU greedy accept.**

### Agentic (`agentic_plan`, n_predict=300)

| n_max | tgp van | tgp spec | Speedup | Accept/step | Verify ms | Match |
|-------|---------|----------|---------|-------------|-----------|-------|
| 2 | 25.61 | 35.86 | **1.40x** | 1.27 | 94 | NO |
| 3 | 25.89 | 35.62 | 1.38x | 1.56 | 107 | NO |

Agentic acceptance ~1.3-1.6/step limits speedup. **1.5x fair target not met** (~7% gap).

## Targets vs fair reality

| Workload | Target | Fair best | Gap to 2x | Gap to 3x |
|----------|--------|-----------|-----------|-----------|
| Coding | 2-3x | **1.62x** | 24% | 85% |
| Agentic | 1.5x | 1.40x | n/a | n/a |

## Experiment log

| Date | Commit | Change | Fair coding speedup |
|------|--------|--------|---------------------|
| 2026-06-29 | 843457e | Fair harness (vanilla-first + cooldown) | 1.61x |
| 2026-06-29 | f360dae | GPU greedy + defer layer + warmup (defer buggy) | regressed |
| 2026-06-29 | (pending) | Fix defer sync; timing; GPU greedy opt-in | **1.62x** |

## Open work (structural, ordered by impact)

1. **Faster 5-token target forward** - llama graph reuse audit, Vulkan small-batch matmul, fused per-row argmax in graph (skip logits D2H; saves little if forward dominates)
2. **Higher coding acceptance** - draft model / Q4 numerics; n_max>4 hurts match
3. **Fix GPU greedy path** - cached argmax graph, correct row stride; opt-in only
4. **Agentic acceptance** - prompt-class limiter, not verify ms

## How to append results

```bash
COMM=$(git rev-parse --short HEAD)
DSPARK_NO_ADAPTIVE_NMAX=1 ./build/bin/compare_vanilla_speculative \
  -m "$TARGET" -md "$DRAFT" --input-ids /tmp/dspark_eval/code_500l.json \
  --spec-type draft-dspark -c 1024 -ngl 99 -ngld 99 --temp 0 --seed 42 \
  --spec-draft-n-max 4 -n 400 2>&1 | tee /tmp/bench.out

grep -E 'generated:|throughput|tgp speedup|mean accepted|token match|verify step|accept \(GPU|layer commit' /tmp/bench.out
```

Record: date, commit, prompt, n, n_max, **pp van/spec, tgp van/spec**, speedup, accept/step, match.
