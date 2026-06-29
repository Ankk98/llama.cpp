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
| `DSPARK_NO_GPU_GREEDY=1` | Force CPU greedy accept (full vocab logits D2H) |
| `DSPARK_NO_DEFER_LAYER_INP=1` | Extract all layer rows during verify decode |

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
| GPU greedy verify accept (skip logits D2H) | TBD | Vulkan argmax on logits rows |
| Deferred partial layer-input D2H | TBD | Only committed rows after accept |
| Verify-sized graph warmup (`n_max+1`) | TBD | `llama_graph_reserve` at DSpark init |
| `n_max=4` sweet spot | ~same tgp as n_max=3, **token match YES** | Config, not thermal |
| Process sync skip (short batches) | Noise | <1% |
| Adaptive n_max | Variance | Disabled for fair runs |

### What is NOT real

| Approach | Fair result | Why |
|----------|-------------|-----|
| Spec-first bench order | +10-15% fake speedup | Vanilla runs hot |
| Split verify (dual ctx) | **0.79-1.04x** | 2nd target forward >> layer-tap savings |
| flash-attn on/off | **neutral** (~1.52-1.58x) | Within variance |
| ubatch 256-2048 | **neutral** | Within variance |
| `-c 512` vs `1024` | **neutral** | Within variance |

### Bottleneck (honest)

Verify step ~110-128 ms/step = **12B Q4 batched forward on 4-5 tokens** + GPU fence.
`decode_submit_ms` ~2 ms (async return); almost all time is GPU matmul, not CPU or layer D2H.
Layer-input taps are views on existing forward activations; disabling them via split-verify
still requires a second forward on committed tokens and loses badly.

**2x fair tgp is not reachable by config tuning alone.** From fair **~1.61x**, need ~25% more.
Real paths: faster target batched decode (graph/kernel), or higher acceptance without larger
verify batches (draft model quality).

## Fair throughput results

All: `-c 1024`, `temp=0`, `seed=42`, vanilla-first + 3s cooldown, `DSPARK_NO_ADAPTIVE_NMAX=1`.

### Coding (`code_500l`, n_predict=400)

| n_max | pp van | tgp van | pp spec* | tgp spec | **Speedup** | Accept/step | Verify ms | Match |
|-------|--------|---------|----------|----------|-------------|-------------|-----------|-------|
| **4** | - | **25.48** | - | **41.15** | **1.61x** | 2.50 | 128 | **YES** |
| 3 | - | 25.58 | - | 40.98 | 1.60x | 2.05 | 111 | NO |
| 2 | - | 25.64 | - | 39.49 | 1.54x | 1.46 | 93 | NO |

\* spec pp includes one-time setup (~270 ms); not compared here.

**Sweet spot: `n_max=4`** - best fair speedup with greedy token match at temp=0.

### Agentic (`agentic_plan`, n_predict=300)

| n_max | tgp van | tgp spec | Speedup | Accept/step | Verify ms | Match |
|-------|---------|----------|---------|-------------|-----------|-------|
| 2 | 25.61 | 35.86 | **1.40x** | 1.27 | 94 | NO |
| 3 | 25.89 | 35.62 | 1.38x | 1.56 | 107 | NO |

Agentic acceptance ~1.3-1.6/step limits speedup. **1.5x fair target not met** (~7% gap).

### Config sweeps (fair order, same session - thermal drift between runs)

Long back-to-back sweeps are unreliable. Isolated fair runs above are authoritative.

| Config | Approx fair speedup | Verdict |
|--------|---------------------|---------|
| flash-attn on | 1.58x | neutral |
| flash-attn off | 1.52x | neutral |
| ubatch=512 | 1.58x | neutral |
| `-c 512/2048` | ~1.12-1.54x | neutral vs 1024 |

### General prose (`essay_100w`)

Low acceptance (~1.1/step) -> spec **slower** than vanilla. Do not use DSpark for this workload.

## Targets vs fair reality

| Workload | Target | Fair best | tgp van -> spec | Gap |
|----------|--------|-----------|-----------------|-----|
| Coding | 2.0x | **1.61x** | 25.5 -> 41.2 tok/s | ~24% |
| Agentic | 1.5x | **1.40x** | 25.6 -> 35.9 tok/s | ~7% |

A **5-10% structural gain** on coding would mean fair **1.69-1.77x** - not yet achieved.

## Per-step timing (fair, code_500l, n_max=4)

| Phase | ms/step |
|-------|---------|
| vanilla forward (1 tok) | ~39 |
| draft (fwd + GPU sample) | ~19 |
| verify (batched + sync) | ~128 |
| process (encode + inject) | ~1 |

## Experiment log

| Date | Commit | Change | Fair coding speedup |
|------|--------|--------|---------------------|
| 2026-06-29 | 05ecf8a | Batched verify + greedy accept | ~1.5x (early, spec-first) |
| 2026-06-29 | 5298fbd | Shorter draft blocks + block_gpu | +config |
| 2026-06-29 | 889ae1063 | Adaptive upscale, profiling | spec-first inflated |
| 2026-06-29 | bdf0f0308 | pp/tgp table, process sync skip | spec-first inflated |
| 2026-06-29 | 843457e | **Fair harness** (vanilla-first + cooldown) | **1.61x** honest |
| 2026-06-29 | (pending) | GPU greedy verify + defer layer D2H + verify graph warmup | re-run fair bench |

## Open work (structural only)

1. **Target verify forward** - llama graph / Vulkan kernel for small batched decode (only path to 2x)
2. **Agentic acceptance** - draft model / prompt class, not verify ms
3. **Token match at temp=0** - n_max=4 matches; n_max=3 drifts on Q4 batched path
4. **Re-benchmark** GPU greedy + defer layer + verify warmup (fair protocol above)

## How to append results

```bash
COMM=$(git rev-parse --short HEAD)
DSPARK_NO_ADAPTIVE_NMAX=1 ./build/bin/compare_vanilla_speculative \
  -m "$TARGET" -md "$DRAFT" --input-ids /tmp/dspark_eval/code_500l.json \
  --spec-type draft-dspark -c 1024 -ngl 99 -ngld 99 --temp 0 --seed 42 \
  --spec-draft-n-max 4 -n 400 2>&1 | tee /tmp/bench.out

grep -E 'generated:|throughput|tgp speedup|mean accepted|token match' /tmp/bench.out
```

Record: date, commit, prompt, n, n_max, **pp van/spec, tgp van/spec**, speedup, accept/step, match.
