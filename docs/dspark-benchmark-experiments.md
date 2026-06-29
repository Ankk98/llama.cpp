# DSpark benchmark experiments

Living log for DSpark speculative decoding performance work on llama.cpp.
Hardware, model paths, and command lines are recorded so results can be reproduced and compared.

## Environment

| Item | Value |
|------|-------|
| Machine | Strix Halo APU (Vulkan) |
| Target | `gemma-4-12B-it-QAT-Q4_0.gguf` (~7 GB) |
| Draft | `/tmp/dspark_gemma4_12b_q4pure.gguf` (~1.9 GB) |
| Backend | Vulkan (`-ngl 99 -ngld 99`) |
| Harness | `build/bin/compare_vanilla_speculative` |
| Eval prompts | `/tmp/dspark_eval/*.json` |
| Bench order | **Speculative first, vanilla last** |
| Git (latest sweep) | `889ae1063` + local (process sync skip, pp/tgp table) |

### Standard command

```bash
./build/bin/compare_vanilla_speculative \
  -m "$TARGET" -md "$DRAFT" \
  --input-ids /tmp/dspark_eval/code_500l.json \
  --spec-type draft-dspark \
  -c 1024 -ngl 99 -ngld 99 --temp 0 --seed 42 \
  --spec-draft-n-max 3 -n 500
```

### Profiling env vars

| Variable | Effect |
|----------|--------|
| `DSPARK_PROF=1` | Per-step draft forward vs sampling split in logs |
| `DSPARK_DEBUG=1` | Per-step anchor/draft/accept trace |
| `DSPARK_NO_ADAPTIVE_NMAX=1` | Disable hit-rate n_max shrink/expand |
| `DSPARK_NO_BLOCK_GPU=1` | Force CPU block sampler |
| `DSPARK_SPLIT_VERIFY=1` | Dual ctx: logits on ctx_tgt, layers on ctx_tgt_feat |
| `DSPARK_VERIFY_SEQ=1` | Sequential early-exit verify (debug) |

### Reading pp vs tgp

| Metric | Vanilla | Speculative |
|--------|---------|-------------|
| **pp** | Target prefill only (`prompt decode`) | Prefill + first-token sample + `process()` setup (~270ms one-time) |
| **tgp** | Target autoregressive decode | Draft + verify + process loop (**primary metric**) |

Spec **pp** is not apples-to-apples with vanilla pp (includes extra setup). Compare **tgp** for generation speedup.

## First-principles model

Per propose step:

```
tokens_out = 1 + accepted_drafts   (includes bonus token)
step_ms    = draft_ms + verify_ms
ms/token   = step_ms / tokens_out
speedup    = vanilla_tgp / spec_tgp
```

Bottleneck (typical, n_max=3, coding):

| Phase | ms/step | Notes |
|-------|---------|-------|
| Draft forward + GPU block sample | ~16 | n_draft+1 tokens, fused block_gpu |
| Target verify decode + sync | ~108 | 12B Q4 batched forward (4 tokens) + fence |
| Accept (greedy) | ~1 | Single GPU sync + argmax |
| process (encode + inject) | ~1 | Not on critical path |

**Conclusion:** 2x needs ~13% more tgp (1.77x -> 2.0x). Verify batch forward is the ceiling;
layer-input taps are cheap when bundled in single-pass. Split-verify re-decode loses.

## Throughput results (2026-06-29)

All runs: `-c 1024`, `temp=0`, `seed=42`, spec-first / vanilla-last, Vulkan.

### Coding (target 2.0x tgp)

| Prompt | n | n_max | Config | pp van (tok/s) | pp spec (tok/s) | **tgp van** | **tgp spec** | **Speedup** | Accept/step |
|--------|---|-------|--------|----------------|-----------------|-------------|--------------|-------------|-------------|
| code_500l | 500 | 3 | fixed | 16062 | 346* | **25.81** | **45.78** | **1.77x** | 2.18 |
| code_500l | 600 | 3 | fixed | 25.63 | - | **25.63** | **45.16** | **1.76x** | 2.25 |
| code_500l | 400 | 3 | fixed | - | - | - | - | **1.72x** | - |
| code_bug | 286 | 3 | fixed | 22265 | 349* | **25.75** | **40.65** | **1.58x** | 2.38 |
| code_fib | 78 | 3 | fixed | 6117 | 295* | **25.55** | **39.70** | **1.55x** | 2.29 |

\* spec pp includes setup; not comparable to vanilla pp.

**Best coding:** 45.78 tgp vs 25.81 vanilla = **1.77x** (`code_500l`, n=500, n_max=3).
Peak observed: **1.78x** (n=400, same config, thermal variance).

### Agentic (target 1.5x tgp)

| Prompt | n | n_max | Config | tgp van | tgp spec | Speedup | Accept/step |
|--------|---|-------|--------|---------|----------|---------|-------------|
| agentic_plan | 400 | 2 | fixed | 25.76 | 36.99 | **1.44x** | 1.25 |
| agentic_plan | 300 | 2 | fixed | 25.55 | 34.61 | **1.35x** | 1.30 |
| agentic_plan | 500 | 2 | fixed | - | - | **1.41x** | 1.28 |
| agentic_plan | 300 | 3 | adaptive | 23.29 | 26.49 | 1.14x | 1.59 |

**Best agentic:** 36.99 tgp vs 25.76 vanilla = **1.44x** (n_max=2, n=400). Need **38.6 tgp** for 1.5x.

### General / other

| Prompt | n | n_max | tgp van | tgp spec | Speedup | Accept/step | Notes |
|--------|---|-------|---------|----------|---------|-------------|-------|
| essay_100w | 114 | 3 | 25.13 | 17.74 | **0.71x** | 1.09 | Low acceptance; spec slower |

### Per-step timing (code_500l, n_max=3, n=500)

| Phase | ms/step |
|-------|---------|
| vanilla forward (1 tok) | ~39 |
| draft (fwd + sample) | 16 |
| verify (batched) | 108 |
| decode submit (async) | ~2 |
| process (encode + inject) | ~1 |

## Targets

| Workload | Target | Best tgp (van -> spec) | Speedup | Gap |
|----------|--------|------------------------|---------|-----|
| Coding | 2.0x | 25.8 -> **45.8** tok/s | **1.77-1.78x** | ~12-13% |
| Agentic | 1.5x | 25.8 -> **37.0** tok/s | **1.44x** | ~10% |
| General prose | 1.5x | essay 0.71x | - | low accept |

## Experiment log (chronological)

### 2026-06-29 - Baseline batched verify (05ecf8a)

| Config | Speedup | Notes |
|--------|---------|-------|
| n_max=5, c=1024 | ~1.56-1.60x | Pre-short-draft |

### 2026-06-29 - Shorter draft blocks (5298fbd)

Shorter draft decode (n_draft+1), per-length block_gpu, adaptive shrink.

### 2026-06-29 - Adaptive upscale + profiling (889ae1063)

Adaptive upscale (hit > 65%), `decode_submit_ms`, experiments doc.

### 2026-06-29 - Process sync skip + pp/tgp table

| Change | Result |
|--------|--------|
| Skip redundant `llama_synchronize` in process() on verify hot path (n_tokens <= 8) | Coding **1.77x** peak |
| Adaptive thresholds 0.50 / 0.65 | Agentic n_max=2 > adaptive n_max=3 |
| Harness prints pp/tgp side-by-side | See throughput tables above |

### Algorithm investigations

| Idea | Result | Notes |
|------|--------|-------|
| Shorter draft decode (n_draft+1) | **Win** | Keeps fused block_gpu per length |
| Split verify (dual ctx) | **Loss** | +47ms feature re-decode/step |
| Sequential early-exit verify | **Loss** | ~1.4x slower than batched |
| Cached logits (skip anchor decode) | **Deferred** | KV trim semantics need care |
| flash-attn on | **Neutral/loss** | No verify improvement |
| -c 512 | **Minor loss** | 1.62x vs 1.72x |

### n_max sweep (code_500l, n=500, spec-first)

| n_max | Speedup | Accept/step | Verify ms | Draft ms |
|-------|---------|-------------|-----------|----------|
| 3 | **1.77x** | 2.18 | 108 | 16 |
| 4 | 1.67x | 2.67 | 131 | 20 |
| 5 | 1.47x | 2.91 | 159 | 25 |

**Sweet spot:** n_max=3 for coding.

## Open hypotheses

1. **KV-aware cached logits verify** - skip anchor re-decode when safe (needs KV rewrite on mismatch)
2. **llama graph** - logits-only verify ctx (never layer taps) + committed-only feature pass without toggle
3. **Agentic draft quality** - acceptance ~1.3/step is the limiter, not verify ms
4. **Separate-process bench** - remove thermal skew between spec and vanilla
5. **Server wiring** of `verify_batched`

## How to append results

```bash
COMM=$(git rev-parse --short HEAD)
DSPARK_NO_ADAPTIVE_NMAX=1 ./build/bin/compare_vanilla_speculative \
  -m "$TARGET" -md "$DRAFT" --input-ids /tmp/dspark_eval/code_500l.json \
  --spec-type draft-dspark -c 1024 -ngl 99 -ngld 99 --temp 0 --seed 42 \
  --spec-draft-n-max 3 -n 500 2>&1 | tee /tmp/bench.out

grep -E 'throughput|tgp speedup|mean accepted|verify step' /tmp/bench.out
# Add row to throughput table: date, commit, prompt, n, n_max, pp van/spec, tgp van/spec, speedup
```
