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
| Bench order | **Speculative first, vanilla last** (user preference) |

### Standard command

```bash
./build/bin/compare_vanilla_speculative \
  -m "$TARGET" -md "$DRAFT" \
  --input-ids /tmp/dspark_eval/code_500l.json \
  --spec-type draft-dspark \
  -c 1024 -ngl 99 -ngld 99 --temp 0 --seed 42 \
  --spec-draft-n-max 4 -n 400
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

## First-principles model

Per propose step:

```
tokens_out = 1 + accepted_drafts   (includes bonus token)
step_ms    = draft_ms + verify_ms
ms/token   = step_ms / tokens_out
speedup    = vanilla_ms_per_token / ms_per_token
```

Bottleneck (typical, n_max=4, coding):

| Phase | ms/step | Notes |
|-------|---------|-------|
| Draft forward + GPU block sample | ~18-24 | Scales with n_draft+1 (capped block) |
| Target verify decode | ~55-75 | 12B batched forward + 5 layer-input taps |
| Accept (greedy) | ~0.5-2 | Single GPU sync + argmax |
| process (encode + inject) | ~1-2 | Not on critical path |

**Conclusion:** 2x needs either ~40% lower verify cost or ~50% more tokens per step at same verify cost.
Split verify (logits ctx + feature re-decode) loses vs single-pass bundled forward on this hardware.

## Experiment log

### 2026-06-29 - Baseline after batched verify (commit 05ecf8a)

| Config | Prompt | n | Speedup | Accept/step | Match | Notes |
|--------|--------|---|---------|-------------|-------|-------|
| n_max=5, c=1024 | code_500l | 600 | ~1.56-1.60x | ~3.08 | sometimes NO | Best pre-short-draft |
| split verify | code_500l | 300 | ~1.04x | - | NO | logits+feature re-decode slower |

### 2026-06-29 - Shorter draft blocks (commit 5298fbd)

| Config | Prompt | n | Speedup | Accept/step | Match |
|--------|--------|---|---------|-------------|-------|
| n_max=4, adaptive | code_500l | 250 | 1.52x | 2.21 | YES |
| n_max=4, no adaptive | code_500l | 400 | 1.61x | 2.50 | YES |
| n_max=4 | code_bug | 250 | 1.52x | 2.55 | YES |
| n_max=3 | code_500l | 300 | 1.74x* | 2.05 | YES | *spec-first cold GPU |
| n_max=4 | agentic_plan | 250 | 1.26x | 1.19 | NO |
| n_max=4 | agentic_plan (prior) | 300 | 0.91x | 1.55 | NO | Before adaptive |

### 2026-06-29 - Algorithm investigations

| Idea | Result | Notes |
|------|--------|-------|
| Shorter draft decode (n_draft+1) | **Win** | Keeps fused block_gpu per length |
| Partial block without block_gpu | **Loss** | CPU chain; reverted |
| Split verify (dual ctx) | **Loss** | +47ms feature re-decode/step |
| Sequential early-exit verify | **Loss** | ~1.4x slower than batched |
| Anchor-elided verify | **N/A** | KV crop semantics require anchor at pos_verify |
| Vanilla-first bench order | Fairer | User prefers spec-first; doc records both |

### 2026-06-29 - Adaptive upscale + verify profiling (post 5298fbd)

| Config | Prompt | n | Speedup | Accept/step | Verify ms | Match |
|--------|--------|---|---------|-------------|-----------|-------|
| n_max=4 adaptive | code_500l | 400 | **1.66x** | 2.38 | 119 | NO |
| n_max=3 fixed | code_500l | 500 | **1.74x** | 2.18 | 108 | NO |
| n_max=4 fixed | code_500l | 500 | 1.67x | 2.67 | 131 | NO |
| n_max=5 fixed | code_500l | 500 | 1.47x | 2.91 | 159 | NO |
| n_max=4 adaptive | code_500l | 500 | 1.70x | 2.54 | 123 | NO |
| n_max=4 adaptive | agentic_plan | 300 | 1.16x | 1.27 | 124 | NO |
| n_max=2 adaptive | agentic_plan | 300 | 1.24x | 1.30 | 151 | NO |

**Profiling insight:** `decode_submit_ms` ~2-3ms (async return). Almost all verify time is GPU
compute + fence in accept (~120ms for 4-5 token batch on 12B Q4 Vulkan). Layer-input taps are
bundled cheaply in single-pass; split-verify re-decode loses.

**Peak coding:** 1.74x (`n_max=3`, n=500, spec-first). Target 2.0x still needs ~15% more throughput.

### n_max sweep (code_500l, n=500, no adaptive, spec-first)

| n_max | Speedup | Accept/step | Verify ms | Draft ms |
|-------|---------|-------------|-----------|----------|
| 3 | 1.74x | 2.18 | 108 | 16 |
| 4 | 1.67x | 2.67 | 131 | 20 |
| 5 | 1.47x | 2.91 | 159 | 25 |

Sweet spot for coding: **n_max=3** (lowest verify cost, same peak speedup as n_max=4).

## Targets

| Workload | Target | Best so far | Gap |
|----------|--------|-------------|-----|
| Coding | 2.0x | **1.74x** (n_max=3, n=500) | ~15% - verify batch cost |
| Agentic | 1.5x | 1.24x (n_max=2 adaptive) | low acceptance (~1.3/step) |
| General | 1.5x | ~1.5-1.7x coding prompts | agentic lags |

## Open hypotheses (next experiments)

1. **Adaptive n_max upscale** (hit > 72% -> full n_max): test on long code runs
2. **Finer verify profiling**: decode_submit vs sync vs layer-extract (decode_submit_ms added)
3. **n_max sweep** 3/4/5 with `DSPARK_NO_ADAPTIVE_NMAX=1` on each eval prompt
4. **Context size sweep** (-c 512/1024/2048): minor effect observed
5. **llama graph**: layer-input taps only for committed rows (needs ctx changes, no per-step toggle)
6. **Confidence threshold** without hidden-state read path - likely not viable
7. **Server wiring** of verify_batched for production path

## How to append results

After a benchmark run, add a row to the experiment log with: date, git commit, config flags, prompt, n_predict, speedup, accept/step, token match, and per-step timing if available.

```bash
git rev-parse --short HEAD
DSPARK_NO_ADAPTIVE_NMAX=1 ./build/bin/compare_vanilla_speculative ... 2>&1 | tee /tmp/bench.out
grep -E 'speedup|mean accepted|draft |verify |decode submit' /tmp/bench.out
```
