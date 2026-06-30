# DSpark Qwen3-8B benchmark log (Vulkan / Strix Halo)

Living append log for Qwen3 DSpark speculative decoding on llama.cpp.
Branch: `ft-dspark-qwen3` (from `ft-dspark`).

## Environment

| Item | Value |
|------|-------|
| Machine | AMD Strix Halo iGPU (RADV STRIX_HALO) |
| Target | `Qwen3-8B-Q4_K_M.gguf` (~4.7 GB) |
| Draft | `dspark_qwen3_8b_block7.q4_k_m.gguf` (~1.5 GB) |
| Backend | Vulkan (`-ngl 99 -ngld 99`, `--device Vulkan0`) |
| Harness | `build/bin/compare_vanilla_speculative` |
| Eval | `scripts/dspark-vps/eval/qwen3/*.json` |

Gemma4 reference numbers: [dspark-benchmark-experiments.md](dspark-benchmark-experiments.md).

### Fair protocol

1. Vanilla first (cool GPU), 3s cooldown, then speculative (`DSPARK_BENCH_NO_COOLDOWN=1` to skip).
2. Fixed `n_max` (`DSPARK_NO_ADAPTIVE_NMAX=1`).
3. **Always `-c 512`** for these prompts.
4. `temp=0`, `seed=42` for token-match checks.
5. Report **tgp** speedup (decode throughput).

### Verify path (2026-06-30)

- **Vulkan/CPU:** batched defer verify is now default at `temp=0` (was sequential via env default).
  Opt out: `DSPARK_VERIFY_SEQ=1`.
- **CUDA:** sequential verify remains default (`DSPARK_VERIFY_FAST_FORCE=1` to opt into batched).
- **Harness fix:** vanilla Vulkan prefill now matches spec (full prompt + first-token sample).
  Old `inp.size()-1` prefill caused false mismatches (e.g. agentic @ gen 78).

---

## 2026-06-30: Phase 4 bring-up

**Smoke:** `smoke_phase4_qwen3 --draft-model $DRAFT` PASS after optional `rope_freqs` for Qwen3 path.

**Draft GGUF metadata:** `dspark.attention_k_eq_v=false`, `target_layers=[2,10,18,26,34]`,
`block_size=7`, `mask_token_id=151669`.

### Short prompts (`n_predict=128`, `DSPARK_BENCH_NO_COOLDOWN=1`, default batched verify)

| Fixture | n_max | Accept % | Accept/step | tgp speedup | Match |
|---------|-------|----------|-------------|-------------|-------|
| general | 4 | 48.3 | 1.93 | 1.06x | NO (gen 27) |
| general | 5 | 43.5 | 2.17 | 1.09x | YES |
| code_short | 4 | 48.3 | 1.93 | 1.04x | YES |
| code_short | 5 | 41.4 | 2.07 | 1.02x | YES |
| agentic | 4 | 47.7 | 1.91 | 1.06x | NO |
| agentic | 5 | 39.1 | 1.95 | 1.00x | YES |

Short prompts are verify-overhead limited (~50 ms verify vs ~44 ms vanilla/token).
Use `n_max=5` when `n_max=4` fails token match on a given fixture.

### Coding (`code_500l`, 92 tokens, `n_predict=400`)

Synthetic fixture (code_short-derived, 92 tokens). Real long coding prompt TBD.

**Fair run** (vanilla-first + 3s cooldown, `n_max=4`):

| tgp van | tgp spec | Speedup | Accept % | Accept/step | Verify ms | Match |
|---------|----------|---------|----------|-------------|-----------|-------|
| 42.1 | 85.3 | **2.03x** | 94.3 | 3.77 | 44 | YES |

**n_max sweep** (`DSPARK_BENCH_NO_COOLDOWN=1`):

| n_max | Speedup | Accept % | Accept/step | Match |
|-------|---------|----------|-------------|-------|
| 3 | 1.78x | 97.1 | 2.91 | YES |
| 4 | 1.80x | 94.3 | 3.77 | YES |
| 5 | 1.95x | 95.1 | 4.76 | YES |
| 6 | 1.87x | 88.8 | 5.33 | YES |

Sweet spot: **`n_max=4` or `5`**, threshold `0` (confidence off).

### Task classes (`n_predict=300`, `n_max=5`, no cooldown)

| Fixture | Accept % | Accept/step | tgp speedup | Match |
|---------|----------|-------------|-------------|-------|
| general | 47.8 | 2.39 | 1.06x | YES |
| code_short | 41.6 | 2.08 | 0.99x | YES |
| agentic | 31.6 | 1.58 | 0.81x | YES |

Agentic acceptance is low on Q4 Vulkan; draft quality / numerics likely limiter.

### Confidence scheduling (`code_500l`, `n_max=4`, `n=400`, no cooldown)

| Threshold | Speedup | Accept/step | Match |
|-----------|---------|-------------|-------|
| 0.0 | 1.68x | 3.77 | YES |
| 0.5 | 1.65x | 6.39 | YES |
| 0.7 | 1.66x | 6.39 | YES |
| 0.9 | 1.61x | 6.35 | YES |

Higher threshold raises accept/step but not net tgp (full `block_size=7` draft forward cost).
Keep threshold `0` for throughput on Vulkan Q4.

---

## Standard commands

```bash
TARGET=~/models/Qwen3-8B-Q4_K_M.gguf
DRAFT=~/models/dspark_qwen3_8b_block7.q4_k_m.gguf

# Fair coding headline (target 1.5-2x)
DSPARK_NO_ADAPTIVE_NMAX=1 ./build/bin/compare_vanilla_speculative \
  -m "$TARGET" -md "$DRAFT" \
  --input-ids scripts/dspark-vps/eval/qwen3/code_500l.json \
  --spec-type draft-dspark -c 512 --device Vulkan0 -ngl 99 -ngld 99 \
  --temp 0 --seed 42 --spec-draft-n-max 4 -n 400

# Quick iteration (skip cooldown)
DSPARK_NO_ADAPTIVE_NMAX=1 DSPARK_BENCH_NO_COOLDOWN=1 ./build/bin/compare_vanilla_speculative ...
```

---

## Open work

1. Real `code_500l` token fixture from Qwen3 chat template (not padded code_short).
2. `n_max=4` token match on `general.json` short run (batched verify edge case).
3. Agentic acceptance / speedup (31% hit rate on Q4).
4. CUDA Qwen3 verify path (inherit sequential default from Gemma).
5. Publish draft GGUF: [ankk98/dspark-qwen3-8b-block7-Q4_K_M-GGUF](https://huggingface.co/ankk98/dspark-qwen3-8b-block7-Q4_K_M-GGUF).

---

## How to append

```bash
COMM=$(git rev-parse --short HEAD)
date -Iseconds
# paste compare_vanilla_speculative stderr summary here
```
