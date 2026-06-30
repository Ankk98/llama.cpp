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
3. **`-c 512`** for short single-fixture runs; **`-c 8096`** for the 20-prompt suite (default in `dspark-bench-qwen3.py`).
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

## 2026-06-30: Full eval suite (20 prompts x thinking x confidence)

**Commit:** `3b6d47010` (`bench: drop MTP from Qwen3 suite`)

**Command:**

```bash
python3 scripts/dspark-bench-qwen3.py --no-cooldown
```

**Suite shape:** 10 code + 10 agentic prompts, thinking on/off, confidence
`{0.0, 0.3, 0.5, 0.7, 0.9}`, `n_max=4`, `n_predict=200`, `-c 8096`,
`temp=0`, `seed=42`, vanilla-first then DSpark per row.

**Runs:** 40 vanilla + 200 DSpark = 240 harness invocations.

**Artifacts:** per-run JSON in `scripts/dspark-vps/eval/qwen3/runs/`,
vanilla references in `expected/`, aggregate CSV in `results.csv`.
Runtime outputs are gitignored (see `.gitignore`).

### Token match (correctness)

Comparison is **token ID equality** against vanilla reference (`expected/`),
not raw UTF-8 bytes. Detokenized text matches iff every generated token ID
matches.

| Scope | Match | Rate |
|-------|-------|------|
| All 200 DSpark runs | 75 / 200 | 37.5% |
| `conf=0.0` only | 16 / 40 | 40.0% |

Mismatches are **not** confined to early tokens: median first mismatch at
gen index **85** (mean 92.6, range 3-199). Only 7/125 mismatches occur
within the first 20 generated tokens.

Early smoke (`--quick`, `n=64`, 4 prompts) showed **100% match** and
1.0-1.4x speedup. The full 20-prompt suite is the realistic picture.

### Acceptance rate and speedup by confidence

| Threshold | Accept % | Accept/step | Token match | Mean tgp speedup |
|-----------|----------|-------------|-------------|------------------|
| **0.0** | **48.7** | 1.95 | 16/40 | **1.01x** |
| 0.3 | 39.2 | 2.43 | 13/40 | 0.81x |
| 0.5 | 50.6 | 2.26 | 15/40 | 0.84x |
| 0.7 | 69.6 | 1.71 | 13/40 | 0.76x |
| 0.9 | 92.4 | 0.81 | 18/40 | 0.54x |

Higher confidence raises per-token acceptance but **truncates drafts** (0.81
tokens/step at 0.9). Verify overhead dominates; throughput drops to 0.54x.

**`conf=0.0` breakdown:**

| Slice | Accept % | Speedup | Token match |
|-------|----------|---------|-------------|
| thinking off | 58.1 | 1.13x | 10/20 |
| thinking on | 39.2 | 0.88x | 6/20 |
| code | 51.3 | 1.04x | 7/20 |
| agentic | 46.1 | 0.97x | 9/20 |

Best per-prompt speedup at `conf=0`: `code09` 1.22x (match YES). Several
fast prompts still mismatch tokens (`code04` 1.16x match NO).

Contrast with padded `code_500l` fixture (above): **94% accept, 2.03x** on
a single repetitive coding prompt. Varied real prompts do not replicate that.

### Timing: vanilla vs speculative (`conf=0.0`, means over 40 runs)

**Vanilla (target only)**

| Metric | Value |
|--------|-------|
| PP one pass (prefill prompt) | 99.4 ms |
| TGP decode phase (`n_predict=200`) | 4615 ms |
| Implied ms/token (`gen_ms / 200`) | 23.1 ms |
| PP throughput | ~390 tok/s |
| TGP throughput | 43.3 tok/s |

**DSpark speculative**

| Metric | Value |
|--------|-------|
| PP one pass | 101.5 ms (~same as vanilla) |
| TGP decode phase | 4700 ms |
| Draft per propose step | 13.2 ms |
| Verify per propose step | 54.2 ms |
| Avg ms/token effective | 23.5 ms |
| TGP throughput | 43.7 tok/s (**1.01x**) |
| Propose steps (avg) | 69.7 |
| Tokens per propose step | 2.95 |

Whole decode phase split: draft **20%** / verify **80%** of `gen_ms`
(~920 ms draft, ~3778 ms verify).

### Bottleneck analysis (`conf=0.0`)

Per speculative **propose step** (~67.4 ms total):

| Stage | ms/step | Share |
|-------|---------|-------|
| Draft model forward + CPU sampling | 13.2 | 20% |
| Target verify (total) | 54.2 | 80% |
| - logits decode (batched target fwd) | 23.1 | 43% of verify |
| - accept / sampling | 30.0 | 55% of verify |
| - feature redecode | 1.1 | ~2% |
| - process | 1.1 | ~2% |

**Verify is ~4.1x draft cost per step.** Draft is cheap; batched target
forward + accept dominates. At ~49% acceptance and ~2.0 tokens accepted per
step with `n_max=4`, extra draft work is not amortized.

At `conf > 0`, draft step rises to **~35-38 ms** (full `block_size=7`
decode + hidden states for confidence head), making throughput worse despite
higher accept %.

### Confidence head

Qwen3 draft GGUF has `enable_confidence_head=true`.

| `confidence_threshold` | Behavior |
|------------------------|----------|
| `0.0` | Confidence head **disabled** at runtime. Full block proposed. |
| `> 0` | Per-position confidence logit from draft hidden states; `sigmoid(conf)` compared to threshold; proposal truncated at first failure (`dspark_confident_prefix_length`). |

Confidence scores are **not exported** in benchmark JSON/CSV; only threshold
and resulting accept/truncation metrics appear.

### Learnings

1. **No net speedup on varied Q4 prompts yet.** Best case `conf=0` is ~1.01x
   mean tgp. Target 1.5-2x requires ~94% accept (see `code_500l`), not ~49%.
2. **Verify is the wall.** ~54 ms/step target work vs ~13 ms draft. Need
   more accepted tokens per verify to beat vanilla's ~23 ms/token.
3. **Correctness gap.** 62.5% of runs diverge from vanilla token stream.
   Thinking-on is worse (39% accept, 0.88x). Investigate batched verify vs
   vanilla sampling path on Q4.
4. **Confidence tuning is a speed trap here.** Higher threshold improves accept
   % but shrinks drafts so much that verify overhead dominates (0.54x at 0.9).
   Keep threshold `0` for throughput on Vulkan Q4.
5. **Short smoke misleads.** `n=64` on 4 prompts: match + 1.2-1.4x. Full
   suite exposes divergence and ~1.0x throughput.
6. **Prompt sensitivity.** Code prompts slightly better than agentic at `conf=0`,
   but neither class reliably hits 1.5x on real prompts.

### CSV note

Rows appended for commit `3b6d47010` have **column drift** (header written
before `draft_model_path` and other fields were added). Use per-run JSON in
`runs/` as authoritative until CSV is regenerated with a clean header.

---

## 2026-06-30: Token match investigation (why spec != vanilla?)

### Your mental model is correct

The **draft model does not define the output**. It only proposes candidate
tokens. The **target model verify step** decides what gets committed:

- At `temp=0`, verify should greedy-argmax the target logits at each position.
- Wrong draft tokens should be **rejected**; the target's token is committed
  instead.
- If verify is correct, **spec output must equal vanilla target-only output**
  token-for-token, regardless of draft quality.

So mismatches are a **verify / KV / sampling correctness bug**, not "the draft
guessed wrong."

### Root cause (confirmed): target KV corruption from batched verify

Batched verify does two things vanilla never does:

1. **Parallel target forward** - one `llama_decode()` with `[anchor, draft...]`
   to read multi-row logits in one graph.
2. **Writes draft positions into target KV**, then `llama_memory_seq_rm()` drops
   rejected tail **metadata only** (K/V tensor bytes in freed cells are not zeroed).

Stale K/V in reused cells corrupts attention over ~100+ steps. Batched row logits
then diverge from vanilla even though matmul is fine on a clean snapshot.

Evidence:

| Test | Result |
|------|--------|
| `smoke_batched_logits_repro --step 4` (clean snapshot) | all rows **MATCH** |
| `DSPARK_VERIFY_PRE_SNAP=1` | token match **YES** |
| Default batched pre-fix | **NO** (agent03 @ gen 111) |
| `DSPARK_TRACE_ORACLE` with pre-snap | every step ids match vanilla chain |

Draft KV (`ctx_dft`) is separate; bug is **target** KV not staying canonical.

### Fix (`common_speculative_dspark_verify_batched`)

**Root cause:** greedy accept used **parallel multi-token target forward** row logits.
That graph does not guarantee the same greedy chain as **one-token-at-a-time** decode
(the only correct causal semantics at temp=0). This is a design bug, not a prompt-specific
edge case.

**Default (correct):** `verify_batched()` delegates to **sequential verify** - one target
decode per token, early exit on draft mismatch, `process()` inline. Matches vanilla.

**Opt-in (experimental):** `DSPARK_VERIFY_PARALLEL=1` enables parallel multi-row forward
on a scratch sequence + `process_committed()`. Do not use for production greedy verify
until row logits are proven equivalent to sequential decode.

Scratch sequence isolation (for parallel path) prevents draft hypotheticals from touching
canonical target KV. Requires `n_parallel >= 2` when parallel verify is enabled.

Debug: `DSPARK_TRACE_KV=1`, `DSPARK_VERIFY_SEQ=1` (alias for sequential).

### Post-fix validation (CPU/Vulkan, n=200, c=8096, temp=0)

| Prompt | Default verify | `DSPARK_VERIFY_PARALLEL=1` |
|--------|----------------|----------------------------|
| `code01_think_off` | **YES** | **NO** @ gen 162 |
| `agent03_think_on` | **YES** | (not re-run) |
| `code_500l` | **YES** | (not re-run) |

Parallel path still fails on `code01_think_off` @ gen 162, confirming the diagnosis:
wrong verify semantics, not a prompt quirk.

### Trace tooling

```bash
DSPARK_TRACE_KV=1           # kv_max before/after rm and trim
DSPARK_TRACE_VERIFY=1       # batched vs sequential row logits
DSPARK_TRACE_ORACLE=1       # full accept chain vs vanilla (uses pre-snap)
build/bin/smoke_batched_logits_repro --step 4 --input-ids ...
```

### Hypotheses (final)

1. **Stale K/V in freed cells after batched draft tail** - **CONFIRMED**
2. **Batched matmul wrong with clean KV** - **RULED OUT**
3. **`defer_layer_inp_extract`** - ~0.15 logit delta, same argmax on clean KV
4. **Draft confidence at verify** - **RULED OUT** at `conf=0`

### Draft confidence scores (clarification)

The draft model **can** emit per-token confidence logits when
`confidence_threshold > 0`. They truncate the draft block only. They are **not**
used in verify and do **not** change which target tokens are accepted.

---

## Standard commands

```bash
# Generate 20 prompts x thinking on/off token fixtures
python3 scripts/gen_qwen3_eval_fixtures.py

# Full benchmark: vanilla vs DSpark (default -c 8096, not full trained ctx)
python3 scripts/dspark-bench-qwen3.py --no-cooldown

# Quick smoke (2 prompts/category, n=64)
python3 scripts/dspark-bench-qwen3.py --quick --no-cooldown
```

Qwen3-8B has no inbuilt MTP (`nextn_predict_layers=0`). This suite compares **vanilla** vs **DSpark** only.

Results append to `scripts/dspark-vps/eval/qwen3/results.csv` (flushed per row).
Vanilla expected outputs: `scripts/dspark-vps/eval/qwen3/expected/`.

```bash
TARGET=~/models/Qwen3-8B-Q4_K_M.gguf
DRAFT=~/models/dspark_qwen3_8b_block7.q4_k_m.gguf

DSPARK_NO_ADAPTIVE_NMAX=1 ./build/bin/compare_vanilla_speculative \
  -m "$TARGET" -md "$DRAFT" \
  --input-ids scripts/dspark-vps/eval/qwen3/code_500l.json \
  --spec-type draft-dspark -c 8096 --device Vulkan0 -ngl 99 -ngld 99 \
  --temp 0 --seed 42 --spec-draft-n-max 4 -n 400
```

---

## Open work

1. **Token match on full suite** - 62.5% of DSpark runs diverge from vanilla;
   thinking-on and agentic prompts worst. Batched verify vs vanilla sampling?
2. **Raise acceptance on real prompts** - ~49% at `conf=0` vs ~94% on padded
   `code_500l`. Draft quality / Q4 numerics / feature injection?
3. Real `code_500l` token fixture from Qwen3 chat template (not padded code_short).
4. Regenerate `results.csv` with stable header (column drift on `3b6d47010` rows).
5. CUDA Qwen3 verify path (inherit sequential default from Gemma).
6. Publish draft GGUF: [ankk98/dspark-qwen3-8b-block7-Q4_K_M-GGUF](https://huggingface.co/ankk98/dspark-qwen3-8b-block7-Q4_K_M-GGUF).

---

## How to append

```bash
COMM=$(git rev-parse --short HEAD)
date -Iseconds
# paste compare_vanilla_speculative stderr summary here
```
