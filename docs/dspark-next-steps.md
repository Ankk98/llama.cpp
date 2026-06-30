# DSpark Qwen3: status and next steps

Living plan after greedy verify fix (sequential default, 2026-06-30).
Branch: `ft-dspark-qwen3`.

Related docs:
- [dspark-benchmark-qwen3.md](dspark-benchmark-qwen3.md) - full benchmark log
- [dspark-refactor-kv-safety.md](dspark-refactor-kv-safety.md) - KV ownership refactor
- [dspark-refactor-pipeline.md](dspark-refactor-pipeline.md) - pipeline extraction plan

---

## Does all output match now?

**No - not yet across the full suite, but the main verify bug is fixed.**

| Status | Detail |
|--------|--------|
| Greedy verify correctness | Fixed. Parallel multi-row accept was wrong semantics at `temp=0`. Default is now sequential one-token decode. |
| Post-fix spot checks (`n=200`, `c=8096`, `conf=0`, `n_max=4`) | `code01_think_off` YES, `agent03_think_on` YES, `code_500l` YES |
| Known remaining mismatch | `code01_think_on` NO @ gen 198 when `n >= 199` (even with `DSPARK_VERIFY_SEQ=1`) - separate from verify semantics; likely thinking/EOS handling in the main spec loop |
| Full 20-prompt suite | **Not re-run post-fix.** Pre-fix run (`3b6d47010`) had **37.5% token match** (75/200 DSpark rows). Stale until regenerated. |

**Bottom line:** Correctness is restored for the verify path we fixed. We still owe a full-suite re-benchmark and one open bug on `code01_think_on` near generation boundary.

---

## Latest speedup results (post-fix, Vulkan Q4)

Measured 2026-06-30 on AMD Strix Halo iGPU after sequential-default verify.
Harness: `compare_vanilla_speculative`, `temp=0`, `seed=42`, `-ngl 99 -ngld 99`.

| Prompt | Token match | Accept % | Accept/step | tgp speedup |
|--------|-------------|----------|-------------|-------------|
| `code01_think_off` | YES | 49.6% | 1.99 | **0.82-0.84x** |
| `code01_think_on` | NO @ gen 198 | 37.2% | 1.49 | 0.80x |
| `agent03_think_on` | YES | 35.7% | 1.43 | **0.81x** |
| `code_500l` | YES | 94.6% | 3.79 | **0.91x** |

Sequential verify trades correctness for speed. Pre-fix batched verify (incorrect) reported ~1.01x mean on the full suite. **We are currently slower than vanilla on Vulkan Q4** until verify cost is reduced while keeping token match.

Contrast: padded `code_500l` with old batched verify reached **2.03x** at 94% accept. Same prompt post-fix at 94.6% accept is only **0.91x** because verify now does one target decode per accepted token.

---

## Where is the bottleneck?

Per propose step on `code01_think_off` (representative):

| Stage | ms/step | Share of step |
|-------|---------|---------------|
| Draft forward + sampling | **12.6** | 14% |
| Target verify (total) | **78.0** | 86% |
| - sequential target decodes | ~78 | dominates verify |
| - accept / argmax | ~0 | inline in decode loop |
| - feature redecode / process | small | negligible |

**Verify is ~6x draft cost per step.**

Whole decode phase (`code01_think_off`, 67 propose steps):

| Component | Total ms | Share |
|-----------|----------|-------|
| Draft | ~800 | 14% |
| Verify | ~4800 | 86% |

On high-accept `code_500l` (42 steps, 94.6% accept):

| Stage | ms/step |
|-------|---------|
| Draft | 11.7 |
| Verify | **115.2** |

Higher acceptance **increases** sequential verify cost (more target decodes per step before early exit).

---

## Forward pass vs verify pass timing

Single-token vanilla target decode (from `gen_ms / n_predict`):

| Metric | Value |
|--------|-------|
| ms/token (vanilla forward + sample) | **~25 ms** |
| tok/s (vanilla tgp) | ~39-43 |

Per speculative **propose step** (`code01_think_off`):

| Metric | Value |
|--------|-------|
| Draft block forward | **~12.6 ms** (one draft graph for up to 7 tokens) |
| Verify step | **~78 ms** (sequential: one decode per accepted token + mismatch probe) |
| Effective ms/token (spec) | **~30 ms** (90.6 ms / 2.99 tokens per step) |
| Ratio verify : single vanilla fwd | **~3.1x** per step wall time vs one vanilla token |

With ~2 tokens accepted per step, sequential verify does ~3 target decodes/step. That is roughly **3 x 25 ms = 75 ms**, matching measured ~78 ms.

**Implication:** To beat vanilla we need either (a) a correct parallel verify proven equivalent to sequential, or (b) enough accepted tokens per verify that amortized cost drops below 25 ms/token. At ~2 tokens/step and 78 ms verify, we lose ~5 ms/token vs vanilla.

---

## Prompt processing (prefill) vs vanilla

Prefill should be nearly identical: one target prefill pass, draft model not run on prompt tokens (only after first generated token).

| Metric | Vanilla | DSpark spec | Delta |
|--------|---------|-------------|-------|
| PP time (`code01_think_off`) | 116.6 ms | ~117-120 ms | ~+0-3 ms |
| PP tok/s | 368 | 336 | ~0.91x (same pass, minor overhead) |
| Full suite mean (pre-fix) | 99.4 ms | 101.5 ms | +2.1 ms |

Prefill is **not** the bottleneck. Any DSpark-specific prefill work (target layer taps for draft features) adds negligible time on short prompts.

---

## Confidence threshold: what DeepSpec uses

| Source | Default / recommendation |
|--------|--------------------------|
| DeepSpec `eval.py` | `--confidence-threshold` default **0.0** |
| DeepSpec `validate_dspark_reference.py` | default **0.0** |
| DeepSpec port plan | `confidence_threshold=0.0` = full block proposed; head active only when threshold > 0 |
| Training config `dspark_qwen3_8b.py` | `confidence_head_alpha=1.0` (train the head); inference threshold still **0.0** |
| DeepSpec `draft_ops.py` | `threshold <= 0.0` -> proposal length = full `block_size` (7) |

**Recommendation for llama.cpp throughput:** keep **`--dspark-confidence-threshold 0.0`**.

Higher thresholds truncate drafts early (`sigmoid(conf) < threshold`). Our Vulkan sweep showed:

| Threshold | Accept/step | tgp speedup |
|-----------|-------------|-------------|
| 0.0 | 1.95 | 1.01x (pre-fix batched) |
| 0.5 | 2.26 | 0.84x |
| 0.9 | 0.81 | 0.54x |

Higher threshold raises per-token acceptance but forces full-block draft forwards for confidence scoring and shrinks effective drafts. Net throughput drops. Use threshold > 0 only for experiments or when verify is cheap enough to absorb the tradeoff.

---

## Prioritized task list

### P0 - Correctness (must finish before claiming production-ready)

| # | Task | Details | Done when |
|---|------|---------|-----------|
| 1 | **Re-run full 20-prompt suite post-fix** | `python3 scripts/dspark-bench-qwen3.py --no-cooldown --confidence 0.0`; regenerate `expected/` if verify fix changed outputs | 200/200 DSpark rows token match YES at `conf=0` (or documented exceptions) |
| 2 | **Fix `code01_think_on` gen-198 mismatch** | Fails at last token when `n >= 199`; reproduces with sequential verify. Check thinking token handling, EOS/stop, and `n_predict` boundary in `run_speculative()` | Token match YES for `n=200` |
| 3 | **Add CI/canary token-match gate** | Quick harness on 4-5 prompts (`code01`, `agent03`, `code_500l`) in smoke or nightly | Fails build on token mismatch at `temp=0` |

### P1 - Performance (correct verify that beats vanilla)

| # | Task | Details | Done when |
|---|------|---------|-----------|
| 4 | **Prove parallel verify equivalence** | Row-by-row logits vs sequential chain on scratch seq; mandatory gate before enabling parallel for greedy | `smoke_batched_logits_repro` + harness oracle pass on full suite with `DSPARK_VERIFY_PARALLEL=1` |
| 5 | **Logits-only / no-KV-write verify forward** | Upstream or local API: multi-row logits without writing draft positions to target KV (see kv-safety doc) | Verify ms/step drops toward single batched forward (~25 ms) with token match YES |
| 6 | **Populate sequential verify timing** | `verify_sequential()` does not fill `common_speculative_dspark_verify_timing`; harness shows 0 for sub-breakdown | Accurate ms/decode in JSON/CSV for profiling |
| 7 | **CUDA verify path** | Confirm sequential default on NVIDIA VPS; compare against Gemma numbers | Documented speedup + token match on H100/L40 |

### P2 - Draft quality and tuning

| # | Task | Details | Done when |
|---|------|---------|-----------|
| 8 | **Raise acceptance on real prompts** | ~35-50% accept on varied prompts vs ~94% on `code_500l`. Check feature injection, Q4 numerics, Markov on GPU | Mean accept/step > 2.5 on suite at `conf=0` without mismatch |
| 9 | **n_max sweep post-fix** | Re-sweep `n_max` 3-7 with sequential verify; old sweet spot may shift | Best `n_max` documented with speedup + match |
| 10 | **Real long coding fixture** | Replace padded `code_500l` with true ~500-token Qwen3 chat-template prompt | Fixture in eval suite |

### P3 - Engineering / release

| # | Task | Details | Done when |
|---|------|---------|-----------|
| 11 | **Pipeline refactor Phase A** | Extract `dspark_pipeline_step()` per pipeline doc; single path for harness + server | No duplicated propose/verify/commit logic |
| 12 | **KV safety asserts** | Debug builds assert canonical seq untouched by draft hypotheticals | Assert fires on regression |
| 13 | **Regenerate clean `results.csv`** | Fix column drift from `3b6d47010` rows | Single stable header |
| 14 | **Publish draft GGUF** | [ankk98/dspark-qwen3-8b-block7-Q4_K_M-GGUF](https://huggingface.co/ankk98/dspark-qwen3-8b-block7-Q4_K_M-GGUF) | Public weights + README |
| 15 | **Update benchmark doc verify section** | `dspark-benchmark-qwen3.md` still says batched defer is default | Docs match sequential-default behavior |

### P4 - Future (after P0-P1)

| # | Task | Details |
|---|------|---------|
| 16 | Batch draft feature inject on multi-token commits | Amortize target layer tap writes |
| 17 | `min_verify_tokens=2` skip | Skip verify when draft block is 1 token |
| 18 | Server integration | Wire DSpark path through `llama-server` per server-dev scope |
| 19 | Temperature > 0 rejection sampling | Phase 3c from port plan |

---

## Recommended immediate sequence

1. Commit sequential-default verify fix (if not already).
2. Run task **#1** (full suite) - establishes new correctness baseline.
3. Parallel task **#2** (`think_on` boundary bug).
4. Start **#4** / **#5** in parallel - only path to >1x with correct greedy output.
5. Do not tune confidence threshold for speed until verify is cheap (threshold 0.0 stays default).

---

## How to re-run key measurements

```bash
# Full suite (post-fix baseline)
python3 scripts/dspark-bench-qwen3.py --no-cooldown --confidence 0.0

# Single prompt with timing JSON
TARGET=~/models/Qwen3-8B-Q4_K_M.gguf
DRAFT=~/models/dspark_qwen3_8b_block7.q4_k_m.gguf
DSPARK_NO_ADAPTIVE_NMAX=1 DSPARK_BENCH_NO_COOLDOWN=1 \
./build/bin/compare_vanilla_speculative \
  -m "$TARGET" -md "$DRAFT" \
  --input-ids scripts/dspark-vps/eval/qwen3/code/code01_think_off.json \
  --spec-type draft-dspark -c 8096 --device Vulkan0 -ngl 99 -ngld 99 \
  --temp 0 --seed 42 --spec-draft-n-max 4 -n 200 \
  --json-results /tmp/dspark_run.json
```
