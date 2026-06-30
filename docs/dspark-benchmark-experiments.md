# DSpark benchmark experiments

Living log for DSpark speculative decoding performance on llama.cpp.
**Fair methodology only** - numbers below use vanilla-first + 3s cooldown unless noted.

**NVIDIA CUDA results:** see [dspark-benchmark-nvidia.md](dspark-benchmark-nvidia.md) (RTX 3090). Post-fix (2026-06-29): sequential verify restores token match on CUDA (~0.90x); batched verify still diverges at gen 54+.

**Upstream vLLM comparison:** see [dspark-vllm-pr46995-comparison.md](dspark-vllm-pr46995-comparison.md).

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

Use **`-c 512`** for this workload (~92-token prompt). `-c 1024` allocates 2x KV for
both target and draft and slows verify ~35% with no benefit here. **Never omit `-c`**:
default `n_ctx=0` loads the model's trained context (262144 for Gemma4) and OOMs.

```bash
DSPARK_NO_ADAPTIVE_NMAX=1 ./build/bin/compare_vanilla_speculative \
  -m "$TARGET" -md "$DRAFT" \
  --input-ids /tmp/dspark_eval/code_500l.json \
  --spec-type draft-dspark \
  -c 512 -ngl 99 -ngld 99 --temp 0 --seed 42 \
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
| `DSPARK_FUSED_ARGMAX=1` | In-graph per-row argmax (experimental; match NO on Vulkan) |
| `DSPARK_VERIFY_SEQ=1` | Force sequential verify (CUDA default for temp=0) |
| `DSPARK_VERIFY_BATCHED=1` | Opt-in batched verify (CUDA temp=0: diverges gen 54+) |
| `DSPARK_PREFILL_DEFER=1` | Defer layer D2H during spec prefill (match YES on CUDA) |
| `DSPARK_FAST_PREFILL=1` | Two-pass n-1 prefill (experimental; match NO, slower pp on CUDA) |

### Safe profiling (avoid OOM)

| Do | Don't |
|----|-------|
| `-c 512`, `-n 20` for quick checks | `GGML_VK_PERF_LOGGER=1` on full `compare_vanilla_speculative` (dual model + vanilla+spec) |
| `llama-bench -d 512 -p 5` for isolated 5-token forward | `-c 0` or omit `-c` on harnesses without the 512 default guard |
| `DSPARK_BENCH_NO_COOLDOWN=1` for iteration | `-c 1024` unless you need long context |

Isolated Vulkan forward (single model):

```bash
./build/bin/llama-bench -m "$TARGET" -ngl 99 -fa off -d 512 -p 5 -r 3
GGML_VK_PERF_LOGGER=1 ./build/bin/llama-bench -m "$TARGET" -ngl 99 -fa off -d 512 -p 5 -r 1 --no-warmup
```

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
| `-c 512` vs `1024` | **real** | Same prompt: verify 69 vs 92 ms; **1.57x vs 1.19x** speedup |
| `n_max=7` on coding | **0.96x, match NO** | Quality drift |

### DeepSpec reference alignment (2026-06-29)

Compared against `~/repos/DeepSpec` (`draft_ops.py`, `base_evaluator.py`, Gemma4 config).

| Topic | DeepSpec | llama.cpp | OK? |
|-------|----------|-----------|-----|
| Confidence at default | `threshold=0` -> full block, head unused | Same (`use_confidence` only if `threshold > 0`) | **YES** |
| Confidence features | `concat(hidden, markov_w1(prev))` | `dspark_confidence_logit()` same layout | **YES** |
| Truncation | first `sigmoid(conf) < threshold` | `dspark_confident_prefix_length()` | **YES** |
| Markov + softcap | autoregressive `w2(w1(prev))` + tanh softcap | CPU/GPU paths match | **YES** |
| Verify | single batched target forward + rejection | `verify_batched()` + greedy at temp=0 | **YES** |
| `block_size` / n_max | always 7 at eval | default `n_max=4`, adaptive optional | **diff** (perf tuning) |
| Draft forward | full `block_size` tokens | `n_draft+1` when confidence off | **diff** (optimization) |
| Target dtype | bf16 eval | Q4 GGUF | **diff** (acceptance drift) |

Confidence head weights are in the draft GGUF (`enable_confidence_head=true`,
`confidence_head_with_markov=true`). We are **not** misusing it; at threshold 0 it is
correctly skipped (same as DeepSpec eval defaults).

### Bottleneck (measured 2026-06-29, `-c 512 -d 512`)

**Vulkan isolated target forward** (`llama-bench`, flash-attn off, Strix Halo):

| Test | ms | Notes |
|------|-----|-------|
| 1-token TG @ d512 | **~43** | 23 tok/s |
| 5-token PP @ d512 (batched) | **~70** | 71 tok/s for 5 tokens |
| 5x1-token TG @ d512 (sequential) | **~212** | 23 tok/s x 5 |

Batched 5-token is **~3x** faster than 5 sequential singles (attention/matmul reuse).

**Single 5-token graph VK breakdown** (~70 ms GPU, no layer taps):

| Component | ms | Share |
|-----------|-----|-------|
| Q4 `MUL_MAT_VEC` (layers) | ~46 | 66% |
| LM head (`q6_K` vocab) | ~4 | 5% |
| RMS/ROPE/norm | ~4 | 5% |
| Other | ~16 | 23% |

**DSpark verify step** (`n_max=4`, defer layer on, `-c 512`):

| Phase | ms/step | Notes |
|-------|---------|-------|
| decode submit (async return) | ~3 | `llama_decode` returns before GPU done |
| **GPU forward + logits fence** | **~65** | Dominates; labeled "accept" in harness |
| layer commit (partial D2H) | ~0.2 | Deferred path |
| process (encode + inject) | ~1.3 | Not the limiter |
| draft step | ~18 | Separate from verify |

At `-c 1024`, verify rises to **~92 ms** (longer KV attention) with no gain for this prompt.

**2-3x coding target math (from ~1.62x fair):**

| Target | Required | Realistic path |
|--------|----------|----------------|
| **2.0x** | +23% tgp | ~3.1 accept/step at same verify ms, OR verify 60->46 ms (-23%) |
| **3.0x** | +85% tgp | Both: verify ~40 ms AND accept ~3.5+/step, OR draft model quality leap |

Config tuning alone will not reach 2x. Need **target batched-decode kernel/graph work**
and/or **higher acceptance without quality loss**.

## Fair throughput results

Default fair settings: `-c 512`, `temp=0`, `seed=42`, vanilla-first + 3s cooldown,
`DSPARK_NO_ADAPTIVE_NMAX=1`.

### Coding (`code_500l`, n_predict=400)

| Config | tgp van | tgp spec | **Speedup** | Accept/step | Verify ms | Match |
|--------|---------|----------|-------------|-------------|-----------|-------|
| **defer layer, `-c 512`** | 25.1 | **39.5** | **1.57x** | 2.50 | 69 | **YES** |
| defer layer, `-c 1024` | 25.0 | 29.6 | 1.19x | 2.50 | 92 | YES |
| defer layer (earlier, c1024) | ~26.1 | ~42.2 | 1.62x | 2.50 | ~64 | YES |
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
| Coding | 2-3x | **1.57x** (`-c 512`) | 27% | 91% |
| Agentic | 1.5x | 1.40x | n/a | n/a |

## Experiment log

| Date | Commit | Change | Fair coding speedup |
|------|--------|--------|---------------------|
| 2026-06-29 | 843457e | Fair harness (vanilla-first + cooldown) | 1.61x |
| 2026-06-29 | f360dae | GPU greedy + defer layer + warmup (defer buggy) | regressed |
| 2026-06-29 | 33fc593 | Fix defer sync; timing; GPU greedy opt-in | **1.62x** |
| 2026-06-29 | (wip) | Defer cleanup fix; DeepSpec audit; VK profile; `-c 512` | **1.57x** |

### 2026-06-29 session: 3x attempts + profiling

| Experiment | tgp speedup | Accept/step | Match | Notes |
|------------|-------------|-------------|-------|-------|
| CPU greedy, `-c 512`, n=400 | **1.57x** | 2.50 | YES | verify ~69ms; recommended fair config |
| CPU greedy, `-c 1024`, n=400 | 1.19x | 2.50 | YES | verify ~92ms; extra KV hurts |
| `DSPARK_FUSED_ARGMAX=1` | 0.44x | 0.98 | NO | in-graph argmax wrong tokens on Vulkan |
| `DSPARK_VERIFY_SEQ=1` | 0.88x | 2.50 | NO | 1 decode/token >> batched savings |
| `GGML_VK_PERF_LOGGER` on compare | OOM | - | - | dual model + logger; do not use |

**3x conclusion:** batched verify GPU graph is ~70ms (66% matmul). Need ~40ms graph AND/OR
~3.5 accept/step. Fused argmax saves little vs forward. Use `-c 512` not 1024 for coding bench.

### 2026-06-29: confidence scheduling sweep

**Why not tried earlier:** default `--dspark-confidence-threshold 0` disables truncation
(same as DeepSpec eval). Work focused on the fast path with `n_max=4` and a short draft forward
(anchor + 4 masks, not full `block_size=7`). Mid-threshold sweeps also hit a defer-layer commit
bug (`extract_layer_inputs` assumed commit rows == tensor rows) - fixed in this session.

Config: `code_500l`, `-c 512`, `-n 400`, `n_max=4`, `DSPARK_NO_ADAPTIVE_NMAX=1`, temp=0, seed=42.

| Threshold | tgp speedup | Accept/step | Draft ms | Verify ms | Proposes | Match vanilla |
|-----------|-------------|-------------|----------|-----------|----------|---------------|
| **0.0** (off) | **1.66x** | 2.50 | 19 | 63 | 4 | YES |
| 0.3 | 0.98x | 3.04 | 62 | 99 | 6 | NO |
| 0.5 | 1.08x | 3.02 | 62 | 87 | 5 | NO |
| 0.7 | 1.04x | 2.77 | 69 | 76 | 4 | NO |
| 0.8 | 0.98x | 2.36 | 68 | 69 | 3 | NO |
| 0.9 | 0.98x | 2.09 | 66 | 63 | 2 | NO |

**Takeaway:** confidence on forces a full `block_size=7` draft forward (~3x draft cost). Higher
accept/step does not compensate on Q4 Vulkan. Verify ms drops slightly at high thresholds (shorter
proposals) but net tgp regresses to ~1.0x. Threshold 0 remains best for throughput; scheduling
may help bf16 setups where draft cost is lower relative to verify.

```bash
# confidence sweep example
for T in 0.0 0.5 0.7 0.9; do
  DSPARK_NO_ADAPTIVE_NMAX=1 ./build/bin/compare_vanilla_speculative \
    -m "$TARGET" -md "$DRAFT" --input-ids /tmp/dspark_eval/code_500l.json \
    --spec-type draft-dspark -c 512 -ngl 99 -ngld 99 --temp 0 --seed 42 \
    --spec-draft-n-max 4 --dspark-confidence-threshold $T -n 400
done
```

## Open work (structural, ordered by impact)

1. **CUDA batched multi-token verify correctness** - sequential fallback works but ~0.90x; batched gives ~1.0x throughput but wrong greedy at gen 54+. KV tail trim and `-fa 0` insufficient; needs llama.cpp CUDA decode fix.
2. **Faster 5-token target forward** - llama graph reuse audit, Vulkan small-batch matmul, fused per-row argmax in graph (skip logits D2H; saves little if forward dominates)
3. **Higher coding acceptance** - draft model / Q4 numerics; n_max>4 hurts match
4. **Fix GPU greedy path** - cached argmax graph, correct row stride; opt-in only
5. **Agentic acceptance** - prompt-class limiter, not verify ms
6. **Spec prefill speed on CUDA** - `DSPARK_PREFILL_DEFER=1` wired in compare harness (match YES, ~0.38x pp). `DSPARK_FAST_PREFILL=1` breaks match @ gen 34 and is slower (~0.21x pp); not recommended.

## How to append results

```bash
COMM=$(git rev-parse --short HEAD)
DSPARK_NO_ADAPTIVE_NMAX=1 ./build/bin/compare_vanilla_speculative \
  -m "$TARGET" -md "$DRAFT" --input-ids /tmp/dspark_eval/code_500l.json \
  --spec-type draft-dspark -c 512 -ngl 99 -ngld 99 --temp 0 --seed 42 \
  --spec-draft-n-max 4 -n 400 2>&1 | tee /tmp/bench.out

grep -E 'generated:|throughput|tgp speedup|mean accepted|token match|verify step|accept \(GPU|layer commit' /tmp/bench.out
```

Record: date, commit, prompt, n, n_max, **pp van/spec, tgp van/spec**, speedup, accept/step, match.
