# DSpark benchmark experiments - NVIDIA CUDA

Living log for DSpark speculative decoding on **NVIDIA CUDA** hardware.
Companion to [dspark-benchmark-experiments.md](dspark-benchmark-experiments.md) (Vulkan / Strix Halo).

## Environment

| Item | Value |
|------|-------|
| Host | `ssh ankk98-gpu-vps` (Ubuntu 22.04, x86_64) |
| GPU | **NVIDIA GeForce RTX 3090** (24 GB VRAM, sm_86) |
| Driver | 580.95.05 |
| CUDA toolkit (build) | 12.6 (`/usr/local/cuda-12.6`) |
| Target | `gemma-4-12B-it-QAT-Q4_0.gguf` (~6.5 GB) |
| Draft | `dspark_gemma4_12b_q4pure.gguf` (~1.9 GB) |
| Backend | CUDA (`-ngl 99 -ngld 99`) |
| Branch / commit | `ft-dspark` (rsync to VPS `/root/llama.cpp`, 2026-06-29) |
| Harness | `build/bin/compare_vanilla_speculative` |
| Eval prompts | `/root/dspark_eval/*.json` |

### VPS setup (one-time)

```bash
# On VPS (root@ubuntu)
apt-get install -y cmake ninja-build build-essential git
# CUDA 12.6 (apt cuda-toolkit-11.5 is too old for current llama.cpp)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb && apt-get update && apt-get install -y cuda-toolkit-12-6

# From dev machine: rsync source + models
rsync -az --exclude=build/ --exclude=.git/ ~/repos/llama.cpp/ ankk98-gpu-vps:/root/llama.cpp/
rsync -az /tmp/dspark_eval/ ankk98-gpu-vps:/root/dspark_eval/
rsync -az TARGET.gguf ankk98-gpu-vps:/root/models/target.gguf
rsync -az DRAFT.gguf  ankk98-gpu-vps:/root/models/draft.gguf

# Build
export PATH=/usr/local/cuda-12.6/bin:$PATH
cd /root/llama.cpp
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc -DLLAMA_CURL=OFF
cmake --build build --target compare_vanilla_speculative llama-bench -j$(nproc)
```

### Standard command

Same fair protocol as Vulkan doc (vanilla-first + 3s cooldown, fixed `n_max`):

```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

TARGET=/root/models/target.gguf
DRAFT=/root/models/draft.gguf

DSPARK_NO_ADAPTIVE_NMAX=1 ./build/bin/compare_vanilla_speculative \
  -m "$TARGET" -md "$DRAFT" \
  --input-ids /root/dspark_eval/code_500l.json \
  --spec-type draft-dspark \
  -c 512 -ngl 99 -ngld 99 --temp 0 --seed 42 \
  --spec-draft-n-max 4 -n 400
```

Use **`-c 512`** for short-prompt benches (same rationale as Vulkan doc).

---

## Isolated forward profile (llama-bench, `-fa off`, `-d 512`)

| Test | tok/s | ms/token (approx) | Vulkan (Strix Halo) |
|------|-------|-------------------|---------------------|
| 1-token TG | 77.9 | **~12.8 ms** | ~43 ms |
| 5-token PP (batched) | 310.3 (5 tok) | **~16.1 ms total** | ~70 ms |
| pp512 prefill | 3134.6 | - | - |

CUDA batched 5-token verify is **~4x faster** than Vulkan on this setup. Single-token TG is **~3.4x faster**.

Draft and verify step times in DSpark track these isolated numbers closely.

---

## Fair throughput results (2026-06-29, post-fix)

Default: `-c 512`, `temp=0`, `seed=42`, `n_max=4`, `DSPARK_NO_ADAPTIVE_NMAX=1`, vanilla-first.
CUDA default verify: **sequential** (batched opt-in via `DSPARK_VERIFY_BATCHED=1`).

### Coding (`code_500l`, n=400)

| Config | tgp van | tgp spec | **Speedup** | Accept/step | Match |
|--------|---------|----------|-------------|-------------|-------|
| **Default (sequential verify)** | ~88 | ~80 | **~0.90x** | 2.54 | **YES** |
| `DSPARK_VERIFY_BATCHED=1` + defer | ~88 | ~88 | ~1.00x | 2.54 | NO @ gen 54 |
| Pre-fix batched (committed) | 88.5 | 177.1 | 2.00x | 2.60 | NO @ gen 34 |

Sequential verify is correct but slower than batched multi-token forward (~16 ms vs ~4x per step).
Batched path still diverges on CUDA (gen 54: vanilla=108, spec=107); `-fa 0` does not fix it.

### Other coding prompts (n=300, sequential default)

| Prompt | Speedup | Accept/step | Token match |
|--------|---------|-------------|-------------|
| `code_fib.json` | ~0.89x | 2.76 | **YES** |
| `code_500l.json` | ~0.90x | 2.54 | **YES** |

### Agentic (`agentic_plan`, n=300, sequential default)

| n_max | Speedup | Accept/step | Match |
|-------|---------|-------------|-------|
| 4 | ~0.88x | 1.85 | **YES** |

Pre-fix batched agentic was ~1.53x but token match NO. Sequential restores correctness at cost of speed.

### n_max sweep (`code_500l`, n=400)

| n_max | Speedup | Accept/step | Draft ms | Verify ms | Match |
|-------|---------|-------------|----------|-----------|-------|
| 2 | 1.87x | 1.55 | 3.0 | 12.6 | NO |
| **4** | **1.99x** | **2.60** | **4.2** | **16.3** | NO |
| 7 | 1.67x | 3.08 | 5.5 | 22.3 | NO |

Sweet spot remains **`n_max=4`** for throughput on coding.

### Confidence scheduling (`code_500l`, n=400)

| Threshold | Speedup | Accept/step | Draft ms | Verify ms | Proposes |
|-----------|---------|-------------|----------|-----------|----------|
| **0.0 (off)** | **2.00x** | 2.60 | 4.2 | 16.3 | 4 |
| 0.5 | 1.02x | 3.00 | 23.6 | 20.8 | 5 |
| 0.7 | 1.08x | 3.04 | 23.2 | 19.4 | 4 |
| 0.9 | 0.89x | 2.09 | 23.2 | 16.4 | 2 |

Same pattern as Vulkan: confidence on adds full-block draft cost (~24 ms vs ~4 ms). Net regression above threshold 0 despite higher accept/step.

---

## CUDA vs Vulkan summary

| Metric | Vulkan (Strix Halo) | CUDA (RTX 3090) | Ratio |
|--------|---------------------|-----------------|-------|
| 1-token TG @ d512 | ~43 ms | ~13 ms | **3.3x** |
| 5-token batched forward | ~70 ms | ~16 ms | **4.4x** |
| DSpark verify step | ~63-69 ms | ~16 ms | **~4x** |
| DSpark draft step | ~19 ms | ~4 ms | **~5x** |
| **Fair coding speedup** | **1.57-1.66x** | **~0.90x (seq)** / ~2.0x (batched, wrong) | see below |
| Accept/step (code_500l) | 2.50 | 2.54 | similar |
| Token match (code_500l) | YES | **YES (seq)** / NO @ gen 54 (batched) | sequential fix |

CUDA is fast enough that draft+verify overhead is small relative to vanilla TG (~11 ms/token vanilla vs ~5.6 ms/token effective spec). Speedup approaches **2x** (theoretical max with ~2.6 accept/step and similar per-step cost).

---

## Correctness notes

### Phase 3a smoke (DeepSpec reference)

```bash
./build/bin/smoke_phase3a_speculative \
  -m /root/models/target.gguf -md /root/models/draft.gguf \
  --input-ids tests/data/dspark_gemma4_12b_input_ids.json \
  --reference tests/data/dspark_gemma4_12b_reference.jsonl \
  -ngl 99 -ngld 99 -c 512 --temp 0 --seed 42
# Result: OK: output matches Phase 0 reference (27 tokens)
```

Short-prompt DeepSpec reference match **passes on CUDA**.

### Harness token match vs vanilla

**Fixed (2026-06-29):** CUDA builds default to sequential target verify inside `verify_batched()` (batched multi-token forward diverges). Compare harness uses aligned full-prompt vanilla prefill on all backends.

On `code_500l`, n=400, temp=0, RTX 3090:

| Config | Match | tgp speedup |
|--------|-------|-------------|
| Default (sequential verify + aligned vanilla) | **YES** | **0.90x** |
| `DSPARK_VERIFY_BATCHED=1` (CUDA batched) | NO @ gen 54 | ~1.00x |
| `DSPARK_VERIFY_BATCHED=1` + `-fa 0` | NO @ gen 54 | ~0.82x |

Vulkan (Strix Halo): batched defer verify + standard vanilla prefill, **YES** n=400, ~1.13x.

Opt-in: `DSPARK_VERIFY_BATCHED=1` (CUDA batched), `DSPARK_VERIFY_DEFER=1` (defer fast path), `DSPARK_PREFILL_DEFER=1` (prefill layer defer).

### Batched logits repro (P0)

`tests/smoke_batched_logits_repro.cpp` isolates the CUDA batched-verify bug:

1. Run DSpark with **sequential** verify to a target step (default `--gen-index 54`).
2. Snapshot target KV with `llama_state_get_data`.
3. Compare **row-0 greedy** from single-token decode vs multi-token batched decode on the same KV.
4. Compare per-row: incremental single-token chain vs batched rows 0..n_max.
5. Run full `verify_sequential` vs `verify_batched` on the snapshot.

```bash
# Find first step where row-0 single vs multi diverges
DSPARK_NO_ADAPTIVE_NMAX=1 ./build/bin/smoke_batched_logits_repro \
  -m /root/models/target.gguf -md /root/models/draft.gguf \
  --input-ids /root/dspark_eval/code_500l.json --spec-type draft-dspark \
  -c 512 -ngl 99 -ngld 99 --temp 0 --seed 42 --spec-draft-n-max 4 -n 400 --scan-all

# Deep dive at gen index 54 (known mismatch with batched verify)
DSPARK_NO_ADAPTIVE_NMAX=1 ./build/bin/smoke_batched_logits_repro \
  -m /root/models/target.gguf -md /root/models/draft.gguf \
  --input-ids /root/dspark_eval/code_500l.json --spec-type draft-dspark \
  -c 512 -ngl 99 -ngld 99 --temp 0 --seed 42 --spec-draft-n-max 4 --gen-index 54
```

**Interpretation:**

| Result | Likely cause |
|--------|----------------|
| Row-0 single == multi | Bug is not in forward logits; check accept/trim/process |
| Row-0 single != multi | CUDA multi-token forward corrupts row-0 (causal mask / FA / graph) |
| Row-0 matches, row-k mismatches | Batched row-k uses wrong KV relative to incremental decode |
| `--scan-all` first mismatch step << 54 | Earlier KV drift; gen 54 is symptom not root step |

---

## Observations

1. **CUDA batched verify blocked.** Multi-token target forward on CUDA produces wrong greedy logits on long runs (gen 54+). KV tail trim and `-fa 0` do not fix. Sequential verify is correct (~0.90x) but loses batched speed. Needs llama.cpp CUDA multi-token decode investigation.
2. **CUDA removes the verify bottleneck (when batched works).** On Vulkan verify (~63 ms) dominates; on RTX 3090 batched verify (~16 ms) is comparable to draft (~4 ms).
3. **~2x was near ceiling with batched verify** but batched is not yet correct on CUDA.
4. **Confidence scheduling** still not worth it on Q4 CUDA (draft cost 6x when enabled).
5. **`-c 512` vs `1024`** has negligible impact on CUDA for this short prompt (unlike Vulkan where KV length hurt verify).
6. **Draft on CPU** (`-ngld 0`) destroys speedup (0.43x) - keep draft on GPU.

---

## Experiment log

| Date | GPU | Prompt | n | n_max | Speedup | Accept/step | Match | Notes |
|------|-----|--------|---|-------|---------|-------------|-------|-------|
| 2026-06-29 | RTX 3090 | code_500l | 400 | 4 | **0.90x** | ~2.6 | **YES** | sequential verify default |
| 2026-06-29 | RTX 3090 | code_500l | 400 | 4 | 2.00x | 2.60 | NO@34 | pre-fix batched verify |
| 2026-06-29 | RTX 3090 | code_fib | 300 | 4 | ~0.89x | 2.76 | **YES** | sequential default |
| 2026-06-29 | RTX 3090 | agentic_plan | 300 | 4 | ~0.88x | 1.85 | **YES** | sequential default |
| 2026-06-29 | RTX 3090 | Phase 3a ref | 32 | 7 | - | - | ref OK | smoke pass |

---

## How to append results

```bash
ssh ankk98-gpu-vps
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
cd /root/llama.cpp

DSPARK_NO_ADAPTIVE_NMAX=1 ./build/bin/compare_vanilla_speculative \
  -m /root/models/target.gguf -md /root/models/draft.gguf \
  --input-ids /root/dspark_eval/code_500l.json \
  --spec-type draft-dspark -c 512 -ngl 99 -ngld 99 --temp 0 --seed 42 \
  --spec-draft-n-max 4 -n 400 2>&1 | tee /tmp/bench_cuda.out

grep -E 'generated:|tgp speedup|mean accepted|token match|first mismatch|verify step|draft  step' /tmp/bench_cuda.out
```

Record: date, GPU, driver, CUDA version, prompt, n, n_max, pp/tgp van/spec, speedup, accept/step, match, mismatch index if any.
