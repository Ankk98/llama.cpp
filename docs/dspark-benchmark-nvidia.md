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

## Fair throughput results (2026-06-29)

Default: `-c 512`, `temp=0`, `seed=42`, `n_max=4`, `DSPARK_NO_ADAPTIVE_NMAX=1`, vanilla-first.

### Coding (`code_500l`, n=400)

| Config | tgp van | tgp spec | **Speedup** | Accept/step | Draft ms | Verify ms | Match |
|--------|---------|----------|-------------|-------------|----------|-----------|-------|
| **Default CUDA** | 88.5 | **177.1** | **2.00x** | 2.60 | 4.2 | 16.3 | NO* |
| `-c 1024` | 88.4 | 176.6 | 2.00x | 2.60 | 4.2 | 16.3 | NO |
| No defer layer | 88.3 | 176.0 | 1.99x | 2.60 | 4.2 | 16.3 | NO |
| Draft CPU (`-ngld 0`) | 88.3 | 38.0 | 0.43x | 2.54 | 73.2 | 20.7 | NO |
| Target+Draft CPU (`-ngl 0 -ngld 0`) | 88.3 | 174.0 | 1.97x | 2.45 | 80.6 | 256.0 | NO |

\*First mismatch at **generated token index 34** (`vanilla=13740`, `spec=24767`). First 34 tokens identical.

### Other coding prompts (n=200, n_max=4)

| Prompt | Speedup | Accept/step | Token match | Notes |
|--------|---------|-------------|-------------|-------|
| `code_fib.json` | 1.98x | 2.59 | **YES** | |
| `code_sql.json` | 1.98x | 2.54 | NO | mismatch @ index 29 |
| `code_bug.json` | 1.95x | 2.49 | NO | mismatch @ index 109 |
| `code_500l.json` | 2.00x | 2.60 | NO | mismatch @ index 34 |

Token match is **prompt-dependent on CUDA**. Not all prompts diverge from vanilla greedy.

### Agentic (`agentic_plan`, n=300)

| n_max | tgp van | tgp spec | Speedup | Accept/step | Verify ms | Match |
|-------|---------|----------|---------|-------------|-----------|-------|
| 4 | 88.6 | 135.4 | **1.53x** | 1.75 | 16.2 | NO |

Agentic acceptance ~1.75/step (vs ~1.3-1.6 on Vulkan). CUDA hits the 1.5x agentic target.

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
| **Fair coding speedup** | **1.57-1.66x** | **2.00x** | **~1.2x** |
| Accept/step (code_500l) | 2.50 | 2.60 | similar |
| Token match (code_500l) | YES | NO @ index 34 | see below |

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

On `code_500l`, spec diverges from vanilla greedy at token 34 despite deterministic reruns (same mismatch index). Possible causes under investigation:

- GPU Markov head + fused block sampler on CUDA (`markov head running on GPU/backend`)
- Longer-context layer injection / KV numerics on CUDA Q4
- Prompt-specific draft acceptance path (some prompts match, e.g. `code_fib.json` YES)

`DSPARK_NO_BLOCK_GPU=1` did not restore match on `code_500l`. Vulkan reported match YES on the same prompt/weights - backend-specific divergence worth tracking before claiming byte-identical output on all CUDA workloads.

---

## Observations

1. **CUDA removes the verify bottleneck.** On Vulkan, verify (~63 ms) dominated; on RTX 3090 verify (~16 ms) is comparable to draft (~4 ms). Further speedup needs higher acceptance or lower vanilla baseline (already ~88 tgp).
2. **~2x is near the ceiling** for this accept rate (~2.6/step) without quality loss or longer proposals.
3. **Confidence scheduling** still not worth it on Q4 CUDA (draft cost 6x when enabled).
4. **`-c 512` vs `1024`** has negligible impact on CUDA for this short prompt (unlike Vulkan where KV length hurt verify).
5. **Defer layer** saves little when verify is already fast (~16 ms); impact within noise.
6. **Draft on CPU** (`-ngld 0`) destroys speedup (0.43x) - keep draft on GPU.

---

## Experiment log

| Date | GPU | Prompt | n | n_max | Speedup | Accept/step | Match | Notes |
|------|-----|--------|---|-------|---------|-------------|-------|-------|
| 2026-06-29 | RTX 3090 | code_500l | 400 | 4 | **2.00x** | 2.60 | NO@34 | default fair bench |
| 2026-06-29 | RTX 3090 | code_fib | 200 | 4 | 1.98x | 2.59 | YES | |
| 2026-06-29 | RTX 3090 | agentic_plan | 300 | 4 | 1.53x | 1.75 | NO | hits 1.5x target |
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
