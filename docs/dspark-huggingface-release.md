# Publishing the DSpark Gemma4 draft (Q4_0) on Hugging Face

Steps to convert [`deepseek-ai/dspark_gemma4_12b_block7`](https://huggingface.co/deepseek-ai/dspark_gemma4_12b_block7) to a **pure Q4_0** GGUF draft and upload it for use with llama.cpp `--spec-type draft-dspark`.

**Pairing target:** [`google/gemma-4-12B-it`](https://huggingface.co/google/gemma-4-12B-it) (any GGUF quant you already use; Q4_0 QAT is what we benchmark).

**Branch:** `ft-dspark` (DSpark converter + speculative driver).

Related docs:

- [dspark-llamacpp-port-plan.md](dspark-llamacpp-port-plan.md) - checkpoint contract and convert details
- [dspark-benchmark-experiments.md](dspark-benchmark-experiments.md) - benchmark setup using `dspark_gemma4_12b_q4pure.gguf`
- [tools/quantize/README.md](../tools/quantize/README.md) - `llama-quantize` options

---

## What you are publishing

| Item | Value |
|------|-------|
| Source HF draft | `deepseek-ai/dspark_gemma4_12b_block7` |
| Target model | `google/gemma-4-12B-it` |
| Output file (example) | `dspark_gemma4_12b_q4pure.gguf` (~1.9 GB) |
| Quantization | **Q4_0**, all tensors (`--pure`) |
| Architecture in GGUF | `dspark` |
| llama.cpp flag | `--spec-type draft-dspark` |

The draft GGUF is **self-contained** (includes `token_embd` and `output` / lm_head). You do not ship the target weights in the same repo.

---

## Prerequisites

1. **llama.cpp built** on `ft-dspark`:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-quantize convert_hf_to_gguf -j"$(nproc)"
```

2. **Python deps** for conversion:

```bash
python3 -m pip install -r requirements.txt
# Gemma4 may need transformers 5:
python3 -m pip install -U transformers huggingface_hub
```

3. **Hugging Face access**

   - Accept the licenses on the source repos (draft + target) if they are gated.
   - Log in: `huggingface-cli login`
   - Or set a write token: `export HF_TOKEN=hf_...`

4. **Disk space** - allow ~8 GB working space (bf16 intermediate + Q4 output).

---

## Step 1 - Convert HF draft to bf16 GGUF

From the llama.cpp repo root:

```bash
DRAFT_REPO="deepseek-ai/dspark_gemma4_12b_block7"
TARGET_REPO="google/gemma-4-12B-it"
BF16_GGUF="/tmp/dspark_gemma4_12b_bf16.gguf"

python convert_hf_to_gguf.py "$DRAFT_REPO" \
  --target-model-dir "$TARGET_REPO" \
  --outtype bf16 \
  --outfile "$BF16_GGUF"
```

`--target-model-dir` is required: the converter pulls tokenizer metadata and target-side config from the Gemma4 target.

**Quick gate** (optional):

```bash
python tests/smoke_phase1_convert.py \
  --hf-repo "$DRAFT_REPO" \
  --target-model-dir "$TARGET_REPO" \
  --outfile "$BF16_GGUF"
```

Inspect metadata:

```bash
./build/bin/gguf-dump "$BF16_GGUF" | head -40
# expect: general.architecture = dspark
#         dspark.block_size = 7
#         dspark.markov_rank = 256
```

---

## Step 2 - Quantize to pure Q4_0

This matches the benchmark artifact `dspark_gemma4_12b_q4pure.gguf`:

```bash
Q4_GGUF="/tmp/dspark_gemma4_12b_q4pure.gguf"

./build/bin/llama-quantize \
  --allow-requantize \
  --pure \
  "$BF16_GGUF" \
  "$Q4_GGUF" \
  Q4_0
```

| Flag | Why |
|------|-----|
| `--pure` | Every tensor is Q4_0 (no K-quant mixes). Name `q4pure` refers to this. |
| `--allow-requantize` | Safe when re-running quant on an already-converted file. |

**Optional quality tweak:** add `--leave-output-tensor` to keep `output.weight` in higher precision (slightly larger file, may help acceptance). Our Vulkan/CUDA benchmarks used full pure Q4_0 without this flag.

Confirm size and arch:

```bash
ls -lh "$Q4_GGUF"
./build/bin/gguf-dump "$Q4_GGUF" | rg 'general.architecture|general.quantization_version|dspark\.'
```

---

## Step 3 - Smoke test before upload

Minimal load + speculative run (needs a target GGUF on disk):

```bash
TARGET_GGUF="/path/to/gemma-4-12B-it-QAT-Q4_0.gguf"   # or your target quant

./build/bin/compare_vanilla_speculative \
  -m "$TARGET_GGUF" -md "$Q4_GGUF" \
  --input-ids tests/data/dspark_gemma4_12b_input_ids.json \
  --spec-type draft-dspark \
  -c 512 -ngl 99 -ngld 99 --temp 0 --seed 42 \
  --spec-draft-n-max 4 -n 32
```

Pass criteria:

- No crash or GGUF load errors.
- `token match: YES` is ideal on bf16/f16 target; on **Q4 target**, some divergence is expected (see [dspark-benchmark-experiments.md](dspark-benchmark-experiments.md)).

---

## Step 4 - Create the Hugging Face model repo

Pick a namespace (`your-user` or org) and a repo name. The ggml-org convention appends `-GGUF`:

```text
your-user/dspark-gemma4-12b-block7-Q4_0-GGUF
```

### Option A - llama.cpp helper scripts

The HF `make` targets live in `examples/model-conversion/`, **not** the repo root
(the root `Makefile` only errors with "Build system changed").

```bash
cd examples/model-conversion

export HF_TOKEN=hf_...   # write token

make hf-create-model \
  MODEL_NAME='dspark-gemma4-12b-block7-Q4_0' \
  NAMESPACE='your-user' \
  ORIGINAL_BASE_MODEL='deepseek-ai/dspark_gemma4_12b_block7'
```

Dry-run first if you want to preview the generated README:

```bash
cd examples/model-conversion

make hf-create-model-dry-run \
  MODEL_NAME='dspark-gemma4-12b-block7-Q4_0' \
  NAMESPACE='your-user' \
  ORIGINAL_BASE_MODEL='deepseek-ai/dspark_gemma4_12b_block7'
```

Or call the script directly (no `make`):

```bash
cd examples/model-conversion
./scripts/utils/hf-create-model.py \
  -m 'dspark-gemma4-12b-block7-Q4_0' -ns 'your-user' \
  -b 'deepseek-ai/dspark_gemma4_12b_block7' -d
```

### Option B - huggingface-cli

```bash
huggingface-cli repo create \
  your-user/dspark-gemma4-12b-block7-Q4_0-GGUF \
  --type model
```

---

## Step 5 - Write the model card (README.md)

Replace the generic causal-LM card from `hf-create-model` with draft-specific usage. Minimum sections:

```markdown
---
base_model:
- deepseek-ai/dspark_gemma4_12b_block7
- google/gemma-4-12B-it
library_name: gguf
tags:
- llama-cpp
- speculative-decoding
- dspark
- gemma4
license: other   # match upstream DeepSpec / Gemma terms
---

# DSpark Gemma4 12B draft (Q4_0 GGUF)

4-bit (pure Q4_0) GGUF draft for **DSpark speculative decoding** with
[`google/gemma-4-12B-it`](https://huggingface.co/google/gemma-4-12B-it).

Converted with llama.cpp `ft-dspark` from
[`deepseek-ai/dspark_gemma4_12b_block7`](https://huggingface.co/deepseek-ai/dspark_gemma4_12b_block7).

## Files

| File | Description |
|------|-------------|
| `dspark_gemma4_12b_q4pure.gguf` | Draft model (~1.9 GB), `LLM_ARCH_DSPARK` |

## Requirements

- llama.cpp built from `ft-dspark` (or a release that includes `draft-dspark`)
- A separate **target** Gemma4 12B GGUF (not included here)

## Usage

```sh
llama-cli \
  -m /path/to/gemma-4-12B-it-Q4_0.gguf \
  -md ./dspark_gemma4_12b_q4pure.gguf \
  --spec-type draft-dspark \
  --spec-draft-n-max 4 \
  -c 512 -ngl 99 -ngld 99 \
  -p "Your prompt" -n 128 --temp 0
```

Server:

```sh
llama-server \
  -m /path/to/gemma-4-12B-it-Q4_0.gguf \
  -md ./dspark_gemma4_12b_q4pure.gguf \
  --spec-type draft-dspark \
  --spec-draft-n-max 4 \
  -c 512 -ngl 99 -ngld 99
```

## Notes

- `confidence_threshold` defaults to 0 (full block proposed). Confidence scheduling is optional.
- Q4 draft + Q4 target can change acceptance vs bf16 PyTorch references; use bf16 target for parity tests.
- Upstream weights: DeepSpec / DeepSeek AI. Target tokenizer: Google Gemma license applies to the pairing.
```

Upload the README:

```bash
huggingface-cli upload your-user/dspark-gemma4-12b-block7-Q4_0-GGUF \
  README.md README.md
```

---

## Step 6 - Upload the GGUF

Large files use Git LFS automatically via the Hub API.

### Option A - make target (from `examples/model-conversion/`)

```bash
cd examples/model-conversion

make hf-upload-gguf-to-model \
  MODEL_PATH=/tmp/dspark_gemma4_12b_q4pure.gguf \
  REPO_ID=your-user/dspark-gemma4-12b-block7-Q4_0-GGUF \
  NAME_IN_REPO=dspark_gemma4_12b_q4pure.gguf
```

Or the script directly:

```bash
cd examples/model-conversion
./scripts/utils/hf-upload-gguf-model.py \
  -m /tmp/dspark_gemma4_12b_q4pure.gguf \
  -r your-user/dspark-gemma4-12b-block7-Q4_0-GGUF \
  -o dspark_gemma4_12b_q4pure.gguf
```

### Option B - huggingface-cli

```bash
huggingface-cli upload \
  your-user/dspark-gemma4-12b-block7-Q4_0-GGUF \
  /tmp/dspark_gemma4_12b_q4pure.gguf \
  dspark_gemma4_12b_q4pure.gguf
```

### Option C - Python

```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="/tmp/dspark_gemma4_12b_q4pure.gguf",
    path_in_repo="dspark_gemma4_12b_q4pure.gguf",
    repo_id="your-user/dspark-gemma4-12b-block7-Q4_0-GGUF",
    repo_type="model",
)
```

After upload, verify download:

```bash
huggingface-cli download your-user/dspark-gemma4-12b-block7-Q4_0-GGUF \
  dspark_gemma4_12b_q4pure.gguf --local-dir /tmp/hf-check
```

---

## Step 7 - (Optional) Pull with `-hf` shorthand

Once published, users can reference the draft by repo id if the filename is standard:

```bash
llama-cli \
  -m /path/to/target.gguf \
  -hf your-user/dspark-gemma4-12b-block7-Q4_0-GGUF \
  --spec-type draft-dspark \
  ...
```

(`-hf` downloads from Hub; target still needs a local path or its own `-hf` repo.)

---

## License and attribution checklist

- [ ] README `base_model` lists `deepseek-ai/dspark_gemma4_12b_block7` and notes Gemma4 target pairing.
- [ ] License field matches upstream (do not pick `apache-2.0` unless it actually applies).
- [ ] Mention conversion tool: llama.cpp `convert_hf_to_gguf.py` + `llama-quantize`.
- [ ] If you fork DeepSeek weights, keep their LICENSE file in the repo or link it explicitly.

---

## End-to-end script (copy-paste)

Adjust `NAMESPACE`, `TARGET_GGUF`, and paths:

```bash
set -euo pipefail
cd /path/to/llama.cpp

NAMESPACE="your-user"
REPO_ID="${NAMESPACE}/dspark-gemma4-12b-block7-Q4_0-GGUF"
DRAFT_REPO="deepseek-ai/dspark_gemma4_12b_block7"
TARGET_REPO="google/gemma-4-12B-it"
BF16="/tmp/dspark_gemma4_12b_bf16.gguf"
Q4="/tmp/dspark_gemma4_12b_q4pure.gguf"

python convert_hf_to_gguf.py "$DRAFT_REPO" \
  --target-model-dir "$TARGET_REPO" \
  --outtype bf16 --outfile "$BF16"

./build/bin/llama-quantize --allow-requantize --pure "$BF16" "$Q4" Q4_0

# validate (optional)
# ./build/bin/compare_vanilla_speculative -m "$TARGET_GGUF" -md "$Q4" ...

cd examples/model-conversion   # HF make targets are here, not repo root
make hf-create-model \
  MODEL_NAME='dspark-gemma4-12b-block7-Q4_0' \
  NAMESPACE="$NAMESPACE" \
  ORIGINAL_BASE_MODEL="$DRAFT_REPO"
# edit README on Hub or locally, then:
make hf-upload-gguf-to-model \
  MODEL_PATH="$Q4" \
  REPO_ID="$REPO_ID" \
  NAME_IN_REPO=dspark_gemma4_12b_q4pure.gguf

echo "Done: https://huggingface.co/${REPO_ID}"
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `gated repo` on download | Accept license on HF model page; `huggingface-cli login` |
| Convert fails on Gemma4 | `pip install -U transformers` (v5 for Gemma4) |
| `architecture` not `dspark` | Build/run from `ft-dspark`, not stock main without DSpark converter |
| Upload timeout | Retry; or use `huggingface-cli upload` with stable network |
| Speculative slower than vanilla | Put draft on GPU (`-ngld 99`); use `-c 512` for short prompts |
| Token mismatch vs reference | Expected with Q4 target; gate parity on bf16 target only |

After publishing, link the repo from your benchmark docs and note the exact filename so others can reproduce [dspark-benchmark-experiments.md](dspark-benchmark-experiments.md) numbers.
