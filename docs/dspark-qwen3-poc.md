# DSpark Qwen3-8B POC (branch `ft-dspark-qwen3-gittree`)

Isolated worktree for Qwen3 DSpark support without conflicting with the main `ft-dspark` branch.

**Worktree path:** `~/repos/llama.cpp-dspark-qwen3-gittree`

## Models (Q4)

| Role | Hugging Face | Local path (after download) |
|------|--------------|-----------------------------|
| Target | [Qwen3-8B-GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF) `Qwen3-8B-Q4_K_M.gguf` | `~/models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf` |
| Draft (HF) | [dspark_qwen3_8b_block7](https://huggingface.co/deepseek-ai/dspark_qwen3_8b_block7) | `~/models/dspark_qwen3_8b_block7` |
| Tokenizer | [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | `~/models/Qwen3-8B` |

Download (when ready):

```bash
bash scripts/download-dspark-qwen3-8b-models.sh
```

## Convert draft to Q4 GGUF

```bash
python convert_hf_to_gguf.py ~/models/dspark_qwen3_8b_block7 \
  --target-model-dir ~/models/Qwen3-8B \
  --outtype q4_k_m \
  --outfile ~/models/dspark_qwen3_8b_block7.q4_k_m.gguf
```

Expected metadata:

- `general.architecture` = `dspark`
- `dspark.attention_k_eq_v` = `false` (Qwen3 path: separate `v_proj`, `1/sqrt(d)` scale, no embed scaling)
- `dspark.target_layers` = `[2, 10, 18, 26, 34]` (HF ids `[1, 9, 17, 25, 33]` + 1)
- `dspark.block_size` = `7`
- `tokenizer.ggml.mask_token_id` = `151669`

## Smoke tests

```bash
# Phase 1: HF -> GGUF (downloads weights on first run)
python tests/smoke_phase1_convert_qwen3.py --outtype q4_k_m \
  --outfile /tmp/dspark_qwen3_8b_smoke.gguf

# Phase 4: graph load + encoder/inject/decode
cmake -B build -DLLAMA_BUILD_TESTS=ON
cmake --build build --target smoke_phase4_qwen3 -j
./build/bin/smoke_phase4_qwen3 --draft-model /tmp/dspark_qwen3_8b_smoke.gguf
```

## Speculative run (after build)

```bash
./build/bin/llama-cli \
  -m ~/models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf \
  -md ~/models/dspark_qwen3_8b_block7.q4_k_m.gguf \
  --spec-type draft-dspark \
  --spec-draft-n-max 7 \
  -p "The capital of France is" \
  -n 32 --temp 0
```

Parity vs DeepSpec reference (bf16 target) is not expected with Q4 target; use for crash-free POC only.

## Code changes (Phase 4)

- `conversion/qwen.py`: `Qwen3DSparkModel` converter
- `src/models/dspark.cpp`: Qwen3 graph branch via `dspark.attention_k_eq_v`
- `gguf-py`: `dspark.attention_k_eq_v` metadata key

Reference: `~/repos/DeepSpec/deepspec/modeling/dspark/qwen3/`, port plan Phase 4 in `docs/dspark-llamacpp-port-plan.md`.
