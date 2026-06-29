# DSpark port validation (Phase 0)

Reference traces for llama.cpp DSpark speculative decoding are generated from the DeepSpec PyTorch implementation. Phase 0 locks inference semantics before any C++ work begins.

**Generator:** `DeepSpec/scripts/validate_dspark_reference.py`  
**Smoke gate:** `DeepSpec/scripts/smoke_phase0_reference.sh`  
**Plan:** [dspark-llamacpp-port-plan.md](dspark-llamacpp-port-plan.md)

## Checkpoints

| Role | Hugging Face ID |
|------|-----------------|
| Target | `google/gemma-4-12B-it` |
| Draft | `deepseek-ai/dspark_gemma4_12b_block7` |

## Tokenization contract

DeepSpec eval uses `encode_chat_messages` with the Gemma4 chat template (`<|turn>user\n`, `<|turn>model\n`, etc.), not raw prompt strings.

Phase 0 prompt (user turn only): `"The capital of France is"`, wrapped in the chat template with `add_generation_prompt=True` and `enable_thinking=False`.

The shared fixture is a **JSON array of token IDs**:

- DeepSpec: `tests/data/dspark_gemma4_12b_input_ids.json`
- llama.cpp Phase 3a loads the same file via `--input-ids` (no re-tokenization from `--prompt`).

The detokenized prompt string is recorded in the reference JSONL metadata row for human readability only. The gate is on **token ID equality**.

## Generation settings

| Parameter | Value |
|-----------|-------|
| `temperature` | 0 |
| `confidence_threshold` | 0 |
| `max_new_tokens` | 32 |
| `seed` | 42 |
| dtype | bf16 |
| device | CPU or CUDA (smoke requires byte-stable output across two consecutive runs) |

## Reference artifacts

| File | Content |
|------|---------|
| `dspark_gemma4_12b_input_ids.json` | Prompt token IDs (JSON array) |
| `dspark_gemma4_12b_reference.jsonl` | Metadata row + one JSON object per propose step |

### Per-step JSONL fields

| Field | Meaning |
|-------|---------|
| `start` | Anchor index before propose |
| `ctx_len` | `target_hidden_states.shape[1]` (verify-window length) |
| `block_size` | Draft block size (7 for Gemma4 12B checkpoint) |
| `cache_len_before` | Draft KV length before forward |
| `cache_len_after_forward` | Draft KV length after forward, before crop |
| `cache_len_after_crop` | Draft KV length after `crop(start)` |
| `draft_position_ids_len` | RoPE slice length |
| `verify_len` | Target verify window length (`proposal_len + 1`) |
| `verify_hidden_states_len` | Target hidden-state sequence length from verify forward |
| `proposal_len` | Effective draft proposal length |
| `accepted` | Accepted draft tokens this step |
| `output_token_ids` | Cumulative output token IDs after this step |
| `draft_token_ids` | Proposed draft tokens (debug) |

## Locked invariants (every propose step)

These are asserted by `scripts/smoke_phase0_reference.sh`:

1. **RoPE length:** `draft_position_ids_len == ctx_len + block_size`
   - Position IDs are `position_ids[:, cache_len_before : start + block_size]`, not `[:, start : start + block_size]`.
   - On the first propose after prefill: `cache_len_before = 0`, `ctx_len = T_in`, slice length = `T_in + block_size`.

2. **Draft KV crop:** `cache_len_after_crop == start`
   - After propose, `past_key_values_draft.crop(start)` retains context keys up to the current anchor and drops transient noise keys.

3. **Verify hidden states:** `verify_hidden_states_len == verify_len`
   - With target KV cache, verify forward returns hidden states for the verify window only, not the full sequence.

## KV cache semantics (context vs noise)

Critical orientation for Phase 2/3 implementers:

- **Context K/V is cached.** The draft KV cache accumulates projected fused target features for the committed prefix. It grows by `accepted + 1` each propose (via incremental inject of the verify-window `target_hidden_states`).
- **Noise K/V is transient.** The noise block is decoded, read out, then cropped away before the next propose (`crop(start)` drops the last `block_size` noise entries).

In the PyTorch reference, attention computes `k = concat(k_ctx, k_noise)` in one forward, then appends to cache and crops to `start`. The first `start` entries are context; the trailing noise entries are discarded.

## Target hidden state buffer

| Step | `target_hidden_states` shape | Notes |
|------|------------------------------|-------|
| After prefill | `[1, T_in, L_t x H]` | From `extract_context_feature` on prefill hidden states |
| After verify + update | `[1, accepted+1, L_t x H]` | **Replaces** prior buffer (does not append prompt prefix) |

With target KV cache, verify `output_hidden_states` contain **only new tokens** (`shape[1] == verify_len`).

## Layer extraction

```python
def extract_context_feature(hidden_states, layer_ids):
    return torch.cat([
        hidden_states[0 if lid == -1 else lid + 1]
        for lid in layer_ids
    ], dim=-1)
```

**GGUF mapping:** HF `target_layer_ids` are 0-based decoder output indices. llama.cpp `dspark.target_layers` = `[i + 1 for i in target_layer_ids]` (layer **input** indices for `llama_set_embeddings_layer_inp`).

For `dspark_gemma4_12b_block7`: HF `target_layer_ids = [6, 18, 30, 42, 47]` -> GGUF `dspark.target_layers = [7, 19, 31, 43, 48]`.

## Attention mode

Draft forward uses `attention_mask=None`, `is_causal=False`. Every noise query attends to all context K/V plus all cached and current noise K/V (bidirectional within the draft graph).

llama.cpp must set `hparams.causal_attn = false` and call `llama_set_causal_attn(ctx_dft, false)` in the spec driver.

## Markov head ordering (Phase 3)

For released `dspark_*` checkpoints (`markov_rank=256`, `markov_head_type="vanilla"`):

```
base_i   = lm_head(h_i)
base_i   = tanh(base_i / softcap) * softcap   # Gemma4 only
bias_i   = markov_w2(markov_w1(prev_token_i))
logits_i = base_i + bias_i
```

Position 0 uses the anchor token as `prev_0`. Softcap is applied **before** Markov bias.

## Phase 0 results (2026-06-29)

Smoke test passed on CPU (bf16, `transformers==5.10.2`):

```bash
cd ~/repos/DeepSpec
bash scripts/smoke_phase0_reference.sh
# checked 2 propose steps; all invariants passed
```

| Metric | Value |
|--------|-------|
| Input tokens | 18 |
| Output tokens | 9 (generation stopped early at EOS) |
| Propose steps | 2 |
| Byte-stable across two runs | yes (seed 42) |

**Final output token IDs** (prompt + generation):

```
[2, 105, 2364, 107, 818, 5279, 529, 7001, 563, 106, 107, 105, 4368, 107, 100, 45518, 107, 101,
 818, 5279, 529, 7001, 563, 5213, 50429, 84750, 106]
```

**Step 0 (first propose after prefill):** `start=18`, `ctx_len=18`, `cache_len_before=0`, `draft_position_ids_len=25` (= 18 + 7), `accepted=4`.

**Step 1:** `start=23`, `ctx_len=5` (= verify-window increment after 4 accepted + 1 correction), `cache_len_before=18` (= start - ctx_len), `draft_position_ids_len=12` (= 5 + 7), `accepted=3`, generation terminated on stop token.

Artifacts live in DeepSpec: `tests/data/dspark_gemma4_12b_input_ids.json`, `tests/data/dspark_gemma4_12b_reference.jsonl`.

## Regenerating the reference

```bash
cd ~/repos/DeepSpec
bash scripts/smoke_phase0_reference.sh
```

Or manually:

```bash
cd ~/repos/DeepSpec
python scripts/validate_dspark_reference.py \
  --seed 42 --temperature 0 --confidence-threshold 0 --max-new-tokens 32
```

Use `--device cpu` to force CPU inference.

## Phase 3a gate

llama.cpp speculative smoke must reproduce `final_output_token_ids` from the reference metadata at `temp=0`, loading prompt tokens from `dspark_gemma4_12b_input_ids.json`, with a **full bf16/f16 Gemma4 12B target** (not a tiny fixture).
