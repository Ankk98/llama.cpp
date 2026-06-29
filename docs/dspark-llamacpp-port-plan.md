# DSpark support in llama.cpp — implementation plan

This document is a grounded, phased plan for adding **DSpark** speculative-decoding support to [llama.cpp](https://github.com/ggml-org/llama.cpp). It is written so that an engineer or AI agent can implement the feature without guessing at semantics.

**Primary target (Phase 1):** [`deepseek-ai/dspark_gemma4_12b_block7`](https://huggingface.co/deepseek-ai/dspark_gemma4_12b_block7) paired with [`google/gemma-4-12B-it`](https://huggingface.co/google/gemma-4-12B-it).

**Reference implementation:** DeepSpec (`deepspec/modeling/dspark/`, `deepspec/eval/dspark/`).

**Not in scope of this doc:** LM Studio integration (llama.cpp CLI/server only).

---

## Table of contents

1. [Executive summary](#executive-summary)
2. [Why DSpark is not DFlash](#why-dspark-is-not-dflash)
3. [Markov head: what “vanilla” means](#markov-head-what-vanilla-means)
4. [Weight sharing (`ctx_other`)](#weight-sharing-ctx_other)
5. [Validated inference semantics](#validated-inference-semantics)
6. [Gemma4 checkpoint contract](#gemma4-checkpoint-contract)
7. [Phased delivery](#phased-delivery)
8. [Smoke test index](#smoke-test-index)
9. [Phase 0 — Reference validation](#phase-0--reference-validation)
10. [Phase 1 — GGUF converter (Gemma4)](#phase-1--gguf-converter-gemma4)
11. [Phase 2 — Model graph (`LLM_ARCH_DSPARK`)](#phase-2--model-graph-llm_arch_dspark)
12. [Phase 3 — Speculative driver (`draft-dspark`)](#phase-3--speculative-driver-draft-dspark)
13. [Phase 4 — Qwen3 + DFlash-mode](#phase-4--qwen3--dflash-mode)
14. [Phase 5 — CI integration & release](#phase-5--ci-integration--release)
15. [Risk register](#risk-register)
16. [File change index](#file-change-index)
17. [Locked decisions](#locked-decisions-binding)

---

## Executive summary

| Question | Answer |
|----------|--------|
| Can we reuse the llama.cpp `draft-dflash` **checkpoint format**? | **No.** Different HF architecture and tensor layout (see contrast table). Add a new `LLM_ARCH_DSPARK`. |
| Can we reuse the llama.cpp DFlash **graph/driver structure** (encoder + KV-injection decoder)? | **Yes — use it as the template (binding).** DFlash already injects fused target features as K/V into the draft cache and runs the noise block over that cache; DSpark needs the same, with incremental injection + Gemma4 graph specifics (see Phase 2). |
| Can we reuse DeepSpec Eagle3/DFlash HF weights in llama.cpp today? | **No.** Different architectures and checkpoint formats. |
| Primary v1 checkpoint | `deepseek-ai/dspark_gemma4_12b_block7` |
| New llama.cpp arch | `LLM_ARCH_DSPARK` (`--spec-type draft-dspark`) |
| Vulkan | No new backend ops; standard attention/matmul/norm/rope/concat. |
| Estimated effort (Gemma4 end-to-end) | ~4–6 weeks (after Phase 0; reuses DFlash encoder + KV-injection decoder structure) |
| E2E token-match gate | **Full Gemma4 12B target, bf16/f16** — not a tiny target fixture |
| CI on every PR | Phases 1–2 + graph NMSE only; Phases 3a–4 E2E run nightly / manual |

---

## Why DSpark is not DFlash

llama.cpp already supports **DFlash** (`--spec-type draft-dflash`) for z-lab `DFlashDraftModel` checkpoints. DeepSpec’s released “DFlash” weights (`deepseek-ai/dflash_*`) are **not** that format — they use `Gemma4DSparkModel` / `Qwen3DSparkModel` with `markov_rank=0`.

The **checkpoint format** differs, so a new `LLM_ARCH_DSPARK` is required. But the **graph/driver structure is reusable** — DFlash and DSpark both fuse target features through an encoder, inject them as K/V into the draft cache, and run a non-causal noise block over that cache. The differences are localized:

| Aspect | llama.cpp DFlash (z-lab) | DSpark (DeepSpec) |
|--------|--------------------------|-------------------|
| HF architecture | `DFlashDraftModel` | `Gemma4DSparkModel` / `Qwen3DSparkModel` |
| Target conditioning | Encoder → inject fused K/V into draft cache (per-batch, position-indexed) | **Same**: encoder → inject fused K/V into draft cache. DSpark injects the verify-window increment each step (the cache accumulates the full-prefix context). |
| Context buffer | Fused features injected as KV at their positions | **Same.** In PyTorch reference this is expressed as `concat(k_ctx, k_noise)` in one forward, which is mathematically equivalent to inject-then-attend (each stream is RoPE’d at its own absolute positions). |
| Draft propose size | `block_size - 1` tokens (reads block positions `1..block-1`) | Up to **`block_size`** draft tokens (reads block positions `0..block-1` — the anchor’s hidden also predicts) |
| Per-layer graph | Qwen-style: `v_proj`, `1/sqrt(d)` scale, standard RoPE | Gemma4: `k_eq_v` (no `v_proj`), attn scale `1.0`, partial proportional RoPE, softcap, four layernorms |
| Extra heads | None | Markov (autoregressive bias) + confidence — applied on **CPU** in the driver |

**Do not extend `LLM_ARCH_DFLASH`** (separate checkpoint format / hparams). Add a new `LLM_ARCH_DSPARK`, but **model its graph and driver on the existing DFlash encoder + KV-injection decoder** (see Phase 2) rather than inventing a new single-forward dual-stream graph.

> **KV-cache orientation (read this before Phase 2):** the draft KV cache holds the **context** keys (projected, fused target features) for the entire committed prefix, growing by `accepted+1` each propose. The **noise** block is the *transient* stream: it is decoded, read out, then dropped/overwritten before the next propose. This is the opposite of a naive reading where “noise is cached.” It is verified against the reference in [Validated inference semantics](#validated-inference-semantics).

---

## Markov head: what “vanilla” means

Released DSpark checkpoints are **not** “plain LM-head only.” They include a **Markov head** with `markov_rank=256` and `markov_head_type="vanilla"`.

Naming is easy to misread:

| Term | Meaning |
|------|---------|
| **`vanilla`** | A **type** of Markov head (memoryless token embedding → logit bias). It is **not** “no Markov head.” |
| **`gated` / `rnn`** | Other Markov head types implemented in DeepSpec but **not used in any released checkpoint**. **Out of scope for v1** — the converter and loader assert-fail on these (locked decision #3). |
| **`markov_rank=0`** | Markov head **disabled** — this is the DeepSpec “DFlash ablation” (`deepseek-ai/dflash_*`), not full DSpark. |

### What the vanilla Markov head does at inference

After the draft backbone produces hidden states `h_i` for each block position `i`:

```
base_i   = lm_head(h_i)
base_i   = tanh(base_i / softcap) * softcap   # Gemma4 only; Qwen3 skips
bias_i   = markov_w2( markov_w1(prev_token_i) )
logits_i = base_i + bias_i
token_i  = sample(logits_i)
prev_{i+1} = token_i
```

- Position 0 uses the **anchor token** (last accepted token) as `prev_0`.
- Weights: `markov_w1` is `Embedding(vocab, 256)`, `markov_w2` is `Linear(256, vocab)`.
- **Order matters:** softcap is applied to `lm_head` output **before** Markov bias (matches `compute_logits` → `sample_draft_tokens` in DeepSpec).
- At `temp > 0`, rejection sampling uses the **post-Markov** logits as `draft_probs` (see Phase 3c).

**Why implement Markov for released weights?** Without it, logits come only from `lm_head(h_i)` and acceptance rates will be wrong — the checkpoint was trained and evaluated **with** Markov bias. Skipping Markov is only valid for `dflash_*` ablation checkpoints (`markov_rank=0`).

### Confidence head (full DSpark only)

Released `dspark_*` checkpoints also have `enable_confidence_head=true`. At inference with default `confidence_threshold=0.0`, the full block is proposed (no truncation). The head still exists in weights and is used when threshold > 0.

**Phased approach:** implement vanilla Markov in Phase 3a (required for `dspark_*`); confidence truncation in Phase 3b; rejection sampling (`draft_probs`) in Phase 3c for `temp > 0`.

---

## Weight sharing (`ctx_other`)

### What DeepSpec stores in the checkpoint

During training, `embed_tokens` and `lm_head` are **copied from the target model and frozen**, but they are still **saved in the draft safetensors** (`embed_tokens.weight`, `lm_head.weight`). The draft checkpoint is self-contained on Hugging Face.

### What llama.cpp DFlash/Eagle3 do

For `LLM_ARCH_DFLASH` and `LLM_ARCH_EAGLE3`, draft GGUFs often **omit** `token_embd` and `output` (lm_head). At context init (`llama-context.cpp`):

```cpp
if (model.arch == LLM_ARCH_EAGLE3 || model.arch == LLM_ARCH_DFLASH) {
    if (model.tok_embd == nullptr || model.output == nullptr) {
        cparams.ctx_other = params.ctx_other;  // borrow from target context
    }
}
```

The draft decoder then calls `get_rows(tok_embd, tokens)` and `mm(output, hidden)` using the **target model’s** embedding and LM head tensors. Benefits:

- Smaller draft GGUF (~saves 2 × vocab × hidden_size)
- Guaranteed vocab alignment with the target

### Approach for DSpark — LOCKED

**Decision (binding):** load `token_embd` and `output` (lm_head) from the **draft GGUF** (Strategy A). The converter always emits both tensors; the loader never sets `ctx_other` for them. Do **not** implement a `--share-embed-with-target` flag or any `ctx_other` sharing path for embed/lm_head in v1 (see [Out of scope](#explicitly-out-of-scope-for-v1)).

Rationale (context only — not a re-litigation point): DeepSpec checkpoints already ship these tensors, and loading them from the draft avoids silent mismatch if the target quant differs from the frozen draft copy.

| Strategy | Status |
|----------|--------|
| **A. Load embed + lm_head from draft GGUF** | **LOCKED for v1** |
| **B. Omit embed + lm_head; use `ctx_other`** | Out of scope (do not implement) |

Gemma4 note: embeddings use `Gemma4TextScaledWordEmbedding` (`embed_scale = sqrt(hidden_size)`); the scale is applied in the graph (Phase 2), not baked into `token_embd`.

Gemma4 note: embeddings use `Gemma4TextScaledWordEmbedding` (`embed_scale = sqrt(hidden_size)`). If sharing via `ctx_other`, the target Gemma4 graph must expose the same scaled embedding path.

**Residual quant risk (Strategy A):** embed/lm_head come from the draft GGUF, but **target layer features** are still extracted from the (possibly quantized) target at runtime. Q4 target hidden states can diverge from bf16 PyTorch references and change acceptance paths. Phase 3a exact-match gates use **bf16/f16 target**; quantized target is a separate crash/perf smoke (Phase 5 release).

---

## Validated inference semantics

Experiment run against `Qwen/Qwen3-4B` + `deepseek-ai/dspark_qwen3_4b_block7` (Gemma4 semantics are the same; dimensions differ). Source: `deepspec/eval/dspark/draft_ops.py`, `deepspec/eval/dspark/evaluator.py`.

### Position IDs for draft forward

```python
draft_position_ids = position_ids[:, cache_len : start + block_size]
```

- **Not** `position_ids[:, start : start + block_size]`.
- On the first propose: `cache_len=0`, `start=T_in` → length `T_in + block_size` (= `ctx_len + block_size`).
- RoPE `cos/sin` length follows `position_ids.shape[1]`, not noise length alone.

### Target hidden state buffer

| Step | `target_hidden_states` shape | Notes |
|------|------------------------------|-------|
| After prefill | `[1, T_in, L_t × H]` | From `extract_context_feature` on prefill hidden states |
| After verify + `_update` | `[1, accepted+1, L_t × H]` | **Replaces** prior buffer (does not append prompt prefix) |

With KV cache, `output_hidden_states` from verify contain **only new tokens** (`shape[1] == verify_len`), not the full sequence.

### Draft KV cache

**What the cache holds (critical — context, not noise):** in the reference, the per-layer attention computes `k = concat(k_ctx, k_noise)` and **appends the whole thing** to the cache; `crop(start)` then keeps the **first `start` entries, which are the context keys** (the projected fused target features). The noise keys (the last `block_size` entries) are dropped. So:

- The draft cache accumulates **context K/V for the full committed prefix**, growing by `accepted+1` each propose.
- The **noise** block K/V is transient — written by the forward, then cropped away.
- Each propose feeds only the **incremental** `target_hidden_states` (`accepted+1` rows after `_update`), not the full prefix; the cache supplies the rest of the context (see [evaluator `_update`](#target-hidden-state-buffer)).

Source: `forward_dspark_draft_block` (`past_key_values_draft.update(concat(k_ctx,k_noise))` then `crop(start)`) and `Gemma4DSparkAttention.forward`.

**Lengths per propose:**

- Before propose: `cache_len = past_key_values_draft.get_seq_length()` = **previous** anchor (`start - ctx_len`), **not** the current `start`.
- After propose forward: cache length = `cache_len + ctx_len + block_size` = `start + block_size`.
- Then `past_key_values_draft.crop(start)` → cache length = `start` (current anchor; retains all context keys, drops noise).

**Crop invariant (critical):** crop uses the **current anchor `start`**. After crop, `cache_len == start`. The propose position slice uses the *pre-forward* `cache_len` as its lower bound:

```
len(position_ids[:, cache_len : start + block_size]) == ctx_len + block_size
```

where `ctx_len = target_hidden_states.shape[1]` (verify-window length after `_update`, not full prompt length) and `cache_len = start - ctx_len`.

In llama.cpp (DFlash-style): the context K/V is injected at absolute positions in `process()`, so it persists correctly across iterations; the noise K/V written by `draft()` is removed before the next propose with `llama_memory_seq_rm(ctx_dft, seq, draft_start, -1)` (crop to **`draft_start` (anchor)**). See Phase 3a.

### Attention mode at inference

During eval, `_forward_backbone` is called with `attention_mask=None` and `is_causal=False`. Every noise query attends to **all** context K/V (current forward) **plus** all cached + current noise K/V — fully bidirectional within the draft graph.

Implementation requirements in llama.cpp:

- Set `hparams.causal_attn = false` on load (`LLM_KV_ATTENTION_CAUSAL = 0` in GGUF).
- Call `llama_set_causal_attn(ctx_dft, false)` in the spec driver ctor (same as DFlash).
- **Reuse DFlash’s two-pass decode** (`ubatch.embd` context-inject → token/noise decode). Attention over `[cached context K/V] + [noise K/V]` is mathematically identical to the reference’s `concat(k_ctx, k_noise)` single forward, because each stream is RoPE’d at its own absolute positions. Do **not** invent a new single-forward dual-stream graph input — it adds risk for no correctness benefit (see Phase 2).
- Phase 2 graph test must include iteration ≥ 2 with non-empty draft KV to catch mask/cache bugs.

### Tokenization contract (Phase 0 / Phase 3)

DeepSpec eval tokenizes via `encode_chat_messages` with the **Gemma4 chat template** (`<|turn>user\n`, `<|turn>model\n`, etc.) — not raw prompt strings.

Phase 0 and Phase 3a **must share the same token IDs**, not just the same English text:

1. Phase 0 writes `tests/data/dspark_gemma4_12b_input_ids.json` (a JSON array of token IDs; see [locked decision #7](#locked-decisions-binding)) alongside the reference JSONL.
2. Phase 3a smoke loads that file and feeds the exact token sequence (no re-tokenization from `--prompt`).
3. Document the chat-template path in `tests/data/README.md`.

Phase 0 also records the detokenized string in the reference JSONL for human readability (informational only); the gate is on **token ID equality**.

### Layer extraction for target features

```python
def extract_context_feature(hidden_states, layer_ids):
    return torch.cat([
        hidden_states[0 if lid == -1 else lid + 1]
        for lid in layer_ids
    ], dim=-1)
```

**GGUF mapping:** HF `target_layer_ids` are 0-based decoder output indices. llama.cpp `dspark.target_layers` = `[i + 1 for i in target_layer_ids]` (layer **input** indices for `llama_set_embeddings_layer_inp`).

---

## Gemma4 checkpoint contract

**Checkpoint:** `deepseek-ai/dspark_gemma4_12b_block7`  
**Target:** `google/gemma-4-12B-it`

| Field | Value |
|-------|-------|
| `architectures` | `["Gemma4DSparkModel"]` |
| `block_size` | 7 |
| `num_hidden_layers` | 5 (draft) |
| `target_layer_ids` | `[5, 17, 29, 41, 46]` |
| `mask_token_id` | 4 |
| `hidden_size` | 3840 |
| `vocab_size` | 262144 |
| `markov_rank` | 256 |
| `markov_head_type` | `"vanilla"` |
| `enable_confidence_head` | true |
| `confidence_head_with_markov` | true |
| `attention_k_eq_v` | true |
| `final_logit_softcapping` | 30.0 |
| `global_head_dim` | 512 |
| `num_global_key_value_heads` | 1 |
| RoPE | proportional, `partial_rotary_factor=0.25` |

**GGUF `dspark.target_layers`:** `[6, 18, 30, 42, 47]` (HF ids + 1).

### Gemma4-specific graph differences (vs Qwen3)

| Feature | Implementation note |
|---------|---------------------|
| Scaled embeddings | `embed * sqrt(hidden_size)` |
| `attention_k_eq_v` | No `v_proj`; **V = K** on both streams; **`v_norm` without scale** |
| Attention scale | `1.0` (not `1/sqrt(d)`) |
| Extra layer norms | `input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`, `post_feedforward_layernorm`, `layer_scalar` |
| Logits | `tanh(logits / 30) * 30` after `lm_head`, **before** Markov bias |
| Inference attention | Fully bidirectional (`attention_mask=None`, `is_causal=False`) |
| MoE / per-layer input | Assert disabled; not in released checkpoint |

### Ablation checkpoint (later phase)

`deepseek-ai/dflash_gemma4_12b_block7`: same `Gemma4DSparkModel`, `markov_rank=0`, `enable_confidence_head=false` — valid for Phase 4 “DFlash-mode” testing without Markov/confidence code paths.

---

## Phased delivery

```
Phase 0  Reference validation (Gemma4)     ──smoke──► gate
Phase 1  GGUF converter (Gemma4)          ──smoke──► gate
Phase 2  Model graph (Gemma4 DSpark)      ──smoke──► gate (NMSE vs PyTorch)
Phase 3a Spec driver + vanilla Markov     ──smoke──► gate (E2E token match, full target)
Phase 3b Confidence head truncation       ──smoke──► gate
Phase 3c Rejection sampling (temp > 0)    ──smoke──► gate (required for temp>0; see locked #5b)
Phase 4  Qwen3 variant + DFlash-mode      ──smoke──► gate
Phase 5  CI wiring + docs + release       (tiered CI; see below)
```

| Phase | Delivers | Smoke test (must pass before next phase) |
|-------|----------|------------------------------------------|
| 0 | Reference trace JSON + shared `input_ids` | `scripts/smoke_phase0_reference.sh` |
| 1 | `dspark_gemma4_12b.gguf` | `tests/smoke_phase1_convert.py` |
| 2 | `LLM_ARCH_DSPARK` graph | `tests/smoke_phase2_graph.cpp` |
| 3a | `--spec-type draft-dspark` | `tests/smoke_phase3a_speculative.cpp` (full 12B target, bf16) |
| 3b | Confidence truncation | `tests/smoke_phase3b_confidence.cpp` |
| 3c | `draft_probs` / rejection sampling | `tests/smoke_phase3c_rejection.cpp` |
| 4 | Qwen3 + dflash ablation | `tests/smoke_phase4_qwen3.cpp` |
| 5 | CI + docs | Tiered: PR = phases 1–2; nightly = 3a–4 |

**Rule:** No phase is complete until its smoke test passes locally and is registered in CI (wired in Phase 5).

---

## Smoke test index

Quick reference for all per-phase smokes.

| Tier | When | Phases |
|------|------|--------|
| **PR (fast)** | Every PR touching dspark code | 1, 2 |
| **Nightly / manual** | Cached full weights; bf16 target | 3a, 3b, 3c, 4 |
| **Release** | Developer machine | Quantized target + Vulkan perf |

PR smokes finish in **< 5 minutes on CPU**. Nightly E2E smokes may take longer (12B target load).

| Phase | Script / binary | Repo | What it checks |
|-------|-----------------|------|----------------|
| 0 | `scripts/smoke_phase0_reference.sh` | DeepSpec | Reference JSON + `input_ids` fixture; invariants |
| 1 | `tests/smoke_phase1_convert.py` | llama.cpp | HF → GGUF; metadata + required tensors |
| 2 | `tests/smoke_phase2_graph.cpp` | llama.cpp | Model loads; encoder + decoder; logits NMSE (synthetic ctx) |
| 3a | `tests/smoke_phase3a_speculative.cpp` | llama.cpp | Full spec loop; token match vs Phase 0 (**full 12B target, bf16**) |
| 3b | `tests/smoke_phase3b_confidence.cpp` | llama.cpp | Proposal length shrinks when threshold > 0 |
| 3c | `tests/smoke_phase3c_rejection.cpp` | llama.cpp | Stochastic accept path uses post-Markov `draft_probs` |
| 4 | `tests/smoke_phase4_qwen3.cpp` | llama.cpp | Qwen3 convert + graph; dflash ablation |
| 5 | CI job `dspark-smoke` | llama.cpp | PR: 1–2; nightly: 3a–4 |

---

## Phase 0 — Reference validation

**Goal:** Lock behavior before C++ work. **Do not start Phase 1 until this passes.**

**Primary checkpoint:** Gemma4 (`google/gemma-4-12B-it` + `deepseek-ai/dspark_gemma4_12b_block7`). Qwen3 semantics were used to validate the doc, but Phase 0 reference artifacts must come from the Gemma4 pair.

### Tasks

1. Add `scripts/validate_dspark_reference.py` in DeepSpec:
   - Target: `google/gemma-4-12B-it`
   - Draft: `deepseek-ai/dspark_gemma4_12b_block7`
   - Tokenize with Gemma4 chat template (same path as `encode_chat_messages` in eval)
   - Fixed user turn: `"The capital of France is"` wrapped in chat template
   - Run eval loop: `temperature=0`, `confidence_threshold=0`, `max_new_tokens=32`, `seed=42`
   - Save shared fixture: `tests/data/dspark_gemma4_12b_input_ids.json`
   - Save trace: `tests/data/dspark_gemma4_12b_reference.jsonl`

2. Per-iteration JSONL fields (minimum):

   | Field | Purpose |
   |-------|---------|
   | `start` | Anchor index before propose |
   | `ctx_len` | `target_hidden_states.shape[1]` |
   | `cache_len_before` / `cache_len_after_crop` | Draft KV lengths |
   | `draft_position_ids_len` | RoPE length (= `ctx_len + block_size`) |
   | `verify_hidden_states_len` | Must equal `verify_len` when target KV set |
   | `proposal_len`, `accepted` | Spec stats |
   | `output_token_ids` | Cumulative output (for final match) |
   | `draft_token_ids` | Tokens proposed this step (optional debug) |

3. Document invariants (also copied to llama.cpp `docs/dspark-port-validation.md`):
   - RoPE length: `draft_position_ids_len == ctx_len + block_size` on **every** propose (not just the first).
   - Draft KV crop: `cache_len_after_crop == start` (anchor at propose time).
   - Verify hidden states: length equals verify window only.

### Acceptance

- [ ] Script runs reproducibly on CPU (bf16 models)
- [ ] `input_ids` fixture + reference JSONL checked in or generated in CI
- [ ] Findings recorded in llama.cpp `docs/dspark-port-validation.md`

### Smoke test — `scripts/smoke_phase0_reference.sh` (DeepSpec)

**Run:**

```bash
cd DeepSpec
bash scripts/smoke_phase0_reference.sh
```

**Script steps:**

1. Run `scripts/validate_dspark_reference.py` with fixed seed (`42`), `max_new_tokens=32`, `temp=0`.
2. Assert output files exist:
   - `tests/data/dspark_gemma4_12b_input_ids.json`
   - `tests/data/dspark_gemma4_12b_reference.jsonl`
3. Assert JSONL has ≥ 1 generation step with required fields (see table above).
4. Assert invariants on **every** propose row:
   - `draft_position_ids_len == ctx_len + block_size`
   - `cache_len_after_crop == start`
   - `verify_hidden_states_len == verify_len` (when present)
5. Exit 0 if all assertions pass.

**Pass criteria:** Script exits 0; reference file is byte-stable across two consecutive runs (same seed).

---

## Phase 1 — GGUF converter (Gemma4)

**Repo:** llama.cpp

### New / modified files

| File | Change |
|------|--------|
| `gguf-py/gguf/constants.py` | `MODEL_ARCH.DSPARK`, metadata keys, tensor list |
| `gguf-py/gguf/gguf_writer.py` | `add_markov_rank()`, `add_markov_head_type()`, confidence metadata |
| `gguf-py/gguf/tensor_mapping.py` | HF → GGUF names including markov/confidence |
| `conversion/gemma.py` | `@ModelBase.register("Gemma4DSparkModel")` — extend existing Gemma4 converter |
| `conversion/__init__.py` | `"Gemma4DSparkModel": "gemma"` |

### Metadata keys

Use the same **llama.cpp KV enum pattern as DFlash/Eagle3**: `LLM_KV_TARGET_LAYERS` maps to `{arch}.target_layers` (i.e. `dspark.target_layers`). Do not invent a parallel target-layer key.

| GGUF key | Example (Gemma4) | C++ enum / writer |
|----------|------------------|-------------------|
| `general.architecture` | `dspark` | `LLM_ARCH_DSPARK` |
| `dspark.block_size` | 7 | new `LLM_KV_*` or arch-specific writer |
| `dspark.target_layers` | `[6, 18, 30, 42, 47]` | `LLM_KV_TARGET_LAYERS` → `llama_model_target_layer_ids()` |
| `tokenizer.ggml.mask_token_id` | 4 | existing mask token key |
| `dspark.markov_rank` | 256 | new |
| `dspark.markov_head_type` | `vanilla` | new |
| `dspark.enable_confidence_head` | true | new |
| `dspark.confidence_head_with_markov` | true | new |
| `dspark.attention.causal` | `0` | `LLM_KV_ATTENTION_CAUSAL` (must be false) |

### Tensor mapping (HF → GGUF)

| HF | GGUF | Notes |
|----|------|-------|
| `embed_tokens.weight` | `token_embd.weight` | scaled embedding (`sqrt(hidden_size)`) applied in graph, not baked in |
| `fc.weight` | `fc.weight` | encoder fusion `Linear(L_t × hidden, hidden)`, no bias |
| `hidden_norm.weight` | `enc.output_norm.weight` | encoder RMSNorm after `fc` |
| `norm.weight` | `output_norm.weight` | decoder final RMSNorm |
| `lm_head.weight` | `output.weight` | |
| `layers.{i}.self_attn.q_proj.weight` | `blk.{i}.attn_q.weight` | |
| `layers.{i}.self_attn.k_proj.weight` | `blk.{i}.attn_k.weight` | |
| `layers.{i}.self_attn.o_proj.weight` | `blk.{i}.attn_output.weight` | **no `v_proj`** (`k_eq_v=true`) |
| `layers.{i}.self_attn.q_norm.weight` | `blk.{i}.attn_q_norm.weight` | RMSNorm over `head_dim` |
| `layers.{i}.self_attn.k_norm.weight` | `blk.{i}.attn_k_norm.weight` | RMSNorm over `head_dim` |
| `layers.{i}.self_attn.v_norm` | **(absent)** | `with_scale=False` → **no `v_norm.weight` tensor**; graph RMS-normalizes V with no learnable scale |
| `layers.{i}.input_layernorm.weight` | `blk.{i}.attn_norm.weight` | |
| `layers.{i}.post_attention_layernorm.weight` | `blk.{i}.post_attention_norm.weight` | |
| `layers.{i}.pre_feedforward_layernorm.weight` | `blk.{i}.ffn_norm.weight` | |
| `layers.{i}.post_feedforward_layernorm.weight` | `blk.{i}.post_ffw_norm.weight` | |
| `layers.{i}.mlp.{gate,up,down}_proj.weight` | `blk.{i}.ffn_{gate,up,down}.weight` | |
| `layers.{i}.layer_scalar` | `blk.{i}.layer_out_scale.weight` | registered buffer; **all-ones (no-op) in released ckpts** — emit for parity or assert-and-skip |
| `markov_head.markov_w1.weight` | `markov.w1.weight` | `Embedding(vocab, markov_rank)` |
| `markov_head.markov_w2.weight` | `markov.w2.weight` | `Linear(markov_rank, vocab)`, no bias |
| `confidence_head.proj.weight` | `confidence.proj.weight` | `Linear(hidden + markov_rank, 1)` when `confidence_head_with_markov` |
| `confidence_head.proj.bias` | `confidence.proj.bias` | |

HF checkpoints have **no** `model.` prefix on tensor names; converter must normalize like DFlash.

**Absent tensors the converter must tolerate (not error on):** `v_proj` (any layer), `v_norm.weight` (any layer). For `dflash_*` ablation checkpoints, also `markov.*` and `confidence.*` (see Phase 4). The Gemma4 norm names above follow the conventions already used by `conversion/gemma.py` for Gemma3/4 (`post_attention_norm`, `post_ffw_norm`); confirm exact GGUF tensor enum spellings against `gguf-py/gguf/tensor_mapping.py` when wiring.

### Convert command

```bash
python convert_hf_to_gguf.py deepseek-ai/dspark_gemma4_12b_block7 \
  --target-model-dir google/gemma-4-12B-it \
  --outtype bf16 \
  --outfile dspark_gemma4_12b.gguf
```

### Acceptance

- [ ] `gguf-dump` shows `dspark` arch and correct metadata
- [ ] Tensor round-trip max abs diff vs HF < 1e-5 (fp32 compare)

### Smoke test — `tests/smoke_phase1_convert.py` (llama.cpp)

**Run:**

```bash
cd llama.cpp
python tests/smoke_phase1_convert.py \
  --hf-repo deepseek-ai/dspark_gemma4_12b_block7 \
  --target-model-dir google/gemma-4-12B-it \
  --outfile /tmp/dspark_gemma4_12b_smoke.gguf
```

**Checks:**

1. Conversion completes without error.
2. `gguf-dump /tmp/dspark_gemma4_12b_smoke.gguf` reports `general.architecture = dspark`.
3. Metadata: `dspark.block_size == 7`, `dspark.markov_rank == 256`, `mask_token_id == 4`.
4. `dspark.target_layers == [6, 18, 30, 42, 47]`.
5. **Required tensors present** (checklist, not a brittle total count):
   `token_embd.weight`, `output.weight`, `fc.weight`, `enc.output_norm.weight`, `output_norm.weight`,
   `markov.w1.weight`, `markov.w2.weight`, `confidence.proj.weight`, `confidence.proj.bias`,
   and per-layer `blk.{0..4}.{attn_q,attn_k,attn_output,attn_q_norm,attn_k_norm,attn_norm,post_attention_norm,ffn_norm,post_ffw_norm}.weight`, `blk.{0..4}.layer_out_scale.weight`.
6. **Absent tensors** (must NOT be present, since `k_eq_v=true`): `blk.{i}.attn_v.weight` and any `v_norm` tensor.
7. Spot-check: reload `fc.weight` and compare max abs diff vs HF safetensors < 1e-5.
8. `dspark.attention.causal == 0` (or key absent with loader default false).

**Pass criteria:** All assertions pass; smoke completes in < 2 min with cached HF weights.

---

## Phase 2 — Model graph (`LLM_ARCH_DSPARK`)

**Repo:** llama.cpp

### Registration

| File | Change |
|------|--------|
| `src/llama-arch.h` / `.cpp` | `LLM_ARCH_DSPARK`, KV keys, tensor enums |
| `src/models/models.h` | `struct llama_model_dspark` |
| `src/models/dspark.cpp` | **NEW** — encoder + KV-injection decoder (model on `src/models/dflash.cpp`) |
| `src/llama-model.cpp` | Factory, `has_encoder()`, RoPE type, `llama_model_target_layer_ids` |
| `src/llama-hparams.h` | Markov/confidence hparams |
| `src/llama-graph.h` / `llama-graph.cpp` | Reuse existing DFlash graph inputs (`llm_graph_input_embd` for inject + tokens); no new dual-stream input needed |

Register `llama_model_has_encoder() == true` for `LLM_ARCH_DSPARK` (same as DFlash).

### Graph structure (reuse DFlash two-pass decode)

**Do not invent a single-forward dual-stream graph.** Model `dspark.cpp` directly on `dflash.cpp`: an encoder graph (`graph<true>`) that fuses raw target features, and a decoder graph (`graph<false>`) with two modes selected by `ubatch.embd`:

| Mode | Selected when | Role | Writes to cache? |
|------|---------------|------|------------------|
| **Context inject** | `ubatch.embd != nullptr` | Project fused features → K/V, write to cache at their absolute positions | **Yes** — context K/V (persists across iterations) |
| **Noise decode** | `ubatch.token != nullptr` | Embed `[anchor, MASK × (block_size-1)]`, attend non-causally over `[cached context K/V] + own noise K/V`, emit hidden/logits | Noise K/V written then removed by the driver before the next propose |

This is exactly the DFlash data flow (`process()` injects, `draft()` decodes noise). The only structural change vs DFlash is that DSpark injects the **verify-window increment** each step so the cache accumulates the full-prefix context (DFlash injects each batch the same way — no new mechanism). Equivalence to the reference’s `concat(k_ctx, k_noise)`: attention over `[cached context] + [noise]` with each stream RoPE’d at its own absolute positions is identical to concatenating then attending.

**Why no new graph input:** the context is supplied through the **KV cache** (inject mode), not through a per-forward `target_ctx` tensor. The noise mode is a plain token batch. Both already exist in the DFlash graph.

### Encoder (reuse DFlash encode path)

```
Input: [n_tokens, n_target_layers × n_embd_tgt]   # raw interleaved layer features
cur = RMSNorm(FC @ input)                          # fc + hidden_norm
output → staging buffer (fused target features per token)
```

Called from spec `process()` via `llama_encode(ctx_dft, embd_batch)` after extracting target layer inputs. The fused output is then injected as K/V (context-inject mode) — the decoder does **not** re-apply `fc`/`hidden_norm` (matches the reference, where `hidden_norm(fc(...))` happens once before `k_proj`).

### Decoder — context-inject mode (model on DFlash `ubatch.embd` branch)

Per layer, for the fused context features `inp_g` at their absolute positions:

```
KV    = Wk @ inp_g              # single k_proj output; k_eq_v means V shares this projection
K_ctx = k_norm(KV)             # k_norm has a learnable scale
V_ctx = v_norm(KV)             # v_norm has NO scale (with_scale=False); applied to the SAME pre-norm KV
K_ctx = rope(K_ctx, ctx_positions)   # V is not RoPE'd
cpy_k/cpy_v → kv_cache         # context K/V persists for the rest of the sequence
```

### Decoder — noise mode (model on DFlash token branch)

Per layer (Gemma4 branch). Port the norm and RoPE building blocks from `src/models/gemma4.cpp`: reuse its `build_norm(..., LLM_NORM_RMS)` calls for all RMSNorms, and its proportional-RoPE frequency setup (`global_head_dim`, `partial_rotary_factor`) for `ggml_rope_ext`. Attention uses `kq_scale = 1.0f`:

Gemma4 uses **sandwich norms** (norm before *and* after each sublayer, residual added around the pair). Track the pre-norm residual explicitly:

```
x = embed(tokens) * sqrt(n_embd)          # scaled embed (layer 0 input)

# --- attention sublayer ---
residual = x
h = input_layernorm(x)
Q       = q_norm(Wq @ h)
KV      = Wk @ h                            # single k_proj output (k_eq_v)
K_noise = k_norm(KV)                        # learnable scale
V_noise = v_norm(KV)                        # no scale; same pre-norm KV, NOT RoPE'd
Q       = rope(Q, noise_positions)         # Q at the block_size noise positions
K_noise = rope(K_noise, noise_positions)
attn = build_attn(Q, K_noise, V_noise)     # over [cached context K/V] + noise; causal_attn=false; scale=1.0
attn = post_attention_layernorm(o_proj(attn))
x = residual + attn

# --- feed-forward sublayer ---
residual = x
h = pre_feedforward_layernorm(x)
h = post_feedforward_layernorm(mlp(h))     # mlp = SwiGLU (gate/up/down)
x = residual + h

x = x * layer_scalar                        # constant 1.0 in released ckpt (no-op)
```

After the last layer: `output_norm(x)` → `lm_head` → raw logits (softcap/Markov applied in the driver).

> The reference’s asymmetric RoPE (`Q` uses the last `block_size` positions, `K` uses all `ctx_len+block_size`) is reproduced automatically by the two-pass split: context K is RoPE’d at its positions during inject, noise Q/K at the noise positions during decode. No manual position slicing is required.

**Gemma4-specific (must implement, not optional):**

| Component | Detail |
|-----------|--------|
| Scaled embedding | `embed * sqrt(hidden_size)` on noise stream |
| `k_eq_v` | No `v_proj`. K and V share one `k_proj` output, then **K gets `k_norm` (scaled), V gets `v_norm` (no scale)** — so K ≠ V after norm. V is not RoPE'd. |
| Attention scale | `1.0` (override default `1/sqrt(d)`) |
| Layer norms | `input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`, `post_feedforward_layernorm` (all four per layer; `v_norm` has **no** learnable scale) |
| `layer_scalar` | Per-layer multiply. **Registered buffer initialized to `ones(1)` and never trained → constant `1.0` (no-op) in released checkpoints.** Implement the multiply for parity, but it can be skipped if the converted tensor is verified all-ones. |
| RoPE | Proportional, `global_head_dim=512`, `partial_rotary_factor=0.25`; reuse Gemma4 proportional freq logic |
| GQA | `num_global_key_value_heads=1`; repeat KV to match Q heads |

**Qwen3 branch (Phase 4):** separate path with `v_proj`, `1/sqrt(d)` scale, standard RoPE, no softcap.

### Outputs

- `t_logits` — raw `lm_head` output (**no** softcap in graph)
- Softcap + Markov applied in **spec driver** (Phase 3) in documented order

### Gate

`tests/test-dspark-graph.cpp`: decoder logits vs PyTorch `_forward_backbone` + `lm_head` on fixed tensors, NMSE < threshold (bf16). Include a case with **non-empty draft KV** (iteration ≥ 2 shape).

### Smoke test — `tests/smoke_phase2_graph.cpp` (llama.cpp)

**Build:**

```bash
cmake -B build -DLLAMA_BUILD_TESTS=ON
cmake --build build --target smoke_phase2_graph -j
```

**Run:**

```bash
./build/bin/smoke_phase2_graph \
  --draft-model /tmp/dspark_gemma4_12b_smoke.gguf
```

**Checks:**

1. `llama_model_load` succeeds; arch is `LLM_ARCH_DSPARK`; `causal_attn == false`.
2. **Encoder smoke:** feed synthetic raw target features `[n_tokens=3, n_embd_enc=5×3840]`; output shape `[3, 3840]`; no NaN/Inf.
3. **Context-inject smoke (ctx_len=3):** `llama_encode` the synthetic features, then inject as K/V at positions `0..2` (embd-batch decode); cache length grows to 3.
4. **Noise-decode smoke (with cached context):** decode a `block_size=7` noise block `[anchor, MASK×6]` over the injected context; forward completes; emits logits for all 7 positions.
5. **Iteration ≥ 2 smoke:** crop noise (`seq_rm` to anchor), inject one more context row (ctx_len grows to 4), decode a second noise block; logits differ from the first iteration (proves the cache accumulates context + non-causal path).
6. **Logits NMSE:** compare position-0 logits (first 100 vocab indices) vs `tests/data/dspark_gemma4_decoder_logits_ref.bin`; NMSE < 0.01 (bf16).
7. **Softcap sanity:** test harness applies `tanh(x/30)*30` separately; output differs from uncapped.

**Pass criteria:** Exit 0; no graph build or decode failures.

**Note:** Reference blob generated once by `tests/gen_dspark_graph_ref.py` from DeepSpec PyTorch on **synthetic** fixed tensors (no full target model needed).

---

## Phase 3 — Speculative driver (`draft-dspark`)

**Repo:** llama.cpp `common/speculative.cpp`, `common/arg.cpp`, `tools/server/server-context.cpp`

Split into **3a** (Markov + core loop), **3b** (confidence), **3c** (rejection sampling). Each has its own smoke test.

### Phase 3a — Core spec driver + vanilla Markov

#### New type

- `COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK`
- CLI / server: `--spec-type draft-dspark`
- Clamp `--spec-draft-n-max` to **`block_size`** (not `block_size - 1` like DFlash)

#### Driver init (constructor)

Mirror DFlash setup, plus DSpark-specific flags:

```cpp
llama_set_causal_attn(ctx_dft, false);
for (k : target_layer_ids) llama_set_embeddings_layer_inp(ctx_tgt, k, true);
```

#### State machine (per sequence)

Because the context lives in the **draft KV cache** (injected per-token at absolute positions, DFlash-style), the driver does **not** keep a rolling fused-feature buffer. The only state it needs:

```
draft_start   # anchor index; mirrors DeepSpec `start` (= committed-token count)
prev_token    # autoregressive anchor for the Markov head (= last accepted token)
```

The draft KV length after each propose’s crop equals `draft_start`. At the *next* propose entry it is still `draft_start` of the **previous** iteration, i.e. `current draft_start - (accepted+1)` — the gap is exactly the context rows that `process()` injects this iteration.

#### `process()` — after each target decode (prefill + each verify)

Identical to DFlash `process()`, with the Gemma4 inject branch:

1. Extract `llama_get_embeddings_layer_inp(ctx_tgt, lid)` for each `target_layer_ids` entry
2. Interleave into `[n_tokens × (L_t × n_embd_tgt)]` (raw concat, same layout as DeepSpec)
3. `llama_encode(ctx_dft, embd_batch)` → fused features (`fc` + `hidden_norm`)
4. Inject the fused features as **context K/V** into the draft cache at the tokens’ **absolute positions** (`batch_inject` with `llama_decode`, `logits=false`). On prefill this injects the whole prompt; on each verify it injects only the `verify_len` new rows, so the cache accumulates the full-prefix context. (Position-indexed injection also overwrites any leftover noise slots from the previous `draft()`.)

> No fused buffer is replaced/truncated in the driver — the cache is the source of truth, and `_update`-style truncation is unnecessary because injection is position-addressed.

#### `draft()`

1. Build noise block `[anchor, MASK × (block_size-1)]` at positions `draft_start .. draft_start+block_size-1`
2. Run the noise-decode forward (token batch) → raw `lm_head` logits + hidden for all `block_size` positions. The noise attends over the cached context (positions `0 .. draft_start-1`) + its own keys; no `cache_len == draft_start` assertion is needed — context comes from the cache, not a position slice.
3. **CPU logits pipeline (per position `i = 0 .. block_size-1`, autoregressive Markov only — backbone is parallel):**
   - `base = lm_head(hidden_i)` (from graph)
   - `base = tanh(base / 30) * 30` (Gemma4 only)
   - `logits = base + markov_bias(prev_token)`   (`prev_token` for `i=0` is the anchor)
   - sample / argmax → emit draft token; set `prev_token = sampled`
4. **Confidence truncation (Phase 3b):** if threshold > 0
5. **Remove noise K/V from the draft cache:** `llama_memory_seq_rm(ctx_dft, seq, draft_start, -1)` — drops positions `≥ draft_start` (the noise block), leaving the accumulated context. The next iteration’s `process()` will inject the newly committed context at positions `draft_start ..`.

#### `accept(n_accepted)` — no-op for cache, advances driver state

Like DFlash, the cache needs no rollback here (the next `process()` injects context at the correct absolute positions and overwrites any stale slots). The driver only advances:

1. `draft_start += n_accepted + 1`
2. `prev_token = last accepted token` (anchor for the next block’s Markov position 0)

#### Run command (local / nightly — bf16 target for parity)

```bash
./build/bin/llama-cli \
  -m gemma-4-12b-it-bf16.gguf \
  -md dspark_gemma4_12b.gguf \
  --spec-type draft-dspark \
  --spec-draft-n-max 7 \
  -ngl 99 \
  --device Vulkan0 \
  --input-ids tests/data/dspark_gemma4_12b_input_ids.json \
  -n 32 --temp 0 --seed 42
```

**Binding:** Phase 3a delivers a `--input-ids <path.json>` CLI flag (in `common/arg.cpp`) that loads a JSON array of token IDs and feeds them directly, bypassing tokenization. This is the *only* sanctioned input path for parity runs — do not use `--prompt`/`--input-file` (raw text) for the token-match gate, because re-tokenization would break the shared-fixture invariant. The same loader backs the smoke binary's `--input-ids` argument.

#### Gate (3a)

Token sequence matches Phase 0 reference at `temp=0`, **same `input_ids`**, **bf16/f16 target**.

#### Smoke test — `tests/smoke_phase3a_speculative.cpp` (llama.cpp)

**Prerequisites:** Phase 1 GGUF, Phase 2 graph, Phase 0 reference + `input_ids` fixture.

**Requires full target:** `google/gemma-4-12B-it` converted to **bf16/f16** GGUF. Do **not** use a truncated “tiny” target — the draft was trained on layer features from the full 12B model; a tiny target produces unrelated hidden states and will fail token match even with a correct graph.

**Run (nightly / manual):**

```bash
./build/bin/smoke_phase3a_speculative \
  --target-model  /path/to/gemma-4-12b-it-bf16.gguf \
  --draft-model   /tmp/dspark_gemma4_12b_smoke.gguf \
  --input-ids     ../DeepSpec/tests/data/dspark_gemma4_12b_input_ids.json \
  --reference     ../DeepSpec/tests/data/dspark_gemma4_12b_reference.jsonl \
  --temp 0 --seed 42 -n 32
```

**Checks:**

1. `--spec-type draft-dspark` registers without error (CLI + server).
2. Target + draft contexts load; vocab check passes.
3. Generation completes without crash.
4. Output token IDs **exact match** Phase 0 reference JSONL final `output_token_ids`.
5. Per-step invariant checks from reference (RoPE length, KV crop) hold in debug build.
6. `stats.accepted_draft_tokens_count > 0` for at least one step.
7. **Markov sanity:** test flag disables Markov → token sequence diverges from reference (proves path is exercised).

**Pass criteria:** Exit 0; token sequence match on bf16 target.

**Optional local smoke (quantized target — no token match expected):**

```bash
./build/bin/llama-cli -m gemma-4-12B-it.Q4_K_M.gguf -md dspark_gemma4_12b.gguf \
  --spec-type draft-dspark --spec-draft-n-max 7 \
  --device Vulkan0 -ngl 99 --input-file ... -n 16 --temp 0
# Pass: completes without backend error (acceptance may differ from bf16 reference)
```

---

### Phase 3b — Confidence head truncation

Add CLI flag `--dspark-confidence-threshold <float>` (exact name, in `common/arg.cpp`); default `0.0` preserves full-block proposals.

**Implementation (match DeepSpec `build_dspark_proposal`):**

1. After Markov sampling loop produces `sampled_tokens[0..block_size-1]` and hidden states:
2. Build `prev_token_ids = [anchor, sampled[0], ..., sampled[block_size-2]]`
3. If `confidence_head_with_markov`:
   - `features = concat(hidden_states, markov_w1(prev_token_ids), dim=-1)` → shape `[block_size, hidden_size + markov_rank]`
4. Else: `features = hidden_states`
5. `conf_logit = confidence.proj(features)` → `[block_size]`
6. Truncation: if `sigmoid(conf_logit[i]) < threshold` at position `i`, proposal length = `i` (may be **0** → empty proposal, verify anchor only)
7. Weights: `confidence.proj` is `Linear(hidden_size + markov_rank, 1)` for Gemma4 checkpoint (4096 → 1)

### Smoke test — `tests/smoke_phase3b_confidence.cpp` (llama.cpp)

**Run (nightly; same full bf16 target as 3a):**

```bash
./build/bin/smoke_phase3b_confidence \
  --target-model  /path/to/gemma-4-12b-it-bf16.gguf \
  --draft-model   /tmp/dspark_gemma4_12b_smoke.gguf \
  --input-ids     ../DeepSpec/tests/data/dspark_gemma4_12b_input_ids.json \
  --temp 0 --seed 42 -n 64
```

**Checks:**

1. With `confidence_threshold=0.0`: mean proposal length == `block_size` (7).
2. With `confidence_threshold=0.9`: mean proposal length < `block_size`.
3. With `confidence_threshold=0.9`: at least one step has `proposal_len == 0`.
4. With `threshold=0.0`: output still matches Phase 0 reference.

**Pass criteria:** Exit 0; proposal length ordering: `mean_len(0.9) < mean_len(0.0)`.

---

### Phase 3c — Rejection sampling (`temp > 0`)

DeepSpec verifies with rejection sampling using **post-Markov** probabilities as draft probabilities (`draft_probs = logits_to_probs(post_markov_logits, temp)` in `verify_draft_tokens`). The accept rule is `accept_prob = clamp(p_target[tok] / p_draft[tok], max=1)`, and on rejection the next token is drawn from the residual `sample_residual(p_target, p_draft)`. Required for parity at non-zero temperature.

> **temp=0 reduces to greedy token equality** (so Phase 3a needs none of this): at `temp=0` both `p_target` and `p_draft` are one-hot argmax distributions, so `accept_prob = 1` iff `target_argmax == draft_token`, else `0`, and the residual draw becomes the target argmax. This is why the Phase 3a gate is exact-match and the `rand` draw in verify is irrelevant there.

**Integration note:** llama.cpp’s built-in speculative verify (`common/speculative.cpp`) does **not** consume an arbitrary per-token draft-probability vector — it accepts/rejects using the draft sampler’s own distribution. Supporting DSpark-correct rejection therefore requires real changes to the verify/accept path, not just “pass `draft_probs`.”

**Implementation (locked decision #5b — option a):** extend the speculative verify path to accept an externally supplied per-token post-Markov `draft_probs` tensor and apply `clamp(p_t/p_d,1)` + residual sampling, matching `verify_draft_tokens`. (The temp=0-only fallback is reserved for the #5b review-time contingency and is not an implementation choice — see [Locked decisions](#locked-decisions-binding).)

**Tasks:**

1. In `draft()`, store the per-position **post-Markov** probability vector for each proposed token (softmax at the run temperature), keyed by sequence + block position.
2. Thread these into the verify path so acceptance uses `min(p_target/p_draft, 1)` with DSpark-correct draft probs, and the rejection fallback uses the residual distribution.
3. Validate: at `temp=0`, identical to Phase 3a (greedy); at `temp>0`, acceptance statistics match DeepSpec within tolerance.

### Smoke test — `tests/smoke_phase3c_rejection.cpp` (llama.cpp)

**Run (nightly):** fixed seed, `temp=0.7`, compare acceptance-length distribution vs Phase 0 reference run at same temperature (statistical tolerance, not exact tokens).

**Pass criteria:** Mean acceptance length within 10% of DeepSpec reference over ≥ 20 propose steps.

---

## Phase 4 — Qwen3 + DFlash-mode

- Add `Qwen3DSparkModel` to converter (`conversion/qwen.py`)
- Qwen3 graph branch: Q/K norms, GQA, `1/sqrt(d)` scale, no logit softcap
- Support `deepseek-ai/dflash_*` (`markov_rank=0`) without Markov/confidence code paths
- Checkpoints: `dspark_qwen3_*`, `dflash_qwen3_*`

### Smoke test — `tests/smoke_phase4_qwen3.cpp` (llama.cpp)

**Run:**

```bash
./build/bin/smoke_phase4_qwen3 \
  --dspark-hf    deepseek-ai/dspark_qwen3_4b_block7 \
  --dflash-hf    deepseek-ai/dflash_qwen3_4b_block7 \
  --target-hf    Qwen/Qwen3-4B
```

**Checks:**

1. **Convert smoke:** `Qwen3DSparkModel` → GGUF with arch `dspark`; `dspark.target_layers == [2, 10, 18, 26, 34]`.
2. **DFlash ablation convert:** `dflash_qwen3_4b_block7` → GGUF; `markov_rank == 0`; markov/confidence tensors **absent**.
3. **Graph smoke:** Qwen3 decoder forward on 4B draft GGUF completes (no Gemma4 softcap path).
4. **Spec smoke (dflash ablation):** `--spec-type draft-dspark` with dflash GGUF runs without Markov code path; generates ≥ 8 tokens without crash.
5. **Spec smoke (full dspark):** Qwen3-4B dspark pair; token match vs Qwen3 Phase 0 reference (`dspark_qwen3_4b_input_ids.json` + `dspark_qwen3_4b_reference.jsonl`, same contract as Gemma4 Phase 0).

**Pass criteria:** Exit 0; both ablation and full Qwen3 paths load and run.

---

## Phase 5 — CI integration & release

Wire smokes into **tiered** CI and document user-facing usage.

### CI tiers

| Job | Trigger | Steps |
|-----|---------|-------|
| `dspark-smoke-pr` | Every PR touching `dspark`/speculative code | Phase 1 convert + Phase 2 graph (synthetic tensors; no 12B download) |
| `dspark-smoke-nightly` | Nightly / manual | Phase 3a–4 with cached bf16 Gemma4 12B + Qwen3 weights |
| `dspark-smoke-vulkan` | Optional nightly | llama-cli `--device Vulkan0`; crash-free only |
| `dspark-reference` (DeepSpec) | DeepSpec CI | Phase 0 smoke |

```yaml
# Pseudocode — llama.cpp GitHub Actions
jobs:
  dspark-smoke-pr:
    steps:
      - run: python tests/smoke_phase1_convert.py ...
      - run: ./build/bin/smoke_phase2_graph ...

  dspark-smoke-nightly:
    steps:
      - run: cache restore gemma-4-12b-it-bf16.gguf
      - run: ./build/bin/smoke_phase3a_speculative ...
      - run: ./build/bin/smoke_phase3b_confidence ...
      - run: ./build/bin/smoke_phase3c_rejection ...
      - run: ./build/bin/smoke_phase4_qwen3 ...
```

### DeepSpec CI (Phase 0)

```yaml
jobs:
  dspark-reference:
    steps:
      - run: bash scripts/smoke_phase0_reference.sh
```

### Docs deliverables

| File | Content |
|------|---------|
| `docs/speculative.md` | DSpark convert + run examples; bf16 vs quant note |
| `docs/dspark-port-validation.md` | Phase 0 findings + invariant summary |
| `tests/data/README.md` | `input_ids` fixture, graph ref blob generation, nightly weight paths |

### Release smoke (manual, pre-merge)

**A. Parity (bf16 target, temp=0):** must match Phase 0 reference.

```bash
./build/bin/llama-cli \
  -m gemma-4-12b-it-bf16.gguf \
  -md dspark_gemma4_12b.gguf \
  --spec-type draft-dspark --spec-draft-n-max 7 \
  --input-file tests/data/dspark_gemma4_12b_input_ids.json \
  -n 32 --temp 0 --seed 42
# Pass: output token IDs match reference
```

**B. Production (quantized target, temp>0):** crash-free + perf only.

```bash
./build/bin/llama-cli \
  -m gemma-4-12b-it.Q4_K_M.gguf \
  -md dspark_gemma4_12b.gguf \
  --spec-type draft-dspark --spec-draft-n-max 7 \
  --device Vulkan0 -ngl 99 \
  -p "Explain speculative decoding." -n 128 --temp 0.7
# Pass: no crash; tokens/sec > non-spec baseline (acceptance may differ from bf16)
```

### Phase 5 acceptance

- [ ] PR CI runs phases 1–2 on every relevant PR
- [ ] Nightly CI runs phases 3a–4 with full weights
- [ ] Phase 0 smoke runs in DeepSpec CI
- [ ] `docs/speculative.md` updated
- [ ] Server accepts `--spec-type draft-dspark`
- [ ] Vulkan nightly smoke green (optional)

---

## Risk register

| Risk | Mitigation |
|------|------------|
| Context vs noise KV roles (easy to invert) | **Context K/V is cached** (accumulates full prefix); **noise is transient** (cropped each propose). Reuse DFlash inject/decode split; Phase 2 iteration-≥2 test |
| Incremental context injection / crop semantics | Inject verify-window increment per `process()` at absolute positions; remove noise with `seq_rm` to anchor `start`; Phase 0 per-step invariants |
| Bidirectional attention at inference | `causal_attn=false` in GGUF + `llama_set_causal_attn`; Phase 2 iteration-2 test |
| Gemma4 proportional RoPE (Q/K asymmetric) | Reproduced by two-pass split (context K rope’d at inject, noise Q/K at decode); port freqs from `src/models/gemma4.cpp`; test vs PyTorch |
| `target_hidden_states` replace vs append | Cache is source of truth; inject increment in `process()`, no driver-side fused buffer to truncate |
| Tokenization mismatch (chat template) | Shared `input_ids` fixture; Phase 0/3a load same file |
| Tiny target + real draft in CI | **Removed** — PR CI uses synthetic graph tests only; E2E requires full 12B bf16 |
| Quantized target vs bf16 reference | Exact match gates use bf16; quant is crash/perf smoke only |
| Markov + softcap ordering | Documented pipeline; CPU driver applies in order |
| Markov on CPU vs GPU | **CPU only** (locked #8); GPU graph fusion is out of scope for v1 |
| Partial accept KV rollback | Noise removed with `llama_memory_seq_rm` to anchor `start` after each propose; context overwritten by next `process()` inject (no explicit rollback) |
| Confidence head feature dim | `hidden_size + markov_rank` when `confidence_head_with_markov`; `confidence.proj` is `Linear(4096, 1)` for Gemma4 |
| Rejection sampling at temp > 0 | Phase 3c; requires verify-API change to accept external post-Markov `draft_probs` (or restrict v1 to temp=0) |
| `v_proj` / `v_norm.weight` absent (`k_eq_v`) | Converter tolerates missing tensors; graph sets `V = K` and applies `v_norm` with no learnable scale |
| Checkpoint/target availability (gated/unreleased) | `gemma-4-12B-it` + `dspark_gemma4_12b_block7` may be gated; nightly CI uses cached weights; converter reads dims from `config.json`, never hardcodes |
| Target layer features from quant target | Strategy A for embed/lm_head; document acceptance drift for Q4 |

---

## File change index

### llama.cpp — new files

- `src/models/dspark.cpp`
- `tests/smoke_phase1_convert.py`
- `tests/smoke_phase2_graph.cpp`
- `tests/smoke_phase3a_speculative.cpp`
- `tests/smoke_phase3b_confidence.cpp`
- `tests/smoke_phase3c_rejection.cpp`
- `tests/smoke_phase4_qwen3.cpp`
- `tests/gen_dspark_graph_ref.py`
- `tests/data/dspark_gemma4_decoder_logits_ref.bin`
- `tests/data/README.md`
- `docs/dspark-port-validation.md`
- `docs/speculative.md` (DSpark section)

### llama.cpp — modified files

- `gguf-py/gguf/constants.py`, `gguf_writer.py`, `tensor_mapping.py`
- `conversion/gemma.py`, `conversion/__init__.py`
- `src/llama-arch.h`, `src/llama-arch.cpp`
- `src/models/models.h`, `src/llama-model.cpp`, `src/llama-hparams.h`
- `common/common.h`, `common/speculative.cpp`, `common/arg.cpp`
- `tools/server/server-context.cpp` (draft-dspark init, same as DFlash/Eagle3)
- `tests/test-llama-archs.cpp`

### DeepSpec — new files

- `scripts/validate_dspark_reference.py`
- `scripts/smoke_phase0_reference.sh`
- `tests/data/dspark_gemma4_12b_input_ids.json` (shared token fixture)
- `tests/data/dspark_gemma4_12b_reference.jsonl` (generated)
- `tests/data/dspark_qwen3_4b_input_ids.json` (Phase 4)
- `tests/data/dspark_qwen3_4b_reference.jsonl` (Phase 4, generated)

---

## Locked decisions (binding)

**Policy:** every item below is a binding choice, not a recommendation. The implementer does **not** re-evaluate these. If a locked decision turns out to be infeasible during implementation, **stop and escalate to the plan owner** — do not silently pick an alternative inline. There are no remaining open choices in this plan.

| # | Question | Locked answer |
|---|----------|---------------|
| 1 | embed/lm_head source | **Load from draft GGUF** (Strategy A). Converter always emits `token_embd` + `output`; no `ctx_other` for these. |
| 2 | Confidence head in v1 | **Implement in Phase 3b**, default threshold `0.0` (full-block). Weights always converted (Phase 1). |
| 3 | `gated` / `rnn` Markov | **Vanilla only.** Converter and loader **assert-fail** on `markov_head_type != "vanilla"`. (Out of scope below.) |
| 4 | Upstream target | PR to `ggml-org/llama.cpp` `master`. |
| 5 | DSpark graph inputs | **Reuse DFlash graph inputs** (`llm_graph_input_embd` for context-inject + token noise batch). No new dual-stream input or `target_ctx` setter. |
| 5b | Phase 3c rejection sampling at temp>0 | **Implement option (a):** extend the speculative verify path to accept an externally supplied per-token post-Markov `draft_probs` tensor and apply `clamp(p_t/p_d,1)` + residual sampling. (Temp=0-only is the fallback **only** if escalation in #5b's contingency below is invoked.) |
| 6 | E2E CI for 12B target | **Nightly only** with cached bf16 weights; PR CI = phases 1–2. |
| 7 | Token fixture format | **JSON array of token IDs** at `tests/data/dspark_*_input_ids.json`, loaded via the `--input-ids` flag (Phase 3a). |
| 8 | Markov + confidence compute location | **CPU** in the spec driver (no GPU graph fusion). |
| 9 | Confidence CLI flag | `--dspark-confidence-threshold <float>` (exact spelling). |
| 10 | Quantized-target support | Convert/run path must not crash, but **no token-match guarantee**; parity gates use bf16/f16 target only. |

**#5b contingency (pre-defined, not an inline choice):** if upstream maintainers reject the verify-API extension during review, the *only* sanctioned fallback is to ship v1 with `temp=0` greedy acceptance and mark `temp>0` as unsupported in `docs/speculative.md`. This is a review-time escalation outcome, not an implementation-time decision.

### Explicitly out of scope for v1 (do NOT implement)

These are listed so no one builds them "just in case." Building any of these is a scope violation, not a judgment call:

- `--share-embed-with-target` / any `ctx_other` sharing of embed/lm_head (Strategy B).
- `gated` and `rnn` Markov head types (assert-fail instead).
- GPU/graph fusion of the Markov or confidence heads (CPU only).
- LM Studio integration (stated in the doc header).
- Gemma4 MoE blocks and per-layer-input gates (the reference asserts these are disabled; the loader must assert the same).
- Any decode path that injects the *whole* prefix context each step (only the verify-window increment is injected; see Phase 2/3).

---

## References

- DeepSpec DSpark modeling: `deepspec/modeling/dspark/`
- DeepSpec eval: `deepspec/eval/dspark/`
- llama.cpp DFlash (**structural template** — encoder + KV-injection decoder + spec driver): `src/models/dflash.cpp`, `common/speculative.cpp`
- llama.cpp speculative docs: `docs/speculative.md`
- Released checkpoint: https://huggingface.co/deepseek-ai/dspark_gemma4_12b_block7
- Target model: https://huggingface.co/google/gemma-4-12B-it

---

*Last updated: 2026-06-29. Each phase includes a mandatory smoke test before proceeding. Critical review amendments: graph input contract, tokenization fixture, tiered CI, bf16 E2E gate. Second review (KV semantics): context K/V is cached/accumulated and noise is transient (roles corrected); Phase 2 reuses the DFlash encoder + KV-injection decoder instead of a bespoke single-forward dual-stream graph; Phase 3 driver cache_len/accept semantics corrected; converter tensor map completed (`v_proj`/`v_norm` absent, `layer_scalar` no-op, four norms); Phase 3c rejection-sampling integration cost made explicit. Third pass (decision lockdown): all "open decisions" converted to binding locked decisions with a no-inline-choice policy; #5b resolved to option (a) with a pre-defined review-time contingency; added an "Explicitly out of scope for v1" list; locked CLI flag names (`--input-ids`, `--dspark-confidence-threshold`), token-fixture format, embed/lm_head source (Strategy A), and CPU-only Markov/confidence; removed remaining `e.g.`/`or`/`where applicable`/`if not yet available` hedges.*
