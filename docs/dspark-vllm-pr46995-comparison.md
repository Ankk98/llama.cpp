# DSpark: vLLM PR #46995 vs llama.cpp `ft-dspark`

Living reference comparing NVIDIA's upstream vLLM DSpark integration
([PR #46995](https://github.com/vllm-project/vllm/pull/46995), author benchislett)
with the llama.cpp port on branch `ft-dspark`.

Related docs:

- [dspark-llamacpp-port-plan.md](dspark-llamacpp-port-plan.md) - implementation plan
- [dspark-port-validation.md](dspark-port-validation.md) - DeepSpec reference contract
- [dspark-benchmark-experiments.md](dspark-benchmark-experiments.md) - Vulkan fair benchmarks
- [dspark-benchmark-nvidia.md](dspark-benchmark-nvidia.md) - CUDA benchmarks

**Last reviewed:** 2026-06-30 (PR open, 9 commits on `benchislett:dspark`)

---

## Executive summary

Both implementations follow the same paper semantics and extend the DFlash pattern.
The llama.cpp architectural choices (DFlash-style KV injection, anchor-first query
layout, sequential Markov after parallel backbone) align with upstream vLLM.

| Area | vLLM #46995 | llama.cpp `ft-dspark` |
|------|-------------|----------------------|
| Primary models | DSV4-Pro-DSpark, Qwen3 + `dspark_qwen3_8b_block7` | Gemma4 12B + `dspark_gemma4_12b_block7` |
| Draft backbone | `DSparkSpeculator` subclasses `DFlashSpeculator` | `LLM_ARCH_DSPARK` (encoder + KV-inject decoder) |
| Non-causal attn | SparseMLA top-k index expansion | `causal_attn=false` in graph |
| Target features | EAGLE3 `aux_hidden_states` pathway | Layer-inp extraction at `dspark.target_layers` |
| CUDA graphs | Full draft step (backbone + Markov sampling) | Verify warmup only; no full propose graph |
| Confidence scheduling | Out of scope for PR | Implemented; net loss on Q4 (see benchmark docs) |
| Main perf gap | N/A (datacenter scale) | CUDA batched verify correctness; Vulkan verify latency |

---

## High-level alignment

| Concept | vLLM #46995 | llama.cpp `ft-dspark` |
|---------|-------------|----------------------|
| Draft backbone | Parallel non-causal block forward | Encoder + KV-inject decoder (`causal_attn=false`) |
| Query layout | N tokens (anchor + N-1 masks); anchor predicts first draft | Same: anchor at pos 0; `n_draft+1` decode when confidence off |
| Markov head | Sequential left-to-right after parallel forward | Same ordering (softcap -> Markov bias -> sample) |
| Target conditioning | Fused hidden states from target layers | Layer-inp extraction at `dspark.target_layers` |
| Verify | Batched target forward + rejection sampling | `verify_batched()` + greedy accept at temp=0 |
| Confidence scheduling | Implemented but deferred | Implemented; benchmarks show net loss on Q4 |

A comment on the PR from ishan5ain documents a failed hand-rolled integration
(custom `_dspark_context_buffer`, hand-rolled non-causal attention) that got 0%
acceptance before pivoting to the upstream PR approach. That validates the
llama.cpp decision to model on DFlash encoder + KV-injection rather than invent
a new attention path.

---

## Where vLLM does things differently

### 1. Non-causal attention via SparseMLA top-k expansion

vLLM does not add a new attention kernel. It reuses existing SparseMLA backends
and expands top-k indices so every query in the draft block includes all other
queries (plus the sliding window). A Triton kernel
(`_compute_dspark_noncausal_swa_indices_kernel`) builds the index list;
`tests/v1/attention/test_dspark_noncausal_sparse_mla.py` locks correctness.

llama.cpp sets `causal_attn=false` in the graph directly. Simpler, works on
Vulkan/CUDA without sparse-MLA infrastructure, but does not get kernel reuse on
DSV4-class models.

**Takeaway:** For sparse-MLA targets (DSV4), the top-k expansion trick is the
path of least resistance. For Gemma4 dense attention, explicit non-causal graph
is fine.

### 2. CUDA graph captures the entire draft step

vLLM captures backbone forward and sequential Markov sampling in one CUDA graph
(`DFlashCudaGraphManager`, `FULL_DECODE_ONLY`). Reported on 8xB300 (BS1, 7 draft
tokens, CUDA graphs + prefix caching):

| Phase | Latency |
|-------|---------|
| Target verify | ~11-13 ms |
| DFlash backbone | ~0.6 ms |
| Markov sampling | ~0.6 ms |
| E2E step | ~14 ms |

With acceptance length ~5, that works out to >350 TPS on SPEED-Bench coding.

llama.cpp has verify graph warmup and optional fused argmax, but no equivalent
full-step CUDA graph for draft + Markov. On RTX 3090: draft ~4 ms, verify ~16 ms
(batched) or ~12.6 ms (sequential).

**Takeaway:** CUDA graph capture of the full propose loop is likely the biggest
infra gap vs vLLM on NVIDIA, separate from acceptance rate.

### 3. Target hidden states via EAGLE3 `aux_hidden_states`

vLLM routes target context through the existing EAGLE3 pathway
(`SupportsEagle3` / `EagleModelMixin`), then `combine_hidden_states()` does
`main_norm(main_proj(concat(aux)))`. For DSV4 they mean-pool hyper-connection
copies before fusion.

llama.cpp extracts layer inputs directly (`llama_set_embeddings_layer_inp`).
Semantically equivalent for Gemma4. A PR review thread asks whether mean-pooling
is correct vs passing full `hidden_states` - worth confirming against DeepSpec
for DSV4 if expanding beyond Gemma4.

### 4. Embed / lm_head sharing

| Model | vLLM | llama.cpp |
|-------|------|-----------|
| DSV4 (weights in target ckpt) | Shares target `embed_tokens` + `lm_head` | Loads from draft GGUF (Strategy A, locked in port plan) |
| Qwen3 dense draft | Self-contained (`dspark_shares_target_embeddings = False`) | Self-contained |

llama.cpp loads embed/lm_head from the draft GGUF to avoid silent mismatch when
the target is quantized differently from the frozen draft copy. vLLM's sharing is
natural when draft weights live inside the target checkpoint.

### 5. Model coverage is inverted

| Checkpoint | vLLM #46995 | llama.cpp |
|------------|-------------|-----------|
| `DeepSeek-V4-Pro-DSpark` | Primary | Not in scope |
| `dspark_qwen3_8b_block7` | Supported | Phase 4 (planned) |
| `dspark_gemma4_12b_block7` | Explicitly deferred | Primary |

PR scope note: Gemma4-DSpark left for follow-up; author suggests training a
Qwen3 DSpark and reusing the current implementation instead.

---

## Results comparison

Numbers are not directly comparable (different models, hardware, quant, concurrency)
but the shape is informative.

| Metric | vLLM (DSV4-Pro, 8xB300) | vLLM (Hopper, fank field report) | llama.cpp Vulkan (Gemma4 Q4) | llama.cpp CUDA (Gemma4 Q4) |
|--------|-------------------------|----------------------------------|------------------------------|----------------------------|
| Draft positions | 7 | 5 | 4 (`n_max`, block_size=7) | 4 |
| Accept/step | ~3.4 avg @ DL6 vs MTP ~2.6 (1.31x) | 3.86 (57.2% per-token) | 2.50 | 2.54 |
| Verify latency | ~11-13 ms | - | ~69 ms (bottleneck) | ~16 ms batched / ~12.6 ms seq |
| Draft latency | ~1.2 ms total | - | ~19 ms | ~4 ms |
| Throughput speedup | >350 TPS (absolute) | ~1.75x vs serial MTP | 1.57x fair | 0.90x (seq, correct) / ~2.0x (batched, wrong) |
| Token match | GSM8k passes | Coherent, no loops (greedy) | YES | YES (sequential) |

### vLLM acceptance vs MTP (thinking ON, probabilistic accept, max output 4096)

DSpark / MTP ratio at draft length 6, category average: **1.31x** (3.42 vs 2.60).
DSpark matches or slightly trails MTP at position 1, then pulls ahead from
position 2 onward. Full table in PR description.

### Hopper field report (2x H100 NVL, TP=2, `DeepSeek-V4-Flash-DSpark`, block=5)

- Overall acceptance: 57.2%
- Mean acceptance length: 3.86 tokens/step
- Per-position: p0 82%, p1 67%, p2 55%, p3 45%, p4 37%
- vs serial MTP baseline: ~1.75x per-step yield; pos-1 67% vs 43%
- Default (non-probabilistic) drafting; no looping/degradation

### Acceptance interpretation

llama.cpp ~2.5/step on Gemma4 coding is in the right ballpark but below vLLM's
DSV4 numbers (~3.4-3.9). Contributing factors:

- Different model/checkpoint (Gemma4 vs DSV4)
- `n_max=4` vs full `block_size=7`
- Q4 target hidden-state drift vs bf16 training

---

## Bugs and pitfalls from the PR

Subtle correctness traps surfaced in fix commits and review threads. Audit
llama.cpp if adding CUDA graphs or probabilistic drafting.

| Issue | Fix / status | Relevance to llama.cpp |
|-------|--------------|------------------------|
| Stale `idx_mapping` from CUDA graph padding | `54b454f` | If adding CUDA graph capture for propose |
| Non-contiguous gumbel sample inputs | `d0d2c85` | `temp > 0` rejection sampling |
| `sample_pos` double-increment in gumbel path | Open review thread | Audit RNG/position contract for probabilistic accept |
| Dummy Q buffer must be zeros, not `torch.empty` | `af8bf12` | Any scratch alloc before graph capture |
| Probabilistic drafting junk/loops on DSV4 | Fixed (two bugs above) | Also affected MTP baseline on DSV4 |
| Qwen3 DSpark IMA with CUDA graphs | Fixed by rebase + FLASH_ATTN | Dense-model CG buffer hygiene |
| Confidence head code unused at inference | Reviewer: remove until used | Aligns with our benchmark finding (threshold > 0 hurts on Q4) |

---

## Shared propose loop (mental model)

```
Target model
  prefill/decode -> extract layer h at target_layers
       |
       v
Draft model (parallel backbone)
  encode fused h -> inject context K/V
  non-causal noise block forward (N positions)
       |
       v
Sequential Markov
  for i in 0..N-1: logits_i = lm_head(h_i) + markov(prev_i)
       |
       v
Target verify
  batched forward (anchor + drafts) -> rejection sampling -> accept prefix
       |
       v
  inject accepted+1 hidden back into target/draft state
```

Both codebases implement this loop. vLLM optimizes the draft box (CUDA graphs,
sparse MLA) at datacenter scale. llama.cpp targets GGUF/Q4/consumer GPUs with
deeper Gemma4 correctness validation; CUDA batched verify is the critical missing
piece for NVIDIA speedup.

---

## What llama.cpp got right

- DFlash-style KV injection + non-causal noise block (not custom cross-attention)
- Anchor-first query layout (N queries, not 1+N masks like DFlash)
- Markov ordering (softcap before bias, per DeepSpec)
- DeepSpec Phase 0/3a reference gate (tighter per-step invariants than vLLM's GSM8k smoke)
- Fair benchmark methodology (vanilla-first + cooldown; see benchmark docs)
- Shorter draft forward when confidence off (`n_draft+1` not full `block_size`)
- Deferred layer D2H on verify path (Vulkan win)

---

## Actionable gaps (ordered by impact)

### 1. CUDA batched multi-token verify correctness

#1 blocker on NVIDIA. vLLM verify works at BS32; llama.cpp diverges at gen 54+
on CUDA with batched verify. KV tail trim and `-fa 0` were insufficient.
Sequential fallback is correct but ~0.90x vs ~2.0x theoretical with batched.

See [dspark-benchmark-nvidia.md](dspark-benchmark-nvidia.md).

### 2. CUDA graph for full propose loop

vLLM's 0.6 + 0.6 ms draft breakdown suggests launch overhead matters at scale.
Graph draft backbone + sequential Markov even if forward is already ~4 ms on
RTX 3090.

### 3. `n_max` vs `block_size`

Default `n_max=4` for speed/match; vLLM runs full 7 and gets higher cumulative
acceptance. Revisit `n_max=7` once CUDA verify is fixed.

### 4. Confidence scheduling

Both projects defer it for good reason. vLLM reviewer wants confidence code
removed until used. llama.cpp sweeps confirm full `block_size=7` draft when
enabled kills throughput on Q4. Revisit on bf16 where draft cost is lower
relative to verify.

### 5. Qwen3 path

vLLM `qwen3_dspark.py` is a thin delta over `qwen3_dflash.py` (Markov +
confidence heads on DFlash Qwen3 backbone). Phase 4 should mirror that.

### 6. Probabilistic drafting

When enabling `temp > 0`, study vLLM gumbel path and the two bug fixes before
trusting acceptance metrics.

### 7. Per-position acceptance breakdown

Reproduce vLLM-style p0, p1, ... reporting on the Gemma4 harness to separate
`n_max` tuning, Q4 numerics, and draft quality effects.

---

## What not to blindly copy

- **Mean-pooling hyper-connection states** for DSV4 - still under PR review
- **SparseMLA top-k trick** - only relevant for DSV4 sparse attention, not Gemma4
- **Skipping Gemma4** - vLLM defers it; llama.cpp is the Gemma4 story
- **Absolute TPS numbers** - dominated by 8xB300 + fp8 KV + prefix cache; compare
  accept/step and per-step ms instead

---

## vLLM PR file index (for cross-reference)

Fetched from `benchislett:dspark` (@ `275d462`):

| Path | Role |
|------|------|
| `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` | `DSparkSpeculator` (subclasses DFlash) |
| `vllm/v1/worker/gpu/spec_decode/dspark/utils.py` | Model load, embed sharing, `use_non_causal` |
| `vllm/models/deepseek_v4/nvidia/dspark.py` | DSV4 draft model |
| `vllm/model_executor/models/qwen3_dspark.py` | Qwen3 draft model |
| `vllm/v1/attention/backends/mla/sparse_swa.py` | Non-causal SWA index expansion |
| `vllm/config/speculative.py` | `method: dspark` config |
| `tests/v1/attention/test_dspark_noncausal_sparse_mla.py` | Non-causal attn correctness tests |

Example launch (from PR):

```bash
vllm serve deepseek-ai/DeepSeek-V4-Pro-DSpark \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --speculative_config '{"method": "dspark", "num_speculative_tokens": 7, "draft_sample_method": "probabilistic"}'

vllm serve Qwen/Qwen3-8B \
    --speculative_config '{"method":"dspark","model":"deepseek-ai/dspark_qwen3_8b_block7","num_speculative_tokens":7, "attention_backend":"FLASH_ATTN", "draft_sample_method": "probabilistic"}'
```

---

## Open PR scope (out of scope for vLLM #46995)

- Dynamic drafting and confidence-based scheduling (track
  [PR #45953](https://github.com/vllm-project/vllm/pull/45953) for DSD + CUDA graphs)
- Gemma4-DSpark (separate arch; author prefers Qwen3 DSpark reuse)

---

## Revision log

| Date | Notes |
|------|-------|
| 2026-06-30 | Initial doc from comparison of PR #46995 vs `ft-dspark` branch, benchmark docs, and DeepSpec reference |
