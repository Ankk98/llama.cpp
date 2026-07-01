#pragma once

#include "common.h"
#include "llama.h"
#include "sampling.h"

#include <cstdint>
#include <vector>

struct common_speculative;

// Target + scratch memory domains (see dspark-refactor-kv-safety.md).
struct dspark_memory_bundle {
    llama_context * ctx_tgt     = nullptr;
    llama_context * ctx_dft     = nullptr;
    llama_context * ctx_tgt_feat = nullptr;
    llama_seq_id    seq_main    = 0;
    llama_seq_id    seq_scratch = 1;
};

struct dspark_verify_timing {
    double decode_submit_ms   = 0;
    double logits_decode_ms   = 0;
    double accept_ms          = 0;
    double layer_commit_ms    = 0;
    double features_decode_ms = 0;
    double process_ms         = 0;
};

// Scratch seq for batched verify logits (seq_main + 1 when n_seq_max allows).
llama_seq_id dspark_target_scratch_seq_id(const llama_context * ctx, llama_seq_id seq_main);

void dspark_target_assert_canonical_kv(
        llama_context * ctx_tgt,
        llama_seq_id seq_main,
        llama_pos pos_verify,
        const char * tag);

void dspark_target_context_reset(llama_context * ctx);

// Phase A: batched forward on scratch seq; canonical KV unchanged.
bool dspark_target_verify_logits(
        dspark_memory_bundle * mem,
        common_speculative * spec,
        llama_pos pos_verify,
        llama_token anchor,
        const llama_tokens & draft,
        llama_batch & batch,
        dspark_verify_timing * timing = nullptr);

// Phase B: greedy/stochastic accept from scratch row logits (no llama_decode).
llama_tokens dspark_target_accept_chain(
        common_sampler * smpl,
        llama_context * ctx_tgt,
        const llama_tokens & draft,
        const std::vector<std::vector<float>> * draft_probs = nullptr,
        float temp = 0.0f);

// Discard hypothetical verify tail on scratch seq after accept.
void dspark_target_verify_scratch_cleanup(
        dspark_memory_bundle * mem,
        llama_pos pos_verify);

// Phase C: append accepted tokens on canonical seq + draft feature inject.
bool dspark_target_commit_tokens(
        common_speculative * spec,
        dspark_memory_bundle * mem,
        llama_pos pos_verify,
        llama_token anchor,
        const llama_tokens & committed,
        llama_batch & batch,
        dspark_verify_timing * timing = nullptr);

// Single-token fallback when draft block is empty (canonical decode + process).
bool dspark_target_commit_one_greedy(
        common_speculative * spec,
        dspark_memory_bundle * mem,
        common_sampler * smpl,
        llama_pos pos_verify,
        llama_token anchor,
        llama_batch & batch,
        llama_tokens & out_one);

// Sequential verify on canonical (DSPARK_VERIFY_SEQ). Verify + commit fused.
bool dspark_target_verify_sequential(
        common_speculative * spec,
        common_sampler * smpl,
        dspark_memory_bundle * mem,
        llama_pos pos_verify,
        llama_token anchor,
        const llama_tokens & draft,
        llama_tokens & out_ids,
        llama_batch & batch);

// Default path: scratch verify + accept + canonical commit.
bool dspark_target_verify_step(
        common_speculative * spec,
        common_sampler * smpl,
        dspark_memory_bundle * mem,
        llama_pos pos_verify,
        llama_token anchor,
        const llama_tokens & draft,
        llama_tokens & out_ids,
        llama_batch & batch,
        dspark_verify_timing * timing = nullptr,
        const std::vector<std::vector<float>> * draft_probs = nullptr,
        float temp = 0.0f);
