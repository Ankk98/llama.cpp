#pragma once

#include "llama.h"
#include "common.h"

struct common_speculative;

// comma separated list the provided types
std::string common_speculative_type_name_str(const std::vector<enum common_speculative_type> & types);

// comma separated list of all types
const char * common_speculative_all_types_str();

// parse user provided types
std::vector<enum common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names);

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// return the max number of draft tokens based on the speculative parameters
int32_t common_speculative_n_max(const common_params_speculative * spec);

common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq);

void common_speculative_free(common_speculative * spec);

struct common_speculative_draft_params {
    // this flag is used to chain the drafts through all the available implementations
    // after the first successful draft from an implementation, we set it
    //   to false to prevent further drafts for that sequence
    // at the end of the draft() call, all drafting flags will be reset to false
    bool drafting = false;

    // overrides individual configurations (-1 disabled)
    // can be used to constraint the max draft based on the remaining context size
    int32_t n_max = -1;

    llama_pos   n_past;
    llama_token id_last;

    // TODO: remove in the future by keeping track of the prompt from the _begin() call and the consecutive accept calls
    const llama_tokens * prompt;

    // the generated draft from the last _draft() call
    llama_tokens * result;

    // optional output: per-position draft probability vectors (DSpark, temp > 0)
    std::vector<std::vector<float>> * draft_probs = nullptr;
};

common_speculative_draft_params & common_speculative_get_draft_params(common_speculative * spec, llama_seq_id seq_id);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt);

// process the batch and update the internal state of the speculative context
bool common_speculative_process(common_speculative * spec, const llama_batch & batch);

// true if any implementation requires target post-norm embeddings to be extracted
bool common_speculative_need_embd(common_speculative * spec);

// true if any implementation requires target nextn embeddings to be extracted
bool common_speculative_need_embd_nextn(common_speculative * spec);

// generate drafts for the sequences specified with `common_speculative_get_draft_params`
void common_speculative_draft(common_speculative * spec);

// update runtime DSpark options (temp/seed/confidence) before drafting
void common_speculative_sync_params(common_speculative * spec, const common_params_speculative & params);

// DSpark: toggle expensive target layer-input extraction (5 hidden tensors per token)
void common_speculative_dspark_target_features_enable(common_speculative * spec, bool enable);

// No-op (legacy). Batched verify uses a scratch sequence; call is harmless at loop start.
void common_speculative_dspark_verify_kv_canon_reset();

struct common_speculative_dspark_prefill_timing {
    double fast_ms  = 0; // target decode with layer taps off (optional first pass)
    double setup_ms = 0; // target decode with layer taps on + process()
};

// DSpark/DeepSpec prefill: target decode inp[0..n-2], then process() into draft KV.
// fast_ttft: layer-tap-free decode first (fast_ms), then feature decode + process (setup_ms).
bool common_speculative_dspark_prefill(
        common_speculative * spec,
        struct llama_context * ctx_tgt,
        const llama_tokens & inp,
        struct llama_batch & batch,
        bool fast_ttft,
        common_speculative_dspark_prefill_timing * timing = nullptr);

struct common_speculative_dspark_verify_timing {
    double decode_submit_ms   = 0; // llama_decode return (async, no GPU sync)
    double logits_decode_ms   = 0; // same as decode_submit_ms (legacy alias)
    double accept_ms          = 0; // GPU fence + greedy argmax on host logits
    double layer_commit_ms    = 0; // partial layer D2H after accept (defer path)
    double features_decode_ms = 0;
    double process_ms         = 0;
};

// DSpark: after verify accept, decode committed tokens with layer taps on and run process().
// When kv_append_only is true, canonical KV ends at pos_verify - 1 (scratch verify path).
bool common_speculative_dspark_process_committed(
        common_speculative * spec,
        struct llama_context * ctx_tgt,
        llama_seq_id seq_id,
        llama_pos pos_verify,
        llama_token anchor,
        const llama_tokens & committed_ids,
        struct llama_batch & batch,
        bool kv_append_only = false);

// DSpark: target verify entry point. Default sequential (correct at temp=0); parallel opt-in via DSPARK_VERIFY_PARALLEL=1
bool common_speculative_dspark_verify_batched(
        common_speculative * spec,
        struct common_sampler * smpl,
        struct llama_context * ctx_tgt,
        llama_seq_id seq_id,
        llama_pos pos_verify,
        llama_token anchor,
        const llama_tokens & draft,
        llama_tokens & out_ids,
        struct llama_batch & batch,
        common_speculative_dspark_verify_timing * timing = nullptr,
        const std::vector<std::vector<float>> * draft_probs = nullptr,
        float temp = 0.0f);

// DSpark: sequential early-exit target verify (one token/decode, stop at first mismatch)
bool common_speculative_dspark_verify_sequential(
        common_speculative * spec,
        struct common_sampler * smpl,
        struct llama_context * ctx_tgt,
        llama_seq_id seq_id,
        llama_pos pos_verify,
        llama_token anchor,
        const llama_tokens & draft,
        llama_tokens & out_ids,
        struct llama_batch & batch);

// informs the speculative context that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, llama_seq_id, uint16_t n_accepted);

// (optional) get/set internal state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data);
void common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;
