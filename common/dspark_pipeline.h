#pragma once

#include "common.h"
#include "dspark_draft.h"
#include "dspark_target.h"
#include "llama.h"
#include "sampling.h"

struct common_speculative;
struct common_params;
struct common_params_speculative;

struct dspark_pipeline_config {
    int32_t block_size        = 7;
    int32_t n_max             = 4;
    int32_t min_verify_tokens = 1;
    float   confidence_threshold = 0.0f;
    float   temp              = 0.0f;
};

struct dspark_step_timing {
    double draft_ms  = 0;
    double verify_ms = 0;
    dspark_verify_timing verify_detail;
};

struct dspark_step_result {
    llama_tokens committed;
    int n_accepted_draft = 0;
    int n_drafted        = 0;
    dspark_step_timing timing;
};

struct dspark_pipeline_state {
    dspark_memory_bundle   mem;
    common_speculative *   spec       = nullptr;
    common_sampler *       smpl       = nullptr;
    common_params_speculative spec_params;
    llama_tokens           prompt;
    llama_pos              n_past     = 0;
    llama_token            anchor     = LLAMA_TOKEN_NULL;
    llama_batch            batch_tgt;
    dspark_pipeline_config cfg;
    bool                   batch_owned = false;
};

struct dspark_run_stats {
    std::vector<llama_token> output;
    double pp_ms  = 0;
    double gen_ms = 0;
    double draft_ms = 0;
    double verify_ms = 0;
    double tgt_decode_ms = 0;
    double verify_accept_ms = 0;
    double verify_layer_commit_ms = 0;
    double verify_features_ms = 0;
    double verify_decode_submit_ms = 0;
    double verify_process_ms = 0;
    int n_prompt = 0;
    int n_generated = 0;
    int n_drafted = 0;
    int n_accepted = 0;
    int n_propose_steps = 0;
};

bool dspark_pipeline_init(
        dspark_pipeline_state * st,
        common_speculative * spec,
        common_sampler * smpl,
        llama_context * ctx_tgt,
        llama_context * ctx_dft,
        llama_context * ctx_tgt_feat,
        const common_params_speculative & spec_params,
        const dspark_pipeline_config & cfg);

void dspark_pipeline_free(dspark_pipeline_state * st);

bool dspark_pipeline_prefill(
        dspark_pipeline_state * st,
        const llama_tokens & prompt,
        llama_token * out_first_gen,
        double * out_pp_ms);

bool dspark_pipeline_step(dspark_pipeline_state * st, dspark_step_result * out);

bool dspark_pipeline_run(
        common_params & params,
        common_speculative * spec,
        common_sampler * smpl,
        llama_context * ctx_tgt,
        llama_context * ctx_dft,
        const llama_tokens & inp,
        int n_predict,
        dspark_run_stats * out);
