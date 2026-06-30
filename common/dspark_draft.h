#pragma once

#include "common.h"
#include "dspark_target.h"
#include "llama.h"

struct common_speculative;
struct common_params_speculative;

struct dspark_draft_propose_result {
    llama_tokens draft;
    int          n_proposed = 0;
};

// Draft propose + confidence filter (domain: ctx_dft only).
bool dspark_draft_propose(
        common_speculative * spec,
        const common_params_speculative & params,
        llama_seq_id seq_id,
        llama_pos n_past,
        llama_token anchor,
        const llama_tokens * prompt,
        dspark_draft_propose_result * out);

// Trim target/draft KV beyond committed n_past.
void dspark_memory_trim_beyond(dspark_memory_bundle * mem, llama_pos n_past);
