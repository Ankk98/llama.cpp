#include "dspark_draft.h"

#include "speculative.h"

#include <cstdlib>

bool dspark_draft_propose(
        common_speculative * spec,
        const common_params_speculative & params,
        llama_seq_id seq_id,
        llama_pos n_past,
        llama_token anchor,
        const llama_tokens * prompt,
        dspark_draft_propose_result * out) {
    if (out == nullptr) {
        return false;
    }

    out->draft.clear();
    out->n_proposed = 0;

    if (spec == nullptr || getenv("DSPARK_NO_DRAFT")) {
        return true;
    }

    common_speculative_sync_params(spec, params);

    common_speculative_get_draft_params(spec, seq_id) = {
        true, -1, n_past, anchor, prompt, &out->draft, nullptr,
    };
    common_speculative_draft(spec);

    out->n_proposed = (int) out->draft.size();
    return true;
}

void dspark_memory_trim_beyond(dspark_memory_bundle * mem, llama_pos n_past) {
    if (mem == nullptr || getenv("DSPARK_NO_KV_TRIM")) {
        return;
    }

    if (mem->ctx_tgt) {
        llama_memory_seq_rm(llama_get_memory(mem->ctx_tgt), mem->seq_main, n_past, -1);
    }
    if (mem->ctx_dft) {
        llama_memory_seq_rm(llama_get_memory(mem->ctx_dft), 0, n_past, -1);
    }
}

bool dspark_draft_process_committed(
        common_speculative * spec,
        dspark_memory_bundle * mem,
        llama_pos pos_verify,
        llama_token anchor,
        const llama_tokens & committed,
        llama_batch & batch) {
    if (spec == nullptr || mem == nullptr || committed.empty()) {
        return true;
    }

    for (size_t i = 0; i < committed.size(); ++i) {
        const llama_token tok_in = (i == 0) ? anchor : committed[i - 1];

        common_batch_clear(batch);
        common_batch_add(batch, tok_in, pos_verify + (llama_pos) i, { mem->seq_main }, false);

        if (!common_speculative_process(spec, batch)) {
            return false;
        }
    }

    return true;
}
