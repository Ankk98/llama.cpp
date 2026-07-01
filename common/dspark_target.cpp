#include "dspark_target.h"

#include "speculative.h"

#include "../src/llama-ext.h"

#include <cstdlib>
#include <cstring>
#include <unordered_set>
#include <vector>

static bool dspark_verify_canary_enabled() {
    return getenv("DSPARK_VERIFY_CANARY") != nullptr;
}

static bool dspark_verify_layer_taps_enabled() {
    return getenv("DSPARK_VERIFY_LAYER_TAPS") != nullptr;
}

static llama_token dspark_greedy_argmax_row(llama_context * ctx, int row) {
    const llama_model * model = llama_get_model(ctx);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float * logits = llama_get_logits_ith_no_sync(ctx, row);
    if (!logits) {
        return LLAMA_TOKEN_NULL;
    }
    return common_sampler_greedy_argmax(logits, n_vocab);
}

static void dspark_verify_prepare_scratch(
        llama_context * ctx_tgt,
        llama_memory_t mem_tgt,
        llama_seq_id seq_main,
        llama_seq_id seq_scratch) {
    llama_synchronize(ctx_tgt);
    llama_memory_seq_rm(mem_tgt, seq_scratch, -1, -1);
    llama_memory_seq_cp(mem_tgt, seq_main, seq_scratch, -1, -1);
}

static bool dspark_state_save(llama_context * ctx, std::vector<uint8_t> & buf) {
    const size_t sz = llama_state_get_size(ctx);
    if (sz == 0) {
        return false;
    }
    buf.resize(sz);
    return llama_state_get_data(ctx, buf.data(), sz) == sz;
}

static bool dspark_state_restore(llama_context * ctx, const std::vector<uint8_t> & buf) {
    if (buf.empty()) {
        return false;
    }
    return llama_state_set_data(ctx, buf.data(), buf.size()) == buf.size();
}

// Debug-only: compare batched row logits vs one-token chain on scratch seq.
static int dspark_find_first_logit_mismatch(
        llama_context * ctx_tgt,
        llama_memory_t mem_tgt,
        llama_seq_id seq_main,
        llama_seq_id seq_scratch,
        llama_pos pos_verify,
        llama_token anchor,
        const llama_tokens & draft,
        llama_batch & batch,
        const std::vector<llama_token> & batched_toks) {
    const int n_rows = (int) draft.size() + 1;

    for (int row = 0; row < n_rows; ++row) {
        dspark_verify_prepare_scratch(ctx_tgt, mem_tgt, seq_main, seq_scratch);

        for (int j = 0; j <= row; ++j) {
            const llama_token tok_in = (j == 0) ? anchor : draft[(size_t) j - 1];
            common_batch_clear(batch);
            common_batch_add(batch, tok_in, pos_verify + (llama_pos) j, { seq_scratch }, true);
            if (llama_decode(ctx_tgt, batch) != 0) {
                return row;
            }
            llama_synchronize(ctx_tgt);
        }

        const llama_token seq_tok = dspark_greedy_argmax_row(ctx_tgt, 0);
        if (seq_tok != batched_toks[(size_t) row]) {
            return row;
        }
    }

    return -1;
}

llama_seq_id dspark_target_scratch_seq_id(const llama_context * ctx, llama_seq_id seq_main) {
    const llama_seq_id scratch = seq_main + 1;
    if ((uint32_t) scratch < llama_n_seq_max(ctx)) {
        return scratch;
    }
    return seq_main;
}

void dspark_target_assert_canonical_kv(
        llama_context * ctx_tgt,
        llama_seq_id seq_main,
        llama_pos pos_verify,
        const char * tag) {
    if (!getenv("DSPARK_KV_ASSERT")) {
        return;
    }

    llama_memory_t mem = llama_get_memory(ctx_tgt);
    const llama_pos kv_max = llama_memory_seq_pos_max(mem, seq_main);
    const llama_pos expect = pos_verify - 1;
    if (kv_max != expect) {
        fprintf(stderr,
                "DSPARK_KV_ASSERT: %s seq=%d pos_verify=%d kv_max=%d expect=%d\n",
                tag, (int) seq_main, (int) pos_verify, (int) kv_max, (int) expect);
        GGML_ABORT("DSpark canonical KV invariant violated");
    }
}

void dspark_target_context_reset(llama_context * ctx) {
    if (ctx == nullptr) {
        return;
    }

    llama_memory_t mem = llama_get_memory(ctx);
    if (mem) {
        llama_memory_clear(mem, true);
    }

    llama_set_skip_host_logits(ctx, false);
    llama_set_defer_layer_inp_extract(ctx, false);
    llama_synchronize(ctx);
    llama_perf_context_reset(ctx);
}

static void dspark_ensure_fused_verify_graph(llama_context * ctx, int32_t n_verify, llama_seq_id seq_id) {
    if (ctx == nullptr || n_verify <= 1) {
        return;
    }

    static std::unordered_set<llama_context *> enabled;
    if (enabled.count(ctx)) {
        return;
    }

    llama_set_skip_host_logits(ctx, true);
    llama_graph_reserve(ctx, (uint32_t) n_verify, 1, (uint32_t) n_verify);

    llama_batch warm = llama_batch_init(llama_n_batch(ctx), 0, 1);
    for (int32_t i = 0; i < n_verify; ++i) {
        common_batch_add(warm, 0, (llama_pos) i, { seq_id }, true);
    }

    llama_decode(ctx, warm);
    llama_synchronize(ctx);
    llama_batch_free(warm);
    dspark_target_context_reset(ctx);

    enabled.insert(ctx);
}

bool dspark_target_verify_logits(
        dspark_memory_bundle * mem,
        common_speculative * spec,
        llama_pos pos_verify,
        llama_token anchor,
        const llama_tokens & draft,
        llama_batch & batch,
        dspark_verify_timing * timing) {
    if (mem == nullptr || mem->ctx_tgt == nullptr || spec == nullptr) {
        return false;
    }

    llama_context * ctx_tgt = mem->ctx_tgt;
    const llama_seq_id seq_main    = mem->seq_main;
    const llama_seq_id seq_scratch = mem->seq_scratch;

    if (seq_scratch == seq_main) {
        fprintf(stderr, "DSpark: scratch seq unavailable (n_seq_max too low for parallel verify)\n");
        return false;
    }

    const bool layer_taps = dspark_verify_layer_taps_enabled();
    const bool defer_layers = layer_taps
            && getenv("DSPARK_VERIFY_DEFER_SCRATCH") != nullptr
            && getenv("DSPARK_NO_DEFER_LAYER_INP") == nullptr;

    llama_memory_t mem_tgt = llama_get_memory(ctx_tgt);

    dspark_target_assert_canonical_kv(ctx_tgt, seq_main, pos_verify, "pre-verify");

    dspark_verify_prepare_scratch(ctx_tgt, mem_tgt, seq_main, seq_scratch);

    common_batch_clear(batch);
    common_batch_add(batch, anchor, pos_verify, { seq_scratch }, true);
    for (size_t i = 0; i < draft.size(); ++i) {
        const llama_pos pos = pos_verify + 1 + (llama_pos) i;
        common_batch_add(batch, draft[i], pos, { seq_scratch }, true);
        if (getenv("DSPARK_TRACE_VERIFY")) {
            fprintf(stderr,
                    "DSPARK_TRACE_VERIFY: batch[%zu] tok=%d pos=%d (expect pos_verify+1+%zu=%d)\n",
                    i + 1, (int) draft[i], (int) pos, i,
                    (int) (pos_verify + 1 + (llama_pos) i));
        }
    }
    if (getenv("DSPARK_TRACE_VERIFY")) {
        fprintf(stderr,
                "DSPARK_TRACE_VERIFY: batch[0] anchor=%d pos=%d pos_verify=%d draft_n=%zu\n",
                (int) anchor, (int) pos_verify, (int) pos_verify, draft.size());
    }

    const int64_t t0 = timing ? ggml_time_us() : 0;

    if (getenv("DSPARK_FUSED_ARGMAX") != nullptr && getenv("DSPARK_GPU_GREEDY") == nullptr) {
        dspark_ensure_fused_verify_graph(ctx_tgt, (int32_t) draft.size() + 1, seq_scratch);
    }

    // Logits-only verify; layer taps and process() run during commit only.
    common_speculative_dspark_target_features_enable(spec, layer_taps);

    if (defer_layers) {
        llama_set_defer_layer_inp_extract(ctx_tgt, true);
    }

    if (llama_decode(ctx_tgt, batch) != 0) {
        if (defer_layers) {
            llama_set_defer_layer_inp_extract(ctx_tgt, false);
        }
        common_speculative_dspark_target_features_enable(spec, false);
        return false;
    }

    if (timing) {
        const int64_t t_decode = ggml_time_us();
        timing->decode_submit_ms = 1e-3 * (t_decode - t0);
        timing->logits_decode_ms = timing->decode_submit_ms;
    }

    if (defer_layers) {
        llama_set_defer_layer_inp_extract(ctx_tgt, false);
    }

    if (dspark_verify_canary_enabled()) {
        llama_synchronize(ctx_tgt);
        const int n_rows = (int) draft.size() + 1;
        std::vector<llama_token> batched_toks((size_t) n_rows, LLAMA_TOKEN_NULL);
        for (int row = 0; row < n_rows; ++row) {
            batched_toks[(size_t) row] = dspark_greedy_argmax_row(ctx_tgt, row);
        }
        const int mismatch_row = dspark_find_first_logit_mismatch(
                ctx_tgt, mem_tgt, seq_main, seq_scratch, pos_verify, anchor, draft, batch, batched_toks);
        if (mismatch_row >= 0) {
            fprintf(stderr,
                    "DSPARK_VERIFY_CANARY: row%d mismatch batched=%d pos_verify=%d anchor=%d draft_n=%zu\n",
                    mismatch_row, (int) batched_toks[(size_t) mismatch_row],
                    (int) pos_verify, (int) anchor, draft.size());
            GGML_ABORT("DSpark parallel verify logits mismatch");
        }

        dspark_verify_prepare_scratch(ctx_tgt, mem_tgt, seq_main, seq_scratch);
        common_batch_clear(batch);
        common_batch_add(batch, anchor, pos_verify, { seq_scratch }, true);
        for (size_t i = 0; i < draft.size(); ++i) {
            common_batch_add(batch, draft[i], pos_verify + 1 + (llama_pos) i, { seq_scratch }, true);
        }
        if (layer_taps) {
            common_speculative_dspark_target_features_enable(spec, true);
        }
        if (defer_layers) {
            llama_set_defer_layer_inp_extract(ctx_tgt, true);
        }
        if (llama_decode(ctx_tgt, batch) != 0) {
            if (defer_layers) {
                llama_set_defer_layer_inp_extract(ctx_tgt, false);
            }
            common_speculative_dspark_target_features_enable(spec, false);
            return false;
        }
        if (defer_layers) {
            llama_set_defer_layer_inp_extract(ctx_tgt, false);
        }
    }

    if (!layer_taps) {
        common_speculative_dspark_target_features_enable(spec, false);
    }

    return true;
}

llama_tokens dspark_target_accept_chain(
        common_sampler * smpl,
        llama_context * ctx_tgt,
        const llama_tokens & draft,
        const std::vector<std::vector<float>> * draft_probs,
        float temp) {
    llama_tokens out_ids;

    if (smpl == nullptr || ctx_tgt == nullptr) {
        return out_ids;
    }

    const bool pure_greedy = draft_probs == nullptr
            && (temp == 0.0f || temp == -1.0f)
            && common_sampler_is_pure_greedy(smpl);

    const bool fused_argmax = pure_greedy && getenv("DSPARK_FUSED_ARGMAX") != nullptr;

    const bool gpu_greedy = pure_greedy
            && !fused_argmax
            && getenv("DSPARK_GPU_GREEDY") != nullptr
            && getenv("DSPARK_NO_GPU_GREEDY") == nullptr;

    if (draft_probs != nullptr && !draft_probs->empty()) {
        std::vector<int> idxs(draft.size() + 1);
        for (size_t i = 0; i < idxs.size(); ++i) {
            idxs[i] = (int) i;
        }
        out_ids = common_sampler_sample_and_accept_n_dspark(
                smpl, ctx_tgt, idxs, draft, *draft_probs, temp);
    } else if (fused_argmax) {
        std::vector<int> idxs(draft.size() + 1);
        for (size_t i = 0; i < idxs.size(); ++i) {
            idxs[i] = (int) i;
        }
        out_ids = common_sampler_greedy_accept_n_fused(smpl, ctx_tgt, idxs, draft);
    } else if (gpu_greedy) {
        std::vector<int> idxs(draft.size() + 1);
        for (size_t i = 0; i < idxs.size(); ++i) {
            idxs[i] = (int) i;
        }
        out_ids = common_sampler_greedy_accept_n_gpu(smpl, ctx_tgt, idxs, draft);
    } else {
        out_ids = common_sampler_sample_and_accept_n(smpl, ctx_tgt, draft);
    }

    llama_set_defer_layer_inp_extract(ctx_tgt, false);

    return out_ids;
}

void dspark_target_verify_scratch_cleanup(dspark_memory_bundle * mem, llama_pos pos_verify) {
    if (mem == nullptr || mem->ctx_tgt == nullptr) {
        return;
    }

    llama_memory_t mem_tgt = llama_get_memory(mem->ctx_tgt);
    llama_memory_seq_rm(mem_tgt, mem->seq_scratch, pos_verify, -1);
}

bool dspark_target_commit_tokens(
        common_speculative * spec,
        dspark_memory_bundle * mem,
        llama_pos pos_verify,
        llama_token anchor,
        const llama_tokens & committed_ids,
        llama_batch & batch,
        dspark_verify_timing * timing) {
    if (mem == nullptr || mem->ctx_tgt == nullptr) {
        return false;
    }

    if (committed_ids.empty()) {
        return false;
    }

    dspark_target_assert_canonical_kv(mem->ctx_tgt, mem->seq_main, pos_verify, "post-verify-pre-commit");

    const int64_t t_rebuild = timing ? ggml_time_us() : 0;

    common_speculative_dspark_target_features_enable(spec, true);

    for (size_t i = 0; i < committed_ids.size(); ++i) {
        const llama_token tok_in = (i == 0) ? anchor : committed_ids[i - 1];

        common_batch_clear(batch);
        common_batch_add(batch, tok_in, pos_verify + (llama_pos) i, { mem->seq_main }, true);

        if (llama_decode(mem->ctx_tgt, batch) != 0) {
            return false;
        }

        if (!common_speculative_process(spec, batch)) {
            return false;
        }
    }

    if (timing) {
        timing->layer_commit_ms    = 0;
        timing->features_decode_ms = 0;
        timing->process_ms         = 1e-3 * (ggml_time_us() - t_rebuild);
    }

    dspark_target_assert_canonical_kv(
            mem->ctx_tgt, mem->seq_main, pos_verify + (llama_pos) committed_ids.size(), "post-commit");

    return true;
}

bool dspark_target_commit_one_greedy(
        common_speculative * spec,
        dspark_memory_bundle * mem,
        common_sampler * smpl,
        llama_pos pos_verify,
        llama_token anchor,
        llama_batch & batch,
        llama_tokens & out_one) {
    out_one.clear();

    if (mem == nullptr || smpl == nullptr) {
        return false;
    }

    common_batch_clear(batch);
    common_batch_add(batch, anchor, pos_verify, { mem->seq_main }, true);

    common_speculative_dspark_target_features_enable(spec, true);

    if (llama_decode(mem->ctx_tgt, batch) != 0) {
        return false;
    }

    const llama_token id = common_sampler_sample(smpl, mem->ctx_tgt, -1, false);
    common_sampler_accept(smpl, id, true);
    out_one.push_back(id);

    if (!common_speculative_process(spec, batch)) {
        return false;
    }

    return true;
}

bool dspark_target_verify_sequential(
        common_speculative * spec,
        common_sampler * smpl,
        dspark_memory_bundle * mem,
        llama_pos pos_verify,
        llama_token anchor,
        const llama_tokens & draft,
        llama_tokens & out_ids,
        llama_batch & batch) {
    out_ids.clear();

    if (spec == nullptr || smpl == nullptr || mem == nullptr || mem->ctx_tgt == nullptr) {
        return false;
    }

    llama_context * ctx_tgt = mem->ctx_tgt;
    const llama_seq_id seq_main = mem->seq_main;

    const bool pure_greedy = common_sampler_is_pure_greedy(smpl);

    auto sample_one = [&](int idx) -> llama_token {
        if (pure_greedy) {
            const llama_model * model = llama_get_model(ctx_tgt);
            const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
            const float * logits = llama_get_logits_ith_no_sync(ctx_tgt, idx);
            GGML_ASSERT(logits);
            return common_sampler_greedy_argmax(logits, n_vocab);
        }
        return common_sampler_sample_after_sync(smpl, ctx_tgt, idx, false);
    };

    llama_synchronize(ctx_tgt);
    llama_set_defer_layer_inp_extract(ctx_tgt, false);
    common_speculative_dspark_target_features_enable(spec, true);

    for (size_t i = 0; i <= draft.size(); ++i) {
        const llama_token tok_in = (i == 0) ? anchor : draft[i - 1];

        common_batch_clear(batch);
        common_batch_add(batch, tok_in, pos_verify + (llama_pos) i, { seq_main }, true);

        if (llama_decode(ctx_tgt, batch) != 0) {
            return false;
        }
        llama_synchronize(ctx_tgt);

        const llama_token id = sample_one(0);
        common_sampler_accept(smpl, id, true);
        out_ids.push_back(id);

        if (!common_speculative_process(spec, batch)) {
            return false;
        }

        if (i < draft.size()) {
            if (id != draft[i]) {
                break;
            }
        }
    }

    return !out_ids.empty();
}

bool dspark_target_verify_step(
        common_speculative * spec,
        common_sampler * smpl,
        dspark_memory_bundle * mem,
        llama_pos pos_verify,
        llama_token anchor,
        const llama_tokens & draft,
        llama_tokens & out_ids,
        llama_batch & batch,
        dspark_verify_timing * timing,
        const std::vector<std::vector<float>> * draft_probs,
        float temp) {
    out_ids.clear();

    if (spec == nullptr || smpl == nullptr || mem == nullptr) {
        return false;
    }

    if (getenv("DSPARK_VERIFY_SEQ") != nullptr) {
        return dspark_target_verify_sequential(
                spec, smpl, mem, pos_verify, anchor, draft, out_ids, batch);
    }

    if (draft.empty()) {
        return dspark_target_commit_one_greedy(
                spec, mem, smpl, pos_verify, anchor, batch, out_ids);
    }

    const int64_t t_accept0 = timing ? ggml_time_us() : 0;

    std::vector<uint8_t> pre_snap;
    const bool use_pre_snap = getenv("DSPARK_NO_VERIFY_PRE_SNAP") == nullptr;
    if (use_pre_snap) {
        dspark_state_save(mem->ctx_tgt, pre_snap);
    }

    if (!dspark_target_verify_logits(mem, spec, pos_verify, anchor, draft, batch, timing)) {
        return false;
    }

    out_ids = dspark_target_accept_chain(smpl, mem->ctx_tgt, draft, draft_probs, temp);

    if (timing) {
        timing->accept_ms = 1e-3 * (ggml_time_us() - t_accept0);
    }

    dspark_target_verify_scratch_cleanup(mem, pos_verify);

    if (out_ids.empty()) {
        if (use_pre_snap && !pre_snap.empty()) {
            dspark_state_restore(mem->ctx_tgt, pre_snap);
        }
        return false;
    }

    if (use_pre_snap && !pre_snap.empty()) {
        dspark_state_restore(mem->ctx_tgt, pre_snap);
    }

    return dspark_target_commit_tokens(
            spec, mem, pos_verify, anchor, out_ids, batch, timing);
}
