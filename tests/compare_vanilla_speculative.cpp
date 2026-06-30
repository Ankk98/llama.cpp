#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include "../src/llama-ext.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

struct bench_flags {
    bool vanilla_only = false;
    bool spec_only    = false;
    std::string reference_path;
    std::string save_output_path;
    std::string json_results_path;
};

struct run_stats {
    std::vector<llama_token> output;
    double pp_ms = 0;
    double gen_ms = 0;
    double draft_ms = 0;  // time inside common_speculative_draft (draft fwd + cpu sampling)
    double verify_ms = 0; // time inside target decode + accept + process
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

static void usage(const char * argv0) {
    fprintf(stderr,
            "Usage: %s -m TARGET.gguf [-md DRAFT.gguf] --input-ids PATH "
            "[--temp 0] [--seed 42] [-n 32] [-c 512] [-ngl N] [-ngld N]\n"
            "       [--vanilla-only] [--spec-only] [--reference PATH.json]\n"
            "       [--save-output PATH.json] [--json-results PATH.json]\n",
            argv0);
}

static bool io_save_tokens(const std::string & path, const std::vector<llama_token> & toks) {
    FILE * f = fopen(path.c_str(), "w");
    if (!f) {
        return false;
    }
    fprintf(f, "[");
    for (size_t i = 0; i < toks.size(); ++i) {
        fprintf(f, "%s%d", i ? "," : "", (int) toks[i]);
    }
    fprintf(f, "]\n");
    fflush(f);
    fclose(f);
    return true;
}

static bool compare_generated_prefix(
        const std::vector<llama_token> & ref,
        const std::vector<llama_token> & out,
        size_t n_inp,
        size_t * first_mismatch_gen) {
    const size_t n_cmp = std::min(ref.size(), out.size());
    size_t first_mismatch = n_cmp;
    for (size_t i = n_inp; i < n_cmp; ++i) {
        if (ref[i] != out[i]) {
            first_mismatch = i;
            break;
        }
    }
    if (first_mismatch_gen) {
        *first_mismatch_gen = first_mismatch == n_cmp ? (size_t) -1 : first_mismatch - n_inp;
    }
    return first_mismatch == n_cmp;
}

static void write_json_results(
        FILE * f,
        const common_params & params,
        size_t n_inp,
        const run_stats * vanilla,
        const run_stats * spec,
        const std::vector<llama_token> * reference,
        bool token_match,
        ssize_t first_mismatch_gen) {
    const auto write_run = [&](const char * key, const run_stats & s, bool is_spec) {
        const double pp  = s.pp_ms  > 0 ? 1000.0 * s.n_prompt    / s.pp_ms  : 0.0;
        const double tgp = s.gen_ms > 0 ? 1000.0 * s.n_generated / s.gen_ms : 0.0;
        fprintf(f, "\"%s\":{", key);
        fprintf(f, "\"pp_ms\":%.4f,\"gen_ms\":%.4f,\"pp_tok_s\":%.4f,\"tgp_tok_s\":%.4f,",
                s.pp_ms, s.gen_ms, pp, tgp);
        fprintf(f, "\"n_prompt\":%d,\"n_generated\":%d", s.n_prompt, s.n_generated);
        if (is_spec) {
            const double accept_rate = s.n_drafted > 0 ? 100.0 * s.n_accepted / s.n_drafted : 0.0;
            const double mean_acc = s.n_propose_steps > 0
                    ? (double) s.n_accepted / (double) s.n_propose_steps : 0.0;
            const double d_step = s.n_propose_steps > 0 ? s.draft_ms / s.n_propose_steps : 0.0;
            const double vf_step = s.n_propose_steps > 0 ? s.verify_ms / s.n_propose_steps : 0.0;
            const double tok_step = s.n_propose_steps > 0
                    ? (double) s.n_generated / s.n_propose_steps : 0.0;
            const double v_fwd = vanilla && vanilla->n_generated > 0
                    ? vanilla->gen_ms / vanilla->n_generated : 0.0;
            fprintf(f, ",\"n_propose_steps\":%d,\"n_drafted\":%d,\"n_accepted\":%d",
                    s.n_propose_steps, s.n_drafted, s.n_accepted);
            fprintf(f, ",\"accept_rate_pct\":%.4f,\"mean_accepted_per_step\":%.4f",
                    accept_rate, mean_acc);
            fprintf(f, ",\"ms_per_step_draft\":%.4f,\"ms_per_step_verify\":%.4f",
                    d_step, vf_step);
            fprintf(f, ",\"ms_per_step_logits_decode\":%.4f",
                    s.n_propose_steps > 0
                        ? (s.tgt_decode_ms - s.verify_features_ms) / s.n_propose_steps : 0.0);
            fprintf(f, ",\"ms_per_step_decode_submit\":%.4f",
                    s.n_propose_steps > 0 ? s.verify_decode_submit_ms / s.n_propose_steps : 0.0);
            fprintf(f, ",\"ms_per_step_accept\":%.4f",
                    s.n_propose_steps > 0 ? s.verify_accept_ms / s.n_propose_steps : 0.0);
            fprintf(f, ",\"ms_per_step_layer_commit\":%.4f",
                    s.n_propose_steps > 0 ? s.verify_layer_commit_ms / s.n_propose_steps : 0.0);
            fprintf(f, ",\"ms_per_step_feature_redecode\":%.4f",
                    s.n_propose_steps > 0 ? s.verify_features_ms / s.n_propose_steps : 0.0);
            fprintf(f, ",\"ms_per_step_process\":%.4f",
                    s.n_propose_steps > 0 ? s.verify_process_ms / s.n_propose_steps : 0.0);
            fprintf(f, ",\"ms_per_propose_total\":%.4f,\"tokens_per_propose\":%.4f",
                    d_step + vf_step, tok_step);
            fprintf(f, ",\"ms_per_token_effective\":%.4f",
                    tok_step > 0 ? (d_step + vf_step) / tok_step : 0.0);
            fprintf(f, ",\"ms_per_token_vanilla_fwd\":%.4f", v_fwd);
            fprintf(f, ",\"draft_ms_total\":%.4f,\"verify_ms_total\":%.4f",
                    s.draft_ms, s.verify_ms);
        }
        fputc('}', f);
    };

    fprintf(f, "{");
    fprintf(f, "\"temp\":%.4f,\"seed\":%u,\"n_predict\":%d,\"n_prompt_tokens\":%zu,\"n_ctx\":%d,",
            params.sampling.temp, params.sampling.seed, params.n_predict, n_inp, params.n_ctx);
    fprintf(f, "\"confidence_threshold\":%.6f,\"n_max\":%d,",
            params.speculative.dspark_confidence_threshold,
            params.speculative.draft.n_max);
    if (vanilla) {
        write_run("vanilla", *vanilla, false);
        fprintf(f, ",");
    }
    if (spec) {
        write_run("spec", *spec, true);
        fprintf(f, ",");
    }
    fprintf(f, "\"token_match\":%s,", token_match ? "true" : "false");
    if (first_mismatch_gen >= 0) {
        fprintf(f, "\"first_mismatch_gen\":%zd,", first_mismatch_gen);
    } else {
        fprintf(f, "\"first_mismatch_gen\":null,");
    }
    if (vanilla && spec) {
        const double tgp_v = vanilla->gen_ms > 0 ? 1000.0 * vanilla->n_generated / vanilla->gen_ms : 0.0;
        const double tgp_s = spec->gen_ms > 0 ? 1000.0 * spec->n_generated / spec->gen_ms : 0.0;
        fprintf(f, "\"tgp_speedup\":%.6f", tgp_v > 0 && tgp_s > 0 ? tgp_s / tgp_v : 0.0);
    } else {
        fprintf(f, "\"tgp_speedup\":null");
    }
    if (reference) {
        fprintf(f, ",\"reference_tokens\":%zu", reference->size());
    }
    fprintf(f, "}\n");
    fflush(f);
}

static double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
            clock::now().time_since_epoch()).count();
}

static bool ctx_save_state(llama_context * ctx, std::vector<uint8_t> & buf) {
    const size_t sz = llama_state_get_size(ctx);
    buf.resize(sz);
    return llama_state_get_data(ctx, buf.data(), sz) == sz;
}

static bool ctx_restore_state(llama_context * ctx, const std::vector<uint8_t> & buf) {
    return llama_state_set_data(ctx, buf.data(), buf.size()) == buf.size();
}

// Vanilla target-only greedy tokens: sequential one-decode-per-token from saved KV.
static llama_tokens vanilla_oracle_chain(
        llama_context * ctx_tgt,
        llama_batch & batch,
        llama_seq_id seq_id,
        llama_pos pos_verify,
        llama_token anchor,
        size_t n_tokens) {
    llama_tokens out;
    out.reserve(n_tokens);

    llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, pos_verify, -1);

    llama_token tok_in = anchor;
    for (size_t i = 0; i < n_tokens; ++i) {
        common_batch_clear(batch);
        common_batch_add(batch, tok_in, pos_verify + (llama_pos) i, { seq_id }, true);
        if (llama_decode(ctx_tgt, batch) != 0) {
            break;
        }
        llama_synchronize(ctx_tgt);
        const llama_model * model = llama_get_model(ctx_tgt);
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        const float * logits = llama_get_logits_ith(ctx_tgt, 0);
        if (!logits) {
            break;
        }
        const llama_token id = common_sampler_greedy_argmax(logits, n_vocab);
        out.push_back(id);
        tok_in = id;
    }
    return out;
}

static int run_vanilla(
        common_params & params,
        llama_context * ctx_tgt,
        common_sampler * smpl,
        std::vector<llama_token> & inp,
        int n_predict,
        run_stats * out) {
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx_tgt));

    llama_batch batch = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);

    // Match spec KV layout: full-prompt prefill + first-token sample. The old
    // inp.size()-1 prefill diverged from DSpark spec verify on Vulkan (e.g. Qwen3
    // agentic mismatch around gen 78).
    llama_token id_last = inp.back();
    int n_past = (int) inp.size();

    llama_batch prefill = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);
    for (size_t i = 0; i < inp.size(); ++i) {
        common_batch_add(prefill, inp[i], (llama_pos) i, { 0 }, i + 1 == inp.size());
    }
    out->n_prompt = (int) inp.size();
    const double tpp = now_ms();
    llama_decode(ctx_tgt, prefill);
    llama_batch_free(prefill);

    id_last = common_sampler_sample(smpl, ctx_tgt, -1, false);
    common_sampler_accept(smpl, id_last, true);
    out->pp_ms = now_ms() - tpp;

    out->output = inp;
    out->output.push_back(id_last);
    out->n_generated = 1;

    const double t0 = now_ms();

    while (!llama_vocab_is_eog(vocab, id_last) && out->n_generated < n_predict) {
        common_batch_clear(batch);
        common_batch_add(batch, id_last, n_past++, { 0 }, true);
        llama_decode(ctx_tgt, batch);

        const llama_token next = common_sampler_sample(smpl, ctx_tgt, -1, false);
        common_sampler_accept(smpl, next, true);

        out->output.push_back(next);
        id_last = next;
        out->n_generated++;

        if (llama_vocab_is_eog(vocab, next)) {
            break;
        }
    }

    out->gen_ms = now_ms() - t0;
    llama_batch_free(batch);
    llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, 0, -1);
    return 0;
}

static bool params_use_dspark_spec(const common_params & params) {
    for (auto t : params.speculative.types) {
        if (t == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK) {
            return true;
        }
    }
    return false;
}

static bool verify_standard_batched(
        common_speculative * spec,
        common_sampler * smpl,
        llama_context * ctx_tgt,
        llama_seq_id seq_id,
        llama_pos pos_verify,
        llama_token anchor,
        const llama_tokens & draft,
        llama_tokens & out_ids,
        llama_batch & batch) {
    out_ids.clear();

    llama_memory_t mem_tgt = llama_get_memory(ctx_tgt);
    llama_memory_seq_rm(mem_tgt, seq_id, pos_verify, -1);

    common_batch_clear(batch);
    common_batch_add(batch, anchor, pos_verify, { seq_id }, true);
    for (size_t i = 0; i < draft.size(); ++i) {
        common_batch_add(batch, draft[i], pos_verify + 1 + (llama_pos) i, { seq_id }, true);
    }

    if (llama_decode(ctx_tgt, batch) != 0) {
        return false;
    }

    if (spec != nullptr) {
        common_speculative_process(spec, batch);
    }

    out_ids = common_sampler_sample_and_accept_n(smpl, ctx_tgt, draft);
    if (out_ids.empty()) {
        return false;
    }

    llama_memory_seq_rm(mem_tgt, seq_id, pos_verify + (llama_pos) out_ids.size(), -1);
    return true;
}

// Generic speculative loop (draft-mtp, draft-simple, etc.) without DSpark verify/process.
static int run_speculative_standard(
        common_params & params,
        llama_context * ctx_tgt,
        llama_context * ctx_dft,
        common_sampler * smpl,
        common_speculative * spec,
        std::vector<llama_token> & inp,
        int n_predict,
        run_stats * out) {
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx_tgt));

    llama_tokens prompt_tgt(inp.begin(), inp.end());
    int n_past = (int) inp.size();

    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);
    llama_batch prefill   = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);

    for (size_t i = 0; i < inp.size(); ++i) {
        common_batch_add(prefill, inp[i], (llama_pos) i, { 0 }, i + 1 == inp.size());
    }
    out->n_prompt = (int) inp.size();
    const double tpp = now_ms();
    llama_decode(ctx_tgt, prefill);
    common_speculative_process(spec, prefill);
    common_speculative_begin(spec, 0, prompt_tgt);

    llama_token id_last = common_sampler_sample(smpl, ctx_tgt, -1, false);
    common_sampler_accept(smpl, id_last, true);
    out->pp_ms = now_ms() - tpp;

    out->output = inp;
    out->output.push_back(id_last);
    out->n_generated = 1;

    llama_tokens draft;
    const double t0 = now_ms();

    while (!llama_vocab_is_eog(vocab, id_last) && out->n_generated < n_predict) {
        draft.clear();
        common_speculative_get_draft_params(spec, 0) = {
            true, -1, n_past, id_last, &prompt_tgt, &draft, nullptr,
        };
        const double td0 = now_ms();
        common_speculative_draft(spec);
        out->draft_ms += now_ms() - td0;

        out->n_propose_steps++;
        out->n_drafted += (int) draft.size();

        const llama_pos pos_verify = n_past;
        llama_tokens ids;
        const double tv0 = now_ms();

        if (!verify_standard_batched(spec, smpl, ctx_tgt, 0, pos_verify, id_last, draft, ids, batch_tgt)) {
            fprintf(stderr, "error: standard verify failed\n");
            return 1;
        }

        out->verify_ms += now_ms() - tv0;

        const int n_acc = (int) ids.size() - 1;
        out->n_accepted += n_acc;

        if (n_acc > 0) {
            common_speculative_accept(spec, 0, (uint16_t) n_acc);
        }

        n_past = pos_verify + (int) ids.size();
        for (auto t : ids) {
            prompt_tgt.push_back(id_last);
            id_last = t;
            out->output.push_back(id_last);
            out->n_generated++;
            if (out->n_generated >= n_predict || llama_vocab_is_eog(vocab, id_last)) {
                break;
            }
        }

        llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, n_past, -1);
        if (ctx_dft) {
            llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, n_past, -1);
        }

        if (llama_vocab_is_eog(vocab, id_last)) {
            break;
        }
    }

    out->gen_ms = now_ms() - t0;
    llama_batch_free(batch_tgt);
    llama_batch_free(prefill);
    llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, 0, -1);
    if (ctx_dft) {
        llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, 0, -1);
    }
    return 0;
}

static int run_speculative(
        common_params & params,
        llama_context * ctx_tgt,
        llama_context * ctx_dft,
        common_sampler * smpl,
        common_speculative * spec,
        std::vector<llama_token> & inp,
        int n_predict,
        run_stats * out) {
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx_tgt));

    llama_tokens prompt_tgt(inp.begin(), inp.end());
    int n_past = (int) inp.size();

    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);

    // warmup target graphs at verify batch size (n_max + 1 tokens)
    if (!getenv("DSPARK_NO_WARMUP")) {
        const int32_t n_verify = params.speculative.draft.n_max + 1;
        llama_batch warm = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);
        const llama_token warm_tok = inp.empty() ? 0 : inp[0];
        for (int32_t i = 0; i < n_verify; ++i) {
            common_batch_add(warm, warm_tok, (llama_pos) i, { 0 }, true);
        }
        llama_decode(ctx_tgt, warm);
        llama_context * const ctx_feat = params.speculative.draft.ctx_tgt_feat;
        if (ctx_feat) {
            llama_decode(ctx_feat, warm);
        }
        llama_synchronize(ctx_tgt);
        llama_batch_free(warm);
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, 0, -1);
        if (ctx_feat) {
            llama_memory_seq_rm(llama_get_memory(ctx_feat), 0, 0, -1);
        }
    }

    // Prefill the FULL prompt and sample the first token from the target, then draft from
    // it (mirrors DeepSpec). This avoids a cold start where the last prompt token's target
    // features are not yet injected into ctx_dft on the first draft - that missing context
    // row makes the draft degenerate (repeat the anchor) for the first few steps.
    // common_speculative_process() reads pos/seq_id/n_seq_id directly, so the prefill batch
    // must be fully formed (llama_batch_get_one leaves those arrays null).
    llama_batch prefill = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);
    for (size_t i = 0; i < inp.size(); ++i) {
        common_batch_add(prefill, inp[i], (llama_pos) i, { 0 }, i + 1 == inp.size());
    }
    out->n_prompt = (int) inp.size();
    const double tpp = now_ms();
    llama_context * const ctx_feat = params.speculative.draft.ctx_tgt_feat;
    llama_context * const ctx_prefill = ctx_tgt;
    const bool prefill_defer = getenv("DSPARK_PREFILL_DEFER") != nullptr;
    if (prefill_defer) {
        llama_set_defer_layer_inp_extract(ctx_prefill, true);
    }
    llama_decode(ctx_prefill, prefill);
    if (ctx_feat) {
        llama_decode(ctx_feat, prefill);
    }
        if (prefill_defer) {
            llama_commit_layer_inputs(ctx_prefill, inp.size());
            llama_set_defer_layer_inp_extract(ctx_prefill, false);
        }
        if (!getenv("DSPARK_NO_PREFILL_PROCESS")) {
            common_speculative_process(spec, prefill);
        }
    if (!getenv("DSPARK_NO_BEGIN")) {
        common_speculative_begin(spec, 0, prompt_tgt);
    }

    llama_token id_last = common_sampler_sample(smpl, ctx_tgt, -1, false);
    common_sampler_accept(smpl, id_last, true);
    out->pp_ms = now_ms() - tpp;

    // fused verify skips full-vocab logits D2H; enable after prefill first sample
    const bool fused_verify = getenv("DSPARK_FUSED_ARGMAX") != nullptr
            && getenv("DSPARK_GPU_GREEDY") == nullptr;
    if (fused_verify) {
        llama_set_skip_host_logits(ctx_tgt, true);
    }

    out->output = inp;
    out->output.push_back(id_last);
    out->n_generated++;
    llama_tokens draft;
    const double t0 = now_ms();

    common_speculative_dspark_verify_kv_canon_reset();

    while (!llama_vocab_is_eog(vocab, id_last) && out->n_generated < n_predict) {
        auto spec_params = params.speculative;
        spec_params.dspark_temp = params.sampling.temp;
        spec_params.dspark_seed = params.sampling.seed;
        common_speculative_sync_params(spec, spec_params);

        draft.clear();
        if (!getenv("DSPARK_NO_DRAFT")) {
            common_speculative_get_draft_params(spec, 0) = {
                true, -1, n_past, id_last, &prompt_tgt, &draft, nullptr,
            };
            const double td0 = now_ms();
            common_speculative_draft(spec);
            out->draft_ms += now_ms() - td0;
        }

        out->n_propose_steps++;
        out->n_drafted += (int) draft.size();

        const llama_pos pos_verify = n_past;
        llama_tokens ids;
        const double tv0 = now_ms();

        std::vector<uint8_t> oracle_snap;
        const bool trace_oracle = getenv("DSPARK_TRACE_ORACLE") != nullptr;
        const bool verify_snap  = trace_oracle || getenv("DSPARK_VERIFY_PRE_SNAP") != nullptr;
        llama_token oracle_tok = LLAMA_TOKEN_NULL;
        llama_tokens oracle_chain;
        if (verify_snap) {
            if (!ctx_save_state(ctx_tgt, oracle_snap)) {
                fprintf(stderr, "DSPARK_TRACE_ORACLE: save_state failed at step %d\n",
                        out->n_propose_steps + 1);
            } else if (trace_oracle) {
                oracle_chain = vanilla_oracle_chain(
                        ctx_tgt, batch_tgt, 0, pos_verify, id_last, draft.size() + 1);
                oracle_tok = oracle_chain.empty() ? LLAMA_TOKEN_NULL : oracle_chain[0];
            }
            if (!oracle_snap.empty() && !ctx_restore_state(ctx_tgt, oracle_snap)) {
                fprintf(stderr, "DSPARK_TRACE_ORACLE: restore failed before verify\n");
            }
        }

        if (getenv("DSPARK_TRACE_VERIFY")) {
            char gen_buf[32];
            snprintf(gen_buf, sizeof(gen_buf), "%d", out->n_generated);
            setenv("DSPARK_TRACE_VERIFY_GEN", gen_buf, 1);
        }

        if (getenv("DSPARK_VANILLA_VERIFY")) {
            common_batch_clear(batch_tgt);
            common_batch_add(batch_tgt, id_last, pos_verify, { 0 }, true);
            llama_decode(ctx_tgt, batch_tgt);
            const llama_token next = common_sampler_sample(smpl, ctx_tgt, -1, false);
            common_sampler_accept(smpl, next, true);
            ids.assign(1, next);
        } else {
            common_speculative_dspark_verify_timing vtim {};
            if (!common_speculative_dspark_verify_batched(
                    spec, smpl, ctx_tgt, 0, pos_verify, id_last, draft, ids, batch_tgt, &vtim)) {
                fprintf(stderr, "error: dspark verify failed\n");
                return 1;
            }
            out->tgt_decode_ms     += vtim.logits_decode_ms + vtim.features_decode_ms;
            out->verify_accept_ms  += vtim.accept_ms;
            out->verify_layer_commit_ms += vtim.layer_commit_ms;
            out->verify_features_ms += vtim.features_decode_ms;
            out->verify_process_ms += vtim.process_ms;
            out->verify_decode_submit_ms += vtim.decode_submit_ms;
        }

        out->verify_ms += now_ms() - tv0;

        if (trace_oracle && !oracle_chain.empty() && !ids.empty()) {
            const size_t n_cmp = std::min(oracle_chain.size(), ids.size());
            for (size_t i = 0; i < n_cmp; ++i) {
                if (ids[i] == oracle_chain[i]) {
                    continue;
                }
                fprintf(stderr,
                        "DSPARK_TRACE_ORACLE: CHAIN MISMATCH step=%d gen_index=%d "
                        "pos_verify=%d at ids[%zu]: batched=%d vanilla=%d draft_n=%zu\n",
                        out->n_propose_steps, out->n_generated, (int) pos_verify,
                        i, (int) ids[i], (int) oracle_chain[i], draft.size());
                break;
            }
        }

        const int n_acc = (int) ids.size() - 1;
        out->n_accepted += n_acc;

        if (getenv("DSPARK_DEBUG")) {
            fprintf(stderr, "\n[step %2d] anchor='%s'\n", out->n_propose_steps,
                    common_token_to_piece(ctx_tgt, id_last, true).c_str());
            fprintf(stderr, "  proposed (%d):", (int) draft.size());
            for (auto d : draft) fprintf(stderr, " '%s'", common_token_to_piece(ctx_tgt, d, true).c_str());
            fprintf(stderr, "\n  accepted %d/%d:", n_acc, (int) draft.size());
            for (int i = 0; i < n_acc; ++i) fprintf(stderr, " '%s'", common_token_to_piece(ctx_tgt, draft[i], true).c_str());
            fprintf(stderr, "\n  committed (+bonus):");
            for (auto t : ids) fprintf(stderr, " '%s'", common_token_to_piece(ctx_tgt, t, true).c_str());
            fputc('\n', stderr);
        }

        if (n_acc > 0) {
            common_speculative_accept(spec, 0, (uint16_t) n_acc);
        }

        n_past = pos_verify + (int) ids.size();
        for (auto t : ids) {
            prompt_tgt.push_back(id_last);
            id_last = t;
            out->output.push_back(id_last);
            out->n_generated++;
            if (out->n_generated >= n_predict || llama_vocab_is_eog(vocab, id_last)) {
                break;
            }
        }

        if (!getenv("DSPARK_NO_KV_TRIM")) {
            llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, n_past, -1);
            llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, n_past, -1);
        }

        if (llama_vocab_is_eog(vocab, id_last)) {
            break;
        }
    }

    out->gen_ms = now_ms() - t0;
    llama_batch_free(batch_tgt);
    llama_batch_free(prefill);
    llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, 0, -1);
    llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, 0, -1);
    return 0;
}

static void print_tokens(const char * label, const std::vector<llama_token> & toks, size_t skip) {
    fprintf(stderr, "%s (%zu tokens):", label, toks.size() > skip ? toks.size() - skip : 0);
    for (size_t i = skip; i < toks.size(); ++i) {
        fprintf(stderr, " %d", (int) toks[i]);
    }
    fputc('\n', stderr);
}

static void print_text(llama_context * ctx, const char * label,
        const std::vector<llama_token> & toks, size_t skip) {
    std::string text;
    for (size_t i = skip; i < toks.size(); ++i) {
        text += common_token_to_piece(ctx, toks[i], true);
    }
    fprintf(stderr, "%s text: %s\n", label, text.c_str());
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();

    std::string input_ids_path;
    bench_flags bflags;

    std::vector<char *> fargv;
    fargv.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--input-ids") == 0 && i + 1 < argc) {
            input_ids_path = argv[++i];
        } else if (strcmp(argv[i], "--vanilla-only") == 0) {
            bflags.vanilla_only = true;
        } else if (strcmp(argv[i], "--spec-only") == 0) {
            bflags.spec_only = true;
        } else if (strcmp(argv[i], "--reference") == 0 && i + 1 < argc) {
            bflags.reference_path = argv[++i];
        } else if (strcmp(argv[i], "--save-output") == 0 && i + 1 < argc) {
            bflags.save_output_path = argv[++i];
        } else if (strcmp(argv[i], "--json-results") == 0 && i + 1 < argc) {
            bflags.json_results_path = argv[++i];
        } else {
            fargv.push_back(argv[i]);
        }
    }

    if (bflags.vanilla_only && bflags.spec_only) {
        fprintf(stderr, "error: --vanilla-only and --spec-only are mutually exclusive\n");
        return 1;
    }

    if (!common_params_parse((int) fargv.size(), fargv.data(), params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.model.path.empty() || input_ids_path.empty()) {
        usage(argv[0]);
        return 1;
    }

    if (bflags.spec_only && params.speculative.draft.mparams.path.empty()) {
        fprintf(stderr, "error: --spec-only requires -md DRAFT.gguf\n");
        return 1;
    }

    if (bflags.spec_only && bflags.reference_path.empty()) {
        fprintf(stderr, "error: --spec-only requires --reference PATH.json\n");
        return 1;
    }

    // n_ctx=0 loads the model's trained context and allocates a full KV cache (OOM risk).
    if (params.n_ctx == 0) {
        params.n_ctx = 8096;
        fprintf(stderr, "note: defaulting context to %d (pass -c to override)\n", params.n_ctx);
    }

    const bool has_draft = !bflags.vanilla_only && !params.speculative.draft.mparams.path.empty();
    if (has_draft && params.speculative.types.empty()) {
        params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK };
    }
    const bool use_dspark_spec = has_draft && params_use_dspark_spec(params);

    if (use_dspark_spec && params.n_parallel < 2 && getenv("DSPARK_VERIFY_PARALLEL") != nullptr) {
        fprintf(stderr, "note: DSpark parallel verify needs n_parallel >= 2; using 2\n");
        params.n_parallel = 2;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init_tgt = common_init_from_params(params);
    llama_context * ctx_tgt = llama_init_tgt->context();

    llama_model_ptr model_dft;
    llama_context_ptr ctx_dft;
    llama_context_ptr ctx_tgt_feat;
    common_speculative * spec = nullptr;

    if (has_draft) {
        auto params_dft = params;
        params_dft.model         = params.speculative.draft.mparams;
        params_dft.n_gpu_layers  = params.speculative.draft.n_gpu_layers;
        params_dft.devices       = params.speculative.draft.devices;
        if (params.speculative.draft.cpuparams.n_threads > 0) {
            params_dft.cpuparams.n_threads = params.speculative.draft.cpuparams.n_threads;
        }
        params_dft.cpuparams_batch.n_threads = params.speculative.draft.cpuparams_batch.n_threads;
        params_dft.tensor_buft_overrides     = params.speculative.draft.tensor_buft_overrides;
        model_dft.reset(llama_model_load_from_file(params_dft.model.path.c_str(),
                    common_model_params_to_llama(params_dft)));
        if (!model_dft) {
            fprintf(stderr, "failed to load draft model\n");
            return 1;
        }
        ctx_dft.reset(llama_init_from_model(model_dft.get(), common_context_params_to_llama(params_dft)));
        params.speculative.draft.ctx_tgt = ctx_tgt;
        params.speculative.draft.ctx_dft = ctx_dft.get();
        if (!getenv("DSPARK_NO_SPLIT_VERIFY") && getenv("DSPARK_SPLIT_VERIFY") && use_dspark_spec) {
            llama_context_params cparams_feat = common_context_params_to_llama(params);
            cparams_feat.ctx_other = ctx_tgt;
            ctx_tgt_feat.reset(llama_init_from_model(
                        const_cast<llama_model *>(llama_get_model(ctx_tgt)), cparams_feat));
            if (!ctx_tgt_feat) {
                fprintf(stderr, "warning: failed to create ctx_tgt_feat; using single-pass verify\n");
            } else {
                params.speculative.draft.ctx_tgt_feat = ctx_tgt_feat.get();
                fprintf(stderr, "note: split target verify enabled (ctx_tgt logits + ctx_tgt_feat layers)\n");
            }
        }
        spec = common_speculative_init(params.speculative, 1);
    }

    std::vector<llama_token> inp;
    if (!common_load_input_ids_json(input_ids_path, inp)) {
        return 1;
    }

    std::vector<llama_token> reference;
    if (!bflags.reference_path.empty()) {
        if (!common_load_input_ids_json(bflags.reference_path, reference)) {
            return 1;
        }
    }

    common_sampler_ptr smpl(common_sampler_init(llama_get_model(ctx_tgt), params.sampling));

    const int n_predict = params.n_predict > 0 ? params.n_predict : 32;
    const size_t n_inp = inp.size();

    run_stats vanilla {};
    run_stats spec_stats {};
    bool have_vanilla = false;
    bool have_spec    = false;

    if (!bflags.spec_only) {
        if (run_vanilla(params, ctx_tgt, smpl.get(), inp, n_predict, &vanilla) != 0) {
            return 1;
        }
        have_vanilla = true;
        if (!bflags.save_output_path.empty()) {
            if (!io_save_tokens(bflags.save_output_path, vanilla.output)) {
                fprintf(stderr, "warning: failed to save output to %s\n", bflags.save_output_path.c_str());
            }
        }
    }

    if (has_draft) {
        llama_synchronize(ctx_tgt);
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, 0, -1);
        if (ctx_dft) {
            llama_memory_seq_rm(llama_get_memory(ctx_dft.get()), 0, 0, -1);
        }
        if (!bflags.spec_only) {
            const int cooldown_s = getenv("DSPARK_BENCH_NO_COOLDOWN") ? 0 : 3;
            if (cooldown_s > 0) {
                fprintf(stderr, "note: cooling down %ds before speculative run (DSPARK_BENCH_NO_COOLDOWN=1 to skip)\n",
                        cooldown_s);
                sleep((unsigned) cooldown_s);
            }
        }
        common_sampler_reset(smpl.get());
        if (use_dspark_spec) {
            if (run_speculative(params, ctx_tgt, ctx_dft.get(), smpl.get(), spec, inp, n_predict, &spec_stats) != 0) {
                return 1;
            }
        } else if (run_speculative_standard(params, ctx_tgt, ctx_dft.get(), smpl.get(), spec, inp, n_predict, &spec_stats) != 0) {
            return 1;
        }
        have_spec = true;
        if (!bflags.save_output_path.empty() && bflags.spec_only) {
            if (!io_save_tokens(bflags.save_output_path, spec_stats.output)) {
                fprintf(stderr, "warning: failed to save output to %s\n", bflags.save_output_path.c_str());
            }
        }
    }

    fprintf(stderr, "\n=== Vanilla (target only) ===\n");
    if (have_vanilla) {
    fprintf(stderr, "prompt: %d tokens, pp %.1f ms (%.2f tok/s)\n",
            vanilla.n_prompt, vanilla.pp_ms,
            vanilla.pp_ms > 0 ? 1000.0 * vanilla.n_prompt / vanilla.pp_ms : 0.0);
    fprintf(stderr, "generated: %d tokens in %.1f ms (tgp %.2f tok/s)\n",
            vanilla.n_generated, vanilla.gen_ms,
            vanilla.n_generated > 0 ? 1000.0 * vanilla.n_generated / vanilla.gen_ms : 0.0);
    print_tokens("output", vanilla.output, n_inp);
    print_text(ctx_tgt, "output", vanilla.output, n_inp);
    } else {
        fprintf(stderr, "(skipped)\n");
    }

    if (have_spec) {
        fprintf(stderr, "\n=== Speculative (%s) ===\n",
                use_dspark_spec ? "draft-dspark" : common_speculative_type_to_str(params.speculative.types[0]).c_str());
        fprintf(stderr, "prompt: %d tokens, pp+setup %.1f ms (%.2f tok/s)\n",
                spec_stats.n_prompt, spec_stats.pp_ms,
                spec_stats.pp_ms > 0 ? 1000.0 * spec_stats.n_prompt / spec_stats.pp_ms : 0.0);
        fprintf(stderr, "generated: %d tokens in %.1f ms (tgp %.2f tok/s)\n",
                spec_stats.n_generated, spec_stats.gen_ms,
                spec_stats.n_generated > 0 ? 1000.0 * spec_stats.n_generated / spec_stats.gen_ms : 0.0);
        fprintf(stderr, "propose steps: %d\n", spec_stats.n_propose_steps);
        fprintf(stderr, "drafted tokens: %d\n", spec_stats.n_drafted);
        fprintf(stderr, "accepted tokens: %d\n", spec_stats.n_accepted);
        if (spec_stats.n_drafted > 0) {
            fprintf(stderr, "accept rate (hit rate): %.1f%%\n", 100.0 * spec_stats.n_accepted / spec_stats.n_drafted);
        }
        if (spec_stats.n_propose_steps > 0) {
            fprintf(stderr, "mean accepted/step: %.2f\n",
                    (double) spec_stats.n_accepted / (double) spec_stats.n_propose_steps);
            const double v_fwd = have_vanilla && vanilla.n_generated > 0
                    ? vanilla.gen_ms / vanilla.n_generated : 0.0;
            const double d_step = spec_stats.draft_ms  / spec_stats.n_propose_steps;
            const double vf_step = spec_stats.verify_ms / spec_stats.n_propose_steps;
            const double tok_step = (double) spec_stats.n_generated / spec_stats.n_propose_steps;
            fprintf(stderr, "--- per-pass timing ---\n");
            fprintf(stderr, "  target forward (vanilla, 1 tok)   : %.2f ms\n", v_fwd);
            fprintf(stderr, "  draft  step (fwd + cpu sampling)  : %.2f ms  (proposes %d)\n", d_step, (int) spec_stats.n_drafted / std::max(1, spec_stats.n_propose_steps));
            fprintf(stderr, "  verify step (batched)            : %.2f ms\n", vf_step);
            fprintf(stderr, "    logits decode (full block)     : %.2f ms\n",
                    (spec_stats.tgt_decode_ms - spec_stats.verify_features_ms) / spec_stats.n_propose_steps);
            fprintf(stderr, "    decode submit (no sync)      : %.2f ms\n",
                    spec_stats.verify_decode_submit_ms / spec_stats.n_propose_steps);
            fprintf(stderr, "    accept (GPU sync + argmax)     : %.2f ms\n",
                    spec_stats.verify_accept_ms / spec_stats.n_propose_steps);
            fprintf(stderr, "    layer commit (partial D2H)     : %.2f ms\n",
                    spec_stats.verify_layer_commit_ms / spec_stats.n_propose_steps);
            fprintf(stderr, "    feature re-decode (committed)  : %.2f ms\n",
                    spec_stats.verify_features_ms / spec_stats.n_propose_steps);
            fprintf(stderr, "    process (encode + KV inject)   : %.2f ms\n",
                    spec_stats.verify_process_ms / spec_stats.n_propose_steps);
            fprintf(stderr, "  => %.2f ms/propose for %.2f tokens = %.2f ms/token (vanilla %.2f ms/token)\n",
                    d_step + vf_step, tok_step,
                    tok_step > 0 ? (d_step + vf_step) / tok_step : 0.0, v_fwd);
        }
        print_tokens("output", spec_stats.output, n_inp);
        print_text(ctx_tgt, "output", spec_stats.output, n_inp);

        fprintf(stderr, "\n=== Comparison ===\n");
        const std::vector<llama_token> & ref_out = !reference.empty()
                ? reference
                : (have_vanilla ? vanilla.output : spec_stats.output);
        const size_t n_cmp = std::min(ref_out.size(), spec_stats.output.size());
        size_t first_mismatch = n_cmp;
        for (size_t i = n_inp; i < n_cmp; ++i) {
            if (ref_out[i] != spec_stats.output[i]) {
                first_mismatch = i;
                break;
            }
        }
        const bool prefix_match = first_mismatch == n_cmp;
        const char * ref_label = !reference.empty() ? "reference" : "vanilla";
        fprintf(stderr, "token match on common prefix (temp=%.2f): %s\n",
                params.sampling.temp, prefix_match ? "YES" : "NO");
        if (!prefix_match) {
            fprintf(stderr, "first mismatch at gen index %zu: %s=%d spec=%d\n",
                    first_mismatch - n_inp, ref_label,
                    (int) ref_out[first_mismatch], (int) spec_stats.output[first_mismatch]);
        }
        // tokens/s already accounts for token count, so this is a fair throughput ratio
        const double pp_v  = have_vanilla && vanilla.pp_ms > 0
                ? 1000.0 * vanilla.n_prompt / vanilla.pp_ms : 0.0;
        const double pp_s  = spec_stats.pp_ms > 0 ? 1000.0 * spec_stats.n_prompt / spec_stats.pp_ms : 0.0;
        const double tgp_v = have_vanilla && vanilla.gen_ms > 0
                ? 1000.0 * vanilla.n_generated / vanilla.gen_ms : 0.0;
        const double tgp_s = spec_stats.gen_ms > 0 ? 1000.0 * spec_stats.n_generated / spec_stats.gen_ms : 0.0;
        if (have_vanilla) {
            fprintf(stderr, "--- throughput ---\n");
            fprintf(stderr, "  pp  (prefill):  vanilla %6.2f tok/s  |  spec %6.2f tok/s", pp_v, pp_s);
            if (pp_v > 0 && pp_s > 0) {
                fprintf(stderr, "  (%.2fx)", pp_s / pp_v);
            }
            fputc('\n', stderr);
            fprintf(stderr, "  tgp (generate): vanilla %6.2f tok/s  |  spec %6.2f tok/s", tgp_v, tgp_s);
            if (tgp_v > 0 && tgp_s > 0) {
                fprintf(stderr, "  (%.2fx)", tgp_s / tgp_v);
            }
            fputc('\n', stderr);
            if (tgp_v > 0 && tgp_s > 0) {
                fprintf(stderr, "tgp speedup: %.2fx\n", tgp_s / tgp_v);
            }
        } else {
            fprintf(stderr, "spec tgp: %.2f tok/s\n", tgp_s);
        }

        if (!bflags.json_results_path.empty()) {
            FILE * jf = fopen(bflags.json_results_path.c_str(), "w");
            if (jf) {
                write_json_results(
                        jf, params, n_inp,
                        have_vanilla ? &vanilla : nullptr,
                        &spec_stats,
                        reference.empty() ? nullptr : &reference,
                        prefix_match,
                        prefix_match ? -1 : (ssize_t) (first_mismatch - n_inp));
                fclose(jf);
            } else {
                fprintf(stderr, "warning: failed to write json results to %s\n", bflags.json_results_path.c_str());
            }
        }
    } else if (have_vanilla && !bflags.json_results_path.empty()) {
        FILE * jf = fopen(bflags.json_results_path.c_str(), "w");
        if (jf) {
            write_json_results(jf, params, n_inp, &vanilla, nullptr, nullptr, true, -1);
            fclose(jf);
        }
    }

    if (spec) {
        common_speculative_free(spec);
    }
    llama_backend_free();
    return 0;
}
