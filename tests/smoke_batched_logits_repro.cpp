#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include "../src/llama-ext.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct step_snapshot {
    int         step        = 0;
    int         gen_index   = 0;
    llama_pos   pos_verify  = 0;
    llama_token anchor      = 0;
    llama_tokens draft;
};

static void usage(const char * argv0) {
    fprintf(stderr,
            "Usage: %s -m TARGET.gguf -md DRAFT.gguf --input-ids PATH "
            "[--step N] [--gen-index N] [-c 512] [-ngl N] [-ngld N] "
            "[--temp 0] [--seed 42] [--spec-draft-n-max 4]\n"
            "\n"
            "Repro: at each DSpark verify step (or one chosen step), compare row-0 greedy\n"
            "logits from single-token decode vs multi-token batched decode on identical KV.\n",
            argv0);
}

static llama_token greedy_argmax(const float * logits, int n_vocab) {
    int best = 0;
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    return (llama_token) best;
}

static bool save_state(llama_context * ctx, std::vector<uint8_t> & buf) {
    const size_t sz = llama_state_get_size(ctx);
    buf.resize(sz);
    return llama_state_get_data(ctx, buf.data(), sz) == sz;
}

static bool restore_state(llama_context * ctx, const std::vector<uint8_t> & buf) {
    return llama_state_set_data(ctx, buf.data(), buf.size()) == buf.size();
}

static llama_token decode_row0(
        llama_context * ctx,
        llama_batch & batch,
        llama_seq_id seq_id,
        llama_pos pos,
        llama_token tok) {
    common_batch_clear(batch);
    common_batch_add(batch, tok, pos, { seq_id }, true);
    if (llama_decode(ctx, batch) != 0) {
        return LLAMA_TOKEN_NULL;
    }
    llama_synchronize(ctx);
    const llama_model * model = llama_get_model(ctx);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float * logits = llama_get_logits_ith(ctx, 0);
    if (!logits) {
        return LLAMA_TOKEN_NULL;
    }
    return greedy_argmax(logits, n_vocab);
}

static llama_token decode_batched_row0(
        llama_context * ctx,
        llama_batch & batch,
        llama_seq_id seq_id,
        llama_pos pos_verify,
        llama_token anchor,
        const llama_tokens & draft) {
    llama_memory_seq_rm(llama_get_memory(ctx), seq_id, pos_verify, -1);

    common_batch_clear(batch);
    common_batch_add(batch, anchor, pos_verify, { seq_id }, true);
    for (size_t i = 0; i < draft.size(); ++i) {
        common_batch_add(batch, draft[i], pos_verify + 1 + (llama_pos) i, { seq_id }, true);
    }

    if (llama_decode(ctx, batch) != 0) {
        return LLAMA_TOKEN_NULL;
    }
    llama_synchronize(ctx);

    const llama_model * model = llama_get_model(ctx);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float * logits = llama_get_logits_ith(ctx, 0);
    if (!logits) {
        return LLAMA_TOKEN_NULL;
    }
    return greedy_argmax(logits, n_vocab);
}

static void logit_diff_stats(
        llama_context * ctx,
        const float * logits_a,
        const float * logits_b,
        int n_vocab,
        int top_k,
        llama_token * out_best_a,
        llama_token * out_best_b,
        float * out_max_abs_diff) {
    *out_best_a = greedy_argmax(logits_a, n_vocab);
    *out_best_b = greedy_argmax(logits_b, n_vocab);

    float max_abs = 0.0f;
    for (int i = 0; i < n_vocab; ++i) {
        max_abs = std::max(max_abs, std::fabs(logits_a[i] - logits_b[i]));
    }
    *out_max_abs_diff = max_abs;

    fprintf(stderr, "  top-%d logit deltas (single vs multi row0):\n", top_k);
    struct entry { int id; float da; };
    std::vector<entry> deltas;
    deltas.reserve((size_t) n_vocab);
    for (int i = 0; i < n_vocab; ++i) {
        deltas.push_back({ i, logits_a[i] - logits_b[i] });
    }
    std::sort(deltas.begin(), deltas.end(), [](const entry & a, const entry & b) {
        return std::fabs(a.da) > std::fabs(b.da);
    });
    for (int i = 0; i < top_k && i < n_vocab; ++i) {
        fprintf(stderr, "    tok %d: single=%.4f multi=%.4f delta=%.4f\n",
                deltas[i].id, logits_a[deltas[i].id], logits_b[deltas[i].id], deltas[i].da);
    }
    (void) ctx;
}

static bool compare_row0_logits(
        llama_context * ctx,
        common_speculative * spec,
        llama_batch & batch,
        const std::vector<uint8_t> & state,
        llama_seq_id seq_id,
        const step_snapshot & snap,
        bool verbose,
        bool with_dspark_features,
        bool defer_layers) {
    const llama_model * model = llama_get_model(ctx);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    if (with_dspark_features && spec) {
        common_speculative_dspark_target_features_enable(spec, defer_layers);
    }

    if (!restore_state(ctx, state)) {
        fprintf(stderr, "restore_state failed\n");
        return false;
    }

    if (defer_layers) {
        llama_set_defer_layer_inp_extract(ctx, true);
    }

    llama_memory_seq_rm(llama_get_memory(ctx), seq_id, snap.pos_verify, -1);
    common_batch_clear(batch);
    common_batch_add(batch, snap.anchor, snap.pos_verify, { seq_id }, true);
    if (llama_decode(ctx, batch) != 0) {
        return false;
    }
    llama_synchronize(ctx);
    std::vector<float> logits_single((size_t) n_vocab);
    {
        const float * row = llama_get_logits_ith(ctx, 0);
        if (!row) {
            return false;
        }
        std::memcpy(logits_single.data(), row, (size_t) n_vocab * sizeof(float));
    }

    if (!restore_state(ctx, state)) {
        return false;
    }

    if (defer_layers) {
        llama_set_defer_layer_inp_extract(ctx, true);
    }

    llama_memory_seq_rm(llama_get_memory(ctx), seq_id, snap.pos_verify, -1);
    common_batch_clear(batch);
    common_batch_add(batch, snap.anchor, snap.pos_verify, { seq_id }, true);
    for (size_t i = 0; i < snap.draft.size(); ++i) {
        common_batch_add(batch, snap.draft[i], snap.pos_verify + 1 + (llama_pos) i, { seq_id }, true);
    }
    if (llama_decode(ctx, batch) != 0) {
        return false;
    }
    llama_synchronize(ctx);
    std::vector<float> logits_multi((size_t) n_vocab);
    {
        const float * row = llama_get_logits_ith(ctx, 0);
        if (!row) {
            return false;
        }
        std::memcpy(logits_multi.data(), row, (size_t) n_vocab * sizeof(float));
    }

    if (defer_layers) {
        llama_set_defer_layer_inp_extract(ctx, false);
    }

    llama_token best_single = 0;
    llama_token best_multi  = 0;
    float max_abs_diff      = 0.0f;
    logit_diff_stats(ctx, logits_single.data(), logits_multi.data(), n_vocab, 5,
            &best_single, &best_multi, &max_abs_diff);

    const bool match = best_single == best_multi;
    fprintf(stderr,
            "step %d gen_index %d pos %d anchor %d draft_n %zu features=%d defer=%d: "
            "row0 single=%d multi=%d max_abs_diff=%.6f %s\n",
            snap.step, snap.gen_index, (int) snap.pos_verify, (int) snap.anchor,
            snap.draft.size(), (int) with_dspark_features, (int) defer_layers,
            (int) best_single, (int) best_multi, max_abs_diff,
            match ? "MATCH" : "MISMATCH");

    if (verbose || !match) {
        fprintf(stderr, "  draft:");
        for (auto t : snap.draft) {
            fprintf(stderr, " %d", (int) t);
        }
        fputc('\n', stderr);
    }

    return match;
}

static bool compare_row0_logits(
        llama_context * ctx,
        common_speculative * spec,
        llama_batch & batch,
        const std::vector<uint8_t> & state,
        llama_seq_id seq_id,
        const step_snapshot & snap,
        bool verbose) {
    return compare_row0_logits(ctx, spec, batch, state, seq_id, snap, verbose, false, false);
}

static void compare_all_rows(
        llama_context * ctx,
        llama_batch & batch,
        const std::vector<uint8_t> & state,
        llama_seq_id seq_id,
        const step_snapshot & snap) {
    const llama_model * model = llama_get_model(ctx);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int n_rows  = (int) snap.draft.size() + 1;

    std::vector<llama_token> seq_argmax((size_t) n_rows, LLAMA_TOKEN_NULL);
    std::vector<llama_token> bat_argmax((size_t) n_rows, LLAMA_TOKEN_NULL);

    if (!restore_state(ctx, state)) {
        return;
    }
    llama_memory_seq_rm(llama_get_memory(ctx), seq_id, snap.pos_verify, -1);
    for (int row = 0; row < n_rows; ++row) {
        const llama_token tok_in = row == 0 ? snap.anchor : snap.draft[(size_t) row - 1];
        common_batch_clear(batch);
        common_batch_add(batch, tok_in, snap.pos_verify + row, { seq_id }, true);
        if (llama_decode(ctx, batch) != 0) {
            return;
        }
        llama_synchronize(ctx);
        const float * logits = llama_get_logits_ith(ctx, 0);
        if (!logits) {
            return;
        }
        seq_argmax[(size_t) row] = greedy_argmax(logits, n_vocab);
    }

    if (!restore_state(ctx, state)) {
        return;
    }
    llama_memory_seq_rm(llama_get_memory(ctx), seq_id, snap.pos_verify, -1);
    common_batch_clear(batch);
    common_batch_add(batch, snap.anchor, snap.pos_verify, { seq_id }, true);
    for (size_t i = 0; i < snap.draft.size(); ++i) {
        common_batch_add(batch, snap.draft[i], snap.pos_verify + 1 + (llama_pos) i, { seq_id }, true);
    }
    if (llama_decode(ctx, batch) != 0) {
        return;
    }
    llama_synchronize(ctx);
    for (int row = 0; row < n_rows; ++row) {
        const float * logits = llama_get_logits_ith(ctx, row);
        if (!logits) {
            return;
        }
        bat_argmax[(size_t) row] = greedy_argmax(logits, n_vocab);
    }

    fprintf(stderr, "per-row greedy (sequential chain vs batched rows):\n");
    for (int row = 0; row < n_rows; ++row) {
        const bool match = seq_argmax[(size_t) row] == bat_argmax[(size_t) row];
        fprintf(stderr, "  row %d: seq=%d bat=%d %s\n",
                row, (int) seq_argmax[(size_t) row], (int) bat_argmax[(size_t) row],
                match ? "MATCH" : "MISMATCH");
    }
}

static int run_spec_until(
        common_params & params,
        llama_context * ctx_tgt,
        llama_context * ctx_dft,
        common_sampler * smpl,
        common_speculative * spec,
        std::vector<llama_token> & inp,
        int n_predict,
        int target_step,
        int target_gen_index,
        step_snapshot * out_snap,
        std::vector<uint8_t> * out_state) {
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx_tgt));

    llama_tokens prompt_tgt(inp.begin(), inp.end());
    int n_past = (int) inp.size();

    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);
    llama_batch prefill   = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);

    for (size_t i = 0; i < inp.size(); ++i) {
        common_batch_add(prefill, inp[i], (llama_pos) i, { 0 }, i + 1 == inp.size());
    }
    llama_context * const ctx_feat = params.speculative.draft.ctx_tgt_feat;
    llama_decode(ctx_tgt, prefill);
    if (ctx_feat) {
        llama_decode(ctx_feat, prefill);
    }
    common_speculative_process(spec, prefill);
    common_speculative_begin(spec, 0, prompt_tgt);

    llama_token id_last = common_sampler_sample(smpl, ctx_tgt, -1, false);
    common_sampler_accept(smpl, id_last, true);

    int n_generated = 1;
    int step        = 0;

    while (!llama_vocab_is_eog(vocab, id_last) && n_generated < n_predict) {
        ++step;

        auto spec_params = params.speculative;
        spec_params.dspark_temp = params.sampling.temp;
        spec_params.dspark_seed = params.sampling.seed;
        common_speculative_sync_params(spec, spec_params);

        llama_tokens draft;
        common_speculative_get_draft_params(spec, 0) = {
            true, -1, n_past, id_last, &prompt_tgt, &draft, nullptr,
        };
        common_speculative_draft(spec);

        const llama_pos pos_verify = n_past;
        const int gen_index        = n_generated;

        const bool hit_step = target_step >= 0 && step == target_step;
        const bool hit_gen  = target_gen_index >= 0 && gen_index == target_gen_index;

        if (hit_step || hit_gen) {
            if (out_snap) {
                out_snap->step       = step;
                out_snap->gen_index  = gen_index;
                out_snap->pos_verify = pos_verify;
                out_snap->anchor     = id_last;
                out_snap->draft      = draft;
            }
            if (out_state && !save_state(ctx_tgt, *out_state)) {
                fprintf(stderr, "failed to save state at step %d\n", step);
                return 1;
            }
            if (target_step >= 0 || target_gen_index >= 0) {
                llama_batch_free(batch_tgt);
                llama_batch_free(prefill);
                return 0;
            }
        }

        llama_tokens ids;
        if (!common_speculative_dspark_verify_sequential(
                spec, smpl, ctx_tgt, 0, pos_verify, id_last, draft, ids, batch_tgt)) {
            fprintf(stderr, "sequential verify failed at step %d\n", step);
            return 1;
        }

        common_speculative_accept(spec, 0, (uint16_t) (ids.size() > 0 ? ids.size() - 1 : 0));
        n_past = pos_verify + (int) ids.size();
        for (auto t : ids) {
            prompt_tgt.push_back(id_last);
            id_last = t;
            ++n_generated;
            if (llama_vocab_is_eog(vocab, id_last)) {
                break;
            }
        }

        llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, n_past, -1);
        llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, n_past, -1);

        if (llama_vocab_is_eog(vocab, id_last)) {
            break;
        }
    }

    llama_batch_free(batch_tgt);
    llama_batch_free(prefill);
    return 0;
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();

    std::string input_ids_path;
    int target_step      = -1;
    int target_gen_index = 54;
    bool scan_all        = false;

    std::vector<char *> fargv;
    fargv.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--input-ids") == 0 && i + 1 < argc) {
            input_ids_path = argv[++i];
        } else if (strcmp(argv[i], "--step") == 0 && i + 1 < argc) {
            target_step = atoi(argv[++i]);
            target_gen_index = -1;
        } else if (strcmp(argv[i], "--gen-index") == 0 && i + 1 < argc) {
            target_gen_index = atoi(argv[++i]);
            target_step = -1;
        } else if (strcmp(argv[i], "--scan-all") == 0) {
            scan_all = true;
            target_step = -1;
            target_gen_index = -1;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fargv.push_back(argv[i]);
        }
    }

    if (!common_params_parse((int) fargv.size(), fargv.data(), params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.model.path.empty() || params.speculative.draft.mparams.path.empty()
            || input_ids_path.empty()) {
        usage(argv[0]);
        return 1;
    }

    if (params.n_ctx == 0) {
        params.n_ctx = 512;
    }

    params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK };
    setenv("DSPARK_NO_ADAPTIVE_NMAX", "1", 0);

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init_tgt = common_init_from_params(params);
    llama_context * ctx_tgt = llama_init_tgt->context();

    auto params_dft = params;
    params_dft.model = params.speculative.draft.mparams;
    params_dft.n_gpu_layers = params.speculative.draft.n_gpu_layers;
    auto model_dft = llama_model_ptr(llama_model_load_from_file(
                params_dft.model.path.c_str(), common_model_params_to_llama(params_dft)));
    if (!model_dft) {
        fprintf(stderr, "failed to load draft model\n");
        return 1;
    }
    auto ctx_dft = llama_context_ptr(llama_init_from_model(
                model_dft.get(), common_context_params_to_llama(params_dft)));

    params.speculative.draft.ctx_tgt = ctx_tgt;
    params.speculative.draft.ctx_dft = ctx_dft.get();

    std::vector<llama_token> inp;
    if (!common_load_input_ids_json(input_ids_path, inp)) {
        return 1;
    }

    common_sampler_ptr smpl(common_sampler_init(llama_get_model(ctx_tgt), params.sampling));
    common_speculative * spec = common_speculative_init(params.speculative, 1);

    const int n_predict = params.n_predict > 0 ? params.n_predict : 400;

    fprintf(stderr, "=== batched logits repro ===\n");
    fprintf(stderr, "backend: %s\n",
#if defined(GGML_USE_CUDA)
            "CUDA"
#elif defined(GGML_USE_VULKAN)
            "Vulkan"
#else
            "CPU/other"
#endif
    );

    if (scan_all) {
        llama_tokens prompt_tgt(inp.begin(), inp.end());
        int n_past = (int) inp.size();
        llama_batch batch = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);
        llama_batch prefill = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);

        for (size_t i = 0; i < inp.size(); ++i) {
            common_batch_add(prefill, inp[i], (llama_pos) i, { 0 }, i + 1 == inp.size());
        }
        llama_decode(ctx_tgt, prefill);
        common_speculative_process(spec, prefill);
        common_speculative_begin(spec, 0, prompt_tgt);

        llama_token id_last = common_sampler_sample(smpl.get(), ctx_tgt, -1, false);
        common_sampler_accept(smpl.get(), id_last, true);

        int n_generated = 1;
        int step        = 0;
        int first_mismatch_step = -1;

        const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx_tgt));

        while (!llama_vocab_is_eog(vocab, id_last) && n_generated < n_predict) {
            ++step;

            std::vector<uint8_t> state;
            if (!save_state(ctx_tgt, state)) {
                fprintf(stderr, "save_state failed at step %d\n", step);
                return 1;
            }

            auto spec_params = params.speculative;
            spec_params.dspark_temp = params.sampling.temp;
            spec_params.dspark_seed = params.sampling.seed;
            common_speculative_sync_params(spec, spec_params);

            llama_tokens draft;
            common_speculative_get_draft_params(spec, 0) = {
                true, -1, n_past, id_last, &prompt_tgt, &draft, nullptr,
            };
            common_speculative_draft(spec);

            step_snapshot snap {
                step, n_generated, n_past, id_last, draft,
            };

            const bool row0_match = compare_row0_logits(
                    ctx_tgt, spec, batch, state, 0, snap, false);

            // compare_row0_logits leaves KV polluted (decoded at pos_verify+..);
            // restore canonical snapshot before sequential verify consumes it.
            restore_state(ctx_tgt, state);

            if (!row0_match && first_mismatch_step < 0) {
                first_mismatch_step = step;
                fprintf(stderr, "*** first row0 mismatch at step %d gen_index %d ***\n",
                        step, n_generated);
            }

            llama_tokens ids;
            if (!common_speculative_dspark_verify_sequential(
                    spec, smpl.get(), ctx_tgt, 0, n_past, id_last, draft, ids, batch)) {
                fprintf(stderr, "sequential verify failed at step %d\n", step);
                return 1;
            }

            common_speculative_accept(spec, 0, (uint16_t) (ids.size() > 0 ? ids.size() - 1 : 0));
            n_past = n_past + (int) ids.size();
            for (auto t : ids) {
                prompt_tgt.push_back(id_last);
                id_last = t;
                ++n_generated;
                if (llama_vocab_is_eog(vocab, id_last)) {
                    break;
                }
            }

            llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, n_past, -1);
            llama_memory_seq_rm(llama_get_memory(ctx_dft.get()), 0, n_past, -1);

            if (llama_vocab_is_eog(vocab, id_last)) {
                break;
            }
        }

        llama_batch_free(batch);
        llama_batch_free(prefill);

        fprintf(stderr, "scan done: first row0 mismatch step = %d\n", first_mismatch_step);
        return first_mismatch_step < 0 ? 0 : 1;
    }

    step_snapshot snap {};
    std::vector<uint8_t> state;
    if (run_spec_until(params, ctx_tgt, ctx_dft.get(), smpl.get(), spec, inp, n_predict,
                target_step, target_gen_index, &snap, &state) != 0) {
        return 1;
    }

    fprintf(stderr, "captured step %d gen_index %d pos %d anchor %d draft_n %zu\n",
            snap.step, snap.gen_index, (int) snap.pos_verify, (int) snap.anchor, snap.draft.size());

    llama_batch batch = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);

    fprintf(stderr, "\n--- row0 logits: single-token vs multi-token (pure target decode) ---\n");
    compare_row0_logits(ctx_tgt, spec, batch, state, 0, snap, true);

    fprintf(stderr, "\n--- row0 with defer layer extract (batched verify config) ---\n");
    compare_row0_logits(ctx_tgt, spec, batch, state, 0, snap, true, true, true);

    fprintf(stderr, "\n--- per-row sequential chain vs batched ---\n");
    compare_all_rows(ctx_tgt, batch, state, 0, snap);

    fprintf(stderr, "\n--- full verify: sequential vs batched (DSpark paths) ---\n");
    setenv("DSPARK_VERIFY_BATCHED", "1", 1);

  {
        common_sampler_reset(smpl.get());
        restore_state(ctx_tgt, state);

        llama_tokens out_seq;
        if (!common_speculative_dspark_verify_sequential(
                spec, smpl.get(), ctx_tgt, 0, snap.pos_verify, snap.anchor,
                snap.draft, out_seq, batch)) {
            fprintf(stderr, "sequential verify failed on snapshot\n");
            return 1;
        }

        fprintf(stderr, "sequential out (%zu):", out_seq.size());
        for (auto t : out_seq) {
            fprintf(stderr, " %d", (int) t);
        }
        fputc('\n', stderr);

        common_sampler_reset(smpl.get());
        restore_state(ctx_tgt, state);

        llama_tokens out_batched;
        if (!common_speculative_dspark_verify_batched(
                spec, smpl.get(), ctx_tgt, 0, snap.pos_verify, snap.anchor,
                snap.draft, out_batched, batch, nullptr, nullptr, params.sampling.temp)) {
            fprintf(stderr, "batched verify failed on snapshot\n");
            return 1;
        }

        fprintf(stderr, "batched   out (%zu):", out_batched.size());
        for (auto t : out_batched) {
            fprintf(stderr, " %d", (int) t);
        }
        fputc('\n', stderr);

        const bool verify_match = out_seq == out_batched;
        fprintf(stderr, "verify outputs: %s\n", verify_match ? "MATCH" : "MISMATCH");
        if (!verify_match) {
            const size_t n = std::min(out_seq.size(), out_batched.size());
            for (size_t i = 0; i < n; ++i) {
                if (out_seq[i] != out_batched[i]) {
                    fprintf(stderr, "  first token diff at accept index %zu: seq=%d batched=%d\n",
                            i, (int) out_seq[i], (int) out_batched[i]);
                    break;
                }
            }
        }
    }

    llama_batch_free(batch);
    return 0;
}
