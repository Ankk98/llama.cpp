#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct run_stats {
    std::vector<llama_token> output;
    double pp_ms = 0;
    double gen_ms = 0;
    double draft_ms = 0;  // time inside common_speculative_draft (draft fwd + cpu sampling)
    double verify_ms = 0; // time inside target decode + accept + process
    double tgt_decode_ms = 0;
    double verify_accept_ms = 0;
    double verify_features_ms = 0;
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
            "[--temp 0] [--seed 42] [-n 32] [-c 512] [-ngl N] [-ngld N]\n",
            argv0);
}

static double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
            clock::now().time_since_epoch()).count();
}

static int run_vanilla(
        common_params & params,
        llama_context * ctx_tgt,
        common_sampler * smpl,
        std::vector<llama_token> & inp,
        int n_predict,
        run_stats * out) {
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx_tgt));

    llama_token id_last = inp.back();
    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    int n_past = (int) inp.size() - 1;

    llama_batch batch = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);
    llama_batch prefill = llama_batch_get_one(inp.data(), (int) inp.size() - 1);
    out->n_prompt = (int) inp.size() - 1;
    const double tpp = now_ms();
    llama_decode(ctx_tgt, prefill);
    out->pp_ms = now_ms() - tpp;

    out->output = inp;
    const double t0 = now_ms();

    for (int n = 0; n < n_predict; ++n) {
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

    // warmup target graphs (especially after ctx_tgt_feat creation / first split-verify step)
    {
        llama_batch warm = llama_batch_init(1, 0, 1);
        common_batch_add(warm, inp.empty() ? 0 : inp[0], 0, { 0 }, true);
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
    llama_decode(ctx_prefill, prefill);
    if (ctx_feat) {
        llama_decode(ctx_feat, prefill);
    }
    common_speculative_process(spec, prefill);
    common_speculative_begin(spec, 0, prompt_tgt);

    llama_token id_last = common_sampler_sample(smpl, ctx_tgt, -1, false);
    common_sampler_accept(smpl, id_last, true);
    out->pp_ms = now_ms() - tpp;

    out->output = inp;
    out->output.push_back(id_last);
    out->n_generated++;
    llama_tokens draft;
    const double t0 = now_ms();

    while (!llama_vocab_is_eog(vocab, id_last) && out->n_generated < n_predict) {
        auto spec_params = params.speculative;
        spec_params.dspark_temp = params.sampling.temp;
        spec_params.dspark_seed = params.sampling.seed;
        common_speculative_sync_params(spec, spec_params);

        draft.clear();
        common_speculative_get_draft_params(spec, 0) = {
            true, -1, n_past, id_last, &prompt_tgt, &draft, nullptr,
        };
        const double td0 = now_ms();
        common_speculative_draft(spec);
        out->draft_ms += now_ms() - td0;

        out->n_propose_steps++;
        out->n_drafted += (int) draft.size();

        if (draft.empty()) {
            break;
        }

        const llama_pos pos_verify = n_past;
        llama_tokens ids;
        if (getenv("DSPARK_VERIFY_SEQ")) {
            const double tv0 = now_ms();
            if (!common_speculative_dspark_verify_sequential(
                    spec, smpl, ctx_tgt, 0, pos_verify, id_last, draft, ids, batch_tgt)) {
                fprintf(stderr, "error: dspark sequential verify failed\n");
                return 1;
            }
            out->tgt_decode_ms += now_ms() - tv0;
            out->verify_ms += now_ms() - tv0;
        } else {
            common_speculative_dspark_verify_timing vtim {};
            if (!common_speculative_dspark_verify_batched(
                    spec, smpl, ctx_tgt, 0, pos_verify, id_last, draft, ids, batch_tgt, &vtim)) {
                fprintf(stderr, "error: dspark batched verify failed\n");
                return 1;
            }
            out->tgt_decode_ms     += vtim.logits_decode_ms + vtim.features_decode_ms;
            out->verify_accept_ms  += vtim.accept_ms;
            out->verify_features_ms += vtim.features_decode_ms;
            out->verify_process_ms += vtim.process_ms;
            out->verify_ms         += vtim.logits_decode_ms + vtim.accept_ms + vtim.process_ms;
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

        common_speculative_accept(spec, 0, (uint16_t) n_acc);
        n_past = pos_verify + (int) ids.size();
        for (auto t : ids) {
            prompt_tgt.push_back(id_last);
            id_last = t;
            out->output.push_back(id_last);
            out->n_generated++;
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

    std::vector<char *> fargv;
    fargv.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--input-ids") == 0 && i + 1 < argc) {
            input_ids_path = argv[++i];
        } else {
            fargv.push_back(argv[i]);
        }
    }

    if (!common_params_parse((int) fargv.size(), fargv.data(), params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.model.path.empty() || input_ids_path.empty()) {
        usage(argv[0]);
        return 1;
    }

    // n_ctx=0 loads the model's trained context (262144 for Gemma4/DSpark) and allocates
    // a full KV cache for both target and draft, which OOMs on typical GPUs.
    if (params.n_ctx == 0) {
        params.n_ctx = 512;
        fprintf(stderr, "note: defaulting context to %d (pass -c to override)\n", params.n_ctx);
    }

    const bool has_draft = !params.speculative.draft.mparams.path.empty();
    if (has_draft) {
        params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK };
        params.speculative.draft.n_max = 5;
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
        if (!getenv("DSPARK_NO_SPLIT_VERIFY") && getenv("DSPARK_SPLIT_VERIFY")) {
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

    common_sampler_ptr smpl(common_sampler_init(llama_get_model(ctx_tgt), params.sampling));

    const int n_predict = params.n_predict > 0 ? params.n_predict : 32;
    const size_t n_inp = inp.size();

    run_stats vanilla {};
    run_stats spec_stats {};

    if (has_draft) {
        if (run_speculative(params, ctx_tgt, ctx_dft.get(), smpl.get(), spec, inp, n_predict, &spec_stats) != 0) {
            return 1;
        }
    }

    common_sampler_reset(smpl.get());

    if (run_vanilla(params, ctx_tgt, smpl.get(), inp, n_predict, &vanilla) != 0) {
        return 1;
    }

    fprintf(stderr, "\n=== Vanilla (target only) ===\n");
    fprintf(stderr, "prompt: %d tokens, pp %.1f ms (%.2f tok/s)\n",
            vanilla.n_prompt, vanilla.pp_ms,
            vanilla.pp_ms > 0 ? 1000.0 * vanilla.n_prompt / vanilla.pp_ms : 0.0);
    fprintf(stderr, "generated: %d tokens in %.1f ms (tgp %.2f tok/s)\n",
            vanilla.n_generated, vanilla.gen_ms,
            vanilla.n_generated > 0 ? 1000.0 * vanilla.n_generated / vanilla.gen_ms : 0.0);
    print_tokens("output", vanilla.output, n_inp);
    print_text(ctx_tgt, "output", vanilla.output, n_inp);

    if (has_draft) {
        fprintf(stderr, "\n=== Speculative (draft-dspark) ===\n");
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
            const double v_fwd = vanilla.n_generated > 0 ? vanilla.gen_ms / vanilla.n_generated : 0.0;
            const double d_step = spec_stats.draft_ms  / spec_stats.n_propose_steps;
            const double vf_step = spec_stats.verify_ms / spec_stats.n_propose_steps;
            const double tok_step = (double) spec_stats.n_generated / spec_stats.n_propose_steps;
            fprintf(stderr, "--- per-pass timing ---\n");
            fprintf(stderr, "  target forward (vanilla, 1 tok)   : %.2f ms\n", v_fwd);
            fprintf(stderr, "  draft  step (fwd + cpu sampling)  : %.2f ms  (proposes %d)\n", d_step, (int) spec_stats.n_drafted / std::max(1, spec_stats.n_propose_steps));
            fprintf(stderr, "  verify step (batched)            : %.2f ms\n", vf_step);
            fprintf(stderr, "    logits decode (full block)     : %.2f ms\n",
                    (spec_stats.tgt_decode_ms - spec_stats.verify_features_ms) / spec_stats.n_propose_steps);
            fprintf(stderr, "    accept (sampling)              : %.2f ms\n",
                    spec_stats.verify_accept_ms / spec_stats.n_propose_steps);
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
        // the two loops may stop at slightly different lengths, so compare the common prefix
        const size_t n_cmp = std::min(vanilla.output.size(), spec_stats.output.size());
        size_t first_mismatch = n_cmp;
        for (size_t i = n_inp; i < n_cmp; ++i) {
            if (vanilla.output[i] != spec_stats.output[i]) {
                first_mismatch = i;
                break;
            }
        }
        const bool prefix_match = first_mismatch == n_cmp;
        fprintf(stderr, "token match on common prefix (temp=%.2f): %s\n",
                params.sampling.temp, prefix_match ? "YES" : "NO");
        if (!prefix_match) {
            fprintf(stderr, "first mismatch at gen index %zu: vanilla=%d spec=%d\n",
                    first_mismatch - n_inp,
                    (int) vanilla.output[first_mismatch], (int) spec_stats.output[first_mismatch]);
        }
        // tokens/s already accounts for token count, so this is a fair throughput ratio
        const double tgp_v = vanilla.gen_ms    > 0 ? vanilla.n_generated    / vanilla.gen_ms    : 0.0;
        const double tgp_s = spec_stats.gen_ms > 0 ? spec_stats.n_generated / spec_stats.gen_ms : 0.0;
        if (tgp_v > 0 && tgp_s > 0) {
            fprintf(stderr, "tgp speedup: %.2fx\n", tgp_s / tgp_v);
        }
    }

    if (spec) {
        common_speculative_free(spec);
    }
    llama_backend_free();
    return 0;
}
