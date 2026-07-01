#include "dspark_pipeline.h"

#include "speculative.h"

#include "../src/llama-ext.h"

#include <chrono>
#include <cstdlib>

static double dspark_now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
            clock::now().time_since_epoch()).count();
}

bool dspark_pipeline_init(
        dspark_pipeline_state * st,
        common_speculative * spec,
        common_sampler * smpl,
        llama_context * ctx_tgt,
        llama_context * ctx_dft,
        llama_context * ctx_tgt_feat,
        const common_params_speculative & spec_params,
        const dspark_pipeline_config & cfg) {
    if (st == nullptr || ctx_tgt == nullptr) {
        return false;
    }

    st->mem.ctx_tgt      = ctx_tgt;
    st->mem.ctx_dft      = ctx_dft;
    st->mem.ctx_tgt_feat = ctx_tgt_feat;
    dspark_memory_bundle_init(&st->mem, ctx_tgt, ctx_dft, ctx_tgt_feat, 0);

    st->spec        = spec;
    st->smpl        = smpl;
    st->spec_params = spec_params;
    st->cfg         = cfg;
    st->batch_tgt   = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);
    st->batch_owned = true;

    return true;
}

void dspark_pipeline_free(dspark_pipeline_state * st) {
    if (st == nullptr || !st->batch_owned) {
        return;
    }
    llama_batch_free(st->batch_tgt);
    st->batch_owned = false;
}

bool dspark_pipeline_prefill(
        dspark_pipeline_state * st,
        const llama_tokens & prompt,
        llama_token * out_first_gen,
        double * out_pp_ms) {
    if (st == nullptr || st->mem.ctx_tgt == nullptr || st->smpl == nullptr || out_first_gen == nullptr) {
        return false;
    }

    st->prompt = prompt;
    st->n_past = (llama_pos) prompt.size();

    llama_context * ctx_tgt = st->mem.ctx_tgt;
    llama_context * ctx_feat = st->mem.ctx_tgt_feat;

    if (!getenv("DSPARK_NO_WARMUP")) {
        const int32_t n_verify = st->cfg.n_max + 1;
        llama_batch warm = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);
        const llama_token warm_tok = prompt.empty() ? 0 : prompt[0];
        for (int32_t i = 0; i < n_verify; ++i) {
            common_batch_add(warm, warm_tok, (llama_pos) i, { st->mem.seq_main }, true);
        }
        llama_decode(ctx_tgt, warm);
        if (ctx_feat) {
            llama_decode(ctx_feat, warm);
        }
        llama_synchronize(ctx_tgt);
        llama_batch_free(warm);
        dspark_target_context_reset(ctx_tgt);
        if (ctx_feat) {
            dspark_target_context_reset(ctx_feat);
        }
        llama_synchronize(ctx_tgt);
    }

    llama_batch prefill = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);
    for (size_t i = 0; i < prompt.size(); ++i) {
        common_batch_add(prefill, prompt[i], (llama_pos) i, { st->mem.seq_main }, i + 1 == prompt.size());
    }

    const double tpp = dspark_now_ms();
    const bool prefill_defer = getenv("DSPARK_PREFILL_DEFER") != nullptr;
    if (prefill_defer) {
        llama_set_defer_layer_inp_extract(ctx_tgt, true);
    }

    llama_decode(ctx_tgt, prefill);
    if (ctx_feat) {
        llama_decode(ctx_feat, prefill);
    }

    if (prefill_defer) {
        llama_commit_layer_inputs(ctx_tgt, prompt.size());
        llama_set_defer_layer_inp_extract(ctx_tgt, false);
    }

    if (!getenv("DSPARK_NO_PREFILL_PROCESS") && st->spec) {
        common_speculative_process(st->spec, prefill);
    }

    if (!getenv("DSPARK_NO_BEGIN") && st->spec) {
        common_speculative_begin(st->spec, st->mem.seq_main, st->prompt);
    }

    *out_first_gen = common_sampler_sample(st->smpl, ctx_tgt, -1, false);
    common_sampler_accept(st->smpl, *out_first_gen, true);

    if (out_pp_ms) {
        *out_pp_ms = dspark_now_ms() - tpp;
    }

    st->anchor = *out_first_gen;

    llama_batch_free(prefill);

    return true;
}

bool dspark_pipeline_step(dspark_pipeline_state * st, dspark_step_result * out) {
    if (st == nullptr || out == nullptr) {
        return false;
    }

    out->committed.clear();
    out->n_accepted_draft = 0;
    out->n_drafted        = 0;
    out->timing = {};

    dspark_draft_propose_result propose {};
    st->spec_params.dspark_temp  = st->cfg.temp;
    st->spec_params.dspark_seed  = 0;

    const double td0 = dspark_now_ms();
    if (!dspark_draft_propose(
                st->spec, st->spec_params, st->mem.seq_main,
                st->n_past, st->anchor, &st->prompt, &propose)) {
        return false;
    }
    out->timing.draft_ms = dspark_now_ms() - td0;
    out->n_drafted = propose.n_proposed;

    const llama_pos pos_verify = st->n_past;
    llama_tokens accepted;

    const double tv0 = dspark_now_ms();

    if (getenv("DSPARK_VANILLA_VERIFY")) {
        common_batch_clear(st->batch_tgt);
        common_batch_add(st->batch_tgt, st->anchor, pos_verify, { st->mem.seq_main }, true);
        llama_decode(st->mem.ctx_tgt, st->batch_tgt);
        const llama_token next = common_sampler_sample(st->smpl, st->mem.ctx_tgt, -1, false);
        common_sampler_accept(st->smpl, next, true);
        accepted.assign(1, next);
    } else if (getenv("DSPARK_VERIFY_SEQ") != nullptr
            && (int) propose.draft.size() >= st->cfg.min_verify_tokens) {
        if (!dspark_target_verify_sequential(
                    st->spec, st->smpl, &st->mem,
                    pos_verify, st->anchor, propose.draft,
                    accepted, st->batch_tgt)) {
            return false;
        }
    } else if ((int) propose.draft.size() >= st->cfg.min_verify_tokens) {
        dspark_verify_timing vtim {};

        if (!dspark_target_verify_logits(
                    &st->mem, st->spec, pos_verify, st->anchor, propose.draft,
                    st->batch_tgt, &vtim)) {
            return false;
        }

        accepted = dspark_target_accept_chain(
                st->smpl, st->mem.ctx_tgt, propose.draft, nullptr, st->cfg.temp);

        dspark_target_verify_scratch_cleanup(&st->mem, pos_verify);

        if (accepted.empty()) {
            return false;
        }

        if (!dspark_target_commit_tokens(
                    st->spec, &st->mem, pos_verify, st->anchor, accepted,
                    st->batch_tgt, &vtim)) {
            return false;
        }

        if (!dspark_draft_process_committed(
                    st->spec, &st->mem, pos_verify, st->anchor, accepted,
                    st->batch_tgt)) {
            return false;
        }

        out->timing.verify_detail = vtim;
    } else {
        if (!dspark_target_commit_one_greedy(
                    st->spec, &st->mem, st->smpl,
                    pos_verify, st->anchor, st->batch_tgt, accepted)) {
            return false;
        }

        if (!dspark_draft_process_committed(
                    st->spec, &st->mem, pos_verify, st->anchor, accepted,
                    st->batch_tgt)) {
            return false;
        }
    }

    out->timing.verify_ms = dspark_now_ms() - tv0;

    if (accepted.empty()) {
        return false;
    }

    out->n_accepted_draft = (int) accepted.size() - 1;
    out->committed = accepted;

    const int n_acc = out->n_accepted_draft;
    if (n_acc > 0 && st->spec) {
        common_speculative_accept(st->spec, st->mem.seq_main, (uint16_t) n_acc);
    }

    st->n_past = pos_verify + (llama_pos) accepted.size();
    st->anchor = accepted.back();

    dspark_memory_trim_beyond(&st->mem, st->n_past);

    return true;
}

bool dspark_pipeline_run(
        common_params & params,
        common_speculative * spec,
        common_sampler * smpl,
        llama_context * ctx_tgt,
        llama_context * ctx_dft,
        const llama_tokens & inp,
        int n_predict,
        dspark_run_stats * out) {
    if (out == nullptr) {
        return false;
    }

    dspark_pipeline_config cfg {};
    cfg.n_max             = params.speculative.draft.n_max;
    cfg.min_verify_tokens = 1;
    cfg.temp              = params.sampling.temp;
    cfg.confidence_threshold = params.speculative.dspark_confidence_threshold;

    dspark_pipeline_state st {};
    if (!dspark_pipeline_init(
                &st, spec, smpl, ctx_tgt, ctx_dft,
                params.speculative.draft.ctx_tgt_feat,
                params.speculative, cfg)) {
        return false;
    }

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx_tgt));

    llama_token first_gen = LLAMA_TOKEN_NULL;
    double pp_ms = 0;
    if (!dspark_pipeline_prefill(&st, inp, &first_gen, &pp_ms)) {
        dspark_pipeline_free(&st);
        return false;
    }

    out->n_prompt = (int) inp.size();
    out->pp_ms    = pp_ms;
    out->output   = inp;
    out->output.push_back(first_gen);
    out->n_generated = 1;

    const double t0 = dspark_now_ms();

    llama_token id_last = first_gen;

    while (!llama_vocab_is_eog(vocab, id_last) && out->n_generated < n_predict) {
        st.cfg.temp = params.sampling.temp;
        st.spec_params = params.speculative;
        st.spec_params.dspark_temp = params.sampling.temp;
        st.spec_params.dspark_seed = params.sampling.seed;

        dspark_step_result step {};
        if (!dspark_pipeline_step(&st, &step)) {
            fprintf(stderr, "error: dspark pipeline step failed\n");
            dspark_pipeline_free(&st);
            return false;
        }

        out->n_propose_steps++;
        out->n_drafted  += step.n_drafted;
        out->n_accepted += step.n_accepted_draft;
        out->draft_ms   += step.timing.draft_ms;
        out->verify_ms  += step.timing.verify_ms;

        const auto & vtim = step.timing.verify_detail;
        out->tgt_decode_ms          += vtim.logits_decode_ms + vtim.features_decode_ms;
        out->verify_accept_ms       += vtim.accept_ms;
        out->verify_layer_commit_ms += vtim.layer_commit_ms;
        out->verify_features_ms     += vtim.features_decode_ms;
        out->verify_process_ms      += vtim.process_ms;
        out->verify_decode_submit_ms += vtim.decode_submit_ms;

        if (getenv("DSPARK_DEBUG")) {
            fprintf(stderr, "\n[step %2d] anchor='%s'\n", out->n_propose_steps,
                    common_token_to_piece(ctx_tgt, id_last, true).c_str());
            fprintf(stderr, "  committed:");
            for (auto t : step.committed) {
                fprintf(stderr, " '%s'", common_token_to_piece(ctx_tgt, t, true).c_str());
            }
            fputc('\n', stderr);
        }

        for (auto t : step.committed) {
            st.prompt.push_back(id_last);
            id_last = t;
            out->output.push_back(id_last);
            out->n_generated++;
            if (out->n_generated >= n_predict || llama_vocab_is_eog(vocab, id_last)) {
                break;
            }
        }

        id_last = st.anchor;

        if (llama_vocab_is_eog(vocab, id_last)) {
            break;
        }
    }

    out->gen_ms = dspark_now_ms() - t0;

    dspark_pipeline_free(&st);
    dspark_target_context_reset(ctx_tgt);
    if (ctx_dft) {
        dspark_target_context_reset(ctx_dft);
    }

    return true;
}
