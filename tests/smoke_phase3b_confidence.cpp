#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Reuses the Phase 3a loop; checks confidence truncation ordering only.

static void reset_dspark_loop(
        llama_context * ctx_tgt,
        llama_context * ctx_dft,
        common_speculative * spec,
        common_sampler * smpl,
        const std::vector<llama_token> & inp,
        llama_tokens & prompt_tgt,
        llama_token & id_last,
        int & n_past) {
    const llama_seq_id seq_id = 0;

    llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, 0, -1);
    llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, 0, -1);

    prompt_tgt.assign(inp.begin(), inp.end() - 1);
    id_last = inp.back();
    n_past = (int) inp.size() - 1;

    common_sampler_reset(smpl);

    // common_speculative_process() reads pos/seq_id/n_seq_id directly, so the prefill
    // batch must be fully formed (llama_batch_get_one leaves those arrays null).
    llama_batch prefill = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);
    for (size_t i = 0; i + 1 < inp.size(); ++i) {
        common_batch_add(prefill, inp[i], (llama_pos) i, { seq_id }, false);
    }
    llama_decode(ctx_tgt, prefill);
    common_speculative_process(spec, prefill);
    llama_decode(ctx_dft, prefill);
    common_speculative_begin(spec, seq_id, prompt_tgt);
    llama_batch_free(prefill);
}

static int run_confidence_smoke(
        common_params & params,
        llama_context * ctx_tgt,
        llama_context * ctx_dft,
        common_speculative * spec,
        common_sampler * smpl,
        llama_batch & batch_tgt,
        const std::vector<llama_token> & inp,
        float confidence_threshold,
        double * mean_proposal_out) {
    llama_tokens prompt_tgt;
    llama_token id_last = 0;
    int n_past = 0;

    params.speculative.dspark_confidence_threshold = confidence_threshold;
    reset_dspark_loop(ctx_tgt, ctx_dft, spec, smpl, inp, prompt_tgt, id_last, n_past);

    llama_tokens draft;
    int steps = 0;
    double sum_proposal = 0.0;
    bool saw_empty = false;

    for (int n = 0; n < params.n_predict; ++n) {
        auto spec_params = params.speculative;
        spec_params.dspark_temp = params.sampling.temp;
        spec_params.dspark_seed = params.sampling.seed;
        common_speculative_sync_params(spec, spec_params);

        draft.clear();
        common_speculative_get_draft_params(spec, 0) = {
            true, -1, n_past, id_last, &prompt_tgt, &draft, nullptr,
        };
        common_speculative_draft(spec);

        sum_proposal += draft.size();
        steps++;
        if (draft.empty()) {
            saw_empty = true;
            break;
        }

        common_batch_clear(batch_tgt);
        common_batch_add(batch_tgt, id_last, n_past++, { 0 }, true);
        for (size_t i = 0; i < draft.size(); ++i) {
            common_batch_add(batch_tgt, draft[i], n_past + (llama_pos) i, { 0 }, true);
        }
        llama_decode(ctx_tgt, batch_tgt);
        common_speculative_process(spec, batch_tgt);

        auto ids = common_sampler_sample_and_accept_n(smpl, ctx_tgt, draft);
        common_speculative_accept(spec, 0, (uint16_t) (ids.size() - 1));
        n_past += (int) ids.size() - 1;
        for (auto t : ids) {
            prompt_tgt.push_back(id_last);
            id_last = t;
        }
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, n_past, -1);
        llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, n_past, -1);
    }

    *mean_proposal_out = steps > 0 ? sum_proposal / steps : 0.0;

    if (confidence_threshold > 0.0f && !saw_empty) {
        return 1;
    }
    return 0;
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

    if (params.model.path.empty() || params.speculative.draft.mparams.path.empty() || input_ids_path.empty()) {
        fprintf(stderr, "Usage: smoke_phase3b_confidence -m TARGET -md DRAFT --input-ids F [-n 64]\n");
        return 1;
    }

    params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK };
    params.speculative.draft.n_max = 7;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init_tgt = common_init_from_params(params);
    llama_context * ctx_tgt = llama_init_tgt->context();

    llama_model_ptr model_dft;
    llama_context_ptr ctx_dft;
    {
        auto params_dft = params;
        params_dft.model = params.speculative.draft.mparams;
        model_dft.reset(llama_model_load_from_file(params_dft.model.path.c_str(),
                    common_model_params_to_llama(params_dft)));
        if (!model_dft) {
            fprintf(stderr, "failed to load draft model\n");
            return 1;
        }
        ctx_dft.reset(llama_init_from_model(model_dft.get(), common_context_params_to_llama(params_dft)));
        params.speculative.draft.ctx_tgt = ctx_tgt;
        params.speculative.draft.ctx_dft = ctx_dft.get();
    }

    std::vector<llama_token> inp;
    if (!common_load_input_ids_json(input_ids_path, inp)) {
        return 1;
    }

    common_sampler_ptr smpl(common_sampler_init(llama_get_model(ctx_tgt), params.sampling));
    common_speculative * spec = common_speculative_init(params.speculative, 1);

    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);

    double mean0 = 0;
    double mean9 = 0;
    if (run_confidence_smoke(params, ctx_tgt, ctx_dft.get(), spec, smpl.get(), batch_tgt, inp,
                             0.0f, &mean0) != 0) {
        llama_batch_free(batch_tgt);
        common_speculative_free(spec);
        llama_backend_free();
        return 1;
    }
    if (run_confidence_smoke(params, ctx_tgt, ctx_dft.get(), spec, smpl.get(), batch_tgt, inp,
                             0.9f, &mean9) != 0) {
        fprintf(stderr, "FAIL: threshold=0.9 never produced empty proposal\n");
        llama_batch_free(batch_tgt);
        common_speculative_free(spec);
        llama_backend_free();
        return 1;
    }

    llama_batch_free(batch_tgt);
    common_speculative_free(spec);

    if (!(mean9 < mean0)) {
        fprintf(stderr, "FAIL: mean proposal len at 0.9 (%.3f) >= at 0.0 (%.3f)\n", mean9, mean0);
        llama_backend_free();
        return 1;
    }

    fprintf(stderr, "OK: mean proposal len 0.0=%.3f 0.9=%.3f\n", mean0, mean9);
    llama_backend_free();
    return 0;
}
