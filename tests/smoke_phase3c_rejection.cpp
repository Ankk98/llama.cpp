#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

static bool load_reference_mean_acceptance(const std::string & path, double * mean_out, int * steps_out) {
    std::ifstream f(path);
    if (!f) {
        return false;
    }

    std::string line;
    int steps = 0;
    int sum = 0;
    while (std::getline(f, line)) {
        if (line.empty()) {
            continue;
        }
        auto row = nlohmann::json::parse(line, nullptr, false);
        if (row.is_discarded() || !row.contains("accepted")) {
            continue;
        }
        sum += row["accepted"].get<int>();
        steps++;
    }

    if (steps == 0) {
        return false;
    }

    *mean_out  = (double) sum / (double) steps;
    *steps_out = steps;
    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();
    params.sampling.temp = 0.7f;

    std::string input_ids_path;
    std::string reference_path;

    std::vector<char *> fargv;
    fargv.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--input-ids") == 0 && i + 1 < argc) {
            input_ids_path = argv[++i];
        } else if (strcmp(argv[i], "--reference") == 0 && i + 1 < argc) {
            reference_path = argv[++i];
        } else {
            fargv.push_back(argv[i]);
        }
    }

    if (!common_params_parse((int) fargv.size(), fargv.data(), params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.model.path.empty() || params.speculative.draft.mparams.path.empty()
            || input_ids_path.empty() || reference_path.empty()) {
        fprintf(stderr,
                "Usage: smoke_phase3c_rejection -m TARGET -md DRAFT "
                "--input-ids F --reference F [--temp 0.7] [--seed 42] [-n 64]\n");
        return 1;
    }

    double ref_mean = 0;
    int ref_steps = 0;
    if (!load_reference_mean_acceptance(reference_path, &ref_mean, &ref_steps) || ref_steps < 20) {
        fprintf(stderr, "reference must contain >= 20 steps with accepted counts\n");
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

    llama_token id_last = inp.back();
    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    int n_past = (int) inp.size() - 1;

    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);
    llama_batch prefill = llama_batch_get_one(inp.data(), (int) inp.size() - 1);
    llama_decode(ctx_tgt, prefill);
    common_speculative_process(spec, prefill);
    llama_decode(ctx_dft.get(), prefill);
    common_speculative_begin(spec, 0, prompt_tgt);

    llama_tokens draft;
    std::vector<std::vector<float>> draft_probs;
    int steps = 0;
    int accept_sum = 0;

    for (int n = 0; n < params.n_predict; ++n) {
        auto spec_params = params.speculative;
        spec_params.dspark_temp = params.sampling.temp;
        spec_params.dspark_seed = params.sampling.seed;
        common_speculative_sync_params(spec, spec_params);

        draft.clear();
        draft_probs.clear();
        common_speculative_get_draft_params(spec, 0) = {
            true, -1, n_past, id_last, &prompt_tgt, &draft, &draft_probs,
        };
        common_speculative_draft(spec);
        if (draft.empty()) {
            break;
        }

        const llama_pos pos_verify = n_past;
        llama_tokens ids;
        if (!common_speculative_dspark_verify_batched(
                spec, smpl.get(), ctx_tgt, 0, pos_verify, id_last, draft, ids, batch_tgt,
                nullptr, &draft_probs, params.sampling.temp)) {
            return 1;
        }

        accept_sum += (int) ids.size() - 1;
        steps++;

        common_speculative_accept(spec, 0, (uint16_t) (ids.size() - 1));
        n_past = pos_verify + (int) ids.size();
        for (auto t : ids) {
            prompt_tgt.push_back(id_last);
            id_last = t;
        }
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, n_past, -1);
        llama_memory_seq_rm(llama_get_memory(ctx_dft.get()), 0, n_past, -1);
    }

    llama_batch_free(batch_tgt);
    common_speculative_free(spec);
    llama_backend_free();

    if (steps < 20) {
        fprintf(stderr, "FAIL: only %d propose steps (need >= 20)\n", steps);
        return 1;
    }

    const double mean = (double) accept_sum / (double) steps;
    const double rel_err = std::abs(mean - ref_mean) / std::max(ref_mean, 1e-6);

    if (rel_err > 0.10) {
        fprintf(stderr, "FAIL: mean acceptance %.3f vs reference %.3f (err %.1f%%)\n",
                mean, ref_mean, rel_err * 100.0);
        return 1;
    }

    fprintf(stderr, "OK: mean acceptance %.3f within 10%% of reference %.3f (%d steps)\n",
            mean, ref_mean, steps);
    return 0;
}
