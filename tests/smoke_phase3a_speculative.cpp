#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

static void usage(const char * argv0) {
    fprintf(stderr,
            "Usage: %s -m TARGET.gguf -md DRAFT.gguf "
            "--input-ids PATH --reference PATH "
            "[--temp 0] [--seed 42] [-n 32] [--disable-markov]\n",
            argv0);
}

static bool load_reference_output(const std::string & path, std::vector<llama_token> & out) {
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "failed to open reference: %s\n", path.c_str());
        return false;
    }

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) {
            continue;
        }
        nlohmann::json row = nlohmann::json::parse(line, nullptr, false);
        if (row.is_discarded()) {
            continue;
        }
        if (row.contains("final_output_token_ids")) {
            out.clear();
            for (const auto & el : row["final_output_token_ids"]) {
                out.push_back((llama_token) el.get<int64_t>());
            }
            return !out.empty();
        }
    }

    fprintf(stderr, "reference JSONL missing final_output_token_ids metadata row\n");
    return false;
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();

    std::string input_ids_path;
    std::string reference_path;
    bool disable_markov = false;

    std::vector<char *> fargv;
    fargv.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--input-ids") == 0 && i + 1 < argc) {
            input_ids_path = argv[++i];
        } else if (strcmp(argv[i], "--reference") == 0 && i + 1 < argc) {
            reference_path = argv[++i];
        } else if (strcmp(argv[i], "--disable-markov") == 0) {
            disable_markov = true;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fargv.push_back(argv[i]);
        }
    }

    const int fargc = (int) fargv.size();
    if (!common_params_parse(fargc, fargv.data(), params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.model.path.empty() || params.speculative.draft.mparams.path.empty()
            || input_ids_path.empty() || reference_path.empty()) {
        usage(argv[0]);
        return 1;
    }

    params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK };
    params.speculative.draft.n_max = 7;
    params.speculative.dspark_disable_markov = disable_markov;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init_tgt = common_init_from_params(params);
    llama_context * ctx_tgt = llama_init_tgt->context();
    const llama_vocab * vocab = llama_model_get_vocab(llama_init_tgt->model());

    llama_model_ptr model_dft;
    llama_context_ptr ctx_dft;
    {
        auto params_dft = params;
        params_dft.model = params.speculative.draft.mparams;
        auto mparams_dft = common_model_params_to_llama(params_dft);
        model_dft.reset(llama_model_load_from_file(params_dft.model.path.c_str(), mparams_dft));
        if (!model_dft) {
            fprintf(stderr, "failed to load draft model\n");
            return 1;
        }
        auto cparams = common_context_params_to_llama(params_dft);
        ctx_dft.reset(llama_init_from_model(model_dft.get(), cparams));
        params.speculative.draft.ctx_tgt = ctx_tgt;
        params.speculative.draft.ctx_dft = ctx_dft.get();
    }

    std::vector<llama_token> inp;
    if (!common_load_input_ids_json(input_ids_path, inp)) {
        return 1;
    }

    if ((uint32_t) inp.size() > llama_n_ctx(ctx_tgt)) {
        fprintf(stderr, "input exceeds context size\n");
        return 1;
    }

    std::vector<llama_token> ref_out;
    if (!load_reference_output(reference_path, ref_out)) {
        return 1;
    }

    common_sampler_ptr smpl(common_sampler_init(llama_get_model(ctx_tgt), params.sampling));
    common_speculative * spec = common_speculative_init(params.speculative, 1);

    llama_token id_last = inp.back();
    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    int n_past = (int) inp.size() - 1;

    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);

    // common_speculative_process() reads pos/seq_id/n_seq_id directly, so the prefill
    // batch must be fully formed (llama_batch_get_one leaves those arrays null).
    llama_batch prefill = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);
    for (size_t i = 0; i + 1 < inp.size(); ++i) {
        common_batch_add(prefill, inp[i], (llama_pos) i, { 0 }, false);
    }
    llama_decode(ctx_tgt, prefill);
    common_speculative_process(spec, prefill);
    llama_decode(ctx_dft.get(), prefill);

    common_speculative_begin(spec, 0, prompt_tgt);

    llama_tokens output = inp;
    llama_tokens draft;
    std::vector<std::vector<float>> draft_probs;
    int n_predict = 0;
    const llama_seq_id seq_id = 0;

    while (params.n_predict < 0 || n_predict < params.n_predict) {
        if (draft.empty()) {
            auto spec_params = params.speculative;
            spec_params.dspark_temp = params.sampling.temp;
            spec_params.dspark_seed = params.sampling.seed;
            common_speculative_sync_params(spec, spec_params);

            draft_probs.clear();
            common_speculative_get_draft_params(spec, seq_id) = {
                /* .drafting    = */ true,
                /* .n_max       = */ -1,
                /* .n_past      = */ n_past,
                /* .id_last     = */ id_last,
                /* .prompt      = */ &prompt_tgt,
                /* .result      = */ &draft,
                /* .draft_probs = */ params.sampling.temp > 1e-5f ? &draft_probs : nullptr,
            };
            common_speculative_draft(spec);

            if (draft.empty()) {
                break;
            }
        }

        common_batch_clear(batch_tgt);
        common_batch_add(batch_tgt, id_last, n_past++, { seq_id }, true);
        for (size_t i = 0; i < draft.size(); ++i) {
            common_batch_add(batch_tgt, draft[i], n_past + (llama_pos) i, { seq_id }, true);
        }
        llama_decode(ctx_tgt, batch_tgt);
        common_speculative_process(spec, batch_tgt);

        std::vector<llama_token> ids;
        if (!draft_probs.empty()) {
            std::vector<int> idxs(draft.size() + 1);
            for (size_t i = 0; i < idxs.size(); ++i) {
                idxs[i] = (int) i;
            }
            ids = common_sampler_sample_and_accept_n_dspark(
                    smpl.get(), ctx_tgt, idxs, draft, draft_probs, params.sampling.temp);
        } else {
            ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);
        }

        common_speculative_accept(spec, seq_id, (uint16_t) (ids.size() - 1));

        n_past += (int) ids.size() - 1;
        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);
            id_last = ids[i];
            output.push_back(id_last);
            n_predict++;
            if (llama_vocab_is_eog(vocab, id_last)) {
                break;
            }
        }

        draft.clear();
        draft_probs.clear();
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, n_past, -1);
        llama_memory_seq_rm(llama_get_memory(ctx_dft.get()), seq_id, n_past, -1);

        if (llama_vocab_is_eog(vocab, id_last)) {
            break;
        }
    }

    llama_batch_free(batch_tgt);
    llama_batch_free(prefill);
    common_speculative_free(spec);

    if (disable_markov) {
        if (output == ref_out) {
            fprintf(stderr, "FAIL: disable-markov run still matched reference (Markov path not exercised)\n");
            return 1;
        }
        fprintf(stderr, "OK: disable-markov run diverged from reference as expected\n");
        llama_backend_free();
        return 0;
    }

    if (output != ref_out) {
        fprintf(stderr, "FAIL: output token mismatch (got %zu, expected %zu)\n", output.size(), ref_out.size());
        const size_t n = std::min(output.size(), ref_out.size());
        for (size_t i = 0; i < n; ++i) {
            if (output[i] != ref_out[i]) {
                fprintf(stderr, "  first mismatch at %zu: got %d expected %d\n",
                        i, (int) output[i], (int) ref_out[i]);
                break;
            }
        }
        llama_backend_free();
        return 1;
    }

    fprintf(stderr, "OK: output matches Phase 0 reference (%zu tokens)\n", output.size());
    llama_backend_free();
    return 0;
}
