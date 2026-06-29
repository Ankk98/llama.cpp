#include "common.h"
#include "llama.h"

#include "../src/llama-ext.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static void usage(const char * argv0) {
    fprintf(stderr,
            "Usage: %s --draft-model PATH [--ref-bin PATH]\n",
            argv0);
}

static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    double mse_ab = 0.0;
    double mse_a0 = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = (double) a[i] - (double) b[i];
        mse_ab += d * d;
        mse_a0 += (double) a[i] * (double) a[i];
    }
    return mse_a0 > 0.0 ? mse_ab / mse_a0 : mse_ab;
}

static bool has_nan_inf(const float * data, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(data[i])) {
            return true;
        }
    }
    return false;
}

static void fill_features(float * dst, size_t n, int seed) {
    // Deterministic fill shared with tests/gen_dspark_graph_ref.py (not RNG-based).
    for (size_t i = 0; i < n; ++i) {
        dst[i] = 0.01f * std::sin(0.001f * (float) (i + (size_t) seed * 7919u));
    }
}

static bool load_ref_logits(const std::string & path, std::vector<float> & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "failed to open reference blob: %s\n", path.c_str());
        return false;
    }
    int32_t n = 0;
    f.read(reinterpret_cast<char *>(&n), sizeof(n));
    if (n <= 0) {
        fprintf(stderr, "invalid reference blob count: %d\n", (int) n);
        return false;
    }
    out.resize((size_t) n);
    f.read(reinterpret_cast<char *>(out.data()), (size_t) n * sizeof(float));
    return f.good();
}

static bool run_encode(llama_context * ctx, const float * features, int32_t n_tokens, int32_t n_embd_enc) {
    llama_batch batch = llama_batch_init(n_tokens, n_embd_enc, 1);
    batch.n_tokens = n_tokens;
    std::memcpy(batch.embd, features, (size_t) n_tokens * n_embd_enc * sizeof(float));
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = false;
    }
    const int rc = llama_encode(ctx, batch);
    llama_batch_free(batch);
    return rc == 0;
}

static bool run_inject(llama_context * ctx, const float * fused, int32_t n_tokens, int32_t n_embd, llama_pos pos0) {
    llama_batch batch = llama_batch_init(n_tokens, n_embd, 1);
    batch.n_tokens = n_tokens;
    std::memcpy(batch.embd, fused, (size_t) n_tokens * n_embd * sizeof(float));
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.pos[i]       = pos0 + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = false;
    }
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return rc == 0;
}

static bool run_noise_decode(
        llama_context * ctx,
        llama_token anchor,
        llama_token mask_id,
        int32_t block_size,
        llama_pos pos0,
        std::vector<float> & logits_out) {
    llama_batch batch = llama_batch_init(block_size, 0, 1);
    batch.n_tokens = block_size;
    for (int32_t i = 0; i < block_size; ++i) {
        batch.token[i]     = i == 0 ? anchor : mask_id;
        batch.pos[i]       = pos0 + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = true;
    }
    const int rc = llama_decode(ctx, batch);
    if (rc != 0) {
        llama_batch_free(batch);
        return false;
    }

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    logits_out.resize((size_t) n_vocab);
    const float * row0 = llama_get_logits_ith(ctx, 0);
    if (!row0) {
        llama_batch_free(batch);
        return false;
    }
    std::memcpy(logits_out.data(), row0, (size_t) n_vocab * sizeof(float));
    llama_batch_free(batch);
    return true;
}

int main(int argc, char ** argv) {
    std::string draft_model = "/tmp/dspark_gemma4_12b_smoke.gguf";
    std::string ref_bin     = "tests/data/dspark_gemma4_decoder_logits_ref.bin";

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--draft-model") == 0 && i + 1 < argc) {
            draft_model = argv[++i];
        } else if (strcmp(argv[i], "--ref-bin") == 0 && i + 1 < argc) {
            ref_bin = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(draft_model.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "failed to load draft model: %s\n", draft_model.c_str());
        return 1;
    }

    char mbuf[64] = {};
    if (llama_model_meta_val_str(model, "general.architecture", mbuf, sizeof(mbuf)) < 0 ||
            strcmp(mbuf, "dspark") != 0) {
        fprintf(stderr, "expected general.architecture=dspark, got '%s'\n", mbuf);
        return 1;
    }

    mbuf[0] = '\0';
    if (llama_model_meta_val_str(model, "dspark.attention.causal", mbuf, sizeof(mbuf)) >= 0) {
        if (!(strcmp(mbuf, "false") == 0 || strcmp(mbuf, "0") == 0)) {
            fprintf(stderr, "expected dspark.attention.causal=false, got '%s'\n", mbuf);
            return 1;
        }
    }

    int32_t block_size = 7;
    mbuf[0] = '\0';
    if (llama_model_meta_val_str(model, "dspark.block_size", mbuf, sizeof(mbuf)) >= 0) {
        block_size = std::atoi(mbuf);
    }

    const uint32_t n_target_layers = llama_model_target_layer_ids_n(model);
    const int32_t  n_embd          = llama_model_n_embd(model);
    const int32_t  n_embd_enc      = (int32_t) n_target_layers * n_embd;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 512;
    cparams.n_batch         = 512;
    cparams.n_ubatch        = 512;
    cparams.no_perf         = true;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }

    llama_set_causal_attn(ctx, false);
    llama_set_embeddings_nextn(ctx, true, false);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const llama_token mask_id = llama_vocab_mask(vocab);
    const llama_token anchor  = 100;

    std::vector<float> raw_ctx((size_t) 3 * n_embd_enc);
    std::vector<float> raw_inc((size_t) n_embd_enc);
    fill_features(raw_ctx.data(), raw_ctx.size(), 42);
    fill_features(raw_inc.data(), raw_inc.size(), 43);

    if (!run_encode(ctx, raw_ctx.data(), 3, n_embd_enc)) {
        fprintf(stderr, "encoder forward failed\n");
        return 1;
    }

    const float * fused = llama_get_embeddings_nextn(ctx);
    if (!fused || has_nan_inf(fused, (size_t) 3 * n_embd)) {
        fprintf(stderr, "encoder output invalid\n");
        return 1;
    }
    if (!run_inject(ctx, fused, 3, n_embd, 0)) {
        fprintf(stderr, "context inject failed\n");
        return 1;
    }

    const llama_pos cache_after_inject = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
    if (cache_after_inject < 2) {
        fprintf(stderr, "unexpected cache length after inject: %d\n", (int) cache_after_inject);
        return 1;
    }

    std::vector<float> logits1;
    if (!run_noise_decode(ctx, anchor, mask_id, block_size, 3, logits1)) {
        fprintf(stderr, "noise decode (iter 1) failed\n");
        return 1;
    }
    if (has_nan_inf(logits1.data(), logits1.size())) {
        fprintf(stderr, "iter 1 logits contain NaN/Inf\n");
        return 1;
    }

    llama_memory_seq_rm(llama_get_memory(ctx), 0, 3, -1);

    if (!run_encode(ctx, raw_inc.data(), 1, n_embd_enc)) {
        fprintf(stderr, "encoder forward (increment) failed\n");
        return 1;
    }
    fused = llama_get_embeddings_nextn(ctx);
    if (!run_inject(ctx, fused, 1, n_embd, 3)) {
        fprintf(stderr, "context inject (increment) failed\n");
        return 1;
    }

    std::vector<float> logits2;
    if (!run_noise_decode(ctx, anchor, mask_id, block_size, 4, logits2)) {
        fprintf(stderr, "noise decode (iter 2) failed\n");
        return 1;
    }

    double diff_norm = 0.0;
    for (size_t i = 0; i < std::min(logits1.size(), logits2.size()); ++i) {
        const double d = (double) logits1[i] - (double) logits2[i];
        diff_norm += d * d;
    }
    if (diff_norm == 0.0) {
        fprintf(stderr, "iter 2 logits identical to iter 1 (cache path likely broken)\n");
        return 1;
    }

    std::vector<float> ref;
    if (!load_ref_logits(ref_bin, ref)) {
        fprintf(stderr, "reference blob missing; run tests/gen_dspark_graph_ref.py first\n");
        return 1;
    }

    std::vector<float> got(ref.size());
    std::memcpy(got.data(), logits1.data(), ref.size() * sizeof(float));

    const double err = nmse(ref, got);
    if (err >= 0.01) {
        fprintf(stderr, "logits NMSE %.6f >= 0.01\n", err);
        return 1;
    }

    const float softcap = 30.0f;
    bool softcap_diff = false;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float capped = tanhf(got[i] / softcap) * softcap;
        if (std::fabs(capped - got[i]) > 1e-3f) {
            softcap_diff = true;
            break;
        }
    }
    if (!softcap_diff) {
        fprintf(stderr, "softcap sanity check failed (graph logits should be uncapped)\n");
        return 1;
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    printf("smoke_phase2_graph: all checks passed (NMSE=%.6f)\n", err);
    return 0;
}
