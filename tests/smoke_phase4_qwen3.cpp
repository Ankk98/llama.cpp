#include "common.h"
#include "llama.h"

#include "../src/llama-ext.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void usage(const char * argv0) {
    fprintf(stderr,
            "Usage: %s --draft-model PATH\n"
            "  Phase 4 smoke: load Qwen3 DSpark GGUF and run encoder + inject + noise decode.\n",
            argv0);
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
    for (size_t i = 0; i < n; ++i) {
        dst[i] = 0.01f * std::sin(0.001f * (float) (i + (size_t) seed * 7919u));
    }
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

static bool meta_is_false(llama_model * model, const char * key) {
    char mbuf[64] = {};
    if (llama_model_meta_val_str(model, key, mbuf, sizeof(mbuf)) < 0) {
        return false;
    }
    return strcmp(mbuf, "false") == 0 || strcmp(mbuf, "0") == 0;
}

int main(int argc, char ** argv) {
    std::string draft_path;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--draft-model") == 0 && i + 1 < argc) {
            draft_path = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (draft_path.empty()) {
        usage(argv[0]);
        return 1;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(draft_path.c_str(), mparams);
    if (model == nullptr) {
        fprintf(stderr, "failed to load draft model: %s\n", draft_path.c_str());
        return 1;
    }

    char mbuf[64] = {};
    if (llama_model_meta_val_str(model, "general.architecture", mbuf, sizeof(mbuf)) < 0 ||
            strcmp(mbuf, "dspark") != 0) {
        fprintf(stderr, "expected general.architecture=dspark, got '%s'\n", mbuf);
        return 1;
    }

    if (!meta_is_false(model, "dspark.attention.causal")) {
        fprintf(stderr, "expected dspark.attention.causal=false\n");
        return 1;
    }

    if (!meta_is_false(model, "dspark.attention_k_eq_v")) {
        fprintf(stderr, "expected dspark.attention_k_eq_v=false for Qwen3\n");
        return 1;
    }

    mbuf[0] = '\0';
    float softcap = 0.0f;
    if (llama_model_meta_val_str(model, "dspark.final_logit_softcapping", mbuf, sizeof(mbuf)) >= 0) {
        softcap = std::atof(mbuf);
    }
    if (softcap > 0.0f) {
        fprintf(stderr, "expected no logit softcap for Qwen3, got %.3f\n", softcap);
        return 1;
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
    cparams.n_ctx    = 512;
    cparams.n_batch  = 512;
    cparams.n_ubatch = 512;
    cparams.no_perf  = true;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "failed to create draft context\n");
        return 1;
    }

    llama_set_causal_attn(ctx, false);
    llama_set_embeddings_nextn(ctx, true, false);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const llama_token mask_id = llama_vocab_mask(vocab);
    const llama_token anchor  = 100;

    std::vector<float> raw_ctx((size_t) 3 * n_embd_enc);
    fill_features(raw_ctx.data(), raw_ctx.size(), 7);

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

    std::vector<float> logits;
    if (!run_noise_decode(ctx, anchor, mask_id, block_size, 3, logits)) {
        fprintf(stderr, "noise decode failed\n");
        return 1;
    }

    if (has_nan_inf(logits.data(), logits.size())) {
        fprintf(stderr, "logits contain NaN/Inf\n");
        return 1;
    }

    printf("smoke_phase4_qwen3: OK (target_layers=%u, block=%d, vocab=%zu)\n",
           n_target_layers, (int) block_size, logits.size());

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
