#!/usr/bin/env bash
# Download Qwen3-8B target (Q4 GGUF) and HF weights for dspark_qwen3_8b_block7 conversion.
# Does not run conversion; see docs/dspark-qwen3-poc.md for convert commands.
set -euo pipefail

MODELS="${MODELS_DIR:-$HOME/models}"
mkdir -p "$MODELS"

if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "huggingface-cli not found; install huggingface_hub[cli]" >&2
    exit 1
fi

echo "==> Target GGUF (Q4_K_M): Qwen/Qwen3-8B-GGUF"
huggingface-cli download Qwen/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf --local-dir "$MODELS/Qwen3-8B-GGUF"

echo "==> Target tokenizer metadata: Qwen/Qwen3-8B"
huggingface-cli download Qwen/Qwen3-8B \
    --include "*.json" --include "tokenizer*" --include "*.txt" --include "LICENSE*" \
    --local-dir "$MODELS/Qwen3-8B"

echo "==> DSpark draft HF weights: deepseek-ai/dspark_qwen3_8b_block7"
huggingface-cli download deepseek-ai/dspark_qwen3_8b_block7 \
    --local-dir "$MODELS/dspark_qwen3_8b_block7"

echo "Done. Files under $MODELS"
echo "  Target GGUF: $MODELS/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"
echo "  Draft HF:    $MODELS/dspark_qwen3_8b_block7"
echo "  Tokenizer:   $MODELS/Qwen3-8B"
