#!/usr/bin/env bash
#
# One-shot DSpark CUDA benchmark VPS setup.
#
# From your dev machine (rsync source, then run remote setup over SSH):
#   ./scripts/setup-dspark-cuda-vps.sh
#
# On the VPS directly (source already at VPS_LLAMA_DIR):
#   ./scripts/setup-dspark-cuda-vps.sh --on-vps
#
# Environment overrides (local and remote):
#   VPS_HOST              SSH host (default: ankk98-gpu-vps)
#   VPS_USER              SSH user (default: root)
#   VPS_LLAMA_DIR         llama.cpp tree on VPS (default: /root/llama.cpp)
#   VPS_MODELS_DIR        model dir on VPS (default: /root/models)
#   VPS_EVAL_DIR          eval JSON dir on VPS (default: /root/dspark_eval)
#   HF_TOKEN              optional Hugging Face token for faster downloads
#   DRAFT_REPO            draft HF repo (default: ankk98/dspark-gemma4-12b-block7-Q4_0-GGUF)
#   DRAFT_FILE            draft GGUF filename (default: dspark_gemma4_12b_q4pure.gguf)
#   TARGET_REPO           target HF repo (default: unsloth/gemma-4-12B-it-qat-GGUF)
#   TARGET_FILE           target GGUF filename (default: gemma-4-12B-it-qat-UD-Q4_K_XL.gguf)
#   CUDA_ARCH             CMAKE_CUDA_ARCHITECTURES (default: 86 for RTX 3090)
#   BUILD_JOBS            parallel build jobs (default: nproc on VPS)
#
# Flags:
#   --on-vps              run VPS-side steps only (no SSH/rsync)
#   --skip-sync           skip rsync from dev machine
#   --skip-models         skip Hugging Face model download
#   --skip-build          skip cmake configure/build
#   --clean-build         remove build/ before configuring
#   -h, --help            show this help
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EVAL_FIXTURE="$SCRIPT_DIR/dspark-vps/eval/code_500l.json"

VPS_HOST="${VPS_HOST:-ankk98-gpu-vps}"
VPS_USER="${VPS_USER:-root}"
VPS_SSH="${VPS_USER}@${VPS_HOST}"
VPS_LLAMA_DIR="${VPS_LLAMA_DIR:-/root/llama.cpp}"
VPS_MODELS_DIR="${VPS_MODELS_DIR:-/root/models}"
VPS_EVAL_DIR="${VPS_EVAL_DIR:-/root/dspark_eval}"

DRAFT_REPO="${DRAFT_REPO:-ankk98/dspark-gemma4-12b-block7-Q4_0-GGUF}"
DRAFT_FILE="${DRAFT_FILE:-dspark_gemma4_12b_q4pure.gguf}"
TARGET_REPO="${TARGET_REPO:-unsloth/gemma-4-12B-it-qat-GGUF}"
TARGET_FILE="${TARGET_FILE:-gemma-4-12B-it-qat-UD-Q4_K_XL.gguf}"
CUDA_ARCH="${CUDA_ARCH:-86}"
BUILD_JOBS="${BUILD_JOBS:-}"

ON_VPS=0
SKIP_SYNC=0
SKIP_MODELS=0
SKIP_BUILD=0
CLEAN_BUILD=0

log() { echo "[setup-dspark-vps] $*" >&2; }
die() { log "ERROR: $*"; exit 1; }

usage() {
    sed -n '3,35p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --on-vps)      ON_VPS=1; shift ;;
        --skip-sync)   SKIP_SYNC=1; shift ;;
        --skip-models) SKIP_MODELS=1; shift ;;
        --skip-build)  SKIP_BUILD=1; shift ;;
        --clean-build) CLEAN_BUILD=1; shift ;;
        -h|--help)     usage ;;
        *) die "unknown argument: $1 (try --help)" ;;
    esac
done

remote_setup() {
    SKIP_MODELS="${SKIP_MODELS:-0}"
    SKIP_BUILD="${SKIP_BUILD:-0}"
    CLEAN_BUILD="${CLEAN_BUILD:-0}"
    EVAL_FIXTURE="${EVAL_FIXTURE:-$SCRIPT_DIR/dspark-vps/eval/code_500l.json}"

    local jobs="${BUILD_JOBS:-$(nproc)}"
    local cuda_home=""
    local nvcc=""

    for d in /usr/local/cuda-12.6 /usr/local/cuda; do
        if [[ -x "$d/bin/nvcc" ]]; then
            cuda_home="$d"
            nvcc="$d/bin/nvcc"
            break
        fi
    done

    log "installing system packages"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq \
        build-essential cmake ninja-build git wget ca-certificates \
        python3 python3-pip

    if [[ -z "$nvcc" ]]; then
        log "CUDA toolkit not found; installing cuda-toolkit-12-6"
        local keyring="/tmp/cuda-keyring_1.1-1_all.deb"
        wget -q -O "$keyring" \
            https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
        dpkg -i "$keyring"
        apt-get update -qq
        apt-get install -y -qq cuda-toolkit-12-6
        rm -f "$keyring"
        for d in /usr/local/cuda-12.6 /usr/local/cuda; do
            if [[ -x "$d/bin/nvcc" ]]; then
                cuda_home="$d"
                nvcc="$d/bin/nvcc"
                break
            fi
        done
    fi

    [[ -n "$nvcc" ]] || die "nvcc not found after CUDA install"
    export PATH="$cuda_home/bin:$PATH"
    export LD_LIBRARY_PATH="${cuda_home}/lib64:${LD_LIBRARY_PATH:-}"

    if ! command -v hf >/dev/null 2>&1; then
        log "installing huggingface_hub CLI"
        pip3 install -q huggingface_hub
    fi

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        die "nvidia-smi not found; install NVIDIA driver before running this script"
    fi
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

    mkdir -p "$VPS_MODELS_DIR" "$VPS_EVAL_DIR"

    if [[ -f "$EVAL_FIXTURE" ]]; then
        cp -f "$EVAL_FIXTURE" "$VPS_EVAL_DIR/code_500l.json"
    elif [[ -f "$VPS_EVAL_DIR/code_500l.json" ]]; then
        log "keeping existing $VPS_EVAL_DIR/code_500l.json"
    else
        die "eval fixture missing: $EVAL_FIXTURE"
    fi

    if [[ "$SKIP_MODELS" -eq 0 ]]; then
        local draft_path="$VPS_MODELS_DIR/hf-draft/$DRAFT_FILE"
        local target_path="$VPS_MODELS_DIR/hf-target/$TARGET_FILE"

        if [[ -s "$draft_path" ]]; then
            log "draft already present: $draft_path"
        else
            log "downloading draft $DRAFT_REPO / $DRAFT_FILE"
            hf download "$DRAFT_REPO" "$DRAFT_FILE" --local-dir "$VPS_MODELS_DIR/hf-draft"
        fi
        ln -sf "$draft_path" "$VPS_MODELS_DIR/draft.gguf"

        if [[ -s "$target_path" ]]; then
            log "target already present: $target_path"
        else
            log "downloading target $TARGET_REPO / $TARGET_FILE"
            hf download "$TARGET_REPO" "$TARGET_FILE" --local-dir "$VPS_MODELS_DIR/hf-target"
        fi
        ln -sf "$target_path" "$VPS_MODELS_DIR/target.gguf"

        ls -lh "$VPS_MODELS_DIR/draft.gguf" "$VPS_MODELS_DIR/target.gguf"
    else
        log "skipping model download (--skip-models)"
    fi

    [[ -d "$VPS_LLAMA_DIR" ]] || die "llama.cpp not found at $VPS_LLAMA_DIR"

    if [[ "$SKIP_BUILD" -eq 0 ]]; then
        log "building compare_vanilla_speculative and smoke_batched_logits_repro"
        cd "$VPS_LLAMA_DIR"
        if [[ "$CLEAN_BUILD" -eq 1 ]]; then
            rm -rf build
        fi
        cmake -S . -B build -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DGGML_CUDA=ON \
            -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
            -DCMAKE_CUDA_COMPILER="$nvcc" \
            -DLLAMA_CURL=OFF
        cmake --build build -j"$jobs" \
            --target compare_vanilla_speculative smoke_batched_logits_repro
        ls -lh build/bin/compare_vanilla_speculative build/bin/smoke_batched_logits_repro
    else
        log "skipping build (--skip-build)"
    fi

    cat <<EOF

Setup complete.

Models:
  target: $VPS_MODELS_DIR/target.gguf
  draft:  $VPS_MODELS_DIR/draft.gguf

Eval:
  $VPS_EVAL_DIR/code_500l.json

Quick bench (from $VPS_LLAMA_DIR):
  export PATH=$cuda_home/bin:\$PATH
  export LD_LIBRARY_PATH=$cuda_home/lib64:\$LD_LIBRARY_PATH
  cd $VPS_LLAMA_DIR
  DSPARK_NO_ADAPTIVE_NMAX=1 ./build/bin/compare_vanilla_speculative \\
    -m $VPS_MODELS_DIR/target.gguf -md $VPS_MODELS_DIR/draft.gguf \\
    --input-ids $VPS_EVAL_DIR/code_500l.json \\
    --spec-type draft-dspark -c 512 -ngl 99 -ngld 99 --temp 0 --seed 42 \\
    --spec-draft-n-max 4 -n 400

P0 batched repro:
  DSPARK_NO_ADAPTIVE_NMAX=1 ./build/bin/smoke_batched_logits_repro \\
    -m $VPS_MODELS_DIR/target.gguf -md $VPS_MODELS_DIR/draft.gguf \\
    --input-ids $VPS_EVAL_DIR/code_500l.json --spec-type draft-dspark \\
    -c 512 -ngl 99 -ngld 99 --temp 0 --seed 42 --spec-draft-n-max 4 -n 400 --scan-all
EOF
}

if [[ "$ON_VPS" -eq 1 ]]; then
    remote_setup
    exit 0
fi

[[ -d "$REPO_ROOT" ]] || die "repo root not found: $REPO_ROOT"
[[ -f "$EVAL_FIXTURE" ]] || die "eval fixture not found: $EVAL_FIXTURE"

if [[ "$SKIP_SYNC" -eq 0 ]]; then
    log "rsync llama.cpp -> $VPS_SSH:$VPS_LLAMA_DIR"
    rsync -az --delete \
        --exclude=build/ \
        --exclude=.git/ \
        "$REPO_ROOT/" "$VPS_SSH:$VPS_LLAMA_DIR/"

    log "rsync eval fixtures -> $VPS_SSH:$VPS_EVAL_DIR"
    rsync -az "$SCRIPT_DIR/dspark-vps/eval/" "$VPS_SSH:$VPS_EVAL_DIR/"
else
    log "skipping rsync (--skip-sync)"
fi

log "running remote setup on $VPS_SSH"
ssh "$VPS_SSH" "bash -s -- --on-vps" <<REMOTE
set -euo pipefail
export VPS_LLAMA_DIR='$VPS_LLAMA_DIR'
export VPS_MODELS_DIR='$VPS_MODELS_DIR'
export VPS_EVAL_DIR='$VPS_EVAL_DIR'
export DRAFT_REPO='$DRAFT_REPO'
export DRAFT_FILE='$DRAFT_FILE'
export TARGET_REPO='$TARGET_REPO'
export TARGET_FILE='$TARGET_FILE'
export CUDA_ARCH='$CUDA_ARCH'
export BUILD_JOBS='${BUILD_JOBS}'
export HF_TOKEN='${HF_TOKEN:-}'
SKIP_MODELS=$SKIP_MODELS
SKIP_BUILD=$SKIP_BUILD
CLEAN_BUILD=$CLEAN_BUILD
EVAL_FIXTURE='$VPS_EVAL_DIR/code_500l.json'
$(declare -f log die remote_setup)
remote_setup
REMOTE

log "done"
