#!/usr/bin/env bash
#
# One-shot DSpark CUDA benchmark VPS setup (idempotent).
#
# From your dev machine (rsync source, then run remote setup over SSH):
#   ./scripts/setup-dspark-cuda-vps.sh
#
# On the VPS directly (source already at VPS_LLAMA_DIR):
#   ./scripts/setup-dspark-cuda-vps.sh --on-vps
#
# Re-run safely after partial setup or driver updates:
#   ./scripts/setup-dspark-cuda-vps.sh --skip-sync
#   ./scripts/setup-dspark-cuda-vps.sh --on-vps --skip-models
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
#   --skip-verify         skip post-setup GPU smoke check
#   --clean-build         remove build/ before configuring
#   --reboot-if-gpu-broken  reboot VPS on driver/library mismatch, then resume
#   --verify-only         print setup status and exit (no changes)
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

# minimum non-empty GGUF sizes (bytes) for sanity checks
MIN_DRAFT_BYTES=$((400 * 1024 * 1024))
MIN_TARGET_BYTES=$((4 * 1024 * 1024 * 1024))

ON_VPS=0
SKIP_SYNC=0
SKIP_MODELS=0
SKIP_BUILD=0
SKIP_VERIFY=0
CLEAN_BUILD=0
REBOOT_IF_GPU_BROKEN=0
VERIFY_ONLY=0

# exit code from remote_setup when a reboot was requested
REBOOT_EXIT_CODE=42

log() { echo "[setup-dspark-vps] $*" >&2; }
die() { log "ERROR: $*"; exit 1; }

usage() {
    sed -n '3,42p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --on-vps)               ON_VPS=1; shift ;;
        --skip-sync)            SKIP_SYNC=1; shift ;;
        --skip-models)          SKIP_MODELS=1; shift ;;
        --skip-build)           SKIP_BUILD=1; shift ;;
        --skip-verify)          SKIP_VERIFY=1; shift ;;
        --clean-build)          CLEAN_BUILD=1; shift ;;
        --reboot-if-gpu-broken) REBOOT_IF_GPU_BROKEN=1; shift ;;
        --verify-only)          VERIFY_ONLY=1; shift ;;
        -h|--help)              usage ;;
        *) die "unknown argument: $1 (try --help)" ;;
    esac
done

resolve_path() {
    readlink -f "$1" 2>/dev/null || realpath "$1" 2>/dev/null || echo "$1"
}

gpu_query() {
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null
}

# prints: ok | mismatch | missing | error
gpu_status() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "missing"
        return 0
    fi
    local out rc=0
    out="$(nvidia-smi 2>&1)" || rc=$?
    if [[ $rc -eq 0 ]] && gpu_query >/dev/null 2>&1; then
        echo "ok"
        return 0
    fi
    if echo "$out" | grep -qi "Driver/library version mismatch"; then
        echo "mismatch"
        return 0
    fi
    echo "error"
}

ensure_gpu_ready() {
    local status
    status="$(gpu_status)"
    case "$status" in
        ok)
            log "GPU: $(gpu_query)"
            return 0
            ;;
        missing)
            die "nvidia-smi not found; install NVIDIA driver before running this script"
            ;;
        mismatch)
            log "NVIDIA driver/library mismatch (kernel driver != userspace libs)."
            log "CUDA runtime will not work until the VPS is rebooted."
            if [[ "$REBOOT_IF_GPU_BROKEN" -eq 1 ]]; then
                log "rebooting now (--reboot-if-gpu-broken)"
                sync
                nohup bash -c 'sleep 2; reboot' >/dev/null 2>&1 &
                exit "$REBOOT_EXIT_CODE"
            fi
            return 1
            ;;
        *)
            die "nvidia-smi failed; check NVIDIA driver installation"
            ;;
    esac
}

find_cuda_home() {
    local d
    for d in /usr/local/cuda-12.6 /usr/local/cuda; do
        if [[ -x "$d/bin/nvcc" ]]; then
            echo "$d"
            return 0
        fi
    done
    return 1
}

export_cuda_env() {
    local cuda_home="$1"
    export PATH="$cuda_home/bin:$PATH"
    export LD_LIBRARY_PATH="${cuda_home}/lib64:${LD_LIBRARY_PATH:-}"
}

ensure_system_packages() {
    log "ensuring system packages"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq \
        build-essential cmake ninja-build git wget ca-certificates \
        python3 python3-pip
}

ensure_cuda_toolkit() {
    local cuda_home nvcc
    cuda_home="$(find_cuda_home || true)"
    if [[ -n "$cuda_home" ]]; then
        echo "$cuda_home"
        return 0
    fi

    log "CUDA toolkit not found; installing cuda-toolkit-12-6"
    local keyring="/tmp/cuda-keyring_1.1-1_all.deb"
    wget -q -O "$keyring" \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i "$keyring"
    apt-get update -qq
    apt-get install -y -qq cuda-toolkit-12-6
    rm -f "$keyring"

    cuda_home="$(find_cuda_home || true)"
    [[ -n "$cuda_home" ]] || die "nvcc not found after CUDA toolkit install"
    echo "$cuda_home"
}

ensure_hf_cli() {
    if command -v hf >/dev/null 2>&1; then
        return 0
    fi
    log "installing huggingface_hub CLI"
    pip3 install -q --upgrade huggingface_hub
    command -v hf >/dev/null 2>&1 || die "hf CLI not available after pip install"
}

hf_auth_if_needed() {
    if [[ -n "${HF_TOKEN:-}" ]]; then
        export HF_TOKEN
        hf auth login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null \
            || hf auth login --token "$HF_TOKEN" 2>/dev/null \
            || true
    fi
}

gguf_ok() {
    local path="$1" min_bytes="$2"
    path="$(resolve_path "$path")"
    [[ -f "$path" ]] || return 1
    local size
    size="$(stat -c%s "$path" 2>/dev/null || stat -f%z "$path" 2>/dev/null || echo 0)"
    [[ "$size" -ge "$min_bytes" ]]
}

ensure_eval_fixture() {
    local src="${EVAL_FIXTURE:-$SCRIPT_DIR/dspark-vps/eval/code_500l.json}"
    local dest="$VPS_EVAL_DIR/code_500l.json"
    mkdir -p "$VPS_EVAL_DIR"
    if [[ -f "$src" ]]; then
        if [[ ! "$src" -ef "$dest" ]]; then
            cp -f "$src" "$dest"
        fi
        log "eval fixture: $dest"
    elif [[ -f "$dest" ]]; then
        log "eval fixture already present: $dest"
    else
        die "eval fixture missing: $src"
    fi
}

ensure_models() {
    local draft_path="$VPS_MODELS_DIR/hf-draft/$DRAFT_FILE"
    local target_path="$VPS_MODELS_DIR/hf-target/$TARGET_FILE"

    mkdir -p "$VPS_MODELS_DIR/hf-draft" "$VPS_MODELS_DIR/hf-target"
    hf_auth_if_needed

    if gguf_ok "$draft_path" "$MIN_DRAFT_BYTES"; then
        log "draft already present: $draft_path"
    else
        log "downloading draft $DRAFT_REPO / $DRAFT_FILE"
        rm -f "$draft_path"
        hf download "$DRAFT_REPO" "$DRAFT_FILE" --local-dir "$VPS_MODELS_DIR/hf-draft"
        gguf_ok "$draft_path" "$MIN_DRAFT_BYTES" || die "draft download looks incomplete: $draft_path"
    fi
    ln -sfn "$(resolve_path "$draft_path")" "$VPS_MODELS_DIR/draft.gguf"

    if gguf_ok "$target_path" "$MIN_TARGET_BYTES"; then
        log "target already present: $target_path"
    else
        log "downloading target $TARGET_REPO / $TARGET_FILE"
        rm -f "$target_path"
        hf download "$TARGET_REPO" "$TARGET_FILE" --local-dir "$VPS_MODELS_DIR/hf-target"
        gguf_ok "$target_path" "$MIN_TARGET_BYTES" || die "target download looks incomplete: $target_path"
    fi
    ln -sfn "$(resolve_path "$target_path")" "$VPS_MODELS_DIR/target.gguf"

    ls -lh "$VPS_MODELS_DIR/draft.gguf" "$VPS_MODELS_DIR/target.gguf"
}

ensure_build() {
    local cuda_home="$1" nvcc="$2" jobs="${BUILD_JOBS:-$(nproc)}"
    local bin_dir="$VPS_LLAMA_DIR/build/bin"
    local targets=(compare_vanilla_speculative smoke_batched_logits_repro)

    [[ -d "$VPS_LLAMA_DIR" ]] || die "llama.cpp not found at $VPS_LLAMA_DIR"
    cd "$VPS_LLAMA_DIR"
    export_cuda_env "$cuda_home"

    if [[ "$CLEAN_BUILD" -eq 1 ]]; then
        log "cleaning build directory"
        rm -rf build
    fi

    local -a cmake_args=(
        -S . -B build -G Ninja
        -DCMAKE_BUILD_TYPE=Release
        -DGGML_CUDA=ON
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
        -DCMAKE_CUDA_COMPILER="$nvcc"
        -DLLAMA_CURL=OFF
    )

    log "cmake configure (CUDA arch=$CUDA_ARCH)"
    cmake "${cmake_args[@]}"

    if ! grep -q 'GGML_CUDA:BOOL=ON' build/CMakeCache.txt; then
        log "CUDA backend not enabled; cleaning build and reconfiguring"
        rm -rf build
        cmake "${cmake_args[@]}"
        grep -q 'GGML_CUDA:BOOL=ON' build/CMakeCache.txt \
            || die "cmake did not enable GGML_CUDA (check toolkit and driver)"
    fi

    log "cmake build targets: ${targets[*]}"
    cmake --build build -j"$jobs" --target "${targets[@]}"

    local t
    for t in "${targets[@]}"; do
        [[ -x "$bin_dir/$t" ]] || die "missing binary after build: $bin_dir/$t"
    done
    ls -lh "$bin_dir/compare_vanilla_speculative" "$bin_dir/smoke_batched_logits_repro"
}

verify_gpu_runtime() {
    local cuda_home="$1"
    export_cuda_env "$cuda_home"

    local bench="$VPS_LLAMA_DIR/build/bin/compare_vanilla_speculative"
    [[ -x "$bench" ]] || die "verify: $bench not found"

    log "GPU smoke: 1-token greedy compare (CUDA)"
    if ! DSPARK_NO_ADAPTIVE_NMAX=1 "$bench" \
        -m "$VPS_MODELS_DIR/target.gguf" -md "$VPS_MODELS_DIR/draft.gguf" \
        --input-ids "$VPS_EVAL_DIR/code_500l.json" \
        --spec-type draft-dspark -c 512 -ngl 99 -ngld 99 --temp 0 --seed 42 \
        --spec-draft-n-max 4 -n 1 2>&1 | tee /tmp/dspark_setup_smoke.log | tail -5; then
        die "GPU smoke test failed; see /tmp/dspark_setup_smoke.log"
    fi

    if grep -qi "no usable GPU found" /tmp/dspark_setup_smoke.log; then
        die "GPU smoke test ran on CPU; CUDA runtime not working"
    fi
    log "GPU smoke test passed"
}

print_status() {
    local cuda_home="${1:-}"
    local gpu
    gpu="$(gpu_status)"
    log "status: gpu=$gpu llama_dir=$VPS_LLAMA_DIR models=$VPS_MODELS_DIR eval=$VPS_EVAL_DIR"
    [[ -f "$VPS_EVAL_DIR/code_500l.json" ]] && log "  eval fixture: OK" || log "  eval fixture: MISSING"
    gguf_ok "$VPS_MODELS_DIR/draft.gguf" "$MIN_DRAFT_BYTES" && log "  draft model: OK" || log "  draft model: MISSING"
    gguf_ok "$VPS_MODELS_DIR/target.gguf" "$MIN_TARGET_BYTES" && log "  target model: OK" || log "  target model: MISSING"
    [[ -x "$VPS_LLAMA_DIR/build/bin/compare_vanilla_speculative" ]] && log "  compare_vanilla_speculative: OK" || log "  compare_vanilla_speculative: MISSING"
    [[ -x "$VPS_LLAMA_DIR/build/bin/smoke_batched_logits_repro" ]] && log "  smoke_batched_logits_repro: OK" || log "  smoke_batched_logits_repro: MISSING"
    if [[ "$gpu" == "ok" ]]; then
        log "  nvidia-smi: $(gpu_query)"
    elif [[ "$gpu" == "mismatch" ]]; then
        log "  nvidia-smi: driver/library mismatch (reboot required)"
    fi
    if [[ -n "$cuda_home" ]]; then
        log "  cuda toolkit: $cuda_home"
    fi
}

print_usage_hint() {
    local cuda_home="$1"
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

remote_setup() {
    SKIP_MODELS="${SKIP_MODELS:-0}"
    SKIP_BUILD="${SKIP_BUILD:-0}"
    SKIP_VERIFY="${SKIP_VERIFY:-0}"
    CLEAN_BUILD="${CLEAN_BUILD:-0}"
    REBOOT_IF_GPU_BROKEN="${REBOOT_IF_GPU_BROKEN:-0}"
    VERIFY_ONLY="${VERIFY_ONLY:-0}"

    local cuda_home nvcc gpu_ok=0

    if [[ "$VERIFY_ONLY" -eq 1 ]]; then
        cuda_home="$(find_cuda_home || true)"
        print_status "$cuda_home"
        exit 0
    fi

    ensure_system_packages
    cuda_home="$(ensure_cuda_toolkit)"
    nvcc="$cuda_home/bin/nvcc"
    export_cuda_env "$cuda_home"

    ensure_hf_cli

    if ensure_gpu_ready; then
        gpu_ok=1
    else
        gpu_ok=0
        log "continuing setup without working GPU (build only); fix driver then re-run"
    fi

    ensure_eval_fixture

    if [[ "$SKIP_MODELS" -eq 0 ]]; then
        ensure_models
    else
        log "skipping model download (--skip-models)"
    fi

    if [[ "$SKIP_BUILD" -eq 0 ]]; then
        ensure_build "$cuda_home" "$nvcc"
    else
        log "skipping build (--skip-build)"
    fi

    print_status "$cuda_home"

    if [[ "$gpu_ok" -eq 1 && "$SKIP_VERIFY" -eq 0 ]]; then
        verify_gpu_runtime "$cuda_home"
    elif [[ "$gpu_ok" -eq 0 ]]; then
        log "skipping GPU smoke test (driver not ready)"
        log "re-run with: ./scripts/setup-dspark-cuda-vps.sh --skip-sync --skip-models --skip-build --reboot-if-gpu-broken"
    else
        log "skipping GPU smoke test (--skip-verify)"
    fi

    print_usage_hint "$cuda_home"
}

wait_for_ssh() {
    local max_wait="${1:-180}" waited=0
    log "waiting for $VPS_SSH to come back (max ${max_wait}s)"
    while [[ "$waited" -lt "$max_wait" ]]; do
        if ssh -o ConnectTimeout=5 -o BatchMode=yes "$VPS_SSH" true 2>/dev/null; then
            log "SSH ready after ${waited}s"
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
    done
    die "VPS did not come back within ${max_wait}s"
}

run_remote_setup() {
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
export MIN_DRAFT_BYTES='$MIN_DRAFT_BYTES'
export MIN_TARGET_BYTES='$MIN_TARGET_BYTES'
export REBOOT_EXIT_CODE='$REBOOT_EXIT_CODE'
SKIP_MODELS=$SKIP_MODELS
SKIP_BUILD=$SKIP_BUILD
SKIP_VERIFY=$SKIP_VERIFY
CLEAN_BUILD=$CLEAN_BUILD
REBOOT_IF_GPU_BROKEN=$REBOOT_IF_GPU_BROKEN
VERIFY_ONLY=$VERIFY_ONLY
EVAL_FIXTURE='$VPS_EVAL_DIR/code_500l.json'
$(declare -f log die resolve_path gpu_query gpu_status ensure_gpu_ready find_cuda_home \
    export_cuda_env ensure_system_packages ensure_cuda_toolkit ensure_hf_cli hf_auth_if_needed \
    gguf_ok ensure_eval_fixture ensure_models ensure_build \
    verify_gpu_runtime print_status print_usage_hint remote_setup)
remote_setup
REMOTE
}

if [[ "$ON_VPS" -eq 1 ]]; then
    remote_setup
    exit 0
fi

[[ -d "$REPO_ROOT" ]] || die "repo root not found: $REPO_ROOT"
[[ -f "$EVAL_FIXTURE" ]] || die "eval fixture not found: $EVAL_FIXTURE"

if [[ "$SKIP_SYNC" -eq 0 && "$VERIFY_ONLY" -eq 0 ]]; then
    log "rsync llama.cpp -> $VPS_SSH:$VPS_LLAMA_DIR"
    rsync -az --delete \
        --exclude=build/ \
        --exclude=.git/ \
        "$REPO_ROOT/" "$VPS_SSH:$VPS_LLAMA_DIR/"

    log "rsync eval fixtures -> $VPS_SSH:$VPS_EVAL_DIR"
    rsync -az "$SCRIPT_DIR/dspark-vps/eval/" "$VPS_SSH:$VPS_EVAL_DIR/"
else
    log "skipping rsync (--skip-sync or --verify-only)"
fi

log "running remote setup on $VPS_SSH"
set +e
run_remote_setup
rc=$?
set -e

if [[ "$rc" -eq "$REBOOT_EXIT_CODE" ]]; then
    wait_for_ssh 180
    log "resuming after reboot (skip sync and models)"
    SKIP_SYNC=1
    SKIP_MODELS=1
    REBOOT_IF_GPU_BROKEN=0
  set +e
  run_remote_setup
  rc=$?
  set -e
fi

[[ "$rc" -eq 0 ]] || die "remote setup failed (exit $rc)"
log "done"
