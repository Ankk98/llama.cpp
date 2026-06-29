#!/usr/bin/env python3
"""Phase 1 smoke test: DSpark Gemma4 HF -> GGUF conversion."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "gguf-py"))

from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402
from gguf.constants import GGMLQuantizationType  # noqa: E402


def meta_val(reader: GGUFReader, key: str):
    field = reader.get_field(key)
    if field is None:
        return None
    return field.contents()


def meta_array(reader: GGUFReader, key: str) -> list[int]:
    val = meta_val(reader, key)
    if val is None:
        return []
    return [int(x) for x in val]


def tensor_names(reader: GGUFReader) -> set[str]:
    return {t.name for t in reader.tensors}


def load_hf_tensor(draft_dir: Path, name: str) -> np.ndarray:
    import torch
    from safetensors import safe_open

    st_path = draft_dir / "model.safetensors"
    with safe_open(str(st_path), framework="pt") as f:
        if name in f.keys():
            return f.get_tensor(name).float().numpy()
        prefixed = f"model.{name}" if not name.startswith("model.") else name
        if prefixed in f.keys():
            return f.get_tensor(prefixed).float().numpy()
        raise KeyError(f"Tensor {name!r} not found in {st_path}")


def ensure_local_repo(repo: str, *, weights: bool = False) -> Path:
    from huggingface_hub import snapshot_download

    patterns = ["LICENSE", "*.json", "*.md", "*.txt", "tokenizer.model", "tokenizer.json"]
    if weights:
        patterns.append("*.safetensors")
    local_dir = snapshot_download(repo_id=repo, allow_patterns=patterns)
    return Path(local_dir)


def run_convert(hf_repo: str, target_model_dir: str, outfile: Path) -> tuple[Path, Path]:
    draft_dir = ensure_local_repo(hf_repo, weights=True)
    target_dir = ensure_local_repo(target_model_dir, weights=False)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "convert_hf_to_gguf.py"),
        str(draft_dir),
        "--outtype", "bf16",
        "--outfile", str(outfile),
        "--target-model-dir", str(target_dir),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    return draft_dir, target_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 DSpark GGUF conversion smoke test")
    parser.add_argument("--hf-repo", default="deepseek-ai/dspark_gemma4_12b_block7")
    parser.add_argument("--target-model-dir", default="google/gemma-4-12B-it")
    parser.add_argument("--outfile", default="/tmp/dspark_gemma4_12b_smoke.gguf")
    args = parser.parse_args()

    outfile = Path(args.outfile)
    draft_dir, _target_dir = run_convert(args.hf_repo, args.target_model_dir, outfile)

    reader = GGUFReader(str(outfile), "r")
    names = tensor_names(reader)

    arch = meta_val(reader, "general.architecture")
    assert arch == "dspark", f"expected architecture dspark, got {arch!r}"

    block_size = meta_val(reader, "dspark.block_size")
    assert block_size == 7, f"expected block_size 7, got {block_size!r}"

    markov_rank = meta_val(reader, "dspark.markov_rank")
    assert markov_rank == 256, f"expected markov_rank 256, got {markov_rank!r}"

    mask_token_id = meta_val(reader, "tokenizer.ggml.mask_token_id")
    assert mask_token_id == 4, f"expected mask_token_id 4, got {mask_token_id!r}"

    target_layers = meta_array(reader, "dspark.target_layers")
    assert target_layers == [6, 18, 30, 42, 47], f"unexpected target_layers: {target_layers}"

    causal = meta_val(reader, "dspark.attention.causal")
    assert causal is False or causal == 0, f"expected causal attention false, got {causal!r}"

    required = [
        "token_embd.weight",
        "output.weight",
        "fc.weight",
        "enc.output_norm.weight",
        "output_norm.weight",
        "markov.w1.weight",
        "markov.w2.weight",
        "confidence.proj.weight",
        "confidence.proj.bias",
    ]
    for tname in required:
        assert tname in names, f"missing required tensor {tname!r}"

    layer_tensors = [
        "attn_q", "attn_k", "attn_output", "attn_q_norm", "attn_k_norm",
        "attn_norm", "post_attention_norm", "ffn_norm", "post_ffw_norm",
        "layer_output_scale",
    ]
    for i in range(5):
        for suffix in layer_tensors:
            tname = f"blk.{i}.{suffix}.weight"
            assert tname in names, f"missing per-layer tensor {tname!r}"

    forbidden = [n for n in names if ".attn_v." in n or "v_norm" in n]
    assert not forbidden, f"unexpected v_proj/v_norm tensors: {forbidden}"

    gguf_fc = next(t for t in reader.tensors if t.name == "fc.weight")
    hf_fc = load_hf_tensor(draft_dir, "fc.weight")
    gguf_shape = tuple(int(x) for x in gguf_fc.shape)
    gguf_data = dequantize(gguf_fc.data, GGMLQuantizationType(gguf_fc.tensor_type)).reshape(gguf_shape)
    hf_data = hf_fc.reshape(gguf_shape)
    max_diff = float(np.max(np.abs(gguf_data - hf_data)))
    assert max_diff < 1e-5, f"fc.weight max abs diff {max_diff} >= 1e-5"

    print("smoke_phase1_convert: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
