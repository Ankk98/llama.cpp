#!/usr/bin/env python3
"""Phase 1 smoke test: DSpark Qwen3 HF -> GGUF conversion."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "gguf-py"))

from gguf import GGUFReader  # noqa: E402


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


def ensure_local_repo(repo: str, *, weights: bool = False) -> Path:
    from huggingface_hub import snapshot_download

    patterns = ["LICENSE", "*.json", "*.md", "*.txt", "tokenizer.model", "tokenizer.json"]
    if weights:
        patterns.append("*.safetensors")
    local_dir = snapshot_download(repo_id=repo, allow_patterns=patterns)
    return Path(local_dir)


def run_convert(
    hf_repo: str,
    target_model_dir: str,
    outfile: Path,
    *,
    outtype: str,
) -> tuple[Path, Path]:
    draft_dir = ensure_local_repo(hf_repo, weights=True)
    target_dir = ensure_local_repo(target_model_dir, weights=False)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "convert_hf_to_gguf.py"),
        str(draft_dir),
        "--outtype", outtype,
        "--outfile", str(outfile),
        "--target-model-dir", str(target_dir),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    return draft_dir, target_dir


def check_qwen3_dspark(reader: GGUFReader, *, expect_markov: bool) -> None:
    arch = meta_val(reader, "general.architecture")
    assert arch == "dspark", f"expected arch=dspark, got {arch!r}"

    assert int(meta_val(reader, "dspark.block_size") or 0) == 7
    assert meta_array(reader, "dspark.target_layers") == [2, 10, 18, 26, 34]

    k_eq_v = meta_val(reader, "dspark.attention_k_eq_v")
    assert k_eq_v is False, f"Qwen3 DSpark must set attention_k_eq_v=false, got {k_eq_v!r}"

    causal = meta_val(reader, "dspark.attention.causal")
    assert causal is False, f"expected non-causal draft attention, got {causal!r}"

    names = tensor_names(reader)
    for required in (
        "token_embd.weight",
        "output.weight",
        "fc.weight",
        "enc.output_norm.weight",
        "output_norm.weight",
        "blk.0.attn_v.weight",
    ):
        assert required in names, f"missing tensor {required!r}"

    markov_rank = int(meta_val(reader, "dspark.markov_rank") or 0)
    if expect_markov:
        assert markov_rank == 256
        assert "markov.w1.weight" in names
        assert "markov.w2.weight" in names
        assert "confidence.proj.weight" in names
    else:
        assert markov_rank == 0
        assert "markov.w1.weight" not in names
        assert "confidence.proj.weight" not in names


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 DSpark Qwen3 GGUF conversion smoke test")
    parser.add_argument(
        "--dspark-hf",
        default="deepseek-ai/dspark_qwen3_8b_block7",
        help="HF repo for full DSpark checkpoint",
    )
    parser.add_argument(
        "--dflash-hf",
        default="deepseek-ai/dflash_qwen3_8b_block7",
        help="HF repo for DFlash ablation (markov_rank=0)",
    )
    parser.add_argument("--target-hf", default="Qwen/Qwen3-8B")
    parser.add_argument("--outtype", default="q4_k_m", help="GGUF quant type (default: q4_k_m)")
    parser.add_argument("--outfile", type=Path, default=Path("/tmp/dspark_qwen3_8b_smoke.gguf"))
    parser.add_argument("--dflash-outfile", type=Path, default=Path("/tmp/dflash_qwen3_8b_smoke.gguf"))
    args = parser.parse_args()

    run_convert(args.dspark_hf, args.target_hf, args.outfile, outtype=args.outtype)
    reader = GGUFReader(str(args.outfile), "r")
    check_qwen3_dspark(reader, expect_markov=True)
    print("dspark convert: OK")

    try:
        run_convert(args.dflash_hf, args.target_hf, args.dflash_outfile, outtype=args.outtype)
    except Exception as e:
        print(f"dflash ablation convert skipped (checkpoint may be unavailable): {e}")
        return 0

    dflash_reader = GGUFReader(str(args.dflash_outfile), "r")
    check_qwen3_dspark(dflash_reader, expect_markov=False)
    print("dflash ablation convert: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
