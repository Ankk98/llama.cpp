#!/usr/bin/env python3
"""Generate DSpark decoder logits reference for Phase 2 graph smoke test."""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
DEEPSPEC_ROOT = Path.home() / "repos" / "DeepSpec"
sys.path.insert(0, str(DEEPSPEC_ROOT))

from deepspec.eval.dspark.draft_ops import forward_dspark_draft_block  # noqa: E402
from deepspec.modeling.dspark.gemma4 import Gemma4DSparkModel  # noqa: E402
from transformers import DynamicCache  # noqa: E402


def fill_raw_features(n_tokens: int, n_layers: int, hidden: int, seed: int) -> np.ndarray:
    # Deterministic fill shared with tests/smoke_phase2_graph.cpp (not RNG-based).
    total = n_tokens * n_layers * hidden
    idx = np.arange(total, dtype=np.float32) + seed * 7919
    return (np.sin(idx * 0.001) * 0.01).reshape(n_tokens, n_layers * hidden)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-repo", default="deepseek-ai/dspark_gemma4_12b_block7")
    parser.add_argument(
        "--outfile",
        default=str(REPO_ROOT / "tests/data/dspark_gemma4_decoder_logits_ref.bin"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-vocab-check", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cpu")
    dtype = torch.bfloat16

    model = Gemma4DSparkModel.from_pretrained(args.hf_repo, torch_dtype=dtype).to(device)
    model.eval()

    hidden = int(model.config.hidden_size)
    block_size = int(model.config.block_size)
    mask_token_id = int(model.config.mask_token_id)
    n_target_layers = len(model.target_layer_ids)

    anchor_token = 100
    raw_ctx = fill_raw_features(3, n_target_layers, hidden, args.seed)
    raw_inc = fill_raw_features(1, n_target_layers, hidden, args.seed + 1)

    position_ids = torch.arange(11, device=device, dtype=torch.long).unsqueeze(0)

    cache = DynamicCache()
    start = 3

    with torch.no_grad():
        target_hidden = torch.from_numpy(raw_ctx).unsqueeze(0).to(device=device, dtype=dtype)
        draft_input_ids = torch.full((1, block_size), mask_token_id, dtype=torch.long, device=device)
        draft_input_ids[:, 0] = anchor_token

        block_hidden_1 = forward_dspark_draft_block(
            model,
            draft_input_ids=draft_input_ids,
            position_ids=position_ids,
            past_key_values_draft=cache,
            target_hidden_states=target_hidden,
            start=start,
            block_size=block_size,
        )
        logits_1 = model.lm_head(block_hidden_1[:, :1, :]).float().squeeze(0).squeeze(0)

        target_hidden_2 = torch.from_numpy(raw_inc).unsqueeze(0).to(device=device, dtype=dtype)
        start_2 = 4
        draft_input_ids_2 = torch.full((1, block_size), mask_token_id, dtype=torch.long, device=device)
        draft_input_ids_2[:, 0] = anchor_token

        block_hidden_2 = forward_dspark_draft_block(
            model,
            draft_input_ids=draft_input_ids_2,
            position_ids=position_ids,
            past_key_values_draft=cache,
            target_hidden_states=target_hidden_2,
            start=start_2,
            block_size=block_size,
        )
        logits_2 = model.lm_head(block_hidden_2[:, :1, :]).float().squeeze(0).squeeze(0)

    n_check = min(args.n_vocab_check, logits_1.numel())
    ref_logits = logits_1[:n_check].cpu().numpy().astype(np.float32)

    out_path = Path(args.outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<i", n_check))
        ref_logits.tofile(f)

    meta_path = out_path.with_suffix(".meta.npz")
    np.savez(
        meta_path,
        seed=args.seed,
        anchor_token=anchor_token,
        raw_ctx=raw_ctx,
        raw_inc=raw_inc,
        logits_iter1=logits_1[:n_check].cpu().numpy(),
        logits_iter2=logits_2[:n_check].cpu().numpy(),
        block_size=block_size,
        mask_token_id=mask_token_id,
        n_embd_enc=n_target_layers * hidden,
    )

    print(f"wrote {out_path} ({n_check} floats)")
    print(f"wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
