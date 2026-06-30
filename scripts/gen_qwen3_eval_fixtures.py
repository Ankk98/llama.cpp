#!/usr/bin/env python3
"""Tokenize Qwen3 eval prompts for DSpark benchmarks (thinking on/off)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("transformers required: pip install transformers") from exc
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def tokenize_prompt(tok, text: str, thinking: bool) -> list[int]:
    msgs = [{"role": "user", "content": text}]
    kwargs = {"add_generation_prompt": True, "tokenize": True, "return_tensors": None}
    try:
        out = tok.apply_chat_template(msgs, enable_thinking=thinking, **kwargs)
    except TypeError:
        out = tok.apply_chat_template(msgs, **kwargs)
    if hasattr(out, "__getitem__"):
        ids = out["input_ids"] if "input_ids" in out else None
        if ids is not None:
            return list(ids)
    return list(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--prompts",
        default="scripts/dspark-vps/eval/qwen3/prompts.json",
        help="prompt manifest JSON",
    )
    ap.add_argument(
        "--out-dir",
        default="scripts/dspark-vps/eval/qwen3",
        help="output root directory",
    )
    ap.add_argument(
        "--tokenizer",
        default="Qwen/Qwen3-8B",
        help="HF tokenizer id or local path",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    prompts_path = root / args.prompts if not Path(args.prompts).is_absolute() else Path(args.prompts)
    out_root = root / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)

    manifest = json.loads(prompts_path.read_text())
    tok = load_tokenizer(args.tokenizer)

    index: dict[str, dict] = {"prompts": []}

    for category in ("code", "agentic"):
        cat_dir = out_root / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        for item in manifest.get(category, []):
            pid = item["id"]
            text = item["text"]
            for thinking in (False, True):
                think_tag = "on" if thinking else "off"
                ids = tokenize_prompt(tok, text, thinking)
                rel = f"{category}/{pid}_think_{think_tag}.json"
                out_path = out_root / rel
                out_path.write_text(json.dumps(ids))
                index["prompts"].append(
                    {
                        "id": pid,
                        "category": category,
                        "subtype": item.get("subtype", ""),
                        "thinking": think_tag,
                        "input_ids": rel,
                        "prompt_tokens": len(ids),
                        "text": text,
                    }
                )
                print(f"wrote {out_path} ({len(ids)} tokens, thinking={think_tag})")

    index_path = out_root / "manifest.json"
    index_path.write_text(json.dumps(index, indent=2))
    print(f"wrote {index_path} ({len(index['prompts'])} fixtures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
