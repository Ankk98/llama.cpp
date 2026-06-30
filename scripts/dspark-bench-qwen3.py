#!/usr/bin/env python3
"""
Qwen3 DSpark benchmark: vanilla target vs DSpark speculative.

Runs one harness process at a time (never two models loaded in parallel).
Appends each CSV row immediately (line-buffered + fsync).

Phases:
  1. vanilla-only (target) -> save expected outputs
  2. dspark spec-only (target + DSpark draft), confidence sweep
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


CSV_FIELDS = [
    "timestamp_utc",
    "git_commit",
    "decode_mode",
    "run_mode",
    "prompt_id",
    "category",
    "subtype",
    "thinking",
    "n_ctx",
    "confidence_threshold",
    "n_max",
    "n_predict",
    "n_prompt_tokens",
    "temp",
    "seed",
    "vanilla_pp_ms",
    "vanilla_gen_ms",
    "vanilla_pp_tok_s",
    "vanilla_tgp_tok_s",
    "vanilla_n_generated",
    "spec_pp_ms",
    "spec_gen_ms",
    "spec_pp_tok_s",
    "spec_tgp_tok_s",
    "spec_n_generated",
    "tgp_speedup_vs_vanilla",
    "accept_rate_pct",
    "mean_accepted_per_step",
    "n_propose_steps",
    "n_drafted",
    "n_accepted",
    "ms_draft_step",
    "ms_verify_step",
    "ms_logits_decode",
    "ms_decode_submit",
    "ms_accept",
    "ms_layer_commit",
    "ms_feature_redecode",
    "ms_process",
    "ms_propose_total",
    "tokens_per_propose",
    "ms_per_token_effective",
    "ms_per_token_vanilla_fwd",
    "draft_ms_total",
    "verify_ms_total",
    "token_match",
    "first_mismatch_gen",
    "input_ids_path",
    "reference_path",
    "draft_model_path",
    "json_results_path",
    "harness_exit_code",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def git_commit(root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True
        )
        return out.strip()
    except Exception:
        return "unknown"


def append_csv_row(csv_path: Path, row: dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8", buffering=1) as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
            f.flush()
            os.fsync(f.fileno())
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
        f.flush()
        os.fsync(f.fileno())


def load_manifest(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data["prompts"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def run_harness(
    compare_bin: Path,
    *,
    target: str,
    draft: str | None,
    input_ids: Path,
    reference: Path | None,
    save_output: Path | None,
    json_results: Path,
    vanilla_only: bool,
    spec_only: bool,
    n_predict: int,
    n_ctx: int,
    n_max: int,
    confidence: float,
    env: dict[str, str],
    extra_args: list[str],
) -> tuple[int, dict | None]:
    cmd = [
        str(compare_bin),
        "-m",
        target,
        "--input-ids",
        str(input_ids),
        "--temp",
        "0",
        "--seed",
        "42",
        "-c",
        str(n_ctx),
        "--device",
        "Vulkan0",
        "-ngl",
        "99",
        "-n",
        str(n_predict),
        "--json-results",
        str(json_results),
        "--spec-type",
        "draft-dspark",
        "--spec-draft-n-max",
        str(n_max),
        "--dspark-confidence-threshold",
        str(confidence),
    ]
    if draft:
        cmd.extend(["-md", draft, "-ngld", "99"])
    if vanilla_only:
        cmd.append("--vanilla-only")
    if spec_only:
        cmd.extend(["--spec-only"])
        if reference:
            cmd.extend(["--reference", str(reference)])
    if save_output:
        cmd.extend(["--save-output", str(save_output)])
    cmd.extend(extra_args)

    json_results.parent.mkdir(parents=True, exist_ok=True)
    if json_results.exists():
        json_results.unlink()

    print(f"\n>>> {' '.join(cmd)}", flush=True)
    proc = subprocess.run(
        cmd,
        cwd=repo_root(),
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
    )
    sys.stderr.write(proc.stderr)
    sys.stderr.flush()

    result = None
    if json_results.exists():
        try:
            result = json.loads(json_results.read_text())
        except json.JSONDecodeError:
            result = None
    return proc.returncode, result


def vanilla_row_from_json(
    commit: str,
    prompt: dict,
    n_predict: int,
    n_ctx: int,
    n_max: int,
    j: dict,
    *,
    input_ids: Path,
    reference: Path,
    json_path: Path,
    exit_code: int,
) -> dict:
    v = j.get("vanilla", {})
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "decode_mode": "vanilla",
        "run_mode": "vanilla",
        "prompt_id": prompt["id"],
        "category": prompt["category"],
        "subtype": prompt.get("subtype", ""),
        "thinking": prompt["thinking"],
        "n_ctx": n_ctx,
        "confidence_threshold": 0.0,
        "n_max": n_max,
        "n_predict": n_predict,
        "n_prompt_tokens": j.get("n_prompt_tokens", prompt.get("prompt_tokens", "")),
        "temp": j.get("temp", 0),
        "seed": j.get("seed", 42),
        "vanilla_pp_ms": v.get("pp_ms", ""),
        "vanilla_gen_ms": v.get("gen_ms", ""),
        "vanilla_pp_tok_s": v.get("pp_tok_s", ""),
        "vanilla_tgp_tok_s": v.get("tgp_tok_s", ""),
        "vanilla_n_generated": v.get("n_generated", ""),
        "token_match": "true",
        "first_mismatch_gen": "",
        "input_ids_path": str(input_ids),
        "reference_path": str(reference),
        "draft_model_path": "",
        "json_results_path": str(json_path),
        "harness_exit_code": exit_code,
    }


def dspark_row_from_json(
    commit: str,
    prompt: dict,
    n_predict: int,
    n_ctx: int,
    confidence: float,
    n_max: int,
    j: dict,
    vanilla_ref: dict,
    *,
    input_ids: Path,
    reference: Path,
    draft_path: str,
    json_path: Path,
    exit_code: int,
) -> dict:
    s = j.get("spec", {})
    v_tgp = vanilla_ref.get("vanilla_tgp_tok_s") or 0
    s_tgp = s.get("tgp_tok_s") or 0
    speedup = (float(s_tgp) / float(v_tgp)) if v_tgp and s_tgp else j.get("tgp_speedup", "")
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "decode_mode": "dspark",
        "run_mode": "spec",
        "prompt_id": prompt["id"],
        "category": prompt["category"],
        "subtype": prompt.get("subtype", ""),
        "thinking": prompt["thinking"],
        "n_ctx": n_ctx,
        "confidence_threshold": confidence,
        "n_max": n_max,
        "n_predict": n_predict,
        "n_prompt_tokens": j.get("n_prompt_tokens", prompt.get("prompt_tokens", "")),
        "temp": j.get("temp", 0),
        "seed": j.get("seed", 42),
        "vanilla_pp_ms": vanilla_ref.get("vanilla_pp_ms", ""),
        "vanilla_gen_ms": vanilla_ref.get("vanilla_gen_ms", ""),
        "vanilla_pp_tok_s": vanilla_ref.get("vanilla_pp_tok_s", ""),
        "vanilla_tgp_tok_s": vanilla_ref.get("vanilla_tgp_tok_s", ""),
        "vanilla_n_generated": vanilla_ref.get("vanilla_n_generated", ""),
        "spec_pp_ms": s.get("pp_ms", ""),
        "spec_gen_ms": s.get("gen_ms", ""),
        "spec_pp_tok_s": s.get("pp_tok_s", ""),
        "spec_tgp_tok_s": s.get("tgp_tok_s", ""),
        "spec_n_generated": s.get("n_generated", ""),
        "tgp_speedup_vs_vanilla": speedup,
        "accept_rate_pct": s.get("accept_rate_pct", ""),
        "mean_accepted_per_step": s.get("mean_accepted_per_step", ""),
        "n_propose_steps": s.get("n_propose_steps", ""),
        "n_drafted": s.get("n_drafted", ""),
        "n_accepted": s.get("n_accepted", ""),
        "ms_draft_step": s.get("ms_per_step_draft", ""),
        "ms_verify_step": s.get("ms_per_step_verify", ""),
        "ms_logits_decode": s.get("ms_per_step_logits_decode", ""),
        "ms_decode_submit": s.get("ms_per_step_decode_submit", ""),
        "ms_accept": s.get("ms_per_step_accept", ""),
        "ms_layer_commit": s.get("ms_per_step_layer_commit", ""),
        "ms_feature_redecode": s.get("ms_per_step_feature_redecode", ""),
        "ms_process": s.get("ms_per_step_process", ""),
        "ms_propose_total": s.get("ms_per_propose_total", ""),
        "tokens_per_propose": s.get("tokens_per_propose", ""),
        "ms_per_token_effective": s.get("ms_per_token_effective", ""),
        "ms_per_token_vanilla_fwd": s.get("ms_per_token_vanilla_fwd", ""),
        "draft_ms_total": s.get("draft_ms_total", ""),
        "verify_ms_total": s.get("verify_ms_total", ""),
        "token_match": j.get("token_match", ""),
        "first_mismatch_gen": j.get("first_mismatch_gen", ""),
        "input_ids_path": str(input_ids),
        "reference_path": str(reference),
        "draft_model_path": draft_path,
        "json_results_path": str(json_path),
        "harness_exit_code": exit_code,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default=os.environ.get("DSPARK_TARGET", "~/models/Qwen3-8B-Q4_K_M.gguf"))
    ap.add_argument("--draft", default=os.environ.get("DSPARK_DRAFT", "~/models/dspark_qwen3_8b_block7.q4_k_m.gguf"))
    ap.add_argument("--compare-bin", default="build/bin/compare_vanilla_speculative")
    ap.add_argument("--manifest", default="scripts/dspark-vps/eval/qwen3/manifest.json")
    ap.add_argument("--csv", default="scripts/dspark-vps/eval/qwen3/results.csv")
    ap.add_argument("--expected-dir", default="scripts/dspark-vps/eval/qwen3/expected")
    ap.add_argument("--json-dir", default="scripts/dspark-vps/eval/qwen3/runs")
    ap.add_argument("-c", "--n-ctx", type=int, default=8096, help="context size (never use full trained ctx)")
    ap.add_argument("-n", "--n-predict", type=int, default=200)
    ap.add_argument("--n-max", type=int, default=4)
    ap.add_argument(
        "--confidence",
        type=float,
        nargs="*",
        default=[0.0, 0.3, 0.5, 0.7, 0.9],
        help="DSpark confidence thresholds",
    )
    ap.add_argument("--categories", nargs="*", default=["code", "agentic"])
    ap.add_argument("--thinking", nargs="*", default=["off", "on"])
    ap.add_argument("--prompt-ids", nargs="*", help="subset of prompt ids")
    ap.add_argument("--vanilla-only", action="store_true", help="only phase 1")
    ap.add_argument("--dspark-only", action="store_true", help="only phase 2 (needs expected)")
    ap.add_argument("--no-cooldown", action="store_true", help="set DSPARK_BENCH_NO_COOLDOWN=1")
    ap.add_argument("--quick", action="store_true", help="2 prompts/category, n=64, conf 0 only")
    args = ap.parse_args()

    root = repo_root()
    target = str(Path(args.target).expanduser())
    draft = str(Path(args.draft).expanduser())
    compare_bin = root / args.compare_bin
    manifest_path = root / args.manifest
    csv_path = root / args.csv
    expected_dir = root / args.expected_dir
    json_dir = root / args.json_dir
    expected_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    if not compare_bin.exists():
        print(f"error: missing harness binary: {compare_bin}", file=sys.stderr)
        return 1
    if not manifest_path.exists():
        print(f"error: missing manifest {manifest_path}; run scripts/gen_qwen3_eval_fixtures.py", file=sys.stderr)
        return 1

    prompts = load_manifest(manifest_path)
    prompts = [p for p in prompts if p["category"] in args.categories]
    prompts = [p for p in prompts if p["thinking"] in args.thinking]
    if args.prompt_ids:
        wanted = set(args.prompt_ids)
        prompts = [p for p in prompts if p["id"] in wanted]

    if args.quick:
        prompts = [p for p in prompts if p["id"].endswith("01") or p["id"].endswith("02")]
        args.n_predict = 64
        args.confidence = [0.0]

    commit = git_commit(root)
    env = os.environ.copy()
    env["DSPARK_NO_ADAPTIVE_NMAX"] = "1"
    if args.no_cooldown:
        env["DSPARK_BENCH_NO_COOLDOWN"] = "1"

    vanilla_cache: dict[str, dict] = {}

    if not args.dspark_only:
        for p in prompts:
            input_ids = root / "scripts/dspark-vps/eval/qwen3" / p["input_ids"]
            ref_path = expected_dir / f"{p['id']}_think_{p['thinking']}.json"
            json_path = json_dir / f"vanilla_{p['id']}_think_{p['thinking']}.json"
            code, j = run_harness(
                compare_bin,
                target=target,
                draft=None,
                input_ids=input_ids,
                reference=None,
                save_output=ref_path,
                json_results=json_path,
                vanilla_only=True,
                spec_only=False,
                n_predict=args.n_predict,
                n_ctx=args.n_ctx,
                n_max=args.n_max,
                confidence=0.0,
                env=env,
                extra_args=[],
            )
            if j is None:
                print(f"warning: no json results for vanilla {p['id']}", file=sys.stderr)
                continue
            row = vanilla_row_from_json(
                commit, p, args.n_predict, args.n_ctx, args.n_max, j,
                input_ids=input_ids, reference=ref_path, json_path=json_path, exit_code=code,
            )
            vanilla_cache[f"{p['id']}_think_{p['thinking']}"] = row
            append_csv_row(csv_path, row)
            print(f"vanilla {p['id']} thinking={p['thinking']} -> csv", flush=True)
            if code != 0:
                return code

    if not args.vanilla_only:
        for p in prompts:
            input_ids = root / "scripts/dspark-vps/eval/qwen3" / p["input_ids"]
            ref_path = expected_dir / f"{p['id']}_think_{p['thinking']}.json"
            if not ref_path.exists():
                print(f"warning: missing reference {ref_path}, skip", file=sys.stderr)
                continue

            key = f"{p['id']}_think_{p['thinking']}"
            van_row = vanilla_cache.get(key)
            if van_row is None:
                vpath = json_dir / f"vanilla_{p['id']}_think_{p['thinking']}.json"
                if vpath.exists():
                    van_row = vanilla_row_from_json(
                        commit, p, args.n_predict, args.n_ctx, args.n_max, load_json(vpath),
                        input_ids=input_ids, reference=ref_path, json_path=vpath, exit_code=0,
                    )
                else:
                    van_row = {}

            for conf in args.confidence:
                json_path = json_dir / f"dspark_{p['id']}_think_{p['thinking']}_c{conf}_n{args.n_max}.json"
                code, j = run_harness(
                    compare_bin,
                    target=target,
                    draft=draft,
                    input_ids=input_ids,
                    reference=ref_path,
                    save_output=None,
                    json_results=json_path,
                    vanilla_only=False,
                    spec_only=True,
                    n_predict=args.n_predict,
                    n_ctx=args.n_ctx,
                    n_max=args.n_max,
                    confidence=conf,
                    env=env,
                    extra_args=[],
                )
                if j is None:
                    print(f"warning: no json for dspark {p['id']} conf={conf}", file=sys.stderr)
                    continue
                row = dspark_row_from_json(
                    commit, p, args.n_predict, args.n_ctx, conf, args.n_max, j, van_row,
                    input_ids=input_ids, reference=ref_path, draft_path=draft,
                    json_path=json_path, exit_code=code,
                )
                append_csv_row(csv_path, row)
                print(
                    f"dspark {p['id']} thinking={p['thinking']} conf={conf} "
                    f"match={row.get('token_match')} speedup={row.get('tgp_speedup_vs_vanilla')}",
                    flush=True,
                )
                if code != 0:
                    return code
                time.sleep(1)

    print(f"results appended to {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
