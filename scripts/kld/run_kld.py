"""End-to-end KLD driver: extract hidden states for two models, then compute.

Each stage is a separate subprocess call so that vLLM's singleton state does
not carry over between baseline and target extractions. Use --skip-extract-*
to resume from a previous run's hidden-state stores.

Writes all outputs under --run-dir:
  baseline_hidden/  (safetensors + manifest + prompts)
  target_hidden/
  results/          (per_request.jsonl, summary.json, kld.log)
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

LOG = logging.getLogger("kld.run")

SCRIPT_DIR = Path(__file__).resolve().parent


def _run(cmd: list[str]) -> None:
    LOG.info("$ %s", " ".join(cmd))
    t0 = time.time()
    r = subprocess.run(cmd)
    LOG.info("exit=%d wall=%.1fs", r.returncode, time.time() - t0)
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline-model", required=True)
    p.add_argument("--target-model", required=True)
    p.add_argument("--run-dir", type=Path, required=True)

    p.add_argument("--dataset-id", default="wikitext")
    p.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    p.add_argument("--dataset-split", default="test")
    p.add_argument("--text-column", default="text")
    p.add_argument("--num-samples", type=int, default=256)
    p.add_argument("--max-seq-length", type=int, default=2048)

    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--compute-device", default="cuda")
    p.add_argument("--compute-dtype", default="float16")
    p.add_argument("--chunk-tokens", type=int, default=512)
    p.add_argument("--apply-baseline-norm", action="store_true")
    p.add_argument("--apply-target-norm", action="store_true")
    p.add_argument("--allow-token-mismatch", action="store_true")

    p.add_argument("--skip-extract-baseline", action="store_true")
    p.add_argument("--skip-extract-target", action="store_true")
    return p.parse_args(list(argv) if argv is not None else None)


def _extract_cmd(model_id: str, output_dir: Path, args: argparse.Namespace, log_file: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "extract_hidden_states.py"),
        "--model-id", model_id,
        "--output-dir", str(output_dir),
        "--dataset-id", args.dataset_id,
        "--dataset-config", args.dataset_config,
        "--dataset-split", args.dataset_split,
        "--text-column", args.text_column,
        "--num-samples", str(args.num_samples),
        "--max-seq-length", str(args.max_seq_length),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--dtype", args.dtype,
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--seed", str(args.seed),
        "--log-file", str(log_file),
    ]
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    return cmd


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.run_dir / "run.log")],
    )

    with (args.run_dir / "run_config.json").open("w") as f:
        json.dump({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, f, indent=2)

    baseline_dir = args.run_dir / "baseline_hidden"
    target_dir = args.run_dir / "target_hidden"
    results_dir = args.run_dir / "results"

    if not args.skip_extract_baseline:
        _run(_extract_cmd(args.baseline_model, baseline_dir, args, args.run_dir / "extract_baseline.log"))
    else:
        LOG.info("skipping baseline extraction per flag")

    if not args.skip_extract_target:
        _run(_extract_cmd(args.target_model, target_dir, args, args.run_dir / "extract_target.log"))
    else:
        LOG.info("skipping target extraction per flag")

    compute_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "compute_kld.py"),
        "--baseline-dir", str(baseline_dir),
        "--target-dir", str(target_dir),
        "--results-dir", str(results_dir),
        "--device", args.compute_device,
        "--dtype", args.compute_dtype,
        "--chunk-tokens", str(args.chunk_tokens),
        "--log-file", str(args.run_dir / "compute_kld.log"),
    ]
    if args.apply_baseline_norm:
        compute_cmd.append("--apply-baseline-norm")
    if args.apply_target_norm:
        compute_cmd.append("--apply-target-norm")
    if args.allow_token_mismatch:
        compute_cmd.append("--allow-token-mismatch")
    _run(compute_cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
