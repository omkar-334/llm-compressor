"""Extract last-layer hidden states from a model running in vLLM.

Rationale: full-vocab logprob dumps for KLD are bottlenecked on disk transfer
(see issue #2646). Hidden states are ~vocab/hidden_size smaller (~30x for
Llama-3-8B) and are enough to recompute logprobs offline via lm_head.

This uses the vLLM >= 0.18 extraction path:
  speculative_config: method="extract_hidden_states"
  kv_transfer_config: ExampleHiddenStatesConnector, role=kv_producer

One safetensors file per request is written under --output-dir, each containing
"token_ids" [L] and "hidden_states" [L, num_layers_to_extract, hidden_size].
A manifest.json records inputs, shapes, dtypes, and the mapping from
request index -> safetensors path so compute_kld.py can align two runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Iterable

LOG = logging.getLogger("kld.extract")


def _build_dataset(
    dataset_id: str,
    dataset_split: str,
    dataset_config: str | None,
    text_column: str,
    num_samples: int,
    max_seq_length: int,
    tokenizer,
    seed: int,
) -> list[str]:
    """Return a list of prompt strings, each at most max_seq_length tokens."""
    from datasets import load_dataset

    LOG.info("loading dataset %s split=%s config=%s", dataset_id, dataset_split, dataset_config)
    if dataset_config is not None:
        ds = load_dataset(dataset_id, dataset_config, split=dataset_split)
    else:
        ds = load_dataset(dataset_id, split=dataset_split)

    ds = ds.shuffle(seed=seed)

    prompts: list[str] = []
    for row in ds:
        text = row.get(text_column)
        if not text or not isinstance(text, str):
            continue
        token_ids = tokenizer(text, truncation=True, max_length=max_seq_length, add_special_tokens=False)["input_ids"]
        if len(token_ids) < 8:
            continue
        prompts.append(tokenizer.decode(token_ids, skip_special_tokens=False))
        if len(prompts) >= num_samples:
            break

    LOG.info("prepared %d prompts", len(prompts))
    return prompts


def _final_layer_idx_from_config(cfg) -> int:
    """Return the 0-indexed final decoder layer from an HF config object."""
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        if hasattr(cfg, attr):
            n = getattr(cfg, attr)
            if isinstance(n, int) and n > 0:
                return n - 1
    raise RuntimeError(f"cannot determine number of decoder layers from config {type(cfg).__name__}")


def _resolve_final_layer_idx(model_id: str) -> int:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    return _final_layer_idx_from_config(cfg)


def extract(
    model_id: str,
    output_dir: Path,
    prompts: list[str],
    tensor_parallel_size: int,
    dtype: str,
    gpu_memory_utilization: float,
    enforce_eager: bool,
    seed: int,
) -> dict:
    """Run vLLM once and return a manifest dict. Files are written to output_dir."""
    from vllm import LLM, SamplingParams

    final_layer_idx = _resolve_final_layer_idx(model_id)
    LOG.info("final decoder layer index for %s: %d", model_id, final_layer_idx)

    output_dir.mkdir(parents=True, exist_ok=True)

    llm = LLM(
        model=model_id,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        seed=seed,
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {"eagle_aux_hidden_state_layer_ids": [final_layer_idx]},
            },
        },
        kv_transfer_config={
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {"shared_storage_path": str(output_dir)},
        },
    )

    sp = SamplingParams(max_tokens=1, temperature=0.0)
    LOG.info("calling vLLM generate on %d prompts", len(prompts))
    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    wall = time.time() - t0
    LOG.info("vLLM generate finished in %.1fs", wall)

    entries = []
    for i, o in enumerate(outputs):
        path = o.kv_transfer_params.get("hidden_states_path") if o.kv_transfer_params else None
        if path is None:
            LOG.warning("no hidden_states_path for prompt index %d; skipping", i)
            continue
        entries.append(
            {
                "index": i,
                "request_id": getattr(o, "request_id", str(i)),
                "path": str(Path(path).resolve()),
                "prompt_preview": prompts[i][:200],
            }
        )

    manifest = {
        "model_id": model_id,
        "final_layer_idx": final_layer_idx,
        "dtype": dtype,
        "num_prompts": len(prompts),
        "num_extracted": len(entries),
        "vllm_wall_seconds": wall,
        "entries": entries,
    }
    with (output_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    LOG.info("wrote manifest with %d entries", len(entries))
    return manifest


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
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
    p.add_argument("--log-file", type=Path, default=None)
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)

    log_handlers: list[logging.Handler] = [logging.StreamHandler()]
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=log_handlers,
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    prompts = _build_dataset(
        dataset_id=args.dataset_id,
        dataset_split=args.dataset_split,
        dataset_config=args.dataset_config,
        text_column=args.text_column,
        num_samples=args.num_samples,
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        seed=args.seed,
    )
    if not prompts:
        LOG.error("no prompts built from dataset; aborting")
        return 2

    prompts_path = args.output_dir / "prompts.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with prompts_path.open("w") as f:
        json.dump(prompts, f)
    LOG.info("wrote %d prompts to %s", len(prompts), prompts_path)

    extract(
        model_id=args.model_id,
        output_dir=args.output_dir,
        prompts=prompts,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
