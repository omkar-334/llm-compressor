"""Compute per-token KL divergence from two hidden-state stores.

Inputs:
  --baseline-dir: output_dir from extract_hidden_states.py for the reference model
  --target-dir:   same, for the compressed/target model

For each matched request:
  1. Load token_ids and final-layer hidden states from both stores.
  2. Verify token_ids are identical (bails otherwise; a mismatch means the two
     models tokenized differently and KLD is ill-defined without a remap).
  3. Apply each model's lm_head (loaded from HF) and optional final norm.
  4. Compute KL(P_baseline || P_target) per token, averaged over tokens.
  5. Write per-row JSONL and summary.json under --results-dir.

KL is computed in float32 with log-softmax for numerical stability. Tokens are
chunked so the vocab-size logit tensor never exceeds --chunk-tokens rows at
once.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Iterable

LOG = logging.getLogger("kld.compute")


def _load_manifest(d: Path) -> dict:
    p = d / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"missing manifest: {p}")
    with p.open() as f:
        return json.load(f)


def _load_entry(path: str):
    from safetensors import safe_open

    with safe_open(path, framework="pt") as f:
        token_ids = f.get_tensor("token_ids")
        hidden = f.get_tensor("hidden_states")
    return token_ids, hidden


_INNER_MODULE_CANDIDATES = ("model", "transformer", "gpt_neox", "backbone")
_FINAL_NORM_CANDIDATES = ("norm", "ln_f", "final_layer_norm", "final_norm")


def _find_final_norm(model):
    """Walk known inner-module names and known final-norm names.

    Returns the norm module if found, else None. Handles:
      - Llama / Mistral / Qwen2 / Gemma / Mixtral / Phi: model.model.norm
      - GPT-2:                                          model.transformer.ln_f
      - GPT-NeoX / Pythia:                              model.gpt_neox.final_layer_norm

    If a model places the final norm elsewhere, extend the candidate lists.
    """
    import torch.nn as nn

    for inner_name in _INNER_MODULE_CANDIDATES:
        inner = getattr(model, inner_name, None)
        if inner is None:
            continue
        for norm_name in _FINAL_NORM_CANDIDATES:
            norm = getattr(inner, norm_name, None)
            if isinstance(norm, nn.Module):
                return norm
    return None


def _load_lm_head_and_norm(model_id: str, device: str, dtype):
    """Return (lm_head Linear, optional final_norm module, config)."""
    from transformers import AutoModelForCausalLM

    LOG.info("loading lm_head for %s on %s", model_id, device)
    # Low-mem load: we only need lm_head and the final norm. AutoModelForCausalLM
    # will materialize the whole model; for KLD this is tolerable (user has at
    # least one GPU that already fits the model). If needed, a future pass can
    # load only the head and final norm via safetensors directly.
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, trust_remote_code=True)
    model.eval()
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise RuntimeError(f"{model_id} has no output embeddings / lm_head")

    final_norm = _find_final_norm(model)
    if final_norm is None:
        LOG.warning(
            "no final norm found for %s; if --apply-*-norm is set, KLD will be wrong. "
            "Extend _INNER_MODULE_CANDIDATES / _FINAL_NORM_CANDIDATES.",
            model_id,
        )

    lm_head = lm_head.to(device)
    if final_norm is not None:
        final_norm = final_norm.to(device)

    return lm_head, final_norm, model.config


def _apply_head(hidden, lm_head, final_norm, apply_norm: bool, softcap=None):
    """hidden: [L, H] -> logits [L, V].

    If softcap is a positive float (e.g. Gemma2's final_logit_softcapping),
    apply logits = tanh(logits / softcap) * softcap to match the model's
    native forward. Without this, KLD numbers are biased on arches that
    softcap.
    """
    import torch

    x = hidden
    if apply_norm and final_norm is not None:
        x = final_norm(x)
    logits = lm_head(x)
    if softcap:
        logits = torch.tanh(logits / softcap) * softcap
    return logits


def _kl_chunked(
    h_base, h_target,
    base_head, base_norm, base_apply_norm,
    target_head, target_norm, target_apply_norm,
    chunk_tokens: int,
    device: str,
    base_softcap=None,
    target_softcap=None,
):
    """Return per-token KL(P_baseline || P_target) as a 1D tensor of length L."""
    import torch
    import torch.nn.functional as F

    L = h_base.shape[0]
    out = torch.empty(L, dtype=torch.float32)

    for start in range(0, L, chunk_tokens):
        end = min(start + chunk_tokens, L)
        hb = h_base[start:end].to(device=device, dtype=next(base_head.parameters()).dtype)
        ht = h_target[start:end].to(device=device, dtype=next(target_head.parameters()).dtype)

        with torch.no_grad():
            logits_b = _apply_head(hb, base_head, base_norm, base_apply_norm, base_softcap).float()
            logits_t = _apply_head(ht, target_head, target_norm, target_apply_norm, target_softcap).float()

            log_p = F.log_softmax(logits_b, dim=-1)
            log_q = F.log_softmax(logits_t, dim=-1)
            p = log_p.exp()

            # KL(P||Q) = sum_v p_v * (log p_v - log q_v)
            kl = (p * (log_p - log_q)).sum(dim=-1)

        out[start:end] = kl.detach().cpu()

    return out


def compute(
    baseline_dir: Path,
    target_dir: Path,
    results_dir: Path,
    device: str,
    dtype_str: str,
    chunk_tokens: int,
    apply_baseline_norm: bool,
    apply_target_norm: bool,
    allow_token_mismatch: bool,
    max_requests: int | None,
) -> dict:
    import torch

    results_dir.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, dtype_str) if dtype_str != "auto" else torch.float16

    base_manifest = _load_manifest(baseline_dir)
    target_manifest = _load_manifest(target_dir)

    LOG.info("baseline model: %s (%d entries)", base_manifest["model_id"], len(base_manifest["entries"]))
    LOG.info("target model:   %s (%d entries)", target_manifest["model_id"], len(target_manifest["entries"]))

    base_head, base_norm, base_cfg = _load_lm_head_and_norm(base_manifest["model_id"], device, dtype)
    target_head, target_norm, target_cfg = _load_lm_head_and_norm(target_manifest["model_id"], device, dtype)

    if base_cfg.vocab_size != target_cfg.vocab_size:
        raise RuntimeError(
            f"vocab size mismatch: baseline={base_cfg.vocab_size} target={target_cfg.vocab_size}"
        )

    base_softcap = getattr(base_cfg, "final_logit_softcapping", None) or None
    target_softcap = getattr(target_cfg, "final_logit_softcapping", None) or None
    if base_softcap or target_softcap:
        LOG.info("softcap: baseline=%s target=%s", base_softcap, target_softcap)

    # Align entries by index. extract_hidden_states.py preserves input order.
    pairs = list(zip(base_manifest["entries"], target_manifest["entries"]))
    if max_requests is not None:
        pairs = pairs[:max_requests]

    jsonl_path = results_dir / "per_request.jsonl"
    per_row = jsonl_path.open("w")

    total_kl = 0.0
    total_tokens = 0
    mismatches = 0
    t0 = time.time()

    for i, (be, te) in enumerate(pairs):
        base_tok, base_hidden = _load_entry(be["path"])
        target_tok, target_hidden = _load_entry(te["path"])

        # hidden: [L, num_layers_to_extract, H]; we requested one layer, so squeeze dim 1.
        if base_hidden.ndim == 3:
            base_hidden = base_hidden[:, -1, :]
        if target_hidden.ndim == 3:
            target_hidden = target_hidden[:, -1, :]

        tok_ok = base_tok.shape == target_tok.shape and bool((base_tok == target_tok).all())
        if not tok_ok:
            mismatches += 1
            LOG.warning("token_ids mismatch at request %d", i)
            if not allow_token_mismatch:
                per_row.close()
                raise RuntimeError(
                    f"token mismatch at request {i}; rerun with matched prompts "
                    f"or pass --allow-token-mismatch"
                )
            # truncate to common prefix
            common = 0
            for a, b in zip(base_tok.tolist(), target_tok.tolist()):
                if a != b:
                    break
                common += 1
            if common == 0:
                continue
            base_hidden = base_hidden[:common]
            target_hidden = target_hidden[:common]
            base_tok = base_tok[:common]

        kl_per_tok = _kl_chunked(
            base_hidden, target_hidden,
            base_head, base_norm, apply_baseline_norm,
            target_head, target_norm, apply_target_norm,
            chunk_tokens=chunk_tokens,
            device=device,
            base_softcap=base_softcap,
            target_softcap=target_softcap,
        )

        n = int(kl_per_tok.numel())
        s = float(kl_per_tok.sum().item())
        mean = s / n if n > 0 else float("nan")

        row = {
            "index": i,
            "num_tokens": n,
            "mean_kl": mean,
            "min_kl": float(kl_per_tok.min().item()) if n > 0 else None,
            "max_kl": float(kl_per_tok.max().item()) if n > 0 else None,
        }
        per_row.write(json.dumps(row) + "\n")
        per_row.flush()

        total_kl += s
        total_tokens += n

        if (i + 1) % 10 == 0 or i == len(pairs) - 1:
            elapsed = time.time() - t0
            running_mean = total_kl / total_tokens if total_tokens else float("nan")
            LOG.info(
                "req %d/%d  tokens=%d  running_mean_kl=%.6f  elapsed=%.1fs",
                i + 1, len(pairs), total_tokens, running_mean, elapsed,
            )

    per_row.close()

    summary = {
        "baseline_model_id": base_manifest["model_id"],
        "target_model_id": target_manifest["model_id"],
        "num_requests": len(pairs),
        "num_tokens": total_tokens,
        "mean_kl": total_kl / total_tokens if total_tokens else None,
        "token_id_mismatches": mismatches,
        "wall_seconds": time.time() - t0,
        "apply_baseline_norm": apply_baseline_norm,
        "apply_target_norm": apply_target_norm,
        "chunk_tokens": chunk_tokens,
        "device": device,
        "dtype": dtype_str,
    }
    with (results_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    LOG.info("wrote summary: mean_kl=%s over %d tokens", summary["mean_kl"], total_tokens)
    return summary


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline-dir", type=Path, required=True)
    p.add_argument("--target-dir", type=Path, required=True)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32", "auto"])
    p.add_argument("--chunk-tokens", type=int, default=512)
    p.add_argument(
        "--apply-baseline-norm",
        action="store_true",
        help="apply model.model.norm before baseline lm_head (some arches need this if vLLM hook is pre-norm)",
    )
    p.add_argument("--apply-target-norm", action="store_true")
    p.add_argument("--allow-token-mismatch", action="store_true")
    p.add_argument("--max-requests", type=int, default=None)
    p.add_argument("--log-file", type=Path, default=None)
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=handlers,
    )

    compute(
        baseline_dir=args.baseline_dir,
        target_dir=args.target_dir,
        results_dir=args.results_dir,
        device=args.device,
        dtype_str=args.dtype,
        chunk_tokens=args.chunk_tokens,
        apply_baseline_norm=args.apply_baseline_norm,
        apply_target_norm=args.apply_target_norm,
        allow_token_mismatch=args.allow_token_mismatch,
        max_requests=args.max_requests,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
