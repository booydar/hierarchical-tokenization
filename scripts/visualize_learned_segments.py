#!/usr/bin/env python3
"""Load trained dynamic-chunking RMT checkpoints and print learned segment splits.

For each run under ``runs/dyn_chunk_*``, loads the latest checkpoint, runs the
routing module on KV-retrieval examples (trained pair count + cross-eval on
other pair counts), and prints a human-readable view of predicted segments.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoConfig, AutoTokenizer

from kv_dataset_utils import generate_sequence
from modeling_rmt.huggingface import RMTConfig, RMTForReasoningDynamicChunking


# Map run name → default data path used during training (from saved config.json).
PAIR_DATA_PATHS = {
    2: "data/P2-K4V4-S4(32-64)_1M",
    4: "data/P4-K4V4-S4(32-64)_1M",
    8: "data/N8-K4V4-S4(32-64)_1M",
}


def find_latest_checkpoint(run_dir: Path) -> Path:
    ckpts = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint-* in {run_dir}")
    return ckpts[-1]


def load_run_config(run_dir: Path) -> Dict[str, Any]:
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            data = json.load(f)
        return data.get("cli_args", data)
    with open(find_latest_checkpoint(run_dir) / "config.json") as f:
        return json.load(f)


def build_model_and_tokenizer(ckpt_dir: Path, run_cfg: Dict[str, Any], device: torch.device):
    """Rebuild model from run CLI config + checkpoint weights (same as training script)."""
    tokenizer_path = run_cfg.get("tokenizer_path", "./tokenizers/kv_alphabet_62/")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    base_model = run_cfg.get("base_model", "gpt2")
    if base_model == "gpt2":
        config = AutoConfig.from_pretrained("gpt2")
        config.n_layer = run_cfg["n_layer"]
        config.n_head = run_cfg["n_head"]
        config.n_embd = run_cfg["n_embd"]
    else:
        raise ValueError(f"Unsupported base_model in run config: {base_model}")

    config.vocab_size = tokenizer.vocab_size
    config.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")
    config.bos_token_id = tokenizer.convert_tokens_to_ids("[BOS]")
    config.eos_token_id = tokenizer.convert_tokens_to_ids("[EOS]")

    rmt_config = RMTConfig()
    rmt_config.base_model_config = config
    rmt_config.num_mem_tokens = run_cfg.get("n_mem_tokens") or run_cfg.get("num_mem_tokens", 8)
    rmt_config.max_n_segments = run_cfg.get("max_n_segments", 96)
    rmt_config.k2 = run_cfg.get("k2", -1)
    rmt_config.chunker_compression_ratio = run_cfg.get("chunker_compression_ratio")
    rmt_config.chunker_aux_loss_weight = run_cfg.get("chunker_aux_loss_weight", 0.01)
    rmt_config.chunker_ratio_loss_exclude_query_start = run_cfg.get(
        "chunker_ratio_loss_exclude_query_start", True,
    )
    rmt_config.query_token_id = tokenizer.convert_tokens_to_ids("?")

    model = RMTForReasoningDynamicChunking(rmt_config)
    weights = ckpt_dir / "model.safetensors"
    if weights.exists():
        from safetensors.torch import load_model
        load_model(model, str(weights), device=str(device))
    else:
        import torch as _torch
        state = _torch.load(ckpt_dir / "pytorch_model.bin", map_location="cpu")
        model.load_state_dict(state, strict=False)

    model.eval()
    model.to(device)
    return model, tokenizer


def encode_sample(sample: Dict[str, str], tokenizer):
    """Full training layout: context + query + target."""
    ctx = sample["context"]
    qry = sample["query"]
    tgt = sample.get("target", "")
    text = ctx + qry + tgt
    full_ids = tokenizer.encode(text, add_special_tokens=False)
    input_ids = torch.tensor([full_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask, text


def load_samples_for_pairs(
    n_pairs: int,
    n_examples: int,
    seed: int,
    data_root: Path,
) -> List[Dict[str, str]]:
    """Prefer on-disk HF dataset; fall back to fresh generation."""
    data_path = data_root / PAIR_DATA_PATHS.get(n_pairs, f"data/P{n_pairs}-K4V4-S4(32-64)_1M")
    if data_path.exists():
        import datasets
        ds = datasets.load_from_disk(str(data_path))
        split = "valid" if "valid" in ds else "test"
        rows = ds[split]
        rng = random.Random(seed)
        idxs = rng.sample(range(len(rows)), min(n_examples, len(rows)))
        return [rows[i] for i in idxs]

    rng = random.Random(seed + n_pairs)
    return [
        generate_sequence(
            num_kv_pairs=n_pairs,
            k_length=2,
            v_length=2,
            n_segments=4,
            min_segment_len=32,
            max_segment_len=64,
        )
        for _ in range(n_examples)
    ]


def segment_spans(boundary_mask: torch.Tensor, attn: torch.Tensor) -> List[Tuple[int, int]]:
    """Return (start, end) token spans for one sequence (exclusive end)."""
    L = int(attn.sum().item())
    mask = boundary_mask[:L].tolist()
    spans: List[Tuple[int, int]] = []
    start = 0
    for i in range(1, L):
        if mask[i]:
            spans.append((start, i))
            start = i
    spans.append((start, L))
    return spans


def highlight_kv_structure(text: str) -> str:
    """Mark pipe delimiters and !K:V! pairs for easier reading."""
    return re.sub(r"(\|)", r"[\1]", text)


def format_segmentation(
    text: str,
    token_strs: List[str],
    boundary_mask: torch.Tensor,
    p_boundary: torch.Tensor,
    attn: torch.Tensor,
) -> str:
    L = int(attn.sum().item())
    spans = segment_spans(boundary_mask, attn)
    lines: List[str] = []
    lines.append(f"  seq_len={L}  n_segments={len(spans)}  "
                 f"mean_seg_len={L / len(spans):.1f}")
    lines.append(f"  text (pipes bracketed): {highlight_kv_structure(text[:200])}"
                 f"{'...' if len(text) > 200 else ''}")

    for seg_i, (s, e) in enumerate(spans):
        seg_text = "".join(token_strs[s:e])
        # Boundary prob at segment start (position 0 is forced to 1.0).
        bp = float(p_boundary[s].item()) if s < L else 0.0
        lines.append(f"  seg {seg_i:2d}  tok[{s:3d}:{e:3d}]  len={e - s:3d}  "
                     f"p(boundary@start)={bp:.3f}  |{seg_text[:80]}{'...' if len(seg_text) > 80 else ''}|")

    # Show high-probability boundaries inside the sequence (excluding pos 0).
    thresh = 0.5
    hot = [
        (i, float(p_boundary[i].item()))
        for i in range(1, L)
        if boundary_mask[i] or p_boundary[i] > thresh
    ]
    if hot:
        hot_str = ", ".join(f"pos{i}:{p:.2f}" for i, p in hot[:20])
        if len(hot) > 20:
            hot_str += f", ... (+{len(hot) - 20} more)"
        lines.append(f"  boundary positions (mask or p>{thresh}): {hot_str}")

    return "\n".join(lines)


@torch.no_grad()
def predict_segments(model, input_ids, attention_mask):
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    bmask = out.boundary_mask[0].cpu()
    pbound = out.boundary_prob[0, :, 1].cpu()
    return bmask, pbound


def discover_runs(runs_root: Path) -> List[Path]:
    return sorted(runs_root.glob("dyn_chunk_*"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_root", type=Path, default=Path("runs"))
    parser.add_argument("--data_root", type=Path, default=Path("."))
    parser.add_argument("--n_examples", type=int, default=2,
                        help="Examples per (model, pair-count) combination.")
    parser.add_argument("--pair_counts", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run_filter", type=str, default=None,
                        help="Only process runs whose name contains this substring.")
    args = parser.parse_args()

    device = torch.device(args.device)
    runs = discover_runs(args.runs_root)
    if args.run_filter:
        runs = [r for r in runs if args.run_filter in r.name]

    if not runs:
        print(f"No runs found under {args.runs_root}")
        return

    print("=" * 80)
    print("Learned dynamic segmentation — KV retrieval checkpoints")
    print("=" * 80)

    for run_dir in runs:
        run_cfg = load_run_config(run_dir)
        train_pairs = run_cfg.get("n_pairs")
        compression = run_cfg.get("chunker_compression_ratio")
        ckpt = find_latest_checkpoint(run_dir)
        print(f"\n{'#' * 80}")
        print(f"RUN: {run_dir.name}")
        print(f"  checkpoint: {ckpt.name}")
        n_mem = run_cfg.get("n_mem_tokens") or run_cfg.get("num_mem_tokens")
        print(f"  trained n_pairs={train_pairs}  compression_ratio={compression}  "
              f"n_mem_tokens={n_mem}")
        print(f"{'#' * 80}")

        model, tokenizer = build_model_and_tokenizer(ckpt, run_cfg, device)

        for n_pairs in args.pair_counts:
            tag = "IN-DIST" if n_pairs == train_pairs else "OOD"
            print(f"\n--- {tag}  n_kv_pairs={n_pairs}  ({args.n_examples} examples) ---")
            samples = load_samples_for_pairs(
                n_pairs, args.n_examples, args.seed, args.data_root,
            )

            for ex_i, sample in enumerate(samples):
                input_ids, attention_mask, text = encode_sample(sample, tokenizer)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                bmask, pbound = predict_segments(model, input_ids, attention_mask)
                token_strs = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

                print(f"\n  Example {ex_i}  (context has {sample['context'].count('!') // 2} "
                      f"kv-pair markers, query={sample['query']!r})")
                print(format_segmentation(
                    text, token_strs, bmask.cpu(), pbound.cpu(), attention_mask[0].cpu(),
                ))

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
