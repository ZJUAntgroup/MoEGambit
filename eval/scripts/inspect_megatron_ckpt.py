#!/usr/bin/env python
"""Dump the state_dict keys + shapes of a Megatron-LM ``--ckpt-format torch``
MoE checkpoint, so we can write a faithful Megatron->HF converter.

Why this exists: the production checkpoint layout depends on which MoE
expert implementation Megatron-LM picked at training time (GroupedMLP,
TEGroupedMLP, SequentialMLP), whether QKV is fused, whether the column-
parallel linear has a fused layernorm, etc. Rather than guess, we run
this script once on the training cluster against a real checkpoint and
use its output to drive the converter (``megatron_moe_to_hf.py``).

Usage on the cluster::

    python eval/scripts/inspect_megatron_ckpt.py \
        --ckpt /mnt/ais-c1/dataset/zds/main_exp/5.27/moeguard/ckpt \
        [--iter 10000]                # default: read latest_checkpointed_iteration.txt
        [--pp-rank 0 --ep-rank 0]     # which shard to dump (default 0/0)
        [--full]                      # dump *every* shard, not just one
        [--out keys.txt]              # write to file instead of stdout

The script never imports Megatron; it only uses ``torch.load`` so it is
safe to run on the eval/login node.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterator, Tuple

import torch


# Megatron pickled ``args`` as an ``argparse.Namespace`` plus helper objects
# (e.g. enums) whose classes live under the ``megatron`` package. ``torch.load``
# uses pickle so the package MUST be importable, otherwise we hit
# ``ModuleNotFoundError: No module named 'megatron'`` before reading any tensor.
def _ensure_megatron_importable() -> None:
    try:
        import megatron  # noqa: F401
        return
    except ImportError:
        pass
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "Megatron-LM"
        if (candidate / "megatron" / "__init__.py").is_file():
            sys.path.insert(0, str(candidate))
            try:
                import megatron  # noqa: F401
                print(f"[inspect] auto-added {candidate} to sys.path",
                      file=sys.stderr)
                return
            except ImportError:
                sys.path.pop(0)
    print("[inspect] WARNING: 'megatron' is not importable; torch.load may "
          "fail on Megatron-pickled args. Set PYTHONPATH to include the "
          "Megatron-LM root.", file=sys.stderr)


_ensure_megatron_importable()


def _iter_shards(iter_dir: Path) -> Iterator[Tuple[int, int, int, Path]]:
    """Yield (tp, pp, ep, shard_path) for every mp_rank_* subdir."""
    for sub in sorted(iter_dir.iterdir()):
        if not sub.is_dir():
            continue
        if not sub.name.startswith("mp_rank_"):
            continue
        # mp_rank_{tp:02d}_{pp:03d}_{ep:03d}  OR  mp_rank_{tp:02d}_{pp:03d}
        parts = sub.name.split("_")
        if len(parts) == 4:
            _, _, tp, pp = parts
            tp, pp, ep = int(tp), int(pp), 0
        elif len(parts) == 5:
            _, _, tp, pp, ep = parts
            tp, pp, ep = int(tp), int(pp), int(ep)
        else:
            continue
        ckpt = sub / "model_optim_rng.pt"
        if ckpt.is_file():
            yield tp, pp, ep, ckpt


def _flatten(d, prefix=""):
    """Walk a (possibly nested) state_dict-like dict yielding (key, value)."""
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _flatten(v, f"{prefix}{k}.")
    else:
        yield prefix.rstrip("."), d


def _shape_str(t) -> str:
    if isinstance(t, torch.Tensor):
        return f"{tuple(t.shape)} {t.dtype}"
    return f"<{type(t).__name__}>"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="Megatron ckpt root (parent of iter_*/).")
    ap.add_argument("--iter", type=int, default=None,
                    help="Iteration to inspect (default: latest).")
    ap.add_argument("--pp-rank", type=int, default=0)
    ap.add_argument("--ep-rank", type=int, default=0)
    ap.add_argument("--tp-rank", type=int, default=0)
    ap.add_argument("--full", action="store_true",
                    help="Dump every shard, not only (tp,pp,ep) selected.")
    ap.add_argument("--out", default=None,
                    help="Write dump to file (default: stdout).")
    ap.add_argument("--values", action="store_true",
                    help="Also print a tiny statistic (mean/std/min/max).")
    args = ap.parse_args()

    ckpt_root = Path(args.ckpt).resolve()
    if not ckpt_root.is_dir():
        print(f"[inspect] ERROR: {ckpt_root} not a directory", file=sys.stderr)
        return 2

    if args.iter is None:
        latest_file = ckpt_root / "latest_checkpointed_iteration.txt"
        if not latest_file.is_file():
            print(f"[inspect] ERROR: no --iter given and no "
                  f"{latest_file.name} present", file=sys.stderr)
            return 2
        args.iter = int(latest_file.read_text().strip())
    iter_dir = ckpt_root / f"iter_{args.iter:07d}"
    if not iter_dir.is_dir():
        print(f"[inspect] ERROR: {iter_dir} does not exist", file=sys.stderr)
        return 2

    shards = list(_iter_shards(iter_dir))
    if not shards:
        print(f"[inspect] ERROR: no mp_rank_* found in {iter_dir}",
              file=sys.stderr)
        return 2

    if args.full:
        selected = shards
    else:
        selected = [s for s in shards if (s[0], s[1], s[2]) ==
                    (args.tp_rank, args.pp_rank, args.ep_rank)]
        if not selected:
            print(f"[inspect] ERROR: shard tp={args.tp_rank} pp={args.pp_rank} "
                  f"ep={args.ep_rank} not found. Available: "
                  f"{[(s[0], s[1], s[2]) for s in shards[:8]]}...",
                  file=sys.stderr)
            return 2

    out_fp = open(args.out, "w") if args.out else sys.stdout
    try:
        print(f"# ckpt root        : {ckpt_root}", file=out_fp)
        print(f"# iteration        : {args.iter}", file=out_fp)
        print(f"# shards total     : {len(shards)}", file=out_fp)
        print(f"# shards selected  : {len(selected)}", file=out_fp)
        print(f"# format           : torch (per-rank .pt)", file=out_fp)
        print("#" + "-" * 78, file=out_fp)
        for tp, pp, ep, ckpt in selected:
            print(f"\n## shard tp={tp} pp={pp} ep={ep}  path={ckpt}", file=out_fp)
            try:
                sd = torch.load(ckpt, map_location="cpu",
                                weights_only=False)
            except Exception as exc:
                print(f"!! torch.load failed: {exc}", file=out_fp)
                continue
            # Megatron ckpt top-level usually has: 'args', 'iteration',
            # 'model', 'optimizer', 'opt_param_scheduler', 'rng_state',
            # 'num_floating_point_operations_so_far', ...
            if isinstance(sd, dict):
                print(f"# top-level keys: {sorted(sd.keys())}", file=out_fp)
            if isinstance(sd, dict) and "args" in sd:
                a = sd["args"]
                interesting = [
                    "tensor_model_parallel_size",
                    "pipeline_model_parallel_size",
                    "expert_model_parallel_size",
                    "num_layers", "hidden_size", "ffn_hidden_size",
                    "num_attention_heads", "num_query_groups",
                    "kv_channels",
                    "num_experts", "moe_router_topk",
                    "moe_ffn_hidden_size",
                    "moe_grouped_gemm",
                    "moe_use_legacy_grouped_gemm",
                    "transformer_impl",
                    "swiglu",
                    "untie_embeddings_and_output_weights",
                ]
                print("# args (relevant):", file=out_fp)
                for k in interesting:
                    v = getattr(a, k, "<missing>")
                    print(f"#   {k} = {v}", file=out_fp)
            print("# model state_dict:", file=out_fp)
            model_blob = sd.get("model") if isinstance(sd, dict) else sd
            if not isinstance(model_blob, dict):
                print(f"!! model is not a dict ({type(model_blob)})", file=out_fp)
                continue
            for k, v in _flatten(model_blob):
                line = f"  {k}\t{_shape_str(v)}"
                if args.values and isinstance(v, torch.Tensor) and v.numel() < 5e7:
                    try:
                        f = v.float().flatten()
                        line += (f"\tmean={f.mean().item(): .3e}"
                                 f" std={f.std().item(): .3e}"
                                 f" min={f.min().item(): .3e}"
                                 f" max={f.max().item(): .3e}")
                    except Exception:
                        pass
                print(line, file=out_fp)
    finally:
        if args.out:
            out_fp.close()
            print(f"[inspect] dump written to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
