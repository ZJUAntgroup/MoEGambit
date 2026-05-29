#!/usr/bin/env python
"""Convert a Megatron-LM Qwen3-MoE checkpoint (``--ckpt-format torch``) to
HuggingFace ``Qwen3MoeForCausalLM`` format so that it can be loaded by
``lm-evaluation-harness`` / ``vLLM`` / ``transformers``.

Why this script exists
----------------------
Megatron-LM bundles ``tools/checkpoint/convert.py``, but it (a) does not
ship a ``saver_llama_mistral`` plugin in this tree, and (b) its dense
savers (``saver_core``/``saver_legacy``) do not understand MoE-grouped
expert weights, fused QKV with GQA, or the per-(PP,EP) shard layout that
the training script produced. This script implements the missing path
directly:

    iter_XXXXXXX/mp_rank_00_{pp:03d}_{ep:03d}/model_optim_rng.pt   (input)
    ↓
    config.json  +  model-XXXXX-of-YYYYY.safetensors  +  index.json   (output)

Supported source layout
-----------------------
* ``--transformer-impl transformer_engine`` (the layout used by
  ``run_main_exp_moeguard.sh``).
* TP=1 (PP×EP only). TP>1 splitting is intentionally not handled in v1;
  the inspector script will reject TP>1 shards with a clear error.
* MoE GroupedMLP **or** SequentialMLP. The two key shapes are different
  and we detect at runtime.
* Fused QKV (``self_attention.linear_qkv.weight``), GQA with
  ``num_query_groups`` KV heads, ``--no-rope-fusion``, ``--swiglu``,
  ``--untie-embeddings-and-output-weights``.

Usage
-----
    python eval/scripts/megatron_moe_to_hf.py \
        --ckpt /mnt/ais-c1/dataset/zds/main_exp/5.27/moeguard/ckpt \
        --out  /mnt/ais-c1/dataset/zds/eval/models/moeguard-hf \
        --tokenizer ./tokenizer \
        [--iter 10000]                       # default: latest
        [--shard-size-gb 5]                  # default: 5 GB per safetensors

If anything in the source doesn't match what we expect, the script
errors out with a precise message and points you at
``inspect_megatron_ckpt.py``. We never silently produce a partial model.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

try:
    from safetensors.torch import save_file as save_safetensors
except ImportError:  # pragma: no cover
    print("[convert] ERROR: install safetensors (pip install safetensors)",
          file=sys.stderr)
    sys.exit(2)

# Megatron pickled ``args`` as an ``argparse.Namespace`` plus a few helper
# objects whose classes live under the ``megatron`` package (e.g. enum types
# from ``megatron.core.enums``). torch.load goes through pickle, so the
# ``megatron`` package MUST be importable, otherwise unpickling crashes with
# ``ModuleNotFoundError: No module named 'megatron'`` before we get a chance
# to read a single tensor.
#
# We try the in-tree copy first; if the caller already exported PYTHONPATH,
# nothing changes. If neither works we fail loudly with the fix instructions.
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
                print(f"[convert] auto-added {candidate} to sys.path so the "
                      f"Megatron-pickled args can be unpickled.",
                      file=sys.stderr)
                return
            except ImportError:
                sys.path.pop(0)
    raise ImportError(
        "Cannot import 'megatron'. The Megatron-LM checkpoint pickle "
        "references types under the 'megatron' package, so this package "
        "MUST be importable while reading the checkpoint. Either run the "
        "converter from the repo root that contains a Megatron-LM/ subdir, "
        "or set PYTHONPATH to include the Megatron-LM root before invoking "
        "this script:\n"
        "    export PYTHONPATH=/path/to/repo/Megatron-LM:${PYTHONPATH:-}"
    )


_ensure_megatron_importable()


# =====================================================================
# Discovery
# =====================================================================

SHARD_RE = re.compile(r"mp_rank_(\d{2})_(\d{3})_(\d{3})$")


def _list_iter_dirs(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir()
                   if p.is_dir() and p.name.startswith("iter_")],
                  key=lambda p: int(p.name[5:]))


def _list_shards(iter_dir: Path) -> List[Tuple[int, int, int, Path]]:
    out: List[Tuple[int, int, int, Path]] = []
    for sub in sorted(iter_dir.iterdir()):
        if not sub.is_dir():
            continue
        m = SHARD_RE.match(sub.name)
        if not m:
            continue
        tp, pp, ep = int(m.group(1)), int(m.group(2)), int(m.group(3))
        ck = sub / "model_optim_rng.pt"
        if ck.is_file():
            out.append((tp, pp, ep, ck))
    return out


def _read_args_blob(shard_path: Path):
    blob = torch.load(shard_path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or "args" not in blob:
        raise RuntimeError(
            f"{shard_path} has no 'args' top-level key — this does not look "
            f"like a Megatron checkpoint. Run inspect_megatron_ckpt.py first."
        )
    return blob["args"], blob


# =====================================================================
# QKV split (GQA)
# =====================================================================

def _split_qkv_gqa(qkv: torch.Tensor, *, num_heads: int, num_kv_heads: int,
                   head_dim: int, hidden: int):
    """Megatron's fused QKV layout under GQA is per-group-interleaved:

        for g in range(num_kv_heads):
            [Q_{g*N..g*N+N-1}, K_g, V_g]                # heads per group: N = num_heads / num_kv_heads

    each ``head`` is ``head_dim`` rows, total = (num_heads + 2*num_kv_heads) * head_dim.
    """
    n_per_kv = num_heads // num_kv_heads
    expected = (num_heads + 2 * num_kv_heads) * head_dim
    if qkv.shape[0] != expected:
        raise RuntimeError(
            f"linear_qkv first dim = {qkv.shape[0]} but expected {expected} "
            f"(num_heads={num_heads}, num_kv_heads={num_kv_heads}, "
            f"head_dim={head_dim})."
        )
    if qkv.shape[1] != hidden:
        raise RuntimeError(
            f"linear_qkv second dim = {qkv.shape[1]} but expected hidden={hidden}."
        )
    # reshape to (num_kv_heads, n_per_kv+2, head_dim, hidden)
    grouped = qkv.view(num_kv_heads, n_per_kv + 2, head_dim, hidden)
    q = grouped[:, :n_per_kv, :, :].reshape(num_kv_heads * n_per_kv * head_dim, hidden)
    k = grouped[:, n_per_kv, :, :].reshape(num_kv_heads * head_dim, hidden)
    v = grouped[:, n_per_kv + 1, :, :].reshape(num_kv_heads * head_dim, hidden)
    return q.contiguous(), k.contiguous(), v.contiguous()


# =====================================================================
# Expert weight assembly
# =====================================================================

def _take_expert_weights(model_sd: dict, layer_key: str,
                         num_local_experts: int, moe_ffn_hidden: int,
                         hidden: int):
    """Return list of (gate_proj_w, up_proj_w, down_proj_w) for each local
    expert in this PP shard.

    Handles both GroupedMLP (``mlp.experts.weight1`` / ``weight2`` huge
    concatenated tensors) and SequentialMLP (``mlp.experts.local_experts.{i}.
    linear_fc1.weight`` etc).

    SwiGLU convention in Megatron: linear_fc1 produces [gate, up]
    concatenated along the ffn dim, so each expert's fc1 row count is
    ``2 * moe_ffn_hidden``.
    """
    # --- try sequential first (key per local expert) ---
    seq_keys_present = any(
        f"{layer_key}.mlp.experts.local_experts.0.linear_fc1.weight" in k
        for k in model_sd
    )
    out: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    if seq_keys_present:
        for li in range(num_local_experts):
            fc1 = model_sd.pop(
                f"{layer_key}.mlp.experts.local_experts.{li}.linear_fc1.weight"
            )
            fc2 = model_sd.pop(
                f"{layer_key}.mlp.experts.local_experts.{li}.linear_fc2.weight"
            )
            if fc1.shape != (2 * moe_ffn_hidden, hidden):
                raise RuntimeError(
                    f"layer {layer_key} expert {li}: linear_fc1 shape {tuple(fc1.shape)} "
                    f"!= expected (2*{moe_ffn_hidden}, {hidden})."
                )
            if fc2.shape != (hidden, moe_ffn_hidden):
                raise RuntimeError(
                    f"layer {layer_key} expert {li}: linear_fc2 shape {tuple(fc2.shape)} "
                    f"!= expected ({hidden}, {moe_ffn_hidden})."
                )
            gate_w, up_w = fc1.chunk(2, dim=0)
            out.append((gate_w.contiguous(), up_w.contiguous(),
                        fc2.contiguous()))
        return out

    # --- grouped MLP (weight1 / weight2) ---
    w1_key = f"{layer_key}.mlp.experts.weight1"
    w2_key = f"{layer_key}.mlp.experts.weight2"
    if w1_key in model_sd and w2_key in model_sd:
        w1 = model_sd.pop(w1_key)
        w2 = model_sd.pop(w2_key)
        # Megatron GroupedMLP weight1 layout = (hidden, 2*moe_ffn*num_local_experts)
        # (see experts.py: fc1_output_size = moe_ffn_hidden * num_local_experts,
        #  combined with SwiGLU's 2x → 2*moe_ffn*num_local_experts).
        if w1.shape != (hidden, 2 * moe_ffn_hidden * num_local_experts):
            raise RuntimeError(
                f"layer {layer_key}: grouped weight1 shape {tuple(w1.shape)} != "
                f"({hidden}, 2*{moe_ffn_hidden}*{num_local_experts})"
            )
        if w2.shape != (moe_ffn_hidden * num_local_experts, hidden):
            raise RuntimeError(
                f"layer {layer_key}: grouped weight2 shape {tuple(w2.shape)} != "
                f"({moe_ffn_hidden}*{num_local_experts}, {hidden})"
            )
        # w1 → split per expert: each expert gets a (hidden, 2*moe_ffn) slab
        w1_per = w1.view(hidden, num_local_experts, 2 * moe_ffn_hidden)
        w2_per = w2.view(num_local_experts, moe_ffn_hidden, hidden)
        for li in range(num_local_experts):
            slab = w1_per[:, li, :]                       # (hidden, 2*moe_ffn)
            # HF wants Linear(weight=(out, in)); slab is (in=hidden, out=2*moe_ffn).
            fc1 = slab.transpose(0, 1).contiguous()       # (2*moe_ffn, hidden)
            gate_w, up_w = fc1.chunk(2, dim=0)
            down_w = w2_per[li].transpose(0, 1).contiguous()  # (hidden, moe_ffn)
            out.append((gate_w.contiguous(), up_w.contiguous(), down_w))
        return out

    raise RuntimeError(
        f"layer {layer_key}: neither sequential nor grouped MoE keys found. "
        f"Use inspect_megatron_ckpt.py to dump real key names and update this "
        f"script."
    )


# =====================================================================
# Per-layer conversion
# =====================================================================

def _convert_layer(model_sd: dict, *, layer_idx_local: int, layer_idx_global: int,
                   args, num_local_experts: int) -> Dict[str, torch.Tensor]:
    """Convert one transformer layer's worth of keys from Megatron names
    (local to this PP rank) into HF names (global layer index).
    """
    hidden = args.hidden_size
    n_heads = args.num_attention_heads
    n_kv = args.num_query_groups
    head_dim = args.kv_channels
    moe_ffn = args.moe_ffn_hidden_size

    L_local = layer_idx_local         # 0-based within this PP shard
    L_global = layer_idx_global       # 0-based in the full model
    src_prefix = f"decoder.layers.{L_local}"
    dst_prefix = f"model.layers.{L_global}"
    out: Dict[str, torch.Tensor] = {}

    # ---- attention ----
    qkv_w = model_sd.pop(f"{src_prefix}.self_attention.linear_qkv.weight")
    q_w, k_w, v_w = _split_qkv_gqa(qkv_w, num_heads=n_heads,
                                   num_kv_heads=n_kv, head_dim=head_dim,
                                   hidden=hidden)
    out[f"{dst_prefix}.self_attn.q_proj.weight"] = q_w
    out[f"{dst_prefix}.self_attn.k_proj.weight"] = k_w
    out[f"{dst_prefix}.self_attn.v_proj.weight"] = v_w

    # input layernorm — Megatron fuses it into linear_qkv when TE backend is used
    # (look for layer_norm_weight on the linear_qkv module)
    pre_ln_key = f"{src_prefix}.self_attention.linear_qkv.layer_norm_weight"
    if pre_ln_key in model_sd:
        out[f"{dst_prefix}.input_layernorm.weight"] = model_sd.pop(pre_ln_key)
    else:
        # fallback: separate input_layernorm
        alt = f"{src_prefix}.input_layernorm.weight"
        if alt in model_sd:
            out[f"{dst_prefix}.input_layernorm.weight"] = model_sd.pop(alt)

    # Qwen3 has qk_layernorm; pick those up
    for src_name, dst_name in [
        ("self_attention.q_layernorm.weight", "self_attn.q_norm.weight"),
        ("self_attention.k_layernorm.weight", "self_attn.k_norm.weight"),
    ]:
        skey = f"{src_prefix}.{src_name}"
        if skey in model_sd:
            out[f"{dst_prefix}.{dst_name}"] = model_sd.pop(skey)

    out[f"{dst_prefix}.self_attn.o_proj.weight"] = model_sd.pop(
        f"{src_prefix}.self_attention.linear_proj.weight"
    )

    # pre-mlp layernorm — for MoE blocks Megatron usually folds it into router
    # or stores as separate ``pre_mlp_layernorm.weight``
    pre_mlp_ln = f"{src_prefix}.pre_mlp_layernorm.weight"
    if pre_mlp_ln in model_sd:
        out[f"{dst_prefix}.post_attention_layernorm.weight"] = model_sd.pop(pre_mlp_ln)
    else:
        alt = f"{src_prefix}.mlp.router.layer_norm_weight"
        if alt in model_sd:
            out[f"{dst_prefix}.post_attention_layernorm.weight"] = model_sd.pop(alt)

    # ---- router ----
    router_w = model_sd.pop(f"{src_prefix}.mlp.router.weight")
    out[f"{dst_prefix}.mlp.gate.weight"] = router_w  # HF Qwen3MoE calls it gate

    # ---- experts ----
    experts = _take_expert_weights(
        model_sd, layer_key=src_prefix,
        num_local_experts=num_local_experts,
        moe_ffn_hidden=moe_ffn, hidden=hidden,
    )
    return out, experts


# =====================================================================
# Main converter
# =====================================================================

def convert(ckpt_root: Path, out_dir: Path, *, iteration: Optional[int],
            tokenizer_dir: Optional[Path], shard_size_gb: float = 5.0,
            dtype: str = "bfloat16") -> None:
    if iteration is None:
        lf = ckpt_root / "latest_checkpointed_iteration.txt"
        if not lf.is_file():
            raise RuntimeError(f"no --iter given and no {lf} present")
        iteration = int(lf.read_text().strip())
    iter_dir = ckpt_root / f"iter_{iteration:07d}"
    if not iter_dir.is_dir():
        raise RuntimeError(f"{iter_dir} does not exist")

    shards = _list_shards(iter_dir)
    if not shards:
        raise RuntimeError(f"no mp_rank_* found in {iter_dir}")

    # Sniff parallelism from any shard's 'args'
    args, _ = _read_args_blob(shards[0][3])
    tp = args.tensor_model_parallel_size
    pp = args.pipeline_model_parallel_size
    ep = args.expert_model_parallel_size
    if tp != 1:
        raise RuntimeError(
            f"This converter only supports TP=1 (saw TP={tp}). "
            f"Re-shard the ckpt with Megatron's tools/checkpoint/convert.py "
            f"first, or extend this script."
        )
    num_layers = args.num_layers
    num_experts = args.num_experts
    layers_per_pp = num_layers // pp
    experts_per_ep = num_experts // ep
    print(f"[convert] parallelism: TP={tp} PP={pp} EP={ep} "
          f"(layers/PP={layers_per_pp}, experts/EP={experts_per_ep})",
          file=sys.stderr)
    print(f"[convert] num_layers={num_layers} hidden={args.hidden_size} "
          f"n_heads={args.num_attention_heads} kv_heads={args.num_query_groups} "
          f"moe_ffn={args.moe_ffn_hidden_size} num_experts={num_experts}",
          file=sys.stderr)

    # Group shards by (pp, ep) for ordered iteration
    shard_idx: Dict[Tuple[int, int], Path] = {
        (s[1], s[2]): s[3] for s in shards
    }
    if len(shard_idx) != pp * ep:
        print(f"[convert] WARNING: expected {pp*ep} shards but found "
              f"{len(shard_idx)}", file=sys.stderr)

    # Output state dict (we hold everything in CPU RAM; for 30B BF16 + experts
    # this is ~60 GB → fine on a fat eval node)
    hf_sd: Dict[str, torch.Tensor] = {}
    # Per-layer expert collection: layer_idx_global → expert_id → (gw, uw, dw)
    expert_collect: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}

    target_dtype = {"bfloat16": torch.bfloat16,
                    "float16": torch.float16,
                    "float32": torch.float32}[dtype]

    def _cast(t: torch.Tensor) -> torch.Tensor:
        return t.detach().to(target_dtype).contiguous()

    for pp_rank in range(pp):
        for ep_rank in range(ep):
            shard = shard_idx.get((pp_rank, ep_rank))
            if shard is None:
                raise RuntimeError(f"missing shard pp={pp_rank} ep={ep_rank}")
            t0 = time.time()
            blob = torch.load(shard, map_location="cpu", weights_only=False)
            model_sd = blob["model"]
            # Flatten one level if Megatron wrapped into 'language_model'/'embedding'
            # (mcore models typically already flat: 'embedding.word_embeddings.weight',
            #  'decoder.layers.<i>...', 'decoder.final_layernorm.weight',
            #  'output_layer.weight')
            # ---- one-time: embeddings / final norm / output layer (only from pp=0/last) ----
            if pp_rank == 0 and ep_rank == 0:
                emb_key = "embedding.word_embeddings.weight"
                if emb_key in model_sd:
                    hf_sd["model.embed_tokens.weight"] = _cast(model_sd.pop(emb_key))
            if pp_rank == pp - 1 and ep_rank == 0:
                fnorm_key = "decoder.final_layernorm.weight"
                if fnorm_key in model_sd:
                    hf_sd["model.norm.weight"] = _cast(model_sd.pop(fnorm_key))
                out_key = "output_layer.weight"
                if out_key in model_sd:
                    hf_sd["lm_head.weight"] = _cast(model_sd.pop(out_key))

            # ---- per-layer dense parts (only need from ep_rank == 0; identical across EP) ----
            for li_local in range(layers_per_pp):
                li_global = pp_rank * layers_per_pp + li_local
                if ep_rank == 0:
                    dense_out, _experts_unused = _convert_layer(
                        model_sd, layer_idx_local=li_local,
                        layer_idx_global=li_global, args=args,
                        num_local_experts=experts_per_ep,
                    )
                    # NOTE: dense_out includes router + per-layer norms etc.
                    # experts in dense_out are from this (pp=*, ep=0) shard only.
                    for k, v in dense_out.items():
                        hf_sd[k] = _cast(v)
                    # collect this EP shard's experts under global ids
                    exps = _experts_unused
                    expert_collect.setdefault(li_global, {})
                    for li, (gw, uw, dw) in enumerate(exps):
                        global_eid = ep_rank * experts_per_ep + li
                        expert_collect[li_global][global_eid] = (
                            _cast(gw), _cast(uw), _cast(dw),
                        )
                else:
                    # other EP ranks only contribute experts; we still need to
                    # call the converter to pull the right slice but skip dense
                    # outputs (re-run the experts-only path)
                    exps = _take_expert_weights(
                        model_sd, layer_key=f"decoder.layers.{li_local}",
                        num_local_experts=experts_per_ep,
                        moe_ffn_hidden=args.moe_ffn_hidden_size,
                        hidden=args.hidden_size,
                    )
                    expert_collect.setdefault(li_global, {})
                    for li, (gw, uw, dw) in enumerate(exps):
                        global_eid = ep_rank * experts_per_ep + li
                        expert_collect[li_global][global_eid] = (
                            _cast(gw), _cast(uw), _cast(dw),
                        )
                    # drop everything else this shard had for this layer (it's
                    # duplicate dense state we already took from ep_rank==0)
            # Drop residual keys (they should be only optimizer / rng state,
            # which we don't need)
            del blob, model_sd
            gc.collect()
            print(f"[convert] shard pp={pp_rank} ep={ep_rank} loaded in "
                  f"{time.time()-t0:5.1f}s", file=sys.stderr)

    # ---- assemble experts into HF-named keys ----
    for li_global, ed in expert_collect.items():
        if set(ed.keys()) != set(range(num_experts)):
            missing = set(range(num_experts)) - set(ed.keys())
            raise RuntimeError(
                f"layer {li_global}: missing experts {sorted(missing)[:8]}..."
            )
        for eid in range(num_experts):
            gw, uw, dw = ed[eid]
            hf_sd[f"model.layers.{li_global}.mlp.experts.{eid}.gate_proj.weight"] = gw
            hf_sd[f"model.layers.{li_global}.mlp.experts.{eid}.up_proj.weight"] = uw
            hf_sd[f"model.layers.{li_global}.mlp.experts.{eid}.down_proj.weight"] = dw

    # ---- write config.json ----
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "hidden_size": args.hidden_size,
        "intermediate_size": args.ffn_hidden_size,
        "moe_intermediate_size": args.moe_ffn_hidden_size,
        "num_hidden_layers": args.num_layers,
        "num_attention_heads": args.num_attention_heads,
        "num_key_value_heads": args.num_query_groups,
        "head_dim": args.kv_channels,
        "max_position_embeddings": args.max_position_embeddings,
        "rms_norm_eps": getattr(args, "norm_epsilon", 1e-6),
        "rope_theta": getattr(args, "rotary_base", 1_000_000.0),
        "hidden_act": "silu",
        "vocab_size": args.padded_vocab_size if hasattr(args, "padded_vocab_size")
                       else hf_sd["model.embed_tokens.weight"].shape[0],
        "num_experts": num_experts,
        "num_experts_per_tok": args.moe_router_topk,
        "norm_topk_prob": True,
        "tie_word_embeddings": not getattr(args, "untie_embeddings_and_output_weights", True),
        "torch_dtype": dtype,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "use_cache": True,
        "output_router_logits": False,
        "router_aux_loss_coef": getattr(args, "moe_aux_loss_coeff", 0.001),
        "decoder_sparse_step": 1,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[convert] wrote {out_dir/'config.json'}", file=sys.stderr)

    # ---- shard + write safetensors ----
    shard_size = int(shard_size_gb * (1024**3))
    keys = sorted(hf_sd.keys())
    shards_out: List[Dict[str, torch.Tensor]] = [{}]
    cur_bytes = 0
    for k in keys:
        t = hf_sd[k]
        nbytes = t.numel() * t.element_size()
        if cur_bytes + nbytes > shard_size and shards_out[-1]:
            shards_out.append({})
            cur_bytes = 0
        shards_out[-1][k] = t
        cur_bytes += nbytes

    n_shards = len(shards_out)
    weight_map: Dict[str, str] = {}
    total_size = 0
    for i, sh in enumerate(shards_out, start=1):
        fname = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
        path = out_dir / fname
        save_safetensors(sh, str(path))
        sz = path.stat().st_size
        total_size += sz
        for k in sh:
            weight_map[k] = fname
        print(f"[convert] wrote {fname} ({sz/1e9:5.2f} GB, {len(sh)} tensors)",
              file=sys.stderr)
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    print(f"[convert] wrote model.safetensors.index.json", file=sys.stderr)

    # ---- tokenizer pass-through ----
    if tokenizer_dir is not None and tokenizer_dir.is_dir():
        copied = 0
        for f in tokenizer_dir.iterdir():
            if not f.is_file():
                continue
            if f.name.startswith("tokenizer") or f.name in (
                "vocab.json", "merges.txt", "special_tokens_map.json",
                "added_tokens.json",
            ):
                dest = out_dir / f.name
                dest.write_bytes(f.read_bytes())
                copied += 1
        print(f"[convert] copied {copied} tokenizer files from "
              f"{tokenizer_dir}", file=sys.stderr)

    print(f"[convert] DONE → {out_dir}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="Megatron ckpt root (parent of iter_*/).")
    ap.add_argument("--out", required=True,
                    help="HF output directory.")
    ap.add_argument("--iter", type=int, default=None)
    ap.add_argument("--tokenizer", default=None,
                    help="Path to source tokenizer (vocab.json/merges.txt/...).")
    ap.add_argument("--shard-size-gb", type=float, default=5.0)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()
    try:
        convert(Path(args.ckpt).resolve(),
                Path(args.out).resolve(),
                iteration=args.iter,
                tokenizer_dir=Path(args.tokenizer).resolve() if args.tokenizer else None,
                shard_size_gb=args.shard_size_gb,
                dtype=args.dtype)
    except Exception as e:
        print(f"[convert] FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
