#!/usr/bin/env python3
"""Real Qwen3-MoE training workload for the local DeepSpeed adapter.

The pipeline path mirrors the existing Megatron experiment with PP=8 and
EP=8. DeepSpeed does not support PipelineModule with ZeRO-2, so the ZeRO-2
validation uses PP=1 while retaining the same 48-layer model, AutoEP=8, and
Megatron mmap dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import time
import traceback
from functools import partial
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.distributed as torch_dist
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn


_POSITION_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)
    parser.add_argument("--case-name", default="deepspeed-real")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--pipeline-parallel-size", type=int, default=8)
    parser.add_argument("--expert-parallel-size", type=int, default=8)
    parser.add_argument("--zero-stage", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--train-iters", type=int, default=100)
    parser.add_argument("--fault-step", type=int, default=-1)
    parser.add_argument("--fault-rank", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--router-aux-loss-coeff", type=float, default=1.0e-3)
    parser.add_argument("--activation-checkpointing", type=int, choices=(0, 1), default=1)
    parser.add_argument("--log-memory", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.pipeline_parallel_size > 1 and args.zero_stage >= 2:
        parser.error("DeepSpeed PipelineModule is incompatible with ZeRO stage 2")
    if args.train_iters <= 0:
        parser.error("--train-iters must be positive")
    if args.sequence_length <= 1:
        parser.error("--sequence-length must be greater than one")
    if args.micro_batch_size <= 0:
        parser.error("--micro-batch-size must be positive")
    if args.gradient_accumulation_steps <= 0:
        parser.error("--gradient-accumulation-steps must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if not 0 <= args.min_learning_rate <= args.learning_rate:
        parser.error(
            "--min-learning-rate must be between zero and --learning-rate"
        )
    return args


def log(message: str, *, rank: int | None = None) -> None:
    if rank is None:
        rank = int(os.environ.get("RANK", "-1"))
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[deepspeed-real][{stamp}][rank={rank}] {message}", flush=True)


def write_fatal_artifact(exc: BaseException) -> None:
    state_dir = None
    for index, argument in enumerate(sys.argv):
        if argument == "--state-dir" and index + 1 < len(sys.argv):
            state_dir = sys.argv[index + 1]
            break
        if argument.startswith("--state-dir="):
            state_dir = argument.split("=", 1)[1]
            break
    if not state_dir:
        return

    rank = int(os.environ.get("RANK", "-1"))
    epoch = int(
        os.environ.get(
            "MOEGAMBIT_RECOVERY_EPOCH",
            os.environ.get("TORCHELASTIC_RESTART_COUNT", "0"),
        )
    )
    error_dir = Path(state_dir) / "errors"
    error_dir.mkdir(parents=True, exist_ok=True)
    destination = error_dir / f"epoch_{epoch}_rank_{rank}.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "epoch": epoch,
                "error": f"{type(exc).__name__}: {exc}",
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "rank": rank,
                "time": time.time(),
                "traceback": traceback.format_exc(),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def load_qwen_config(path: str, sequence_length: int):
    from transformers import Qwen3MoeConfig

    config = Qwen3MoeConfig.from_pretrained(path, local_files_only=True)
    config.use_cache = False
    config.output_router_logits = True
    config.max_position_embeddings = max(
        int(config.max_position_embeddings), int(sequence_length)
    )
    config._attn_implementation = "sdpa"
    required = {
        "num_hidden_layers": 48,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 768,
    }
    mismatches = {
        name: (getattr(config, name, None), expected)
        for name, expected in required.items()
        if getattr(config, name, None) != expected
    }
    if mismatches:
        raise ValueError(
            "model config no longer matches test_hotspare_replace.sh: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return config


def _rope_theta(config) -> float:
    value = getattr(config, "rope_theta", None)
    if value is not None:
        return float(value)
    parameters = getattr(config, "rope_parameters", None) or {}
    return float(parameters.get("rope_theta", 1_000_000.0))


def position_state(
    hidden_states: torch.Tensor, config
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequence_length = int(hidden_states.shape[1])
    head_dim = int(
        getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
    )
    theta = _rope_theta(config)
    device = hidden_states.device
    key = (
        device.type,
        device.index,
        hidden_states.dtype,
        sequence_length,
        head_dim,
        theta,
    )
    cached = _POSITION_CACHE.get(key)
    if cached is not None:
        return cached

    positions = torch.arange(sequence_length, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
            / head_dim
        )
    )
    frequencies = torch.outer(positions, inv_freq)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    cos = embedding.cos().to(hidden_states.dtype).unsqueeze(0)
    sin = embedding.sin().to(hidden_states.dtype).unsqueeze(0)
    causal_mask = torch.full(
        (sequence_length, sequence_length),
        float("-inf"),
        dtype=hidden_states.dtype,
        device=device,
    )
    causal_mask = torch.triu(causal_mask, diagonal=1).unsqueeze(0).unsqueeze(0)
    _POSITION_CACHE[key] = (causal_mask, cos, sin)
    return causal_mask, cos, sin


def initialize_pipe_module(module: nn.Module, std: float) -> None:
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeExperts,
        Qwen3MoeRMSNorm,
        Qwen3MoeTopKRouter,
    )

    def initialize(item: nn.Module) -> None:
        if isinstance(item, nn.Linear):
            nn.init.normal_(item.weight, mean=0.0, std=std)
            if item.bias is not None:
                nn.init.zeros_(item.bias)
        elif isinstance(item, nn.Embedding):
            nn.init.normal_(item.weight, mean=0.0, std=std)
        elif isinstance(item, Qwen3MoeExperts):
            nn.init.normal_(item.gate_up_proj, mean=0.0, std=std)
            nn.init.normal_(item.down_proj, mean=0.0, std=std)
        elif isinstance(item, Qwen3MoeTopKRouter):
            nn.init.normal_(item.weight, mean=0.0, std=std)
        elif isinstance(item, Qwen3MoeRMSNorm):
            nn.init.ones_(item.weight)

    module.apply(initialize)


class Qwen3MoeEmbeddingPipe(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        initialize_pipe_module(self, float(config.initializer_range))

    def forward(self, input_ids: torch.Tensor):
        hidden_states = self.embedding(input_ids)
        aux_loss = hidden_states.new_zeros((), dtype=torch.float32)
        return hidden_states, aux_loss


def router_load_balancing_loss(
    router_logits: torch.Tensor, num_experts: int, top_k: int
) -> torch.Tensor:
    routing_weights = torch.softmax(router_logits.float(), dim=-1)
    selected_experts = torch.topk(
        routing_weights, top_k, dim=-1, sorted=False
    ).indices
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts)
    tokens_per_expert = expert_mask.float().mean(dim=0)
    probability_per_expert = routing_weights.mean(dim=0)
    return (
        torch.sum(tokens_per_expert * probability_per_expert.unsqueeze(0))
        * num_experts
    )


def qwen_decoder_pipe_type():
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeDecoderLayer,
    )

    class Qwen3MoeDecoderPipe(Qwen3MoeDecoderLayer):
        def __init__(self, config, layer_idx: int) -> None:
            super().__init__(config, layer_idx)
            self.pipe_config = config
            self.router_aux_loss_coeff = float(
                getattr(config, "_moegambit_router_aux_loss_coeff", 0.0)
            )
            self._captured_router_logits: torch.Tensor | None = None
            initialize_pipe_module(self, float(config.initializer_range))

        def forward(self, inputs):
            hidden_states, aux_loss = inputs
            causal_mask, cos, sin = position_state(
                hidden_states, self.pipe_config
            )

            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_ids=None,
                past_key_values=None,
                use_cache=False,
                position_embeddings=(cos, sin),
            )
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            self._captured_router_logits = None
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

            router_logits = self._captured_router_logits
            if router_logits is None:
                raise RuntimeError(
                    f"router logits were not captured for pipeline layer {self.self_attn.layer_idx}"
                )
            layer_aux = router_load_balancing_loss(
                router_logits,
                int(self.pipe_config.num_experts),
                int(self.pipe_config.num_experts_per_tok),
            )
            aux_loss = aux_loss + (
                layer_aux
                * self.router_aux_loss_coeff
                / int(self.pipe_config.num_hidden_layers)
            )
            return hidden_states, aux_loss

    Qwen3MoeDecoderPipe.__name__ = "Qwen3MoeDecoderPipe"
    Qwen3MoeDecoderPipe.__qualname__ = "Qwen3MoeDecoderPipe"
    return Qwen3MoeDecoderPipe


class Qwen3MoeFinalNormPipe(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            Qwen3MoeRMSNorm,
        )

        self.norm = Qwen3MoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        initialize_pipe_module(self, float(config.initializer_range))

    def forward(self, inputs):
        hidden_states, aux_loss = inputs
        return self.norm(hidden_states), aux_loss


class Qwen3MoeLMHeadPipe(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        initialize_pipe_module(self, float(config.initializer_range))

    def forward(self, inputs):
        hidden_states, aux_loss = inputs
        return self.lm_head(hidden_states), aux_loss


def pipeline_loss(outputs, labels: torch.Tensor) -> torch.Tensor:
    logits, aux_loss = outputs
    language_loss = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
    )
    return language_loss + aux_loss.float()


def install_pipeline_router_hooks(model: nn.Module) -> int:
    decoder_type = qwen_decoder_pipe_type()
    installed = 0
    for module in model.modules():
        if module.__class__.__name__ != decoder_type.__name__:
            continue
        router = getattr(getattr(module, "mlp", None), "router", None)
        gate = getattr(router, "gate", None)
        if gate is None:
            raise RuntimeError(
                "AutoEP did not replace a Qwen3-MoE pipeline layer"
            )

        def capture_router_logits(_gate, _inputs, output, owner=module):
            owner._captured_router_logits = output

        gate.register_forward_hook(capture_router_logits)
        installed += 1
    return installed


def build_pipeline_model(
    config,
    pipeline_parallel_size: int,
    checkpoint: bool,
    world_size: int | None = None,
):
    from deepspeed.pipe import LayerSpec, PipelineModule
    from deepspeed.runtime.pipe.topology import ProcessTopology

    decoder_type = qwen_decoder_pipe_type()
    layers = [LayerSpec(Qwen3MoeEmbeddingPipe, config)]
    layers.extend(
        LayerSpec(decoder_type, config, layer_idx)
        for layer_idx in range(config.num_hidden_layers)
    )
    layers.extend(
        [
            LayerSpec(Qwen3MoeFinalNormPipe, config),
            LayerSpec(Qwen3MoeLMHeadPipe, config),
        ]
    )
    checkpoint_fn = partial(
        torch.utils.checkpoint.checkpoint, use_reentrant=False
    )
    topology = None
    if os.environ.get(
        "MOEGAMBIT_DEEPSPEED_HYBRID_RESTORE", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}:
        if world_size is None or world_size % pipeline_parallel_size:
            raise ValueError(
                "hybrid restore requires a valid world_size divisible by PP"
            )
        topology = ProcessTopology(
            axes=["data", "pipe"],
            dims=[world_size // pipeline_parallel_size, pipeline_parallel_size],
        )
    model = PipelineModule(
        layers=layers,
        num_stages=pipeline_parallel_size,
        topology=topology,
        loss_fn=pipeline_loss,
        partition_method=f"type:{decoder_type.__name__}",
        activation_checkpoint_interval=1 if checkpoint else 0,
        activation_checkpoint_func=checkpoint_fn,
        checkpointable_layers=[decoder_type.__name__],
        seed_layers=True,
        base_seed=1234,
    )
    model.config = config
    return model


def build_full_model(config):
    from transformers import Qwen3MoeForCausalLM

    return Qwen3MoeForCausalLM(config)


class MegatronMMapTokenSource:
    def __init__(self, path_prefix: str, sequence_length: int) -> None:
        from megatron.core.datasets.indexed_dataset import IndexedDataset

        self.dataset = IndexedDataset(path_prefix, multimodal=False, mmap=True)
        self.sequence_length = int(sequence_length)
        if len(self.dataset) == 0:
            raise ValueError(f"empty Megatron dataset: {path_prefix}")

    def sample(self, ordinal: int) -> tuple[torch.Tensor, torch.Tensor]:
        target = self.sequence_length + 1
        tokens: list[np.ndarray] = []
        count = 0
        document = int(ordinal) % len(self.dataset)
        while count < target:
            value = self.dataset[document]
            if isinstance(value, tuple):
                value = value[0]
            array = np.asarray(value, dtype=np.int64)
            if array.size:
                tokens.append(array)
                count += int(array.size)
            document = (document + 1) % len(self.dataset)
        joined = np.concatenate(tokens)[:target]
        input_ids = torch.from_numpy(joined[:-1].copy()).long()
        labels = torch.from_numpy(joined[1:].copy()).long()
        return input_ids, labels


class DeterministicBatchIterator(Iterator):
    def __init__(
        self,
        source: MegatronMMapTokenSource,
        micro_batch_size: int,
        data_parallel_rank: int,
        data_parallel_world_size: int,
        start_micro_batch: int,
    ) -> None:
        self.source = source
        self.micro_batch_size = int(micro_batch_size)
        self.data_parallel_rank = int(data_parallel_rank)
        self.data_parallel_world_size = int(data_parallel_world_size)
        self.micro_batch = int(start_micro_batch)

    def __iter__(self):
        return self

    def __next__(self):
        base = (
            self.micro_batch
            * self.data_parallel_world_size
            * self.micro_batch_size
            + self.data_parallel_rank * self.micro_batch_size
        )
        samples = [
            self.source.sample(base + offset)
            for offset in range(self.micro_batch_size)
        ]
        self.micro_batch += 1
        inputs = torch.stack([sample[0] for sample in samples])
        labels = torch.stack([sample[1] for sample in samples])
        return inputs, labels


def resolve_grouped_mm() -> bool:
    setting = os.environ.get(
        "MOEGAMBIT_AUTOEP_GROUPED_MM", "auto"
    ).strip().lower()
    available = callable(getattr(torch, "_grouped_mm", None))
    if setting == "auto":
        return available
    if setting in {"0", "false", "no", "off"}:
        return False
    if setting in {"1", "true", "yes", "on"}:
        if not available:
            raise RuntimeError(
                "MOEGAMBIT_AUTOEP_GROUPED_MM requested grouped GEMM, "
                "but this PyTorch build does not provide torch._grouped_mm"
            )
        return True
    raise ValueError(
        "MOEGAMBIT_AUTOEP_GROUPED_MM must be auto, 0/1, or false/true"
    )


def deepspeed_config(
    args: argparse.Namespace,
    world_size: int,
    use_grouped_mm: bool,
) -> dict:
    data_parallel_size = world_size // args.pipeline_parallel_size
    train_batch_size = (
        args.micro_batch_size
        * data_parallel_size
        * args.gradient_accumulation_steps
    )
    expert_parallel = {
        "enabled": True,
        "autoep_size": args.expert_parallel_size,
        "preset_model": "qwen3_moe",
        "use_grouped_mm": use_grouped_mm,
    }
    if args.pipeline_parallel_size > 1:
        expert_parallel["moe_layer_pattern"] = r"\d+\.mlp"
    zero_optimization = {
        "stage": args.zero_stage,
        "overlap_comm": False,
        "contiguous_gradients": True,
    }
    if args.zero_stage == 2 and os.environ.get("MOEGAMBIT_HOT_SWAP") == "1":
        zero_optimization["elastic_checkpoint"] = True

    return {
        "log_level": os.environ.get(
            "MOEGAMBIT_DEEPSPEED_LOG_LEVEL", "info"
        ),
        "train_micro_batch_size_per_gpu": args.micro_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "train_batch_size": train_batch_size,
        "steps_per_print": 1,
        "gradient_clipping": 1.0,
        "wall_clock_breakdown": True,
        "bf16": {"enabled": True},
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.learning_rate,
                "betas": [0.9, 0.95],
                "eps": 1.0e-8,
                "weight_decay": 0.1,
                "torch_adam": True,
            },
        },
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                "total_num_steps": args.train_iters,
                "warmup_num_steps": args.warmup_steps,
                "warmup_min_ratio": 0.0,
                "cos_min_ratio": args.min_learning_rate / args.learning_rate,
                "warmup_type": "linear",
            },
        },
        "zero_optimization": zero_optimization,
        "pipeline": {
            "activation_checkpoint_interval": (
                1 if args.activation_checkpointing else 0
            ),
            "use_reentrant": False,
        },
        "expert_parallel": expert_parallel,
    }


def maybe_log_memory(rank: int, label: str, enabled: bool) -> None:
    if not enabled:
        return
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    log(
        f"memory label={label} allocated_gib={allocated:.2f} reserved_gib={reserved:.2f}",
        rank=rank,
    )


def notify_hot_spare(kind: str, rank: int, **payload) -> bool:
    from moegambit.runtime.hot_spare import send_worker_event

    return send_worker_event(kind, rank, **payload)


def maybe_inject_fault(
    args: argparse.Namespace, rank: int, global_step: int
) -> None:
    recovery_epoch = int(
        os.environ.get(
            "MOEGAMBIT_RECOVERY_EPOCH",
            os.environ.get("TORCHELASTIC_RESTART_COUNT", "0"),
        )
    )
    if (
        args.fault_step < 0
        or recovery_epoch != 0
        or global_step != args.fault_step
        or rank != args.fault_rank
    ):
        return
    marker = Path(args.state_dir) / "fault_injected.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "rank": rank,
                "step": global_step,
                "pid": os.getpid(),
                "time": time.time(),
                "recovery_epoch": recovery_epoch,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    log(
        f"FAULT_INJECT rank={rank} step={global_step} signal=SIGKILL",
        rank=rank,
    )
    try:
        if notify_hot_spare(
            "rank_failure",
            rank,
            reason="injected_sigkill",
            global_step=global_step,
        ):
            log(
                "FAULT_REPORTED_TO_HOT_SPARE "
                f"logical_node={rank // int(os.environ.get('LOCAL_WORLD_SIZE', '1'))}",
                rank=rank,
            )
    except Exception as exc:
        log(
            f"FAULT_REPORT_FAILED {type(exc).__name__}: {exc}",
            rank=rank,
        )
    os.kill(os.getpid(), signal.SIGKILL)


def main() -> int:
    args = parse_args()
    if args.local_rank < 0:
        args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    model_config_path = Path(args.model_config)
    data_prefix = Path(args.data_path)
    if not model_config_path.exists():
        raise FileNotFoundError(model_config_path)
    if not Path(str(data_prefix) + ".idx").exists():
        raise FileNotFoundError(str(data_prefix) + ".idx")
    if not Path(str(data_prefix) + ".bin").exists():
        raise FileNotFoundError(str(data_prefix) + ".bin")

    import deepspeed

    torch.cuda.set_device(args.local_rank)
    deepspeed.init_distributed(dist_backend="nccl")
    rank = torch_dist.get_rank()
    world_size = torch_dist.get_world_size()
    from moegambit.runtime.hot_spare import report_worker_phase

    report_worker_phase("distributed_ready", rank)
    if world_size % args.pipeline_parallel_size:
        raise ValueError(
            f"world_size={world_size} is not divisible by PP={args.pipeline_parallel_size}"
        )
    if (world_size // args.pipeline_parallel_size) % args.expert_parallel_size:
        raise ValueError(
            "EP must divide world_size / PP: "
            f"world={world_size} PP={args.pipeline_parallel_size} "
            f"EP={args.expert_parallel_size}"
        )

    # Model-parallel and AutoEP ranks must start from the same logical model.
    # PipelineModule adds a deterministic global-layer offset when it builds
    # each local stage.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    config = load_qwen_config(args.model_config, args.sequence_length)
    config._moegambit_router_aux_loss_coeff = args.router_aux_loss_coeff
    use_grouped_mm = resolve_grouped_mm()
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.state_dir).mkdir(parents=True, exist_ok=True)

    log(
        "initializing "
        f"case={args.case_name} world={world_size} "
        f"PP={args.pipeline_parallel_size} EP={args.expert_parallel_size} "
        f"ZeRO={args.zero_stage} recovery_epoch="
        f"{os.environ.get('MOEGAMBIT_RECOVERY_EPOCH', os.environ.get('TORCHELASTIC_RESTART_COUNT', '0'))} "
        f"grouped_mm={use_grouped_mm}",
        rank=rank,
    )

    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    report_worker_phase("model_build_start", rank)
    log("MODEL_BUILD_START", rank=rank)
    try:
        if args.pipeline_parallel_size > 1:
            model = build_pipeline_model(
                config,
                args.pipeline_parallel_size,
                bool(args.activation_checkpointing),
                world_size,
            )
        else:
            model = build_full_model(config)
    finally:
        torch.set_default_dtype(original_dtype)
    log("MODEL_BUILD_DONE", rank=rank)
    report_worker_phase("model_build_done", rank)
    maybe_log_memory(rank, "model-built-before-autoep", args.log_memory)

    report_worker_phase("engine_init_start", rank)
    log("DEEPSPEED_ENGINE_INIT_START", rank=rank)
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=None,
        config=deepspeed_config(args, world_size, use_grouped_mm),
    )
    log("DEEPSPEED_ENGINE_INIT_DONE", rank=rank)
    report_worker_phase("engine_init_done", rank)
    maybe_log_memory(rank, "engine-ready-after-autoep", args.log_memory)

    from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer

    local_autoep_layers = sum(
        isinstance(module, AutoEPMoELayer) for module in engine.module.modules()
    )
    expected_local_layers = (
        config.num_hidden_layers // args.pipeline_parallel_size
        if args.pipeline_parallel_size > 1
        else config.num_hidden_layers
    )
    if local_autoep_layers != expected_local_layers:
        raise RuntimeError(
            f"AutoEP replaced {local_autoep_layers} local layers; "
            f"expected {expected_local_layers}"
        )
    report_worker_phase("pipeline_validation_start", rank)
    if args.pipeline_parallel_size > 1:
        hooks = install_pipeline_router_hooks(engine.module)
        if hooks != expected_local_layers:
            raise RuntimeError(
                f"installed {hooks} router hooks; expected {expected_local_layers}"
            )
        data_parallel_rank = int(engine.grid.data_parallel_id)
        data_parallel_world_size = int(engine.grid.data_parallel_size)
    else:
        data_parallel_rank = rank
        data_parallel_world_size = world_size
    report_worker_phase("pipeline_validation_done", rank)

    report_worker_phase("dataset_init_start", rank)
    source = MegatronMMapTokenSource(args.data_path, args.sequence_length)
    report_worker_phase("dataset_init_done", rank)
    start_micro_batch = (
        int(engine.global_steps) * args.gradient_accumulation_steps
    )
    batches = DeterministicBatchIterator(
        source,
        args.micro_batch_size,
        data_parallel_rank,
        data_parallel_world_size,
        start_micro_batch,
    )

    log(
        f"TRAIN_READY global_step={engine.global_steps} "
        f"local_autoep_layers={local_autoep_layers} "
        f"data_rank={data_parallel_rank}/{data_parallel_world_size}",
        rank=rank,
    )
    if args.local_rank == 0:
        try:
            notify_hot_spare(
                "worker_ready",
                rank,
                global_step=int(engine.global_steps),
            )
        except Exception as exc:
            raise RuntimeError(
                "failed to report TRAIN_READY to hot-spare coordinator"
            ) from exc
    report_worker_phase("train_barrier_start", rank)
    torch_dist.barrier()
    report_worker_phase("train_barrier_done", rank)

    while int(engine.global_steps) < args.train_iters:
        iteration_started = time.monotonic()
        if int(engine.global_steps) == 0:
            report_worker_phase("first_iteration_start", rank)
        if args.pipeline_parallel_size > 1:
            loss = engine.train_batch(data_iter=batches)
        else:
            input_ids, _ = next(batches)
            input_ids = input_ids.to(
                engine.device, non_blocking=True
            )
            outputs = engine(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            engine.backward(loss)
            engine.step()
        step = int(engine.global_steps)
        if step == 1:
            report_worker_phase("first_iteration_done", rank)
        elapsed = time.monotonic() - iteration_started
        if rank == 0:
            loss_value = float(loss.detach().float().mean().cpu())
            log(
                f"iteration {step}/{args.train_iters} "
                f"loss={loss_value:.6f} elapsed_s={elapsed:.3f}",
                rank=rank,
            )
        maybe_inject_fault(args, rank, step)

    runtime = getattr(engine, "_moegambit_runtime", None)
    if runtime is not None and runtime.zero2 is not None:
        for manager in runtime.zero2.managers.values():
            manager.wait_until_replicated(int(engine.global_steps))
    torch_dist.barrier()
    if rank == 0:
        zero2_replication = None
        if runtime is not None and runtime.zero2 is not None:
            zero2_replication = {
                namespace: {
                    "local_replicated_step": manager.local_replicated_step,
                    "peer_committed_step": manager.peer_committed_step,
                }
                for namespace, manager in runtime.zero2.managers.items()
            }
        completion = Path(args.state_dir) / "completed.json"
        completion.write_text(
            json.dumps(
                {
                    "case": args.case_name,
                    "global_step": int(engine.global_steps),
                    "restart_count": int(
                        os.environ.get(
                            "MOEGAMBIT_RECOVERY_EPOCH",
                            os.environ.get(
                                "TORCHELASTIC_RESTART_COUNT", "0"
                            ),
                        )
                    ),
                    "recovery_epoch": int(
                        os.environ.get(
                            "MOEGAMBIT_RECOVERY_EPOCH",
                            os.environ.get(
                                "TORCHELASTIC_RESTART_COUNT", "0"
                            ),
                        )
                    ),
                    "physical_node": int(
                        os.environ.get(
                            "MOEGAMBIT_PHYSICAL_NODE_RANK", "0"
                        )
                    ),
                    "world_size": world_size,
                    "pipeline_parallel_size": args.pipeline_parallel_size,
                    "expert_parallel_size": args.expert_parallel_size,
                    "zero_stage": args.zero_stage,
                    "zero2_replication": zero2_replication,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        log(f"TRAIN_COMPLETE state={completion}", rank=rank)

    if runtime is not None:
        runtime.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        try:
            write_fatal_artifact(exc)
        except Exception as artifact_exc:
            log(
                "FATAL_ARTIFACT_WRITE_FAILED "
                f"{type(artifact_exc).__name__}: {artifact_exc}"
            )
        log(f"FATAL {type(exc).__name__}: {exc}")
        raise
