#!/usr/bin/env bash
# main_exp_common.sh — shared helpers for the 3-system main experiment.
#
# This file is sourced by:
#   run_main_exp_moeguard.sh   (MoEGuard — full hybrid + two-phase)
#   run_main_exp_mocsystem.sh  (MoC-System emulated via BSR_MOC_PEC_EMULATE=1)
#   run_main_exp_baseline.sh   (Megatron full-ckpt restart, BSR fully disabled)
#
# Defines:
#   build_main_plan          -- emits the canonical 10-fault plan string on stdout
#                               and a human-readable summary on stderr.
#   export_common_runtime    -- exports cluster/NCCL/Megatron runtime env vars.
#   require_torchrun         -- ensures torchrun is on PATH or aliased.
#
# All systems share the same PLAN_SEED, so the rank identities are bit-identical
# across runs. This guarantees apples-to-apples accuracy comparison.

set -uo pipefail

export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export NNODES="${NNODES:-8}"
export PP_SIZE="${PP_SIZE:-8}"
export EP_SIZE="${EP_SIZE:-8}"
export PLAN_WORLD_SIZE="${PLAN_WORLD_SIZE:-$((NPROC_PER_NODE * NNODES))}"
export PLAN_SEED="${PLAN_SEED:-42}"

export_common_runtime() {
  export NCCL_IB_DISABLE=1
  export NCCL_DEBUG=WARN
  export PYTHONPATH="${PYTHONPATH:-}:./Megatron-LM"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export CUDA_DEVICE_MAX_CONNECTIONS=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
  export TORCH_CUDA_ARCH_LIST="9.0"
}

# Canonical 10-fault plan, seeded by PLAN_SEED. Stdout -> machine plan; stderr -> debug.
build_main_plan() {
  python3 - "${PLAN_WORLD_SIZE}" "${PP_SIZE}" "${PLAN_SEED}" <<'PY'
import random
import sys

world_size = int(sys.argv[1])
pp_size    = int(sys.argv[2])
seed       = int(sys.argv[3])

if pp_size <= 0 or world_size <= 0 or world_size % pp_size != 0:
    raise SystemExit(f"invalid world_size={world_size}, pp_size={pp_size}")
stage_width   = world_size // pp_size
max_per_stage = stage_width - 1                # DP safety: >=1 healthy per stage

# Fixed (step, |F|) schedule — 7 single-GPU + 2 eight-card + 1 sixteen-card.
events = [
    ( 723,  1, "single-GPU"),
    (1188,  1, "single-GPU HBM"),
    (2461,  1, "single-GPU (small Δ, Δ_min test)"),
    (3517,  1, "single-GPU SSD/PCIe"),
    (4309,  1, "single-GPU REPEAT (same rank as #1)"),
    (5640,  1, "single-GPU"),
    (6855,  1, "single-GPU"),
    (7912,  8, "8-card burst (DP-safe spread)"),
    (8689,  8, "8-card burst (DP-safe spread, other PP region)"),
    (9304,  8, "8-card burst (downgraded from 16-card; NCCL stability)"),
]

def sample_ranks(rng, fault_count):
    if fault_count >= world_size:
        raise SystemExit("fault_count must leave at least one healthy rank globally")
    if fault_count > pp_size * max_per_stage:
        raise SystemExit(f"fault_count {fault_count} exceeds DP-safe cap "
                         f"{pp_size}*{max_per_stage}={pp_size*max_per_stage}")
    quotas = [0] * pp_size
    stage_order = list(range(pp_size)); rng.shuffle(stage_order)
    remaining = fault_count
    while remaining > 0:
        progressed = False
        for s in stage_order:
            if remaining == 0: break
            if quotas[s] < max_per_stage:
                quotas[s] += 1; remaining -= 1; progressed = True
        if not progressed:
            raise SystemExit("could not assign DP-safe quotas")
    if any(q > max_per_stage for q in quotas):
        raise SystemExit(f"DP safety violated: quotas={quotas}")
    ranks = []
    for stage, quota in enumerate(quotas):
        if quota == 0: continue
        candidates = list(range(stage * stage_width, (stage + 1) * stage_width))
        ranks.extend(sorted(rng.sample(candidates, quota)))
    return sorted(ranks)

plan_parts  = []
debug_lines = []
event_1_ranks = None
for idx, (step, fcnt, cat) in enumerate(events, start=1):
    if idx == 5:
        if event_1_ranks is None:
            raise SystemExit("event #5 needs event #1 ranks but they are missing")
        ranks = list(event_1_ranks)
    else:
        rng = random.Random(seed + step * 1009 + fcnt * 9176 + idx * 31337)
        ranks = sample_ranks(rng, fcnt)
    if idx == 1:
        event_1_ranks = list(ranks)
    plan_parts.append(f"{step}:{','.join(str(r) for r in ranks)}")
    stage_hist = [0] * pp_size
    for r in ranks:
        stage_hist[r // stage_width] += 1
    debug_lines.append(
        f"  #{idx:>2}  step={step:>4}  |F|={fcnt:>2}  ranks={ranks}  "
        f"stage_hist={stage_hist}  cat={cat}"
    )

print(";".join(plan_parts))
for ln in debug_lines:
    print(ln, file=sys.stderr)
print(f"  PLAN_SEED={seed}, world_size={world_size}, pp_size={pp_size}, "
      f"stage_width={stage_width}, max_per_stage={max_per_stage}", file=sys.stderr)
PY
}

# ----------------------------------------------------------------------
# Simplified plan for MoC-System emulation.
# Same 10 step indices as build_main_plan (so loss-trajectory comparisons
# remain step-aligned across systems), but ranks are collapsed to a
# single triggering rank ([0]) because MoC-System's recovery is a full
# CHECKPOINT_RESTART regardless of which / how many ranks fail: every
# fault simply reloads the latest ckpt. Stage-width / DP-safety
# constraints therefore have no semantic effect on this system and are
# omitted; the PEC accuracy overlay (BSR_MOC_PEC_EMULATE=1) still
# rewrites the per-expert load paths to match MoC-System's byte-level
# checkpoint state.
build_mocsystem_plan() {
  python3 - "${PLAN_SEED}" <<'PY'
import sys
seed = int(sys.argv[1])
steps = [723, 1188, 2461, 3517, 4309, 5640, 6855, 7912, 8689, 9304]
plan_parts = [f"{s}:0" for s in steps]
print(";".join(plan_parts))
for i, s in enumerate(steps, 1):
    print(f"  #{i:>2}  step={s:>4}  ranks=[0]  (MoC-System: full restart from ckpt)",
          file=sys.stderr)
print(f"  PLAN_SEED={seed} (informational; ranks are deterministic [0])",
      file=sys.stderr)
PY
}
