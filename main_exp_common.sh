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
#
# Optional env: RESUME_FROM_ITER (default 0)
#   When >0, events whose scheduled step <= RESUME_FROM_ITER are *dropped* from
#   the emitted plan (but still drive the deterministic RNG state for later
#   events, so rank identities for surviving events are bit-identical to a
#   from-scratch run). This is required when resuming from a checkpoint that
#   is already past some of the planned faults: without this filter the
#   injector would back-fire all historical events at the first iterations
#   after resume (we observed exactly this in 5281.log: a 9200 ckpt was
#   resumed and faults 1--9 were all replayed on steps 9201--9215, which is
#   semantically wrong because those faults are supposed to be 1000+ steps
#   apart and to interact with the running Φ'(t) window).
build_main_plan() {
  python3 - "${PLAN_WORLD_SIZE}" "${PP_SIZE}" "${PLAN_SEED}" "${RESUME_FROM_ITER:-0}" <<'PY'
import random
import sys

world_size      = int(sys.argv[1])
pp_size         = int(sys.argv[2])
seed            = int(sys.argv[3])
resume_from     = int(sys.argv[4])

if pp_size <= 0 or world_size <= 0 or world_size % pp_size != 0:
    raise SystemExit(f"invalid world_size={world_size}, pp_size={pp_size}")
stage_width   = world_size // pp_size
max_per_stage = stage_width - 1                # DP safety: >=1 healthy per stage

# Fixed (step, |F|) schedule — 7 single-GPU + 3 eight-card.
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
skipped_lines = []
event_1_ranks = None
for idx, (step, fcnt, cat) in enumerate(events, start=1):
    # Always run the same RNG sequence so that surviving events keep the
    # bit-identical rank identities they would have in a from-scratch run.
    if idx == 5:
        if event_1_ranks is None:
            raise SystemExit("event #5 needs event #1 ranks but they are missing")
        ranks = list(event_1_ranks)
    else:
        rng = random.Random(seed + step * 1009 + fcnt * 9176 + idx * 31337)
        ranks = sample_ranks(rng, fcnt)
    if idx == 1:
        event_1_ranks = list(ranks)

    stage_hist = [0] * pp_size
    for r in ranks:
        stage_hist[r // stage_width] += 1

    if step <= resume_from:
        skipped_lines.append(
            f"  #{idx:>2}  step={step:>4}  |F|={fcnt:>2}  ranks={ranks}  "
            f"cat={cat}  [SKIPPED: step<=RESUME_FROM_ITER={resume_from}]"
        )
        continue

    plan_parts.append(f"{step}:{','.join(str(r) for r in ranks)}")
    debug_lines.append(
        f"  #{idx:>2}  step={step:>4}  |F|={fcnt:>2}  ranks={ranks}  "
        f"stage_hist={stage_hist}  cat={cat}"
    )

if not plan_parts:
    raise SystemExit(f"no events remain after RESUME_FROM_ITER={resume_from} filter")

print(";".join(plan_parts))
for ln in skipped_lines:
    print(ln, file=sys.stderr)
for ln in debug_lines:
    print(ln, file=sys.stderr)
print(f"  PLAN_SEED={seed}, world_size={world_size}, pp_size={pp_size}, "
      f"stage_width={stage_width}, max_per_stage={max_per_stage}, "
      f"RESUME_FROM_ITER={resume_from}, kept={len(plan_parts)}/{len(events)}", file=sys.stderr)
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
  python3 - "${PLAN_SEED}" "${RESUME_FROM_ITER:-0}" <<'PY'
import sys
seed        = int(sys.argv[1])
resume_from = int(sys.argv[2])
steps = [723, 1188, 2461, 3517, 4309, 5640, 6855, 7912, 8689, 9304]
kept = [(i, s) for i, s in enumerate(steps, 1) if s > resume_from]
if not kept:
    raise SystemExit(f"no events remain after RESUME_FROM_ITER={resume_from} filter")
plan_parts = [f"{s}:0" for _, s in kept]
print(";".join(plan_parts))
for i, s in enumerate(steps, 1):
    if s <= resume_from:
        print(f"  #{i:>2}  step={s:>4}  ranks=[0]  [SKIPPED: step<=RESUME_FROM_ITER={resume_from}]",
              file=sys.stderr)
    else:
        print(f"  #{i:>2}  step={s:>4}  ranks=[0]  (MoC-System: full restart from ckpt)",
              file=sys.stderr)
print(f"  PLAN_SEED={seed} (informational; ranks are deterministic [0]); "
      f"RESUME_FROM_ITER={resume_from}, kept={len(kept)}/{len(steps)}",
      file=sys.stderr)
PY
}
