# MoC-System core-mechanism timing port (short experiment)

This is an independently implemented, **physical PEC timing port**, based on
[MoC-System, ASPLOS 2025](https://jyhuang91.github.io/papers/asplos2025-moc-system.pdf),
Sections 3.2, 4, and 5. It is not the authors' implementation. It does not use the
old `MOEGAMBIT_MOC_PEC_EMULATE` overlay, which still writes full checkpoints and
cannot measure PEC's saving-time benefit.

## What actually runs

1. Load step 4000 of the existing bias baseline with native Megatron, including
   BF16 weights, FP32 master weights, and Adam moments. MBS=1, the original
   architecture, dataset and 10k LR schedule are retained.
2. Build an explicit parameter-name/master/Adam mapping; reject unsupported
   grouped experts, distributed optimizers, TP, VPP, and EDP>1 rather than
   silently omit state. Checkpoint non-expert state is sharded across dense DP
   peers at coarse module granularity; replicated weights AND optimizer state
   are saved once. With EDP=1 the expert state already has no replicas to shard.
3. Write one complete seed outside the timed samples, so every expert can be
   recovered even before the sequential PEC schedule has completed a cycle.
4. Compare `full_sync` (N/N), `pec_sync` (16/16), and `pec_2level_async` (32/16).
   Experts are selected through an EP-interleaved sequential schedule staggered
   across layers. Persisted experts are a subset of the CPU snapshot experts.
   Only selected expert files are written; no full-checkpoint write disguised
   as partial saving. CPU persistence uses one writer and up to three in-flight
   snapshot/persist/recovery generations; payloads never alias mutable GPU state.
5. Simulate 1, 2, 3, 4 ranks losing their tensor state, with one repetition
   by default. All ranks' live tensor state is cleared before timing, because
   this is global checkpoint-state recovery. The simulated failed ranks read
   their latest persisted units (including older seed units); healthy ranks
   recover from CPU snapshots. Sharded dense state is broadcast back to peers.
   Verify weights, masters, moments/scalars exactly and compare dense-replica
   SHA256 digests **outside** the timing interval.
6. Run five real optimizer steps after the final recovery, check no skipped or
   nonfinite loss, then produce the complete summary. No training checkpoint is
   created for these smoke steps.

The source tensors stay fixed during the microbenchmark: the measured repeated
rounds are **not** 200-iteration training intervals. `MOC_START_STEP` selects a
200-multiple checkpoint, but no fake 200-step training progress is manufactured.
The five smoke steps prove runnability of the final restored state. They do not
validate stale-expert convergence, eta tolerances, or alpha_run.

## Run on all eight nodes

Execute from the MoEGambit source checkout. Use the
same settings on every node except NODE_RANK. A separate idle set of 64 GPUs is
needed; do not launch over ongoing experiments using the same GPUs or port.

```bash
export BASE_DIR=/shared/moegambit/baseline/seed_1234
export LOG_DIR=/personal/moegambit/moc_timing
export RUN_ID=moc_timing_01
export MOC_CKPT_ROOT=/shared/moegambit/moc_timing_ckpts/$RUN_ID
export MASTER_ADDR=192.0.2.1
export MASTER_PORT=20130
export NNODES=8 NPROC_PER_NODE=8 PP_SIZE=8 EP_SIZE=8
export NODE_RANK=0  # 0..7, distinct on each node
export MOC_START_STEP=4000
export MOC_REPEATS=1 MOC_SMOKE_STEPS=5
export MOC_SNAPSHOT_K=32 MOC_PERSIST_K=16
mkdir -p "$LOG_DIR"
nohup bash examples/moc_system/run_moc_timing_short.sh \
  >> "$LOG_DIR/${RUN_ID}_node${NODE_RANK}.log" 2>&1 &
```

Plan-only inspection creates no experiment artifacts or GPU work:

```bash
DRY_RUN=1 bash examples/moc_system/run_moc_timing_short.sh
```

Use `MOC_REPEATS=1` for the fastest smoke/timing check: three save samples and
twelve restore samples across the three arms, plus five real training steps.
Three repetitions produce nine save samples and 36 restore samples. This is a
5-step training budget; loading, checkpoint I/O, transfers, and exact verification
can still take substantial wall time on this large model. There is no fixed
minutes estimate before measuring the storage bandwidth.

## Outputs and failure evidence

Everything useful stays under `$LOG_DIR/$RUN_ID`:

- `plan.json`: frozen configuration and explicit exclusions.
- `node_0.log` ... `node_7.log`: torchrun stdout/stderr, full tracebacks.
- `ranks/rank_000.jsonl` ... `rank_063.jsonl`: immediate durable local timings,
  GPU/PyTorch info, snapshot/written/read byte counts, failure sets and checks.
- `ranks/smoke.rank_*.json`: five optimizer-step completion records.
- `worker_errors/FAILED.rank_*.json`, NCCL timeout dumps, and
  `orchestrator/FAILED.node_*.json`: errors when Python/torchrun can report them.
- `timing_samples.csv`, `timing_summary.csv`, `timing_summary.json`,
  `COMPLETE.json`: generated only when all 64 ranks and all expected samples
  pass coverage checks. The summary uses MAX across ranks per sample, then
  median/min/max and interpolated descriptive p95 across repetitions. With
  n=3, p95 is descriptive; it is not a reliable tail estimate or confidence bound.

If the final automatic summary is interrupted after all rank files and smoke
files were written, regenerate on CPU from the repository:

```bash
python3 examples/moc_system/moc_timing_summarize.py \
  --result-dir /personal/moegambit/moc_timing/moc_timing_01
```

The launcher locks each node and refuses reuse of an already-started RUN_ID.
If a run fails, keep the logs, use a **new RUN_ID**, and rerun; partial repeats are
not mixed into a new timing experiment. Existing baseline files are read-only.

## Storage and memory

Results must be under `/personal`. Only large temporary state goes to
`MOC_CKPT_ROOT`; it must be disjoint from baseline and results. Preflight probes
both filesystems with fsync and reports tensor bytes, estimated node RAM and
scratch high-water size. Persistent units overwrite their own older versions;
there is no per-training-step checkpoint accumulation.

The seed and current full-state arm can coexist, so peak scratch is approximately
two **nonredundant full-state copies**, plus file/metadata overhead. The estimate
reserves 2.3 copies; actual bytes are recorded, and every write exception fails
the experiment. It may still be hundreds of GB for the original 48-layer model.
CPU RAM must hold recovery state and temporary snapshot/read buffers; preflight
requires 2.5 copies of owned node tensors plus 2 GiB headroom, but allocator and
filesystem overhead can require more. No small surrogate model is substituted.

By default each successful arm removes only its own rank scratch directory;
after all arms the seed is removed. Failed arms retain their remaining files.
Set `MOC_KEEP_SCRATCH=1` to preserve all materialized state for inspection.
`MOC_FSYNC=1` includes file/directory fsync in persistence timings. Set it to 0
only for an explicitly labeled buffered-write study, consistently on all nodes.

## What the paper can report

Report this as **a MoC-inspired core-mechanism timing port on the same model and
cluster**, with actual bytes and the reproduced subset specified. Do not label
it the original MoC-System, or apply its timings to the published implementation.

The full comparator uses the same nonredundant unit format and sharding, so it
isolates PEC and persistence scheduling. It is not native Megatron checkpoint
format, the original MoC baseline, or RQ1's rank-local FullLoad.
Recovery is state restoration within already-running workers. Failure detection,
process recreation, communication-group rebuild, scheduler/RNG/data-iterator
reconstruction, rollback/replay and first-post-recovery-step latency are excluded;
the native baseline's scheduler, RNG and iterator remain intact for the smoke
check. No run-level or 35.6x end-to-end speedup follows from these measurements.

GPU-to-CPU snapshot copying is blocking. Asynchronous **CPU persistence** is
implemented; GPU snapshot overlap with forward/backward is not. Async enqueue
time excludes pending persistence; report persistence and final drain separately.
No synthetic sleeps or fabricated compute overlap are used. Dynamic-K, adaptive
configuration, load-aware expert selection and ZeRO-2 training integration are
not reproduced. OS/storage caches are uncontrolled, with no cold-cache claim.

The old paper's limitation concerning no same-cluster reference implementation
can be updated once this port actually completes, while retaining this scope.
# End-to-end extension

For real training, GPU snapshot overlap, worker/communication-group relaunch and
actual replay, use [`README_e2e.md`](README_e2e.md) and
`examples/moc_system/run_moc_e2e.sh`. The fixed-state suite below remains a component timing
benchmark; its old results are not automatically reclassified as end-to-end.
