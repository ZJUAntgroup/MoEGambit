# Short end-to-end MoC mechanism experiment

Entry point: `examples/moc_system/run_moc_e2e.sh`. This extends the fixed-state timing suite with
real Megatron training, GPU snapshot overlap, actual worker/communication-group
relaunch, restoration of optimizer/scheduler/RNG, data-iterator rebuilding and
actual replay, following the mechanisms in the
[MoC-System paper](https://jyhuang91.github.io/papers/asplos2025-moc-system.pdf).
This is an **independent fixed-K mechanism port**, not the original
authors' Megatron-DeepSpeed/ZeRO-2 artifact. Dynamic-K and physical failure
detection/replacement-node scheduling are excluded. Do not label its results as
the complete original MoC-System implementation.

## Default workload and measured quantities

- Existing native baseline checkpoint: step 4000, read-only.
- Train to 4200; save **one** checkpoint per arm (paper cadence: 200 steps).
- Continue to 4205; wait for global durable completion, then exit all training
  workers normally at a safe boundary. No SIGKILL during NCCL collectives.
- Supervisors stay alive. Discard failed-rank CPU cache (default one rank).
- Launch **new** torchrun workers and new communication groups.
- Restore at 4200; replay the five lost steps; continue to 4215.
- Three arms: native synchronous full checkpoint/restart; sharded synchronous
  PEC (`K=16`); sharded two-level PEC (`K_snapshot=32`, `K_persist=16`). Both PEC
  arms preserve non-expert state; healthy async ranks use the supervisor's CPU
  snapshots and failed ranks read persisted selected experts. Unselected experts
  use their step-4000 native checkpoint state, including FP32 masters and Adam
  moments. All ranks restore scheduler/RNG and consumed-sample counters at 4200.
- Actual training updates: `3 × (200 + 2×5 + 10) = 660`. At 3 s/update, compute
  alone is 33 minutes; real checkpoint I/O, state loading, process/group startup
  and data rebuilding add time. No promised speedup before measured results.

The measured window starts immediately before the first real training step and
ends at the common endpoint. This excludes the **initial** baseline loading, but
includes every **recovery** process/group initialization, baseline/partial-state
loading, snapshot checkpoint operation, CPU-cache transfer, serialization, fsync,
data fingerprints, restoration checks, failure-boundary persistence drain and
replay. A single node-0 monotonic timeline measures the endpoint; component timers
are diagnostics and are never summed to manufacture end-to-end duration.

The safe-boundary failure is deliberately delayed until checkpoint persistence
completes. Its drain time is recorded and included in the window. Thus this
measures a controlled failure after a durable checkpoint, not failures at an
arbitrary instant during persistence. New allocations/replacement machines and
fault detection are not timed. All workers restart; this does not measure
MoEGambit's in-process replacement-rank fast path.

Async snapshots use pinned CPU destinations on a dedicated CUDA stream. Router
bias/other model buffers are captured synchronously because forward/backward can
mutate them. Parameters/FP32 masters/Adam moments overlap the next forward/backward;
the optimizer waits for the copy event before changing them. GPU event timelines
report actual observed overlap; a zero overlap measurement is not rewritten as a
successful speedup. The supervisor cache is a conservative socket-copy port: its
transfer/serialization costs are included. Only one immutable checkpoint
generation is needed here; this does not evaluate sustained triple-buffer backpressure.

## Launch on every node

Use the same source commit, `RUN_ID`, settings and node-0 address on all 8 nodes.
Change only `NODE_RANK`. Use a new `RUN_ID` on retry to prevent mixed timings.

```bash
cd /path/to/MoEGambit
export BASE_DIR=/shared/moegambit/baseline/seed_1234
export LOG_DIR=/personal/moegambit/moc_e2e
export RUN_ID=moc_e2e_01
export MOC_CKPT_ROOT=/shared/moegambit/moc_e2e_ckpts/${RUN_ID}
export MASTER_ADDR=192.0.2.1  # replace with the routable node-0 IP
export MASTER_PORT=20140
export NNODES=8 NPROC_PER_NODE=8
export NODE_RANK=0              # each node: 0..7
mkdir -p "$LOG_DIR"
nohup bash examples/moc_system/run_moc_e2e.sh \
  >> "$LOG_DIR/${RUN_ID}_node${NODE_RANK}.log" 2>&1 &
```

For all four rank counts: `export MOC_FAILURE_COUNTS=1,2,3,4` (2640 updates).
For three repetitions of the default one-rank experiment: `export MOC_REPEATS=3`
(1980 updates). Repeated arms rotate order; OS cache is uncontrolled, so do not
claim cold-storage timing. A single run supplies an observation, not confidence
intervals, quality acceptance or `alpha_run` guarantees.

Optional initial smoke: `MOC_CKPT_INTERVAL=10` with a new run ID (90 updates).
The manifest explicitly marks this as a compressed timing smoke and **not** the
paper's 200-step checkpoint cadence. Default experiments use 200.

## Results, storage and failures

All logs and results are under `$LOG_DIR/$RUN_ID`:

- `e2e_results.csv`: individual measured windows, recovery to first resumed/new
  commit, catching up, replay length, bytes written and actual GPU overlap.
- `e2e_summary.json`, `e2e_report.md`: complete checked results, grouped medians
  and speedups against native full restart.
- `e2e_timing.pdf` / `e2e_timing.svg`: generated if matplotlib is installed.
- `COMPLETE.json`: written only if all scheduled jobs and all 64 rank records
  validate, actual workers restart, replay sample counters/LR match, and actual
  token/label fingerprints match during replay and across comparison arms.
- `<job>/<prefix|resume>/node_N.log`: full stdout/stderr; `worker_errors/`:
  elastic failure/NCCL diagnostics and Python traceback JSON.
- `resources.node_N.start.json` / `resources.node_N.end.json`: per-phase disk
  availability, memory availability and cgroup OOM event counters, where exposed
  by the container. Counters are evidence to inspect, not automatic attribution.
- `orchestrator/FAILED.node_N.json`: supervisor/preflight failure.

Large temporary checkpoints live only under `MOC_CKPT_ROOT`. Each completed job
is validated and its measured result persisted **before** its own scratch is
removed. The original baseline is never modified or deleted. Peak disk use is
one native full checkpoint for the full arm, or one selected-expert generation
for PEC. The final experiment does not retain hundreds of checkpoints. To keep
scratch for debugging, set `MOC_KEEP_SCRATCH=1` (then storage accumulates).

CPU memory preflight checks both host availability and the container's cgroup
limit, reserving headroom for pinned copies, worker serialization, supervisor
cache and recovery copies. Insufficient memory fails explicitly rather than
quietly falling back to a different timing treatment.

To re-summarize complete evidence without GPUs:

```bash
python3 examples/moc_system/moc_e2e_summarize.py "$LOG_DIR/$RUN_ID"
```

CPU tests: `python3 -m pytest tests/moc_system -q`.
Test fixture numbers are synthetic test inputs and never performance results.
