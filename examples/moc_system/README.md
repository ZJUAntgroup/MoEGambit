# MoC-System mechanism reproduction

This directory contains an independent fixed-K port of partial-expert
checkpointing (PEC), based on the
[MoC-System paper (ASPLOS 2025)](https://jyhuang91.github.io/papers/asplos2025-moc-system.pdf).
It writes actual selected-expert state and includes CPU regression tests. It
is not the original authors' implementation or a drop-in production backend.

| Experiment | Entry point from repository root | What is measured |
|---|---|---|
| Component timing | `bash examples/moc_system/run_moc_timing_short.sh` | Physical partial saves and restores on a fixed loaded model; five real smoke updates |
| End-to-end timing | `bash examples/moc_system/run_moc_e2e.sh` | Real training, checkpoint work, new worker/process groups, state restoration and actual replay |

The end-to-end experiment compares `full_sync_native`, `pec_sync`, and
`pec_2level_async`. The default 200-step checkpoint interval, five replayed
updates and ten new updates require **660 actual updates** across the three
arms. The component benchmark needs five updates. See
[end-to-end instructions](README_e2e.md) and
[component timing instructions](README_timing.md) for measurement boundaries,
storage estimates, failure evidence and result schemas.

## Requirements

Use the source checkout and the [Megatron environment](../../README.md#environment-requirements),
including Transformer Engine and the tokenizer dependencies. The supplied
workload requires 64 GPUs, TP=1, PP=8, EP=8, EDP=1, non-grouped
SequentialMLP experts, and standard BF16 Adam rather than a distributed
optimizer. An existing native Megatron checkpoint at a 200-multiple step is
required, with its original model configuration, data order, tokenizer and
10k learning-rate schedule. The default profile is Qwen3-style 48-layer,
hidden-size-2048 MoE with 128 experts and top-k=8. Micro batch size is 1;
sigmoid routing uses Megatron's built-in expert bias with auxiliary loss off.
The original model and dataset are not replaced with a small synthetic workload.

Set `BASE_DIR`, `DATA_PATH`, `TOKENIZER_DIR` and `MASTER_ADDR` to your deployment.
Paths under `/shared/moegambit` and the address `192.0.2.1` in these instructions
are placeholders. `BASE_DIR/ckpt/iter_0004000` is the default starting checkpoint.
The baseline is read-only. The scripts create no files and require no GPUs in
plan-only mode:

```bash
MASTER_ADDR=127.0.0.1 bash examples/moc_system/run_moc_e2e.sh --plan-only
MASTER_ADDR=127.0.0.1 bash examples/moc_system/run_moc_timing_short.sh --plan-only
python -m pytest tests/moc_system -q
```

Every participating node must use the same source commit, run ID, paths,
model/data settings and node-0 address. Set a distinct `NODE_RANK` on each node.
`--train-iters 10000` retains the baseline LR schedule; native Megatron
`--exit-interval` stops each short phase at its committed endpoint. No bsr2
training-loop patch is needed. The scripts disable MoEGambit recovery and the
legacy PEC overlay while running the independent benchmark.

## Results and compatibility

All logs, manifests, CSV/JSON summaries and generated plots stay under
`LOG_DIR/RUN_ID`, with `LOG_DIR` under `/personal`. Large temporary state goes
under `MOC_CKPT_ROOT`, disjoint from the baseline and result directory. A
successful job validates complete rank coverage before cleaning its own
checkpoint scratch. Failed jobs retain diagnostics; retry with a new run ID.

For existing experiment environments, `FSE_CKPT_ROOT` remains an alias for
`MOC_CKPT_ROOT`, and `FSE_MODEL_PROFILE` remains an alias for `MOC_MODEL_PROFILE`.
The canonical `MOC_*` variable takes precedence. CSV/JSON output schemas retain
the original experiment field names for analysis compatibility.

The legacy `MOEGAMBIT_MOC_PEC_EMULATE` module only changes checkpoint-state
selection while writing full checkpoints. It cannot measure partial-saving
benefits. These physical benchmarks are the timing reproduction. Dynamic-K,
the authors' ZeRO-2 integration, physical failure detection and replacement-node
scheduling are outside this port. CPU tests verify state transport, planning and
result integrity; measured GPU performance must come from a completed cluster
run. No performance numbers are generated from test fixtures.
