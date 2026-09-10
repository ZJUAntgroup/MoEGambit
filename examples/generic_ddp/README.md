# Generic DDP conformance example

`train_loop.py` shows the five explicit lifecycle hooks. `fault_replacement.py`
is the stronger test: it kills either logical rank, starts a new process for the
same logical rank, rebuilds c10d and DDP, restores parameters, committed model
buffers, AdamW slots, and optimizer param-group options from a survivor, then
checks that both ranks finish with the same complete training-state digest.

```bash
pip install -e '.[torch]'
python examples/generic_ddp/fault_replacement.py
python examples/generic_ddp/fault_replacement.py --fail-rank 0
```

For NCCL, use a machine with two visible GPUs:

```bash
python examples/generic_ddp/fault_replacement.py --backend nccl
```

The script is a conformance harness, not a production launcher. The production
deployment path uses the watcher and node supervisor to allocate the spare
process and distribute the same frozen `RecoveryPlan`.

## Ordinary synchronized training

```bash
torchrun --nnodes=1 --nproc-per-node=2 \
  --master-addr=127.0.0.1 --master-port=24000 \
  examples/generic_ddp/train_loop.py --steps 8 \
  --checkpoint-dir /tmp/moegambit-ddp --checkpoint-interval 2
```

The loop initializes DDP when launched with rank environment variables, honors
`--steps`, and reports each rank's final model digest. Only rank 0 writes the
checkpoint; all ranks wait for successful publication before advertising it.
Batches are deterministic by seed, logical rank and committed step, so a cold
resume uses the same input stream. Use `--backend nccl` for CUDA; Gloo/CPU is the
default. Ordinary training leaves recovery disabled unless `MOEGAMBIT_ENABLED=1`.

## Watcher and checkpoint relaunch

`train_loop.py` also demonstrates the cold-relaunch contract. It writes a
checkpoint to a temporary file, atomically renames it, and only then calls
`runtime.record_checkpoint(...)`. If in-process recovery fails, the watcher can
therefore issue a directive for a checkpoint that is known to be complete.

Start one watcher (the rendezvous endpoint is used by rebuilt c10d groups):

```bash
MOEGAMBIT_REQUIRE_TOKEN=1 MOEGAMBIT_JOB_TOKEN=secret \
  moegambit-watcher \
  --rendezvous-host 127.0.0.1 --rendezvous-port 24100
```

Then start the node agent. The example consumes the injected checkpoint path
from the environment, so no framework-specific `--checkpoint-arg` is needed:

```bash
MOEGAMBIT_ENABLED=1 \
MOEGAMBIT_FALLBACK_RELAUNCH=1 \
MOEGAMBIT_JOB_ID=ddp-demo \
MOEGAMBIT_ATTEMPT_ID=attempt-0 \
MOEGAMBIT_JOB_TOKEN=secret \
MOEGAMBIT_CHECKPOINT_DIR=/tmp/moegambit-ddp-checkpoints \
MOEGAMBIT_CHECKPOINT_INTERVAL=5 \
  moegambit-launch \
  --nproc-per-node 2 --nnodes 1 --node-rank 0 \
  --master-addr 127.0.0.1 --master-port 24000 \
  --watcher-host 127.0.0.1 --watcher-port 20200 \
  -- python examples/generic_ddp/train_loop.py
```

For a framework that needs CLI restore flags, pass repeated arguments such as
`--checkpoint-arg=--load` and
`--checkpoint-arg={checkpoint_locator}`. They are appended as argv elements;
no shell interpolation is used.
