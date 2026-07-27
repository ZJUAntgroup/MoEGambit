# Generic DDP conformance example

`train_loop.py` shows the five explicit lifecycle hooks. `fault_replacement.py`
is the stronger test: it kills logical rank 1, starts a new process for the same
logical rank, rebuilds c10d and DDP, restores parameters and AdamW state from a
survivor, then checks that both ranks finish with the same model digest.

```bash
pip install -e '.[torch]'
python examples/generic_ddp/fault_replacement.py
```

For NCCL, use a machine with two visible GPUs:

```bash
python examples/generic_ddp/fault_replacement.py --backend nccl
```

The script is a conformance harness, not a production launcher. The production
deployment path uses the watcher and node supervisor to allocate the spare
process and distribute the same frozen `RecoveryPlan`.
