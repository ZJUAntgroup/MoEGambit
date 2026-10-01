# Validation and deployment

The CPU checks exercise the runtime; GPU recovery still needs a matching Linux
cluster. A passing CPU check does not establish NCCL, Megatron or DeepSpeed MoE
recovery performance.

## Reproducible CPU smoke test

```bash
python -m pip install -e '.[dev,generic-ddp]'
python -m pytest tests -q
python examples/generic_ddp/fault_replacement.py --fail-rank 0
python examples/generic_ddp/fault_replacement.py --fail-rank 1
python tools/check_publication.py
```

The test suite includes real two-process DDP with one checkpoint writer, checks
that both ranks finish with the same model digest, and compares a cold resume
against uninterrupted training. A missing PyTorch installation skips those
checks, so install the `generic-ddp` extra before interpreting a green test result.
That extra includes NumPy for optimizer memory transport.

GitHub Actions runs the suite and both fail-stop cases on Linux with Python
3.10/3.12 and CPU PyTorch 2.7.1. Rank replacement requires DDP's `init_sync=False`
option, which is available from PyTorch 2.7. The workflow limits CPU thread pools
to avoid oversubscription in the two-process examples and retains JUnit results
with failure annotations. The workflow is a configured check; consult its
actual run status before claiming a CI result.

On macOS, select the loopback interface if Gloo cannot resolve the host name:

```bash
GLOO_SOCKET_IFNAME=lo0 python -m pytest tests/test_runtime_examples.py -q
```

## Two-GPU smoke test

With CUDA PyTorch and two visible GPUs:

```bash
torchrun --nnodes=1 --nproc-per-node=2 \
  --master-addr=127.0.0.1 --master-port=24000 \
  examples/generic_ddp/train_loop.py --backend nccl --steps 8 \
  --checkpoint-dir /tmp/moegambit-gpu-smoke --checkpoint-interval 2
python examples/generic_ddp/fault_replacement.py --backend nccl --fail-rank 0
python examples/generic_ddp/fault_replacement.py --backend nccl --fail-rank 1
```

This is a small Generic DDP check, not a reduced Qwen3 model. The supplied
Megatron/DeepSpeed examples retain their documented large-model topology and
require the external mmap dataset. Before a full experiment, validate the
framework build, dataset `.idx`/`.bin`, tokenizer, shared checkpoint directory,
GPU memory and cross-node networking on every active and spare node.

The Megatron experiment intentionally exits if recovery fails: it sets
`ELASTIC_FALLBACK_RELAUNCH=0` and permits no relaunch retries, so checkpoint
restart cannot be mistaken for successful in-process recovery. This is distinct
from the runtime's checkpoint-relaunch deployment contract.

## Source checkout and wheel installation

The independent [MoC-System benchmark](../examples/moc_system/README.md) has CPU
checks under `tests/moc_system`, included in the full CI suite. They exercise
physical partial-state I/O, BF16/master/Adam mapping, immutable supervisor cache
after producer exit, source coverage, bounded phase schedules and strict result
summaries. They do not execute the 64-GPU workload or establish performance.

The bundled framework examples use an editable source checkout. The runtime
wheel does not bundle Megatron-LM, DeepSpeed, datasets or root example scripts.
When installing the wheel with separately installed frameworks, provide
`MOEGAMBIT_REPOSITORY_ROOT` for the bundled examples, or set
`MOEGAMBIT_SPARE_SCRIPT` to your own Megatron replacement script. That script
must run the same workload and consume the watcher-provided rank environment.
The watcher itself runs as a Python module, including from a wheel installation.

## Publication hygiene

Example paths under `/shared/moegambit` and documentation addresses under
`192.0.2.0/24` are placeholders. Replace them locally with routable addresses and
consistent shared mounts. Do not commit datasets, credentials, `.env` files,
run-state databases or training logs. Public upstream notices and author credits
remain part of the vendored source licenses.

`tools/check_publication.py` scans tracked source/resources for common credentials
and private deployment paths. It reports filenames and categories rather than
printing a matched secret. It is a release guard, not a full secret scanner or
an assertion that every historical commit is sanitized.
