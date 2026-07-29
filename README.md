# MoEGambit

MoEGambit is a framework-neutral recovery runtime for distributed MoE
training. It provides coordinated rank replacement, deterministic process-group
rebuild, version-aware parameter and optimizer restoration, resident hot-spare
workers, and checkpoint fallback for both Megatron-LM and DeepSpeed.

This repository contains:

- one installable `moegambit` runtime under `src/`;
- a Megatron adapter and compatibility hooks;
- a DeepSpeed adapter and compatibility hooks;
- Generic DDP as a framework-neutral reference implementation;
- real nine-node validation scripts for Megatron and DeepSpeed.

The implementation is experimental. Test recovery on the exact target
PyTorch, CUDA, NCCL, network, and topology before using it for production
training.

## Repository layout

```text
.
├── src/moegambit/
│   ├── core/                  # framework-free recovery decisions and events
│   ├── runtime/               # orchestration, hot spare, watcher client, wire protocol
│   ├── interfaces/            # EngineAdapter launch protocol
│   ├── adapters/megatron/     # Megatron-specific state and topology access
│   ├── control/               # authenticated control plane and frozen recovery plans
│   ├── distributed/           # topology models and c10d compatibility
│   └── replication/           # optimizer memory replication
├── deepspeed_adapter/
│   └── moegambit_deepspeed/   # DeepSpeed engine, ZeRO, group, and checkpoint logic
├── Megatron-LM/               # Megatron Core 0.15.3 source with integration hooks
├── DeepSpeed/                 # DeepSpeed 0.19.3 source with integration hooks
├── examples/
│   ├── megatron/run_hot_spare.sh
│   ├── deepspeed/run_hot_spare.sh
│   └── generic_ddp/
├── elastic_launcher.py        # Megatron compatibility launcher
├── elastic_watcher.py         # Megatron compatibility watcher
├── run_spare_single_rank.sh   # prearmed Megatron replacement worker
├── deepspeed_qwen3_moe_pretrain.py
├── test_hotspare_replace.sh
└── test_deepspeed_hotspare_replace.sh
```

The dependency direction is:

```text
framework hook -> framework adapter -> moegambit interfaces/runtime/core
```

Framework-independent code must not be added back into `Megatron-LM/`,
`DeepSpeed/`, or `deepspeed_adapter/moegambit_deepspeed/`.

## Supported recovery modes

### Megatron

- fixed logical world size with one failed rank moved to a spare node;
- resident survivor processes and CUDA state;
- deterministic Megatron TP/PP/EP/DP process-group rebuild;
- current-step peer transfer for non-expert state;
- checkpoint or prefetched sidecar restore for expert state;
- optional distributed-optimizer host-memory replication;
- fail-closed fallback when topology or state versions disagree.

The supplied validation shape uses 8 active nodes plus 1 spare node, 8 GPUs per
node, PP=8, EP=8, and TP=1.

### DeepSpeed

`TEST_MODE` selects one of the supported validations:

| Mode | Topology | Recovery behavior |
| --- | --- | --- |
| `hot_swap` | PP=8, EP=8, ZeRO-1 | one failed rank is replaced on node 8 |
| `zero2` | PP=1, EP=8, ZeRO-2 | optimizer shards are replicated through D2H/TCP |
| `combined` | PP=1, EP=8, ZeRO-2 | rank replacement plus optimizer replication |
| `all` | sequential | runs `hot_swap`, then `zero2` |

DeepSpeed `PipelineEngine` does not support ZeRO-2/3, so the validation does
not claim support for PP=8 plus ZeRO-2.

## Environment requirements

### Operating system and hardware

- Linux x86_64;
- NVIDIA GPUs with a CUDA-capable PyTorch build;
- NCCL available on every training and spare node;
- homogeneous GPU count per node for the supplied scripts;
- one network interface reachable by every active and spare node;
- shared or identically mounted dataset and checkpoint paths;
- enough host memory for optimizer replicas and prefetched expert state.

The supplied real validation defaults to:

- 9 physical nodes;
- nodes 0-7: 8 active GPUs each;
- node 8: 8 resident spare workers;
- 64 logical training ranks;
- failure at step 17 after a checkpoint at step 10.

### Software

- Python 3.10 or newer;
- CUDA driver and toolkit compatible with the selected PyTorch wheel;
- PyTorch 2.x with `torch.distributed`;
- NCCL and, for the Megatron configuration, Transformer Engine;
- Megatron Core 0.15.3 from this repository;
- DeepSpeed 0.19.3 from this repository;
- `transformers>=5.0.0,<6` for the Qwen3-MoE DeepSpeed workload;
- `pytest>=7` for local tests.

The project intentionally does not pin a CUDA-specific PyTorch wheel. Install
PyTorch from the index recommended for the cluster's CUDA version.

### Network ports

Allow the selected ports between all participating nodes:

| Purpose | Default |
| --- | ---: |
| Megatron rendezvous | `20117` |
| Megatron watcher | `20200` |
| DeepSpeed rendezvous | `20121` |
| DeepSpeed hot-spare coordinator | `MASTER_PORT + 100` |
| optimizer replicas | configured by the runtime, normally starting at `20300` |

Do not advertise `127.0.0.1` for a multi-node run. `MASTER_ADDR`,
`ELASTIC_WATCHER_ADDR`, and any explicit advertise address must be reachable
from every node.

## Installation

### 1. Create an environment

Run on every node:

```bash
cd /path/to/bsr
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 2. Install CUDA PyTorch

Example only; select the command matching the cluster:

```bash
python -m pip install torch --index-url <PYTORCH_CUDA_WHEEL_INDEX>
```

Verify CUDA and distributed support:

```bash
python - <<'PY'
import torch

assert torch.cuda.is_available()
assert torch.distributed.is_available()
print(torch.__version__, torch.version.cuda, torch.cuda.device_count())
PY
```

### 3. Install MoEGambit

```bash
python -m pip install -e '.[dev]'
```

This installs the common runtime and these commands:

- `moegambit-launch`
- `moegambit-watch`
- `moegambit-watcher`
- `moegambit-doctor`

### 4. Install framework dependencies

For Megatron:

```bash
python -m pip install -e ./Megatron-LM
python -m pip install transformer-engine[pytorch]
```

For DeepSpeed:

```bash
python -m pip install -r deepspeed_requirements.txt
python -m pip install -e ./DeepSpeed
```

Repository scripts also set the source paths explicitly, so local DeepSpeed
and Megatron modifications take precedence over an unrelated site-package:

```bash
export PYTHONPATH="$PWD/src:$PWD/deepspeed_adapter:$PWD/DeepSpeed:$PWD/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"
```

### 5. Verify the installation

```bash
python - <<'PY'
import moegambit
import moegambit_deepspeed
from moegambit.runtime.discovery import discover_adapters

print(moegambit.__file__)
print(moegambit_deepspeed.__file__)
print(sorted(discover_adapters()))
PY

moegambit-doctor
python -m pytest tests -q
```

`moegambit` must resolve from `src/moegambit`; there must not be another
top-level `moegambit` package under a framework adapter.

## Dataset, tokenizer, and storage

The scripts currently retain the validated internal paths:

```text
dataset:
  /mnt/ais-c1/dataset/zds/bigdata/my_qwen3_data_text_document

Megatron checkpoint:
  /mnt/ais-c1/dataset/zds/731hotspare/test_replace_ckpt

Megatron logs:
  /mnt/ais-c1/dataset/zds/log/test_replace

DeepSpeed run root:
  /mnt/ais-c1/dataset/zds/89hotspare/deepspeed_real
```

The Megatron mmap dataset requires both:

```text
${DATA_PATH}.idx
${DATA_PATH}.bin
```

The default tokenizer/model configuration is `./tokenizer`. Override any path
with `DATA_PATH`, `TOKENIZER_DIR`, `MODEL_CONFIG`, `CKPT_DIR`,
`TRAIN_LOG_DIR`, or `RUN_ROOT`.

## Megatron launch

Use [examples/megatron/run_hot_spare.sh](examples/megatron/run_hot_spare.sh).
Run the same command on all nine nodes and change only `NODE_RANK`.

Active node example:

```bash
cd /path/to/bsr
source .venv/bin/activate

export NODE_RANK=0
export MASTER_ADDR=<node-0-routable-ip>
export ELASTIC_WATCHER_ADDR=<node-8-routable-ip>

bash examples/megatron/run_hot_spare.sh
```

Repeat on active nodes with `NODE_RANK=1` through `7`.

Spare node:

```bash
cd /path/to/bsr
source .venv/bin/activate

export NODE_RANK=8
export MASTER_ADDR=<node-0-routable-ip>
export ELASTIC_WATCHER_ADDR=<node-8-routable-ip>

bash examples/megatron/run_hot_spare.sh
```

Important Megatron overrides:

```bash
export FAULT_INJECT_STEP=17
export FAULT_INJECT_NODE=0
export FAULT_INJECT_LOCAL_RANK=1
export TRAIN_ITERS=100
export SAVE_INTERVAL=10
export DATA_PATH=/mnt/ais-c1/dataset/zds/bigdata/my_qwen3_data_text_document
export CKPT_DIR=/mnt/ais-c1/dataset/zds/731hotspare/test_replace_ckpt
```

Dry-run the active and spare commands without starting distributed workers:

```bash
DRY_RUN=1 NODE_RANK=0 \
  MASTER_ADDR=10.0.0.1 ELASTIC_WATCHER_ADDR=10.0.0.9 \
  bash examples/megatron/run_hot_spare.sh

DRY_RUN=1 NODE_RANK=8 \
  MASTER_ADDR=10.0.0.1 ELASTIC_WATCHER_ADDR=10.0.0.9 \
  bash examples/megatron/run_hot_spare.sh
```

## DeepSpeed launch

Use [examples/deepspeed/run_hot_spare.sh](examples/deepspeed/run_hot_spare.sh).
Run the same command on all nine nodes and change only `NODE_RANK`.

Active node example:

```bash
cd /path/to/bsr
source .venv/bin/activate

export NODE_RANK=0
export MASTER_ADDR=<node-0-routable-ip>
export ELASTIC_WATCHER_ADDR=<node-8-routable-ip>
export TEST_MODE=hot_swap

bash examples/deepspeed/run_hot_spare.sh
```

Repeat with `NODE_RANK=1` through `7`.

Spare node:

```bash
cd /path/to/bsr
source .venv/bin/activate

export NODE_RANK=8
export MASTER_ADDR=<node-0-routable-ip>
export ELASTIC_WATCHER_ADDR=<node-8-routable-ip>
export TEST_MODE=hot_swap

bash examples/deepspeed/run_hot_spare.sh
```

Run ZeRO-2 replication without a spare activation:

```bash
TEST_MODE=zero2 bash examples/deepspeed/run_hot_spare.sh
```

Run rank replacement and ZeRO-2 together:

```bash
TEST_MODE=combined bash examples/deepspeed/run_hot_spare.sh
```

Dry-run all generated commands:

```bash
DRY_RUN=1 TEST_MODE=all NODE_RANK=0 \
  MASTER_ADDR=10.0.0.1 ELASTIC_WATCHER_ADDR=10.0.0.9 \
  bash examples/deepspeed/run_hot_spare.sh

DRY_RUN=1 TEST_MODE=all NODE_RANK=8 \
  MASTER_ADDR=10.0.0.1 ELASTIC_WATCHER_ADDR=10.0.0.9 \
  bash examples/deepspeed/run_hot_spare.sh
```

For multi-NIC hosts, explicitly set:

```bash
export MOEGAMBIT_HOT_SPARE_ADVERTISE_ADDR=<this-node-routable-ip>
export MOEGAMBIT_REPLICA_ADVERTISE_ADDR=<this-node-routable-ip>
```

## Generic DDP example

The Generic DDP example proves that the common recovery contract is not a
Megatron- or DeepSpeed-specific API:

```bash
torchrun --standalone --nproc-per-node=2 \
  examples/generic_ddp/train_loop.py \
  --steps 8 \
  --checkpoint-dir /tmp/moegambit-generic-ddp
```

See [examples/generic_ddp/README.md](examples/generic_ddp/README.md) for the
fault-replacement flow.

## Runtime CLI

Prepare a DeepSpeed launch without executing it:

```bash
moegambit-launch \
  --adapter deepspeed \
  --zero2 \
  --dry-run \
  -- python train.py
```

Run the authenticated common watcher:

```bash
moegambit-watcher \
  --rendezvous-host <rank-0-ip> \
  --rendezvous-port 20400 \
  --bind-host <watcher-ip> \
  --port 20200
```

The real Megatron and DeepSpeed scripts use compatibility coordinators because
their already validated recovery sequences include framework-specific
safe-point and resident-worker behavior.

## Success criteria

A successful rank replacement must show:

- the fault marker was written at the configured step;
- survivor Python processes were not restarted;
- the replacement retained the failed logical rank;
- all process groups rebuilt with the same manifest;
- `resume_step` equals the failure step;
- `rollback_steps=0` for the in-process hybrid path;
- the first complete post-recovery iteration committed;
- final `global_step` equals `TRAIN_ITERS`.

DeepSpeed additionally writes `completed.json` under the selected run state
directory and validates optimizer replica versions when ZeRO-2 is enabled.

## Troubleshooting

### Watcher is unreachable

Check routing and firewall rules:

```bash
nc -vz <watcher-ip> 20200
nc -vz <rank-0-ip> 20117
```

Do not use loopback addresses for multi-node runs.

### NCCL aborts before Python handles the failure

The Megatron script sets:

```bash
TORCH_NCCL_ASYNC_ERROR_HANDLING=0
TORCH_NCCL_ENABLE_MONITORING=0
```

These values are recovery-path requirements for the validated environment.
Revalidate them when changing PyTorch or NCCL.

### Recovery hangs during process-group rebuild

- verify every survivor reached the same safe point;
- verify all nodes use identical code and environment;
- verify group creation order and timeout logs;
- set `NCCL_DEBUG=INFO` and `NCCL_DEBUG_SUBSYS=INIT,NET,ENV`;
- ensure the replacement advertises a routable IPv4 address.

### Host memory is exhausted

Reduce optimizer replica slots or disable expert prefetch:

```bash
export MOEGAMBIT_ZERO2_BUFFER_SLOTS=1
export MOEGAMBIT_STANDBY_PREFETCH_MAX_GIB=64
export MOEGAMBIT_STANDBY_PACKED_EXPERT_CACHE=0
```

## Development checks

```bash
bash -n \
  examples/megatron/run_hot_spare.sh \
  examples/deepspeed/run_hot_spare.sh \
  test_hotspare_replace.sh \
  test_deepspeed_hotspare_replace.sh

python -m compileall -q src deepspeed_adapter
python -m pytest tests -q
git diff --check
```

The CPU/local tests validate contracts, control-plane behavior, script
generation, and supervisor state machines. They do not replace a real
multi-node CUDA recovery run.

## Security

The control plane is intended for a trusted training network. Use a job token,
restrict watcher ports with firewall rules, and do not expose the watcher to
the public Internet. See the configuration in `moegambit.config.SecurityConfig`.

## License and third-party code

MoEGambit runtime code is provided under the root [LICENSE](LICENSE). Vendored
Megatron-LM and DeepSpeed sources retain their original licenses and notices.
Review [LEGAL.md](LEGAL.md), `Megatron-LM/LICENSE`, and
`DeepSpeed/LICENSE` before redistribution.
