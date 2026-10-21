<div align="center">

<div style="margin: 20px 0;">
  <a href="docs/assets/moegambit-chess.png">
    <img src="docs/assets/moegambit-chess.png"
         width="420"
         alt="MoEGambit chessboard">
  </a>
</div>

# MoEGambit

### Selective State Repair for Distributed Mixture-of-Experts Training

Framework-neutral hot rank replacement, version-aware state restoration, and
transactional recovery for Megatron-LM and DeepSpeed.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-3776AB?style=flat-square&logo=python&logoColor=white)](pyproject.toml)
[![Megatron Core](https://img.shields.io/badge/Megatron_Core-0.15.3-76B900?style=flat-square)](Megatron-LM)
[![DeepSpeed](https://img.shields.io/badge/DeepSpeed-0.19.3-00539C?style=flat-square)](DeepSpeed)

[![English](https://img.shields.io/badge/🇺🇸_English-1a1a2e?style=for-the-badge)](README.md)
[![简体中文](https://img.shields.io/badge/🇨🇳_简体中文-1a1a2e?style=for-the-badge)](README-zh.md)

[Overview](#overview) · [Quick Start](#quick-start) ·
[Architecture](#architecture) · [Examples](#multi-node-examples) ·
[Recovery Contract](#recovery-contract) · [Benchmarks](#benchmarks)

<a href="docs/assets/moegambit-runtime-architecture.png">
  <img src="docs/assets/moegambit-runtime-architecture.png"
       alt="MoEGambit recovery architecture"
       width="100%">
</a>

<sub>Recovery architecture. Click to view the full-resolution image.</sub>

</div>

## Overview

MoEGambit is a selective recovery runtime for distributed MoE and dense training.
It keeps the distributed training job alive after
a fail-stop rank failure, activates a resident replacement worker, rebuilds communication
groups in a deterministic order, and restores state from the safest available
source.

The key idea is hybrid recovery:

- replicated non-expert state is pulled from a healthy peer at the current
  committed version;
- when no live expert replica exists, expert weights and their corresponding
  optimizer state are restored together from a checkpoint;
- live expert replicas and acknowledged host-memory optimizer replicas are used
  where the adapter and version checks support them;
- source and version checks select peer repair or complete-checkpoint relaunch.

MoEGambit separates recovery policy and orchestration from framework-specific
code. Megatron-LM and DeepSpeed retain only lifecycle hooks; their adapters
translate framework objects into one common recovery contract.

### EDP distribution · State-layout schematic

![State layout with EDP=1 versus EDP=2: unique expert shards versus replicated expert groups.](docs/assets/benchmarks/edp_distribution.png)

[View original vector PDF](docs/assets/benchmarks/edp_distribution.pdf)

At EDP=1, each expert shard has no live replica, while non-expert state remains
replicated. At EDP=2, another expert-data-parallel group can supply current
expert state. The main 64-GPU layout has PP=8, EP=8 and ETP=1, giving EDP=1.

## Highlights

- **One recovery runtime:** common controller, watcher protocol, policy,
  process-group orchestration, observability, and CLI under `src/moegambit/`.
- **Two framework adapters:** Megatron-LM and DeepSpeed use the same adapter
  boundary and the same launcher/watcher entry points.
- **Resident hot spares:** survivors keep their Python and CUDA processes while
  a spare assumes the failed logical rank.
- **Version-aware restoration:** model, optimizer, scheduler, RNG, data cursor,
  and checkpoint records are admitted only when their versions are compatible.
- **Phase-aware transactions:** failures in forward, backward, optimizer, and
  checkpoint publication have explicit replay or fallback semantics.
- **Quality-risk admission (opt-in):** the common coordinator can audit or enforce
  the `R <= 1` quality-risk rule with qualified whole-run risk evidence. Existing engine
  compatibility paths retain their legacy policies; see [audit tools](docs/ARTIFACT_AUDIT.md).
- **Commit integrity:** partial optimizer updates require model-and-optimizer
  restoration before replay.

## Quick Start

### Install

Python 3.10 or newer is required. Install a CUDA-enabled PyTorch build that
matches the cluster before installing MoEGambit.

```bash
# From a local checkout or extracted source archive.
cd /path/to/MoEGambit

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Install the PyTorch wheel appropriate for the cluster first.
python -m pip install torch --index-url <PYTORCH_CUDA_WHEEL_INDEX>

# Install MoEGambit and its development checks.
python -m pip install -e '.[dev]'
```

Install the framework required by the workload. The packages below are the
required set for the supplied Megatron and DeepSpeed Qwen3-MoE examples:

```bash
# Megatron-LM
python -m pip install \
  'numpy<2.0.0' 'packaging>=24.2' \
  pybind11 Cython sentencepiece tiktoken
python -m pip install 'transformer-engine[pytorch]'
python -m pip install -e ./Megatron-LM

# DeepSpeed
python -m pip install -r deepspeed_requirements.txt
python -m pip install --upgrade 'transformers>=5.0.0,<6'
python -m pip install -e ./DeepSpeed
```

`deepspeed_requirements.txt` installs `accelerate`, `einops`, `hjson`,
`msgpack`, `ninja`, `numpy`, `packaging`, `psutil`, `py-cpuinfo`, `pydantic`,
`tqdm`, and `transformers`. The explicit upgrade is intentional:
`deepspeed_qwen3_moe_pretrain.py` and the DeepSpeed launch preflight require
`transformers>=5.0.0`. An environment that still contains
`transformers==4.45.0` will be rejected before the distributed job starts.

The repository scripts explicitly prefer the bundled framework sources:

```bash
export PYTHONPATH="$PWD/src:$PWD/DeepSpeed:$PWD/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"
```

### Verify

```bash
python - <<'PY'
import torch
import moegambit
import moegambit.adapters.deepspeed
from moegambit.runtime.discovery import discover_adapters

assert torch.cuda.is_available()
assert torch.distributed.is_available()
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("moegambit:", moegambit.__file__)
print("adapters:", sorted(discover_adapters()))
PY

moegambit-doctor
python -m pytest tests -q
```

The package installs these commands:

| Command | Purpose |
| --- | --- |
| `moegambit-launch` | build or launch an adapter-selected training command |
| `moegambit-watch` | common watcher client entry point |
| `moegambit-watcher` | authenticated framework-neutral coordinator |
| `moegambit-doctor` | validate the local runtime environment |
| `moegambit-elastic-launcher` | compatibility launcher for Megatron/DeepSpeed |
| `moegambit-elastic-watcher` | compatibility watcher for Megatron/DeepSpeed |

## Architecture

```text
framework hook
      │
      ▼
EngineAdapter (Megatron / DeepSpeed / Generic DDP)
      │
      ├── framework lifecycle and state translation
      ▼
moegambit runtime + policy + control plane
      │
      ├── failure classification and recovery planning
      ├── spare assignment and topology manifest
      ├── deterministic process-group rebuild
      ├── peer / replica / checkpoint state selection
      └── commit, rollback, fallback, and observability
```

The dependency direction is strict:

```text
framework hook -> framework adapter -> moegambit interfaces/runtime/core
```

Framework-independent recovery logic belongs in `src/moegambit/`;
framework adapters translate engine state and lifecycle events. Megatron exposes
one public integration contract:

```python
from moegambit.adapters.megatron.hooks import megatron_hooks
```

Megatron publishes framework events and objects through that singleton.
Recovery configuration, failure classification, rollback/replay policy,
process-group rebuild state, and diagnostics live under
`src/moegambit/adapters/megatron/`.

For the detailed design, see
[Unified Recovery Architecture](docs/design/UNIFIED_RECOVERY_ARCHITECTURE.md).

### Repository layout

```text
.
├── src/moegambit/
│   ├── core/                  # framework-free contracts and decisions
│   ├── runtime/               # orchestration, hot spare, watcher client, protocol
│   ├── interfaces/            # EngineAdapter protocol
│   ├── adapters/
│   │   ├── megatron/          # Megatron state/topology integration
│   │   ├── deepspeed/         # DeepSpeed engine/ZeRO integration
│   │   └── generic_ddp/       # framework-neutral reference adapter
│   ├── control/               # authenticated control plane and frozen plans
│   ├── distributed/           # topology and c10d compatibility
│   └── replication/           # optimizer host-memory replication
├── Megatron-LM/               # Megatron Core 0.15.3 + minimal hooks
├── DeepSpeed/                 # DeepSpeed 0.19.3 + minimal hooks
├── examples/
│   ├── megatron/run_hot_spare.sh
│   ├── deepspeed/run_hot_spare.sh
│   └── generic_ddp/
├── elastic_launcher.py        # adapter-aware compatibility launcher
├── elastic_watcher.py         # adapter-aware compatibility watcher
├── test_hotspare_replace.sh
└── test_deepspeed_hotspare_replace.sh
```

### Adapter support

| Capability | Megatron-LM | DeepSpeed | Generic DDP |
| --- | :---: | :---: | :---: |
| Common launcher/watcher dispatch | ✓ | ✓ | reference |
| Resident rank replacement | ✓ | ✓ | worked example |
| Deterministic group rebuild | TP/PP/EP/DP | engine groups | DDP |
| Peer model-state restore | ✓ | ✓ | adapter contract |
| Host optimizer replication | distributed optimizer | ZeRO-2 | adapter contract |
| Checkpoint fallback | ✓ | ✓ | ✓ |
| Phase-aware transaction contract | ✓ | ✓ | ✓ |

## Recovery Contract

For a fail-stop event `<r, t, c>`—failed logical rank `r`, current iteration
`t`, and last checkpoint `c`—the recovery protocol is:

1. Freeze a recovery plan and increment the recovery epoch.
2. Mark the failed rank as `RECOVERING`, block optimizer commit, and discard
   the in-flight iteration.
3. Assign a physical spare while preserving the failed logical rank.
4. Rebuild all process groups from one canonical topology manifest.
5. Select compatible state sources by component and committed version.
6. Restore weights first; attach optimizer state under an update barrier.
7. Rewind the data cursor only when replay is required.
8. Execute one complete post-recovery iteration.
9. Commit the epoch and transition
   `RECOVERING → REPAIRED → BARRIER → HEALTHY`.

The opt-in common-coordinator quality policy admits a candidate hybrid restore
only when a qualified whole-run quality-risk upper bound satisfies `R <= 1`,
with fixed `eta_final=0.005`, `eta_peak=0.01`, and `alpha_run=0.05`.
It validates supported model/telemetry/policy versions and binds evidence to
the exact fault, source snapshot and whole-run history. Missing or unsupported
evidence selects checkpoint fallback (or abort if no complete source exists).

`--policy quality-risk` defaults to audit mode. Enforce mode uses a trusted
risk provider and externally qualified calibration; supply the trained predictor
for the target workload. `MoeHybridPolicy` and engine-specific compatibility
paths keep their existing policies. See
[quality admission and evidence audits](docs/ARTIFACT_AUDIT.md) for policy
selection, calibration requirements, state audits and per-rank completion checks.

Quality telemetry survives training-worker loss through an independent watcher.
Enable `--quality-cpu-features` and connect `AsyncQualityOffloader` at the
committed-step boundary. Immutable CUDA feature snapshots move to pinned CPU
memory and upload in the background; the watcher retains routing, sensitivity,
parameter/moment drift and whole-run history. Snapshot validation selects repair
or checkpoint fallback. See [CPU feature offload](docs/QUALITY_CPU_OFFLOAD.md)
for deployment across failure domains and the worker-exit smoke run.

### Failure boundary semantics

The last committed optimizer version is distinct from the currently executing
iteration:

| Failure phase | Required recovery |
| --- | --- |
| forward, backward, optimizer-before | discard gradients, rewind data, and replay from the last committed step |
| optimizer-during | restore model and optimizer from the previous committed replica before replay |
| optimizer-after, replica not committed | checkpoint relaunch; bookkeeping-only rollback is forbidden |
| committed step | resume from that committed optimizer version |
| checkpoint before/while publishing commit record | ignore the incomplete checkpoint |
| checkpoint after commit record | allow the new checkpoint for restart |

An optimizer step becomes a safe point when its host-memory peer replica
acknowledges the same version. Optimizer-during recovery restores that committed
model-and-optimizer version; failed restoration checks select checkpoint
relaunch or abort.

## Multi-node Examples

The supplied validation topology uses nine homogeneous nodes:

- nodes `0-7`: eight active GPUs each;
- node `8`: eight resident replacement workers;
- 64 logical ranks;
- fault injection at step `17`, after a checkpoint at step `10`.

Run the same script on every node and change only `NODE_RANK`. `MASTER_ADDR`
and `ELASTIC_WATCHER_ADDR` must be routable from every node.

### Megatron-LM

```bash
# Active nodes: run once with NODE_RANK=0, then 1 ... 7.
NODE_RANK=0 \
MASTER_ADDR=<node-0-routable-ip> \
ELASTIC_WATCHER_ADDR=<node-8-routable-ip> \
bash examples/megatron/run_hot_spare.sh

# Spare node.
NODE_RANK=8 \
MASTER_ADDR=<node-0-routable-ip> \
ELASTIC_WATCHER_ADDR=<node-8-routable-ip> \
bash examples/megatron/run_hot_spare.sh
```

The validated Megatron shape is PP=8, EP=8, TP=1. Common overrides:

```bash
export FAULT_INJECT_STEP=17
export FAULT_INJECT_NODE=0
export FAULT_INJECT_LOCAL_RANK=1
export TRAIN_ITERS=100
export SAVE_INTERVAL=10
export DATA_PATH=/shared/moegambit/data/train_text_document
export CKPT_DIR=/shared/moegambit/checkpoints/megatron
```

### DeepSpeed

```bash
# Active nodes: run once with NODE_RANK=0, then 1 ... 7.
NODE_RANK=0 \
MASTER_ADDR=<node-0-routable-ip> \
ELASTIC_WATCHER_ADDR=<node-8-routable-ip> \
TEST_MODE=hot_swap \
bash examples/deepspeed/run_hot_spare.sh

# Spare node.
NODE_RANK=8 \
MASTER_ADDR=<node-0-routable-ip> \
ELASTIC_WATCHER_ADDR=<node-8-routable-ip> \
TEST_MODE=hot_swap \
bash examples/deepspeed/run_hot_spare.sh
```

DeepSpeed validation modes:

| `TEST_MODE` | Topology | Behavior |
| --- | --- | --- |
| `hot_swap` | PP=8, EP=8, ZeRO-1 | replace one failed rank on node 8 |
| `zero2` | PP=1, EP=8, ZeRO-2 | replicate optimizer shards through D2H/TCP |
| `combined` | PP=1, EP=8, ZeRO-2 | rank replacement plus optimizer replication |
| `all` | sequential | run `hot_swap`, then `zero2` |

The supplied DeepSpeed ZeRO-2 profile uses data/expert parallelism.
`PipelineEngine` excludes ZeRO-2/3.

### Dense models

Megatron-LM and DeepSpeed can also recover dense models. Set
`MOEGAMBIT_MODEL_KIND=dense` on every training and watcher/spare node, and use
a model without expert layers. The training adapter rejects a mismatched model
kind. The two self-contained dense examples below use synthetic data; the
Qwen3-MoE validation scripts above remain MoE-specific.

Run each example on two one-GPU training nodes and a one-GPU spare. Install the
dependencies described above, choose a new shared `RUN_ROOT`, and set the two
routable addresses on every node. Start the spare (`NODE_RANK=2`) first, then
training nodes `0` and `1`. Replace the documentation-only addresses below with
your own; no private dataset, tokenizer, or model weights are needed.

```bash
export RUN_ROOT=/shared/runs/dense-example
export MASTER_ADDR=192.0.2.10
export ELASTIC_WATCHER_ADDR=192.0.2.12

# On the spare node:
NODE_RANK=2 bash examples/megatron/run_dense.sh
# On training nodes 0 and 1, respectively:
NODE_RANK=0 bash examples/megatron/run_dense.sh
NODE_RANK=1 bash examples/megatron/run_dense.sh
```

For the DeepSpeed example, choose a separate `RUN_ROOT` and a unique `RUN_ID`
for each run:

```bash
export RUN_ROOT=/shared/runs/dense-deepspeed-example
export RUN_ID=dense-demo-001
export MASTER_ADDR=192.0.2.10
export ELASTIC_WATCHER_ADDR=192.0.2.12

# On the spare node:
NODE_RANK=2 bash examples/deepspeed/run_dense.sh
# On training nodes 0 and 1, respectively:
NODE_RANK=0 bash examples/deepspeed/run_dense.sh
NODE_RANK=1 bash examples/deepspeed/run_dense.sh
```

The script writes `completed.json` and checks the final step,
`mode=rank_in_process_peer`, and zero expert staleness. `DRY_RUN=1` prints the
commands. By default, these synthetic recovery checks inject a fail-stop fault
at committed step 5 after checkpoint step 4; latency and quality measurements
are reported in [Benchmarks](#benchmarks).

For Megatron-LM, run a normal dense `pretrain_gpt.py` configuration: omit
`--num-experts`, set `EP_SIZE=1`, retain `--moe-moegambit-enable` for recovery
hooks, and provide at least two DP replicas of each TP/PP/CP shard. A replacement
loads a checkpoint base and receives all dense parameters, persistent buffers,
and optimizer state from the current-step DP peer. Dense recovery uses the DP-peer state path. A missing peer rejects hot repair;
checkpoint relaunch is possible only when a valid checkpoint exists.

For DeepSpeed, retain `MOEGAMBIT_HOT_SWAP=1`, set `MOEGAMBIT_ZERO2=1` for
ZeRO-2, and configure the same checkpoint settings. Model parameters, buffers,
and RNG come from a live DP peer; ZeRO-1/2 optimizer shards come from the host
replica. Packed-expert checkpoints are unnecessary. The environment strategy
name remains `rank_in_process_hybrid` for protocol compatibility, while the
recovery contract reports `mode=rank_in_process_peer` and
`expert_staleness=0`. Pipeline parallelism with ZeRO-2 remains unsupported.

The dense path passed multi-node GPU failure-recovery tests on the target
cluster. Automated tests cover dense selection, state-source contracts, and
configuration errors. Quantitative latency and quality comparisons
use the MoE configurations listed in [Benchmarks](#benchmarks).

### Dry run

Generate commands without starting distributed workers:

```bash
DRY_RUN=1 NODE_RANK=0 \
MASTER_ADDR=192.0.2.1 ELASTIC_WATCHER_ADDR=192.0.2.9 \
bash examples/megatron/run_hot_spare.sh

DRY_RUN=1 TEST_MODE=all NODE_RANK=8 \
MASTER_ADDR=192.0.2.1 ELASTIC_WATCHER_ADDR=192.0.2.9 \
bash examples/deepspeed/run_hot_spare.sh
```

For multi-NIC hosts:

```bash
export MOEGAMBIT_HOT_SPARE_ADVERTISE_ADDR=<this-node-routable-ip>
export MOEGAMBIT_REPLICA_ADVERTISE_ADDR=<this-node-routable-ip>
```

### Generic DDP

The Generic DDP example implements the recovery contract through a third
adapter:

```bash
torchrun --standalone --nproc-per-node=2 \
  examples/generic_ddp/train_loop.py \
  --steps 8 \
  --checkpoint-dir /tmp/moegambit-generic-ddp
```

See [examples/generic_ddp/README.md](examples/generic_ddp/README.md) for the
fault-replacement walkthrough.

Generic DDP rank replacement requires PyTorch 2.7 or newer and can be installed
with `python -m pip install -e '.[generic-ddp]'`. This requirement applies to DDP
reconstruction; the bundled Megatron experiment's environment is listed below.

### MoC-System comparison benchmark

An independent physical PEC reproduction is included in
[`examples/moc_system`](examples/moc_system/README.md). It provides a five-update
component timing check and a 660-update end-to-end benchmark with real worker
restart, state restoration and replay. Both use the existing native Megatron
baseline, micro batch size 1 and built-in expert bias. Logs/results remain under
`/personal`; completed jobs clean their own temporary checkpoints.

```bash
MASTER_ADDR=127.0.0.1 bash examples/moc_system/run_moc_e2e.sh --plan-only
```

The benchmark independently ports fixed-K PEC checkpointing and recovery to
Megatron. Its instructions distinguish physical partial checkpointing, the
original ZeRO-2 implementation, and the checkpoint-state overlay, and specify
the GPU requirements and measured intervals.

## Benchmarks

The main benchmarks used Qwen3-30B-A3B on
64 NVIDIA H20 GPUs, with built-in expert bias, auxiliary load-balancing loss
disabled, and micro batch size 1. Operational checkpoints were saved every
200 steps. Full Hybrid restored affected expert **weights and optimizer state**;
current replicated non-expert/router state came from healthy peers.

| Result | Reported outcome | Comparison / scope |
| --- | --- | --- |
| Selective restoration + two-phase attachment | **20.6%** lower latency; **55.0%** in the DeepSeek-V2-Lite setting | Replacement-ready latency vs rank-local **FullLoad**, excluding replay |
| Retaining committed progress | **35.6×** recovery speedup at a 100-iteration checkpoint gap | Whole-job **Restart**, including restoration and replay to the same committed progress |
| Full-state quality, 64/128 experts | Largest sampled 500-step peak increase: **0.393%**, below the 1% tolerance | Fixed-validation loss vs whole-job Restart; early/middle/late, load and rank-count cases |
| Completion with one/repeated faults | Maximum final increase **0.00306%**; sampled peak **0.00454%** | Four selected histories through step 10,000; shared seed/prefixes |
| Architecture comparison | Largest sampled peak: **0.0893%** for GQA MoE, **0.0230%** for DeepSeek-style MLA MoE | Six Hybrid branches; 100-step paired windows vs Restart |
| R2 policy audit | **66%** admission; **95%** interception of unsafe candidates; **3/200** actual-run violations | Author-reported independent, frozen-policy audit; one-sided 95% run-risk upper limit **3.83%** |
| Downstream quality | **45.06% / 44.67% / 45.32%** equal-task means | Restart / MoC PEC / MoEGambit; eight zero-shot tasks, with per-task acc/acc_norm metrics |
| Failure-free control path | Mean step-time change **−0.003%**, observed range **−0.09% to +0.07%** across 20 repetitions | Includes device-completion confirmation and all-rank fence; the range is not a confidence interval |
| Web model-update workflow | **1.13–1.53×** faster; **11.3–34.8%** fewer allocated GPU-hours | Six 64-GPU cold-start Hybrid/Restart pairs; R2 admission before Hybrid restoration, checkpoint-simulated state loss |

### Benchmark visualizations

The gallery covers training stability, recovery quality, model configurations
and restoration performance. The recovery architecture appears above; each
result below links to its vector asset and measurement description.

#### Training through ten faults

![Training loss for Restart, MoC PEC reproduction and full-state MoEGambit.](docs/assets/benchmarks/train_loss.png)

[View original vector PDF](docs/assets/benchmarks/train_loss.pdf)

Last-200-step means were **2.7919 / 2.8254 / 2.7910** for Restart / MoC PEC / MoEGambit. Training loss is a stability diagnostic; the quality boundary uses fixed-validation loss.

#### Quality across training stages and affected-rank counts

![Twelve 500-step continuations and three selected step-10000 outcomes relative to whole-job Restart.](docs/assets/benchmarks/quality_checkpoint_study.png)

[View original vector PDF](docs/assets/benchmarks/quality_checkpoint_study.pdf)

Panel (a) covers twelve recovery-window cases; panel (b) reports three selected long continuations. The reference is whole-job Restart, with final/peak tolerances of **0.5% / 1%**.

#### Full Hybrid quality with 64 and 128 experts

![Full Hybrid restores old expert weights and optimizer state: phase, load and rank-count comparisons.](docs/assets/benchmarks/quality_full_state_500.png)

[View original vector PDF](docs/assets/benchmarks/quality_full_state_500.pdf)

Affected experts restored both old weights and optimizer state. The largest sampled 500-step peak was **0.393%**, below the 1% peak tolerance. A recovery-window endpoint is distinct from final training quality.

#### Completion and repeated faults

![Four full-state Hybrid recovery histories through step 10000 and normalized quality outcomes.](docs/assets/benchmarks/quality_full_state_terminal.png)

[View original vector PDF](docs/assets/benchmarks/quality_full_state_terminal.pdf)

Across four selected histories, the largest final increase was **0.00306%**, and sampled peak **0.00454%**. These histories share a seed/prefixes; they are separate from the independent R2 audit.

#### GPU scaling and parallel layouts

![Recovery excluding replay: 64/128 GPUs and four 64-GPU layouts with MoEGambit and two MoC PEC reproduction modes.](docs/assets/benchmarks/recovery_scaling.png)

[View original vector PDF](docs/assets/benchmarks/recovery_scaling.pdf)

All layout ratios use the same **rank-local FullLoad** denominator. A/B have EDP=1 and checkpointed experts; C/D have EDP=2 and can use live expert peers. These restoration ratios are separate from the **35.6×** replay-inclusive whole-job Restart result.

#### GQA and DeepSeek-style MLA MoE

![Small-model full-state recovery: five evaluation offsets, NoFault and whole-job Restart references.](docs/assets/benchmarks/quality_architecture_transfer.png)

[View original vector PDF](docs/assets/benchmarks/quality_architecture_transfer.pdf)

Relative to Restart, largest sampled peaks were **0.0893%** for GQA MoE and **0.0230%** for MLA MoE over the 100-step window. Shared experts, routing and layer placement also differ; this is a configuration comparison.

### Eight downstream tasks

![Eight zero-shot task scores and percentage-point changes vs Restart.](docs/assets/benchmarks/downstream_accuracy.png)

[View vector PDF](docs/assets/benchmarks/downstream_accuracy.pdf)

Equal-task means were **45.06% / 44.67% / 45.32%** for Restart / MoC PEC /
MoEGambit. PEC retained 16 of 128 experts. The plot preserves each task's
`acc`/`acc_norm` metric and shows both improvements and regressions.

### Additional Web experiments · Update pipeline and FiQA quality

#### Cold-start model-update workflow

![Restart and R2-admitted full-state Hybrid request-to-first-HTTP-response times across six 64-GPU cases.](docs/assets/benchmarks/web_update_pipeline.png)

[Vector PDF](docs/assets/benchmarks/web_update_pipeline.pdf) · [SVG](docs/assets/benchmarks/web_update_pipeline.svg)

Across expert ages **50/100/150 steps** and **1/4 affected ranks**, Hybrid
completed 100 new updates plus validation/model loading and the first real HTTP
rerank response in **892–916 s**, compared with Restart's **1,017–1,367 s**.
The measured workflow speedup was **1.13–1.53×**, saving **11.3–34.8%** of
allocated 64-GPU hours. Endpoint validation-loss changes were
**−0.0628% to +0.0355%**. Each case has one measured timing pair.

#### FiQA reranking quality

![FiQA MRR at 10 and nDCG at 10 differences, Hybrid minus Restart, with paired-query 95 percent intervals.](docs/assets/benchmarks/fiqa_reranking_quality.png)

[Vector PDF](docs/assets/benchmarks/fiqa_reranking_quality.pdf) · [SVG](docs/assets/benchmarks/fiqa_reranking_quality.svg)

Two cases used **500 fixed FiQA queries** and BM25 top-20 candidates at offsets
0 and 100 new updates. The plot retains the reported **95% paired-query
bootstrap intervals** (2,000 resamples). All four endpoint intervals cross zero;
the initial age-150/four-rank differences are positive. Top-20 query coverage was
56.6% and mean candidate recall 36.8%. The intervals quantify query-sampling
uncertainty; equivalence testing additionally requires a prespecified task margin.

The two charts show update completion and retrieval quality separately.
Both experiments use **full expert weights/optimizer restoration, cold starts,
checkpoint-simulated state loss, and R2 admission before Hybrid restoration**.
The HTTP probe invokes real inference over loopback. The workflow gains are
distinct from the 35.6× hot recovery result and the independent R2 audit.
The author confirmed the R2-enabled rerun retained these numerical outcomes;
the benchmark bundle records the protocol correction and original exports. See the
[data, protocol and redraw command](examples/benchmarks/README.md#web-update-and-fiqa-experiments-2026-10-20).

### R2 audit · Reported counts

The independent frozen-policy audit reported **3/200** actual-run quality
violations; its one-sided 95% upper limit is **3.83%**. Among admitted
candidates, **3/132** violated the boundary; its upper limit is **5.77%**.
The run-level and admission-conditional denominators differ. Aggregate
verification and provenance are documented in the result bundle.

### Scripts and reproducibility

[**Benchmark plots and result checks**](examples/benchmarks/README.md) maps
each figure to its inputs, script, comparator and measurement scope. It includes
loss-consistency checks, exact risk-bound arithmetic and scripts to redraw the
new quality, architecture and recovery figures:

```bash
python -m pip install matplotlib numpy
export BENCHMARK_RESULTS_DIR=/personal/moegambit/benchmarks
mkdir -p "$BENCHMARK_RESULTS_DIR"
nohup bash examples/benchmarks/run_benchmarks.sh \
  >> "$BENCHMARK_RESULTS_DIR/reproduce.log" 2>&1 &
```

This CPU workflow verifies the supplied exports and writes figures, reports
and stage logs to `$BENCHMARK_RESULTS_DIR`. GPU experiment entry points are the
[MoC component/end-to-end runners](examples/moc_system/README.md),
[Megatron hot replacement](#multi-node-examples) and
[DeepSpeed MoE/dense examples](examples/deepspeed/run_hot_spare.sh).
[State, completion and risk audits](docs/ARTIFACT_AUDIT.md) validate new run
records. The result bundle contains aggregates and redraw scripts; full
checkpoint-splice training and predictor fitting use workload-specific launchers
and predictor packages supplied separately.

## Environment and Configuration

### Environment requirements

- Linux x86_64;
- Python 3.10+;
- NVIDIA GPUs and a compatible CUDA-enabled PyTorch build;
- NCCL available on active and spare nodes;
- Transformer Engine for the supplied Megatron configuration;
- `transformers>=5.0.0,<6` for the Qwen3-MoE DeepSpeed workload;
- shared or identically mounted datasets and checkpoints;
- enough host memory for optimizer replicas and prefetched expert state.

Required dependency sets:

| Component | Required dependencies |
| --- | --- |
| MoEGambit runtime | Python `>=3.10`; PyTorch with `torch.distributed`; NCCL for GPU jobs |
| Megatron-LM 0.15.3 | `torch>=2.6.0`, `numpy<2.0.0`, `packaging>=24.2`, Transformer Engine; `pybind11` and a C++17 compiler for the optional dataset helper |
| Supplied Megatron workload | `sentencepiece`, `tiktoken`, a Hugging Face-compatible tokenizer directory, and CUDA/NCCL |
| DeepSpeed 0.19.3 | `torch>=2.0.0`, `einops`, `hjson`, `msgpack`, `ninja`, `numpy`, `packaging>=20.0`, `psutil`, `py-cpuinfo`, `pydantic>=2.0.0`, `tqdm` |
| Supplied DeepSpeed Qwen3-MoE workload | `transformers>=5.0.0,<6`, `accelerate`, CUDA/NCCL, and the local `./DeepSpeed` installation |
| Development checks | `pytest>=7.0` |

### Validated software stack

The provided environment snapshot corresponds to the following cluster stack:

| Package/runtime | Validated version |
| --- | --- |
| PyTorch | `2.6.0+cu126` |
| CUDA runtime | `12.6.77` |
| NCCL | `2.21.5` |
| NumPy | `1.26.4` |
| Transformer Engine | `2.4.0.dev0+3b411e79` |
| Triton | `3.2.0` |
| Accelerate | `1.10.1` |
| Einops | `0.8.1` |
| HJSON / msgpack | `3.1.0` / `1.1.0` |
| Ninja / pybind11 / Cython | `1.11.1.4` / `2.11.1` / `3.0.12` |
| Packaging / psutil / py-cpuinfo | `24.2` / `7.0.0` / `9.0.0` |
| Pydantic / tqdm | `2.10.3` / `4.67.1` |
| sentencepiece / tiktoken | `0.2.1` / `0.7.0` |

The snapshot records the pre-upgrade environment, including
`transformers==4.45.0` and `deepspeed==0.16.2`. For the supplied runtime, upgrade
Transformers to `>=5.0.0,<6` and install bundled `./DeepSpeed`, which provides
`deepspeed.__version__ == 0.19.3`.

`flash_attn`, `grouped_gemm`, `megablocks`, `torchvision`, and `torchaudio`
appear in the source environment but are not mandatory for the supplied
recovery scripts. Install them only when the selected model or kernel path
requires them.

Use the CUDA 12.6 PyTorch build above to reproduce the supplied environment.
For another cluster, select a mutually compatible PyTorch/CUDA/NCCL stack.

Verify the effective environment after all editable installs:

```bash
python - <<'PY'
from packaging.version import Version
import deepspeed
import numpy
import torch
import transformers

assert Version(torch.__version__.split("+", 1)[0]) >= Version("2.6.0")
assert Version(transformers.__version__) >= Version("5.0.0")
assert Version(transformers.__version__) < Version("6")
assert Version(deepspeed.__version__) >= Version("0.19.3")
assert Version(numpy.__version__) < Version("2.0.0")
assert torch.cuda.is_available()
assert torch.distributed.is_available()
print(
    f"torch={torch.__version__} cuda={torch.version.cuda} "
    f"deepspeed={deepspeed.__version__} "
    f"transformers={transformers.__version__} numpy={numpy.__version__}"
)
PY
```

### Example paths

Configure the following example paths on the same shared mount across active
and spare nodes, pointing to your dataset, tokenizer and output directories:

```text
dataset:
  /shared/moegambit/data/train_text_document

Megatron checkpoint:
  /shared/moegambit/checkpoints/megatron

Megatron logs:
  /shared/moegambit/logs/megatron

DeepSpeed run root:
  /shared/moegambit/runs/deepspeed
```

Override them with `DATA_PATH`, `TOKENIZER_DIR`, `MODEL_CONFIG`, `CKPT_DIR`,
`TRAIN_LOG_DIR`, or `RUN_ROOT`. The Megatron mmap dataset requires both
`${DATA_PATH}.idx` and `${DATA_PATH}.bin`.

### Network ports

| Purpose | Default |
| --- | ---: |
| Megatron rendezvous | `20117` |
| Megatron watcher | `20200` |
| DeepSpeed rendezvous | `20121` |
| DeepSpeed hot-spare coordinator | `MASTER_PORT + 100` |
| optimizer replicas | normally starts at `20300` |

Do not advertise `127.0.0.1` for multi-node jobs.

## Runtime CLI

The root compatibility entry points dispatch by adapter. Omitting `--adapter`
retains historical Megatron behavior:

```bash
python elastic_launcher.py --adapter megatron \
  --nproc-per-node 8 --nnodes 8 --node-rank "${NODE_RANK}" \
  --master-addr "${MASTER_ADDR}" --master-port 20117 \
  -- python Megatron-LM/pretrain_gpt.py ...

python elastic_watcher.py --adapter megatron \
  --port 20200 --training-nnodes 8 --nproc-per-node 8 \
  --master-addr "${MASTER_ADDR}" --master-port 20117
```

For DeepSpeed, the launcher runs on active nodes and the watcher runs the
coordinator plus resident spare agent:

```bash
# Active nodes 0-7
python elastic_launcher.py --adapter deepspeed \
  --coordinator-host "${SPARE_ADDR}" --coordinator-port 20221 \
  --run-id ds-run-001 --training-nodes 8 --spare-node 8 \
  --physical-node "${NODE_RANK}" --local-world-size 8 \
  --base-master-port 20121 --rank-hot-swap \
  -- python -m deepspeed.launcher.runner ...

# Spare node 8
python elastic_watcher.py --adapter deepspeed \
  --coordinator-host "${SPARE_ADDR}" --coordinator-port 20221 \
  --listen-host 0.0.0.0 --run-id ds-run-001 \
  --training-nodes 8 --spare-node 8 --physical-node 8 \
  --local-world-size 8 --base-master-port 20121 --rank-hot-swap \
  -- python -m deepspeed.launcher.runner ...
```

## Validation and Success Criteria

A successful rank replacement must show:

- a fault marker at the configured step;
- survivor Python processes were not restarted;
- the replacement retained the failed logical rank;
- every process group rebuilt from the same manifest;
- recovery resumed at the contract-selected committed version;
- the first complete post-recovery iteration committed;
- the final `global_step` equals `TRAIN_ITERS`.

DeepSpeed also writes `completed.json` below the selected run-state directory
and validates optimizer replica versions when ZeRO-2 is enabled.

Development checks:

```bash
bash -n \
  examples/megatron/run_hot_spare.sh \
  examples/megatron/run_dense.sh \
  examples/deepspeed/run_hot_spare.sh \
  examples/deepspeed/run_dense.sh \
  test_hotspare_replace.sh \
  test_deepspeed_hotspare_replace.sh

python -m py_compile examples/deepspeed/dense_workload.py
python -m compileall -q src
python -m pytest tests -q
git diff --check
```

[Validation and deployment](docs/VALIDATION.md) provides CPU contract checks,
multi-node CUDA/NCCL smoke tests, checkpoint semantics and wheel deployment.
Validate each target cluster with its GPU recovery workflow.

## Troubleshooting

<details>
<summary><strong>Watcher is unreachable</strong></summary>

```bash
nc -vz <watcher-ip> 20200
nc -vz <rank-0-ip> 20117
```

Check routing and firewall policy; never use loopback addresses between nodes.
</details>

<details>
<summary><strong>NCCL aborts before recovery handles the failure</strong></summary>

The Megatron validation script uses:

```bash
export TORCH_NCCL_ASYNC_ERROR_HANDLING=0
export TORCH_NCCL_ENABLE_MONITORING=0
```

Revalidate these recovery-path settings when changing PyTorch or NCCL.
</details>

<details>
<summary><strong>Recovery hangs during group rebuild</strong></summary>

- confirm every survivor reached the same safe point;
- confirm every node uses identical code and environment;
- inspect group creation order and timeout logs;
- set `NCCL_DEBUG=INFO` and `NCCL_DEBUG_SUBSYS=INIT,NET,ENV`;
- confirm the replacement advertises a routable IPv4 address.
</details>

<details>
<summary><strong>Host memory is exhausted</strong></summary>

```bash
export MOEGAMBIT_ZERO2_BUFFER_SLOTS=1
export MOEGAMBIT_STANDBY_PREFETCH_MAX_GIB=64
export MOEGAMBIT_STANDBY_PACKED_EXPERT_CACHE=0
```
</details>

## Security

The control plane is designed for a trusted training network. Use a per-job
token, restrict watcher and replica ports with firewall rules, and never expose
the watcher directly to the public Internet. See
`moegambit.config.SecurityConfig`.

## License

MoEGambit-owned runtime code is licensed under the root
[Apache License 2.0](LICENSE). Vendored Megatron-LM and DeepSpeed sources retain
their upstream licenses and notices. Review [LEGAL.md](LEGAL.md),
[Megatron-LM/LICENSE](Megatron-LM/LICENSE), and
[DeepSpeed/LICENSE](DeepSpeed/LICENSE) before redistribution.
