# DeepSpeed Adapter

This directory contains the engine adapter and engine-neutral recovery runtime
used by the vendored DeepSpeed 0.19.3 tree in `../DeepSpeed`.

## Real validation

`../test_deepspeed_hotspare_replace.sh` uses the same Qwen3-MoE shape and
Megatron mmap dataset as `../test_hotspare_replace.sh`.

The supported cases are deliberately separate:

| Case | Topology | Validation |
| --- | --- | --- |
| `hot_swap` | 8 active nodes + 1 standby, 64 active GPUs, PP=8, EP=8, ZeRO-1 | kill rank 1 at step 17, keep the other 63 workers and their CUDA state resident, start only rank 1 on node 8, and resume at step 17 |
| `zero2` | 8 nodes, 64 GPUs, PP=1, EP=8, ZeRO-2 | replicate each rank-local optimizer shard through asynchronous D2H and TCP H2H |
| `combined` | 8 active nodes + 1 standby, 64 active GPUs, PP=1, EP=8, ZeRO-2 | combine single-rank replacement and optimizer replication with the same in-process hybrid contract |

DeepSpeed's `PipelineEngine` rejects ZeRO-2 and ZeRO-3, including when AutoEP
is enabled. For that reason, `PP=8 + ZeRO-2` is not offered as a fake or
unsupported test.

The default hot-swap path is rank-granular. Node 8 stays outside the healthy
world with resident GPU workers. At a committed safe point, only the failed
worker exits; the other 63 Python processes, model parameters, and optimizer
parameters stay resident on CUDA. All participating ranks retire the old c10d
generation and deterministically rebuild the DeepSpeed Pipeline, AutoEP, and
ZeRO process-group handles around those tensors. The replacement loads its
expert model and optimizer state from the latest checkpoint, then overwrites
non-expert model and optimizer state from current-step peers. The logical
64-rank topology and resume step do not change. Any incomplete topology,
missing peer replica, or stale state aborts the whole run.

## Environment

The cluster environment needs CUDA PyTorch, `transformers>=5.0.0`, Triton, and
the packages in the vendored DeepSpeed requirements. The launcher prepends
these source directories to `PYTHONPATH`, so an editable install is optional.

Use one optimizer replica slot for the 30B-class test:

```bash
export MOEGAMBIT_ZERO2_BUFFER_SLOTS=1
```

This keeps one local staging snapshot and one peer snapshot. The default value
of two keeps double buffers on both sides and consumes roughly four optimizer
shards of host memory per rank.

The launcher places every optimizer replica outside its owner's
`LOCAL_WORLD_SIZE` failure domain, so a whole-node exit cannot remove both
copies.

The `hot_swap` case also enables packed AutoEP recovery checkpoints and the
resident standby cache:

```bash
export DEEPSPEED_MOEGAMBIT_PACKED_EXPERT_CHECKPOINT=1
export MOEGAMBIT_STANDBY_PACKED_EXPERT_CACHE=1
export MOEGAMBIT_STANDBY_PACKED_EXPERT_PIN_MEMORY=1
export MOEGAMBIT_STANDBY_PACKED_EXPERT_MAX_GIB_PER_RANK=16
```

Each resident worker prefetches only its `(mp_rank, ep_rank)` fused expert
shards. Set `DEEPSPEED_MOEGAMBIT_PACKED_EXPERT_CHECKPOINT=0` to use the
legacy per-expert checkpoint/load path. Disabling only
`MOEGAMBIT_STANDBY_PACKED_EXPERT_CACHE` keeps the packed format and reads its
shards from checkpoint storage during recovery.

## Launch

Run the same command on physical nodes 0 through 8, changing only `NODE_RANK`.
`ELASTIC_WATCHER_ADDR` is the routable address of standby node 8:

```bash
cd /path/to/bsr
export NODE_RANK=0
export MASTER_ADDR=<rank-0-ip>
export ELASTIC_WATCHER_ADDR=<node-8-ip>
export MASTER_PORT=20121
export RUN_ID=ds-real-001
export TEST_MODE=hot_swap
nohup bash ./test_deepspeed_hotspare_replace.sh \
  > "/personal/hotspare94/deepspeed-node${NODE_RANK}.log" 2>&1 &
```

On node 8, `NODE_RANK=8` starts the coordinator/supervisor and resident standby
GPU workers, but none joins the healthy `torch.distributed` world. Recovery
activates only the standby worker whose local rank matches the failed global
rank. On multi-NIC hosts, set `MOEGAMBIT_HOT_SPARE_ADVERTISE_ADDR` separately
on every node to the address reachable by the other training nodes.

`TEST_MODE=all` runs the `hot_swap` and `zero2` cases sequentially. A case is
reported as successful only after its completion manifest proves the expected
restart and optimizer replica commit.

For a failure at step 17 with checkpoint step 10, the recovery log must contain
`resume_step=17`, `rollback_steps=0`, and then `TRAIN_READY global_step=17`.
Seeing `TRAIN_READY global_step=10` means the old checkpoint-relaunch mode is
still running.
