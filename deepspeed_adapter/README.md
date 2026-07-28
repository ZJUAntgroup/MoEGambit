# DeepSpeed Adapter

This directory contains the engine adapter and engine-neutral recovery runtime
used by the vendored DeepSpeed 0.19.3 tree in `../DeepSpeed`.

## Real validation

`../test_deepspeed_hotspare_replace.sh` uses the same Qwen3-MoE shape and
Megatron mmap dataset as `../test_hotspare_replace.sh`.

The supported cases are deliberately separate:

| Case | Topology | Validation |
| --- | --- | --- |
| `hot_swap` | 8 active nodes + 1 standby, 64 active GPUs, PP=8, EP=8, ZeRO-1 | kill rank 1 at step 17, move its logical node to node 8, and restore the step-10 DeepSpeed checkpoint in a new recovery epoch |
| `zero2` | 8 nodes, 64 GPUs, PP=1, EP=8, ZeRO-2 | replicate each rank-local optimizer shard through asynchronous D2H and TCP H2H |
| `combined` | 8 active nodes + 1 standby, 64 active GPUs, PP=1, EP=8, ZeRO-2 | combine node replacement and optimizer replication; recovery still comes from the durable checkpoint |

DeepSpeed's `PipelineEngine` rejects ZeRO-2 and ZeRO-3, including when AutoEP
is enabled. For that reason, `PP=8 + ZeRO-2` is not offered as a fake or
unsupported test.

DeepSpeed cannot replace one rank inside a live c10d/NCCL world. The adapter
therefore implements node-level hot replacement: node 8 stays outside the
healthy world, takes over all eight logical ranks of the failed node, and the
seven survivors enter the same checkpoint-backed recovery epoch. The logical
64-rank topology does not change. A second failure without another spare
aborts every remaining agent.

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

On node 8, `NODE_RANK=8` starts only the coordinator/supervisor. It does not
join `torch.distributed` until a recovery epoch assigns it a failed logical
node. On multi-NIC hosts, set `MOEGAMBIT_HOT_SPARE_ADVERTISE_ADDR` separately
on every node to the address reachable by the other training nodes.

`TEST_MODE=all` runs the `hot_swap` and `zero2` cases sequentially. A case is
reported as successful only after its completion manifest proves the expected
restart and optimizer replica commit.
