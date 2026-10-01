"""Physical PEC / sharded-state timing port on a loaded Megatron model.

Fixed GPU state, no synthetic tensors, no fault injection, no training rollback.
Snapshot copies finish before return; only CPU persistence is asynchronous.
The original MoC GPU snapshot overlap, ZeRO-2 integration and Dynamic-K are
deliberately outside this short benchmark. The output records these boundaries.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time

import torch
import torch.distributed as dist

from moc_timing_plan import EXPERT, dense_owners, selected_experts
from moc_timing_launch import atomic_json


def tensor_bytes(value):
    return value.numel() * value.element_size() if torch.is_tensor(value) else 0


def cpu_copy(value):
    if torch.is_tensor(value):
        # clone() matters on CPU too: views must not mutate under later restores.
        return value.detach().to(device="cpu", copy=True)
    return copy.deepcopy(value)


@dataclass
class Entry:
    key: str
    name: str
    container: object
    slot: object

    def value(self):
        return self.container[self.slot]

    def restore(self, source):
        target = self.value()
        if torch.is_tensor(target):
            if not torch.is_tensor(source) or source.shape != target.shape or source.dtype != target.dtype:
                raise ValueError(f"shape/dtype mismatch for {self.key}")
            with torch.no_grad():
                target.copy_(source)
        else:
            self.container[self.slot] = copy.deepcopy(source)


class LiveState:
    """Explicit parameter -> FP32 master -> Adam mapping, independent of IDs."""
    def __init__(self, model, optimizer, args):
        from megatron.training.utils import unwrap_model
        from megatron.core import parallel_state as mpu
        modules = unwrap_model(model)
        if len(modules) != 1 or mpu.get_tensor_model_parallel_world_size() != 1:
            raise ValueError("requires one model chunk and TP=1")
        if mpu.get_expert_data_parallel_world_size() != 1:
            raise ValueError("this timing port supports EDP=1 only")
        self.rank = dist.get_rank()
        self.dp_group = mpu.get_data_parallel_group(with_context_parallel=True)
        self.dp_ranks = dist.get_process_group_ranks(self.dp_group)
        self.dp_rank = dist.get_rank(self.dp_group)
        self.ep = mpu.get_expert_model_parallel_world_size()
        self.ep_rank = mpu.get_expert_model_parallel_rank()
        self.n = args.num_experts
        self.layer_offset = mpu.get_pipeline_model_parallel_rank() * (args.num_layers // args.pipeline_model_parallel_size)
        module = modules[0]
        named = dict(module.named_parameters())
        reverse = {id(p): n for n, p in named.items()}
        self.entries = {}
        model_state = module.state_dict(keep_vars=True)
        for name, value in model_state.items():
            if not torch.is_tensor(value) and value is not None:
                raise ValueError(f"unsupported non-tensor model state: {name}")
            self.add(Entry("model/" + name, name, model_state, name))
        mapped = set()
        wrappers = getattr(optimizer, "chained_optimizers", [optimizer])
        for wi, wrapper in enumerate(wrappers):
            if "DistributedOptimizer" in type(wrapper).__name__:
                raise ValueError("use the baseline standard BF16 Adam, not distributed optimizer")
            base = wrapper.optimizer
            main_to_name = dict(reverse)
            if hasattr(wrapper, "float16_groups"):
                for models, masters in zip(wrapper.float16_groups, wrapper.fp32_from_float16_groups):
                    if len(models) != len(masters):
                        raise ValueError("master group length mismatch")
                    for parameter, master in zip(models, masters):
                        name = reverse[id(parameter)]
                        main_to_name[id(master)] = name
                        # A dictionary retains the live master tensor reference.
                        self.add(Entry("master/" + name, name, {name: master}, name))
            for gi, group in enumerate(base.param_groups):
                for parameter in group["params"]:
                    name = main_to_name.get(id(parameter))
                    if name is None or name in mapped:
                        raise ValueError("unmapped or duplicate optimizer parameter")
                    mapped.add(name)
                    state = base.state.get(parameter)
                    if not state or not {"exp_avg", "exp_avg_sq"} <= state.keys():
                        raise ValueError(f"baseline Adam moments are missing for {name}")
                    for key, value in state.items():
                        if not torch.is_tensor(value) and not isinstance(value, (int, float, bool, type(None))):
                            raise ValueError(f"unsupported Adam field: {name}/{key}")
                        self.add(Entry(f"adam/{name}/{key}", name, state, key))
                for key in group:
                    if key != "params":
                        self.add(Entry(f"meta/group/{wi}/{gi}/{key}", None, group, key))
            # Fused optimizers can store a common scalar step outside parameter states.
            parameter_ids = {id(p) for g in base.param_groups for p in g["params"]}
            for key in base.state:
                if id(key) not in parameter_ids:
                    if not isinstance(key, str):
                        raise ValueError("unrecognized optimizer state key")
                    self.add(Entry(f"meta/common/{wi}/{key}", None, base.state, key))
        if mapped != {n for n, p in named.items() if p.requires_grad}:
            raise ValueError("optimizer mapping does not cover all trainable parameters")
        if not any(EXPERT.search(n) for n in named):
            raise ValueError("no SequentialMLP experts; fused/grouped layouts need a separate port")
        self.units, self.unit_expert = {}, {}
        for key, entry in self.entries.items():
            name = entry.name
            match = EXPERT.search(name or "")
            if match:
                layer = self.layer_offset + int(match[1])
                eid = self.ep_rank * (self.n // self.ep) + int(match[2])
                if not 0 <= int(match[2]) < self.n // self.ep:
                    raise ValueError("local expert ID outside EP layout")
                unit = f"expert/{layer}/{eid}"
                self.unit_expert[unit] = (layer, eid)
            elif name is None:
                unit = "metadata"
            else:
                # Attention / router / normalization / embeddings as coarse modules.
                match = re.search(r"^(.*decoder\.layers\.\d+\.[^.]+)", name)
                unit = "dense/" + (match[1] if match else name.rsplit(".", 1)[0])
            self.units.setdefault(unit, []).append(key)
        sizes = {u: sum(tensor_bytes(self.entries[k].value()) for k in keys)
                 for u, keys in self.units.items() if u.startswith("dense/")}
        self.owners = dense_owners(sizes, len(self.dp_ranks))
        descriptions = [None] * len(self.dp_ranks)
        dist.all_gather_object(descriptions, (sizes, self.owners), group=self.dp_group)
        if any(row != descriptions[0] for row in descriptions):
            raise ValueError("dense DP tensor layouts/ownership differ")
        self.owned = {u: keys for u, keys in self.units.items()
                      if not u.startswith("dense/") or self.owners[u] == self.dp_rank}

    def add(self, entry):
        if entry.key in self.entries:
            raise ValueError(f"duplicate state key {entry.key}")
        self.entries[entry.key] = entry

    def unit_selected(self, unit, k, round_index, stride=None):
        if unit not in self.unit_expert:
            return True
        layer, eid = self.unit_expert[unit]
        return eid in selected_experts(self.n, self.ep, k, round_index, layer, stride)

    def snapshot(self, k, round_index, stride=None):
        return {unit: {key: cpu_copy(self.entries[key].value()) for key in keys}
                for unit, keys in self.owned.items() if self.unit_selected(unit, k, round_index, stride)}

    def clear_tensor_state(self):
        with torch.no_grad():
            for entry in self.entries.values():
                value = entry.value()
                if torch.is_tensor(value):
                    value.zero_()

    @torch.no_grad()
    def restore_units(self, payload):
        if set(payload) != set(self.owned):
            raise ValueError("incomplete recovery unit coverage")
        for unit, row in payload.items():
            if set(row) != set(self.owned[unit]):
                raise ValueError(f"incomplete state coverage in {unit}")
            for key, value in row.items():
                self.entries[key].restore(value)
        # Every rank rolls back. Redistribute sharded dense state to all DP peers.
        for unit in sorted(self.owners):
            src = self.dp_ranks[self.owners[unit]]
            for key in sorted(self.units[unit]):
                value = self.entries[key].value()
                if torch.is_tensor(value):
                    if value.is_cuda:
                        dist.broadcast(value.detach(), src=src, group=self.dp_group)
                    else:
                        temporary = value.to(device=torch.cuda.current_device())
                        dist.broadcast(temporary, src=src, group=self.dp_group)
                        value.copy_(temporary.cpu())
                elif value is not None:
                    raise ValueError("unsupported dense non-tensor state")

    def verify(self, expected):
        checked = 0
        for unit, row in expected.items():
            for key, source in row.items():
                value = self.entries[key].value()
                if torch.is_tensor(source):
                    if not torch.equal(value.detach().cpu(), source):
                        raise ValueError(f"recovery byte-value check failed: {key}")
                elif value != source:
                    raise ValueError(f"recovery scalar check failed: {key}")
                checked += 1
        # Check all replicas agree on restored dense values, independently of ownership.
        digest = hashlib.sha256()
        for unit in sorted(self.owners):
            for key in sorted(self.units[unit]):
                value = self.entries[key].value()
                if torch.is_tensor(value):
                    data = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
                    digest.update(data.tobytes())
        digests = [None] * len(self.dp_ranks)
        dist.all_gather_object(digests, digest.hexdigest(), group=self.dp_group)
        if len(set(digests)) != 1:
            raise ValueError("dense replicas disagree after recovery")
        return checked


def unit_filename(unit):
    return hashlib.sha256(unit.encode()).hexdigest() + ".pt"


def write_units(directory, payload, durable):
    """Real partial writes: only selected expert units reach disk."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths, byte_count = {}, 0
    t0 = time.perf_counter()
    for unit, row in payload.items():
        path = directory / unit_filename(unit)
        temporary = path.with_suffix(".tmp")
        try:
            with temporary.open("wb") as f:
                torch.save({"unit": unit, "state": row}, f)
                f.flush()
                if durable:
                    os.fsync(f.fileno())
            os.replace(temporary, path)
        except OSError as e:
            raise OSError(f"persist unit {unit} to {path} failed: {e}") from e
        finally:
            temporary.unlink(missing_ok=True)
        byte_count += path.stat().st_size
        paths[unit] = str(path)
    if durable:
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return {"persist_local_s": time.perf_counter() - t0,
            "written_bytes_local": byte_count, "paths": paths}


def read_units(paths):
    payload, read_bytes = {}, 0
    for unit, filename in paths.items():
        path = Path(filename)
        # These are our own plain tensor/scalar files, never baseline pickle objects.
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if blob["unit"] != unit:
            raise ValueError("unit identity mismatch")
        payload[unit] = blob["state"]
        read_bytes += path.stat().st_size
    return payload, read_bytes


class Suite:
    def __init__(self, plan, model, optimizer, args):
        self.plan = plan
        self.live = LiveState(model, optimizer, args)
        self.root = Path(plan.result_dir)
        self.scratch = Path(plan.scratch)
        self.rank = dist.get_rank()
        self.raw = self.root / "ranks" / f"rank_{self.rank:03d}.jsonl"
        self.raw.parent.mkdir(exist_ok=True)
        self.rows = []

    def barrier(self):
        torch.cuda.synchronize()
        dist.barrier()

    def metric(self, value, operation=dist.ReduceOp.MAX):
        v = torch.tensor(float(value), dtype=torch.float64, device="cuda")
        dist.all_reduce(v, op=operation)
        return v.item()

    def emit(self, row):
        row = {"rank": self.rank, **row}
        with self.raw.open("a") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.rows.append(row)
        if self.rank == 0:
            print("[moc-timing] " + json.dumps(row), flush=True)

    def memory_preflight(self):
        experts = [None] * dist.get_world_size()
        dist.all_gather_object(experts, list(self.live.unit_expert.values()))
        all_experts = [pair for shard in experts for pair in shard]
        layer_ids = sorted({layer for layer, eid in all_experts})
        if (len(set(all_experts)) != len(all_experts)
                or any({eid for layer, eid in all_experts if layer == target} != set(range(self.plan.experts))
                       for target in layer_ids)):
            raise ValueError("expert units are duplicated or the distributed layer layout is incomplete")
        self.expert_layer_count = len(layer_ids)
        local_bytes = sum(tensor_bytes(self.live.entries[k].value())
                          for keys in self.live.owned.values() for k in keys)
        totals = [None] * dist.get_world_size()
        dist.all_gather_object(totals, local_bytes)
        local_rank = int(os.environ["LOCAL_RANK"])
        first_rank = self.rank - local_rank
        node_bytes = sum(totals[first_rank:first_rank + self.plan.per_node])
        free = None
        if Path("/proc/meminfo").exists():
            mem = dict(re.findall(r"^(\w+):\s+(\d+)", Path("/proc/meminfo").read_text(), re.M))
            free = int(mem["MemAvailable"]) * 1024
        # Full snapshot plus replacement or disk-load buffer, with allocator headroom.
        required = int(2.5 * node_bytes) + 2 * 1024**3
        self.emit({"event": "preflight", "owned_tensor_bytes": local_bytes,
                   "node_estimated_peak_cpu_bytes": required, "node_mem_available": free,
                   "scratch_peak_estimate_bytes": int(2.3 * sum(totals)),
                   "expert_layer_count": self.expert_layer_count,
                   "cuda_device": torch.cuda.get_device_name(),
                   "torch_version": torch.__version__})
        if free is not None and required > free:
            raise MemoryError(f"node needs about {required / 1024**3:.1f} GiB available CPU RAM; "
                              f"only {free / 1024**3:.1f} GiB available")
        if self.rank == 0:
            free_disk = shutil.disk_usage(self.scratch).free
            if free_disk < 2.3 * sum(totals):
                raise OSError(f"scratch free space {free_disk} below conservative peak estimate")
        self.barrier()

    def run(self):
        self.memory_preflight()
        self.emit({"event": "phase", "stage": "seed_snapshot_and_persist", "time": time.time()})
        self.barrier()
        seed = self.live.snapshot(self.plan.experts, 0)
        self.barrier()
        seed_result = write_units(self.scratch / "seed" / f"rank_{self.rank:03d}", seed, self.plan.fsync)
        self.barrier()
        seed_paths = seed_result["paths"]
        self.emit({"event": "seed", "excluded_from_measured_rounds": True,
                   "persist_local_s": seed_result["persist_local_s"],
                   "written_bytes_local": seed_result["written_bytes_local"]})
        # Files are shared across arms; seed in RAM is released before each new snapshot.
        del seed
        for ai, arm in enumerate(self.plan.arms()):
            self.run_arm(arm, seed_paths, ai)
        if not self.plan.keep_scratch:
            shutil.rmtree(self.scratch / "seed" / f"rank_{self.rank:03d}")
        self.barrier()

    def run_arm(self, arm, seed_paths, arm_index):
        self.emit({"event": "phase", "stage": "arm_start", "arm": arm["name"], "time": time.time()})
        self.barrier()
        latest_memory = self.live.snapshot(self.plan.experts, 0)
        latest_disk = dict(seed_paths)
        directory = self.scratch / arm["name"] / f"rank_{self.rank:03d}"
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending, all_futures = [], []
            for repeat in range(self.plan.repeats):
                self.barrier()
                t0 = time.perf_counter()
                wait_s = 0.0
                # Snapshot / persist / recovery: at most three live generations.
                # Old expert units remain keyed in the recovery cache, not a dense copy.
                if len(pending) >= 2:
                    waiting = time.perf_counter()
                    finished = pending.pop(0)
                    finished[0].result()
                    wait_s = time.perf_counter() - waiting
                payload = self.live.snapshot(arm["snapshot_k"], repeat, arm["persist_k"])
                torch.cuda.synchronize()
                snapshot_s = time.perf_counter() - t0 - wait_s
                latest_memory.update(payload)
                persist_payload = {u: row for u, row in payload.items()
                                   if self.live.unit_selected(u, arm["persist_k"], repeat, arm["persist_k"])}
                # No payload aliases live GPU tensors. Immutable CPU payloads stay
                # alive until their writer finishes. Single writer avoids torn units.
                future = executor.submit(write_units, directory, persist_payload, self.plan.fsync)
                if not arm["async"]:
                    future.result()
                enqueue_s = time.perf_counter() - t0
                pending.append((future, repeat))
                # Update file provenance only after the write completes below.
                snapshot_bytes = sum(tensor_bytes(v) for row in payload.values() for v in row.values())
                self.emit({"event": "snapshot", "arm": arm["name"], "repeat": repeat,
                           "snapshot_local_s": snapshot_s, "backpressure_local_s": wait_s,
                           "enqueue_local_s": enqueue_s, "snapshot_tensor_bytes_local": snapshot_bytes,
                           "snapshot_expert_units_local": sum(u in self.live.unit_expert for u in payload),
                           "persist_expert_units_local": sum(u in self.live.unit_expert for u in persist_payload),
                           "snapshot_k": arm["snapshot_k"], "persist_k": arm["persist_k"]})
                # Completed records are emitted at drain, not in a writer thread.
                # Retain only scalar results/futures after payload scope ends.
                all_futures.append((future, repeat))
                del payload, persist_payload
            drain_start = time.perf_counter()
            completed = [(future.result(), repeat) for future, repeat in all_futures]
            drain_local = time.perf_counter() - drain_start
            for result, repeat in completed:
                latest_disk.update(result["paths"])
                self.emit({"event": "persist", "arm": arm["name"], "repeat": repeat,
                           "persist_local_s": result["persist_local_s"],
                           "written_bytes_local": result["written_bytes_local"]})
        drain_max = self.metric(drain_local)
        self.emit({"event": "drain", "arm": arm["name"], "drain_global_max_s": drain_max})
        for count in (1, 2, 3, 4):
            for repeat in range(self.plan.repeats):
                # Rotate subsets between repeats; same subsets for every arm.
                first = (repeat * 17) % dist.get_world_size()
                failed = {(first + i) % dist.get_world_size() for i in range(count)}
                self.live.clear_tensor_state()
                self.barrier()
                t0 = time.perf_counter()
                if self.rank in failed:
                    recovered, read_bytes = read_units(latest_disk)
                    source = "storage"
                else:
                    recovered, read_bytes = latest_memory, 0
                    source = "memory"
                load_s = time.perf_counter() - t0
                copy_start = time.perf_counter()
                self.live.restore_units(recovered)
                torch.cuda.synchronize()
                copy_s = time.perf_counter() - copy_start
                self.barrier()
                restore_global = self.metric(time.perf_counter() - t0)
                # Byte-value and replica equality outside measured recovery.
                # Fixed-state rounds make memory/storage versions identical; this
                # benchmark cannot validate stale-expert training quality.
                checked = self.live.verify(latest_memory)
                self.emit({"event": "restore", "arm": arm["name"], "repeat": repeat,
                           "failed_ranks": sorted(failed), "failed_count": count,
                           "source": source, "storage_read_bytes_local": read_bytes,
                           "load_local_s": load_s, "copy_and_broadcast_local_s": copy_s,
                           "restore_global_max_s": restore_global, "verified_entries": checked,
                           "dense_replicas_verified": True})
                del recovered
        del latest_memory
        if not self.plan.keep_scratch:
            shutil.rmtree(directory)
        self.barrier()
