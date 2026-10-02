"""Real training, immutable GPU snapshots, process restart and actual replay.

This is an independent fixed-K port, not the original ZeRO-2 artifact. A
controlled failure is scheduled only after the checkpoint is globally durable.
"""
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import runpy
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]


def tensor_digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(__import__('torch').uint8).numpy().tobytes()).hexdigest()


def install_training_hooks(package, setup, step, pretrain):
    # pretrain_gpt imports the re-export, not training.training.pretrain.
    package.training.setup_model_and_optimizer = setup
    package.training.train_step = step
    package.training.pretrain = package.pretrain = pretrain


def main():
    sys.path.insert(0, str(ROOT / "Megatron-LM"))
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.distributed.elastic.multiprocessing.errors import record
    import megatron.training as training_package
    from megatron.training import training, get_args
    from megatron.training.checkpointing import get_rng_state
    from megatron.core import tensor_parallel
    from megatron.core.num_microbatches_calculator import update_num_microbatches
    from megatron.core.rerun_state_machine import get_rerun_state_machine
    from megatron.training.utils import unwrap_model
    from moc_timing_runtime import LiveState, cpu_copy, write_units, read_units, tensor_bytes
    from moc_timing_launch import atomic_json
    from moc_e2e_launch import cache_request

    cfg = json.loads((Path(os.environ["MOC_RESULT_DIR"]) / "plan.json").read_text())
    job = next(x for x in cfg["jobs"] if x["id"] == os.environ["MOC_E2E_JOB"])
    phase = os.environ["MOC_E2E_PHASE"]
    job_root = Path(cfg["result_dir"]) / job["id"]
    log_root = job_root / phase
    disk_root = Path(cfg["scratch"]) / job["id"]
    rows, batches = [], []
    live = None
    scheduler = None
    future = None
    snapshot_event = None
    snapshot_stream = None
    checkpoint_metrics = {}
    pool = ThreadPoolExecutor(max_workers=1)
    original_setup, original_step, original_pretrain = training.setup_model_and_optimizer, training.train_step, training.pretrain
    from moc_timing_data import install_timing_data_hook
    install_timing_data_hook(training, get_args)

    def context(settings, data_iterator):
        state = {"step": cfg["checkpoint_step"],
                 "counters": {key: getattr(settings, key) for key in
                              ("consumed_train_samples", "consumed_valid_samples", "skipped_train_samples")
                              if hasattr(settings, key)},
                 "scheduler": copy.deepcopy(scheduler.state_dict()),
                 "rng": cpu_copy_rng(get_rng_state("torch")),
                 "rerun": get_rerun_state_machine().state_dict(data_iterator=data_iterator, ckpt_format="torch")}
        return state

    def cpu_copy_rng(values):
        # Tracker tensors can alias CUDA storage. Keep metadata immutable too.
        if torch.is_tensor(values):
            return cpu_copy(values)
        if isinstance(values, dict):
            return {key: cpu_copy_rng(value) for key, value in values.items()}
        if isinstance(values, list):
            return [cpu_copy_rng(value) for value in values]
        return copy.deepcopy(values)

    def restore_context(settings, metadata):
        if metadata["step"] != cfg["checkpoint_step"]:
            raise ValueError("checkpoint metadata step mismatch")
        settings.iteration = metadata["step"]
        for key, value in metadata["counters"].items():
            setattr(settings, key, value)
        update_num_microbatches(consumed_samples=settings.consumed_train_samples, verbose=True)
        scheduler.load_state_dict(metadata["scheduler"])
        if metadata["rerun"] is not None:
            get_rerun_state_machine().load_state_dict(metadata["rerun"])
        values = metadata["rng"]
        from megatron.core import parallel_state as mpu
        rng = values[mpu.get_data_parallel_rank()] if settings.data_parallel_random_init else values[0]
        random.setstate(rng["random_rng_state"])
        np.random.set_state(rng["np_rng_state"])
        torch.set_rng_state(rng["torch_rng_state"])
        torch.cuda.set_rng_state(rng["cuda_rng_state"])
        tensor_parallel.get_cuda_rng_tracker().set_states(rng["rng_tracker_states"])

    def setup(*args, **kwargs):
        nonlocal live, scheduler
        model, optimizer, scheduler = original_setup(*args, **kwargs)
        settings, rank = get_args(), dist.get_rank()
        expected = cfg["checkpoint_step"] if phase == "resume" and job["arm"] == "full_sync_native" else cfg["step"]
        if settings.iteration != expected or settings.micro_batch_size != 1:
            raise ValueError("wrong loaded step or micro batch size")
        live = LiveState(model, optimizer, settings)
        if phase == "prefix":
            sizes = {"native": sum(tensor_bytes(entry.value()) for entry in live.entries.values()),
                     "owned": sum(tensor_bytes(live.entries[key].value()) for keys in live.owned.values() for key in keys),
                     "selected": sum(tensor_bytes(live.entries[key].value()) for unit, keys in live.owned.items()
                                     if live.unit_selected(unit, cfg["snapshot_k"], 0, cfg["persist_k"]) for key in keys)}
            local = [None] * cfg["nnodes"] * cfg["per_node"]
            dist.all_gather_object(local, sizes)
            # Supervisor + worker + serialization + pinned copies can coexist.
            memory = sum(x["selected"] for x in local[(rank // cfg["per_node"]) * cfg["per_node"]:
                                                       (rank // cfg["per_node"] + 1) * cfg["per_node"]])
            available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                             if line.startswith("MemAvailable:"))
            # Also honor container cgroup limits, not only host MemAvailable.
            for limit_file, current_file in [("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                                              ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes")]:
                if Path(limit_file).is_file() and Path(current_file).is_file():
                    limit = Path(limit_file).read_text().strip()
                    if limit != "max":
                        available = min(available, int(limit) - int(Path(current_file).read_text()))
            if job["arm"] == "pec_2level_async" and available < memory * 4 + (2 << 30):
                raise MemoryError(f"CPU snapshot/cache needs at least {memory * 4 + (2 << 30)} free bytes; available={available}")
            import shutil
            if shutil.disk_usage(cfg["scratch"]).free < sum(x["native"] for x in local) * 1.5:
                raise OSError("insufficient scratch capacity for one full checkpoint plus temporary writes")
            atomic_json(log_root / f"environment.rank_{rank:03d}.json",
                        {"rank": rank, "torch": torch.__version__, "cuda": torch.version.cuda,
                         "device": torch.cuda.get_device_name(), "memory_available": available, "sizes": sizes,
                         "pid": os.getpid(), "step": settings.iteration})
        elif job["arm"] != "full_sync_native":
            disk = torch.load(disk_root / f"rank_{rank:03d}" / "context.pt", map_location="cpu", weights_only=False)
            paths = json.loads((disk_root / f"rank_{rank:03d}" / "paths.json").read_text())
            if job["arm"] == "pec_2level_async" and rank not in job["failed_ranks"]:
                overlay = cache_request(os.environ["MOC_E2E_CACHE_SOCKET"], {"op": "get", "rank": rank})
                source = "supervisor_cpu"
            else:
                overlay, _ = read_units(paths)
                source = "durable_partial_checkpoint"
            expected_units = {unit for unit in live.owned if live.unit_selected(unit,
                              cfg["snapshot_k"] if source == "supervisor_cpu" else cfg["persist_k"], 0, cfg["persist_k"])}
            if set(overlay) != expected_units:
                raise ValueError("selected expert/dense/optimizer recovery coverage mismatch")
            # Unselected experts retain native baseline state, including masters/moments.
            payload = {unit: {key: live.entries[key].value() for key in keys} for unit, keys in live.owned.items()}
            payload.update(overlay)
            live.restore_units(payload)
            checked = live.verify(overlay)
            restore_context(settings, disk)
            atomic_json(log_root / f"restore.rank_{rank:03d}.json",
                        {"rank": rank, "source": source, "verified_fields": checked,
                         "checkpoint_step": settings.iteration, "pid": os.getpid(),
                         "expert_units": sum(unit.startswith("expert/") for unit in overlay),
                         "base_expert_step": cfg["step"], "selected_expert_step": cfg["checkpoint_step"]})
        else:
            atomic_json(log_root / f"restore.rank_{rank:03d}.json",
                        {"rank": rank, "source": "native_full_checkpoint", "checkpoint_step": settings.iteration,
                         "pid": os.getpid()})
        return model, optimizer, scheduler

    def save_partial(metadata, payload, event):
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        if event is not None:
            event.synchronize()
        start = time.monotonic()
        rank = live.rank
        if job["arm"] == "pec_2level_async":
            cache_request(os.environ["MOC_E2E_CACHE_SOCKET"], {"op": "put", "rank": rank, "payload": payload})
        checkpoint_metrics["cpu_cache_transfer_s"] = time.monotonic() - start
        persist = {unit: value for unit, value in payload.items() if live.unit_selected(unit, cfg["persist_k"], 0, cfg["persist_k"])}
        directory = disk_root / f"rank_{rank:03d}"
        result = write_units(directory, persist, True)
        temporary = directory / "context.tmp"
        with temporary.open("wb") as output:
            torch.save(metadata, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, directory / "context.pt")
        atomic_json(directory / "paths.json", result["paths"])
        checkpoint_metrics.update(result)
        checkpoint_metrics["persisted_expert_units"] = sum(unit.startswith("expert/") for unit in persist)
        checkpoint_metrics["snapshotted_expert_units"] = sum(unit.startswith("expert/") for unit in payload)
        checkpoint_metrics["context_bytes"] = (directory / "context.pt").stat().st_size
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def async_snapshot(model):
        nonlocal snapshot_event, snapshot_stream
        parameter_names = set(dict(unwrap_model(model)[0].named_parameters()))
        snapshot_stream = torch.cuda.Stream()
        snapshot_stream.wait_stream(torch.cuda.current_stream())
        payload = {}
        # Router bias and other buffers may be mutated during forward/backward.
        # Capture them synchronously; only optimizer-protected state overlaps.
        for unit, keys in live.owned.items():
            if not live.unit_selected(unit, cfg["snapshot_k"], 0, cfg["persist_k"]):
                continue
            payload[unit] = {}
            for key in keys:
                entry, value = live.entries[key], live.entries[key].value()
                if not torch.is_tensor(value) or not value.is_cuda or (key.startswith("model/") and entry.name not in parameter_names):
                    payload[unit][key] = cpu_copy(value)
                else:
                    payload[unit][key] = torch.empty_like(value, device="cpu", pin_memory=True)
        with torch.cuda.stream(snapshot_stream):
            begin = torch.cuda.Event(enable_timing=True)
            begin.record()
            for unit, values in payload.items():
                for key, destination in values.items():
                    source = live.entries[key].value()
                    if torch.is_tensor(source) and source.is_cuda and destination.is_pinned():
                        destination.copy_(source.detach(), non_blocking=True)
            snapshot_event = torch.cuda.Event(enable_timing=True)
            snapshot_event.record()
        return payload, begin

    def checkpoint(arguments):
        nonlocal future
        _, data_iterator, model, optimizer, _, _, _ = arguments
        settings = get_args()
        torch.cuda.synchronize()
        dist.barrier()
        start = time.monotonic()
        if job["arm"] == "full_sync_native":
            # Genuine native full checkpoint, including RNG/scheduler/iterator.
            settings.save = str(disk_root / "native")
            try:
                training.save_checkpoint(cfg["checkpoint_step"], model, optimizer, scheduler, 0,
                                         train_data_iterator=data_iterator)
            finally:
                settings.save = None  # suppress automatic final full checkpoint
            # torch-format native saving does not promise fsync: include it here.
            from megatron.core import parallel_state as mpu
            from megatron.training.checkpointing import get_checkpoint_name
            filename = get_checkpoint_name(str(disk_root / "native"), cfg["checkpoint_step"])
            # Native save owns one EP shard per EDP group (EDP=1 here).
            if mpu.get_expert_data_parallel_rank() == 0:
                with open(filename, "rb") as checkpoint_file:
                    os.fsync(checkpoint_file.fileno())
                fd = os.open(str(Path(filename).parent), os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
                checkpoint_metrics["written_bytes_local"] = Path(filename).stat().st_size
            checkpoint_metrics["checkpoint_blocking_s"] = time.monotonic() - start
        else:
            metadata = context(settings, data_iterator)
            if job["arm"] == "pec_2level_async":
                payload, begin = async_snapshot(model)
                future = pool.submit(save_partial, metadata, payload, snapshot_event)
                checkpoint_metrics["gpu_copy_begin_event"] = begin
            else:
                payload = live.snapshot(cfg["persist_k"], 0, cfg["persist_k"])
                save_partial(metadata, payload, None)
            checkpoint_metrics["checkpoint_blocking_s"] = time.monotonic() - start
        # A GPU barrier after enqueuing D2H could serialize the intended overlap.
        if job["arm"] != "pec_2level_async":
            dist.barrier()

    def checked_step(*args, **kwargs):
        nonlocal future
        if kwargs or len(args) != 7:
            raise ValueError("native train_step API changed; refuse unverified integration")
        settings, rank = get_args(), dist.get_rank()
        current = int(settings.curr_iteration)
        if not rows:
            torch.cuda.synchronize()
            dist.barrier()
            if rank == 0:
                atomic_json(log_root / "first_step_start.json", {"monotonic": time.monotonic(), "step": current})
        if phase == "prefix" and current == cfg["checkpoint_step"]:
            checkpoint(args)
        optimizer = args[3]
        samples_before = int(settings.consumed_train_samples)
        learning_rates_before = [float(group["lr"]) for group in optimizer.param_groups]
        native_update = optimizer.step
        def protected_update(*update_args, **update_kwargs):
            if snapshot_event is not None:
                before = time.monotonic()
                snapshot_event.synchronize()  # weights/moments cannot race D2H
                checkpoint_metrics["optimizer_snapshot_wait_s"] = checkpoint_metrics.get("optimizer_snapshot_wait_s", 0) + time.monotonic() - before
            return native_update(*update_args, **update_kwargs)
        optimizer.step = protected_update
        start = time.monotonic()
        fb_start = fb_end = None
        arguments = list(args)
        if snapshot_event is not None and current == cfg["checkpoint_step"]:
            native_fb = arguments[6]
            fb_start, fb_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            def traced_fb(*fb_args, **fb_kwargs):
                fb_start.record()
                value = native_fb(*fb_args, **fb_kwargs)
                fb_end.record()
                return value
            arguments[6] = traced_fb
        try:
            output = original_step(*arguments)
        finally:
            optimizer.step = native_update
        torch.cuda.synchronize()
        if output[1] or output[3]:
            raise RuntimeError("training update skipped or requested unexpected early exit")
        loss = {}
        for key, value in output[0].items():
            if not torch.isfinite(torch.as_tensor(value)).all():
                raise RuntimeError("nonfinite training loss")
            loss[key] = float(torch.as_tensor(value).float().mean().item())
        committed = current + 1
        rows.append({"step": committed, "loss": loss, "elapsed_s": time.monotonic() - start,
                     "consumed_samples_before": samples_before, "learning_rates_before": learning_rates_before,
                     "learning_rates_after": [float(group["lr"]) for group in optimizer.param_groups]})
        if fb_start is not None:
            begin = checkpoint_metrics["gpu_copy_begin_event"]
            copy_ms = begin.elapsed_time(snapshot_event)
            fb_offset, fb_duration = begin.elapsed_time(fb_start), fb_start.elapsed_time(fb_end)
            checkpoint_metrics.update(gpu_snapshot_copy_ms=copy_ms, forward_backward_ms=fb_duration,
                                      gpu_timeline_overlap_ms=max(0.0, min(copy_ms, fb_offset + fb_duration) - max(0.0, fb_offset)))
        if phase == "prefix" and committed == cfg["failure_step"]:
            drain_start = time.monotonic()
            if future is not None:
                future.result()
            checkpoint_metrics["failure_boundary_drain_s"] = time.monotonic() - drain_start
            if snapshot_event is not None:
                checkpoint_metrics["gpu_snapshot_copy_ms"] = checkpoint_metrics.pop("gpu_copy_begin_event").elapsed_time(snapshot_event)
            # All ranks acknowledge durable completion before controlled failure.
            dist.barrier()
            if rank == 0:
                atomic_json(job_root / "fault.json", {"monotonic": time.monotonic(), "step": committed,
                            "failed_ranks": job["failed_ranks"], "mode": "safe-boundary worker exit + real relaunch",
                            "detection_included": False})
        if phase == "resume" and committed in (cfg["checkpoint_step"] + 1, cfg["failure_step"], cfg["failure_step"] + 1, cfg["endpoint"]):
            dist.barrier()
            if rank == 0:
                atomic_json(log_root / f"commit_{committed}.json", {"monotonic": time.monotonic(), "step": committed})
        return output

    def pretrain(*args, **kwargs):
        forward_step = args[3] if len(args) > 3 else kwargs["forward_step_func"]
        globals_ = forward_step.__globals__
        get_batch = globals_["get_batch"]
        def audited_batch(*batch_args, **batch_kwargs):
            result = tuple(get_batch(*batch_args, **batch_kwargs))
            if result[0] is not None:
                # Actual token/label contents, not just consumed sample counts.
                batches.append({"step": int(get_args().curr_iteration) + 1,
                                "tokens": tensor_digest(result[0]),
                                "labels": tensor_digest(result[1]) if result[1] is not None else None})
            return result
        globals_["get_batch"] = audited_batch
        try:
            return original_pretrain(*args, **kwargs)
        finally:
            globals_["get_batch"] = get_batch

    install_training_hooks(training_package, setup, checked_step, pretrain)
    @record
    def execute():
        runpy.run_path(str(ROOT / "Megatron-LM" / "pretrain_gpt.py"), run_name="__main__")
        rank = dist.get_rank()
        start = cfg["step"] if phase == "prefix" else cfg["checkpoint_step"]
        endpoint = cfg["failure_step"] if phase == "prefix" else cfg["endpoint"]
        if [row["step"] for row in rows] != list(range(start + 1, endpoint + 1)):
            raise ValueError("incomplete or duplicate committed training steps")
        atomic_json(log_root / f"training.rank_{rank:03d}.json",
                    {"rank": rank, "pid": os.getpid(), "phase": phase, "rows": rows, "batches": batches})
        if phase == "prefix":
            atomic_json(log_root / f"checkpoint.rank_{rank:03d}.json", checkpoint_metrics)
        dist.barrier()
    try:
        execute()
    except BaseException:
        error = {"rank": os.environ.get("RANK"), "time": time.time(), "traceback": traceback.format_exc()}
        print(json.dumps(error), file=sys.stderr, flush=True)
        try:
            directory = log_root / "worker_errors"
            directory.mkdir(parents=True, exist_ok=True)
            atomic_json(directory / f"FAILED.rank_{os.environ.get('RANK', 'unknown')}.json", error)
        except Exception as exception:
            print(f"cannot save worker error: {exception}", file=sys.stderr, flush=True)
        raise
    finally:
        pool.shutdown(wait=True)


if __name__ == "__main__":
    main()
