"""Real two-process Generic DDP fail-stop/replacement conformance run.

This is intentionally a small controller rather than a production launcher.
It proves the architecture's critical claim without importing Megatron:

1. train ordinary DDP for two committed steps;
2. kill the selected logical rank;
3. launch a new process that owns that logical rank;
4. rebuild WORLD and the DDP reducer at a new rendezvous;
5. restore parameters, committed buffers, AdamW slots, and param-group options;
6. run additional steps and compare complete training-state digests.

Run from a source checkout (PyTorch required)::

    python examples/generic_ddp/fault_replacement.py

Use ``--backend nccl`` on a machine with at least two GPUs.
Use ``--fail-rank 0`` to exercise the rank-0 replacement path.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import os
import queue
import socket
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _topology_manifest(backend: str):
    from moegambit.distributed.topology import GroupSpec, TopologySpec

    return TopologySpec(
        world_size=2,
        rank=0,
        logical_axes={"dp": 2},
        coordinates={"dp": 0},
        groups=(
            GroupSpec("world", (0, 1), backend, "world", 0),
            GroupSpec("data_parallel", (0, 1), backend, "data_parallel", 1),
        ),
        generation=0,
    ).sealed().manifest_hash


class _QueueCoordinator:
    def __init__(self, commands: Any) -> None:
        self.commands = commands

    def prepare(self, request, adapter):
        del adapter
        assignment = self.commands.get(timeout=120)
        if assignment.plan.recovery_epoch != request.recovery_epoch:
            raise RuntimeError("controller returned an assignment for another epoch")
        return assignment

    def committed(self, assignment, step):
        del assignment, step

    def failed(self, assignment, exc):
        del assignment, exc


def _device(torch: Any, backend: str, rank: int):
    if backend == "nccl":
        torch.cuda.set_device(rank)
        return torch.device("cuda", rank)
    return torch.device("cpu")


def _new_model_optimizer(torch: Any, device: Any):
    torch.manual_seed(1234)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.BatchNorm1d(16),
        torch.nn.Tanh(),
        torch.nn.Linear(16, 2),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    return model, optimizer


def _wrap_ddp(
    torch: Any,
    module: Any,
    group: Any,
    device: Any,
    *,
    init_sync: bool = True,
):
    if hasattr(module, "module"):
        cleanup = getattr(module, "_remove_autograd_hooks", None)
        if callable(cleanup):
            cleanup()
        module = module.module
    kwargs = {"process_group": group}
    if not init_sync:
        parameters = inspect.signature(
            torch.nn.parallel.DistributedDataParallel
        ).parameters
        if "init_sync" not in parameters:
            raise RuntimeError(
                "this PyTorch version cannot safely disable DDP constructor sync"
            )
        kwargs["init_sync"] = False
    if device.type == "cuda":
        kwargs.update(device_ids=[device.index], output_device=device.index)
    return torch.nn.parallel.DistributedDataParallel(module, **kwargs)


def _batch(torch: Any, step: int, device: Any):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(9000 + int(step))
    inputs = torch.randn(12, 8, generator=generator).to(device)
    targets = torch.randint(0, 2, (12,), generator=generator).to(device)
    return inputs, targets


def _tensor_bytes(torch: Any, tensor: Any) -> bytes:
    raw = (
        tensor.detach()
        .cpu()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .tolist()
    )
    return bytes(raw)


def _digest_value(torch: Any, hasher: Any, value: Any) -> None:
    if torch.is_tensor(value):
        hasher.update(str(value.dtype).encode())
        hasher.update(str(tuple(value.shape)).encode())
        hasher.update(_tensor_bytes(torch, value))
        return
    hasher.update(repr(value).encode())


def _model_digest(torch: Any, model: Any) -> str:
    hasher = hashlib.sha256()
    current = model.current
    module = current.module if hasattr(current, "module") else current
    for name, tensor in sorted(module.state_dict().items()):
        hasher.update(name.encode())
        _digest_value(torch, hasher, tensor)
    return hasher.hexdigest()


def _optimizer_digest(torch: Any, model: Any, optimizer: Any) -> str:
    hasher = hashlib.sha256()
    current = model.current
    module = current.module if hasattr(current, "module") else current
    parameter_names = {parameter: name for name, parameter in module.named_parameters()}
    for parameter, values in sorted(
        optimizer.state.items(), key=lambda item: parameter_names[item[0]]
    ):
        hasher.update(parameter_names[parameter].encode())
        for key, value in sorted(values.items(), key=lambda item: repr(item[0])):
            hasher.update(repr(key).encode())
            _digest_value(torch, hasher, value)
    for index, group in enumerate(optimizer.param_groups):
        hasher.update(f"group:{index}".encode())
        names = [parameter_names[parameter] for parameter in group["params"]]
        hasher.update(repr(names).encode())
        for key, value in sorted(group.items(), key=lambda item: repr(item[0])):
            if key in ("params", "param_names"):
                continue
            hasher.update(repr(key).encode())
            _digest_value(torch, hasher, value)
    return hasher.hexdigest()


def _training_state_digest(torch: Any, model: Any, optimizer: Any) -> str:
    hasher = hashlib.sha256()
    hasher.update(_model_digest(torch, model).encode())
    hasher.update(_optimizer_digest(torch, model, optimizer).encode())
    return hasher.hexdigest()


def _build_runtime(
    torch: Any,
    rank: int,
    backend: str,
    owner: Any,
    optimizer: Any,
    device: Any,
    coordinator: Any,
    committed_step: int,
    failed_rank: int,
):
    from moegambit.adapters.generic_ddp import build_generic_ddp_adapter
    from moegambit.config import RuntimeConfig
    from moegambit.distributed.c10d_backend import FailureClassification
    from moegambit.runtime.runtime import RecoveryRuntime

    rebuild_events = []

    def rebuild(old, group):
        rebuild_events.append(group)
        return _wrap_ddp(torch, old, group, device, init_sync=False)

    def validate_restored_state(plan):
        expected = plan.policy_evidence.get("expected_state_digest")
        return bool(
            expected
            and _training_state_digest(torch, owner, optimizer) == expected
        )

    adapter = build_generic_ddp_adapter(
        module=owner,
        optimizer=optimizer,
        backend=backend,
        rank=rank,
        world_size=2,
        committed_step=committed_step,
        replacement_loader=lambda plan: None,
        warmup_validator=validate_restored_state,
        module_rebuilder=rebuild,
        error_classifier=lambda exc: FailureClassification(
            True,
            failure_class="injected_fail_stop",
            failed_ranks=(failed_rank,),
            evidence={"demo": True, "exception": type(exc).__name__},
        ),
    )
    runtime = RecoveryRuntime(
        adapter,
        RuntimeConfig(enabled=True, framework="generic_ddp"),
        coordinator=coordinator,
    )
    return adapter, runtime, rebuild_events


def _initial_worker(
    rank: int,
    backend: str,
    initial_port: int,
    commands: Any,
    reports: Any,
    total_steps: int,
    fail_step: int,
    failed_rank: int,
) -> None:
    import torch
    import torch.nn.functional as functional

    from moegambit.adapters.generic_ddp import RebindableModel, peer_state_sources

    device = _device(torch, backend, rank)
    torch.distributed.init_process_group(
        backend=backend,
        init_method=f"tcp://127.0.0.1:{initial_port}",
        rank=rank,
        world_size=2,
        timeout=__import__("datetime").timedelta(seconds=30),
    )
    module, optimizer = _new_model_optimizer(torch, device)
    owner = RebindableModel(_wrap_ddp(torch, module, None, device))
    adapter, runtime, rebuild_events = _build_runtime(
        torch,
        rank,
        backend,
        owner,
        optimizer,
        device,
        _QueueCoordinator(commands),
        -1,
        failed_rank,
    )

    step = 0
    recovered = False
    while step < total_steps:
        step = runtime.iteration_boundary(step)
        if rank == failed_rank and not recovered and step == fail_step:
            # os._exit models fail-stop: no Python cleanup and no graceful PG leave.
            os._exit(42)
        inputs, targets = _batch(torch, step, device)
        try:
            optimizer.zero_grad(set_to_none=True)
            loss = functional.cross_entropy(owner(inputs), targets)
            loss.backward()
            runtime.before_optimizer_step(step)
            optimizer.step()
            optimizer.param_groups[0]["lr"] *= 0.9
            runtime.after_optimizer_step(step, committed=True)
            runtime.commit_iteration(step)
            step += 1
        except RuntimeError as exc:
            if rank == failed_rank or recovered:
                raise
            expected_state_digest = _training_state_digest(torch, owner, optimizer)
            source_rank = rank
            sources = dict(peer_state_sources(adapter.state, source_rank=source_rank))
            reports.put(
                {
                    "type": "recovery_needed",
                    "rank": rank,
                    "resume_step": fail_step - 1,
                    "sources": sources,
                    "source_rank": source_rank,
                    "expected_state_digest": expected_state_digest,
                    "error": str(exc),
                }
            )
            if not runtime.on_distributed_error(exc):
                raise RuntimeError("MoEGambit rejected the injected DDP failure") from exc
            recovered = True
            step = runtime.resume_step + 1

    reports.put(
        {
            "type": "done",
            "rank": rank,
            "model_digest": _model_digest(torch, owner),
            "optimizer_digest": _optimizer_digest(torch, owner, optimizer),
            "state_digest": _training_state_digest(torch, owner, optimizer),
            "recovered": recovered,
            "rebuild_count": len(rebuild_events),
            "is_ddp": hasattr(owner.current, "reducer"),
        }
    )
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _replacement_worker(
    backend: str,
    assignment: Any,
    reports: Any,
    total_steps: int,
    failed_rank: int,
) -> None:
    import torch
    import torch.nn.functional as functional

    from moegambit.adapters.generic_ddp import RebindableModel
    from moegambit.control.coordinator import StaticRecoveryCoordinator

    rank = failed_rank
    device = _device(torch, backend, rank)
    module, optimizer = _new_model_optimizer(torch, device)
    # The replacement starts before the new process group exists.  Rebind wraps
    # this bare module in a fresh DDP object after rendezvous.
    owner = RebindableModel(module)
    optimizer_was_empty = not bool(optimizer.state)
    _adapter, runtime, rebuild_events = _build_runtime(
        torch,
        rank,
        backend,
        owner,
        optimizer,
        device,
        StaticRecoveryCoordinator(assignment),
        assignment.plan.resume_step,
        failed_rank,
    )
    runtime.execute_assignment(assignment, at_step=assignment.plan.resume_step)

    step = runtime.resume_step + 1
    while step < total_steps:
        step = runtime.iteration_boundary(step)
        inputs, targets = _batch(torch, step, device)
        optimizer.zero_grad(set_to_none=True)
        loss = functional.cross_entropy(owner(inputs), targets)
        loss.backward()
        runtime.before_optimizer_step(step)
        optimizer.step()
        optimizer.param_groups[0]["lr"] *= 0.9
        runtime.after_optimizer_step(step, committed=True)
        runtime.commit_iteration(step)
        step += 1

    reports.put(
        {
            "type": "done",
            "rank": rank,
            "model_digest": _model_digest(torch, owner),
            "optimizer_digest": _optimizer_digest(torch, owner, optimizer),
            "state_digest": _training_state_digest(torch, owner, optimizer),
            "recovered": True,
            "rebuild_count": len(rebuild_events),
            "is_ddp": hasattr(owner.current, "reducer"),
            "optimizer_was_empty": optimizer_was_empty,
        }
    )
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--fail-step", type=int, default=2)
    parser.add_argument("--fail-rank", type=int, choices=(0, 1), default=1)
    args = parser.parse_args(argv)
    try:
        import torch
        import torch.multiprocessing as multiprocessing
    except ImportError:
        print("PyTorch is required for the real rank-replacement example.")
        return 2
    if args.backend == "nccl" and torch.cuda.device_count() < 2:
        print("--backend nccl requires at least two visible GPUs.")
        return 2
    if not 0 < args.fail_step < args.steps:
        parser.error("--fail-step must be between 1 and --steps-1")

    from moegambit.adapters.base import StateSources, StoreHandle
    from moegambit.control.coordinator import RecoveryAssignment
    from moegambit.runtime.recovery_plan import RecoveryMode, RecoveryPlan

    context = multiprocessing.get_context("spawn")
    reports = context.Queue()
    commands = context.Queue()
    initial_port = _free_port()
    workers = {
        rank: context.Process(
            target=_initial_worker,
            args=(
                rank,
                args.backend,
                initial_port,
                commands,
                reports,
                args.steps,
                args.fail_step,
                args.fail_rank,
            ),
        )
        for rank in (0, 1)
    }
    for worker in workers.values():
        worker.start()
    victim = workers[args.fail_rank]
    survivor = workers[1 - args.fail_rank]
    victim.join(timeout=90)
    if victim.is_alive():
        victim.terminate()
        raise RuntimeError("injected victim did not exit")
    if victim.exitcode != 42:
        raise RuntimeError(f"victim exited unexpectedly: {victim.exitcode}")

    try:
        needed = reports.get(timeout=90)
    except queue.Empty as exc:
        raise RuntimeError("survivor did not request recovery") from exc
    if needed.get("type") != "recovery_needed":
        raise RuntimeError(f"unexpected controller report: {needed}")
    new_port = _free_port()
    plan = RecoveryPlan(
        protocol_version=1,
        recovery_epoch=1,
        failed_ranks=(args.fail_rank,),
        resume_step=int(needed["resume_step"]),
        mode=RecoveryMode.PEER,
        topology_generation=1,
        group_manifest_hash=_topology_manifest(args.backend),
        state_sources=dict(needed["sources"]),
        policy_evidence={
            "source_rank": int(needed["source_rank"]),
            "expected_state_digest": needed["expected_state_digest"],
        },
    )
    assignment = RecoveryAssignment(
        plan=plan,
        store=StoreHandle("127.0.0.1", new_port, "ddp-demo/epoch-1", 90.0),
        sources=StateSources(dict(plan.state_sources)),
    )
    commands.put(assignment)
    replacement = context.Process(
        target=_replacement_worker,
        args=(args.backend, assignment, reports, args.steps, args.fail_rank),
    )
    replacement.start()

    completed = {}
    while len(completed) < 2:
        try:
            report = reports.get(timeout=180)
        except queue.Empty as exc:
            raise RuntimeError("recovered workers did not finish") from exc
        if report.get("type") == "done":
            completed[int(report["rank"])] = report
    survivor.join(timeout=30)
    replacement.join(timeout=30)
    if survivor.is_alive() or replacement.is_alive():
        survivor.terminate()
        replacement.terminate()
        raise RuntimeError("recovered workers did not exit after reporting completion")
    if survivor.exitcode != 0 or replacement.exitcode != 0:
        raise RuntimeError(
            "recovered process exit codes: "
            f"survivor={survivor.exitcode}, replacement={replacement.exitcode}"
        )
    if not all(report["recovered"] for report in completed.values()):
        raise RuntimeError(f"a worker did not execute recovery: {completed}")
    if not all(report["is_ddp"] for report in completed.values()):
        raise RuntimeError(f"a worker did not install a rebuilt DDP wrapper: {completed}")
    if not all(report["rebuild_count"] == 1 for report in completed.values()):
        raise RuntimeError(f"unexpected DDP rebuild count: {completed}")
    replacement_report = completed[args.fail_rank]
    if not replacement_report.get("optimizer_was_empty"):
        raise RuntimeError("replacement optimizer was not lazy/unmaterialized")
    for field in ("model_digest", "optimizer_digest", "state_digest"):
        digests = {report[field] for report in completed.values()}
        if len(digests) != 1:
            raise RuntimeError(f"post-recovery {field} mismatch: {completed}")
    print(
        f"PASS: logical rank {args.fail_rank} was replaced, model/buffer/optimizer "
        "state restored, and rebuilt DDP continued; "
        f"digest={completed[0]['state_digest'][:16]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
