"""Megatron implementation of the MoEGambit framework contract.

All Megatron and torch imports are deliberately lazy and contained in this
adapter package.  The recovery driver is a migration bridge around the proven
in-tree sequence; the four narrow adapter components expose the same state and
topology through the framework-neutral contract so the bridge can be retired
incrementally without changing the public training API.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from ...capabilities import AdapterCapabilities, SupportLevel
from ...distributed.topology import GroupSpec, TopologySpec, ValidationReport
from ...errors import AdapterUnsupportedError, ContractViolation, RecoverableDistributedError
from ...runtime.recovery_plan import RecoveryPlan
from ...state.catalog import Placement, StateCatalog, StateKind, StateRef
from ...state.version import StateVersion
from ..base import (
    FrameworkAdapter,
    PauseRequest,
    ProgressToken,
    QuiescenceProof,
    RebuildHandle,
    StateSources,
    StoreHandle,
)

__all__ = [
    "MEGATRON_CAPABILITIES",
    "MegatronAdapterPlugin",
    "build_megatron_adapter",
]


def _module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise AdapterUnsupportedError(
            f"Megatron adapter requires importable module {name!r}: {exc}"
        ) from exc


def _client() -> Any:
    return _module("moegambit.adapters.megatron.elastic_client")


def _parallel_state() -> Any:
    return _module("megatron.core.parallel_state")


def _torch() -> Any:
    return _module("torch")


def _iter_model_chunks(model: Any) -> Iterable[Any]:
    chunks = model if isinstance(model, (tuple, list)) else (model,)
    for chunk in chunks:
        current = chunk
        visited = set()
        while hasattr(current, "module") and id(current) not in visited:
            visited.add(id(current))
            current = current.module
        yield current


def _named_parameters(model: Any) -> Tuple[Tuple[str, Any], ...]:
    result = []
    for chunk_index, chunk in enumerate(_iter_model_chunks(model)):
        named = getattr(chunk, "named_parameters", None)
        if not callable(named):
            continue
        prefix = f"chunk{chunk_index}/" if len(tuple(_iter_model_chunks(model))) > 1 else ""
        result.extend((prefix + str(name), parameter) for name, parameter in named())
    return tuple(result)


def _unwrap_optimizer(optimizer: Any) -> Any:
    current = optimizer
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        state = getattr(current, "state", None)
        if isinstance(state, Mapping):
            return current
        nested = getattr(current, "optimizer", None)
        if nested is None:
            break
        current = nested
    return optimizer


def _is_expert_name(name: str) -> bool:
    lowered = name.lower()
    return "expert" in lowered or "local_experts" in lowered


@dataclass
class _Context:
    model: Any
    optimizer: Any
    scheduler: Any = None
    args: Any = None
    current_step: int = 0
    recovery_epoch: int = 0
    resume_step: int = 0
    topology: Optional[TopologySpec] = None
    catalog: Optional[StateCatalog] = None
    last_rebuild_info: Dict[str, Any] = field(default_factory=dict)


class MegatronRecoveryDriver:
    """Bridge the public runtime to the existing hot-spare state machine."""

    _MESSAGE_MARKERS = (
        "nccl",
        "connection reset",
        "broken pipe",
        "peer failure",
        "remote process exited",
        "unhandled system error",
    )

    def __init__(self, context: _Context) -> None:
        self._context = context

    def classify_error(self, exc: BaseException) -> bool:
        if isinstance(exc, RecoverableDistributedError):
            return True
        torch = _torch()
        distributed = getattr(torch, "distributed", None)
        typed = getattr(distributed, "DistBackendError", None)
        if isinstance(typed, type) and isinstance(exc, typed):
            return True
        # Older target torch versions surface NCCL fail-stop errors as plain
        # RuntimeError.  Require an active distributed job plus a narrow marker
        # allowlist; arbitrary RuntimeError instances still propagate.
        active = bool(
            distributed is not None
            and distributed.is_available()
            and distributed.is_initialized()
        )
        message = str(exc).lower()
        return (
            active
            and isinstance(exc, RuntimeError)
            and any(marker in message for marker in self._MESSAGE_MARKERS)
        )

    def _do_rebuild(self) -> Optional[int]:
        client = _client()
        resume = client.elastic_do_rebuild(
            self._context.model,
            self._context.optimizer,
            self._context.scheduler,
        )
        if resume is None:
            return None
        self._context.resume_step = int(resume)
        self._context.current_step = int(resume)
        try:
            self._context.recovery_epoch = int(
                os.environ.get("ELASTIC_RECOVERY_EPOCH", self._context.recovery_epoch + 1)
            )
        except ValueError:
            self._context.recovery_epoch += 1
        return int(resume)

    def recover(self, exc: BaseException) -> Optional[int]:
        if not self.classify_error(exc):
            return None
        client = _client()
        client.elastic_on_nccl_error(exc)
        if not client.elastic_check_pause():
            return None
        return self._do_rebuild()

    def poll(self, step: int) -> Optional[int]:
        self._context.current_step = int(step)
        client = _client()
        client.elastic_client_update_step(
            int(step), phase="iteration_safe_point", step_tag=int(step)
        )
        if not client.elastic_check_pause():
            return None
        return self._do_rebuild()

    def commit(self, step: int) -> bool:
        result = _client().elastic_commit_post_rebuild_iteration(int(step))
        return bool(result)


class MegatronTopologyAdapter:
    def __init__(self, context: _Context) -> None:
        self._context = context

    def inspect(self) -> TopologySpec:
        torch = _torch()
        dist = torch.distributed
        if not dist.is_available() or not dist.is_initialized():
            raise AdapterUnsupportedError("Megatron distributed is not initialized")
        mpu = _parallel_state()
        rank = int(dist.get_rank())
        world_size = int(dist.get_world_size())
        args = self._context.args
        axes = {
            "dp": int(mpu.get_data_parallel_world_size()),
            "tp": int(mpu.get_tensor_model_parallel_world_size()),
            "pp": int(mpu.get_pipeline_model_parallel_world_size()),
        }
        optional_axes = (
            ("ep", "get_expert_model_parallel_world_size"),
            ("etp", "get_expert_tensor_parallel_world_size"),
            ("cp", "get_context_parallel_world_size"),
        )
        for axis, getter_name in optional_axes:
            getter = getattr(mpu, getter_name, None)
            if callable(getter):
                try:
                    axes[axis] = int(getter())
                except Exception:
                    pass
        coordinates = {}
        for axis, getter_name in (
            ("dp", "get_data_parallel_rank"),
            ("tp", "get_tensor_model_parallel_rank"),
            ("pp", "get_pipeline_model_parallel_rank"),
            ("ep", "get_expert_model_parallel_rank"),
            ("etp", "get_expert_tensor_parallel_rank"),
            ("cp", "get_context_parallel_rank"),
        ):
            getter = getattr(mpu, getter_name, None)
            if callable(getter):
                try:
                    coordinates[axis] = int(getter())
                except Exception:
                    pass

        groups = []
        pg_dict = _client()._elastic_current_pg_dict()
        seen = set()
        for ordinal, (purpose, group) in enumerate(pg_dict.items()):
            if group is None or id(group) in seen:
                continue
            seen.add(id(group))
            try:
                ranks = tuple(int(item) for item in dist.get_process_group_ranks(group))
            except Exception:
                size = int(dist.get_world_size(group=group))
                ranks = tuple(range(size))
            try:
                backend = str(dist.get_backend(group))
            except Exception:
                backend = str(dist.get_backend())
            groups.append(
                GroupSpec(str(purpose), ranks, backend, str(purpose), ordinal)
            )
        if not groups:
            groups.append(
                GroupSpec(
                    "world",
                    tuple(range(world_size)),
                    str(dist.get_backend()),
                    "world",
                    0,
                )
            )
        try:
            generation = int(os.environ.get("ELASTIC_PG_GENERATION", "0"))
        except ValueError:
            generation = 0
        topology = TopologySpec(
            world_size=world_size,
            rank=rank,
            logical_axes=axes,
            coordinates=coordinates,
            groups=tuple(groups),
            generation=generation,
        ).sealed()
        self._context.topology = topology
        return topology

    def prepare_rebuild(self, plan: RecoveryPlan) -> RebuildHandle:
        return RebuildHandle(plan.recovery_epoch, {"legacy_bridge": True})

    def rebuild(self, plan: RecoveryPlan, store: StoreHandle) -> TopologySpec:
        del store
        resume = MegatronRecoveryDriver(self._context)._do_rebuild()
        if resume is None:
            raise ContractViolation("Megatron rebuild returned no resume step")
        return self.inspect()

    def rebind(self, topology: TopologySpec) -> None:
        # elastic_do_rebuild performs model/optimizer rebind before returning.
        self._context.topology = topology

    def validate(self, expected: TopologySpec) -> ValidationReport:
        return self.inspect().agrees_with(expected)


class MegatronStateAdapter:
    def __init__(self, context: _Context) -> None:
        self._context = context

    def catalog(self) -> StateCatalog:
        torch = _torch()
        try:
            owner = int(torch.distributed.get_rank())
        except Exception:
            owner = int(os.environ.get("RANK", "0"))
        version = StateVersion(
            self._context.current_step,
            optimizer_generation=self._context.current_step,
            recovery_epoch=self._context.recovery_epoch,
        )
        refs = []
        parameters = dict(_named_parameters(self._context.model))
        parameter_names = {parameter: name for name, parameter in parameters.items()}
        for name, parameter in parameters.items():
            expert = _is_expert_name(name)
            refs.append(
                StateRef(
                    identity=f"parameter/{name}",
                    kind=StateKind.PARAMETER,
                    placement=Placement.UNIQUE if expert else Placement.REPLICATED,
                    owner=owner,
                    version=version,
                    tensor=parameter.data,
                    tags=frozenset({"expert" if expert else "dense"}),
                )
            )

        raw_optimizer = _unwrap_optimizer(self._context.optimizer)
        state = getattr(raw_optimizer, "state", {})
        if isinstance(state, Mapping):
            for parameter, values in state.items():
                name = parameter_names.get(parameter)
                if name is None or not isinstance(values, Mapping):
                    continue
                expert = _is_expert_name(name)
                for key, value in values.items():
                    identity = f"optimizer/{name}/{key}"
                    if hasattr(value, "copy_"):
                        refs.append(
                            StateRef(
                                identity=identity,
                                kind=StateKind.OPTIMIZER_TENSOR,
                                placement=(
                                    Placement.UNIQUE if expert else Placement.SHARDED
                                ),
                                owner=owner,
                                version=version,
                                tensor=value,
                                tags=frozenset({"expert" if expert else "dense"}),
                            )
                        )
        self._context.catalog = StateCatalog.from_iterable(refs)
        return self._context.catalog

    def load_replacement_base(self, plan: RecoveryPlan) -> None:
        # Replacement setup loads its checkpoint base before the adapter is
        # constructed; peer state overwrites it during the legacy rebuild.
        del plan

    def restore(self, plan: RecoveryPlan, sources: StateSources) -> None:
        del plan
        catalog = self._context.catalog or self.catalog()
        for ref in catalog:
            payload = sources.for_ref(ref)
            if payload is None:
                continue
            if ref.tensor is not None:
                source_tensor = getattr(payload, "tensor", payload)
                ref.tensor.copy_(source_tensor)
            elif ref.scalar_set is not None:
                ref.scalar_set(payload)

    def validate_state(self, plan: RecoveryPlan) -> ValidationReport:
        refs = self._context.catalog or self.catalog()
        missing = [
            ref.identity
            for ref in refs
            if ref.version.committed_step < plan.resume_step
        ]
        if missing:
            return ValidationReport.failure(
                "state versions precede resume step", missing=missing[:20]
            )
        return ValidationReport.success(state_count=len(refs))


class MegatronOptimizerAdapter:
    def __init__(self, context: _Context, state: MegatronStateAdapter) -> None:
        self._context = context
        self._state = state

    def local_state_refs(self) -> Sequence[StateRef]:
        return tuple(
            ref
            for ref in self._state.catalog()
            if ref.kind in (StateKind.OPTIMIZER_TENSOR, StateKind.OPTIMIZER_SCALAR)
        )

    def before_step(self, step: int) -> None:
        _client().elastic_zero2_wait_before_optimizer_step(int(step))

    def after_step(self, step: int, committed: bool) -> None:
        next_step = int(step) + 1
        _client().elastic_zero2_schedule_after_optimizer_step(next_step)
        if committed:
            self._context.current_step = next_step

    def rebind(self, topology: TopologySpec) -> None:
        del topology
        # The legacy rebuild calls this before state synchronization.  Calling
        # validation here catches any wrapper the bridge missed.
        _client().elastic_validate_optimizer_process_groups(
            self._context.optimizer,
            iteration=self._context.current_step,
        )


class MegatronTrainingAdapter:
    def __init__(self, context: _Context) -> None:
        self._context = context

    def current_progress(self) -> ProgressToken:
        return ProgressToken(
            step=self._context.current_step,
            recovery_epoch=self._context.recovery_epoch,
            committed=True,
        )

    def quiesce(self, request: PauseRequest) -> QuiescenceProof:
        summary = _client().elastic_zero2_quiesce_for_recovery(
            self._context.current_step
        )
        self._context.recovery_epoch = request.recovery_epoch
        return QuiescenceProof(
            rank=int(os.environ.get("RANK", "0")),
            progress=self.current_progress(),
            collectives_drained=True,
            process_groups_destroyed=False,
            details={"optimizer_replica": summary},
        )

    def apply_resume(self, plan: RecoveryPlan) -> None:
        self._context.resume_step = plan.resume_step
        self._context.current_step = plan.resume_step
        self._context.recovery_epoch = plan.recovery_epoch
        if self._context.args is not None:
            _client().elastic_align_resume_state(
                self._context.args,
                self._context.scheduler,
                plan.resume_step,
            )

    def reset_transients(self, plan: RecoveryPlan) -> None:
        _client()._elastic_reset_rerun_state_machine(plan.resume_step)

    def warmup_and_validate(self, plan: RecoveryPlan) -> ValidationReport:
        del plan
        try:
            _client().elastic_validate_optimizer_process_groups(
                self._context.optimizer,
                iteration=self._context.current_step,
            )
        except Exception as exc:
            return ValidationReport.failure(str(exc))
        return ValidationReport.success(optimizer_groups=True)


MEGATRON_CAPABILITIES = AdapterCapabilities(
    static_world_replacement=True,
    selective_group_rebuild=True,
    full_group_rebuild=True,
    optimizer_memory_replication=True,
    peer_parameter_restore=True,
    moe_state_classification=True,
    two_phase_optimizer_restore=True,
    supported_zero_stages=frozenset({0, 2}),
    supported_parallel_axes=frozenset({"dp", "tp", "pp", "ep", "etp", "cp"}),
)


def build_megatron_adapter(
    model: Any,
    optimizer: Any,
    scheduler: Any = None,
    args: Any = None,
) -> FrameworkAdapter:
    context = _Context(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args,
        current_step=int(getattr(args, "iteration", 0) if args is not None else 0),
        resume_step=int(getattr(args, "iteration", 0) if args is not None else 0),
    )
    state = MegatronStateAdapter(context)
    return FrameworkAdapter(
        name="megatron",
        topology=MegatronTopologyAdapter(context),
        state=state,
        optimizer=MegatronOptimizerAdapter(context, state),
        training=MegatronTrainingAdapter(context),
        capabilities=MEGATRON_CAPABILITIES,
        version="0.1",
        support_level=SupportLevel.EXPERIMENTAL,
        recovery_driver=MegatronRecoveryDriver(context),
    )


class MegatronAdapterPlugin:
    name = "megatron"

    @classmethod
    def build(cls, **kwargs: Any) -> FrameworkAdapter:
        return build_megatron_adapter(**kwargs)
