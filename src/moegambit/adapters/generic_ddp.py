"""Framework adapter for ordinary PyTorch DistributedDataParallel training.

This is the small, worked conformance implementation for adapter authors.  It
does not import PyTorch until an operation actually needs it, so discovery and
configuration remain usable on control-plane machines without PyTorch.
"""

from __future__ import annotations

import inspect
import pickle
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple
from urllib.parse import quote, unquote

from ..capabilities import AdapterCapabilities, SupportLevel
from ..distributed.c10d_backend import ConservativeDistributedBackend
from ..distributed.topology import GroupSpec, TopologySpec, ValidationReport
from ..errors import AdapterUnsupportedError, ContractViolation, StateUnavailable
from ..runtime.recovery_plan import RecoveryPlan
from ..policy.resolver import SourceQuery
from ..state.catalog import (
    Placement,
    StateCatalog,
    StateKind,
    StateRef,
    StateSource,
    StateSourceKind,
)
from ..state.version import StateVersion
from .base import (
    FrameworkAdapter,
    PauseRequest,
    ProgressToken,
    QuiescenceProof,
    RebuildHandle,
    StateSources,
    StoreHandle,
)

__all__ = [
    "GenericDDPTopologyAdapter",
    "GenericDDPStateAdapter",
    "GenericDDPOptimizerAdapter",
    "GenericDDPTrainingAdapter",
    "GenericDDPPlugin",
    "RebindableModel",
    "peer_state_sources",
    "build_generic_ddp_adapter",
]

_MAX_SERIALIZED_SCALAR_BYTES = 1 << 20


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise AdapterUnsupportedError(
            "the generic_ddp adapter requires PyTorch; install moegambit[torch]"
        ) from exc
    return torch


class RebindableModel:
    """Stable training-loop handle whose wrapped DDP object can be replaced.

    Reconstructing DDP necessarily creates a new Python object and reducer.  A
    caller holding the old wrapper cannot be repaired by metadata mutation, so
    the public example and adapter use this indirection to preserve object
    ownership across recovery.
    """

    def __init__(self, module: Any) -> None:
        object.__setattr__(self, "_current", module)

    @property
    def current(self) -> Any:
        return object.__getattribute__(self, "_current")

    def replace(self, module: Any) -> Any:
        if module is None:
            raise ValueError("replacement model cannot be None")
        previous = self.current
        object.__setattr__(self, "_current", module)
        return previous

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.current(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.current, name)


@dataclass
class _DDPContext:
    module: Any = None
    optimizer: Any = None
    backend: str = "gloo"
    rank: Optional[int] = None
    world_size: Optional[int] = None
    generation: int = 0
    topology: Optional[TopologySpec] = None
    dp_group: Any = None
    committed_step: int = 0
    optimizer_generation: int = 0
    recovery_epoch: int = 0
    pending_step: Optional[int] = None
    replacement_loader: Optional[Callable[[RecoveryPlan], None]] = None
    transient_reset: Optional[Callable[[RecoveryPlan], None]] = None
    warmup_validator: Optional[Callable[[RecoveryPlan], Any]] = None
    module_rebuilder: Optional[Callable[[Any, Any], Any]] = None
    module_setter: Optional[Callable[[Any], None]] = None
    module_requires_wrap: bool = False
    buffers_replicated: bool = True
    committed_buffers: Dict[str, Any] = field(default_factory=dict)
    committed_buffers_step: Optional[int] = None

    def version(self) -> StateVersion:
        return StateVersion(
            committed_step=self.committed_step,
            optimizer_generation=self.optimizer_generation,
            recovery_epoch=self.recovery_epoch,
        )


def _named_parameters(context: _DDPContext) -> Tuple[Tuple[str, Any], ...]:
    module = _state_module(context)
    if module is None or not callable(
        getattr(module, "named_parameters", None)
    ):
        raise ContractViolation("generic_ddp requires a module with named_parameters()")
    named = tuple(module.named_parameters())
    names = [name for name, _ in named]
    if any(not isinstance(name, str) or not name for name in names):
        raise ContractViolation("every DDP parameter must have a non-empty name")
    if len(names) != len(set(names)):
        raise ContractViolation("DDP parameter names must be unique")
    return named


def _named_buffers(context: _DDPContext) -> Tuple[Tuple[str, Any], ...]:
    module = _state_module(context)
    named_buffers = getattr(module, "named_buffers", None)
    if not callable(named_buffers):
        return ()
    named = tuple(named_buffers())
    names = [name for name, _ in named]
    if any(not isinstance(name, str) or not name for name in names):
        raise ContractViolation("every DDP buffer must have a non-empty name")
    if len(names) != len(set(names)):
        raise ContractViolation("DDP buffer names must be unique")
    if named and not context.buffers_replicated:
        raise AdapterUnsupportedError(
            "generic peer restore requires broadcast_buffers=True when the "
            "model contains buffers"
        )
    return named


def _active_module(context: _DDPContext) -> Any:
    if isinstance(context.module, RebindableModel):
        return context.module.current
    return context.module


def _state_module(context: _DDPContext) -> Any:
    """Return the underlying nn.Module with wrapper-independent names."""

    module = _active_module(context)
    if _is_live_ddp(module) and hasattr(module, "module"):
        return module.module
    return module


def _is_live_ddp(module: Any) -> bool:
    return module is not None and (
        hasattr(module, "process_group") or hasattr(module, "reducer")
    )


def _synchronize_and_snapshot_buffers(
    context: _DDPContext,
    committed_step: int,
) -> None:
    if context.committed_buffers_step == int(committed_step):
        return
    buffers = _named_buffers(context)
    if not buffers:
        context.committed_buffers = {}
        context.committed_buffers_step = int(committed_step)
        return
    torch = _require_torch()
    dist = torch.distributed
    if not dist.is_available() or not dist.is_initialized():
        raise ContractViolation(
            "buffer snapshot requires initialized torch.distributed"
        )
    group = context.dp_group
    backend = str(dist.get_backend(group)).lower()
    with torch.no_grad():
        for name, buffer in buffers:
            device_type = str(buffer.device.type)
            if "nccl" in backend and device_type != "cuda":
                raise AdapterUnsupportedError(
                    f"NCCL cannot commit CPU model buffer {name!r}"
                )
            if "gloo" in backend and device_type != "cpu":
                raise AdapterUnsupportedError(
                    f"Gloo cannot commit non-CPU model buffer {name!r}"
                )
            dist.broadcast(buffer, src=0, group=group)
        context.committed_buffers = {
            name: buffer.detach().clone() for name, buffer in buffers
        }
    context.committed_buffers_step = int(committed_step)


def _restore_committed_buffers(context: _DDPContext) -> None:
    if not context.committed_buffers:
        return
    torch = _require_torch()
    with torch.no_grad():
        for name, buffer in _named_buffers(context):
            snapshot = context.committed_buffers.get(name)
            if snapshot is None:
                raise ContractViolation(
                    f"committed buffer snapshot lacks {name!r}"
                )
            buffer.copy_(snapshot)


def _default_ddp_kwargs(old_wrapper: Any, process_group: Any) -> Dict[str, Any]:
    torch = _require_torch()
    ddp_type = torch.nn.parallel.DistributedDataParallel
    if not isinstance(old_wrapper, ddp_type):
        raise AdapterUnsupportedError(
            "default DDP reconstruction only supports torch DistributedDataParallel; "
            "provide module_rebuilder for custom wrappers"
        )
    parameters = inspect.signature(ddp_type).parameters
    if "init_sync" not in parameters:
        raise AdapterUnsupportedError(
            "this PyTorch DDP cannot disable constructor parameter sync; "
            "provide module_rebuilder so a replacement rank cannot overwrite "
            "the surviving peer before state restore"
        )
    if getattr(old_wrapper, "device_mesh", None) is not None:
        raise AdapterUnsupportedError(
            "default DDP reconstruction does not support device_mesh; "
            "provide module_rebuilder"
        )
    if getattr(old_wrapper, "_delay_all_reduce_params", None):
        raise AdapterUnsupportedError(
            "default DDP reconstruction does not preserve delayed all-reduce "
            "hooks; provide module_rebuilder"
        )
    if getattr(old_wrapper, "_comm_hooks", None):
        raise AdapterUnsupportedError(
            "default DDP reconstruction does not preserve communication hooks; "
            "provide module_rebuilder"
        )
    kwargs = {
        "process_group": process_group,
        # All model state is restored explicitly after rebind. Constructor
        # sync would be destructive when logical rank 0 is the replacement:
        # its stale baseline would overwrite the surviving source rank.
        "init_sync": False,
        "device_ids": getattr(old_wrapper, "device_ids", None),
        "output_device": getattr(old_wrapper, "output_device", None),
        "broadcast_buffers": getattr(old_wrapper, "broadcast_buffers", True),
        "find_unused_parameters": getattr(
            old_wrapper, "find_unused_parameters", False
        ),
        "gradient_as_bucket_view": getattr(
            old_wrapper, "gradient_as_bucket_view", False
        ),
        "static_graph": getattr(old_wrapper, "static_graph", False),
    }
    bucket_bytes = getattr(old_wrapper, "bucket_bytes_cap", None)
    if bucket_bytes is not None:
        kwargs["bucket_cap_mb"] = float(bucket_bytes) / (1024.0 * 1024.0)
    optional_attributes = {
        "dim": "dim",
        "mixed_precision": "_mixed_precision",
        "skip_all_reduce_unused_params": "skip_all_reduce_unused_params",
        "batched_grad_copy": "batched_grad_copy",
        "bucket_cap_mb_list": "bucket_cap_mb_list",
    }
    for argument, attribute in optional_attributes.items():
        if argument not in parameters or not hasattr(old_wrapper, attribute):
            continue
        value = getattr(old_wrapper, attribute)
        if value is not None:
            kwargs[argument] = value
    return kwargs


def _default_ddp_rebuilder(old_wrapper: Any, process_group: Any) -> Any:
    torch = _require_torch()
    ddp_type = torch.nn.parallel.DistributedDataParallel
    kwargs = _default_ddp_kwargs(old_wrapper, process_group)
    base_module = old_wrapper.module
    cleanup = getattr(old_wrapper, "_remove_autograd_hooks", None)
    if callable(cleanup):
        cleanup()
    return ddp_type(base_module, **kwargs)


def _identity_component(value: Any) -> str:
    if not isinstance(value, (str, int, float, bool)):
        raise ContractViolation(
            "optimizer state keys must be strings or primitive scalar values"
        )
    return quote(str(value), safe="._-")


def _typed_identity_component(value: Any) -> str:
    """Keep optimizer keys such as ``1`` and ``"1"`` distinct."""

    if isinstance(value, bool):
        kind = "bool"
    elif isinstance(value, int):
        kind = "int"
    elif isinstance(value, float):
        kind = "float"
    elif isinstance(value, str):
        kind = "str"
    else:
        raise ContractViolation(
            "optimizer state keys must be strings or primitive scalar values"
        )
    return quote(f"{kind}:{value}", safe="._-")


def _decode_typed_identity_component(value: str) -> Any:
    decoded = unquote(value)
    kind, separator, payload = decoded.partition(":")
    if not separator:
        # Compatibility with the initial Phase E identity format.
        return decoded
    if kind == "str":
        return payload
    if kind == "int":
        return int(payload)
    if kind == "float":
        return float(payload)
    if kind == "bool":
        return payload == "True"
    return decoded


class GenericDDPTopologyAdapter:
    """Own the default world group and one explicitly-created DP group."""

    def __init__(
        self,
        backend: str = "gloo",
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        generation: int = 0,
        _context: Optional[_DDPContext] = None,
    ) -> None:
        self._context = _context or _DDPContext(
            backend=backend,
            rank=rank,
            world_size=world_size,
            generation=generation,
        )

    def inspect(self) -> TopologySpec:
        torch = _require_torch()
        dist = torch.distributed
        if not dist.is_available() or not dist.is_initialized():
            raise AdapterUnsupportedError("torch.distributed is not initialized")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        backend = str(dist.get_backend())
        ranks = tuple(range(world_size))
        topology = TopologySpec(
            world_size=world_size,
            rank=rank,
            logical_axes={"dp": world_size},
            coordinates={"dp": rank},
            groups=(
                GroupSpec("world", ranks, backend, "world", 0),
                GroupSpec("data_parallel", ranks, backend, "data_parallel", 1),
            ),
            generation=self._context.generation,
        ).sealed()
        self._context.rank = rank
        self._context.world_size = world_size
        # Rebuild must use the backend that is actually active, not the
        # constructor default.  Otherwise an NCCL job would silently recreate
        # WORLD with Gloo after recovery.
        self._context.backend = backend
        self._context.topology = topology
        return topology

    def prepare_rebuild(self, plan: RecoveryPlan) -> RebuildHandle:
        module = _active_module(self._context)
        can_replace = isinstance(self._context.module, RebindableModel) or (
            self._context.module_rebuilder is not None
            and self._context.module_setter is not None
        )
        if _is_live_ddp(module) and not can_replace:
            raise AdapterUnsupportedError(
                "a live DDP reducer requires RebindableModel or both "
                "module_rebuilder/module_setter; refusing before process-group "
                "teardown"
            )
        if _is_live_ddp(module) and self._context.module_rebuilder is None:
            # Validate every default-rebuilder assumption while the old WORLD
            # is still intact. Any unsupported DDP feature must fail before
            # quiesce can retire communicators.
            _default_ddp_kwargs(module, None)
        is_replacement = self._context.rank in plan.failed_ranks
        if not _is_live_ddp(module) and is_replacement:
            if self._context.module_rebuilder is None:
                raise AdapterUnsupportedError(
                    "a bare replacement module requires module_rebuilder to "
                    "construct its DDP wrapper after rendezvous"
                )
        self._context.module_requires_wrap = bool(
            not _is_live_ddp(module)
            and self._context.module_rebuilder is not None
            and is_replacement
        )
        return RebuildHandle(
            recovery_epoch=plan.recovery_epoch,
            payload={"next_generation": plan.topology_generation},
        )

    def rebuild(self, plan: RecoveryPlan, store: StoreHandle) -> TopologySpec:
        torch = _require_torch()
        dist = torch.distributed
        rank = self._context.rank
        world_size = self._context.world_size
        if rank is None or world_size is None:
            raise ContractViolation(
                "rank and world_size must be supplied or captured before DDP rebuild"
            )
        if dist.is_initialized():
            if self._context.dp_group is not None:
                dist.destroy_process_group(self._context.dp_group)
                self._context.dp_group = None
            dist.destroy_process_group()
        rendezvous = dist.TCPStore(
            store.host,
            store.port,
            world_size,
            rank == 0,
            timedelta(seconds=store.timeout_s),
        )
        prefixed_store = dist.PrefixStore(store.prefix, rendezvous)
        dist.init_process_group(
            backend=self._context.backend,
            store=prefixed_store,
            rank=rank,
            world_size=world_size,
        )
        # The explicit DP group is recreated after WORLD on every rank.  That
        # creation order is part of the c10d communicator identity contract.
        self._context.dp_group = dist.new_group(ranks=list(range(world_size)))
        self._context.generation = plan.topology_generation
        return self.inspect()

    def rebind(self, topology: TopologySpec) -> None:
        module = _active_module(self._context)
        if _is_live_ddp(module) or self._context.module_requires_wrap:
            can_replace = isinstance(self._context.module, RebindableModel) or (
                self._context.module_rebuilder is not None
                and self._context.module_setter is not None
            )
            if not can_replace:
                raise AdapterUnsupportedError(
                    "cannot rebind a live DDP reducer without a replaceable owner"
                )
            rebuild = self._context.module_rebuilder or _default_ddp_rebuilder
            replacement = rebuild(module, self._context.dp_group)
            if isinstance(self._context.module, RebindableModel):
                self._context.module.replace(replacement)
            elif self._context.module_setter is not None:
                self._context.module_setter(replacement)
                self._context.module = replacement
            else:  # protected by prepare_rebuild; retain fail-closed invariant
                raise AdapterUnsupportedError(
                    "rebuilt DDP wrapper has no owner capable of accepting it"
                )
            self._context.module_requires_wrap = False
        self._context.topology = topology
        self._context.generation = topology.generation

    def validate(self, expected: TopologySpec) -> ValidationReport:
        return self.inspect().agrees_with(expected)


class GenericDDPOptimizerAdapter:
    """Expose the naturally replicated optimizer state of plain DDP."""

    def __init__(
        self,
        module: Any = None,
        optimizer: Any = None,
        _context: Optional[_DDPContext] = None,
    ) -> None:
        self._context = _context or _DDPContext(module=module, optimizer=optimizer)

    def local_state_refs(self) -> Sequence[StateRef]:
        torch = _require_torch()
        optimizer = self._context.optimizer
        if optimizer is None or not isinstance(getattr(optimizer, "state", None), Mapping):
            raise ContractViolation("generic_ddp requires an optimizer with state")
        parameter_names = {
            parameter: name for name, parameter in _named_parameters(self._context)
        }
        refs = []
        for parameter, state in optimizer.state.items():
            if parameter not in parameter_names:
                # StateRef identities must derive from stable module parameter
                # names, never id(), memory addresses, or iteration positions.
                raise ContractViolation("optimizer contains a parameter not named by the module")
            parameter_name = _identity_component(parameter_names[parameter])
            if not isinstance(state, Mapping):
                raise ContractViolation("optimizer parameter state must be a mapping")
            for state_key, value in state.items():
                key = _typed_identity_component(state_key)
                identity = f"optim/{parameter_name}/{key}"
                if torch.is_tensor(value):
                    refs.append(
                        StateRef(
                            identity=identity,
                            kind=StateKind.OPTIMIZER_TENSOR,
                            placement=Placement.REPLICATED,
                            owner=self._context.rank or 0,
                            version=self._context.version(),
                            tensor=value,
                            metadata={
                                "parameter_name": parameter_names[parameter],
                                "state_key": state_key,
                                "shape": [int(size) for size in value.shape],
                                "dtype": str(value.dtype),
                                "device_type": str(value.device.type),
                                "tensor": True,
                            },
                        )
                    )
                else:
                    def get_scalar(
                        parameter: Any = parameter, state_key: Any = state_key
                    ) -> Any:
                        return optimizer.state[parameter][state_key]

                    def set_scalar(
                        new_value: Any,
                        parameter: Any = parameter,
                        state_key: Any = state_key,
                    ) -> None:
                        optimizer.state[parameter][state_key] = new_value

                    refs.append(
                        StateRef(
                            identity=identity,
                            kind=StateKind.OPTIMIZER_SCALAR,
                            placement=Placement.REPLICATED,
                            owner=self._context.rank or 0,
                            version=self._context.version(),
                            scalar_get=get_scalar,
                            scalar_set=set_scalar,
                            metadata={
                                "parameter_name": parameter_names[parameter],
                                "state_key": state_key,
                                "tensor": False,
                                "python_type": type(value).__name__,
                            },
                        )
                    )
        param_groups = getattr(optimizer, "param_groups", None)
        if not isinstance(param_groups, Sequence):
            raise ContractViolation("generic_ddp optimizer requires ordered param_groups")
        for group_index, group in enumerate(param_groups):
            if not isinstance(group, Mapping):
                raise ContractViolation("optimizer param_group must be a mapping")
            group_parameters = tuple(group.get("params", ()))
            try:
                group_parameter_names = tuple(
                    parameter_names[parameter] for parameter in group_parameters
                )
            except KeyError as exc:
                raise ContractViolation(
                    "optimizer param_group contains a parameter not named by the module"
                ) from exc
            for option_key, value in group.items():
                if option_key in ("params", "param_names"):
                    continue
                identity = (
                    f"optim_group/{group_index}/"
                    f"{_typed_identity_component(option_key)}"
                )
                metadata = {
                    "group_index": group_index,
                    "option_key": option_key,
                    "parameter_names": list(group_parameter_names),
                    "python_type": type(value).__name__,
                }
                if torch.is_tensor(value):
                    metadata.update(
                        {
                            "shape": [int(size) for size in value.shape],
                            "dtype": str(value.dtype),
                            "device_type": str(value.device.type),
                            "tensor": True,
                        }
                    )
                    refs.append(
                        StateRef(
                            identity=identity,
                            kind=StateKind.OPTIMIZER_TENSOR,
                            placement=Placement.REPLICATED,
                            owner=self._context.rank or 0,
                            version=self._context.version(),
                            tensor=value,
                            tags=frozenset({"optimizer_group"}),
                            metadata=metadata,
                        )
                    )
                    continue

                def get_group_option(
                    group: Mapping[str, Any] = group,
                    option_key: Any = option_key,
                ) -> Any:
                    return group[option_key]

                def set_group_option(
                    new_value: Any,
                    group: Any = group,
                    option_key: Any = option_key,
                ) -> None:
                    group[option_key] = new_value

                metadata["tensor"] = False
                refs.append(
                    StateRef(
                        identity=identity,
                        kind=StateKind.OPTIMIZER_SCALAR,
                        placement=Placement.REPLICATED,
                        owner=self._context.rank or 0,
                        version=self._context.version(),
                        scalar_get=get_group_option,
                        scalar_set=set_group_option,
                        tags=frozenset({"optimizer_group"}),
                        metadata=metadata,
                    )
                )
        return tuple(sorted(refs, key=lambda ref: ref.identity))

    def before_step(self, step: int) -> None:
        if self._context.pending_step is not None:
            raise ContractViolation("optimizer step already in progress")
        self._context.pending_step = step

    def after_step(self, step: int, committed: bool) -> None:
        if self._context.pending_step != step:
            raise ContractViolation(
                f"optimizer step boundary mismatch: {self._context.pending_step} != {step}"
            )
        if committed:
            # Model buffers can mutate during forward independently on each
            # rank. Synchronize rank 0's authoritative value and snapshot it
            # before declaring this state version committed.
            _synchronize_and_snapshot_buffers(self._context, step)
            self._context.committed_step = step
            self._context.optimizer_generation = step
        else:
            _restore_committed_buffers(self._context)
        self._context.pending_step = None

    def rebind(self, topology: TopologySpec) -> None:
        self._context.topology = topology


class GenericDDPStateAdapter:
    """Catalog and restore named parameters plus optimizer state."""

    def __init__(
        self,
        module: Any = None,
        optimizer: Any = None,
        replacement_loader: Optional[Callable[[RecoveryPlan], None]] = None,
        _context: Optional[_DDPContext] = None,
    ) -> None:
        self._context = _context or _DDPContext(
            module=module,
            optimizer=optimizer,
            replacement_loader=replacement_loader,
        )
        self._optimizer_adapter = GenericDDPOptimizerAdapter(_context=self._context)

    def catalog(self) -> StateCatalog:
        _require_torch()
        parameter_refs = (
            StateRef(
                # Stable names are mandatory: replacement processes cannot
                # reproduce id(), addresses, or incidental iteration indices.
                identity=f"param/{_identity_component(name)}",
                kind=StateKind.PARAMETER,
                placement=Placement.REPLICATED,
                owner=self._context.rank or 0,
                version=self._context.version(),
                tensor=parameter,
                metadata={
                    "parameter_name": name,
                    "shape": [int(size) for size in parameter.shape],
                    "dtype": str(parameter.dtype),
                    "device_type": str(parameter.device.type),
                    "tensor": True,
                },
            )
            for name, parameter in _named_parameters(self._context)
        )
        buffer_refs = (
            StateRef(
                identity=f"buffer/{_identity_component(name)}",
                kind=StateKind.BUFFER,
                placement=Placement.REPLICATED,
                owner=self._context.rank or 0,
                version=self._context.version(),
                tensor=buffer,
                tags=frozenset({"buffer"}),
                metadata={
                    "buffer_name": name,
                    "shape": [int(size) for size in buffer.shape],
                    "dtype": str(buffer.dtype),
                    "device_type": str(buffer.device.type),
                    "tensor": True,
                },
            )
            for name, buffer in _named_buffers(self._context)
        )
        return StateCatalog.from_iterable(
            tuple(parameter_refs)
            + tuple(buffer_refs)
            + tuple(self._optimizer_adapter.local_state_refs())
        )

    def source_candidates(
        self,
        query: SourceQuery,
    ) -> Mapping[str, Sequence[StateSource]]:
        """Advertise a current peer copy on every surviving DDP rank.

        Plain DDP state is fully replicated.  The locator remains a logical
        global rank even when this method runs on another rank, making the
        serialized inventory identical across participants.
        """

        catalog = self.catalog()
        if catalog.manifest_digest() != query.state_catalog.manifest_digest():
            raise ContractViolation(
                "source query catalog differs from Generic DDP catalog"
            )
        survivors = tuple(
            rank
            for rank in range(int(query.world_size))
            if rank not in query.failed_ranks
        )
        if not survivors:
            raise StateUnavailable("no surviving DDP peer can provide state")
        candidates: Dict[str, Sequence[StateSource]] = {}
        for ref in catalog:
            if ref.placement is not Placement.REPLICATED:
                candidates[ref.identity] = ()
                continue
            metadata = {
                "kind": ref.kind.value,
                "placement": ref.placement.value,
                "owner": -1,
                "tags": sorted(ref.tags),
                **dict(ref.metadata),
            }
            candidates[ref.identity] = tuple(
                StateSource(
                    kind=StateSourceKind.PEER,
                    version=ref.version,
                    locator=f"rank://{rank}",
                    metadata=metadata,
                )
                for rank in survivors
            )
        return candidates

    def load_replacement_base(self, plan: RecoveryPlan) -> None:
        if self._context.replacement_loader is not None:
            self._context.replacement_loader(plan)
        elif self._context.module is None:
            raise ContractViolation("replacement worker has no module baseline")

    @staticmethod
    def _source_value(source: Any) -> Any:
        if isinstance(source, StateRef):
            return source.tensor if source.tensor is not None else source.scalar_get()
        if isinstance(source, StateSource):
            if source.kind is StateSourceKind.LOCAL:
                return source
            raise StateUnavailable(
                f"state source {source.kind.value!r} has no in-memory payload"
            )
        if isinstance(source, Mapping):
            for key in ("tensor", "value"):
                if key in source:
                    return source[key]
        if callable(source):
            return source()
        return source

    @staticmethod
    def _peer_rank(source: StateSource) -> int:
        """Parse the stable peer locator used by the control plane.

        Accepted forms intentionally remain small and unambiguous.  A network
        address is not a logical rank and must not be guessed from one.
        """

        locator = source.locator.strip()
        for prefix in ("rank://", "peer://", "rank:", "peer:"):
            if locator.startswith(prefix):
                value = locator[len(prefix) :]
                break
        else:
            value = locator
        try:
            rank = int(value)
        except ValueError as exc:
            raise StateUnavailable(
                f"invalid Generic DDP peer locator {source.locator!r}"
            ) from exc
        if rank < 0:
            raise StateUnavailable("peer rank must be non-negative")
        return rank

    def _collective_device(self, torch: Any, dist: Any, group: Any) -> Any:
        backend = str(dist.get_backend(group)).lower()
        if "nccl" not in backend:
            return torch.device("cpu")
        for _, parameter in _named_parameters(self._context):
            if str(parameter.device.type) == "cuda":
                return parameter.device
        raise StateUnavailable("NCCL recovery requires at least one CUDA parameter")

    def _broadcast_bytes(
        self,
        payload: Optional[bytes],
        *,
        source_rank: int,
        torch: Any,
        dist: Any,
        group: Any,
    ) -> bytes:
        rank = int(dist.get_rank())
        device = self._collective_device(torch, dist, group)
        length = len(payload) if rank == source_rank and payload is not None else 0
        if rank == source_rank and length > _MAX_SERIALIZED_SCALAR_BYTES:
            length = -1
        length_tensor = torch.tensor([length], dtype=torch.int64, device=device)
        dist.broadcast(length_tensor, src=source_rank, group=group)
        length = int(length_tensor.item())
        if length < 0:
            raise StateUnavailable(
                "serialized optimizer option exceeds the 1 MiB safety limit"
            )
        if rank == source_rank:
            data = torch.tensor(list(payload or b""), dtype=torch.uint8, device=device)
        else:
            data = torch.empty(length, dtype=torch.uint8, device=device)
        dist.broadcast(data, src=source_rank, group=group)
        return bytes(data.cpu().tolist())

    def _restore_from_peer(self, ref: StateRef, source: StateSource) -> None:
        torch = _require_torch()
        dist = torch.distributed
        if not dist.is_available() or not dist.is_initialized():
            raise StateUnavailable(
                f"cannot restore {ref.identity!r}: torch.distributed is not initialized"
            )
        source_rank = self._peer_rank(source)
        world_size = int(dist.get_world_size())
        if source_rank >= world_size:
            raise StateUnavailable(
                f"peer rank {source_rank} is outside rebuilt world size {world_size}"
            )
        group = self._context.dp_group
        rank = int(dist.get_rank())
        if ref.kind is StateKind.BUFFER and rank == source_rank:
            buffer_name = str(ref.metadata.get("buffer_name", ""))
            snapshot = self._context.committed_buffers.get(buffer_name)
            if snapshot is None:
                raise StateUnavailable(
                    f"committed buffer snapshot lacks {buffer_name!r}"
                )
            ref.tensor.copy_(snapshot)
        if ref.tensor is not None:
            backend = str(dist.get_backend(group)).lower()
            device_type = str(ref.tensor.device.type)
            if "nccl" in backend and device_type != "cuda":
                device = self._collective_device(torch, dist, group)
                flat = ref.tensor.contiguous().reshape(-1).view(torch.uint8)
                staging = (
                    flat.to(device)
                    if rank == source_rank
                    else torch.empty(flat.numel(), dtype=torch.uint8, device=device)
                )
                dist.broadcast(staging, src=source_rank, group=group)
                if rank != source_rank:
                    flat.copy_(staging.cpu())
                return
            if "gloo" in backend and device_type != "cpu":
                raise StateUnavailable(
                    f"cannot restore {ref.identity!r}: Gloo tensor transfer "
                    f"requires CPU state, got {device_type}"
                )
            dist.broadcast(ref.tensor, src=source_rank, group=group)
            return
        if ref.scalar_get is None or ref.scalar_set is None:
            raise StateUnavailable(
                f"scalar state {ref.identity!r} has no read/write accessors"
            )
        serialized = (
            pickle.dumps(ref.scalar_get(), protocol=pickle.HIGHEST_PROTOCOL)
            if rank == source_rank
            else None
        )
        payload = self._broadcast_bytes(
            serialized,
            source_rank=source_rank,
            torch=torch,
            dist=dist,
            group=group,
        )
        try:
            ref.scalar_set(pickle.loads(payload))
        except Exception as exc:
            raise StateUnavailable(
                f"could not deserialize scalar state {ref.identity!r}"
            ) from exc

    @staticmethod
    def _torch_dtype(torch: Any, name: str) -> Any:
        short = str(name).split(".")[-1]
        dtype = getattr(torch, short, None)
        if dtype is None:
            raise StateUnavailable(f"unsupported optimizer tensor dtype {name!r}")
        return dtype

    def _materialize_optimizer_slots(self, plan: RecoveryPlan) -> None:
        """Create empty replacement optimizer slots from the frozen manifest.

        Adam-style optimizers allocate state lazily on their first step.  A
        replacement has not run that step, so peer restore must materialize the
        exact slots without performing a fake update.  All shape/dtype/key
        information comes from digest-covered source metadata.
        """

        optimizer = self._context.optimizer
        state = getattr(optimizer, "state", None)
        if not isinstance(state, Mapping) or not hasattr(state, "__setitem__"):
            return
        torch = _require_torch()
        parameters = {name: parameter for name, parameter in _named_parameters(self._context)}
        for identity, source in sorted(plan.state_sources.items()):
            if not str(identity).startswith("optim/") or not isinstance(source, StateSource):
                continue
            metadata = dict(source.metadata)
            parameter_name = metadata.get("parameter_name")
            state_key = metadata.get("state_key")
            if parameter_name is None or state_key is None:
                # Compatibility with early plans whose identity was the only
                # descriptor.  Quoting is reversible for ordinary string keys.
                parts = str(identity).split("/", 2)
                if len(parts) != 3:
                    raise StateUnavailable(f"invalid optimizer state identity {identity!r}")
                parameter_name = unquote(parts[1])
                state_key = _decode_typed_identity_component(parts[2])
            parameter = parameters.get(str(parameter_name))
            if parameter is None:
                raise StateUnavailable(
                    f"replacement model lacks optimizer parameter {parameter_name!r}"
                )
            parameter_state = state.setdefault(parameter, {})
            if state_key in parameter_state:
                continue
            if metadata.get("tensor", True):
                shape = tuple(int(size) for size in metadata.get("shape", ()))
                dtype = self._torch_dtype(torch, str(metadata.get("dtype", parameter.dtype)))
                device_type = str(metadata.get("device_type", parameter.device.type))
                device = (
                    torch.device("cpu")
                    if device_type == "cpu"
                    else getattr(parameter, "device", None)
                )
                parameter_state[state_key] = torch.zeros(shape, dtype=dtype, device=device)
            else:
                kind = str(metadata.get("python_type", "int"))
                parameter_state[state_key] = 0.0 if kind == "float" else 0

    @staticmethod
    def _planned_optimizer_generation(plan: RecoveryPlan) -> int:
        versions = {
            (source.version.committed_step, source.version.optimizer_generation)
            for source in plan.state_sources.values()
            if isinstance(source, StateSource)
        }
        if not versions:
            return plan.resume_step
        if any(committed_step != plan.resume_step for committed_step, _ in versions):
            raise StateUnavailable(
                "Generic DDP state source version does not match plan resume_step"
            )
        optimizer_generations = {generation for _, generation in versions}
        if len(optimizer_generations) != 1:
            raise StateUnavailable(
                "Generic DDP plan mixes incompatible optimizer generations"
            )
        return next(iter(optimizer_generations))

    @staticmethod
    def _source_contract_error(ref: StateRef, source: Any) -> Optional[str]:
        if not isinstance(source, StateSource):
            return None
        metadata = dict(source.metadata)
        expected_kind = metadata.get("kind")
        if expected_kind is not None and expected_kind != ref.kind.value:
            return f"{ref.identity}: kind {ref.kind.value} != {expected_kind}"
        expected_placement = metadata.get("placement")
        if expected_placement is not None and expected_placement != ref.placement.value:
            return (
                f"{ref.identity}: placement {ref.placement.value} "
                f"!= {expected_placement}"
            )
        if ref.tensor is not None:
            shape = metadata.get("shape")
            dtype = metadata.get("dtype")
            if shape is None or dtype is None:
                return f"{ref.identity}: peer tensor source lacks shape/dtype metadata"
            observed_shape = [int(size) for size in ref.tensor.shape]
            if observed_shape != [int(size) for size in shape]:
                return f"{ref.identity}: shape {observed_shape} != {list(shape)}"
            if str(ref.tensor.dtype) != str(dtype):
                return f"{ref.identity}: dtype {ref.tensor.dtype} != {dtype}"
            expected_device_type = metadata.get("device_type")
            if (
                expected_device_type is not None
                and str(ref.tensor.device.type) != str(expected_device_type)
            ):
                return (
                    f"{ref.identity}: device type {ref.tensor.device.type} "
                    f"!= {expected_device_type}"
                )
        expected_parameter_names = metadata.get("parameter_names")
        if expected_parameter_names is not None:
            actual_parameter_names = list(ref.metadata.get("parameter_names", ()))
            if actual_parameter_names != list(expected_parameter_names):
                return (
                    f"{ref.identity}: optimizer group parameter membership differs"
                )
        return None

    def _agree_catalog_before_transfer(
        self,
        catalog: StateCatalog,
        plan: RecoveryPlan,
        sources: StateSources,
    ) -> None:
        catalog_identities = {ref.identity for ref in catalog}
        planned_identities = set(plan.state_sources)
        resolved_identities = set(sources.by_identity)
        errors = []
        if catalog_identities != planned_identities:
            errors.append(
                "catalog identities differ from frozen plan "
                f"(missing={sorted(planned_identities - catalog_identities)}, "
                f"unexpected={sorted(catalog_identities - planned_identities)})"
            )
        if resolved_identities != planned_identities:
            errors.append("resolved source identities differ from frozen plan")
        by_identity = {ref.identity: ref for ref in catalog}
        for identity in sorted(catalog_identities & resolved_identities):
            ref = by_identity[identity]
            error = self._source_contract_error(ref, sources.by_identity[identity])
            if error:
                errors.append(error)
            if ref.tensor is None and ref.scalar_get is not None:
                try:
                    encoded = pickle.dumps(
                        ref.scalar_get(), protocol=pickle.HIGHEST_PROTOCOL
                    )
                    if len(encoded) > _MAX_SERIALIZED_SCALAR_BYTES:
                        errors.append(
                            f"{identity}: serialized optimizer option exceeds 1 MiB"
                        )
                except Exception as exc:
                    errors.append(
                        f"{identity}: optimizer option is not serializable "
                        f"({type(exc).__name__})"
                    )
            source = sources.by_identity[identity]
            if (
                ref.kind is StateKind.BUFFER
                and isinstance(source, StateSource)
                and source.kind is StateSourceKind.PEER
                and self._peer_rank(source) == int(self._context.rank or 0)
                and self._context.committed_buffers_step != plan.resume_step
            ):
                errors.append(
                    f"{identity}: source rank lacks a buffer snapshot for "
                    f"committed step {plan.resume_step}"
                )

        peer_transfer = any(
            isinstance(source, StateSource)
            and source.kind is StateSourceKind.PEER
            for source in sources.by_identity.values()
        )
        if not peer_transfer:
            if errors:
                raise StateUnavailable("; ".join(errors))
            return
        torch = _require_torch()
        dist = torch.distributed
        if not dist.is_available() or not dist.is_initialized():
            raise StateUnavailable("peer catalog agreement requires initialized c10d")
        group = self._context.dp_group
        world_size = int(dist.get_world_size(group=group))
        digest = catalog.manifest_digest()
        digest_bytes = bytes.fromhex(digest.removeprefix("sha256:"))
        device = self._collective_device(torch, dist, group)
        local = torch.tensor(
            [1 if errors else 0, *digest_bytes],
            dtype=torch.uint8,
            device=device,
        )
        gathered = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(gathered, local, group=group)
        rank_errors = [
            f"rank {rank}: catalog/source contract rejected locally"
            for rank, item in enumerate(gathered)
            if int(item[0].item()) != 0
        ]
        manifests = {
            bytes(item[1:].cpu().tolist())
            for item in gathered
        }
        if len(manifests) != 1:
            rank_errors.append("state catalog manifest differs across rebuilt ranks")
        if errors:
            rank_errors.extend(errors)
        if rank_errors:
            raise StateUnavailable("; ".join(rank_errors))

    def restore(self, plan: RecoveryPlan, sources: StateSources) -> None:
        torch = _require_torch()
        optimizer_generation = self._planned_optimizer_generation(plan)
        self._materialize_optimizer_slots(plan)
        catalog = self.catalog()
        self._agree_catalog_before_transfer(catalog, plan, sources)
        restored_identities = set()
        with torch.no_grad():
            for ref in catalog:
                source = sources.for_ref(ref)
                # Agreement above proves every source exists.
                restored_identities.add(ref.identity)
                if isinstance(source, StateSource):
                    if source.kind is StateSourceKind.PEER:
                        self._restore_from_peer(ref, source)
                        continue
                    if source.kind in (
                        StateSourceKind.CHECKPOINT,
                        StateSourceKind.LOCAL,
                        StateSourceKind.REINITIALIZE,
                    ):
                        # load_replacement_base() owns whole-checkpoint loading;
                        # LOCAL/REINITIALIZE explicitly retain the current value.
                        continue
                value = self._source_value(source)
                if isinstance(value, StateSource) and value.kind is StateSourceKind.LOCAL:
                    continue
                if ref.tensor is not None:
                    if not torch.is_tensor(value):
                        raise StateUnavailable(
                            f"tensor state {ref.identity!r} has a non-tensor source"
                        )
                    ref.tensor.copy_(value)
                elif ref.scalar_set is not None:
                    ref.scalar_set(value)
        planned = set(plan.state_sources)
        if planned and planned != restored_identities:
            absent = sorted(planned - restored_identities)
            unexpected = sorted(restored_identities - planned)
            raise StateUnavailable(
                "rebuilt Generic DDP state catalog differs from the frozen plan "
                f"(absent={absent}, unexpected={unexpected})"
            )
        self._context.committed_step = plan.resume_step
        self._context.optimizer_generation = optimizer_generation
        self._context.recovery_epoch = plan.recovery_epoch
        self._context.committed_buffers = {
            name: buffer.detach().clone()
            for name, buffer in _named_buffers(self._context)
        }
        self._context.committed_buffers_step = plan.resume_step

    def validate_state(self, plan: RecoveryPlan) -> ValidationReport:
        torch = _require_torch()
        findings = []
        refs = tuple(self.catalog())
        if self._context.committed_step != plan.resume_step:
            findings.append(
                "state committed step does not match the recovery plan resume step"
            )
        for ref in refs:
            if ref.tensor is not None and not bool(torch.isfinite(ref.tensor).all().item()):
                findings.append(f"non-finite tensor state: {ref.identity}")
        if findings:
            return ValidationReport.failure(*findings, state_count=len(refs))
        return ValidationReport.success(state_count=len(refs))


def peer_state_sources(
    state_adapter: GenericDDPStateAdapter,
    source_rank: int,
) -> Mapping[str, StateSource]:
    """Describe a complete, digest-safe peer snapshot for policy/control use."""

    if source_rank < 0:
        raise ValueError("source_rank must be non-negative")
    return {
        ref.identity: StateSource(
            kind=StateSourceKind.PEER,
            version=ref.version,
            locator=f"rank://{int(source_rank)}",
            metadata={
                "kind": ref.kind.value,
                "placement": ref.placement.value,
                "owner": int(ref.owner),
                "tags": sorted(ref.tags),
                **dict(ref.metadata),
            },
        )
        for ref in state_adapter.catalog()
    }


class GenericDDPTrainingAdapter:
    """Coordinate safe-point progress for a conventional DDP loop."""

    def __init__(
        self,
        module: Any = None,
        transient_reset: Optional[Callable[[RecoveryPlan], None]] = None,
        warmup_validator: Optional[Callable[[RecoveryPlan], Any]] = None,
        _context: Optional[_DDPContext] = None,
    ) -> None:
        self._context = _context or _DDPContext(
            module=module,
            transient_reset=transient_reset,
            warmup_validator=warmup_validator,
        )

    def current_progress(self) -> ProgressToken:
        return ProgressToken(
            step=self._context.committed_step,
            recovery_epoch=self._context.recovery_epoch,
            committed=self._context.pending_step is None,
        )

    def iteration_boundary(self, step: int) -> None:
        if self._context.pending_step is not None:
            raise ContractViolation(
                "iteration boundary reached with an uncommitted optimizer step"
            )
        _synchronize_and_snapshot_buffers(
            self._context, self._context.committed_step
        )

    def quiesce(self, request: PauseRequest) -> QuiescenceProof:
        if self._context.pending_step is not None:
            raise ContractViolation(
                "cannot peer-restore Generic DDP while optimizer step "
                f"{self._context.pending_step} is uncommitted"
            )
        torch = _require_torch()
        dist = torch.distributed
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        destroyed = False
        if dist.is_available() and dist.is_initialized():
            # A failed rank cannot participate in the old WORLD barrier.  Safe
            # point agreement is a control-plane proof, never a collective on
            # the communicator being retired.
            if self._context.dp_group is not None:
                dist.destroy_process_group(self._context.dp_group)
                self._context.dp_group = None
            dist.destroy_process_group()
            destroyed = True
        self._context.recovery_epoch = request.recovery_epoch
        return QuiescenceProof(
            rank=self._context.rank or 0,
            progress=self.current_progress(),
            collectives_drained=True,
            process_groups_destroyed=destroyed,
        )

    def apply_resume(self, plan: RecoveryPlan) -> None:
        self._context.committed_step = plan.resume_step
        self._context.recovery_epoch = plan.recovery_epoch
        self._context.pending_step = None

    def reset_transients(self, plan: RecoveryPlan) -> None:
        if self._context.transient_reset is not None:
            self._context.transient_reset(plan)

    def warmup_and_validate(self, plan: RecoveryPlan) -> ValidationReport:
        if self._context.warmup_validator is not None:
            result = self._context.warmup_validator(plan)
            if isinstance(result, ValidationReport):
                return result
            if result is False:
                return ValidationReport.failure("DDP warmup validation failed")
            return ValidationReport.success(warmup="callback")
        torch = _require_torch()
        findings = [
            f"non-finite parameter: {name}"
            for name, parameter in _named_parameters(self._context)
            if not bool(torch.isfinite(parameter).all().item())
        ]
        if findings:
            return ValidationReport.failure(*findings)
        return ValidationReport.success(warmup="parameter_finiteness")


GENERIC_DDP_CAPABILITIES = AdapterCapabilities(
    static_world_replacement=False,
    selective_group_rebuild=False,
    # This constant describes a caller that did not provide a replaceable
    # owner. build_generic_ddp_adapter() upgrades the per-instance capability
    # only when it can construct and install a new DDP wrapper/reducer.
    full_group_rebuild=False,
    optimizer_memory_replication=False,
    peer_parameter_restore=True,
    moe_state_classification=False,
    two_phase_optimizer_restore=False,
    supported_zero_stages=frozenset({0}),
    supported_parallel_axes=frozenset({"dp"}),
)


def build_generic_ddp_adapter(
    module: Any = None,
    optimizer: Any = None,
    backend: str = "gloo",
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    generation: int = 0,
    committed_step: int = 0,
    recovery_epoch: int = 0,
    replacement_loader: Optional[Callable[[RecoveryPlan], None]] = None,
    transient_reset: Optional[Callable[[RecoveryPlan], None]] = None,
    warmup_validator: Optional[Callable[[RecoveryPlan], Any]] = None,
    module_rebuilder: Optional[Callable[[Any, Any], Any]] = None,
    module_setter: Optional[Callable[[Any], None]] = None,
    error_classifier: Optional[Callable[[BaseException], Any]] = None,
) -> FrameworkAdapter:
    """Build a Generic DDP adapter around a module and its optimizer."""

    context = _DDPContext(
        module=module,
        optimizer=optimizer,
        backend=backend,
        rank=rank,
        world_size=world_size,
        generation=generation,
        committed_step=committed_step,
        recovery_epoch=recovery_epoch,
        replacement_loader=replacement_loader,
        transient_reset=transient_reset,
        warmup_validator=warmup_validator,
        module_rebuilder=module_rebuilder,
        module_setter=module_setter,
        buffers_replicated=bool(
            getattr(
                module.current if isinstance(module, RebindableModel) else module,
                "broadcast_buffers",
                True,
            )
        ),
    )
    has_replaceable_owner = isinstance(module, RebindableModel) or (
        module_rebuilder is not None and module_setter is not None
    )
    active_module = module.current if isinstance(module, RebindableModel) else module
    state_module = (
        active_module.module
        if _is_live_ddp(active_module) and hasattr(active_module, "module")
        else active_module
    )
    named_buffers = getattr(state_module, "named_buffers", None)
    has_buffers = bool(tuple(named_buffers())) if callable(named_buffers) else False
    buffers_replicated = bool(getattr(active_module, "broadcast_buffers", True))
    optimizer_state = getattr(optimizer, "state", None)
    optimizer_groups = getattr(optimizer, "param_groups", None)
    peer_restore_ready = bool(
        state_module is not None
        and isinstance(optimizer_state, Mapping)
        and isinstance(optimizer_groups, Sequence)
        and (buffers_replicated or not has_buffers)
    )
    can_rebind = bool(
        has_replaceable_owner
        and (_is_live_ddp(active_module) or module_rebuilder is not None)
    )
    can_replace_failed_rank = bool(
        has_replaceable_owner
        and module_rebuilder is not None
        and peer_restore_ready
    )
    capabilities = AdapterCapabilities(
        static_world_replacement=can_replace_failed_rank,
        selective_group_rebuild=False,
        full_group_rebuild=can_rebind,
        optimizer_memory_replication=False,
        peer_parameter_restore=peer_restore_ready,
        moe_state_classification=False,
        two_phase_optimizer_restore=False,
        supported_zero_stages=frozenset({0}),
        supported_parallel_axes=frozenset({"dp"}),
    )
    return FrameworkAdapter(
        name="generic_ddp",
        topology=GenericDDPTopologyAdapter(_context=context),
        state=GenericDDPStateAdapter(_context=context),
        optimizer=GenericDDPOptimizerAdapter(_context=context),
        training=GenericDDPTrainingAdapter(_context=context),
        capabilities=capabilities,
        version="0.2",
        support_level=(
            SupportLevel.EXPERIMENTAL
            if capabilities.static_world_replacement
            else SupportLevel.DETECTION_ONLY
        ),
        backend=ConservativeDistributedBackend(error_classifier),
    )


class GenericDDPPlugin:
    """Entry-point object for the Generic DDP reference adapter."""

    name = "generic_ddp"

    @classmethod
    def build(cls, **kwargs: Any) -> FrameworkAdapter:
        return build_generic_ddp_adapter(**kwargs)
