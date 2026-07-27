"""Framework adapter for ordinary PyTorch DistributedDataParallel training.

This is the small, worked conformance implementation for adapter authors.  It
does not import PyTorch until an operation actually needs it, so discovery and
configuration remain usable on control-plane machines without PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple
from urllib.parse import quote, unquote

from ..capabilities import AdapterCapabilities, SupportLevel
from ..distributed.c10d_backend import ConservativeDistributedBackend
from ..distributed.topology import GroupSpec, TopologySpec, ValidationReport
from ..errors import AdapterUnsupportedError, ContractViolation, StateUnavailable
from ..runtime.recovery_plan import RecoveryPlan
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

    def version(self) -> StateVersion:
        return StateVersion(
            committed_step=self.committed_step,
            optimizer_generation=self.optimizer_generation,
            recovery_epoch=self.recovery_epoch,
        )


def _named_parameters(context: _DDPContext) -> Tuple[Tuple[str, Any], ...]:
    module = _active_module(context)
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


def _active_module(context: _DDPContext) -> Any:
    if isinstance(context.module, RebindableModel):
        return context.module.current
    return context.module


def _is_live_ddp(module: Any) -> bool:
    return module is not None and (
        hasattr(module, "process_group") or hasattr(module, "reducer")
    )


def _default_ddp_rebuilder(old_wrapper: Any, process_group: Any) -> Any:
    torch = _require_torch()
    ddp_type = torch.nn.parallel.DistributedDataParallel
    if not isinstance(old_wrapper, ddp_type):
        raise AdapterUnsupportedError(
            "default DDP reconstruction only supports torch DistributedDataParallel; "
            "provide module_rebuilder for custom wrappers"
        )
    base_module = old_wrapper.module
    kwargs = {
        "process_group": process_group,
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
        self._context.module_requires_wrap = bool(
            not _is_live_ddp(module)
            and self._context.module_rebuilder is not None
            and self._context.rank in plan.failed_ranks
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
        parameter_names = {parameter: name for name, parameter in _named_parameters(self._context)}
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
                key = _identity_component(state_key)
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
        self._context.pending_step = None
        if committed:
            self._context.committed_step = step

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
                    "tensor": True,
                },
            )
            for name, parameter in _named_parameters(self._context)
        )
        return StateCatalog.from_iterable(
            tuple(parameter_refs) + tuple(self._optimizer_adapter.local_state_refs())
        )

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
        if ref.tensor is not None:
            dist.broadcast(ref.tensor, src=source_rank, group=group)
            return
        if ref.scalar_get is None or ref.scalar_set is None:
            raise StateUnavailable(
                f"scalar state {ref.identity!r} has no read/write accessors"
            )
        rank = int(dist.get_rank())
        payload = [ref.scalar_get() if rank == source_rank else None]
        dist.broadcast_object_list(payload, src=source_rank, group=group)
        ref.scalar_set(payload[0])

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
                state_key = unquote(parts[2])
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
                device = getattr(parameter, "device", None)
                parameter_state[state_key] = torch.zeros(shape, dtype=dtype, device=device)
            else:
                kind = str(metadata.get("python_type", "int"))
                parameter_state[state_key] = 0.0 if kind == "float" else 0

    def restore(self, plan: RecoveryPlan, sources: StateSources) -> None:
        torch = _require_torch()
        self._materialize_optimizer_slots(plan)
        missing = []
        restored_identities = set()
        with torch.no_grad():
            for ref in self.catalog():
                source = sources.for_ref(ref)
                if source is None:
                    missing.append(ref.identity)
                    continue
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
        if missing:
            raise StateUnavailable(
                "missing recovery sources for: " + ", ".join(sorted(missing))
            )
        planned = set(plan.state_sources)
        if planned and planned != restored_identities:
            absent = sorted(planned - restored_identities)
            unexpected = sorted(restored_identities - planned)
            raise StateUnavailable(
                "rebuilt Generic DDP state catalog differs from the frozen plan "
                f"(absent={absent}, unexpected={unexpected})"
            )
        self._context.committed_step = plan.resume_step
        self._context.recovery_epoch = plan.recovery_epoch

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

    def quiesce(self, request: PauseRequest) -> QuiescenceProof:
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
    )
    can_rebind = isinstance(module, RebindableModel) or (
        module_rebuilder is not None and module_setter is not None
    )
    capabilities = AdapterCapabilities(
        static_world_replacement=bool(can_rebind and replacement_loader is not None),
        selective_group_rebuild=False,
        full_group_rebuild=can_rebind,
        optimizer_memory_replication=False,
        peer_parameter_restore=True,
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
        version="0.1",
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
