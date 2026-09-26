# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

"""Packed AutoEP checkpoint shards used by MoEGambit recovery."""

from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from deepspeed.checkpoint.constants import FOLDING_METADATA_KEY


PACKED_EXPERT_CHECKPOINT_ENV = "DEEPSPEED_MOEGAMBIT_PACKED_EXPERT_CHECKPOINT"
PACKED_EXPERT_CACHE_ENV = "MOEGAMBIT_STANDBY_PACKED_EXPERT_CACHE"
PACKED_EXPERT_FORMAT = "autoep_fused_local_shard"
PACKED_EXPERT_FORMAT_VERSION = 1
PACKED_EXPERT_FORMAT_KEY = "packed_expert_format"
PACKED_EXPERT_VERSION_KEY = "packed_expert_format_version"
PACKED_EXPERT_TENSORS_KEY = "expert_tensors"
PACKED_EXPERT_SUFFIX = "_packed_expert_states.pt"

_PACKED_NAME = re.compile(
    r"^layer_(?P<layer>\d+)_ep_rank_(?P<ep>\d+)_"
    r"mp_rank_(?P<mp>\d+)_packed_expert_states\.pt$"
)
_CACHE_LOCK = threading.Lock()
_PACKED_CACHE: dict[str, dict[str, Any]] = {}


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def packed_expert_checkpoint_enabled() -> bool:
    return (
        os.environ.get("MOEGAMBIT_MODEL_KIND", "moe").lower() != "dense"
        and _env_flag(PACKED_EXPERT_CHECKPOINT_ENV)
    )


def packed_expert_cache_enabled() -> bool:
    return (
        os.environ.get("MOEGAMBIT_MODEL_KIND", "moe").lower() != "dense"
        and _env_flag(PACKED_EXPERT_CACHE_ENV)
    )


def packed_expert_checkpoint_name(
    checkpoints_path: str | os.PathLike[str],
    *,
    layer_id: int,
    ep_rank: int,
    mp_rank: int,
    tag: str | None,
) -> str:
    directory = Path(checkpoints_path)
    if tag is not None:
        directory /= str(tag)
    filename = (
        f"layer_{int(layer_id)}_ep_rank_{int(ep_rank):04d}_"
        f"mp_rank_{int(mp_rank):02d}{PACKED_EXPERT_SUFFIX}"
    )
    return str(directory / filename)


def build_packed_expert_state(
    *,
    layer_id: int,
    ep_rank: int,
    mp_rank: int,
    module_path: str,
    global_expert_start: int,
    num_local_experts: int,
    tensors: Mapping[str, torch.Tensor],
    folding_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        PACKED_EXPERT_FORMAT_KEY: PACKED_EXPERT_FORMAT,
        PACKED_EXPERT_VERSION_KEY: PACKED_EXPERT_FORMAT_VERSION,
        "layer_id": int(layer_id),
        "ep_rank": int(ep_rank),
        "mp_rank": int(mp_rank),
        "module_path": str(module_path),
        "global_expert_start": int(global_expert_start),
        "num_local_experts": int(num_local_experts),
        PACKED_EXPERT_TENSORS_KEY: dict(tensors),
    }
    if folding_metadata is not None:
        state[FOLDING_METADATA_KEY] = dict(folding_metadata)
    return state


def validate_packed_expert_state(
    state: Mapping[str, Any],
    *,
    layer_id: int | None = None,
    ep_rank: int | None = None,
    mp_rank: int | None = None,
    module_path: str | None = None,
    num_local_experts: int | None = None,
) -> Mapping[str, torch.Tensor]:
    if not isinstance(state, Mapping):
        raise RuntimeError(
            "packed expert checkpoint root must be a mapping"
        )
    if state.get(PACKED_EXPERT_FORMAT_KEY) != PACKED_EXPERT_FORMAT:
        raise RuntimeError(
            "unsupported packed expert checkpoint format: "
            f"{state.get(PACKED_EXPERT_FORMAT_KEY)!r}"
        )
    if state.get(PACKED_EXPERT_VERSION_KEY) != PACKED_EXPERT_FORMAT_VERSION:
        raise RuntimeError(
            "unsupported packed expert checkpoint version: "
            f"{state.get(PACKED_EXPERT_VERSION_KEY)!r}"
        )
    required_metadata = {
        "layer_id",
        "ep_rank",
        "mp_rank",
        "module_path",
        "global_expert_start",
        "num_local_experts",
    }
    missing = required_metadata - state.keys()
    if missing:
        raise RuntimeError(
            "packed expert checkpoint is missing metadata: "
            f"{sorted(missing)}"
        )
    expected_start = (
        int(state["ep_rank"]) * int(state["num_local_experts"])
    )
    if int(state["global_expert_start"]) != expected_start:
        raise RuntimeError(
            "packed expert checkpoint has inconsistent expert range: "
            f"start={state['global_expert_start']} "
            f"expected={expected_start}"
        )

    expected = {
        "layer_id": layer_id,
        "ep_rank": ep_rank,
        "mp_rank": mp_rank,
        "module_path": module_path,
        "num_local_experts": num_local_experts,
    }
    for key, value in expected.items():
        if value is not None and state.get(key) != value:
            raise RuntimeError(
                f"packed expert checkpoint {key} mismatch: "
                f"{state.get(key)!r} != {value!r}"
            )

    tensors = state.get(PACKED_EXPERT_TENSORS_KEY)
    if not isinstance(tensors, Mapping) or not tensors:
        raise RuntimeError("packed expert checkpoint has no expert tensors")
    expected_count = int(
        num_local_experts
        if num_local_experts is not None
        else state["num_local_experts"]
    )
    for name, tensor in tensors.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise RuntimeError(
                "packed expert checkpoint contains a non-tensor entry"
            )
        if tensor.dim() != 3 or tensor.shape[0] != expected_count:
            raise RuntimeError(
                f"packed expert tensor {name!r} has shape "
                f"{tuple(tensor.shape)}; expected "
                f"[{expected_count}, ..., ...]"
            )
    return tensors


def packed_expert_tensor_bytes(state: Mapping[str, Any]) -> int:
    tensors = state.get(PACKED_EXPERT_TENSORS_KEY, {})
    if not isinstance(tensors, Mapping):
        return 0
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in tensors.values()
        if isinstance(tensor, torch.Tensor)
    )


def packed_expert_pinned_bytes(state: Mapping[str, Any]) -> int:
    tensors = state.get(PACKED_EXPERT_TENSORS_KEY, {})
    if not isinstance(tensors, Mapping):
        return 0
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in tensors.values()
        if isinstance(tensor, torch.Tensor)
        and tensor.device.type == "cpu"
        and tensor.is_pinned()
    )


def _cache_key(path: str | os.PathLike[str]) -> str:
    return str(Path(path).resolve(strict=False))


def cache_packed_expert_state(
    path: str | os.PathLike[str], state: dict[str, Any]
) -> None:
    with _CACHE_LOCK:
        _PACKED_CACHE[_cache_key(path)] = state


def has_cached_packed_expert(
    path: str | os.PathLike[str],
) -> bool:
    with _CACHE_LOCK:
        return _cache_key(path) in _PACKED_CACHE


def cached_packed_expert_bytes(
    path: str | os.PathLike[str],
) -> tuple[int, int]:
    with _CACHE_LOCK:
        state = _PACKED_CACHE.get(_cache_key(path))
        if state is None:
            return 0, 0
        return (
            packed_expert_tensor_bytes(state),
            packed_expert_pinned_bytes(state),
        )


def take_cached_packed_expert(
    path: str | os.PathLike[str],
) -> dict[str, Any] | None:
    with _CACHE_LOCK:
        return _PACKED_CACHE.pop(_cache_key(path), None)


def clear_cached_packed_experts() -> None:
    with _CACHE_LOCK:
        _PACKED_CACHE.clear()


def _pin_packed_tensors(state: dict[str, Any]) -> tuple[int, str | None]:
    tensors = state.get(PACKED_EXPERT_TENSORS_KEY)
    if not isinstance(tensors, dict):
        return 0, "expert_tensors is not mutable"
    pinned_bytes = 0
    for name in list(tensors):
        tensor = tensors[name]
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        try:
            if not tensor.is_pinned():
                tensor = tensor.pin_memory()
        except RuntimeError as exc:
            tensors[name] = tensor
            return pinned_bytes, f"{type(exc).__name__}: {exc}"
        tensors[name] = tensor
        pinned_bytes += tensor.numel() * tensor.element_size()
    return pinned_bytes, None


class PackedExpertPrefetcher:
    """Watch the committed checkpoint and cache one local packed EP shard."""

    def __init__(
        self,
        checkpoint_dir: str | os.PathLike[str],
        *,
        mp_rank: int,
        ep_rank: int,
        expected_layers: int,
        max_bytes: int,
        pin_memory: bool = True,
        poll_interval: float = 0.5,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.mp_rank = int(mp_rank)
        self.ep_rank = int(ep_rank)
        self.expected_layers = int(expected_layers)
        self.max_bytes = max(0, int(max_bytes))
        self.pin_memory = bool(pin_memory)
        self.poll_interval = max(0.05, float(poll_interval))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state_lock = threading.Lock()
        self._active_tag: str | None = None
        self._snapshot: dict[str, Any] = {
            "tag": None,
            "files": 0,
            "bytes": 0,
            "pinned_bytes": 0,
            "completed": False,
            "error": None,
        }
        self._last_report: tuple[Any, ...] | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=(
                "packed-expert-prefetch-"
                f"mp{self.mp_rank}-ep{self.ep_rank}"
            ),
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 0.25) -> dict[str, Any]:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(0.0, timeout))
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self._state_lock:
            result = dict(self._snapshot)
        thread = self._thread
        result["thread_alive"] = bool(
            thread is not None and thread.is_alive()
        )
        return result

    def _set_snapshot(self, **values: Any) -> None:
        with self._state_lock:
            self._snapshot.update(values)

    def _read_latest(self) -> str | None:
        try:
            tag = (self.checkpoint_dir / "latest").read_text(
                encoding="utf-8"
            ).strip()
        except OSError:
            return None
        return tag or None

    def _paths_for_tag(self, tag: str) -> list[Path]:
        pattern = (
            f"layer_*_ep_rank_{self.ep_rank:04d}_"
            f"mp_rank_{self.mp_rank:02d}{PACKED_EXPERT_SUFFIX}"
        )
        return sorted(
            (self.checkpoint_dir / tag).glob(pattern),
            key=lambda path: (
                (0, int(_PACKED_NAME.match(path.name).group("layer")))
                if _PACKED_NAME.match(path.name)
                else (1, path.name)
            ),
        )

    def _report(self, *, elapsed: float) -> None:
        state = self.snapshot()
        signature = (
            state.get("tag"),
            state.get("files"),
            state.get("bytes"),
            state.get("pinned_bytes"),
            state.get("completed"),
            state.get("error"),
        )
        if signature == self._last_report:
            return
        self._last_report = signature
        print(
            "PACKED_EXPERT_PREFETCH "
            f"tag={state.get('tag')} mp_rank={self.mp_rank} "
            f"ep_rank={self.ep_rank} files={state.get('files')} "
            f"bytes={state.get('bytes')} "
            f"pinned_bytes={state.get('pinned_bytes')} "
            f"completed={state.get('completed')} "
            f"elapsed_s={elapsed:.2f} error={state.get('error')}",
            flush=True,
        )

    def _prefetch_tag(self, tag: str) -> None:
        if (
            tag == self._active_tag
            and bool(self.snapshot().get("completed"))
        ):
            return
        paths = self._paths_for_tag(tag)
        if len(paths) != self.expected_layers:
            return
        if self._active_tag != tag:
            clear_cached_packed_experts()
            self._active_tag = tag
            self._set_snapshot(
                tag=tag,
                files=0,
                bytes=0,
                pinned_bytes=0,
                completed=False,
                error=None,
            )

        started = time.monotonic()
        loaded_files = 0
        loaded_bytes = 0
        pinned_bytes = 0
        error: str | None = None
        for path in paths:
            if has_cached_packed_expert(path):
                cached_bytes, cached_pinned = (
                    cached_packed_expert_bytes(path)
                )
                loaded_files += 1
                loaded_bytes += cached_bytes
                pinned_bytes += cached_pinned
                continue
            if self._stop.is_set():
                return
            try:
                file_size = path.stat().st_size
                if loaded_bytes + file_size > self.max_bytes:
                    error = (
                        "packed expert cache budget exceeded: "
                        f"need_at_least={loaded_bytes + file_size} "
                        f"max={self.max_bytes}"
                    )
                    break
                state = torch.load(
                    path, map_location="cpu", weights_only=False
                )
                match = _PACKED_NAME.match(path.name)
                if not isinstance(state, dict) or match is None:
                    raise RuntimeError("invalid packed expert checkpoint")
                validate_packed_expert_state(
                    state,
                    layer_id=int(match.group("layer")),
                    ep_rank=self.ep_rank,
                    mp_rank=self.mp_rank,
                )
                current_bytes = packed_expert_tensor_bytes(state)
                if self.pin_memory:
                    current_pinned, pin_error = _pin_packed_tensors(state)
                    pinned_bytes += current_pinned
                    if pin_error is not None and error is None:
                        error = f"pin_memory fallback: {pin_error}"
                if self._stop.is_set():
                    return
                cache_packed_expert_state(path, state)
                loaded_files += 1
                loaded_bytes += current_bytes
                self._set_snapshot(
                    tag=tag,
                    files=loaded_files,
                    bytes=loaded_bytes,
                    pinned_bytes=pinned_bytes,
                    completed=False,
                    error=error,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                break

        completed = loaded_files == self.expected_layers
        self._set_snapshot(
            tag=tag,
            files=loaded_files,
            bytes=loaded_bytes,
            pinned_bytes=pinned_bytes,
            completed=completed,
            error=error,
        )
        self._report(elapsed=time.monotonic() - started)

    def _run(self) -> None:
        while not self._stop.is_set():
            tag = self._read_latest()
            if tag is not None:
                try:
                    self._prefetch_tag(tag)
                except Exception as exc:
                    self._set_snapshot(
                        tag=tag,
                        completed=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    self._report(elapsed=0.0)
            self._stop.wait(self.poll_interval)
