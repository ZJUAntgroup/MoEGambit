"""Adapter-owned process-group hooks for Megatron.

Megatron keeps the public compatibility entry points because the recovery
client calls them through ``megatron.core.mpu``.  All recovery state and c10d
registry manipulation lives here.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class MegatronParallelStateHooks:
    def __init__(self) -> None:
        self.group_ordinal = 0
        self.group_specs: list[dict[str, Any]] = []
        self.retained_groups: dict[Any, list[dict[str, Any]]] = {}
        self.selective_rebuild_active = False
        self.selective_rebuild_rank: Optional[int] = None
        self.stats = self._empty_stats()

    @staticmethod
    def _empty_stats() -> dict[str, int]:
        return {"retained": 0, "reused": 0, "rebuilt": 0, "skipped_nonmember": 0}

    @staticmethod
    def _signature(ranks, backend, group_desc, local):
        return (
            str(group_desc or ""),
            None if ranks is None else tuple(int(rank) for rank in ranks),
            "default" if backend is None else str(backend).lower(),
            bool(local),
        )

    @staticmethod
    def _is_nonmember(torch_module: Any, group: Any) -> bool:
        return group == torch_module.distributed.GroupMember.NON_GROUP_MEMBER

    def reset_ordinal(self) -> None:
        self.group_ordinal = 0
        self.group_specs = []
        if os.environ.get("ELASTIC_SELECTIVE_GROUP_REBUILD", "0") == "1":
            replacement_rank = os.environ.get("ELASTIC_REPLACEMENT_RANK")
            if replacement_rank is not None:
                self.selective_rebuild_active = True
                self.selective_rebuild_rank = int(replacement_rank)

    @staticmethod
    def _snapshot_registered_group(torch_module: Any, group: Any) -> dict[str, Any]:
        c10d = torch_module.distributed.distributed_c10d
        world = c10d._world
        required = (
            "pg_map",
            "pg_names",
            "pg_group_ranks",
            "pg_backend_config",
            "pg_to_tag",
            "tags_to_pg",
        )
        missing = [name for name in required if not hasattr(world, name)]
        if missing or not hasattr(c10d, "_register_process_group"):
            raise RuntimeError(
                "[elastic] selective group rebuild requires c10d registry fields: "
                + ",".join(missing or ["_register_process_group"])
            )
        if group not in world.pg_map or group not in world.pg_names:
            raise RuntimeError("[elastic] retained process group is not registered")
        return {
            "group": group,
            "pg_map": world.pg_map[group],
            "pg_name": world.pg_names[group],
            "pg_group_ranks": world.pg_group_ranks.get(group),
            "pg_backend_config": world.pg_backend_config.get(group),
            "pg_tag": world.pg_to_tag.get(group),
            "in_default_tag": group in world.tags_to_pg.get("", []),
        }

    @staticmethod
    def _detach_registered_group(torch_module: Any, snapshot: dict[str, Any]) -> None:
        world = torch_module.distributed.distributed_c10d._world
        group = snapshot["group"]
        world.pg_map.pop(group, None)
        world.pg_names.pop(group, None)
        world.pg_group_ranks.pop(group, None)
        world.pg_backend_config.pop(group, None)
        tag = world.pg_to_tag.pop(group, None)
        if tag is not None:
            for key in (tag, ""):
                groups = world.tags_to_pg.get(key)
                if groups is None:
                    continue
                try:
                    groups.remove(group)
                except ValueError:
                    pass
                if not groups:
                    world.tags_to_pg.pop(key, None)
        if hasattr(world, "pg_coalesce_state"):
            world.pg_coalesce_state.pop(group, None)

    @staticmethod
    def _restore_registered_group(
        torch_module: Any, snapshot: dict[str, Any], alias: str
    ) -> Any:
        c10d = torch_module.distributed.distributed_c10d
        world = c10d._world
        group = snapshot["group"]
        if group in world.pg_map:
            return group
        if alias in set(world.pg_names.values()):
            raise RuntimeError(f"[elastic] retained process-group alias collision: {alias}")
        world.pg_map[group] = snapshot["pg_map"]
        world.pg_names[group] = alias
        if snapshot["pg_group_ranks"] is not None:
            world.pg_group_ranks[group] = snapshot["pg_group_ranks"]
        if snapshot["pg_backend_config"] is not None:
            world.pg_backend_config[group] = snapshot["pg_backend_config"]
        tag = snapshot.get("pg_tag")
        if tag is not None:
            world.pg_to_tag[group] = tag
            world.tags_to_pg.setdefault(tag, []).append(group)
        if snapshot.get("in_default_tag"):
            world.tags_to_pg.setdefault("", []).append(group)
        if hasattr(group, "_set_group_name"):
            group._set_group_name(alias)
        c10d._register_process_group(alias, group)
        return group

    def prepare_selective_rebuild(
        self, replacement_rank: int, *, torch_module: Any
    ) -> dict[str, Any]:
        if os.environ.get("ELASTIC_SELECTIVE_GROUP_REBUILD", "0") != "1":
            return {"enabled": False, "retained": 0}
        if not torch_module.distributed.is_initialized():
            raise RuntimeError("[elastic] cannot retain groups before WORLD initialization")
        if replacement_rank is None or int(replacement_rank) < 0:
            raise RuntimeError("[elastic] selective rebuild requires a replacement rank")
        if not self.group_specs:
            raise RuntimeError("[elastic] no Megatron process-group manifest is available")

        torch_module.cuda.synchronize()
        replacement_rank = int(replacement_rank)
        retained: dict[Any, list[dict[str, Any]]] = {}
        retained_count = 0
        for spec in self.group_specs:
            ranks = spec["ranks"]
            group = spec["group"]
            if (
                ranks is None
                or replacement_rank in ranks
                or spec["use_local_synchronization"]
                or self._is_nonmember(torch_module, group)
            ):
                continue
            backend = str(torch_module.distributed.get_backend(group)).lower()
            if "nccl" not in backend:
                continue
            snapshot = self._snapshot_registered_group(torch_module, group)
            self._detach_registered_group(torch_module, snapshot)
            retained.setdefault(spec["signature"], []).append(snapshot)
            retained_count += 1

        self.retained_groups = retained
        self.group_specs = []
        self.selective_rebuild_active = True
        self.selective_rebuild_rank = replacement_rank
        self.stats = self._empty_stats()
        self.stats["retained"] = retained_count
        return {"enabled": True, "retained": retained_count}

    def finalize_selective_rebuild(
        self, *, torch_module: Any, track_group: Callable[[Any], None]
    ) -> dict[str, Any]:
        if not self.selective_rebuild_active:
            return {"enabled": False}
        epoch = os.environ.get("ELASTIC_PG_GENERATION", "0")
        auxiliary_index = 0
        for signature, retained in list(self.retained_groups.items()):
            for snapshot in retained:
                group = self._restore_registered_group(
                    torch_module,
                    snapshot,
                    f"elastic_retained_{epoch}_aux_{auxiliary_index}",
                )
                self.group_specs.append(
                    {
                        "signature": signature,
                        "ranks": signature[1],
                        "backend": None
                        if signature[2] == "default"
                        else signature[2],
                        "group_desc": signature[0] or None,
                        "use_local_synchronization": signature[3],
                        "group": group,
                    }
                )
                track_group(group)
                self.stats["reused"] += 1
                auxiliary_index += 1
        result = dict(self.stats)
        result["enabled"] = True
        self.retained_groups = {}
        self.selective_rebuild_active = False
        self.selective_rebuild_rank = None
        logger.warning("[elastic] selective Megatron subgroup rebuild complete: %s", result)
        return result

    @staticmethod
    def _c10d_group_count(torch_module: Any, local: bool) -> Optional[int]:
        if local:
            return None
        try:
            return int(torch_module.distributed.distributed_c10d._world.group_count)
        except Exception:
            return None

    @staticmethod
    def _barrier_enabled(torch_module: Any) -> bool:
        if os.environ.get("ELASTIC_TRACE_MPU_GROUPS", "0").lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return False
        if os.environ.get("ELASTIC_MPU_GROUP_ORDINAL_BARRIER", "1").lower() in {
            "0",
            "false",
            "no",
            "off",
        }:
            return False
        if (
            os.environ.get("ELASTIC_REBUILD_MODE") != "1"
            and "ELASTIC_REPLACEMENT_RANK" not in os.environ
        ):
            return False
        if not os.environ.get("ELASTIC_WATCHER_ADDR"):
            return False
        return (
            torch_module.distributed.is_available()
            and torch_module.distributed.is_initialized()
        )

    def _next_ordinal(self, torch_module: Any) -> Optional[int]:
        if not self._barrier_enabled(torch_module):
            return None
        self.group_ordinal += 1
        return self.group_ordinal

    @staticmethod
    def _barrier_timeout(timeout: Any) -> float:
        value = os.environ.get(
            "ELASTIC_MPU_GROUP_ORDINAL_TIMEOUT_SECONDS",
            os.environ.get(
                "ELASTIC_PHASE_TIMEOUT_SECONDS",
                os.environ.get("ELASTIC_REBUILD_PHASE_TIMEOUT"),
            ),
        )
        if value:
            return float(value)
        if timeout is not None:
            try:
                return max(300.0, float(timeout.total_seconds()) + 120.0)
            except (TypeError, ValueError):
                pass
        return 300.0

    @staticmethod
    def _manifest(
        torch_module: Any,
        *,
        ranks,
        timeout,
        backend,
        group_desc,
        ordinal,
        stage,
        local,
        c10d_count,
    ) -> dict[str, Any]:
        ranks_list = None if ranks is None else [int(rank) for rank in ranks]
        size = (
            torch_module.distributed.get_world_size()
            if ranks_list is None
            else len(ranks_list)
        )
        representative = 0 if ranks_list is None else min(ranks_list, default=-1)
        result = {
            "group_ordinal": int(ordinal) if ordinal is not None else -1,
            "barrier_stage": stage,
            "group_desc": str(group_desc),
            "group_backend": "default" if backend is None else str(backend),
            "group_init_mode": "lazy",
            "group_use_local_synchronization": bool(local),
            "group_size": size,
            "group_representative_rank": representative,
        }
        if c10d_count is not None:
            result["group_c10d_count"] = int(c10d_count)
        if timeout is not None:
            result["group_timeout_seconds"] = float(timeout.total_seconds())
        if ranks_list is None:
            result["group_ranks"] = "ALL"
        elif len(ranks_list) <= 16:
            result["group_ranks"] = ranks_list
        else:
            result["group_first_rank"] = ranks_list[0]
            result["group_last_rank"] = ranks_list[-1]
        return result

    def _wait_ordinal(
        self,
        torch_module: Any,
        *,
        ordinal,
        stage,
        ranks,
        timeout,
        backend,
        group_desc,
        local,
        c10d_count,
    ) -> None:
        if ordinal is None:
            return
        from .elastic_client import elastic_wait_for_ordinal_barrier

        rank = torch_module.distributed.get_rank()
        world_size = torch_module.distributed.get_world_size()
        token = os.environ.get(
            "ELASTIC_MPU_GROUP_BARRIER_TOKEN",
            "iter%s:replacement%s"
            % (
                os.environ.get("ELASTIC_RESUME_ITERATION", "-1"),
                os.environ.get("ELASTIC_REPLACEMENT_RANK", "-1"),
            ),
        )
        barrier_id = f"mpu_group:{token}:{int(ordinal):06d}:{stage}"
        manifest = self._manifest(
            torch_module,
            ranks=ranks,
            timeout=timeout,
            backend=backend,
            group_desc=group_desc,
            ordinal=ordinal,
            stage=stage,
            local=local,
            c10d_count=c10d_count,
        )
        if not elastic_wait_for_ordinal_barrier(
            barrier_id,
            rank,
            world_size,
            self._barrier_timeout(timeout),
            **manifest,
        ):
            raise RuntimeError(
                f"[elastic] timed out waiting for Megatron group ordinal barrier "
                f"{barrier_id} rank={rank} desc={group_desc}"
            )

    def _trace(
        self,
        torch_module: Any,
        *,
        phase,
        ranks,
        timeout,
        backend,
        group_desc,
        ordinal,
        local,
        c10d_count,
    ) -> None:
        if os.environ.get("ELASTIC_TRACE_MPU_GROUPS", "0").lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return
        if not torch_module.distributed.is_initialized():
            return
        rank = torch_module.distributed.get_rank()
        ranks_list = None if ranks is None else [int(item) for item in ranks]
        if ranks_list is not None and rank not in ranks_list:
            return
        representative = 0 if ranks_list is None else min(ranks_list)
        replacement_rank = int(os.environ.get("ELASTIC_REPLACEMENT_RANK", "-1"))
        report_members = (
            os.environ.get("ELASTIC_TRACE_REPLACEMENT_GROUP_MEMBERS", "0").lower()
            in {"1", "true", "yes", "on"}
            and ranks_list is not None
            and replacement_rank in ranks_list
        )
        if rank != representative and not report_members:
            return
        from .elastic_client import elastic_report_recovery_phase

        extra = self._manifest(
            torch_module,
            ranks=ranks,
            timeout=timeout,
            backend=backend,
            group_desc=group_desc,
            ordinal=ordinal,
            stage=phase,
            local=local,
            c10d_count=c10d_count,
        )
        extra.pop("barrier_stage", None)
        extra["group_report_rank"] = rank
        elastic_report_recovery_phase(phase, **extra)

    def create_group(
        self,
        *,
        torch_module: Any,
        is_torch_min_version: Callable[[str], bool],
        track_group: Callable[[Any], None],
        ranks=None,
        timeout=None,
        backend=None,
        pg_options=None,
        use_local_synchronization=False,
        group_desc=None,
    ) -> Any:
        signature = self._signature(
            ranks, backend, group_desc, use_local_synchronization
        )
        ranks_tuple = signature[1]
        backend_key = signature[2]
        ordinal = self._next_ordinal(torch_module)
        reuse = (
            self.selective_rebuild_active
            and ranks_tuple is not None
            and self.selective_rebuild_rank not in ranks_tuple
            and not use_local_synchronization
            and (backend_key == "default" or "nccl" in backend_key)
        )
        if reuse:
            if torch_module.distributed.get_rank() in ranks_tuple:
                retained = self.retained_groups.get(signature)
                if not retained:
                    raise RuntimeError(
                        f"[elastic] missing retained group {group_desc} {list(ranks_tuple)}"
                    )
                snapshot = retained.pop(0)
                if not retained:
                    self.retained_groups.pop(signature, None)
                suffix = ordinal if ordinal is not None else len(self.group_specs)
                group = self._restore_registered_group(
                    torch_module,
                    snapshot,
                    f"elastic_retained_{os.environ.get('ELASTIC_PG_GENERATION', '0')}_{suffix}",
                )
                self.stats["reused"] += 1
            else:
                group = torch_module.distributed.GroupMember.NON_GROUP_MEMBER
                self.stats["skipped_nonmember"] += 1
            self._record(signature, backend, group_desc, use_local_synchronization, group)
            if not self._is_nonmember(torch_module, group):
                track_group(group)
            return group

        count = self._c10d_group_count(torch_module, use_local_synchronization)
        if ordinal is not None and not use_local_synchronization and count is None:
            raise RuntimeError("[elastic] c10d group_count is unavailable")
        common = dict(
            ranks=ranks,
            timeout=timeout,
            backend=backend,
            group_desc=group_desc,
            ordinal=ordinal,
            local=use_local_synchronization,
            c10d_count=count,
        )
        self._trace(torch_module, phase="mpu_group_start", **common)
        self._wait_ordinal(torch_module, stage="enter", **common)
        kwargs = {
            "ranks": ranks,
            "timeout": timeout,
            "backend": backend,
            "pg_options": pg_options,
            "use_local_synchronization": use_local_synchronization,
            "group_desc": group_desc,
        }
        if not is_torch_min_version("2.4.0"):
            kwargs.pop("group_desc")
            if timeout is None:
                kwargs.pop("timeout")
        group = torch_module.distributed.new_group(**kwargs)
        if self.selective_rebuild_active:
            self.stats["rebuilt"] += 1
        count = self._c10d_group_count(torch_module, use_local_synchronization)
        common["c10d_count"] = count
        self._trace(torch_module, phase="mpu_group_done", **common)
        self._wait_ordinal(torch_module, stage="exit", **common)
        self._record(signature, backend, group_desc, use_local_synchronization, group)
        if ranks is None or torch_module.distributed.get_rank() in ranks:
            track_group(group)
        return group

    def _record(self, signature, backend, group_desc, local, group) -> None:
        self.group_specs.append(
            {
                "signature": signature,
                "ranks": signature[1],
                "backend": backend,
                "group_desc": group_desc,
                "use_local_synchronization": bool(local),
                "group": group,
            }
        )


parallel_state_hooks = MegatronParallelStateHooks()
