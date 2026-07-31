"""Recovery-only MoE token-dispatch diagnostics owned by the Megatron adapter."""

from __future__ import annotations

import logging
import os
import time
from datetime import timedelta
from typing import Optional

import torch
from megatron.core import utils

logger = logging.getLogger(__name__)

_ELASTIC_MOE_FIRST_COLLECTIVE_BARRIERS = set()
_ELASTIC_MOE_FIRST_COLLECTIVE_CHECKS = set()
_ELASTIC_MOE_FIRST_COLLECTIVE_STORES = set()


class TokenDispatcherRecoveryHooks:
    def _elastic_refresh_post_rebuild_groups(self) -> bool:
        """Refresh MoE process groups on the first real post-rebuild forward.

        The elastic rebuild path reinitializes Megatron's global parallel_state
        before resuming training. Existing module instances may still hold old
        ProcessGroup handles, so refresh the dispatcher itself immediately
        before its first replacement-facing collective.
        """
        if not TokenDispatcherRecoveryHooks._elastic_post_rebuild_trace_active():
            return False

        token = os.environ.get("ELASTIC_POST_REBUILD_TRACE_TOKEN", "")
        if getattr(self, "_elastic_post_rebuild_group_refresh_token", None) == token:
            return False

        old_tp_ep_group = getattr(self, "tp_ep_group", None)
        old_tp_ep_ranks = TokenDispatcherRecoveryHooks._elastic_group_ranks(old_tp_ep_group)
        changed = []
        try:
            from megatron.core import parallel_state as mpu

            new_groups = (
                ("ep_group", mpu.get_expert_model_parallel_group(check_initialized=False)),
                ("tp_group", mpu.get_expert_tensor_parallel_group(check_initialized=False)),
                (
                    "tp_ep_group",
                    mpu.get_expert_tensor_and_model_parallel_group(check_initialized=False),
                ),
            )
            for attr_name, new_group in new_groups:
                if new_group is None:
                    continue
                if getattr(self, attr_name, None) is not new_group:
                    setattr(self, attr_name, new_group)
                    changed.append(attr_name)

            self.ep_size = utils.get_pg_size(self.ep_group)
            self.tp_size = utils.get_pg_size(self.tp_group)
            self.tp_rank = utils.get_pg_rank(self.tp_group)
        except Exception as exc:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
            logger.warning(
                "[elastic] Rank %d: failed refreshing MoE dispatcher groups "
                "before post-rebuild collective: %s",
                rank,
                exc,
            )
            return False

        self._elastic_post_rebuild_group_refresh_token = token
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
        trace = os.environ.get("ELASTIC_TRACE_MOE_GROUP_REBIND", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if changed or trace:
            logger.warning(
                "[elastic] Rank %d: refreshed MoE dispatcher groups before "
                "post-rebuild collective token=%s changed=%s tp_ep_old=%s tp_ep_new=%s",
                rank,
                token,
                changed or [],
                old_tp_ep_ranks,
                TokenDispatcherRecoveryHooks._elastic_group_ranks(self.tp_ep_group),
            )
        return bool(changed)

    @staticmethod
    def _elastic_post_rebuild_trace_active():
        try:
            from moegambit.adapters.megatron.hooks import elastic_is_post_rebuild_trace_active

            return elastic_is_post_rebuild_trace_active()
        except Exception:
            return False

    def _elastic_trace_once(self, key: str, message: str, *args):
        if not TokenDispatcherRecoveryHooks._elastic_post_rebuild_trace_active():
            return
        if os.environ.get("ELASTIC_TRACE_MOE_DISPATCH", "0") != "1":
            return
        attr_name = f"_elastic_trace_{key}"
        trace_token = os.environ.get("ELASTIC_POST_REBUILD_TRACE_TOKEN", "rebuild")
        traced_tokens = getattr(self, attr_name, set())
        if trace_token in traced_tokens:
            return
        traced_tokens.add(trace_token)
        setattr(self, attr_name, traced_tokens)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
        logger.warning("[elastic] Rank %d: " + message, rank, *args)

    @staticmethod
    def _elastic_group_ranks(group):
        try:
            return list(torch.distributed.get_process_group_ranks(group))
        except Exception:
            return None

    @staticmethod
    def _elastic_replacement_facing_group(group) -> bool:
        if not TokenDispatcherRecoveryHooks._elastic_post_rebuild_trace_active():
            return False
        ranks = TokenDispatcherRecoveryHooks._elastic_group_ranks(group)
        if not ranks:
            return False
        try:
            replacement_rank = int(os.environ.get("ELASTIC_REPLACEMENT_RANK", "-1"))
        except ValueError:
            return False
        return replacement_rank in ranks

    @staticmethod
    def _elastic_process_group_contract(group):
        contract = {
            "group_name": str(getattr(group, "group_name", "unavailable")),
            "group_desc": str(getattr(group, "group_desc", "unavailable")),
            "group_rank": int(group.rank()),
            "group_size": int(group.size()),
            "store_type": "unavailable",
        }
        try:
            world = torch.distributed.distributed_c10d._world
            contract["c10d_name"] = str(world.pg_names.get(group, "unregistered"))
            contract["store_type"] = type(world.pg_map[group][1]).__name__
        except Exception as exc:
            contract["registry_error"] = str(exc)
        return contract

    @staticmethod
    def _elastic_validate_first_collective_store(name: str, group, ranks):
        """Verify that all members see rank zero through this PG's exact PrefixStore."""
        replacement_rank = int(os.environ.get("ELASTIC_REPLACEMENT_RANK", "-1"))
        token = os.environ.get("ELASTIC_POST_REBUILD_TRACE_TOKEN", "rebuild")
        store_key = (token, name, tuple(ranks), replacement_rank)
        if store_key in _ELASTIC_MOE_FIRST_COLLECTIVE_STORES:
            return

        contract = TokenDispatcherRecoveryHooks._elastic_process_group_contract(group)
        timeout = float(os.environ.get("ELASTIC_MOE_STORE_CANARY_TIMEOUT", "30"))
        canary_key = f"elastic_canary:{token}:{name}"
        expected = (
            f"{token}|{name}|{contract['group_name']}|"
            + "-".join(str(rank) for rank in ranks)
        ).encode()
        rank = torch.distributed.get_rank()

        try:
            world = torch.distributed.distributed_c10d._world
            pg_store = world.pg_map[group][1]
            if int(group.rank()) == 0:
                pg_store.set(canary_key, expected)
            pg_store.wait([canary_key], timedelta(seconds=timeout))
            observed = bytes(pg_store.get(canary_key))
            if observed != expected:
                raise RuntimeError(
                    f"payload mismatch expected={expected!r} observed={observed!r}"
                )
        except Exception as exc:
            TokenDispatcherRecoveryHooks._elastic_report_recovery_phase(
                "moe_first_collective_store_error",
                collective=name,
                group_ranks=ranks,
                pg_contract=contract,
                error=str(exc),
            )
            raise RuntimeError(
                f"[elastic] first MoE {name} ProcessGroup store namespace is not shared "
                f"rank={rank} ranks={ranks} contract={contract}: {exc}"
            ) from exc

        _ELASTIC_MOE_FIRST_COLLECTIVE_STORES.add(store_key)
        TokenDispatcherRecoveryHooks._elastic_report_recovery_phase(
            "moe_first_collective_store_ready",
            collective=name,
            group_ranks=ranks,
            pg_contract=contract,
        )
        logger.warning(
            "[elastic] Rank %d: first MoE %s ProcessGroup store verified contract=%s",
            rank,
            name,
            contract,
        )

    @staticmethod
    def _elastic_wait_first_collective_barrier(name: str, group):
        """Align replacement-facing MoE collectives on the first post-rebuild step."""
        if not TokenDispatcherRecoveryHooks._elastic_post_rebuild_trace_active():
            return
        if os.environ.get("ELASTIC_MOE_FIRST_COLLECTIVE_BARRIER", "1") == "0":
            return
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return

        ranks = TokenDispatcherRecoveryHooks._elastic_group_ranks(group)
        if not ranks:
            return
        try:
            rank = torch.distributed.get_rank()
            replacement_rank = int(os.environ.get("ELASTIC_REPLACEMENT_RANK", "-1"))
        except (TypeError, ValueError):
            return
        if rank not in ranks or replacement_rank not in ranks:
            return

        token = os.environ.get("ELASTIC_POST_REBUILD_TRACE_TOKEN", "rebuild")
        barrier_key = (token, name, tuple(ranks), replacement_rank)
        if barrier_key in _ELASTIC_MOE_FIRST_COLLECTIVE_BARRIERS:
            return

        timeout = float(
            os.environ.get(
                "ELASTIC_MOE_FIRST_COLLECTIVE_BARRIER_TIMEOUT",
                os.environ.get(
                    "ELASTIC_PHASE_TIMEOUT_SECONDS",
                    os.environ.get("ELASTIC_REBUILD_PHASE_TIMEOUT", "720"),
                ),
            )
        )
        resume_iteration = os.environ.get("ELASTIC_RESUME_ITERATION", "-1")
        token = os.environ.get(
            "ELASTIC_MOE_FIRST_COLLECTIVE_BARRIER_TOKEN",
            f"iter{resume_iteration}:replacement{replacement_rank}",
        )
        ranks_token = "-".join(str(r) for r in ranks)
        barrier_id = f"moe_first_collective:{token}:{name}:{ranks_token}"

        logger.warning(
            "[elastic] Rank %d: waiting before first MoE %s collective ranks=%s",
            rank,
            name,
            ranks,
        )
        try:
            from moegambit.adapters.megatron.hooks import elastic_wait_for_ordinal_barrier

            ok = elastic_wait_for_ordinal_barrier(
                barrier_id,
                rank,
                len(ranks),
                timeout,
                group_desc=f"MOE_{name}",
                group_backend="default",
                group_size=len(ranks),
                group_ranks=ranks,
                barrier_stage="first_collective",
            )
        except Exception as exc:
            raise RuntimeError(
                f"[elastic] failed waiting before first MoE {name} collective "
                f"rank={rank} ranks={ranks}: {exc}"
            ) from exc
        if not ok:
            raise RuntimeError(
                f"[elastic] timed out before first MoE {name} collective "
                f"rank={rank} ranks={ranks}"
            )
        _ELASTIC_MOE_FIRST_COLLECTIVE_BARRIERS.add(barrier_key)
        logger.warning(
            "[elastic] Rank %d: aligned before first MoE %s collective ranks=%s",
            rank,
            name,
            ranks,
        )

    @staticmethod
    def _elastic_first_collective_timeout() -> float:
        timeout = os.environ.get("ELASTIC_MOE_FIRST_COLLECTIVE_TIMEOUT")
        if timeout:
            return float(timeout)
        phase_timeout = float(
            os.environ.get(
                "ELASTIC_PHASE_TIMEOUT_SECONDS",
                os.environ.get("ELASTIC_REBUILD_PHASE_TIMEOUT", "180"),
            )
        )
        return min(phase_timeout, 180.0)

    @staticmethod
    def _elastic_report_recovery_phase(phase: str, **extra):
        try:
            from moegambit.adapters.megatron.hooks import elastic_report_recovery_phase

            elastic_report_recovery_phase(phase, **extra)
        except Exception:
            pass

    @staticmethod
    def _elastic_gather_first_dim_fail_fast(
        input_: torch.Tensor, group, name: str, output: Optional[torch.Tensor] = None
    ):
        """All-gather along dim 0 with a recovery-only timeout.

        This mirrors gather_from_sequence_parallel_region for metadata tensors
        that do not need autograd, but uses async_op so a broken rebuilt NCCL
        communicator fails at the recovery boundary instead of hanging forever
        inside the first post-rebuild forward pass.
        """
        assert group is not None, "group should not be None"
        world_size = group.size()
        if world_size == 1:
            return input_

        if output is None:
            dim_size = list(input_.size())
            dim_size[0] = dim_size[0] * world_size
            output = torch.empty(
                dim_size, dtype=input_.dtype, device=torch.cuda.current_device()
            )

        ranks = TokenDispatcherRecoveryHooks._elastic_group_ranks(group)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
        token = os.environ.get("ELASTIC_POST_REBUILD_TRACE_TOKEN", "")
        timeout = TokenDispatcherRecoveryHooks._elastic_first_collective_timeout()
        TokenDispatcherRecoveryHooks._elastic_report_recovery_phase(
            "moe_first_collective_start",
            collective=name,
            group_ranks=ranks,
            timeout=timeout,
        )
        logger.warning(
            "[elastic] Rank %d: launching first MoE %s collective with timeout %.1fs ranks=%s",
            rank,
            name,
            timeout,
            ranks,
        )

        try:
            gather_fn = (
                torch.distributed.all_gather_into_tensor
                if hasattr(torch.distributed, "all_gather_into_tensor")
                else torch.distributed._all_gather_base
            )
            work = gather_fn(output, input_.contiguous(), group=group, async_op=True)
        except Exception as exc:
            TokenDispatcherRecoveryHooks._elastic_report_recovery_phase(
                "moe_first_collective_error",
                collective=name,
                group_ranks=ranks,
                error=str(exc),
            )
            raise RuntimeError(
                f"[elastic] failed launching first MoE {name} collective "
                f"rank={rank} ranks={ranks} token={token}: {exc}"
            ) from exc

        deadline = time.monotonic() + max(timeout, 0.0)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                TokenDispatcherRecoveryHooks._elastic_report_recovery_phase(
                    "moe_first_collective_timeout",
                    collective=name,
                    group_ranks=ranks,
                    timeout=timeout,
                )
                raise RuntimeError(
                    f"[elastic] first MoE {name} collective timed out after {timeout:.1f}s "
                    f"rank={rank} ranks={ranks} token={token}. "
                    "The rebuilt replacement-facing NCCL communicator is not usable; "
                    "abort this in-place recovery run."
                )

            try:
                if work.is_completed():
                    break
            except Exception:
                try:
                    wait_result = work.wait(timeout=timedelta(seconds=max(remaining, 0.0)))
                except TypeError:
                    wait_result = work.wait()
                if wait_result is not False:
                    break
                continue

            time.sleep(0.05)

        try:
            work.wait()
        except Exception as exc:
            TokenDispatcherRecoveryHooks._elastic_report_recovery_phase(
                "moe_first_collective_error",
                collective=name,
                group_ranks=ranks,
                error=str(exc),
            )
            raise RuntimeError(
                f"[elastic] first MoE {name} collective failed "
                f"rank={rank} ranks={ranks} token={token}: {exc}"
            ) from exc

        TokenDispatcherRecoveryHooks._elastic_report_recovery_phase(
            "moe_first_collective_done",
            collective=name,
            group_ranks=ranks,
            timeout=timeout,
        )
        logger.warning(
            "[elastic] Rank %d: first MoE %s collective completed ranks=%s",
            rank,
            name,
            ranks,
        )
        return output

    @staticmethod
    def _elastic_gather_first_dim_ready_aligned(input_: torch.Tensor, group, name: str):
        """Fail fast once, then use the normal collective for later MoE layers."""
        world_size = group.size()
        if world_size == 1:
            return input_

        ranks = TokenDispatcherRecoveryHooks._elastic_group_ranks(group)
        token = os.environ.get("ELASTIC_POST_REBUILD_TRACE_TOKEN", "rebuild")
        replacement_rank = int(os.environ.get("ELASTIC_REPLACEMENT_RANK", "-1"))
        check_key = (token, name, tuple(ranks or ()), replacement_rank)
        dim_size = list(input_.size())
        dim_size[0] *= world_size
        output = torch.empty(dim_size, dtype=input_.dtype, device=input_.device)
        contiguous_input = input_.contiguous()
        gather_fn = (
            torch.distributed.all_gather_into_tensor
            if hasattr(torch.distributed, "all_gather_into_tensor")
            else torch.distributed._all_gather_base
        )

        if check_key in _ELASTIC_MOE_FIRST_COLLECTIVE_CHECKS:
            gather_fn(output, contiguous_input, group=group)
            return output

        # The old barrier ran before these CUDA operations.  A fresh replacement
        # could consequently enter store->get while rank zero was still blocked
        # in allocator/stream work and had not published the NCCL unique ID.
        torch.cuda.current_stream().synchronize()
        rank = torch.distributed.get_rank()
        contract = TokenDispatcherRecoveryHooks._elastic_process_group_contract(group)
        TokenDispatcherRecoveryHooks._elastic_report_recovery_phase(
            "moe_first_collective_prepared",
            collective=name,
            group_ranks=ranks,
            pg_contract=contract,
        )
        logger.warning(
            "[elastic] Rank %d: prepared first MoE %s tensors contract=%s",
            rank,
            name,
            contract,
        )

        TokenDispatcherRecoveryHooks._elastic_wait_first_collective_barrier(name, group)
        TokenDispatcherRecoveryHooks._elastic_validate_first_collective_store(name, group, ranks)
        logger.warning(
            "[elastic] Rank %d: starting prepared first MoE %s collective contract=%s",
            rank,
            name,
            contract,
        )

        if os.environ.get("ELASTIC_MOE_FIRST_COLLECTIVE_FAIL_FAST", "0") != "0":
            result = TokenDispatcherRecoveryHooks._elastic_gather_first_dim_fail_fast(
                contiguous_input, group, name, output=output
            )
        else:
            gather_fn(output, contiguous_input, group=group)
            result = output

        _ELASTIC_MOE_FIRST_COLLECTIVE_CHECKS.add(check_key)
        return result
