"""Legacy PEC checkpoint-state selection overlay.

Enabled by ``MOEGAMBIT_MOC_PEC_EMULATE=1``. When disabled,
``write_pec_metadata`` and ``apply_pec_to_plan`` are no-ops.

This overlay leaves full checkpoint saving unchanged. A sidecar identifies the
experts selected by a round-robin schedule and the earlier checkpoint directory
for each remaining expert; recovery rewrites the expert load paths accordingly.
It is a checkpoint-state experiment, not a physical PEC save/restore benchmark
or the original MoC-System implementation. Its full-write path cannot measure
partial-saving benefits or establish end-to-end performance.

For physical selected-expert I/O and a real-training restart/replay benchmark,
see ``examples/moc_system/README.md``. Those independent fixed-K benchmarks run
with this overlay disabled and report their own measured scope.

Environment knobs (read once at module init):

    MOEGAMBIT_MOC_PEC_EMULATE      -- '1' to enable (default '0')
    MOEGAMBIT_MOC_PEC_K            -- number of fresh experts per save (default 16)
    MOEGAMBIT_MOC_PEC_N_EXPERT     -- total experts in the model (default 128)
    MOEGAMBIT_MOC_PEC_SCHEDULE     -- 'round_robin' (default; only option for now)
    MOEGAMBIT_MOC_PEC_LOG_LEVEL    -- python logging level (default 'INFO')
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "moc_pec_metadata.json"
_BOOL_TRUE = {"1", "true", "yes", "on", "y"}


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in _BOOL_TRUE


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def is_enabled() -> bool:
    """Single source of truth: emulation only runs when this returns True."""
    return _flag("MOEGAMBIT_MOC_PEC_EMULATE", "0")


def _k_pec() -> int:
    return _int("MOEGAMBIT_MOC_PEC_K", 16)


def _n_expert() -> int:
    return _int("MOEGAMBIT_MOC_PEC_N_EXPERT", 128)


def fresh_experts_for_round(save_idx: int,
                            k_pec: Optional[int] = None,
                            n_expert: Optional[int] = None) -> List[int]:
    """MoC-System round-robin schedule: contiguous group of K_pec experts.

    save_idx = 0  -> [0..K-1]
    save_idx = 1  -> [K..2K-1]
    ... wraps after ceil(N/K) rounds.
    """
    k = k_pec if k_pec is not None else _k_pec()
    n = n_expert if n_expert is not None else _n_expert()
    if k <= 0 or n <= 0:
        return []
    k = min(k, n)
    n_groups = (n + k - 1) // k
    g = save_idx % n_groups
    start = g * k
    end = min(start + k, n)
    return list(range(start, end))


def _list_iter_dirs(save_root: str) -> List[Path]:
    """Return iter_XXXXXXX subdirectories under save_root, ascending."""
    root = Path(save_root)
    if not root.is_dir():
        return []
    out = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if not name.startswith("iter_"):
            continue
        try:
            it = int(name[len("iter_"):])
        except ValueError:
            continue
        out.append((it, p))
    out.sort(key=lambda x: x[0])
    return [p for _, p in out]


def _save_root_from_iter_dir(iter_dir: str) -> str:
    """``.../foo/iter_0000040`` -> ``.../foo``."""
    return str(Path(iter_dir).parent)


def write_pec_metadata(iter_dir: str, iteration: int) -> None:
    """Write moc_pec_metadata.json into ``iter_dir``.

    Idempotent and rank-0-only by convention; caller (moegambit_save_manifest)
    already checks ``torch.distributed.get_rank() == 0`` before calling
    us. Safe to call when ``is_enabled()`` is False — it short-circuits.
    """
    if not is_enabled():
        return
    try:
        save_root = _save_root_from_iter_dir(iter_dir)
        iter_dirs = _list_iter_dirs(save_root)
        # Determine this checkpoint's PEC round index (0-based). We use the
        # rank of this iteration in the ascending list of existing iter_*
        # directories that include this one. This is robust to non-uniform
        # save intervals or resumed runs.
        my_path = Path(iter_dir)
        if my_path not in iter_dirs:
            iter_dirs = sorted(set(iter_dirs + [my_path]), key=lambda p: int(p.name[5:]))
        round_idx = iter_dirs.index(my_path)

        k = _k_pec()
        n = _n_expert()
        fresh = fresh_experts_for_round(round_idx, k, n)
        fresh_set = set(fresh)

        # For every non-fresh expert, walk backwards through round indices
        # until we find the one where that expert was fresh; resolve to the
        # corresponding iter_* directory (if present on disk).
        stale_map: Dict[str, Dict[str, Any]] = {}
        for eid in range(n):
            if eid in fresh_set:
                continue
            # Find the most recent earlier round where eid was fresh.
            chosen_round = None
            for back in range(1, round_idx + 1):
                cand_round = round_idx - back
                if eid in set(fresh_experts_for_round(cand_round, k, n)):
                    chosen_round = cand_round
                    break
            if chosen_round is None or chosen_round >= len(iter_dirs):
                # No earlier ckpt covers this expert yet — fall back to the
                # current dir (matches MoC-System behavior at the very start
                # of training, where no full ckpt exists yet).
                chosen_dir = iter_dir
                chosen_iteration = iteration
            else:
                chosen_dir = str(iter_dirs[chosen_round])
                try:
                    chosen_iteration = int(iter_dirs[chosen_round].name[5:])
                except ValueError:
                    chosen_iteration = -1
            stale_map[str(eid)] = {
                "last_fresh_round": chosen_round if chosen_round is not None else -1,
                "last_fresh_iteration": chosen_iteration,
                "last_fresh_ckpt_dir": chosen_dir,
            }

        metadata = {
            "version": 1,
            "iteration": iteration,
            "round_idx": round_idx,
            "k_pec": k,
            "n_expert": n,
            "schedule": "round_robin",
            "fresh_experts": fresh,
            "stale_experts": stale_map,
            "_note": ("MoC-System (PEC) emulation: ckpt physically holds all "
                      "N experts. This sidecar declares which K_pec experts "
                      "MoC-System would have written this round; the rest map "
                      "to their last-fresh iter_* directory for restore-time "
                      "byte-identical emulation."),
        }
        out_path = Path(iter_dir) / MANIFEST_FILENAME
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as fp:
            json.dump(metadata, fp, indent=2)
        logger.info(
            "[moc_pec_emulation] wrote %s (round=%d, k_pec=%d, n_expert=%d, "
            "fresh=[%d..%d])",
            out_path, round_idx, k, n,
            (fresh[0] if fresh else -1),
            (fresh[-1] if fresh else -1),
        )
    except Exception as exc:  # pragma: no cover — never crash training save
        logger.error("[moc_pec_emulation] write_pec_metadata failed: %s", exc)


def _load_metadata(iter_dir: str) -> Optional[Dict[str, Any]]:
    path = Path(iter_dir) / MANIFEST_FILENAME
    if not path.is_file():
        return None
    try:
        with path.open("r") as fp:
            return json.load(fp)
    except Exception as exc:  # pragma: no cover
        logger.warning("[moc_pec_emulation] failed to read %s: %s", path, exc)
        return None


def apply_pec_to_plan(plan, latest_ckpt_dir: str) -> int:
    """Rewrite ``plan.entries[i].checkpoint_dir`` according to PEC metadata.

    Returns the number of entries that were redirected to an earlier
    ckpt (i.e., entries that are NOT fresh in the latest PEC round).
    No-op when emulation is disabled or when the metadata file is
    missing.

    The ``plan`` object is expected to expose an iterable ``entries``
    attribute whose items have ``expert_id`` and ``checkpoint_dir``
    fields (see ``stale_expert_restore.ExpertRestoreEntry``).
    """
    if not is_enabled():
        return 0
    if plan is None or not getattr(plan, "entries", None):
        return 0
    md = _load_metadata(latest_ckpt_dir)
    if md is None:
        logger.warning(
            "[moc_pec_emulation] enabled but no %s found at %s — "
            "leaving plan unchanged",
            MANIFEST_FILENAME, latest_ckpt_dir,
        )
        return 0
    fresh = set(md.get("fresh_experts", []))
    stale_map: Dict[str, Dict[str, Any]] = md.get("stale_experts", {})
    n_redirected = 0
    for entry in plan.entries:
        eid = getattr(entry, "expert_id", None)
        if eid is None:
            continue
        if eid in fresh:
            continue
        info = stale_map.get(str(eid))
        if not info:
            continue
        new_dir = info.get("last_fresh_ckpt_dir")
        if not new_dir:
            continue
        old_dir = getattr(entry, "checkpoint_dir", None)
        if new_dir == old_dir:
            continue
        try:
            entry.checkpoint_dir = new_dir
            # Also record the historical iteration so downstream stats are
            # correct (e.g. Φ'(t) accounting in MoEGuard sees the right gap).
            if hasattr(entry, "checkpoint_step"):
                entry.checkpoint_step = int(info.get("last_fresh_iteration", -1))
            n_redirected += 1
        except Exception as exc:
            logger.warning(
                "[moc_pec_emulation] could not redirect entry expert_id=%d: %s",
                eid, exc,
            )
    if n_redirected:
        logger.info(
            "[moc_pec_emulation] redirected %d/%d expert load entries to "
            "earlier PEC-fresh checkpoints (latest=%s)",
            n_redirected, len(plan.entries), latest_ckpt_dir,
        )
    return n_redirected


def snapshot_config() -> Dict[str, Any]:
    """Read-only snapshot of current emulation knobs (for logging)."""
    return {
        "enabled": is_enabled(),
        "k_pec": _k_pec(),
        "n_expert": _n_expert(),
        "schedule": os.environ.get("MOEGAMBIT_MOC_PEC_SCHEDULE", "round_robin"),
    }
