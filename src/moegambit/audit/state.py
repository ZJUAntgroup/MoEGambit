"""Exact fingerprints of explicitly inventoried recovery state.

The expected inventory must come from independently captured source payloads,
with logical destination ownership and the recovery plan's selected versions.
Hashing two incomplete catalogs cannot establish complete engine recovery.
"""
from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping

from .io import digest, integer, nonempty
from ..state.catalog import StateCatalog, StateKind, Placement


def capture_catalog(catalog: StateCatalog) -> dict:
    rows = []
    for ref in catalog:
        if ref.tensor is not None:
            # Optional torch dependency is only needed when capturing tensors.
            import torch
            tensor = ref.tensor.detach().cpu().contiguous()
            if tensor.layout is not torch.strided:
                raise ValueError(f"unsupported tensor layout for {ref.identity}")
            if tensor.is_floating_point() or tensor.is_complex():
                if not bool(torch.isfinite(tensor).all()):
                    raise ValueError(f"non-finite tensor state: {ref.identity}")
            content = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            fingerprint = hashlib.sha256(content).hexdigest()
            shape, dtype = list(tensor.shape), str(tensor.dtype)
        else:
            fingerprint = digest(ref.scalar_get())
            shape, dtype = [], "json"
        rows.append({
            "identity": ref.identity, "kind": ref.kind.value,
            "placement": ref.placement.value, "owner": ref.owner,
            "version": {"committed_step": ref.version.committed_step,
                        "optimizer_generation": ref.version.optimizer_generation,
                        "recovery_epoch": ref.version.recovery_epoch},
            "shape": shape, "dtype": dtype, "sha256": fingerprint,
        })
    result = {"schema_version": 1, "states": rows}
    _inventory(result)
    return result


def _inventory(manifest: Mapping[str, Any]) -> dict:
    if not isinstance(manifest, Mapping):
        raise ValueError("state manifest must be an object")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("unsupported state manifest schema")
    rows = manifest.get("states")
    if not isinstance(rows, list) or not rows:
        raise ValueError("state manifest requires a non-empty states array")
    result = {}
    for row in rows:
        identity = nonempty(row["identity"], "state identity")
        if identity in result:
            raise ValueError(f"duplicate state identity: {identity}")
        StateKind(row["kind"])
        Placement(row["placement"])
        integer(row["owner"], "owner", -1)
        for key in ("committed_step", "optimizer_generation", "recovery_epoch"):
            integer(row["version"][key], key)
        if not isinstance(row["shape"], list):
            raise ValueError("shape must be an array")
        for dim in row["shape"]:
            integer(dim, "shape dimension")
        nonempty(row["dtype"], "dtype")
        fingerprint = row["sha256"]
        if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(
            c not in "0123456789abcdef" for c in fingerprint
        ):
            raise ValueError("state sha256 must be a lowercase SHA-256 digest")
        result[identity] = row
    return result


def audit_state(expected: Mapping[str, Any], observed: Mapping[str, Any], *,
                before: Mapping[str, Any] | None = None,
                affected: Iterable[str] = ()) -> dict:
    """Compare values, kinds, ownership, shapes, dtype and exact provenance.

    With before, values and versions of every unaffected identity must remain
    identical except recovery_epoch, which may advance during reintegration.
    """
    want, got = _inventory(expected), _inventory(observed)
    affected = set(affected)
    if not affected <= set(want):
        raise ValueError("affected identities must belong to the expected inventory")
    failures = []
    for identity in sorted(set(want) - set(got)):
        failures.append({"identity": identity, "reason": "missing"})
    for identity in sorted(set(got) - set(want)):
        failures.append({"identity": identity, "reason": "unexpected"})
    fields = ("kind", "placement", "owner", "version", "shape", "dtype", "sha256")
    for identity in sorted(set(want) & set(got)):
        for field in fields:
            if want[identity][field] != got[identity][field]:
                failures.append({"identity": identity, "reason": "mismatch", "field": field})
    if before is not None:
        previous = _inventory(before)
        if set(previous) != set(want):
            raise ValueError("before and expected inventories must have identical identities")
        for identity in sorted((set(want) & set(got)) - affected):
            for field in fields:
                old, new = previous[identity][field], got[identity][field]
                if field == "version":
                    old = {k: v for k, v in old.items() if k != "recovery_epoch"}
                    new = {k: v for k, v in new.items() if k != "recovery_epoch"}
                if old != new:
                    failures.append({"identity": identity, "reason": "unaffected_changed", "field": field})
    return {"schema_version": 1, "passed": not failures,
            "expected_count": len(want), "observed_count": len(got),
            "expected_digest": digest(expected), "observed_digest": digest(observed),
            "unaffected_checked": before is not None, "failures": failures,
            "scope": "explicit inventory only; exact content and declared provenance"}
