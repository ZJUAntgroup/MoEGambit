"""Per-rank completion and committed-recovery evidence, without checkpoints."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

from .io import canonical, digest, integer, nonempty, read_json, write_json


def _manifest(value: Mapping[str, Any]) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError("run manifest must be an object")
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("unsupported run manifest schema")
    for key in ("job_id", "attempt_id", "code_commit"):
        nonempty(value.get(key), key)
    integer(value["world_size"], "world_size", 1)
    integer(value["final_step"], "final_step")
    epochs = value.get("required_recovery_epochs")
    if not isinstance(epochs, list) or len(epochs) != len(set(epochs)):
        raise ValueError("required_recovery_epochs must be a unique array")
    for epoch in epochs:
        integer(epoch, "recovery_epoch", 1)
    if not isinstance(value.get("required_files", []), list):
        raise ValueError("required_files must be an array")
    for path in value.get("required_files", []):
        _relative(path)
    return dict(value)


def _relative(path: str) -> Path:
    value = Path(nonempty(path, "relative evidence path"))
    if value.is_absolute() or ".." in value.parts:
        raise ValueError("evidence paths must be relative and cannot contain '..'")
    return value


def _safe_file(root: Path, path: Path) -> bool:
    """Reject linked files or linked directories in the evidence path."""
    relative = path.relative_to(root)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return False
    return path.is_file()


class RunEvidenceWriter:
    """One writer per rank/process; refuses duplicates rather than overwriting.

    Construct with the same manifest on every rank, then pass record_recovery
    to RecoveryRuntime(event_sink=...). Call complete ONLY after the loop ends.
    A restarted process needs a new attempt_id/directory, not stale rank files.
    """
    def __init__(self, root: str | Path, manifest: Mapping[str, Any], rank: int):
        self.root = Path(root)
        self.manifest = _manifest(manifest)
        self.rank = integer(rank, "rank")
        if self.rank >= self.manifest["world_size"]:
            raise ValueError("rank is outside world_size")
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "rank_results").mkdir(exist_ok=True)
        (self.root / "recovery_events").mkdir(exist_ok=True)
        manifest_path = self.root / "run_manifest.json"
        try:
            write_json(manifest_path, self.manifest, exclusive=True)
        except FileExistsError:
            if read_json(manifest_path) != self.manifest:
                raise ValueError("run manifest differs between ranks")
        self.claim = self.root / "rank_results" / f"rank_{rank}.claim.json"
        with self.claim.open("xb") as handle:
            handle.write(canonical({"rank": rank, "pid": os.getpid(),
                                    "manifest_digest": digest(self.manifest)}) + b"\n")
        self.finished = False
        self.events = self.root / "recovery_events" / f"rank_{rank}.jsonl"
        if self.events.exists():
            raise FileExistsError("existing recovery events cannot be reused by a new writer")

    def record_recovery(self, record) -> None:
        if self.finished:
            raise ValueError("cannot append recovery evidence after completion")
        value = dict(record.to_dict() if hasattr(record, "to_dict") else record)
        for key in ("job_id", "attempt_id"):
            if value.get(key) != self.manifest[key]:
                raise ValueError(f"recovery record {key} differs from manifest")
        value["rank"] = self.rank
        value["schema_version"] = 1
        content = canonical(value) + b"\n"
        with self.events.open("ab") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())

    def complete(self, final_step: int, *, exit_code: int = 0) -> None:
        if self.finished:
            raise ValueError("rank completion already published")
        integer(final_step, "final_step")
        integer(exit_code, "exit_code")
        value = {"schema_version": 1, "job_id": self.manifest["job_id"],
                 "attempt_id": self.manifest["attempt_id"], "rank": self.rank,
                 "world_size": self.manifest["world_size"], "final_step": final_step,
                 "code_commit": self.manifest["code_commit"], "exit_code": exit_code,
                 "status": "completed" if exit_code == 0 else "failed",
                 "manifest_digest": digest(self.manifest)}
        write_json(self.root / "rank_results" / f"rank_{self.rank}.json", value, exclusive=True)
        self.finished = True


def verify_run(root: str | Path) -> dict:
    root = Path(root)
    failures, completed, commits, provisional = [], [], {}, set()
    try:
        if not _safe_file(root, root / "run_manifest.json"):
            raise ValueError("run manifest is missing or linked")
        manifest = _manifest(read_json(root / "run_manifest.json"))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"schema_version": 1, "passed": False, "failures": [f"manifest: {exc}"]}
    world = manifest["world_size"]
    rank_dir = root / "rank_results"
    # Reject unrecognized/duplicate completion names; claim files aren't completions.
    for path in sorted(rank_dir.glob("*.json")):
        if path.name.endswith(".claim.json"):
            continue
        try:
            if not _safe_file(root, path):
                raise ValueError("completion file or directory is linked")
            result = read_json(path)
            if not isinstance(result, dict) or type(result.get("schema_version")) is not int or result["schema_version"] != 1:
                raise ValueError("unsupported completion schema")
            rank = integer(result["rank"], "rank")
            if rank >= world or path.name != f"rank_{rank}.json":
                raise ValueError("unexpected completion filename/rank")
            for key in ("job_id", "attempt_id", "world_size", "code_commit", "final_step"):
                if result[key] != manifest[key] or type(result[key]) != type(manifest[key]):
                    raise ValueError(f"rank {key} differs from manifest")
            if result.get("manifest_digest") != digest(manifest):
                raise ValueError("completion is bound to a different manifest")
            if result.get("status") != "completed" or type(result.get("exit_code")) is not int or result["exit_code"] != 0:
                raise ValueError("rank did not exit successfully")
            completed.append(rank)
        except (OSError, KeyError, ValueError, TypeError) as exc:
            failures.append(f"{path.name}: {exc}")
    missing = sorted(set(range(world)) - set(completed))
    if missing:
        failures.append(f"missing successful ranks: {missing}")
    for path in sorted((root / "recovery_events").glob("*.jsonl")):
        try:
            if not _safe_file(root, path):
                raise ValueError("recovery event file or directory is linked")
            rank = int(path.stem.removeprefix("rank_"))
            if rank not in range(world) or path.name != f"rank_{rank}.jsonl":
                raise ValueError("unexpected event filename/rank")
            for line_number, line in enumerate(path.read_text().splitlines(), 1):
                # Same strict decoder as JSON files (including duplicate keys).
                from .io import _pairs
                record = json.loads(line, object_pairs_hook=_pairs,
                                    parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
                if not isinstance(record, dict):
                    raise ValueError("recovery event must be an object")
                for key in ("job_id", "attempt_id"):
                    if record[key] != manifest[key]:
                        raise ValueError(f"line {line_number}: stale {key}")
                if type(record.get("rank")) is not int or record["rank"] != rank:
                    raise ValueError("event rank differs from file")
                if type(record.get("schema_version")) is not int or record["schema_version"] != 1:
                    raise ValueError("unsupported recovery event schema")
                epoch = integer(record["recovery_epoch"], "recovery_epoch", 1)
                failed = record.get("failed_ranks")
                if not isinstance(failed, list) or not failed or len(failed) != len(set(failed)):
                    raise ValueError("failed_ranks must be a non-empty unique array")
                for failed_rank in failed:
                    if integer(failed_rank, "failed rank") >= world:
                        raise ValueError("failed rank is outside world_size")
                if record["result"] in ("aborted", "fallback"):
                    failures.append(f"rank {rank} epoch {epoch}: {record['result']}")
                elif record["result"] == "committed":
                    nonempty(record.get("topology_manifest"), "topology_manifest")
                    if record.get("decision") not in ("peer", "hybrid", "checkpoint"):
                        raise ValueError("unrecognized committed decision")
                    validation = record["validation"]
                    step = integer(validation["committed_step"], "committed_step")
                    resume = integer(record["resume_step"], "resume_step")
                    if step <= resume or step > manifest["final_step"] or validation.get("provisional") is not False:
                        raise ValueError("recovery lacks a valid committed full step")
                    key = (epoch, rank)
                    if key in commits:
                        raise ValueError("duplicate committed recovery event")
                    commits[key] = record
                elif record["result"] == "provisional":
                    if (epoch, rank) in commits:
                        raise ValueError("provisional event after commit")
                    provisional.add((epoch, rank))
                else:
                    raise ValueError("unrecognized recovery outcome")
        except (OSError, KeyError, ValueError, TypeError) as exc:
            failures.append(f"{path.name}: {exc}")
    for epoch, rank in sorted(provisional - set(commits)):
        failures.append(f"rank {rank} epoch {epoch}: unresolved provisional recovery")
    expected_epochs = set(manifest["required_recovery_epochs"])
    if {epoch for epoch, _ in commits} - expected_epochs:
        failures.append("unexpected committed recovery epoch")
    for epoch in sorted(expected_epochs):
        records = [commits.get((epoch, rank)) for rank in range(world)]
        if any(record is None for record in records):
            failures.append(f"epoch {epoch}: missing per-rank commit evidence")
            continue
        if any(record["validation"]["committed_step"] != records[0]["validation"]["committed_step"] for record in records[1:]):
            failures.append(f"epoch {epoch}: ranks disagree on committed step")
        for field in ("resume_step", "failed_ranks", "decision", "topology_manifest"):
            if any(record.get(field) != records[0].get(field) for record in records[1:]):
                failures.append(f"epoch {epoch}: ranks disagree on {field}")
    for name in manifest.get("required_files", []):
        path = root / _relative(name)
        if not _safe_file(root, path) or path.stat().st_size == 0:
            failures.append(f"missing/empty/linked required evidence: {name}")
    return {"schema_version": 1, "passed": not failures, "world_size": world,
            "successful_ranks": sorted(completed), "required_recovery_epochs": sorted(expected_epochs),
            "committed_events": len(commits), "failures": failures,
            "scope": "recorded completion and recovery consistency; not a quality-risk proof"}


_ALLOWED_NAMES = {"run_manifest.json", "pipeline_summary.json", "quality_results.csv",
                  "static_summary.csv", "static_cases.csv", "selection.json",
                  "selected_500.txt", "selected_full.txt", "short_plan.json",
                  "short_experiments_summary.json", "short_static_vs_continuation.csv",
                  "short_static_vs_continuation.json", "baseline_audit.json",
                  "baseline_training_loss_audit.json", "router_checkpoint_state.json",
                  "paired_quality.json", "replay_check.json", "training_loss_comparison.json",
                  "state_audit.json", "expected_state.json", "observed_state.json",
                  "before_state.json", "risk_audit.json", "verification.json"}
_PRUNED = {"ckpt", "checkpoints", "checkpoint", "model_states", "optimizer", "optim",
           ".git", ".env", "__pycache__", "wandb", "tensorboard"}


def collect_evidence(root: str | Path, output: str | Path, *, include_logs: bool = False,
                     max_bytes: int = 100 * 1024 * 1024, max_files: int = 10_000) -> dict:
    """Copy only small result/evidence files, pruning checkpoint trees.

    Existing result pipelines can be exported without claiming that their
    older logs fulfill the new per-rank completion contract. No torch.load.
    """
    root, output = Path(root).resolve(), Path(output).absolute()
    integer(max_bytes, "max_bytes", 1)
    integer(max_files, "max_files", 1)
    if not root.is_dir():
        raise ValueError("evidence root is not a directory")
    if output.exists() or output.is_symlink():
        raise FileExistsError("output already exists; refusing to overwrite evidence")
    resolved_output = output.resolve()
    if resolved_output == root or root in resolved_output.parents:
        raise ValueError("output must be outside the input tree")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".moegambit-export-", dir=output.parent))
    copied, omitted, total = [], [], 0
    try:
        for directory, dirs, files in os.walk(root, followlinks=False):
            base = Path(directory)
            dirs[:] = sorted(d for d in dirs if d.lower() not in _PRUNED
                             and not d.startswith(".") and not (base / d).is_symlink())
            for name in sorted(files):
                path = base / name
                relative = path.relative_to(root)
                audit_folder = relative.parts[0] in ("rank_results", "recovery_events")
                static_case = relative.parts[0] == "static" and path.suffix == ".json"
                allowed = (name in _ALLOWED_NAMES or static_case or
                           (audit_folder and path.suffix in (".json", ".jsonl")) or
                           (include_logs and path.suffix == ".log" and not name.startswith(".")))
                if not allowed or path.is_symlink() or not path.is_file():
                    continue
                if len(copied) >= max_files:
                    raise ValueError("evidence file limit exceeded; narrow the input root")
                size = path.stat().st_size
                if total + size > max_bytes:
                    omitted.append({"path": str(relative), "reason": "byte_budget"})
                    continue
                target = temporary / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                fingerprint = hashlib.sha256()
                actual = 0
                with path.open("rb") as source, target.open("xb") as destination:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        actual += len(chunk)
                        if total + actual > max_bytes:
                            raise ValueError("source grew during export and exceeded the byte budget")
                        fingerprint.update(chunk)
                        destination.write(chunk)
                copied.append({"path": str(relative), "bytes": actual,
                               "sha256": fingerprint.hexdigest()})
                total += actual
        report = {"schema_version": 1, "files": copied, "total_bytes": total,
                  "omitted": omitted, "include_logs": include_logs,
                  "verification": verify_run(root),
                  "checkpoint_payloads_included": False}
        write_json(temporary / "evidence_manifest.json", report)
        os.rename(temporary, output)
        return report
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
