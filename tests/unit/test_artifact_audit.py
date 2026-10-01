"""Missing state, corrupt evidence, concurrent writers and exact risk bounds."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import math

import pytest
from moegambit.audit.io import read_json, write_json
from moegambit.audit.state import audit_state, capture_catalog
from moegambit.audit.run import RunEvidenceWriter, collect_evidence, verify_run
from moegambit.audit.risk import binomial_upper, risk_audit
from moegambit.cli.audit import main
from moegambit.state.catalog import StateCatalog, StateRef, StateKind, Placement
from moegambit.state.version import StateVersion


def state_manifest():
    return {"schema_version": 1, "states": [
        {"identity": identity, "kind": kind, "placement": "replicated", "owner": -1,
         "version": {"committed_step": 1000, "optimizer_generation": 1000, "recovery_epoch": 1},
         "shape": [2] if kind != "rng" else [], "dtype": "float32" if kind != "rng" else "json",
         "sha256": str(index + 1) * 64}
        for index, (identity, kind) in enumerate([
            ("parameter", "parameter"), ("fp32_master", "optimizer_tensor"),
            ("exp_avg", "optimizer_tensor"), ("exp_avg_sq", "optimizer_tensor"),
            ("router_bias", "buffer"), ("rng", "rng")])]}


def test_full_state_matches_with_explicit_source_versions():
    want = state_manifest(); got = deepcopy(want)
    assert audit_state(want, got, before=want, affected=["parameter"])["passed"]


@pytest.mark.parametrize("identity", ["fp32_master", "exp_avg", "exp_avg_sq", "router_bias", "rng"])
def test_state_audit_detects_missing_non_parameter_state(identity):
    expected = state_manifest(); got = deepcopy(expected)
    got["states"] = [s for s in got["states"] if s["identity"] != identity]
    report = audit_state(expected, got)
    assert {"identity": identity, "reason": "missing"} in report["failures"]


@pytest.mark.parametrize("field,value", [("owner", 1), ("dtype", "bfloat16"),
    ("shape", [1, 2]), ("sha256", "f" * 64), ("kind", "buffer"),
    ("version", {"committed_step": 800, "optimizer_generation": 1000, "recovery_epoch": 1})])
def test_state_audit_detects_value_layout_owner_and_provenance(field, value):
    expected = state_manifest(); got = deepcopy(expected)
    got["states"][0][field] = value
    assert not audit_state(expected, got)["passed"]


def test_unaffected_state_cannot_change_even_if_expected_changed():
    before = state_manifest(); expected = deepcopy(before)
    expected["states"][1]["sha256"] = "f" * 64
    report = audit_state(expected, expected, before=before, affected=["parameter"])
    assert any(f["reason"] == "unaffected_changed" for f in report["failures"])


def test_state_audit_rejects_duplicate_inventory_and_invalid_shape():
    expected = state_manifest()
    expected["states"].append(expected["states"][0])
    with pytest.raises(ValueError, match="duplicate"):
        audit_state(expected, expected)
    expected = state_manifest(); expected["states"][0]["shape"] = [True]
    with pytest.raises(ValueError): audit_state(expected, expected)


def test_capture_actual_scalar_state():
    ref = StateRef("rng/python", StateKind.RNG, Placement.UNIQUE, 1,
                   StateVersion(100, 100, 1), scalar_get=lambda: [1, 2, 3])
    catalog = StateCatalog.from_iterable([ref])
    first = capture_catalog(catalog)
    ref.scalar_get = lambda: [1, 2, 4]
    assert not audit_state(first, capture_catalog(catalog))["passed"]


def test_capture_tensor_values_dtype_and_bfloat16():
    torch = pytest.importorskip("torch")
    ref = StateRef("master", StateKind.OPTIMIZER_TENSOR, Placement.REPLICATED, -1,
                   StateVersion(100, 100, 1), tensor=torch.tensor([1, 2], dtype=torch.bfloat16))
    catalog = StateCatalog.from_iterable([ref]); expected = capture_catalog(catalog)
    assert audit_state(expected, capture_catalog(catalog))["passed"]
    ref.tensor[0] = 3
    assert not audit_state(expected, capture_catalog(catalog))["passed"]
    ref.tensor[0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"): capture_catalog(catalog)


def manifest(epochs=None):
    return {"schema_version": 1, "job_id": "job", "attempt_id": "a1", "world_size": 2,
            "final_step": 100, "code_commit": "abcdef", "required_recovery_epochs": epochs or []}


def record(rank=0, epoch=1):
    return {"schema_version": 1, "job_id": "job", "attempt_id": "a1", "rank": rank, "recovery_epoch": epoch,
            "result": "committed", "resume_step": 90, "failed_ranks": [1], "decision": "peer",
            "topology_manifest": "topo", "validation": {"committed_step": 91, "provisional": False}}


def finish(root, epochs=None):
    writers = [RunEvidenceWriter(root, manifest(epochs), rank) for rank in range(2)]
    for writer in writers:
        for epoch in epochs or []: writer.record_recovery(record(writer.rank, epoch))
        writer.complete(100)
    return writers


def test_verify_requires_every_rank_and_every_commit(tmp_path):
    writer = RunEvidenceWriter(tmp_path, manifest([1]), 0)
    writer.record_recovery(record()); writer.complete(100)
    assert not verify_run(tmp_path)["passed"]
    writer = RunEvidenceWriter(tmp_path, manifest([1]), 1)
    writer.record_recovery(record(1)); writer.complete(100)
    assert verify_run(tmp_path)["passed"]


def test_manifest_publication_is_atomic_across_ranks(tmp_path):
    with ThreadPoolExecutor(2) as pool:
        writers = list(pool.map(lambda rank: RunEvidenceWriter(tmp_path, manifest(), rank), range(2)))
    for writer in writers: writer.complete(100)
    assert verify_run(tmp_path)["passed"]


@pytest.mark.parametrize("mutation", ["wrong_step", "stale", "failed", "empty", "duplicate", "wrong_commit"])
def test_verify_rejects_incomplete_or_conflicting_results(tmp_path, mutation):
    finish(tmp_path, [1])
    p = tmp_path / "rank_results/rank_1.json"; value = read_json(p)
    if mutation == "wrong_step": value["final_step"] = 99
    elif mutation == "stale": value["attempt_id"] = "a0"
    elif mutation == "failed": value["exit_code"] = 137
    elif mutation == "empty": p.write_text(""); assert not verify_run(tmp_path)["passed"]; return
    elif mutation == "duplicate": write_json(p.with_name("duplicate.json"), value)
    else: value["code_commit"] = "wrong"
    write_json(p, value)
    assert not verify_run(tmp_path)["passed"]


@pytest.mark.parametrize("mutation", ["duplicate", "provisional", "truncated", "fallback", "disagreement"])
def test_verify_recovery_commit_is_not_first_forward(tmp_path, mutation):
    finish(tmp_path, [1]); path = tmp_path / "recovery_events/rank_1.jsonl"
    value = record(1)
    if mutation == "duplicate": path.write_text(path.read_text() * 2)
    elif mutation == "provisional": value["result"] = "provisional"; path.write_text(json.dumps(value) + "\n")
    elif mutation == "truncated": path.write_text('{"result":')
    elif mutation == "fallback": value["result"] = "fallback"; path.write_text(json.dumps(value) + "\n")
    else: value["failed_ranks"] = [0]; path.write_text(json.dumps(value) + "\n")
    assert not verify_run(tmp_path)["passed"]


def test_duplicate_writer_rejected_and_no_double_completion(tmp_path):
    writer = RunEvidenceWriter(tmp_path, manifest(), 0)
    with pytest.raises(FileExistsError): RunEvidenceWriter(tmp_path, manifest(), 0)
    writer.complete(100)
    with pytest.raises(ValueError): writer.complete(100)
    with pytest.raises(ValueError): writer.record_recovery(record())


def test_compact_export_skips_checkpoints_secrets_symlinks_and_logs_by_default(tmp_path):
    root = tmp_path / "run"; finish(root)
    (root / "ckpt").mkdir(); write_json(root / "ckpt/paired_quality.json", {"large": True})
    (root / ".env").write_text("secret")
    (root / "tokens.json").write_text("secret")
    (root / "node0.log").write_text("optional log")
    (root / "quality_results.csv").write_text("case,Y\na,0.1\n")
    (root / "paired_quality.json").symlink_to(root / "tokens.json")
    target = tmp_path / "bundle"
    result = collect_evidence(root, target)
    names = {v["path"] for v in result["files"]}
    assert result["verification"]["passed"]
    assert "quality_results.csv" in names
    assert not {"paired_quality.json", "ckpt/paired_quality.json", ".env", "node0.log", "tokens.json"} & names
    assert read_json(target / "evidence_manifest.json")["checkpoint_payloads_included"] is False
    with pytest.raises(FileExistsError): collect_evidence(root, target)
    with pytest.raises(ValueError): collect_evidence(root, root / "nested")


def test_export_budget_and_legacy_incompleteness_are_reported(tmp_path):
    root = tmp_path / "legacy"; root.mkdir()
    (root / "quality_results.csv").write_text("x" * 100)
    report = collect_evidence(root, tmp_path / "bundle", max_bytes=10)
    assert report["omitted"][0]["reason"] == "byte_budget"
    assert report["verification"]["passed"] is False


def test_exact_one_sided_bounds_and_sample_size():
    assert binomial_upper(59, 0) < .05 < binomial_upper(58, 0)
    assert binomial_upper(15, 0) == pytest.approx(1 - .05 ** (1 / 15))
    assert binomial_upper(10, 10) == 1
    assert binomial_upper(10, 1) == pytest.approx(.3941633024)
    assert not risk_audit(15, 0)["within_budget"]
    assert risk_audit(59, 0)["conditional_guarantee"] is False
    with pytest.raises(ValueError): binomial_upper(0, 0)
    with pytest.raises(ValueError): binomial_upper(5, 6)
    with pytest.raises(ValueError): binomial_upper(True, 0)
    with pytest.raises(ValueError): binomial_upper(5, 0, math.nan)


def test_cli_writes_failures_and_returns_nonzero(tmp_path):
    output = tmp_path / "verify.json"
    assert main(["verify-run", "--root", str(tmp_path), "--output", str(output)]) == 1
    assert read_json(output)["passed"] is False
    assert main(["risk", "--trials", "59", "--violations", "0", "--output", str(tmp_path / "risk.json")]) == 0
    assert main(["risk", "--trials", "0", "--violations", "0", "--output", str(tmp_path / "bad.json")]) == 2
    assert "error" in read_json(tmp_path / "bad.json")


def test_strict_json_rejects_duplicate_fields(tmp_path):
    p = tmp_path / "duplicate.json"; p.write_text('{"rank":0,"rank":1}')
    with pytest.raises(ValueError, match="duplicate"): read_json(p)


def test_verify_rejects_symlinked_evidence_and_empty_manifest_object(tmp_path):
    root = tmp_path / "run"; finish(root)
    original = root / "rank_results/rank_1.json"
    external = tmp_path / "outside.json"; original.rename(external)
    original.symlink_to(external)
    assert not verify_run(root)["passed"]
    write_json(root / "run_manifest.json", [])
    assert not verify_run(root)["passed"]
    with pytest.raises(ValueError): audit_state([], [])
