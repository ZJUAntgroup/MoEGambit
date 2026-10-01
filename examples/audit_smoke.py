"""CPU-only synthetic audit-tool smoke run; NOT recovery/quality validation."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from moegambit.audit.io import write_json
from moegambit.audit.state import audit_state, capture_catalog
from moegambit.audit.run import RunEvidenceWriter, verify_run
from moegambit.audit.risk import risk_audit
from moegambit.state.catalog import StateCatalog, StateRef, StateKind, Placement
from moegambit.state.version import StateVersion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", required=True)
    args = parser.parse_args()
    root = Path(args.log_dir)
    version = StateVersion(100, 100, 1)
    catalog = StateCatalog.from_iterable([
        StateRef("optimizer/step", StateKind.OPTIMIZER_SCALAR, Placement.REPLICATED,
                 -1, version, scalar_get=lambda: 100),
        StateRef("data/progress", StateKind.DATALOADER, Placement.REPLICATED,
                 -1, version, scalar_get=lambda: {"consumed_samples": 6400}),
    ])
    expected = capture_catalog(catalog)
    observed = capture_catalog(catalog)
    write_json(root / "expected_state.json", expected)
    write_json(root / "observed_state.json", observed)
    write_json(root / "state_audit.json", audit_state(expected, observed))
    manifest = {"schema_version": 1, "job_id": "synthetic-audit-smoke", "attempt_id": "a1",
                "world_size": 2, "final_step": 100, "code_commit": "synthetic-no-training",
                "required_recovery_epochs": [1], "required_files": ["state_audit.json"]}
    for rank in range(2):
        writer = RunEvidenceWriter(root, manifest, rank)
        writer.record_recovery({
            "job_id": manifest["job_id"], "attempt_id": "a1", "recovery_epoch": 1,
            "result": "committed", "resume_step": 90, "failed_ranks": [1], "decision": "peer",
            "topology_manifest": "synthetic-topology",
            "validation": {"committed_step": 91, "provisional": False}})
        writer.complete(100)
    write_json(root / "verification.json", verify_run(root))
    write_json(root / "risk_audit.json", risk_audit(59, 0))
    print(f"Synthetic smoke run finished: {root}")
    print("These synthetic records do not establish model quality or recovery correctness.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
