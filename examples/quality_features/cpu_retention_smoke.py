"""CPU-only transport smoke: acknowledged features survive an abrupt worker exit.

Uses synthetic telemetry. Does not benchmark recovery, GPU offload or quality.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

SOURCE = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SOURCE))

from moegambit.audit.io import write_json
from moegambit.control.service import RecoveryCoordinatorService
from moegambit.control.watcher import ControlRequestProcessor, ControlServer
from moegambit.quality import AsyncQualityOffloader, CPUQualityFeatureStore
from moegambit.runtime.client import ControlClient, ControlClientConfig

IDENTITY = dict(model_id="synthetic-moe", telemetry_version="smoke-v1", policy_version="smoke-v1")


def worker(host, port, rank):
    client = ControlClient(ControlClientConfig(host, port, "cpu-retention-smoke", "a1",
        {"global_rank": rank}, job_token="local-smoke-token", request_timeout_s=5))
    uploader = AsyncQualityOffloader(client, **IDENTITY)
    future = uploader.submit({"layer_ids": [0], "expert_ids": [rank],
        "routed_mass": [.125], "sensitivity": [1.], "parameter_drift": [.01],
        "momentum_drift": [.02], "variance_drift": [.03], "learning_rate": .0001,
        "training_fraction": .5, "routing_change": .001},
        committed_step=1000, checkpoint_step=800, topology_generation=0,
        group_manifest_hash="synthetic-layout", world_size=2, run_history=[])
    future.result(timeout=10)
    os._exit(23 if rank == 1 else 0)  # skip teardown: the worker's RAM is lost


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default="/personal/moegambit_quality_cpu_smoke")
    parser.add_argument("--worker", nargs=3, metavar=("HOST", "PORT", "RANK"))
    args = parser.parse_args()
    if args.worker:
        worker(args.worker[0], int(args.worker[1]), int(args.worker[2]))
        return
    root = Path(args.log_dir)
    root.mkdir(parents=True, exist_ok=True)
    store = CPUQualityFeatureStore()
    service = RecoveryCoordinatorService(store_provider=lambda payload: {}, quality_feature_store=store)
    processor = ControlRequestProcessor(service, job_token="local-smoke-token", require_token=True)
    with ControlServer("127.0.0.1", 0, processor) as server:
        host, port = server.address
        exits = []
        for rank in (0, 1):
            child = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker",
                                    host, str(port), str(rank)], capture_output=True, timeout=20)
            (root / f"worker_{rank}.log").write_bytes(child.stdout + child.stderr)
            exits.append(child.returncode)
        if exits != [0, 23]:
            raise RuntimeError(f"worker smoke failed: exits={exits}; inspect {root}")
        query = {"quality_context": IDENTITY, "quality_source_topology_generation": 0,
                 "recovery_epoch": 1, "quality_source_recovery_epoch": 0,
                 "topology_generation": 1, "group_manifest_hash": "synthetic-layout",
                 "world_size": 2, "at_step": 1000, "latest_checkpoint_step": 800}
        retained = store.recovery_context(query, job_id="cpu-retention-smoke", attempt_id="a1")
        missing = store.recovery_context({**query, "at_step": 1001},
                                         job_id="cpu-retention-smoke", attempt_id="a1")
        assert retained["retained_features_complete"] is True
        assert missing["retained_features_complete"] is False
        write_json(root / "result.json", {"kind": "synthetic_cpu_retention_smoke",
            "worker_exit_codes": exits, "retained_context": retained,
            "unpublished_next_step": missing, "statistical_risk_certificate": False})
    print(f"CPU retention smoke passed; synthetic results: {root / 'result.json'}")


if __name__ == "__main__":
    main()
