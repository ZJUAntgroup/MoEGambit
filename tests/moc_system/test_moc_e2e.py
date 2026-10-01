"""CPU tests for surviving cache, actual worker lifetime, and result integrity."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from moc_e2e_launch import CacheServer, cache_request, phase_command
from moc_e2e_plan import manifest
from moc_e2e_summarize import validate_job, summarize
from moc_timing_launch import atomic_json
from moc_e2e_pretrain import install_training_hooks


class EndToEndTests(unittest.TestCase):
    def test_public_entrypoint_plan_is_portable_and_preserves_legacy_scratch_alias(self):
        script = Path(__file__).resolve().parents[2] / "examples/moc_system/run_moc_e2e.sh"
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            scratch = directory / "scratch"
            tokenizer = directory / "tokenizer with spaces"
            env = {"PATH": os.environ["PATH"], "MASTER_ADDR": "127.0.0.1",
                   "BASE_DIR": str(directory / "baseline"), "FSE_CKPT_ROOT": str(scratch),
                   "TOKENIZER_DIR": str(tokenizer), "RUN_ID": "moc_plan_test",
                   "LOG_DIR": "/personal/moegambit/test_plans", "NODE_RANK": "7"}
            run = subprocess.run(["bash", str(script), "--plan-only"], cwd=directory,
                                 env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr)
            cfg, _ = json.JSONDecoder().raw_decode(run.stdout)
            self.assertEqual(cfg["benchmark_training_steps"], 660)
            self.assertEqual(cfg["scratch"], str(scratch))
            self.assertEqual(cfg["result_dir"], "/personal/moegambit/test_plans/moc_plan_test")
            command = shlex.split(run.stdout.split("[moc-e2e] worker template:", 1)[1])
            self.assertEqual(command[command.index("--tokenizer-model") + 1], str(tokenizer))
            self.assertEqual(command[command.index("--train-iters") + 1], "10000")
            self.assertNotIn("--save", command)
            self.assertFalse(scratch.exists())
            self.assertEqual(list(directory.iterdir()), [])

    def test_phase_stop_and_load_use_native_megatron_without_changing_lr_schedule(self):
        with patch.dict(os.environ, {"MASTER_ADDR": "127.0.0.1"}, clear=True):
            cfg = manifest()
        command = ["torchrun", "--master_port=20140", "worker.py", "--train-iters", "10000"]
        for index, job in enumerate(cfg["jobs"]):
            for phase in ("prefix", "resume"):
                argv = phase_command(command, cfg, job, phase, index)
                self.assertEqual(argv[argv.index("--train-iters") + 1], "10000")
                stop = cfg["failure_step"] if phase == "prefix" else cfg["endpoint"]
                self.assertEqual(argv[argv.index("--exit-interval") + 1], str(stop))
                native_resume = phase == "resume" and job["arm"] == "full_sync_native"
                step = cfg["checkpoint_step"] if native_resume else cfg["step"]
                self.assertEqual(argv[argv.index("--ckpt-step") + 1], str(step))
                load = (Path(cfg["scratch"]) / job["id"] / "native" if native_resume
                        else Path(cfg["base_dir"]) / "ckpt")
                self.assertEqual(argv[argv.index("--load") + 1], str(load))
                self.assertIn(f"--master_port={cfg['master_port'] + index * 2 + (phase == 'resume')}", argv)

    def test_native_reexport_is_patched_for_data_audit(self):
        old = lambda: "unwrapped"
        wrapped = lambda: "audited"
        package = SimpleNamespace(pretrain=old, training=SimpleNamespace(pretrain=old))
        install_training_hooks(package, "setup", "step", wrapped)
        self.assertEqual(package.pretrain(), "audited")
        self.assertIs(package.pretrain, package.training.pretrain)
        self.assertEqual(package.training.train_step, "step")

    def test_plan_budget_and_real_cadence(self):
        with patch.dict(os.environ, {"RUN_ID": "test", "LOG_DIR": "/personal/test", "MOC_FAILURE_COUNTS": "1",
                                     "MOC_REPEATS": "1", "MOC_CKPT_INTERVAL": "200"}, clear=True):
            cfg = manifest()
            self.assertEqual(cfg["checkpoint_step"], 4200)
            self.assertEqual(cfg["endpoint"], 4215)
            self.assertEqual(cfg["benchmark_training_steps"], 660)
            self.assertEqual(len(cfg["jobs"]), 3)
        with patch.dict(os.environ, {"MOC_FAILURE_COUNTS": "1,2,3,4", "MOC_REPEATS": "3"}, clear=True):
            self.assertEqual(manifest()["benchmark_training_steps"], 7920)
        with patch.dict(os.environ, {"MOC_FAILURE_COUNTS": "1,1"}, clear=True):
            with self.assertRaises(ValueError):
                manifest()

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch needed for tensor cache test")
    def test_cpu_cache_survives_real_producer_exit_and_discards_failed_rank(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            server = CacheServer(str(Path(tmp) / "cache.sock"), [0, 1])
            try:
                code = """import sys, torch
from moc_e2e_launch import cache_request
x = torch.arange(10, dtype=torch.float32)
cache_request(sys.argv[1], {'op':'put', 'rank':0, 'payload':{'x':x[1::2]}})
x.zero_()
"""
                subprocess.run([sys.executable, "-c", code, server.address], cwd=Path(__file__).resolve().parents[2] / "examples/moc_system", check=True, timeout=60)
                result = cache_request(server.address, {"op": "get", "rank": 0})
                self.assertTrue(torch.equal(result["x"], torch.tensor([1., 3., 5., 7., 9.])))
                server.discard_failed([0])
                with self.assertRaises(RuntimeError):
                    cache_request(server.address, {"op": "get", "rank": 0})
                with self.assertRaises(RuntimeError):
                    cache_request(server.address, {"op": "put", "rank": 3, "payload": {}})
            finally:
                server.close()

    def fixture(self, directory):
        cfg = {"nnodes": 1, "per_node": 2, "step": 10, "checkpoint_step": 12, "failure_step": 14,
               "endpoint": 16, "scope": "test fixture", "benchmark_training_steps": 24,
               "failure_counts": [1], "excluded": [], "clock": "test", "checkpoint_interval": 2,
               "cache_policy": "test", "cadence_note": "test"}
        jobs = [{"id": arm, "arm": arm, "repeat": 0, "failed_ranks": [0]}
                for arm in ("full_sync_native", "pec_sync", "pec_2level_async")]
        cfg["jobs"] = jobs
        atomic_json(directory / "plan.json", cfg)
        for job in jobs:
            root = directory / job["id"]
            for phase in ("prefix", "resume"):
                (root / phase).mkdir(parents=True)
                for rank in range(2):
                    steps = range(11, 15) if phase == "prefix" else range(13, 17)
                    rows = [{"step": step, "loss": {"lm loss": 2.0}, "elapsed_s": 1.0,
                             "consumed_samples_before": (step - 1) * 64, "learning_rates_before": [0.001],
                             "learning_rates_after": [0.001]} for step in steps]
                    batches = [{"step": step, "tokens": str(step), "labels": str(step)} for step in steps]
                    atomic_json(root / phase / f"training.rank_{rank:03d}.json",
                                {"rank": rank, "pid": rank + (10 if phase == "prefix" else 100), "rows": rows, "batches": batches})
                    if phase == "prefix":
                        atomic_json(root / phase / f"checkpoint.rank_{rank:03d}.json",
                                    {"checkpoint_blocking_s": 2.0, "written_bytes_local": 100, "failure_boundary_drain_s": 0.0,
                                     "persisted_expert_units": 1, "snapshotted_expert_units": 2})
                    else:
                        source = "native_full_checkpoint" if job["arm"] == "full_sync_native" else (
                            "supervisor_cpu" if job["arm"] == "pec_2level_async" and rank == 1 else "durable_partial_checkpoint")
                        atomic_json(root / phase / f"restore.rank_{rank:03d}.json",
                                    {"rank": rank, "checkpoint_step": 12, "source": source, "verified_fields": 20,
                                     "expert_units": 2 if source == "supervisor_cpu" else 1})
            atomic_json(root / "fault.json", {"monotonic": 10.0, "step": 14, "failed_ranks": [0]})
            atomic_json(root / "prefix" / "first_step_start.json", {"monotonic": 1.0})
            for step, timestamp in [(13, 20.0), (14, 21.0), (15, 22.0), (16, 23.0)]:
                atomic_json(root / "resume" / f"commit_{step}.json", {"monotonic": timestamp, "step": step})
        return cfg, jobs

    def test_summary_uses_direct_endpoint_not_sum_of_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, jobs = self.fixture(root)
            result = validate_job(cfg, jobs[0], root / jobs[0]["id"])
            self.assertEqual(result["window_e2e_s"], 22.0)
            self.assertEqual(result["recovery_caught_up_s"], 11.0)
            self.assertEqual(result["replayed_steps"], 2)
            summarize(root)
            self.assertTrue((root / "COMPLETE.json").exists())
            self.assertEqual(len((root / "e2e_results.csv").read_text().splitlines()), 4)

    def test_summary_rejects_missing_records_bad_replay_and_wrong_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, jobs = self.fixture(Path(tmp))
            job = jobs[-1]
            root = Path(tmp) / job["id"]
            file = root / "resume" / "training.rank_000.json"
            original = json.loads(file.read_text())
            damaged = copy.deepcopy(original)
            damaged["batches"][0]["tokens"] = "different real sample"
            atomic_json(file, damaged)
            with self.assertRaisesRegex(ValueError, "tokens/labels differ"):
                validate_job(cfg, job, root)
            atomic_json(file, original)
            bad_schedule = copy.deepcopy(original)
            bad_schedule["rows"][0]["learning_rates_before"] = [0.1]
            atomic_json(file, bad_schedule)
            with self.assertRaisesRegex(ValueError, "LR schedule"):
                validate_job(cfg, job, root)
            atomic_json(file, original)
            damaged["batches"] = original["batches"]
            damaged["rows"].pop()
            atomic_json(file, damaged)
            with self.assertRaisesRegex(ValueError, "commits"):
                validate_job(cfg, job, root)
            atomic_json(file, original)
            file = root / "resume" / "restore.rank_000.json"
            damaged = json.loads(file.read_text())
            damaged["source"] = "supervisor_cpu"
            atomic_json(file, damaged)
            with self.assertRaisesRegex(ValueError, "recovery source"):
                validate_job(cfg, job, root)
            file.unlink()
            with self.assertRaises(FileNotFoundError):
                validate_job(cfg, job, root)


if __name__ == "__main__":
    unittest.main()
