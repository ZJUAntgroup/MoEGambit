"""Small CPU checks for physical PEC state coverage and strict summaries."""
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from moc_timing_plan import Plan, dense_owners, selected_experts
from moc_timing_summarize import summarize

try:
    import torch
except ImportError:
    torch = None


class PlanTests(unittest.TestCase):
    def test_sequential_cover_balance_and_two_level_subset(self):
        for n in (8, 64, 128, 256):
            for k in (1, 2, 8):
                for layer in range(6):
                    cycle = [selected_experts(n, 8, k, r, layer) for r in range(n // k)]
                    self.assertEqual(set.union(*cycle), set(range(n)))
                    self.assertEqual(sum(map(len, cycle)), n)
        for layer in range(48):
            for r in range(8):
                snap = selected_experts(128, 8, 32, r, layer, stride=16)
                disk = selected_experts(128, 8, 16, r, layer, stride=16)
                self.assertTrue(disk <= snap)
                for ep in range(8):
                    self.assertEqual(sum(e // 16 == ep for e in disk), 2)

    def test_plan_rejects_overlap_and_unbounded_steps(self):
        with patch.dict(os.environ, {}, clear=True):
            p = Plan.from_env()
        for bad in (replace(p, scratch=p.base_dir + "/tmp"),
                    replace(p, result_dir="/zds/results"),
                    replace(p, step=4050), replace(p, smoke_steps=500),
                    replace(p, snapshot_k=8, persist_k=16)):
            with self.assertRaises(ValueError):
                bad.validate()
        self.assertEqual(p.manifest()["benchmark_training_steps"], 0)

    def test_owner_assignment_deterministic_and_complete(self):
        sizes = {"a": 9, "b": 7, "c": 3, "d": 1}
        self.assertEqual(dense_owners(sizes, 2), dense_owners(dict(reversed(list(sizes.items()))), 2))
        owners = dense_owners(sizes, 2)
        self.assertEqual([sum(v for u, v in sizes.items() if owners[u] == i) for i in range(2)], [10, 10])

    def test_dry_run_has_no_save_and_keeps_bias_mbs_schedule(self):
        script = Path(__file__).resolve().parents[2] / "examples/moc_system/run_moc_timing_short.sh"
        env = {**os.environ, "DRY_RUN": "1", "MASTER_ADDR": "127.0.0.1", "NODE_RANK": "0",
               "BSR_FAULT_INJECT_STEP": "5", "FSE_QUALITY_BRANCH_RANKS": "0"}
        p = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        command = p.stdout.split("[moc-timing] command:")[1]
        self.assertIn("--micro-batch-size 1", command)
        self.assertIn("--moe-router-load-balancing-type none", command)
        self.assertIn("--moe-router-enable-expert-bias", command)
        self.assertIn("--train-iters 10000", command)
        self.assertIn("--ckpt-step 4000", command)
        self.assertIn("--exit-interval 4005", command)
        self.assertNotIn("--save ", command)
        self.assertNotIn("--moe-moegambit-fault-injection", command)


class SummaryTests(unittest.TestCase):
    def fixture(self, root):
        plan = {"nnodes": 1, "per_node": 2, "step": 4000, "repeats": 1, "smoke_steps": 1,
                "failure_counts": [1], "arms": [{"name": "fixture"}], "scope": "fixture"}
        (root / "plan.json").write_text(json.dumps(plan))
        (root / "ranks").mkdir()
        for rank in (0, 1):
            rows = [{"rank": rank, "arm": "fixture", "repeat": 0, "event": "snapshot",
                     "snapshot_local_s": 1 + rank, "enqueue_local_s": 1 + rank,
                     "backpressure_local_s": 0, "snapshot_tensor_bytes_local": 100},
                    {"rank": rank, "arm": "fixture", "repeat": 0, "event": "persist",
                     "persist_local_s": 3 + rank, "written_bytes_local": 110},
                    {"rank": rank, "arm": "fixture", "repeat": 0, "event": "restore",
                     "failed_count": 1, "failed_ranks": [0], "verified_entries": 10,
                     "dense_replicas_verified": True, "source": "storage" if rank == 0 else "memory",
                     "storage_read_bytes_local": 110 if rank == 0 else 0, "restore_global_max_s": 5}]
            (root / "ranks" / f"rank_{rank:03d}.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
            (root / "ranks" / f"smoke.rank_{rank:03d}.json").write_text(json.dumps(
                {"rank": rank, "count": 1, "rows": [{"step": 4001}]}))

    def test_bottleneck_and_complete_output(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root)
            result = summarize(root)
            self.assertEqual(result["summary"][0]["snapshot_median_s"], 2)
            self.assertEqual(result["summary"][0]["persist_median_s"], 4)
            self.assertEqual(result["summary"][0]["written_bytes_median"], 220)
            self.assertTrue((root / "COMPLETE.json").exists())

    def test_missing_duplicate_and_unverified_samples_are_rejected(self):
        for kind in ("missing", "duplicate", "unverified", "wrong_smoke"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                self.fixture(root)
                path = root / "ranks" / "rank_000.jsonl"
                rows = [json.loads(l) for l in path.read_text().splitlines()]
                if kind == "missing": rows.pop()
                if kind == "duplicate": rows.append(copy.deepcopy(rows[-1]))
                if kind == "unverified": rows[-1]["verified_entries"] = 0
                path.write_text("\n".join(map(json.dumps, rows)) + "\n")
                if kind == "wrong_smoke":
                    (root / "ranks" / "smoke.rank_000.json").write_text(json.dumps(
                        {"rank": 0, "count": 1, "rows": [{"step": 4002}]}))
                with self.assertRaises(ValueError): summarize(root)
                self.assertFalse((root / "COMPLETE.json").exists())


@unittest.skipUnless(torch, "torch needed for tensor/physical-I/O tests")
class PhysicalTests(unittest.TestCase):
    def test_real_partial_write_roundtrip_and_missing_unit(self):
        from moc_timing_runtime import write_units, read_units, LiveState, cpu_copy
        source = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        snap = cpu_copy(source[:, :2])
        source.zero_()
        self.assertTrue(torch.equal(snap, torch.tensor([[0, 1], [4, 5], [8, 9]], dtype=torch.float32)))
        with tempfile.TemporaryDirectory() as d:
            payload = {"expert/0/1": {"weight": snap, "master": snap.clone(),
                                       "exp_avg": snap.clone(), "exp_avg_sq": snap.clone()}}
            out = write_units(Path(d), payload, True)
            self.assertEqual(len(list(Path(d).glob("*.pt"))), 1)
            restored, size = read_units(out["paths"])
            self.assertEqual(size, out["written_bytes_local"])
            self.assertTrue(torch.equal(restored["expert/0/1"]["master"], snap))
            live = LiveState.__new__(LiveState)
            live.owned = {"expert/0/1": list(payload["expert/0/1"])}
            with self.assertRaises(ValueError): live.restore_units({})

    def test_actual_bf16_master_adam_mapping_and_restore(self):
        from moc_timing_runtime import LiveState
        class Fixture(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.router = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
                self.decoder = torch.nn.Module()
                self.decoder.layers = torch.nn.ModuleList([torch.nn.Module()])
                mlp = self.decoder.layers[0].mlp = torch.nn.Module()
                mlp.experts = torch.nn.Module()
                mlp.experts.local_experts = torch.nn.ModuleList([
                    torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)])
                self.register_buffer("expert_bias", torch.ones(2))
        model = Fixture()
        models = list(model.parameters())
        masters = [torch.nn.Parameter(p.detach().float()) for p in models]
        adam = torch.optim.Adam(masters)
        for p in masters: p.grad = torch.ones_like(p)
        adam.step()
        with torch.no_grad():
            for p, q in zip(models, masters): p.copy_(q)
        wrapper = types.SimpleNamespace(optimizer=adam, float16_groups=[models], fp32_from_float16_groups=[masters])
        mpu = types.SimpleNamespace(
            get_tensor_model_parallel_world_size=lambda: 1,
            get_expert_data_parallel_world_size=lambda: 1,
            get_data_parallel_group=lambda **kw: "dp",
            get_expert_model_parallel_world_size=lambda: 8,
            get_expert_model_parallel_rank=lambda: 0,
            get_pipeline_model_parallel_rank=lambda: 0)
        modules = {"megatron": types.ModuleType("megatron"),
                   "megatron.training": types.ModuleType("megatron.training"),
                   "megatron.training.utils": types.SimpleNamespace(unwrap_model=lambda m: m),
                   "megatron.core": types.SimpleNamespace(parallel_state=mpu)}
        def gather(out, value, group=None): out[:] = [value]
        with patch.dict(sys.modules, modules), \
                patch("torch.distributed.get_rank", return_value=0), \
                patch("torch.distributed.get_process_group_ranks", return_value=[0]), \
                patch("torch.distributed.all_gather_object", side_effect=gather), \
                patch("torch.distributed.broadcast", side_effect=lambda *a, **k: None), \
                patch("torch.cuda.current_device", return_value="cpu"):
            live = LiveState([model], wrapper, types.SimpleNamespace(
                num_experts=8, num_layers=8, pipeline_model_parallel_size=8))
            expected = live.snapshot(8, 0)
            live.clear_tensor_state()
            self.assertTrue(all(torch.count_nonzero(p) == 0 for p in models))
            live.restore_units(expected)
            checked = live.verify(expected)
            self.assertGreater(checked, 10)
            self.assertTrue(all(torch.count_nonzero(p) > 0 for p in models))
            for p in masters:
                self.assertTrue(torch.equal(adam.state[p]["exp_avg"], torch.full_like(p, .1)))
            # Missing moments are a hard error, not a weights-only fallback.
            adam.state[masters[0]].pop("exp_avg_sq")
            with self.assertRaises(ValueError):
                LiveState([model], wrapper, types.SimpleNamespace(
                    num_experts=8, num_layers=8, pipeline_model_parallel_size=8))



class TimingDataTests(unittest.TestCase):
    def fixture(self, eval_iters=0, full_validation=False, fail=False):
        from types import SimpleNamespace
        from moc_timing_data import install_timing_data_hook
        settings = SimpleNamespace(eval_iters=eval_iters, full_validation=full_validation,
                                   consumed_valid_samples=12800, consumed_train_samples=256000,
                                   iteration=4000)
        seen = []
        def original(provider):
            seen.append((settings.consumed_valid_samples, settings.consumed_train_samples,
                         settings.iteration, provider))
            if fail:
                raise RuntimeError("loader failed")
            return "loaders"
        training = SimpleNamespace(build_train_valid_test_data_loaders=original,
                                   print_rank_0=lambda message: None)
        install_timing_data_hook(training, lambda: settings)
        return training, settings, seen

    def test_disabled_validation_preserves_checkpoint_and_train_cursor(self):
        for counts in (0, [0, 0]):
            with self.subTest(counts=counts):
                training, settings, seen = self.fixture(eval_iters=counts)
                self.assertEqual(training.build_train_valid_test_data_loaders("provider"), "loaders")
                self.assertEqual(seen, [(0, 256000, 4000, "provider")])
                self.assertEqual(settings.consumed_valid_samples, 12800)

    def test_enabled_validation_is_unmodified(self):
        for counts, full in ((20, False), ([0, 20], False), (0, True)):
            with self.subTest(counts=counts, full=full):
                training, settings, seen = self.fixture(counts, full)
                training.build_train_valid_test_data_loaders("provider")
                self.assertEqual(seen[0][0], 12800)
                self.assertEqual(settings.consumed_valid_samples, 12800)

    def test_counter_is_restored_on_loader_failure(self):
        training, settings, seen = self.fixture(fail=True)
        with self.assertRaisesRegex(RuntimeError, "loader failed"):
            training.build_train_valid_test_data_loaders("provider")
        self.assertEqual(settings.consumed_valid_samples, 12800)
        self.assertEqual(settings.consumed_train_samples, 256000)

if __name__ == "__main__":
    unittest.main()
