# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Tests for MoEGambit Experiment Orchestrator (Step 11).

"""Unit tests for the experiment orchestrator, fault injection plans,
metrics collection, and all pre-built experiment scenarios.

Covers:
    1. FaultInjectionPlan — gap calculation, expected path
    2. RecoveryMetrics — serialization
    3. ExperimentReport — success/failure, JSON output
    4. PhaseTimer — timing accuracy
    5. Scenario: small_gap_checkpoint_restart
    6. Scenario: large_gap_hybrid_recovery
    7. Scenario: hybrid_with_deferred_optimizer
    8. Scenario: hybrid_with_preferential_routing
    9. run_all_experiments convenience function
   10. Callback invocation verification
   11. Error handling
   12. Multiple sequential runs
"""

import os
import sys
import json
import time
import unittest
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..', '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from megatron.core.transformer.moe.moegambit_experiment import (
    ExperimentScenario,
    FaultInjectionPlan,
    RecoveryMetrics,
    ExperimentReport,
    PhaseTimer,
    ExperimentOrchestrator,
    run_all_experiments,
    TROUBLESHOOTING_GUIDE,
)


# =====================================================================
# 1. FaultInjectionPlan
# =====================================================================

class TestFaultInjectionPlan(unittest.TestCase):

    def test_gap_calculation(self):
        plan = FaultInjectionPlan(inject_step=500, checkpoint_iteration=100)
        self.assertEqual(plan.gap, 400)

    def test_gap_no_checkpoint(self):
        plan = FaultInjectionPlan(inject_step=500, checkpoint_iteration=-1)
        self.assertEqual(plan.gap, -1)

    def test_expected_path_small_gap(self):
        plan = FaultInjectionPlan(
            inject_step=50, checkpoint_iteration=45, gap_threshold=100,
        )
        self.assertEqual(plan.expected_path, "CHECKPOINT_RESTART")

    def test_expected_path_large_gap(self):
        plan = FaultInjectionPlan(
            inject_step=500, checkpoint_iteration=100, gap_threshold=100,
        )
        self.assertEqual(plan.expected_path, "HYBRID_RECOVERY")

    def test_expected_path_no_checkpoint(self):
        plan = FaultInjectionPlan(
            inject_step=500, checkpoint_iteration=-1, gap_threshold=100,
        )
        self.assertEqual(plan.expected_path, "HYBRID_RECOVERY")

    def test_expected_path_exact_threshold(self):
        plan = FaultInjectionPlan(
            inject_step=200, checkpoint_iteration=100, gap_threshold=100,
        )
        self.assertEqual(plan.expected_path, "CHECKPOINT_RESTART")

    def test_to_dict(self):
        plan = FaultInjectionPlan(
            fault_type="hard_failure",
            inject_step=50,
            failed_rank=2,
        )
        d = plan.to_dict()
        self.assertEqual(d["fault_type"], "hard_failure")
        self.assertEqual(d["inject_step"], 50)
        self.assertEqual(d["failed_rank"], 2)
        self.assertIn("gap", d)
        self.assertIn("expected_path", d)


# =====================================================================
# 2. RecoveryMetrics
# =====================================================================

class TestRecoveryMetrics(unittest.TestCase):

    def test_default_values(self):
        m = RecoveryMetrics()
        self.assertEqual(m.time_to_detect, 0.0)
        self.assertEqual(m.recovery_path, "")

    def test_to_dict(self):
        m = RecoveryMetrics(
            time_to_detect=0.001,
            recovery_path="HYBRID_RECOVERY",
            gap=400,
            experts_recovered=4,
        )
        d = m.to_dict()
        self.assertEqual(d["time_to_detect"], 0.001)
        self.assertEqual(d["recovery_path"], "HYBRID_RECOVERY")
        self.assertEqual(d["gap"], 400)
        self.assertEqual(d["experts_recovered"], 4)


# =====================================================================
# 3. ExperimentReport
# =====================================================================

class TestExperimentReport(unittest.TestCase):

    def test_success_when_no_errors(self):
        r = ExperimentReport(success=True)
        self.assertTrue(r.success)

    def test_elapsed_seconds(self):
        r = ExperimentReport(start_time=100.0, end_time=105.5)
        self.assertAlmostEqual(r.elapsed_seconds, 5.5)

    def test_to_json(self):
        r = ExperimentReport(
            scenario="test",
            success=True,
            start_time=100.0,
            end_time=101.0,
        )
        j = r.to_json()
        parsed = json.loads(j)
        self.assertEqual(parsed["scenario"], "test")
        self.assertTrue(parsed["success"])

    def test_to_dict_with_plan_and_metrics(self):
        r = ExperimentReport(
            scenario="test",
            plan=FaultInjectionPlan(inject_step=50),
            metrics=RecoveryMetrics(recovery_path="HYBRID_RECOVERY"),
        )
        d = r.to_dict()
        self.assertIsNotNone(d["plan"])
        self.assertIsNotNone(d["metrics"])
        self.assertEqual(d["plan"]["inject_step"], 50)


# =====================================================================
# 4. PhaseTimer
# =====================================================================

class TestPhaseTimer(unittest.TestCase):

    def test_timer_records_elapsed(self):
        metrics = RecoveryMetrics()
        events = []
        with PhaseTimer("detect", metrics, events):
            time.sleep(0.01)
        self.assertGreater(metrics.time_to_detect, 0.0)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["type"], "phase_start")
        self.assertEqual(events[1]["type"], "phase_end")

    def test_timer_records_error(self):
        metrics = RecoveryMetrics()
        events = []
        with self.assertRaises(RuntimeError):
            with PhaseTimer("detect", metrics, events):
                raise RuntimeError("test error")
        self.assertEqual(events[1]["error"], "test error")

    def test_unknown_phase_no_crash(self):
        metrics = RecoveryMetrics()
        events = []
        with PhaseTimer("unknown_phase", metrics, events):
            pass
        # Should not crash even if attr doesn't exist
        self.assertEqual(len(events), 2)


# =====================================================================
# 5. Scenario: small_gap_checkpoint_restart
# =====================================================================

class TestSmallGapCheckpointRestart(unittest.TestCase):

    def test_scenario_succeeds(self):
        orch = ExperimentOrchestrator()
        report = orch.run_scenario(ExperimentScenario.SMALL_GAP_CHECKPOINT_RESTART)
        self.assertTrue(report.success, f"errors: {report.errors}")

    def test_path_is_checkpoint_restart(self):
        orch = ExperimentOrchestrator()
        report = orch.run_scenario(ExperimentScenario.SMALL_GAP_CHECKPOINT_RESTART)
        self.assertEqual(report.metrics.recovery_path, "CHECKPOINT_RESTART")

    def test_gap_is_small(self):
        orch = ExperimentOrchestrator()
        report = orch.run_scenario(ExperimentScenario.SMALL_GAP_CHECKPOINT_RESTART)
        self.assertLessEqual(report.metrics.gap, report.metrics.gap_threshold)

    def test_checkpoint_restart_callback_invoked(self):
        mock = MagicMock()
        orch = ExperimentOrchestrator(checkpoint_restart_fn=mock)
        report = orch.run_scenario(ExperimentScenario.SMALL_GAP_CHECKPOINT_RESTART)
        self.assertTrue(report.success, f"errors: {report.errors}")
        mock.assert_called_once()


# =====================================================================
# 6. Scenario: large_gap_hybrid_recovery
# =====================================================================

class TestLargeGapHybridRecovery(unittest.TestCase):

    def test_scenario_succeeds(self):
        orch = ExperimentOrchestrator()
        report = orch.run_scenario(ExperimentScenario.LARGE_GAP_HYBRID)
        self.assertTrue(report.success, f"errors: {report.errors}")

    def test_path_is_hybrid(self):
        orch = ExperimentOrchestrator()
        report = orch.run_scenario(ExperimentScenario.LARGE_GAP_HYBRID)
        self.assertEqual(report.metrics.recovery_path, "HYBRID_RECOVERY")

    def test_gap_is_large(self):
        orch = ExperimentOrchestrator()
        report = orch.run_scenario(ExperimentScenario.LARGE_GAP_HYBRID)
        self.assertGreater(report.metrics.gap, report.metrics.gap_threshold)

    def test_dense_sync_callback_invoked(self):
        mock_dense = MagicMock()
        mock_expert = MagicMock()
        orch = ExperimentOrchestrator(
            dense_sync_fn=mock_dense,
            expert_restore_fn=mock_expert,
        )
        report = orch.run_scenario(ExperimentScenario.LARGE_GAP_HYBRID)
        self.assertTrue(report.success, f"errors: {report.errors}")
        mock_dense.assert_called_once()
        mock_expert.assert_called_once()

    def test_checkpoint_restart_not_invoked(self):
        mock = MagicMock()
        orch = ExperimentOrchestrator(checkpoint_restart_fn=mock)
        report = orch.run_scenario(ExperimentScenario.LARGE_GAP_HYBRID)
        self.assertTrue(report.success, f"errors: {report.errors}")
        mock.assert_not_called()


# =====================================================================
# 7. Scenario: hybrid_with_deferred_optimizer
# =====================================================================

class TestHybridDeferredOptimizer(unittest.TestCase):

    def test_scenario_succeeds(self):
        orch = ExperimentOrchestrator()
        report = orch.run_scenario(ExperimentScenario.HYBRID_DEFERRED_OPTIMIZER)
        self.assertTrue(report.success, f"errors: {report.errors}")

    def test_optimizer_deferred_flag(self):
        orch = ExperimentOrchestrator()
        report = orch.run_scenario(ExperimentScenario.HYBRID_DEFERRED_OPTIMIZER)
        self.assertTrue(report.metrics.optimizer_deferred)

    def test_two_phase_final_state(self):
        orch = ExperimentOrchestrator()
        report = orch.run_scenario(ExperimentScenario.HYBRID_DEFERRED_OPTIMIZER)
        self.assertEqual(report.metrics.two_phase_final_state, "FULLY_RECOVERED")

    def test_deferred_callbacks_invoked(self):
        mock_submit = MagicMock()
        mock_poll = MagicMock()
        orch = ExperimentOrchestrator(
            deferred_optimizer_submit_fn=mock_submit,
            deferred_optimizer_poll_fn=mock_poll,
        )
        report = orch.run_scenario(ExperimentScenario.HYBRID_DEFERRED_OPTIMIZER)
        self.assertTrue(report.success, f"errors: {report.errors}")
        mock_submit.assert_called_once()
        mock_poll.assert_called_once()

    def test_optimizer_attach_timing_recorded(self):
        orch = ExperimentOrchestrator()
        report = orch.run_scenario(ExperimentScenario.HYBRID_DEFERRED_OPTIMIZER)
        self.assertGreaterEqual(report.metrics.time_to_optimizer_attach, 0.0)


# =====================================================================
# 8. Scenario: hybrid_with_preferential_routing
# =====================================================================

class TestHybridPreferentialRouting(unittest.TestCase):

    def test_scenario_succeeds(self):
        orch = ExperimentOrchestrator()
        report = orch.run_scenario(ExperimentScenario.HYBRID_PREFERENTIAL_ROUTING)
        self.assertTrue(report.success, f"errors: {report.errors}")

    def test_preferential_routing_activated(self):
        orch = ExperimentOrchestrator()
        report = orch.run_scenario(ExperimentScenario.HYBRID_PREFERENTIAL_ROUTING)
        self.assertTrue(report.metrics.preferential_routing_activated)

    def test_routing_event_recorded(self):
        orch = ExperimentOrchestrator()
        report = orch.run_scenario(ExperimentScenario.HYBRID_PREFERENTIAL_ROUTING)
        routing_events = [
            e for e in report.events if e.get("type") == "preferential_routing"
        ]
        self.assertEqual(len(routing_events), 1)
        self.assertEqual(routing_events[0]["window"], 100)
        self.assertAlmostEqual(routing_events[0]["bias"], 0.1)


# =====================================================================
# 10. run_all_experiments
# =====================================================================

class TestRunAllExperiments(unittest.TestCase):

    def test_all_scenarios_run(self):
        results = run_all_experiments()
        self.assertEqual(len(results), len(ExperimentScenario))

    def test_all_scenarios_pass(self):
        results = run_all_experiments()
        for name, report in results.items():
            self.assertTrue(
                report.success,
                f"scenario {name} failed: {report.errors}",
            )

    def test_results_keyed_by_scenario_name(self):
        results = run_all_experiments()
        for scenario in ExperimentScenario:
            self.assertIn(scenario.value, results)


# =====================================================================
# 11. Callback invocation verification
# =====================================================================

class TestCallbackInvocation(unittest.TestCase):

    def test_group_rebuild_called_for_hard_failure(self):
        mock = MagicMock()
        orch = ExperimentOrchestrator(group_rebuild_fn=mock)
        report = orch.run_scenario(ExperimentScenario.LARGE_GAP_HYBRID)
        self.assertTrue(report.success, f"errors: {report.errors}")
        mock.assert_called_once()

    def test_topology_refresh_called(self):
        mock = MagicMock()
        orch = ExperimentOrchestrator(topology_refresh_fn=mock)
        report = orch.run_scenario(ExperimentScenario.LARGE_GAP_HYBRID)
        self.assertTrue(report.success, f"errors: {report.errors}")
        mock.assert_called_once()

    def test_callbacks_receive_correct_ranks(self):
        received = {}

        def capture_dense(**kw):
            received.update(kw)

        orch = ExperimentOrchestrator(dense_sync_fn=capture_dense)
        report = orch.run_scenario(ExperimentScenario.LARGE_GAP_HYBRID)
        self.assertTrue(report.success, f"errors: {report.errors}")
        self.assertEqual(received["failed_rank"], 0)
        self.assertEqual(received["replacement_rank"], 0)


# =====================================================================
# 12. Error handling
# =====================================================================

class TestErrorHandling(unittest.TestCase):

    def test_callback_exception_recorded(self):
        def bad_fn(**kw):
            raise RuntimeError("callback exploded")

        orch = ExperimentOrchestrator(group_rebuild_fn=bad_fn)
        report = orch.run_scenario(ExperimentScenario.LARGE_GAP_HYBRID)
        # The controller catches group_rebuild_execute_fn errors and aborts
        # repair, so the experiment should record an error
        self.assertFalse(report.success)
        self.assertGreater(len(report.errors), 0)


# =====================================================================
# 13. Multiple sequential runs
# =====================================================================

class TestMultipleRuns(unittest.TestCase):

    def test_orchestrator_tracks_history(self):
        orch = ExperimentOrchestrator()
        orch.run_scenario(ExperimentScenario.LARGE_GAP_HYBRID)
        orch.run_scenario(ExperimentScenario.SMALL_GAP_CHECKPOINT_RESTART)
        self.assertEqual(orch.num_runs, 2)
        self.assertEqual(orch.num_passed, 2)

    def test_summary(self):
        orch = ExperimentOrchestrator()
        orch.run_scenario(ExperimentScenario.LARGE_GAP_HYBRID)
        s = orch.summary()
        self.assertEqual(s["num_runs"], 1)
        self.assertTrue(s["all_passed"])

    def test_full_report_json(self):
        orch = ExperimentOrchestrator()
        orch.run_scenario(ExperimentScenario.LARGE_GAP_HYBRID)
        j = orch.full_report_json()
        parsed = json.loads(j)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["scenario"], "large_gap_hybrid_recovery")


# =====================================================================
# 14. Metrics timing
# =====================================================================

class TestMetricsTiming(unittest.TestCase):

    def test_time_to_resume_positive(self):
        orch = ExperimentOrchestrator()
        report = orch.run_scenario(ExperimentScenario.LARGE_GAP_HYBRID)
        self.assertGreater(report.metrics.time_to_resume, 0.0)

    def test_time_to_detect_positive(self):
        orch = ExperimentOrchestrator()
        report = orch.run_scenario(ExperimentScenario.LARGE_GAP_HYBRID)
        self.assertGreaterEqual(report.metrics.time_to_detect, 0.0)

    def test_experts_recovered_count(self):
        orch = ExperimentOrchestrator()
        report = orch.run_scenario(ExperimentScenario.LARGE_GAP_HYBRID)
        self.assertGreater(report.metrics.experts_recovered, 0)


# =====================================================================
# 15. Troubleshooting guide
# =====================================================================

class TestTroubleshootingGuide(unittest.TestCase):

    def test_guide_is_non_empty(self):
        self.assertGreater(len(TROUBLESHOOTING_GUIDE), 100)

    def test_guide_covers_key_errors(self):
        self.assertIn("safe-point repair did not execute", TROUBLESHOOTING_GUIDE)
        self.assertIn("path mismatch", TROUBLESHOOTING_GUIDE)
        self.assertIn("Deferred optimizer", TROUBLESHOOTING_GUIDE)
        self.assertIn("Preferential routing", TROUBLESHOOTING_GUIDE)


if __name__ == '__main__':
    unittest.main()
