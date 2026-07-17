# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for MOEGAMBIT-MoE RankExposureTracker.

Tests cover:
1. Same rank multiple records → stale_iters correctly accumulated
2. Different ranks independently tracked
3. Events outside window not counted
4. Checkpoint restarts should NOT be recorded
5. Stale exposure ratio calculation
6. get_all_rank_exposures returns dict with non-zero ranks
7. Negative gap ignored
8. Persistence: to_state_dict / from_state_dict round-trip
9. prune() removes old events
10. max_events enforcement
11. reset() clears all events
12. Global singleton lifecycle
"""

import unittest

from megatron.core.transformer.moe.rank_exposure_tracker import (
    HybridRecoveryEvent,
    RankExposureTracker,
    get_rank_exposure_tracker,
    set_rank_exposure_tracker,
    reset_rank_exposure_tracker,
)


# =====================================================================
# Test: Same rank multiple records → stale_iters correctly accumulated
# =====================================================================


class TestStaleItersAccumulation(unittest.TestCase):
    """Same rank multiple records → stale_iters correctly accumulated."""

    def setUp(self):
        self.tracker = RankExposureTracker()

    def test_single_event(self):
        """Single event: stale_iters == gap."""
        self.tracker.record_hybrid_recovery(step=100, rank=3, gap=30)
        stale = self.tracker.get_rank_stale_iters(
            rank=3, current_step=200, window_steps=1000,
        )
        self.assertEqual(stale, 30)

    def test_two_events_same_rank(self):
        """Two events on same rank: stale_iters == sum of gaps."""
        self.tracker.record_hybrid_recovery(step=100, rank=3, gap=30)
        self.tracker.record_hybrid_recovery(step=200, rank=3, gap=50)
        stale = self.tracker.get_rank_stale_iters(
            rank=3, current_step=300, window_steps=1000,
        )
        self.assertEqual(stale, 80)

    def test_three_events_same_rank(self):
        """Three events on same rank: cumulative stale_iters."""
        self.tracker.record_hybrid_recovery(step=100, rank=5, gap=10)
        self.tracker.record_hybrid_recovery(step=200, rank=5, gap=20)
        self.tracker.record_hybrid_recovery(step=300, rank=5, gap=30)
        stale = self.tracker.get_rank_stale_iters(
            rank=5, current_step=400, window_steps=1000,
        )
        self.assertEqual(stale, 60)

    def test_zero_gap_event(self):
        """Zero gap event adds nothing to stale_iters."""
        self.tracker.record_hybrid_recovery(step=100, rank=3, gap=0)
        stale = self.tracker.get_rank_stale_iters(
            rank=3, current_step=200, window_steps=1000,
        )
        self.assertEqual(stale, 0)


# =====================================================================
# Test: Different ranks independently tracked
# =====================================================================


class TestDifferentRanksIndependent(unittest.TestCase):
    """Different ranks are independently tracked."""

    def setUp(self):
        self.tracker = RankExposureTracker()

    def test_different_ranks_no_crosstalk(self):
        """Events on rank 3 don't affect rank 5."""
        self.tracker.record_hybrid_recovery(step=100, rank=3, gap=30)
        self.tracker.record_hybrid_recovery(step=200, rank=5, gap=50)
        # rank 3 only has gap=30
        stale_3 = self.tracker.get_rank_stale_iters(
            rank=3, current_step=300, window_steps=1000,
        )
        # rank 5 only has gap=50
        stale_5 = self.tracker.get_rank_stale_iters(
            rank=5, current_step=300, window_steps=1000,
        )
        self.assertEqual(stale_3, 30)
        self.assertEqual(stale_5, 50)

    def test_untracked_rank_returns_zero(self):
        """Querying a rank with no events returns 0."""
        self.tracker.record_hybrid_recovery(step=100, rank=3, gap=30)
        stale = self.tracker.get_rank_stale_iters(
            rank=7, current_step=200, window_steps=1000,
        )
        self.assertEqual(stale, 0)

    def test_multiple_events_per_rank_independent(self):
        """Multiple events on different ranks are summed independently."""
        self.tracker.record_hybrid_recovery(step=100, rank=1, gap=10)
        self.tracker.record_hybrid_recovery(step=150, rank=1, gap=20)
        self.tracker.record_hybrid_recovery(step=200, rank=2, gap=30)
        self.tracker.record_hybrid_recovery(step=250, rank=2, gap=40)

        stale_1 = self.tracker.get_rank_stale_iters(
            rank=1, current_step=300, window_steps=1000,
        )
        stale_2 = self.tracker.get_rank_stale_iters(
            rank=2, current_step=300, window_steps=1000,
        )
        self.assertEqual(stale_1, 30)  # 10 + 20
        self.assertEqual(stale_2, 70)  # 30 + 40


# =====================================================================
# Test: Events outside window not counted
# =====================================================================


class TestWindowExclusion(unittest.TestCase):
    """Events outside the sliding window are not counted."""

    def setUp(self):
        self.tracker = RankExposureTracker()

    def test_old_events_excluded(self):
        """Events before the window_start are not counted."""
        self.tracker.record_hybrid_recovery(step=100, rank=3, gap=30)
        # Window: [900, 1000], event is at step 100 → excluded
        stale = self.tracker.get_rank_stale_iters(
            rank=3, current_step=1000, window_steps=100,
        )
        self.assertEqual(stale, 0)

    def test_mixed_in_and_out_of_window(self):
        """Old events excluded, recent events included."""
        self.tracker.record_hybrid_recovery(step=100, rank=3, gap=30)  # outside
        self.tracker.record_hybrid_recovery(step=950, rank=3, gap=50)  # inside
        # Window: [900, 1000]
        stale = self.tracker.get_rank_stale_iters(
            rank=3, current_step=1000, window_steps=100,
        )
        self.assertEqual(stale, 50)

    def test_exactly_at_window_boundary(self):
        """Event at exactly window_start is included."""
        self.tracker.record_hybrid_recovery(step=900, rank=3, gap=30)
        # Window: [900, 1000]
        stale = self.tracker.get_rank_stale_iters(
            rank=3, current_step=1000, window_steps=100,
        )
        self.assertEqual(stale, 30)

    def test_one_step_before_window(self):
        """Event one step before window_start is excluded."""
        self.tracker.record_hybrid_recovery(step=899, rank=3, gap=30)
        # Window: [900, 1000]
        stale = self.tracker.get_rank_stale_iters(
            rank=3, current_step=1000, window_steps=100,
        )
        self.assertEqual(stale, 0)

    def test_window_zero_returns_zero(self):
        """window_steps=0 returns 0 stale iters."""
        self.tracker.record_hybrid_recovery(step=500, rank=3, gap=30)
        stale = self.tracker.get_rank_stale_iters(
            rank=3, current_step=500, window_steps=0,
        )
        self.assertEqual(stale, 0)


# =====================================================================
# Test: Checkpoint restarts should NOT be recorded
# =====================================================================


class TestCheckpointRestartNotRecorded(unittest.TestCase):
    """Checkpoint restarts should NOT be recorded by tracker.

    The tracker only has record_hybrid_recovery().  There is no
    record_checkpoint_restart() method.  This test verifies that the
    API does not provide a way to record restarts, and simulates the
    expected usage pattern.
    """

    def test_no_restart_recording_method(self):
        """RankExposureTracker has no record_checkpoint_restart method."""
        self.assertFalse(
            hasattr(RankExposureTracker, "record_checkpoint_restart"),
            "Tracker should not have a record_checkpoint_restart method",
        )

    def test_only_hybrid_events_affect_exposure(self):
        """After only hybrid recoveries, exposure reflects hybrid gaps.

        Checkpoint restart is implied to not affect exposure because
        the tracker simply has no method to record it.
        """
        tracker = RankExposureTracker()
        # Simulate two faults: both are hybrid recovery
        tracker.record_hybrid_recovery(step=100, rank=3, gap=30)
        tracker.record_hybrid_recovery(step=500, rank=3, gap=50)
        # (A checkpoint restart at step 300 is NOT recorded)
        exposure = tracker.get_rank_stale_exposure(
            rank=3, current_step=600, window_steps=1000,
        )
        # Only 30+50 = 80 stale iters from hybrid recoveries
        self.assertAlmostEqual(exposure, 80 / 1000)

    def test_hybrid_recovery_events_marked_correctly(self):
        """HybridRecoveryEvent defaults to recovery_path='hybrid_recovery'."""
        event = HybridRecoveryEvent(step=100, rank=3, gap=30)
        self.assertEqual(event.recovery_path, "hybrid_recovery")


# =====================================================================
# Test: Stale exposure ratio calculation
# =====================================================================


class TestStaleExposureRatio(unittest.TestCase):
    """Stale exposure ratio = stale_iters / window_steps."""

    def setUp(self):
        self.tracker = RankExposureTracker()

    def test_simple_exposure(self):
        """Simple exposure ratio calculation."""
        self.tracker.record_hybrid_recovery(step=500, rank=3, gap=100)
        # window_steps=1000 → exposure = 100/1000 = 0.1
        exposure = self.tracker.get_rank_stale_exposure(
            rank=3, current_step=600, window_steps=1000,
        )
        self.assertAlmostEqual(exposure, 0.1)

    def test_accurate_exposure_multiple_events(self):
        """Exposure from multiple events on same rank."""
        self.tracker.record_hybrid_recovery(step=500, rank=3, gap=100)
        self.tracker.record_hybrid_recovery(step=550, rank=3, gap=200)
        # total stale = 300, window = 1000 → 0.3
        exposure = self.tracker.get_rank_stale_exposure(
            rank=3, current_step=600, window_steps=1000,
        )
        self.assertAlmostEqual(exposure, 0.3)

    def test_zero_window_returns_zero(self):
        """window_steps=0 returns exposure 0."""
        self.tracker.record_hybrid_recovery(step=500, rank=3, gap=100)
        exposure = self.tracker.get_rank_stale_exposure(
            rank=3, current_step=600, window_steps=0,
        )
        self.assertEqual(exposure, 0.0)

    def test_no_events_returns_zero(self):
        """No events for a rank returns exposure 0."""
        exposure = self.tracker.get_rank_stale_exposure(
            rank=3, current_step=600, window_steps=1000,
        )
        self.assertEqual(exposure, 0.0)


# =====================================================================
# Test: get_all_rank_exposures
# =====================================================================


class TestGetAllRankExposures(unittest.TestCase):
    """get_all_rank_exposures returns dict with only non-zero ranks."""

    def setUp(self):
        self.tracker = RankExposureTracker()

    def test_multiple_ranks(self):
        """Multiple ranks with different exposures."""
        self.tracker.record_hybrid_recovery(step=500, rank=1, gap=100)
        self.tracker.record_hybrid_recovery(step=500, rank=2, gap=200)
        self.tracker.record_hybrid_recovery(step=500, rank=3, gap=300)
        all_exp = self.tracker.get_all_rank_exposures(
            current_step=600, window_steps=1000,
        )
        self.assertAlmostEqual(all_exp[1], 0.1)
        self.assertAlmostEqual(all_exp[2], 0.2)
        self.assertAlmostEqual(all_exp[3], 0.3)
        self.assertNotIn(4, all_exp)  # untracked rank not in dict

    def test_empty_tracker(self):
        """Empty tracker returns empty dict."""
        all_exp = self.tracker.get_all_rank_exposures(
            current_step=600, window_steps=1000,
        )
        self.assertEqual(all_exp, {})

    def test_all_outside_window(self):
        """All events outside window returns empty dict."""
        self.tracker.record_hybrid_recovery(step=100, rank=1, gap=100)
        all_exp = self.tracker.get_all_rank_exposures(
            current_step=600, window_steps=100,
        )
        self.assertEqual(all_exp, {})


# =====================================================================
# Test: Negative gap ignored
# =====================================================================


class TestNegativeGapIgnored(unittest.TestCase):
    """Negative gap values should be ignored."""

    def setUp(self):
        self.tracker = RankExposureTracker()

    def test_negative_gap_not_recorded(self):
        """Negative gap events are silently ignored."""
        self.tracker.record_hybrid_recovery(step=100, rank=3, gap=-5)
        stale = self.tracker.get_rank_stale_iters(
            rank=3, current_step=200, window_steps=1000,
        )
        self.assertEqual(stale, 0)

    def test_mixed_negative_and_positive(self):
        """Negative gap ignored; positive gap recorded."""
        self.tracker.record_hybrid_recovery(step=100, rank=3, gap=-5)
        self.tracker.record_hybrid_recovery(step=150, rank=3, gap=30)
        stale = self.tracker.get_rank_stale_iters(
            rank=3, current_step=200, window_steps=1000,
        )
        self.assertEqual(stale, 30)

    def test_negative_gap_event_count(self):
        """Negative gap does not increment event count."""
        self.tracker.record_hybrid_recovery(step=100, rank=3, gap=-5)
        self.assertEqual(self.tracker.get_event_count(), 0)


# =====================================================================
# Test: Persistence round-trip
# =====================================================================


class TestPersistence(unittest.TestCase):
    """to_state_dict / from_state_dict round-trip."""

    def setUp(self):
        self.tracker = RankExposureTracker(max_events=500)

    def test_round_trip_empty(self):
        """Empty tracker round-trip preserves state."""
        state = self.tracker.to_state_dict()
        restored = RankExposureTracker.from_state_dict(state)
        self.assertEqual(restored.get_event_count(), 0)
        self.assertEqual(restored._max_events, 500)

    def test_round_trip_with_events(self):
        """Tracker with events round-trip preserves all data."""
        self.tracker.record_hybrid_recovery(step=100, rank=3, gap=30)
        self.tracker.record_hybrid_recovery(step=200, rank=5, gap=50)
        state = self.tracker.to_state_dict()
        restored = RankExposureTracker.from_state_dict(state)
        self.assertEqual(restored.get_event_count(), 2)
        self.assertEqual(restored._max_events, 500)
        # Verify restored data is queryable
        stale_3 = restored.get_rank_stale_iters(
            rank=3, current_step=300, window_steps=1000,
        )
        stale_5 = restored.get_rank_stale_iters(
            rank=5, current_step=300, window_steps=1000,
        )
        self.assertEqual(stale_3, 30)
        self.assertEqual(stale_5, 50)

    def test_state_dict_structure(self):
        """to_state_dict returns correct structure."""
        self.tracker.record_hybrid_recovery(step=100, rank=3, gap=30)
        state = self.tracker.to_state_dict()
        self.assertEqual(state["version"], 2)
        self.assertEqual(state["max_events"], 500)
        self.assertIsInstance(state["events"], list)
        self.assertEqual(len(state["events"]), 1)
        ev = state["events"][0]
        self.assertEqual(ev["step"], 100)
        self.assertEqual(ev["rank"], 3)
        self.assertEqual(ev["gap"], 30)
        self.assertEqual(ev["recovery_path"], "hybrid_recovery")


# =====================================================================
# Test: Pruning
# =====================================================================


class TestPrune(unittest.TestCase):
    """prune() removes events outside the sliding window."""

    def setUp(self):
        self.tracker = RankExposureTracker()

    def test_prune_removes_old_events(self):
        """prune removes events outside the window."""
        self.tracker.record_hybrid_recovery(step=100, rank=3, gap=30)
        self.tracker.record_hybrid_recovery(step=950, rank=3, gap=50)
        pruned = self.tracker.prune(current_step=1000, window_steps=100)
        self.assertEqual(pruned, 1)  # step=100 removed
        self.assertEqual(self.tracker.get_event_count(), 1)

    def test_prune_keeps_recent_events(self):
        """prune keeps events within the window."""
        self.tracker.record_hybrid_recovery(step=950, rank=3, gap=50)
        pruned = self.tracker.prune(current_step=1000, window_steps=100)
        self.assertEqual(pruned, 0)
        self.assertEqual(self.tracker.get_event_count(), 1)

    def test_prune_zero_window(self):
        """prune with window_steps=0 does nothing."""
        self.tracker.record_hybrid_recovery(step=100, rank=3, gap=30)
        pruned = self.tracker.prune(current_step=200, window_steps=0)
        self.assertEqual(pruned, 0)


# =====================================================================
# Test: max_events enforcement
# =====================================================================


class TestMaxEvents(unittest.TestCase):
    """max_events limits the number of stored events."""

    def test_max_events_enforced(self):
        """Events beyond max_events are dropped (oldest first)."""
        tracker = RankExposureTracker(max_events=3)
        tracker.record_hybrid_recovery(step=100, rank=1, gap=10)
        tracker.record_hybrid_recovery(step=200, rank=2, gap=20)
        tracker.record_hybrid_recovery(step=300, rank=3, gap=30)
        # At limit (3 events)
        self.assertEqual(tracker.get_event_count(), 3)
        # Add one more → oldest should be dropped
        tracker.record_hybrid_recovery(step=400, rank=4, gap=40)
        self.assertEqual(tracker.get_event_count(), 3)
        # The first event (step=100, rank=1) should be gone
        stale_1 = tracker.get_rank_stale_iters(
            rank=1, current_step=500, window_steps=1000,
        )
        self.assertEqual(stale_1, 0)
        # The last event (step=400, rank=4) should be present
        stale_4 = tracker.get_rank_stale_iters(
            rank=4, current_step=500, window_steps=1000,
        )
        self.assertEqual(stale_4, 40)

    def test_unlimited_max_events(self):
        """max_events=0 means unlimited."""
        tracker = RankExposureTracker(max_events=0)
        for i in range(20):
            tracker.record_hybrid_recovery(step=i * 10, rank=0, gap=1)
        self.assertEqual(tracker.get_event_count(), 20)


# =====================================================================
# Test: reset()
# =====================================================================


class TestReset(unittest.TestCase):
    """reset() clears all events."""

    def test_reset_clears_events(self):
        tracker = RankExposureTracker()
        tracker.record_hybrid_recovery(step=100, rank=3, gap=30)
        self.assertEqual(tracker.get_event_count(), 1)
        tracker.reset()
        self.assertEqual(tracker.get_event_count(), 0)

    def test_reset_allows_new_events(self):
        tracker = RankExposureTracker()
        tracker.record_hybrid_recovery(step=100, rank=3, gap=30)
        tracker.reset()
        tracker.record_hybrid_recovery(step=200, rank=5, gap=50)
        stale = tracker.get_rank_stale_iters(
            rank=5, current_step=300, window_steps=1000,
        )
        self.assertEqual(stale, 50)


# =====================================================================
# Test: Global singleton lifecycle
# =====================================================================


class TestGlobalSingleton(unittest.TestCase):
    """Tests for global singleton management functions."""

    def tearDown(self):
        reset_rank_exposure_tracker()

    def test_get_creates_default(self):
        """get_rank_exposure_tracker creates a default tracker."""
        tracker = get_rank_exposure_tracker()
        self.assertIsNotNone(tracker)
        self.assertIsInstance(tracker, RankExposureTracker)

    def test_get_returns_same_instance(self):
        """Repeated calls return the same singleton."""
        t1 = get_rank_exposure_tracker()
        t2 = get_rank_exposure_tracker()
        self.assertIs(t1, t2)

    def test_set_replaces_singleton(self):
        """set_rank_exposure_tracker replaces the global tracker."""
        custom = RankExposureTracker(max_events=42)
        set_rank_exposure_tracker(custom)
        tracker = get_rank_exposure_tracker()
        self.assertIs(tracker, custom)
        self.assertEqual(tracker._max_events, 42)

    def test_reset_clears_singleton(self):
        """reset_rank_exposure_tracker sets global to None."""
        t1 = get_rank_exposure_tracker()
        reset_rank_exposure_tracker()
        t2 = get_rank_exposure_tracker()
        self.assertIsNot(t1, t2)

    def test_reset_on_none_is_safe(self):
        """reset_rank_exposure_tracker when no tracker exists is safe."""
        reset_rank_exposure_tracker()  # should not raise


# =====================================================================
# Test: HybridRecoveryEvent serialization
# =====================================================================


class TestHybridRecoveryEventSerialization(unittest.TestCase):
    """Tests for HybridRecoveryEvent to_dict / from_dict."""

    def test_to_dict(self):
        event = HybridRecoveryEvent(step=100, rank=3, gap=30)
        d = event.to_dict()
        self.assertEqual(d["step"], 100)
        self.assertEqual(d["rank"], 3)
        self.assertEqual(d["gap"], 30)
        self.assertEqual(d["recovery_path"], "hybrid_recovery")

    def test_from_dict(self):
        d = {"step": 100, "rank": 3, "gap": 30, "recovery_path": "hybrid_recovery"}
        event = HybridRecoveryEvent.from_dict(d)
        self.assertEqual(event.step, 100)
        self.assertEqual(event.rank, 3)
        self.assertEqual(event.gap, 30)
        self.assertEqual(event.recovery_path, "hybrid_recovery")

    def test_from_dict_missing_recovery_path(self):
        """from_dict defaults recovery_path to 'hybrid_recovery'."""
        d = {"step": 100, "rank": 3, "gap": 30}
        event = HybridRecoveryEvent.from_dict(d)
        self.assertEqual(event.recovery_path, "hybrid_recovery")

    def test_round_trip(self):
        event = HybridRecoveryEvent(step=200, rank=5, gap=50)
        restored = HybridRecoveryEvent.from_dict(event.to_dict())
        self.assertEqual(restored.step, event.step)
        self.assertEqual(restored.rank, event.rank)
        self.assertEqual(restored.gap, event.gap)
        self.assertEqual(restored.recovery_path, event.recovery_path)

    def test_v1_event_defaults_to_one_affected_expert(self):
        event = HybridRecoveryEvent.from_dict({"step": 10, "rank": 2, "gap": 7})
        self.assertEqual(event.num_affected_experts, 1)
        self.assertEqual(event.expert_iteration_debt, 7)


class TestExpertStalenessDensity(unittest.TestCase):

    def test_debt_is_weighted_and_aggregated_across_ranks(self):
        tracker = RankExposureTracker()
        tracker.record_hybrid_recovery(
            step=100, rank=1, gap=10, num_affected_experts=8,
        )
        tracker.record_hybrid_recovery(
            step=200, rank=7, gap=20, num_affected_experts=4,
        )
        self.assertEqual(
            tracker.get_window_expert_iteration_debt(300, 1000),
            160,
        )
        self.assertAlmostEqual(
            tracker.get_expert_staleness_density(300, 1000, 128),
            160 / (128 * 1000),
        )


# =====================================================================
# Test: get_events_for_rank
# =====================================================================


class TestGetEventsForRank(unittest.TestCase):
    """get_events_for_rank returns filtered event list."""

    def setUp(self):
        self.tracker = RankExposureTracker()

    def test_returns_matching_events(self):
        self.tracker.record_hybrid_recovery(step=100, rank=3, gap=10)
        self.tracker.record_hybrid_recovery(step=200, rank=5, gap=20)
        self.tracker.record_hybrid_recovery(step=300, rank=3, gap=30)
        events = self.tracker.get_events_for_rank(
            rank=3, current_step=400, window_steps=1000,
        )
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].gap, 10)
        self.assertEqual(events[1].gap, 30)

    def test_no_matching_events(self):
        self.tracker.record_hybrid_recovery(step=100, rank=3, gap=10)
        events = self.tracker.get_events_for_rank(
            rank=7, current_step=200, window_steps=1000,
        )
        self.assertEqual(len(events), 0)


if __name__ == "__main__":
    unittest.main()
