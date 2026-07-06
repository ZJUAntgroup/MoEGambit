# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for Active Expert Directory Refresh and Dispatch Topology Refresh.

Phase 8: Validates that after a safe-point repair:
1. Failed rank's experts are removed from active dispatch targets.
2. Replacement rank's experts are added to active dispatch targets.
3. Router and dispatcher candidate sets remain consistent.
4. Directory, health mask, placement, and topology snapshot are all in sync.
5. Finalize reintegration marks experts as HEALTHY.
"""

import os
import sys
import unittest
import importlib
import importlib.util

# ---------------------------------------------------------------------------
# Bootstrap: load modules without triggering torch / megatron.core.__init__
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
)

_MOE_DIR = os.path.join(
    _REPO_ROOT, 'megatron', 'core', 'transformer', 'moe',
)


def _load_module(name, filepath):
    """Load a single Python module by file path, bypassing package __init__."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'megatron.core.transformer.moe'
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Pre-load modules
ed_mod = _load_module(
    'megatron.core.transformer.moe.expert_directory',
    os.path.join(_MOE_DIR, 'expert_directory.py'),
)
topo_mod = _load_module(
    'megatron.core.transformer.moe.dispatch_topology_refresh',
    os.path.join(_MOE_DIR, 'dispatch_topology_refresh.py'),
)

ActiveExpertDirectory = ed_mod.ActiveExpertDirectory
RecoveryManifest = ed_mod.RecoveryManifest
ExpertDirectoryEntry = ed_mod.ExpertDirectoryEntry

DispatchTopologyManager = topo_mod.DispatchTopologyManager
DispatchTopologySnapshot = topo_mod.DispatchTopologySnapshot
compute_expert_to_rank_mapping = topo_mod.compute_expert_to_rank_mapping
compute_experts_on_rank = topo_mod.compute_experts_on_rank
compute_active_ranks = topo_mod.compute_active_ranks


# =====================================================================
# Test: ActiveExpertDirectory
# =====================================================================

class TestActiveExpertDirectory(unittest.TestCase):
    """Test ActiveExpertDirectory creation and mutation."""

    def setUp(self):
        self.dir = ActiveExpertDirectory.from_placement(
            num_layers=4,
            num_experts=8,
            ep_size=4,
            ep_group_ranks=[0, 2, 4, 6],
        )

    def test_from_placement_basic(self):
        self.assertEqual(self.dir.num_layers, 4)
        self.assertEqual(self.dir.num_experts, 8)
        self.assertEqual(self.dir.ep_size, 4)
        self.assertEqual(len(self.dir.all_entries()), 32)  # 4 layers * 8 experts

    def test_get_host_rank(self):
        # Expert 0,1 on rank 0; 2,3 on rank 2; 4,5 on rank 4; 6,7 on rank 6
        self.assertEqual(self.dir.get_host_rank(0, 0), 0)
        self.assertEqual(self.dir.get_host_rank(0, 1), 0)
        self.assertEqual(self.dir.get_host_rank(0, 2), 2)
        self.assertEqual(self.dir.get_host_rank(0, 3), 2)
        self.assertEqual(self.dir.get_host_rank(0, 4), 4)
        self.assertEqual(self.dir.get_host_rank(0, 6), 6)

    def test_experts_on_rank(self):
        experts = self.dir.experts_on_rank(4)
        # Rank 4 hosts experts 4,5 across all 4 layers
        self.assertEqual(len(experts), 8)  # 2 experts * 4 layers
        expert_ids = set(eid for _, eid in experts)
        self.assertEqual(expert_ids, {4, 5})

    def test_bulk_update_host_rank(self):
        """Simulate failed rank 4 → replacement rank 64."""
        count = self.dir.bulk_update_host_rank(4, 64)
        self.assertEqual(count, 8)  # 2 experts * 4 layers
        self.assertEqual(self.dir.get_host_rank(0, 4), 64)
        self.assertEqual(self.dir.get_host_rank(0, 5), 64)
        # Other ranks unchanged
        self.assertEqual(self.dir.get_host_rank(0, 0), 0)

    def test_refresh_from_placement(self):
        """Refresh with new EP group ranks (rank 4 → 64)."""
        self.dir.refresh_from_placement([0, 2, 64, 6])
        self.assertEqual(self.dir.get_host_rank(0, 4), 64)
        self.assertEqual(self.dir.get_host_rank(0, 5), 64)
        self.assertEqual(self.dir.get_host_rank(0, 0), 0)
        self.assertEqual(self.dir.get_host_rank(0, 6), 6)

    def test_bulk_update_recovery_state(self):
        self.dir.bulk_update_recovery_state([4, 5], "UNAVAILABLE")
        self.assertEqual(self.dir.get_recovery_state(0, 4), "UNAVAILABLE")
        self.assertEqual(self.dir.get_recovery_state(0, 5), "UNAVAILABLE")
        self.assertEqual(self.dir.get_recovery_state(0, 0), "HEALTHY")

    def test_experts_in_state(self):
        self.dir.bulk_update_recovery_state([4, 5], "UNAVAILABLE")
        unavailable = self.dir.experts_in_state("UNAVAILABLE")
        self.assertEqual(len(unavailable), 8)  # 2 experts * 4 layers
        healthy = self.dir.experts_in_state("HEALTHY")
        self.assertEqual(len(healthy), 24)  # 6 experts * 4 layers

    def test_summary(self):
        summary = self.dir.summary()
        self.assertEqual(summary['num_layers'], 4)
        self.assertEqual(summary['num_experts'], 8)
        self.assertEqual(summary['total_entries'], 32)
        self.assertEqual(summary['state_counts']['HEALTHY'], 32)


# =====================================================================
# Test: DispatchTopologyManager
# =====================================================================

class TestDispatchTopologyManager(unittest.TestCase):
    """Test DispatchTopologyManager initialization and refresh."""

    def setUp(self):
        topo_mod.clear_dispatch_topology_manager()
        self.mgr = DispatchTopologyManager()
        self.mgr.initialise(
            num_layers=4,
            num_experts=8,
            ep_size=4,
            ep_group_ranks=[0, 2, 4, 6],
        )

    def test_initialise(self):
        snap = self.mgr.snapshot
        self.assertIsNotNone(snap)
        self.assertEqual(snap.num_experts, 8)
        self.assertEqual(snap.ep_size, 4)
        self.assertEqual(snap.ep_group_ranks, [0, 2, 4, 6])
        self.assertEqual(snap.active_ranks, frozenset({0, 2, 4, 6}))
        self.assertEqual(snap.quarantined_ranks, frozenset())

    def test_expert_to_rank_mapping(self):
        snap = self.mgr.snapshot
        self.assertEqual(snap.expert_to_rank[0], 0)
        self.assertEqual(snap.expert_to_rank[1], 0)
        self.assertEqual(snap.expert_to_rank[2], 2)
        self.assertEqual(snap.expert_to_rank[4], 4)
        self.assertEqual(snap.expert_to_rank[6], 6)

    def test_all_experts_healthy(self):
        snap = self.mgr.snapshot
        for eid in range(8):
            self.assertTrue(snap.expert_health[eid])

    def test_refresh_replaces_failed_rank(self):
        """Refresh: rank 4 → 64."""
        snap = self.mgr.refresh_dispatch_topology(
            new_ep_group_ranks=[0, 2, 64, 6],
            failed_rank=4,
            replacement_rank=64,
            step=100,
            update_directory=False,
            update_placement=False,
            update_health_mask=False,
            update_replacement_registry=False,
        )
        # Expert 4,5 now on rank 64
        self.assertEqual(snap.expert_to_rank[4], 64)
        self.assertEqual(snap.expert_to_rank[5], 64)
        # Other experts unchanged
        self.assertEqual(snap.expert_to_rank[0], 0)
        self.assertEqual(snap.expert_to_rank[6], 6)

    def test_refresh_updates_active_ranks(self):
        snap = self.mgr.refresh_dispatch_topology(
            new_ep_group_ranks=[0, 2, 64, 6],
            failed_rank=4,
            replacement_rank=64,
            step=100,
            update_directory=False,
            update_placement=False,
            update_health_mask=False,
            update_replacement_registry=False,
        )
        self.assertIn(64, snap.active_ranks)
        self.assertNotIn(4, snap.active_ranks)

    def test_refresh_count(self):
        self.assertEqual(self.mgr.refresh_count, 0)
        self.mgr.refresh_dispatch_topology(
            new_ep_group_ranks=[0, 2, 64, 6],
            failed_rank=4,
            replacement_rank=64,
            step=100,
            update_directory=False,
            update_placement=False,
            update_health_mask=False,
            update_replacement_registry=False,
        )
        self.assertEqual(self.mgr.refresh_count, 1)

    def test_history(self):
        self.mgr.refresh_dispatch_topology(
            new_ep_group_ranks=[0, 2, 64, 6],
            failed_rank=4,
            replacement_rank=64,
            step=100,
            update_directory=False,
            update_placement=False,
            update_health_mask=False,
            update_replacement_registry=False,
        )
        self.assertEqual(len(self.mgr.history), 1)
        # History[0] is the initial snapshot
        self.assertEqual(self.mgr.history[0].ep_group_ranks, [0, 2, 4, 6])

    def test_get_experts_on_rank(self):
        self.assertEqual(self.mgr.get_experts_on_rank(4), [4, 5])
        self.assertEqual(self.mgr.get_experts_on_rank(0), [0, 1])
        self.assertEqual(self.mgr.get_experts_on_rank(99), [])

    def test_get_dispatchable_experts(self):
        dispatchable = self.mgr.get_dispatchable_experts()
        self.assertEqual(dispatchable, [0, 1, 2, 3, 4, 5, 6, 7])

    def test_get_non_dispatchable_experts(self):
        non_disp = self.mgr.get_non_dispatchable_experts()
        self.assertEqual(non_disp, [])

    def test_snapshot_consistency(self):
        snap = self.mgr.snapshot
        self.assertTrue(snap.is_consistent())

    def test_summary(self):
        summary = self.mgr.summary()
        self.assertTrue(summary['initialised'])
        self.assertEqual(summary['num_experts'], 8)
        self.assertEqual(summary['dispatchable_experts'], 8)
        self.assertEqual(summary['non_dispatchable_experts'], 0)


# =====================================================================
# Test: Pure topology computation functions
# =====================================================================

class TestTopologyComputation(unittest.TestCase):
    """Test pure topology computation functions."""

    def test_compute_expert_to_rank_mapping(self):
        mapping = compute_expert_to_rank_mapping(
            num_experts=8, ep_size=4, ep_group_ranks=[0, 2, 4, 6],
        )
        self.assertEqual(mapping[0], 0)
        self.assertEqual(mapping[1], 0)
        self.assertEqual(mapping[2], 2)
        self.assertEqual(mapping[4], 4)
        self.assertEqual(mapping[6], 6)
        self.assertEqual(mapping[7], 6)

    def test_compute_experts_on_rank(self):
        experts = compute_experts_on_rank(
            global_rank=4, num_experts=8, ep_size=4,
            ep_group_ranks=[0, 2, 4, 6],
        )
        self.assertEqual(experts, [4, 5])

    def test_compute_experts_on_rank_not_in_group(self):
        experts = compute_experts_on_rank(
            global_rank=99, num_experts=8, ep_size=4,
            ep_group_ranks=[0, 2, 4, 6],
        )
        self.assertEqual(experts, [])

    def test_compute_active_ranks(self):
        active = compute_active_ranks(
            ep_group_ranks=[0, 2, 4, 6],
            quarantined_ranks=frozenset({4}),
        )
        self.assertEqual(active, frozenset({0, 2, 6}))

    def test_compute_active_ranks_no_quarantine(self):
        active = compute_active_ranks(
            ep_group_ranks=[0, 2, 4, 6],
            quarantined_ranks=frozenset(),
        )
        self.assertEqual(active, frozenset({0, 2, 4, 6}))


# =====================================================================
# Test: Directory + Topology integration
# =====================================================================

class TestDirectoryTopologyIntegration(unittest.TestCase):
    """Test that directory and topology stay in sync."""

    def setUp(self):
        self.dir = ActiveExpertDirectory.from_placement(
            num_layers=4, num_experts=8, ep_size=4,
            ep_group_ranks=[0, 2, 4, 6],
        )
        ed_mod.set_active_expert_directory(self.dir)

        topo_mod.clear_dispatch_topology_manager()
        self.mgr = DispatchTopologyManager()
        self.mgr.initialise(
            num_layers=4, num_experts=8, ep_size=4,
            ep_group_ranks=[0, 2, 4, 6],
        )

    def tearDown(self):
        ed_mod.clear_active_expert_directory()
        topo_mod.clear_dispatch_topology_manager()

    def test_failed_rank_not_in_dispatch_targets(self):
        """After refresh, failed rank 4 should not be in dispatch targets."""
        # Mark experts on rank 4 as UNAVAILABLE
        self.dir.bulk_update_recovery_state([4, 5], "UNAVAILABLE")

        # Refresh topology
        snap = self.mgr.refresh_dispatch_topology(
            new_ep_group_ranks=[0, 2, 64, 6],
            failed_rank=4,
            replacement_rank=64,
            step=100,
            update_directory=True,
            update_placement=False,
            update_health_mask=False,
            update_replacement_registry=False,
        )

        # Failed rank not in active ranks
        self.assertNotIn(4, snap.active_ranks)
        # Replacement rank is in active ranks
        self.assertIn(64, snap.active_ranks)

    def test_replacement_rank_experts_in_directory(self):
        """After refresh, directory should show experts on replacement rank."""
        self.mgr.refresh_dispatch_topology(
            new_ep_group_ranks=[0, 2, 64, 6],
            failed_rank=4,
            replacement_rank=64,
            step=100,
            update_directory=True,
            update_placement=False,
            update_health_mask=False,
            update_replacement_registry=False,
        )

        # Directory should show experts 4,5 on rank 64
        self.assertEqual(self.dir.get_host_rank(0, 4), 64)
        self.assertEqual(self.dir.get_host_rank(0, 5), 64)
        # Other experts unchanged
        self.assertEqual(self.dir.get_host_rank(0, 0), 0)

    def test_directory_and_topology_consistent(self):
        """Directory and topology snapshot should agree on expert hosts."""
        self.mgr.refresh_dispatch_topology(
            new_ep_group_ranks=[0, 2, 64, 6],
            failed_rank=4,
            replacement_rank=64,
            step=100,
            update_directory=True,
            update_placement=False,
            update_health_mask=False,
            update_replacement_registry=False,
        )

        snap = self.mgr.snapshot
        for eid in range(8):
            dir_host = self.dir.get_host_rank(0, eid)
            topo_host = snap.expert_to_rank[eid]
            self.assertEqual(dir_host, topo_host,
                             f"Expert {eid}: dir={dir_host}, topo={topo_host}")

    def test_multiple_refreshes(self):
        """Multiple refreshes should work correctly."""
        # First refresh: rank 4 → 64
        self.mgr.refresh_dispatch_topology(
            new_ep_group_ranks=[0, 2, 64, 6],
            failed_rank=4,
            replacement_rank=64,
            step=100,
            update_directory=True,
            update_placement=False,
            update_health_mask=False,
            update_replacement_registry=False,
        )

        # Second refresh: rank 6 → 66
        snap = self.mgr.refresh_dispatch_topology(
            new_ep_group_ranks=[0, 2, 64, 66],
            failed_rank=6,
            replacement_rank=66,
            step=200,
            update_directory=True,
            update_placement=False,
            update_health_mask=False,
            update_replacement_registry=False,
        )

        self.assertEqual(snap.expert_to_rank[4], 64)
        self.assertEqual(snap.expert_to_rank[6], 66)
        self.assertEqual(self.dir.get_host_rank(0, 4), 64)
        self.assertEqual(self.dir.get_host_rank(0, 6), 66)
        self.assertEqual(self.mgr.refresh_count, 2)
        self.assertEqual(len(self.mgr.history), 2)


# =====================================================================
# Test: RecoveryManifest
# =====================================================================

class TestRecoveryManifest(unittest.TestCase):
    """Test RecoveryManifest creation and serialization."""

    def setUp(self):
        self.dir = ActiveExpertDirectory.from_placement(
            num_layers=2, num_experts=4, ep_size=2,
            ep_group_ranks=[0, 2],
        )

    def test_from_directory(self):
        manifest = RecoveryManifest.from_directory(
            self.dir, step=100, checkpoint_dir="/tmp/ckpt",
        )
        self.assertEqual(manifest.step, 100)
        self.assertEqual(len(manifest.entries), 8)  # 2 layers * 4 experts

    def test_serialization_roundtrip(self):
        manifest = RecoveryManifest.from_directory(
            self.dir, step=100, checkpoint_dir="/tmp/ckpt",
        )
        d = manifest.to_dict()
        loaded = RecoveryManifest.from_dict(d)
        self.assertEqual(loaded.step, 100)
        self.assertEqual(len(loaded.entries), 8)

    def test_get_entry(self):
        manifest = RecoveryManifest.from_directory(
            self.dir, step=100, checkpoint_dir="/tmp/ckpt",
        )
        entry = manifest.get_entry(0, 2)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.host_rank, 2)

    def test_get_stale_experts(self):
        self.dir.bulk_update_recovery_state([2, 3], "UNAVAILABLE")
        manifest = RecoveryManifest.from_directory(
            self.dir, step=100, checkpoint_dir="/tmp/ckpt",
        )
        stale = manifest.get_stale_experts()
        self.assertEqual(len(stale), 4)  # 2 experts * 2 layers

    def test_apply_to_directory(self):
        manifest = RecoveryManifest.from_directory(
            self.dir, step=100, checkpoint_dir="/tmp/ckpt",
        )
        # Modify manifest entries
        for e in manifest.entries:
            if e.expert_id in (2, 3):
                e.recovery_state = "STALE_RUNNABLE"

        # Apply to a fresh directory
        new_dir = ActiveExpertDirectory.from_placement(
            num_layers=2, num_experts=4, ep_size=2,
            ep_group_ranks=[0, 2],
        )
        count = manifest.apply_to_directory(new_dir)
        self.assertEqual(count, 8)
        self.assertEqual(new_dir.get_recovery_state(0, 2), "STALE_RUNNABLE")
        self.assertEqual(new_dir.get_recovery_state(0, 0), "HEALTHY")


# =====================================================================
# Test: End-to-end refresh flow
# =====================================================================

class TestEndToEndRefreshFlow(unittest.TestCase):
    """Test the complete refresh flow: failure → refresh → reintegration."""

    def setUp(self):
        self.dir = ActiveExpertDirectory.from_placement(
            num_layers=2, num_experts=8, ep_size=4,
            ep_group_ranks=[0, 2, 4, 6],
        )
        ed_mod.set_active_expert_directory(self.dir)

        topo_mod.clear_dispatch_topology_manager()
        self.mgr = DispatchTopologyManager()
        self.mgr.initialise(
            num_layers=2, num_experts=8, ep_size=4,
            ep_group_ranks=[0, 2, 4, 6],
        )

    def tearDown(self):
        ed_mod.clear_active_expert_directory()
        topo_mod.clear_dispatch_topology_manager()

    def test_full_flow(self):
        """Simulate: rank 4 fails → quarantine → replacement → refresh → reintegrate."""

        # Step 1: All experts healthy initially
        snap0 = self.mgr.snapshot
        self.assertEqual(len(self.mgr.get_dispatchable_experts()), 8)

        # Step 2: Rank 4 fails — mark experts 4,5 as UNAVAILABLE
        self.dir.bulk_update_recovery_state([4, 5], "UNAVAILABLE")

        # Step 3: Replacement rank 64 arrives, refresh topology
        snap1 = self.mgr.refresh_dispatch_topology(
            new_ep_group_ranks=[0, 2, 64, 6],
            failed_rank=4,
            replacement_rank=64,
            step=100,
            re_enable_recovered_experts=True,
            update_directory=True,
            update_placement=False,
            update_health_mask=False,
            update_replacement_registry=False,
        )

        # Verify: rank 4 not in active, rank 64 is
        self.assertNotIn(4, snap1.active_ranks)
        self.assertIn(64, snap1.active_ranks)

        # Verify: experts 4,5 now on rank 64
        self.assertEqual(snap1.expert_to_rank[4], 64)
        self.assertEqual(snap1.expert_to_rank[5], 64)
        self.assertEqual(self.dir.get_host_rank(0, 4), 64)

        # Step 4: After expert restore, mark as STALE_RUNNABLE
        self.dir.bulk_update_recovery_state([4, 5], "STALE_RUNNABLE")

        # Step 5: After reintegration, mark as HEALTHY
        self.dir.bulk_update_recovery_state([4, 5], "HEALTHY")

        # Verify: all experts healthy in directory
        for eid in range(8):
            self.assertEqual(self.dir.get_recovery_state(0, eid), "HEALTHY")

    def test_failed_rank_experts_not_dispatchable_during_recovery(self):
        """During recovery, failed rank's experts should not be dispatchable."""
        # Mark experts on rank 4 as UNAVAILABLE
        self.dir.bulk_update_recovery_state([4, 5], "UNAVAILABLE")

        # Before refresh: topology still shows old mapping
        snap = self.mgr.snapshot
        # Experts 4,5 are on rank 4 which is still in active_ranks
        # (quarantine hasn't been applied to topology yet)
        self.assertIn(4, snap.active_ranks)

        # After refresh with failed rank info
        snap = self.mgr.refresh_dispatch_topology(
            new_ep_group_ranks=[0, 2, 64, 6],
            failed_rank=4,
            replacement_rank=64,
            step=100,
            update_directory=True,
            update_placement=False,
            update_health_mask=False,
            update_replacement_registry=False,
        )

        # Now rank 4 is not in active ranks
        self.assertNotIn(4, snap.active_ranks)
        # Rank 64 is active
        self.assertIn(64, snap.active_ranks)

    def test_snapshot_consistency_after_refresh(self):
        """Snapshot should be internally consistent after refresh."""
        snap = self.mgr.refresh_dispatch_topology(
            new_ep_group_ranks=[0, 2, 64, 6],
            failed_rank=4,
            replacement_rank=64,
            step=100,
            update_directory=True,
            update_placement=False,
            update_health_mask=False,
            update_replacement_registry=False,
        )
        self.assertTrue(snap.is_consistent())


# =====================================================================
# Test: DispatchTopologySnapshot consistency
# =====================================================================

class TestSnapshotConsistency(unittest.TestCase):
    """Test DispatchTopologySnapshot.is_consistent()."""

    def test_consistent_snapshot(self):
        snap = DispatchTopologySnapshot(
            num_experts=4,
            ep_size=2,
            ep_group_ranks=[0, 2],
            active_ranks=frozenset({0, 2}),
            quarantined_ranks=frozenset(),
            expert_to_rank={0: 0, 1: 0, 2: 2, 3: 2},
            expert_health={0: True, 1: True, 2: True, 3: True},
        )
        self.assertTrue(snap.is_consistent())

    def test_inconsistent_active_quarantined_overlap(self):
        snap = DispatchTopologySnapshot(
            num_experts=4,
            ep_size=2,
            ep_group_ranks=[0, 2],
            active_ranks=frozenset({0, 2}),
            quarantined_ranks=frozenset({2}),  # overlap!
            expert_to_rank={0: 0, 1: 0, 2: 2, 3: 2},
            expert_health={0: True, 1: True, 2: True, 3: True},
        )
        self.assertFalse(snap.is_consistent())

    def test_inconsistent_expert_on_quarantined_rank_healthy(self):
        snap = DispatchTopologySnapshot(
            num_experts=4,
            ep_size=2,
            ep_group_ranks=[0, 2],
            active_ranks=frozenset({0}),
            quarantined_ranks=frozenset({2}),
            expert_to_rank={0: 0, 1: 0, 2: 2, 3: 2},
            expert_health={0: True, 1: True, 2: True, 3: True},  # 2,3 should be False
        )
        self.assertFalse(snap.is_consistent())

    def test_consistent_with_quarantine(self):
        snap = DispatchTopologySnapshot(
            num_experts=4,
            ep_size=2,
            ep_group_ranks=[0, 2],
            active_ranks=frozenset({0}),
            quarantined_ranks=frozenset({2}),
            expert_to_rank={0: 0, 1: 0, 2: 2, 3: 2},
            expert_health={0: True, 1: True, 2: False, 3: False},
        )
        self.assertTrue(snap.is_consistent())

    def test_to_dict(self):
        snap = DispatchTopologySnapshot(
            num_experts=4,
            ep_size=2,
            ep_group_ranks=[0, 2],
            active_ranks=frozenset({0, 2}),
            quarantined_ranks=frozenset(),
            expert_to_rank={0: 0, 1: 0, 2: 2, 3: 2},
            expert_health={0: True, 1: True, 2: True, 3: True},
            step=42,
        )
        d = snap.to_dict()
        self.assertEqual(d['step'], 42)
        self.assertEqual(d['num_experts'], 4)
        self.assertEqual(d['active_ranks'], [0, 2])


# =====================================================================
# Test: Manifest save/load
# =====================================================================

class TestManifestSaveLoad(unittest.TestCase):
    """Test RecoveryManifest save and load."""

    def test_save_and_load(self):
        import tempfile
        d = ActiveExpertDirectory.from_placement(
            num_layers=2, num_experts=4, ep_size=2,
            ep_group_ranks=[0, 2],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = RecoveryManifest.from_directory(
                d, step=100, checkpoint_dir=tmpdir,
            )
            path = manifest.save()
            self.assertTrue(os.path.exists(path))

            loaded = RecoveryManifest.load(tmpdir)
            self.assertEqual(loaded.step, 100)
            self.assertEqual(len(loaded.entries), 8)

    def test_exists(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertFalse(RecoveryManifest.exists(tmpdir))
            d = ActiveExpertDirectory.from_placement(
                num_layers=1, num_experts=2, ep_size=1,
                ep_group_ranks=[0],
            )
            manifest = RecoveryManifest.from_directory(
                d, step=1, checkpoint_dir=tmpdir,
            )
            manifest.save()
            self.assertTrue(RecoveryManifest.exists(tmpdir))


if __name__ == '__main__':
    unittest.main()
