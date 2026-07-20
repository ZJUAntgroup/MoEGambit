import tempfile
from argparse import Namespace

from elastic_watcher import ElasticWatcher


class _Connection:
    def __init__(self):
        self.messages = []

    def sendall(self, data):
        self.messages.append(data.decode())


def _args(fault_dir, training_nnodes=2, nproc_per_node=2):
    return Namespace(
        port=20200,
        training_nnodes=training_nnodes,
        nproc_per_node=nproc_per_node,
        master_addr="10.0.0.1",
        master_port="20123",
        fault_dir=fault_dir,
        heartbeat_timeout=30.0,
        startup_heartbeat_timeout=600.0,
        forward_heartbeat_timeout=180.0,
        checkpoint_heartbeat_timeout=900.0,
        disconnect_grace_timeout=15.0,
        fallback_relaunch=True,
        fallback_exit_code=75,
        fallback_restart_standby=True,
        fault_inject_step=17,
        fault_inject_node=0,
        fault_inject_local_rank=1,
    )


def _manifest(node_rank, attempt=0, master_port="20123", digest="same"):
    return {
        "attempt": attempt,
        "node_rank": node_rank,
        "nnodes": 2,
        "nproc_per_node": 2,
        "world_size": 4,
        "master_addr": "10.0.0.1",
        "master_port": master_port,
        "command_sha256": digest,
    }


def _record(watcher, node_rank, manifest, connection):
    with watcher.lock:
        return watcher._record_startup_manifest_locked(
            node_rank, manifest, connection
        )


def test_startup_requires_a_matching_launcher_quorum():
    with tempfile.TemporaryDirectory() as fault_dir:
        watcher = ElasticWatcher(_args(fault_dir))
        connections = [_Connection(), _Connection()]

        assert _record(watcher, 0, _manifest(0), connections[0]) is None
        action = _record(watcher, 1, _manifest(1), connections[1])
        assert action[:2] == ("release", 0)

        watcher._complete_startup_action(*action)
        assert watcher.startup_released_attempt == 0
        assert all('"startup_release"' in conn.messages[-1] for conn in connections)


def test_startup_rejects_a_mismatched_master_port():
    with tempfile.TemporaryDirectory() as fault_dir:
        watcher = ElasticWatcher(_args(fault_dir))
        connection = _Connection()

        action = _record(
            watcher,
            0,
            _manifest(0, master_port="29999"),
            connection,
        )
        assert action[0] == "reject"
        assert "master_port" in action[3]
        assert watcher.startup_released_attempt == -1


def test_fallback_stays_latched_until_the_next_attempt_quorum():
    with tempfile.TemporaryDirectory() as fault_dir:
        watcher = ElasticWatcher(_args(fault_dir))
        standby_restarts = []
        watcher._start_standby_worker = lambda: standby_restarts.append("started")
        connections = [_Connection(), _Connection()]

        _record(watcher, 0, _manifest(0), connections[0])
        watcher._complete_startup_action(
            *_record(watcher, 1, _manifest(1), connections[1])
        )

        watcher.recovery_in_progress = True
        watcher.recovery_epoch = 1
        watcher.failed_node = 0
        watcher.killed_local_rank = 1
        watcher.node_connections = {0: connections[0], 1: connections[1]}
        watcher._initiate_fallback_relaunch("test")

        assert watcher.fallback_initiated
        assert not watcher.recovery_in_progress
        assert standby_restarts == []

        watcher._handle_fault(1, killed_local_rank=0, failure_scope="software")
        assert watcher.recovery_epoch == 1

        retry_connections = [_Connection(), _Connection()]
        assert (
            _record(watcher, 0, _manifest(0, attempt=1), retry_connections[0])
            is None
        )
        watcher._complete_startup_action(
            *_record(watcher, 1, _manifest(1, attempt=1), retry_connections[1])
        )

        assert watcher.startup_released_attempt == 1
        assert not watcher.fallback_initiated
        assert standby_restarts == ["started"]
