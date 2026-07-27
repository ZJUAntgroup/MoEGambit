"""Phase C node-agent and worker-supervisor contracts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from moegambit.errors import ContractViolation
from moegambit.agent.node_agent import NodeAgent, NodeLaunchSpec
from moegambit.agent.worker_supervisor import WorkerSupervisor
from moegambit.runtime.recovery_plan import RecoveryMode, RecoveryPlan, WorkerEndpoint


class _FakeProcess:
    next_pid = 100

    def __init__(self, argv, **kwargs):
        self.argv = argv
        self.kwargs = kwargs
        self.pid = _FakeProcess.next_pid
        _FakeProcess.next_pid += 1
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        del timeout
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def _launch_spec():
    return NodeLaunchSpec(
        nnodes=2,
        nproc_per_node=2,
        node_rank=1,
        master_addr="127.0.0.1",
        master_port=24000,
        argv=("python", "train.py"),
        env={"TORCHELASTIC_RUN_ID": "old", "CUSTOM": "1"},
    )


def test_node_agent_starts_argv_workers_with_clean_distributed_identity(monkeypatch):
    monkeypatch.setattr("os.killpg", lambda pid, signal: None)
    supervisor = WorkerSupervisor(process_factory=_FakeProcess, base_env={})
    agent = NodeAgent(_launch_spec(), supervisor=supervisor)

    handles = agent.start_all()

    assert [handle.spec.logical_rank for handle in handles] == [2, 3]
    first_env = handles[0].process.kwargs["env"]
    assert handles[0].process.kwargs["shell"] is False
    assert first_env["RANK"] == "2"
    assert first_env["WORLD_SIZE"] == "4"
    assert first_env["MOEGAMBIT_REPLACEMENT"] == "0"
    assert "TORCHELASTIC_RUN_ID" not in first_env


def test_node_agent_replacement_retains_rank_and_carries_frozen_plan(monkeypatch):
    monkeypatch.setattr("os.killpg", lambda pid, signal: None)
    supervisor = WorkerSupervisor(process_factory=_FakeProcess, base_env={})
    agent = NodeAgent(_launch_spec(), supervisor=supervisor)
    agent.start_all()
    plan = RecoveryPlan(
        protocol_version=1,
        recovery_epoch=3,
        failed_ranks=(2,),
        resume_step=8,
        mode=RecoveryMode.PEER,
        replacements={2: WorkerEndpoint("127.0.0.1", 24001, 1, 0)},
    )

    replaced = agent.apply_plan(plan)

    assert len(replaced) == 1
    handle = replaced[0]
    assert handle.spec.logical_rank == 2
    assert handle.generation == 1
    assert handle.process.kwargs["env"]["MOEGAMBIT_REPLACEMENT"] == "1"
    assert handle.process.kwargs["env"]["MOEGAMBIT_RECOVERY_EPOCH"] == "3"
    assert (
        handle.process.kwargs["env"]["MOEGAMBIT_RECOVERY_PLAN_DIGEST"]
        == plan.digest()
    )


def test_control_agent_modules_import_without_torch():
    code = """
import sys
import moegambit.agent.node_agent
import moegambit.agent.worker_supervisor
import moegambit.control.protocol
import moegambit.control.service
import moegambit.control.watcher
runtime = moegambit.initialize(config=moegambit.RuntimeConfig(enabled=False))
assert not runtime.enabled
assert not any(name == 'torch' or name.startswith('torch.') for name in sys.modules)
"""
    environment = os.environ.copy()
    source_root = Path(__file__).parents[2] / "src"
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(source_root), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_node_agent_wait_rejects_an_unstarted_agent():
    agent = NodeAgent(
        _launch_spec(),
        supervisor=WorkerSupervisor(process_factory=_FakeProcess, base_env={}),
    )

    with pytest.raises(ContractViolation, match="no workers"):
        agent.wait(poll_interval_s=0.01)
