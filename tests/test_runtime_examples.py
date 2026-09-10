"""Execute the documented entry points, including real CPU DDP checkpointing."""

import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def test_node_launcher_preserves_workload_flags(tmp_path):
    output = tmp_path / "arguments.json"
    command = [sys.executable, "-m", "moegambit.cli.launch",
               "--nnodes", "1", "--nproc-per-node", "1", "--node-rank", "0",
               "--master-addr", "127.0.0.1", "--master-port", _free_port(), "--",
               sys.executable, "-c",
               "import json,sys;from pathlib import Path;Path(sys.argv[1]).write_text(json.dumps(sys.argv[2:]))",
               str(output), "--dry-run", "--adapter", "workload-adapter"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text()) == ["--dry-run", "--adapter", "workload-adapter"]


def test_megatron_dry_run_keeps_active_and_spare_configuration(tmp_path):
    env = dict(os.environ, DRY_RUN="1", TRAINING_NNODES="1", NPROC_PER_NODE="2",
               TP_SIZE="1", PP_SIZE="1", EP_SIZE="1", CP_SIZE="1", SAVE_INTERVAL="3",
               MASTER_ADDR="192.0.2.1", ELASTIC_WATCHER_ADDR="192.0.2.9",
               DATA_PATH=str(tmp_path / "data with spaces"),
               TOKENIZER_DIR=str(tmp_path / "tokenizer with spaces"),
               CKPT_DIR=str(tmp_path / "checkpoints"),
               ELASTIC_FAULT_DIR=str(tmp_path / "faults"))
    commands = []
    for node in ("0", "1"):
        result = subprocess.run(["bash", "examples/megatron/run_hot_spare.sh"],
                                cwd=ROOT, env=dict(env, NODE_RANK=node),
                                text=True, capture_output=True, timeout=15)
        assert result.returncode == 0, result.stderr
        command = next(line.split("DRY_RUN", 1)[1] for line in result.stdout.splitlines()
                       if "DRY_RUN" in line and "pretrain_gpt.py" in line)
        argv = shlex.split(command)
        for option, value in [("--data-path", env["DATA_PATH"]),
                              ("--tokenizer-model", env["TOKENIZER_DIR"]),
                              ("--save-interval", "3")]:
            assert argv[argv.index(option) + 1] == value
        commands.append(argv)
    assert commands[0][commands[0].index("--nnodes") + 1] == "1"
    assert not (tmp_path / "faults").exists()
    assert not (tmp_path / "checkpoints").exists()


def _train(tmp_path, steps, extra_env=None):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("MOEGAMBIT_", "ELASTIC_"))}
    env.update(extra_env or {})
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc-per-node=2",
         "--master-addr=127.0.0.1", "--master-port=" + _free_port(),
         "examples/generic_ddp/train_loop.py", "--steps", str(steps),
         "--checkpoint-dir", str(tmp_path), "--checkpoint-interval", "1"],
        env=env, cwd=ROOT, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    records = [json.loads(line.split("completed:", 1)[1])
               for line in result.stdout.splitlines() if line.startswith("completed:")]
    assert len(records) == 2, result.stdout
    assert {record["rank"] for record in records} == {0, 1}
    assert all(record["ddp"] and record["completed_steps"] == steps for record in records)
    assert len({record["model_digest"] for record in records}) == 1
    return records[0]["model_digest"]


def test_real_ddp_checkpoint_and_cold_resume_match_uninterrupted_run(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("CPU DDP requires Gloo")
    baseline = _train(tmp_path / "baseline", 4)
    _train(tmp_path / "resume", 2)
    checkpoint = tmp_path / "resume" / "step-2.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["resume_step"] == 2
    assert payload["optimizer"]["state"]
    resumed = _train(tmp_path / "resume", 4, {
        "MOEGAMBIT_CHECKPOINT_RELAUNCH": "1",
        "MOEGAMBIT_CHECKPOINT_LOCATOR": str(checkpoint),
        "MOEGAMBIT_CHECKPOINT_STEP": "2",
    })
    assert resumed == baseline
    assert len(list((tmp_path / "resume").glob("step-*.pt"))) == 4
    assert not list(tmp_path.rglob("*.tmp"))
