import json
from types import SimpleNamespace

from moegambit.cli.launch import main
from moegambit.cli.watch import main as watch_main


def test_launch_dry_run_reports_switches(capsys):
    result = main(
        [
            "--adapter",
            "megatron",
            "--hot-swap",
            "--no-zero2",
            "--dry-run",
            "--",
            "python",
            "pretrain_gpt.py",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["features"] == {"hot_swap": True, "zero2": False}
    assert payload["environment"]["ELASTIC_ZERO2_MEMORY_REPLICATION"] == "0"
    assert "--use-distributed-optimizer" not in payload["command"]


def test_watcher_is_not_started_when_hot_swap_is_disabled(capsys):
    result = watch_main(
        ["--adapter", "megatron", "--no-hot-swap", "--no-zero2"]
    )
    assert result == 0
    assert "no watcher was started" in capsys.readouterr().out


def test_megatron_watcher_receives_runtime_switches(monkeypatch):
    observed = {}

    def fake_run(command, env, check):
        observed["command"] = command
        observed["env"] = env
        observed["check"] = check
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("moegambit.cli.watch.subprocess.run", fake_run)
    result = watch_main(
        [
            "--adapter",
            "megatron",
            "--hot-swap",
            "--zero2",
            "--port",
            "20300",
            "--",
            "--training-nnodes",
            "8",
        ]
    )

    assert result == 0
    assert observed["env"]["MOEGAMBIT_HOT_SWAP"] == "1"
    assert observed["env"]["MOEGAMBIT_ZERO2"] == "1"
    assert "--port" in observed["command"]
    assert "20300" in observed["command"]
