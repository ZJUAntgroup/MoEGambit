"""Small dependency-free recovery metric registry."""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Dict, Mapping

from .events import RecoveryOutcome, RecoveryRecord

__all__ = ["RecoveryMetrics"]


class RecoveryMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._latency_sum_ms: Dict[str, float] = defaultdict(float)
        self._latency_count: Dict[str, int] = defaultdict(int)

    def increment(self, name: str, value: float = 1.0) -> None:
        if not name or value < 0:
            raise ValueError("counter name must be set and value non-negative")
        with self._lock:
            self._counters[name] += float(value)

    def gauge(self, name: str, value: float) -> None:
        if not name:
            raise ValueError("gauge name must be non-empty")
        with self._lock:
            self._gauges[name] = float(value)

    def observe_latency(self, phase: str, milliseconds: float) -> None:
        if not phase or milliseconds < 0:
            raise ValueError("latency phase must be set and duration non-negative")
        with self._lock:
            self._latency_sum_ms[phase] += float(milliseconds)
            self._latency_count[phase] += 1

    def observe_record(self, record: RecoveryRecord) -> None:
        outcome = (
            record.result.value
            if isinstance(record.result, RecoveryOutcome)
            else str(record.result)
        )
        self.increment("recovery_attempts_total")
        self.increment(f"recovery_result_{outcome}_total")
        self.gauge("recovery_epoch", record.recovery_epoch)
        self.gauge(
            "uncommitted_epochs",
            1.0 if record.validation.get("provisional") else 0.0,
        )
        for phase, duration in record.phase_latency_ms.items():
            self.observe_latency(str(phase), float(duration))
        replication_bytes = record.validation.get("replication_bytes")
        if isinstance(replication_bytes, (int, float)):
            self.increment("replication_bytes_total", float(replication_bytes))

    def snapshot(self) -> Mapping[str, object]:
        with self._lock:
            return {
                "counters": dict(sorted(self._counters.items())),
                "gauges": dict(sorted(self._gauges.items())),
                "latency_ms": {
                    phase: {
                        "count": self._latency_count[phase],
                        "sum": self._latency_sum_ms[phase],
                        "mean": self._latency_sum_ms[phase]
                        / self._latency_count[phase],
                    }
                    for phase in sorted(self._latency_count)
                },
            }

    def prometheus_text(self, namespace: str = "moegambit") -> str:
        snapshot = self.snapshot()
        lines = []
        counters = snapshot["counters"]
        gauges = snapshot["gauges"]
        latencies = snapshot["latency_ms"]
        for name, value in counters.items():
            lines.append(f"{namespace}_{name} {value}")
        for name, value in gauges.items():
            lines.append(f"{namespace}_{name} {value}")
        for phase, summary in latencies.items():
            label = phase.replace('"', "")
            lines.append(
                f'{namespace}_phase_latency_ms_sum{{phase="{label}"}} '
                f'{summary["sum"]}'
            )
            lines.append(
                f'{namespace}_phase_latency_ms_count{{phase="{label}"}} '
                f'{summary["count"]}'
            )
        return "\n".join(lines) + ("\n" if lines else "")
