from __future__ import annotations

from collections import defaultdict, deque


class LatencyTracker:
    def __init__(self, window_size: int) -> None:
        self._samples: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window_size))

    def record(self, conversation_key: str, elapsed_seconds: float) -> None:
        self._samples[conversation_key].append(elapsed_seconds)

    def summary(self) -> dict[str, dict[str, float | int]]:
        summary: dict[str, dict[str, float | int]] = {}
        for conversation_key, samples in self._samples.items():
            if not samples:
                continue
            values = sorted(samples)
            summary[conversation_key] = {
                "count": len(values),
                "p50_ms": round(self._percentile(values, 0.50) * 1000, 2),
                "p95_ms": round(self._percentile(values, 0.95) * 1000, 2),
                "p99_ms": round(self._percentile(values, 0.99) * 1000, 2),
            }
        return summary

    @staticmethod
    def _percentile(sorted_values: list[float], percentile: float) -> float:
        if not sorted_values:
            return 0.0
        index = int((len(sorted_values) - 1) * percentile)
        return sorted_values[index]
