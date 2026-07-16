"""
Time-series primitives: EWMA rate smoothing, histogram quantile estimation,
and a bounded history ring-buffer.

``Rate``
    Tracks the per-second rate of a monotonically-increasing counter using an
    Exponentially Weighted Moving Average.  On the first sample it initialises
    state; on subsequent samples it computes an instantaneous rate
    ``(Δcounter / Δt)`` and blends it with the smoothed value:
    ``smoothed = α × inst + (1 − α) × smoothed``.
    A negative Δcounter (server restart / counter wrap) resets state without
    crashing.  Default ``α = 0.3`` balances responsiveness against noise.

``histogram_quantile``
    Prometheus-style linear interpolation of a quantile from cumulative
    ``[(le, cum_count)]`` buckets.  Returns ``None`` when there are no buckets
    or the total count is zero.

``windowed_buckets``
    Computes per-``le`` deltas between two consecutive histogram snapshots,
    returning a window-scoped bucket list for more accurate recent quantiles.
    Falls back to the raw current buckets on counter resets (negative deltas).

``Series``
    A fixed-length deque of float values.  The default ``maxlen=120`` keeps
    120 seconds of history at a 1-second polling interval.

``History``
    A named collection of ``Series`` objects.  ``push(name, value)`` appends to
    the named series, creating it on first use.  ``series(name)`` returns the
    ``Series`` (read-only access from render code).
"""

from __future__ import annotations

from collections import deque


class Rate:
    def __init__(self, alpha: float = 0.3) -> None:
        self.alpha = alpha
        self.value = 0.0
        self._prev_value: float | None = None
        self._prev_t: float | None = None
        self._primed = False

    def update(self, raw: float, t: float) -> float:
        if self._prev_value is None or self._prev_t is None:
            self._prev_value, self._prev_t = raw, t
            return self.value
        dt = t - self._prev_t
        if dt <= 0:
            return self.value
        if raw < self._prev_value:
            self._prev_value, self._prev_t = raw, t
            return self.value
        inst = (raw - self._prev_value) / dt
        # PRIME with the first real delta instead of blending it against the fake 0
        # seed: a 0-seeded EWMA under-reports by (1-alpha)^n, and the two-poll
        # api/CLI snapshot path computes exactly ONE delta -- every gen_tps/
        # prompt_tps/req_rate it ever reported was 0.3x the truth (verified in the
        # 2026-07-15 run data: prompt_tps_peak 96.9 == 0.3 * 323 actual).
        if not self._primed:
            self.value = inst
            self._primed = True
        else:
            self.value = self.alpha * inst + (1 - self.alpha) * self.value
        self._prev_value, self._prev_t = raw, t
        return self.value


def histogram_quantile(buckets: list[tuple[float, float]], q: float) -> float | None:
    if not buckets:
        return None
    total = buckets[-1][1]
    if total <= 0:
        return None
    target = q * total
    prev_le = 0.0
    prev_count = 0.0
    for le, count in buckets:
        if count >= target:
            if le == float("inf"):
                return prev_le
            bucket_count = count - prev_count
            if bucket_count <= 0:
                return le
            frac = (target - prev_count) / bucket_count
            return prev_le + frac * (le - prev_le)
        prev_le, prev_count = le, count
    return buckets[-1][0]


def windowed_buckets(
    prev: list[tuple[float, float]], cur: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    prev_map = dict(prev)
    out: list[tuple[float, float]] = []
    for le, count in cur:
        delta = count - prev_map.get(le, 0.0)
        if delta < 0:
            return cur
        out.append((le, delta))
    return out


class Series:
    def __init__(self, maxlen: int = 120) -> None:
        self.values: deque[float] = deque(maxlen=maxlen)

    def push(self, v: float) -> None:
        self.values.append(v)


class History:
    def __init__(self, maxlen: int = 120) -> None:
        self._maxlen = maxlen
        self._series: dict[str, Series] = {}

    def series(self, name: str) -> Series:
        if name not in self._series:
            self._series[name] = Series(self._maxlen)
        return self._series[name]

    def push(self, name: str, value: float) -> None:
        self.series(name).push(value)
