"""
Metrics package for cage_stats.

Contains the full pipeline from raw Prometheus text to a typed ``Snapshot``:

  - ``state``      — data classes (``Snapshot``, ``GpuSnapshot``, ``Instance``, …)
  - ``parse``      — Prometheus exposition text → ``Families`` dict
  - ``timeseries`` — EWMA rate smoothing, histogram quantile estimation, history buffer
  - ``kv``         — KV-cache capacity/compression maths and model-dimension loading
  - ``engine``     — ``MetricsEngine``: stateful transformer from raw families to ``Snapshot``
"""
