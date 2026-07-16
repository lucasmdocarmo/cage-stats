"""Public, headless API for embedding cage-stats in other tools (e.g. CAGE).

Neither function imports Textual, so they are safe to call from a headless
process (a benchmark harness, a cron job, a notebook).

``fetch_snapshot(url, ...) -> Snapshot``
    Poll a vLLM server's ``/metrics`` twice (priming the EWMA rates) and return
    the derived :class:`~cage_stats.metrics.state.Snapshot` — the full dashboard
    payload (concurrency, throughput, latency quantiles, cache/KV, spec-decode,
    efficiency, GPU).

``snapshot_dict(url, ...) -> dict``
    Same, serialised to a plain JSON-able dict.

``dashboard_text(url, ...) -> str``
    Same, rendered as a static, one-shot terminal dashboard string.
"""

from __future__ import annotations

import asyncio
import time

from cage_stats.metrics.engine import MetricsEngine
from cage_stats.metrics.kv import load_model_dims
from cage_stats.metrics.parse import parse_metrics
from cage_stats.metrics.state import Snapshot, snapshot_to_dict
from cage_stats.providers.mock import MockProvider


def fetch_snapshot(
    url: str = "http://localhost:8000",
    *,
    metrics_path: str = "/metrics",
    api_key: str | None = None,
    interval: float = 1.0,
    mock: bool = False,
) -> Snapshot:
    """Return a derived :class:`Snapshot` for one vLLM instance.

    Raises ``RuntimeError`` if the server cannot be reached / scraped.
    """
    if mock:
        eng = MetricsEngine(dims=None, max_model_len=None)
        mp = MockProvider()
        eng.derive(parse_metrics(mp.metrics_text()), now=0.0)
        return eng.derive(parse_metrics(mp.metrics_text()), now=1.0)

    from cage_stats.providers.vllm import VllmProvider

    async def _go():
        p = VllmProvider(base_url=url, metrics_path=metrics_path, api_key=api_key)
        info = await p.fetch_model_info()
        r0 = await p.fetch_metrics()
        time.sleep(min(interval, 1.0))
        r1 = await p.fetch_metrics()
        await p.aclose()
        return info, r0, r1

    info, r0, r1 = asyncio.run(_go())
    if not r1.fetched_ok:
        raise RuntimeError(r1.error or "failed to fetch /metrics")
    # Guard against a 200 response whose body carries NO vLLM metrics (wrong metrics_path, a
    # proxy/error page, or a vLLM build that renamed the series). Without this, the engine's
    # `sum_value(...) or 0.0` coalescing fabricates an all-zero snapshot that is
    # indistinguishable from a real idle-zero measurement -- a silent data-integrity hole for
    # any downstream consumer (e.g. CAGE telemetry). Fail loud so the caller records the run
    # as "telemetry unavailable" (None) instead of recording fabricated zeros.
    if "vllm:" not in (r1.text or ""):
        raise RuntimeError(
            f"/metrics at {url}{metrics_path} returned no vLLM metrics "
            "(check metrics_path, or that this endpoint is a vLLM server)"
        )
    # r0 PRIMES every rate baseline and the session accounting: a failed/empty FIRST
    # poll with a healthy second one would zero-prime the rates (turning them into
    # fractions of lifetime totals) and make per-window deltas read as lifetime
    # values. Fail exactly as loud as for r1.
    if not r0.fetched_ok:
        raise RuntimeError(r0.error or "failed to fetch /metrics (first/priming poll)")
    if "vllm:" not in (r0.text or ""):
        raise RuntimeError(
            f"/metrics first poll at {url}{metrics_path} returned no vLLM metrics "
            "(priming poll must be valid; rates would be computed against garbage)"
        )
    md = load_model_dims(info.root, info.max_model_len)
    eng = MetricsEngine(dims=md.dims, max_model_len=md.max_model_len)
    eng.derive(parse_metrics(r0.text), now=0.0)
    return eng.derive(parse_metrics(r1.text), now=1.0)


def snapshot_dict(url: str = "http://localhost:8000", **kwargs) -> dict:
    """Headless snapshot as a JSON-able dict (the full dashboard payload)."""
    return snapshot_to_dict(fetch_snapshot(url, **kwargs))


def dashboard_text(url: str = "http://localhost:8000", *, interval: float = 1.0, **kwargs) -> str:
    """Headless one-shot static terminal dashboard for one vLLM instance."""
    from cage_stats.ui.text import render_dashboard

    snap = fetch_snapshot(url, interval=interval, **kwargs)
    return render_dashboard(snap, url=url, interval=interval)
