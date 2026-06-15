"""Static dashboard renderer tests (no Textual)."""

from cage_stats import api
from cage_stats.metrics.state import Snapshot
from cage_stats.ui.text import render_dashboard


def test_render_dashboard_full():
    snap = api.fetch_snapshot(mock=True)
    out = render_dashboard(snap, url="http://localhost:8000", interval=1.0)
    assert out.splitlines()[0].startswith("cage_stats")
    # Spec-decode + KV-compression surface in the dashboard.
    assert "SPEC DECODE" in out
    assert "vs fp16" in out  # KV compression ratio line


def test_render_dashboard_disconnected_is_graceful():
    snap = Snapshot(ts=0.0, connected=False, error="boom")
    out = render_dashboard(snap, url="http://down:8000")
    assert "not connected" in out
    # No panels rendered when down — must not raise.
    assert "CONCURRENCY" not in out
