"""Headless API tests using cage-stats' synthetic (mock) data — no server needed."""

from cage_stats import api
from cage_stats.metrics.state import Snapshot


def test_fetch_snapshot_mock_is_full_and_connected():
    snap = api.fetch_snapshot(mock=True)
    assert isinstance(snap, Snapshot)
    assert snap.connected is True
    # The mock exercises the headline panels CAGE cares about.
    assert snap.spec_active and snap.spec_acceptance is not None
    assert snap.kv_dtype  # e.g. fp8_e4m3
    assert snap.src_compute is not None and snap.src_cache_hit is not None


def test_snapshot_dict_payload_shape():
    d = api.snapshot_dict(mock=True)
    assert isinstance(d, dict)
    # Top-level + nested kv group present.
    for key in ("spec_acceptance", "src_compute", "kv", "ttft", "running", "gen_tps"):
        assert key in d, key
    for kvk in ("dtype", "ratio", "ratio_kind", "usage"):
        assert kvk in d["kv"], kvk


def test_dashboard_text_mock_has_panels():
    text = api.dashboard_text(mock=True)
    assert isinstance(text, str) and text
    for panel in ("CONCURRENCY", "THROUGHPUT", "LATENCY", "CACHE & KV"):
        assert panel in text, panel
