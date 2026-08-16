"""kv_usage absence semantics (CAGE code-assertion walkthrough 2026-08-12, E2b).

An absent ``vllm:kv_cache_usage_perc`` gauge must surface as None, never a
fabricated 0.0: downstream, the CAGE memory-pressure regime gate reads the
snapshot's occupancy, and a fabricated zero flips "occupancy unknown" into
"unpressured". The old ``first_value(...) or 0.0`` also could not represent a
genuine 0.0 reading distinctly, since 0.0 is falsy. Mirrors the
``preemptions_total`` / ``external_kv_active`` precedent: absence is NEVER zero.
"""

import time

from cage_stats.metrics.engine import MetricsEngine
from cage_stats.metrics.parse import parse_metrics

BASE = 'vllm:num_requests_running{model_name="m",engine="0"} 0.0\n'


def _derive(text: str):
    return MetricsEngine().derive(parse_metrics(text), now=time.time())


def test_kv_usage_none_when_gauge_absent():
    # No occupancy gauge in the scrape -> unknown, NOT a fabricated 0.0.
    assert _derive(BASE).kv_usage is None


def test_kv_usage_zero_when_gauge_reports_zero():
    # A REAL 0.0 gauge value (empty cache) must survive as 0.0, not None: the
    # explicit `is None` check distinguishes what `or 0.0` collapsed together.
    text = BASE + 'vllm:kv_cache_usage_perc{model_name="m",engine="0"} 0.0\n'
    snap = _derive(text)
    assert snap.kv_usage == 0.0
    assert snap.kv_usage is not None


def test_kv_usage_passes_through_nonzero_value():
    text = BASE + 'vllm:kv_cache_usage_perc{model_name="m",engine="0"} 0.42\n'
    assert _derive(text).kv_usage == 0.42


def test_kv_used_tokens_none_when_usage_absent():
    # Capacity is knowable from cache_config_info, but without the occupancy
    # gauge the used-token count is not -> None, never a fabricated 0.
    cc = (
        'vllm:cache_config_info{block_size="16",cache_dtype="auto",'
        'num_gpu_blocks="100",engine="0"} 1.0\n'
    )
    snap = _derive(BASE + cc)
    assert snap.kv_capacity_tokens == 1600
    assert snap.kv_used_tokens is None
