"""kv_usage absence + multi-engine refusal semantics (E2b + T4.4).

An absent ``vllm:kv_cache_usage_perc`` gauge must surface as None, never a
fabricated 0.0: downstream, the CAGE memory-pressure regime gate reads the
snapshot's occupancy, and a fabricated zero flips "occupancy unknown" into
"unpressured". The old ``first_value(...) or 0.0`` also could not represent a
genuine 0.0 reading distinctly, since 0.0 is falsy. Mirrors the
``preemptions_total`` / ``external_kv_active`` precedent: absence is NEVER zero.

Multi-label-set scrapes (vLLM data-parallel) are a REFUSAL, not an average:
each engine's gauge is a fraction of a DIFFERENT KV pool, so a mean fabricates
a number with no physical meaning, and the old ``first_value`` silently dropped
every engine but the first while token counters summed across all of them.
Pinned: multiple label sets -> ``kv_usage`` None + ``kv_usage_multi_engine``
True; a single label set (and the absent-gauge case) is byte-identical to the
pre-T4.4 behavior with the flag False.
"""

import time

from cage_stats.metrics.engine import MetricsEngine
from cage_stats.metrics.parse import parse_metrics
from cage_stats.metrics.state import snapshot_to_dict

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


def test_kv_usage_refused_when_multi_engine():
    # Two label sets = two DIFFERENT KV pools: occupancy is refused (None) and
    # flagged, never averaged into a physically meaningless number and never
    # silently narrowed to the first engine.
    text = (
        BASE
        + 'vllm:kv_cache_usage_perc{model_name="m",engine="0"} 0.42\n'
        + 'vllm:kv_cache_usage_perc{model_name="m",engine="1"} 0.10\n'
    )
    snap = _derive(text)
    assert snap.kv_usage is None
    assert snap.kv_usage_multi_engine is True


def test_kv_usage_multi_engine_flag_false_for_single_label_set():
    # Single-engine scrape: value passes through unchanged, flag stays False --
    # pre-T4.4 behavior byte-identical.
    text = BASE + 'vllm:kv_cache_usage_perc{model_name="m",engine="0"} 0.42\n'
    snap = _derive(text)
    assert snap.kv_usage == 0.42
    assert snap.kv_usage_multi_engine is False


def test_kv_usage_multi_engine_flag_false_when_gauge_absent():
    # Absence is NOT multi-engine: kv_usage=None + flag False means "gauge
    # missing", while None + flag True means "refused over averaging".
    snap = _derive(BASE)
    assert snap.kv_usage is None
    assert snap.kv_usage_multi_engine is False


def test_snapshot_dict_carries_multi_engine_flag():
    text = (
        BASE
        + 'vllm:kv_cache_usage_perc{model_name="m",engine="0"} 0.42\n'
        + 'vllm:kv_cache_usage_perc{model_name="m",engine="1"} 0.10\n'
    )
    d = snapshot_to_dict(_derive(text))
    assert d["kv_usage_multi_engine"] is True
    assert d["kv"]["usage_multi_engine"] is True
    assert d["kv"]["usage"] is None
    d_single = snapshot_to_dict(
        _derive(BASE + 'vllm:kv_cache_usage_perc{model_name="m",engine="0"} 0.42\n')
    )
    assert d_single["kv"]["usage_multi_engine"] is False
    assert d_single["kv"]["usage"] == 0.42


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
