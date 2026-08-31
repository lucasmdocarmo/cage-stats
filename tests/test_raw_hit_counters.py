"""Raw prefix-cache hit counters kept alongside the derived ratios (T4.2).

The engine derives ``prefix_hit_window`` / ``prefix_hit_lifetime`` from
``vllm:prefix_cache_queries_total`` / ``vllm:prefix_cache_hits_total`` and used
to DISCARD the raw cumulative values. Contrast #19 (dedup before/after the wire)
needs per-window DELTAS of the raw counters per instance, which cannot be
reconstructed from the ratios -- so the snapshot now carries the raw counters
too. Pinned here:

* present counters surface verbatim (summed across label sets, matching how
  every other counter in the snapshot aggregates);
* ABSENT families surface as None, never a fabricated 0.0, and a GENUINE zero
  counter survives as 0.0 (the E2b absence-is-not-zero doctrine, matching the
  ``preemptions_total`` / ``kv_usage`` precedent);
* ``snapshot_to_dict`` carries both fields;
* the derived ratio fields are unchanged by the addition.
"""

import time

from cage_stats.metrics.engine import MetricsEngine
from cage_stats.metrics.parse import parse_metrics
from cage_stats.metrics.state import snapshot_to_dict

BASE = 'vllm:num_requests_running{model_name="m",engine="0"} 0.0\n'

WITH_COUNTERS = (
    BASE
    + 'vllm:prefix_cache_queries_total{model_name="m",engine="0"} 100.0\n'
    + 'vllm:prefix_cache_hits_total{model_name="m",engine="0"} 97.0\n'
)


def _derive(text: str):
    return MetricsEngine().derive(parse_metrics(text), now=time.time())


def test_raw_counters_exposed_when_present():
    snap = _derive(WITH_COUNTERS)
    assert snap.prefix_cache_queries_total == 100.0
    assert snap.prefix_cache_hits_total == 97.0


def test_raw_counters_summed_across_label_sets():
    # Data-parallel scrape: counters aggregate the same way every other counter
    # in the snapshot does, so downstream deltas stay comparable to token totals.
    text = (
        WITH_COUNTERS
        + 'vllm:prefix_cache_queries_total{model_name="m",engine="1"} 40.0\n'
        + 'vllm:prefix_cache_hits_total{model_name="m",engine="1"} 3.0\n'
    )
    snap = _derive(text)
    assert snap.prefix_cache_queries_total == 140.0
    assert snap.prefix_cache_hits_total == 100.0


def test_raw_counters_none_when_family_absent():
    # No prefix-cache families in the scrape -> unknown, NOT a fabricated 0.0:
    # a zero here would read downstream as "cache exists and saw no queries".
    snap = _derive(BASE)
    assert snap.prefix_cache_queries_total is None
    assert snap.prefix_cache_hits_total is None


def test_raw_counters_genuine_zero_survives_as_zero():
    # A REAL 0.0 counter (cache enabled, nothing queried yet) must stay 0.0,
    # not collapse to None: `is None` is the only absence signal.
    text = (
        BASE
        + 'vllm:prefix_cache_queries_total{model_name="m",engine="0"} 0.0\n'
        + 'vllm:prefix_cache_hits_total{model_name="m",engine="0"} 0.0\n'
    )
    snap = _derive(text)
    assert snap.prefix_cache_queries_total == 0.0
    assert snap.prefix_cache_queries_total is not None
    assert snap.prefix_cache_hits_total == 0.0
    assert snap.prefix_cache_hits_total is not None


def test_snapshot_dict_carries_raw_counters():
    d = snapshot_to_dict(_derive(WITH_COUNTERS))
    assert d["prefix_cache_queries_total"] == 100.0
    assert d["prefix_cache_hits_total"] == 97.0
    d_absent = snapshot_to_dict(_derive(BASE))
    assert d_absent["prefix_cache_queries_total"] is None
    assert d_absent["prefix_cache_hits_total"] is None


def test_derived_ratios_unchanged_by_raw_counter_addition():
    eng = MetricsEngine()
    snap1 = eng.derive(parse_metrics(WITH_COUNTERS), now=time.time())
    assert snap1.prefix_hit_lifetime == 0.97
    text2 = (
        BASE
        + 'vllm:prefix_cache_queries_total{model_name="m",engine="0"} 110.0\n'
        + 'vllm:prefix_cache_hits_total{model_name="m",engine="0"} 102.0\n'
    )
    snap2 = eng.derive(parse_metrics(text2), now=time.time())
    assert snap2.prefix_hit_window == 0.5
    assert snap2.prefix_cache_queries_total == 110.0
    assert snap2.prefix_cache_hits_total == 102.0
