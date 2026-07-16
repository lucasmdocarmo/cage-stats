"""external_kv_active tri-state semantics (CAGE audit 2026-07-16 SANITY-7).

vLLM 0.11.0 exposes NEITHER external-KV metric family
(``vllm:external_prefix_cache_queries_total`` nor
``vllm:prompt_tokens_by_source_total``); the old derivation fabricated a False
from the absent counters, so "connector idle" was indistinguishable from
"metric missing". The field is now Optional: None when neither family is in
the scrape, and a real True/False only when at least one family exists.
"""

import time

from cage_stats.metrics.engine import MetricsEngine
from cage_stats.metrics.parse import parse_metrics

BASE = 'vllm:num_requests_running{model_name="m",engine="0"} 0.0\n'


def _derive(text: str):
    return MetricsEngine().derive(parse_metrics(text), now=time.time())


def test_external_kv_none_when_neither_family_present():
    # vLLM 0.11.0 case: no external-KV series at all -> unknown, NOT a fabricated False.
    assert _derive(BASE).external_kv_active is None


def test_external_kv_false_when_family_present_but_zero():
    text = BASE + "vllm:external_prefix_cache_queries_total 0.0\n"
    assert _derive(text).external_kv_active is False


def test_external_kv_true_when_queries_positive():
    text = BASE + "vllm:external_prefix_cache_queries_total 3.0\n"
    assert _derive(text).external_kv_active is True


def test_external_kv_true_via_by_source_transfer():
    text = (
        BASE
        + 'vllm:prompt_tokens_by_source_total{source="external_kv_transfer"} 5.0\n'
    )
    assert _derive(text).external_kv_active is True


def test_external_kv_false_when_by_source_has_no_transfer():
    # The by-source family exists (so activity IS knowable) but shows no external
    # transfer -> a real False, not None.
    text = BASE + 'vllm:prompt_tokens_by_source_total{source="local_compute"} 7.0\n'
    assert _derive(text).external_kv_active is False
