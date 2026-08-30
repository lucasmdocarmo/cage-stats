"""Pins for the SGLang metric dialect (CAGE gap G-P2, 2026-08-26).

What is pinned and why:

- The translation maps the DOCUMENTED SGLang names (docs.sglang.io production
  metrics, fetched 2026-08-26) onto the vLLM-dialect names the engine
  consumes, value-preserving, input-unmutated.
- ABSENCE STAYS ABSENCE: with no retraction/preemption family in the scrape,
  the translated families must NOT contain ``vllm:num_preemptions_total`` —
  the engine then derives ``preemptions_total=None`` and CAGE's regime
  labeler refuses the window instead of reading a fabricated 0 (the E2b
  lesson, applied to SGLang).
- ``sglang:cache_hit_rate`` (a rate gauge) must NEVER be turned into the
  vLLM query/hit counters — deriving counters from a rate fabricates data.
- End-to-end: exposition text -> parse -> translate -> MetricsEngine.derive
  yields kv_usage from ``sglang:token_usage`` on the plain Snapshot path.
"""

from __future__ import annotations

import pytest

from cage_stats.api import fetch_snapshot
from cage_stats.metrics.engine import MetricsEngine
from cage_stats.metrics.parse import parse_metrics
from cage_stats.metrics.sglang_dialect import (
    RETRACTION_CANDIDATES,
    SGLANG_TO_VLLM,
    translate_sglang_families,
)

_LBL = 'model_name="test-model"'


def _sglang_text(*, token_usage: float = 0.93, retraction: str | None = None,
                 retraction_value: float = 7.0) -> str:
    """Exposition text using the documented SGLang names (mock.py style)."""
    lines = [
        "# TYPE sglang:token_usage gauge",
        f"sglang:token_usage{{{_LBL}}} {token_usage}",
        "# TYPE sglang:num_running_reqs gauge",
        f"sglang:num_running_reqs{{{_LBL}}} 4.0",
        "# TYPE sglang:num_queue_reqs gauge",
        f"sglang:num_queue_reqs{{{_LBL}}} 2.0",
        "# TYPE sglang:prompt_tokens_total counter",
        f"sglang:prompt_tokens_total{{{_LBL}}} 1000.0",
        "# TYPE sglang:generation_tokens_total counter",
        f"sglang:generation_tokens_total{{{_LBL}}} 500.0",
        "# TYPE sglang:cache_hit_rate gauge",
        f"sglang:cache_hit_rate{{{_LBL}}} 0.75",
        "# TYPE sglang:gen_throughput gauge",
        f"sglang:gen_throughput{{{_LBL}}} 123.0",
    ]
    if retraction is not None:
        lines += [
            f"# TYPE {retraction} counter",
            f"{retraction}{{{_LBL}}} {retraction_value}",
        ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Pure translation
# ---------------------------------------------------------------------------

def test_documented_names_map_value_preserving() -> None:
    fam = parse_metrics(_sglang_text(token_usage=0.93))
    out = translate_sglang_families(fam)
    assert out["vllm:kv_cache_usage_perc"][0][1] == pytest.approx(0.93)
    assert out["vllm:num_requests_running"][0][1] == 4.0
    assert out["vllm:num_requests_waiting"][0][1] == 2.0
    assert out["vllm:prompt_tokens_total"][0][1] == 1000.0
    assert out["vllm:generation_tokens_total"][0][1] == 500.0
    # labels survive translation (model attribution depends on them)
    assert out["vllm:num_requests_running"][0][0]["model_name"] == "test-model"
    # every mapped source name is gone from the output
    assert not (set(SGLANG_TO_VLLM) & set(out))


def test_absence_stays_absence_no_preemptions_family() -> None:
    out = translate_sglang_families(parse_metrics(_sglang_text(retraction=None)))
    assert "vllm:num_preemptions_total" not in out, (
        "no retraction family in the scrape must mean NO preemptions family — "
        "the engine derives None and the CAGE regime labeler refuses (never 0)"
    )


@pytest.mark.parametrize("candidate", RETRACTION_CANDIDATES)
def test_each_retraction_candidate_maps_to_preemptions(candidate: str) -> None:
    out = translate_sglang_families(
        parse_metrics(_sglang_text(retraction=candidate, retraction_value=7.0))
    )
    assert out["vllm:num_preemptions_total"][0][1] == 7.0
    # the source family is gone in BOTH spellings (the legacy parser appends
    # ``_total`` to counter sample names — verified empirically 2026-08-26)
    assert candidate not in out and f"{candidate}_total" not in out


def test_cache_hit_rate_is_never_counterfeited_into_counters() -> None:
    out = translate_sglang_families(parse_metrics(_sglang_text()))
    assert "vllm:prefix_cache_queries_total" not in out
    assert "vllm:prefix_cache_hits_total" not in out
    # the rate gauge itself passes through untouched (engine ignores it)
    assert out["sglang:cache_hit_rate"][0][1] == pytest.approx(0.75)


def test_unknown_families_pass_through_and_input_not_mutated() -> None:
    fam = parse_metrics(_sglang_text())
    before = {k: list(v) for k, v in fam.items()}
    out = translate_sglang_families(fam)
    assert fam == before, "translation must not mutate its input"
    assert out["sglang:gen_throughput"][0][1] == 123.0


def test_vllm_families_are_identity() -> None:
    text = (
        "# TYPE vllm:kv_cache_usage_perc gauge\n"
        f"vllm:kv_cache_usage_perc{{{_LBL}}} 0.5\n"
        "# TYPE vllm:num_preemptions_total counter\n"
        f"vllm:num_preemptions_total{{{_LBL}}} 3.0\n"
    )
    fam = parse_metrics(text)
    assert translate_sglang_families(fam) == fam


# ---------------------------------------------------------------------------
# Engine end-to-end (parse -> translate -> derive)
# ---------------------------------------------------------------------------

def _derive(text: str):
    eng = MetricsEngine(dims=None, max_model_len=None)
    fam = translate_sglang_families(parse_metrics(text))
    eng.derive(fam, now=0.0)
    return eng.derive(fam, now=1.0)


def test_engine_kv_usage_from_token_usage_and_none_preemptions() -> None:
    snap = _derive(_sglang_text(token_usage=0.93, retraction=None))
    assert snap.kv_usage == pytest.approx(0.93)
    assert snap.preemptions_total is None, (
        "documented SGLang set has no scarcity counter -> None end-to-end "
        "([VERIFY-LIVE at S0]: whether the pinned build exposes one)"
    )


def test_engine_preemptions_flow_when_counter_present() -> None:
    snap = _derive(
        _sglang_text(retraction="sglang:num_retracted_reqs_total", retraction_value=7.0)
    )
    assert snap.preemptions_total == 7.0
    assert snap.kv_usage == pytest.approx(0.93)


# ---------------------------------------------------------------------------
# api.fetch_snapshot dialect plumbing
# ---------------------------------------------------------------------------

def test_fetch_snapshot_rejects_unknown_dialect_before_any_io() -> None:
    with pytest.raises(ValueError, match="unknown metrics dialect"):
        fetch_snapshot(dialect="tgi", mock=True)


def test_fetch_snapshot_default_dialect_mock_path_unchanged() -> None:
    snap = fetch_snapshot(mock=True)  # pre-dialect behavior must be intact
    assert snap is not None
