"""SGLang → vLLM metric-name dialect translation (CAGE gap G-P2, 2026-08-26).

The :class:`~cage_stats.metrics.engine.MetricsEngine` consumes vLLM-prefixed
Prometheus family names. SGLang exposes the same *physical* quantities under
``sglang:``-prefixed names (docs.sglang.io/references/production_metrics.html,
fetched 2026-08-26). This module renames the parsed families so ONE engine
serves both backends — the S4 "Rosetta" pattern: engine dialects are recorded
translations, never silent assumptions.

Semantics notes (the honest part):

- ``sglang:token_usage`` is the fraction of the token pool in use — the same
  0..1 occupancy semantics as ``vllm:kv_cache_usage_perc`` (which, despite the
  ``_perc`` suffix, is also a 0..1 fraction). Values pass through UNCHANGED —
  no clamping, no fabrication.
- SGLang's documented metric set carries **NO preemption/retraction counter**.
  Its scarcity event is a *retraction* (a running request evicted back to the
  queue under KV pressure) and some builds expose a counter for it; the
  documented set does not. We therefore map the first present name among
  :data:`RETRACTION_CANDIDATES` onto ``vllm:num_preemptions_total`` and map
  NOTHING when none is present — the engine then reports
  ``preemptions_total=None`` and CAGE's regime labeler REFUSES the window
  (fail-closed) instead of reading a fabricated 0. Whether the pinned SGLang
  build exposes such a counter is a [VERIFY-LIVE] fact for S0's engine-parity
  row.
- ``sglang:cache_hit_rate`` is a *rate gauge*; vLLM's prefix-cache fields are
  cumulative *query/hit counters*. Deriving counters from a rate would
  fabricate data, so it is deliberately NOT mapped; SGLang cache-hit evidence
  rides CAGE's own V9 warm-repeat probe instead.

Families not named in the map pass through untouched (the engine ignores
unknown names), so no information is destroyed by translation.
"""

from __future__ import annotations

from cage_stats.metrics.parse import Families

__all__ = [
    "RETRACTION_CANDIDATES",
    "SGLANG_TO_VLLM",
    "translate_sglang_families",
]

#: Direct one-to-one renames: same physical quantity, different dialect.
SGLANG_TO_VLLM: dict[str, str] = {
    "sglang:token_usage": "vllm:kv_cache_usage_perc",
    "sglang:num_running_reqs": "vllm:num_requests_running",
    "sglang:num_queue_reqs": "vllm:num_requests_waiting",
    "sglang:prompt_tokens_total": "vllm:prompt_tokens_total",
    "sglang:generation_tokens_total": "vllm:generation_tokens_total",
}

#: Scarcity-counter names seen across SGLang builds, in preference order.
#: The FIRST present one maps to ``vllm:num_preemptions_total``; when none is
#: present no preemption family is emitted at all (absence must stay absence).
RETRACTION_CANDIDATES: tuple[str, ...] = (
    "sglang:num_retracted_reqs_total",
    "sglang:num_retracted_reqs",
    "sglang:retracted_reqs_total",
    "sglang:num_preempted_reqs",
)

_PREEMPT_TARGET = "vllm:num_preemptions_total"


def _find_retraction_source(families: Families) -> str | None:
    """First present retraction family, in candidate preference order.

    prometheus_client's legacy text parser appends ``_total`` to counter
    sample names (verified empirically 2026-08-26: a counter declared
    ``sglang:num_preempted_reqs`` parses as
    ``sglang:num_preempted_reqs_total``), so each candidate is checked in
    BOTH spellings — a build exposing the metric as a gauge arrives raw,
    as a counter arrives suffixed.
    """
    for cand in RETRACTION_CANDIDATES:
        for name in (cand, cand + "_total"):
            if name in families:
                return name
    return None


def translate_sglang_families(families: Families) -> Families:
    """Return a new Families dict with SGLang names renamed to engine dialect.

    Pure function: the input dict is not mutated. Collisions are impossible in
    practice (a scrape is either vLLM- or SGLang-prefixed); if a target name
    somehow already exists, the SGLang samples are APPENDED so nothing is
    silently dropped.
    """
    out: Families = {}
    retraction_source = _find_retraction_source(families)
    for name, samples in families.items():
        if name == retraction_source:
            target = _PREEMPT_TARGET
        else:
            target = SGLANG_TO_VLLM.get(name, name)
        out.setdefault(target, []).extend(samples)
    return out
