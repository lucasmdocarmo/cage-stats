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
  ``preemptions_total=None``, which CAGE records as absent scarcity (it never
  gates the regime label since CAGE C11, 2026-10-10), never a fabricated 0. Whether the pinned SGLang
  build exposes such a counter is a [VERIFY-LIVE] fact for S0's engine-parity
  row.
  CAGE S1 (2026-10-23, live): the pinned 0.5.10.post1 build runs its metrics in
  prometheus_client multiprocess mode and creates the labeled counter only at
  its first retraction, so a server that has not retracted shows the
  ``sglang:num_retracted_reqs`` gauge and no counter at all. That gauge is the
  witness that the metrics collector is up; with it present and no counter,
  the count is 0 (see :func:`translate_sglang_families`). Limitation: a build
  with the gauge but no counter would also read 0; CAGE installs the exact
  0.5.10.post1 build, which defines the counter.
- ``sglang:num_requests_total`` maps to ``vllm:request_success_total`` so the
  engine derives ``req_rate`` from a scraped counter (ADR-0148 Batch C); the
  two counters' semantics (processed vs finished) is recorded as [A] and
  VERIFY-LIVE in the map.
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
    # ADR-0148 Batch C (CAGE, 2026-10-08): the request counter, so req_rate has
    # a producer under SGLang (without it the engine derived a rate from an
    # absent family). [A] SGLang documents num_requests_total as "requests
    # processed"; vLLM's request_success_total counts finished requests. The
    # equivalence is VERIFY-LIVE at the next S0 rehearsal: the 2026-10-07
    # landing probed no request family.
    "sglang:num_requests_total": "vllm:request_success_total",
}

#: Scarcity-counter names seen across SGLang builds, in preference order.
#: The FIRST present one maps to ``vllm:num_preemptions_total``; when none is
#: present no preemption family is emitted (absence stays absence), unless the
#: lazy-counter witness gauge below is sampled, which reads 0 (CAGE S1 2026-10-23).
#: CAGE A7 (owner 2026-10-20): the pinned SGLang 0.5.10.post1 exports a
#: CUMULATIVE counter, sglang:num_retracted_requests_total, and a last-batch
#: GAUGE, sglang:num_retracted_reqs. Only counters map onto the preemption
#: counter (the regime record takes its change over the window); the gauge is
#: no candidate and passes through under its own name as a diagnostic.
RETRACTION_CANDIDATES: tuple[str, ...] = (
    "sglang:num_retracted_requests_total",
    "sglang:num_retracted_requests",
    "sglang:num_retracted_reqs_total",
    "sglang:retracted_reqs_total",
    "sglang:num_preempted_reqs",
)

_PREEMPT_TARGET = "vllm:num_preemptions_total"

#: CAGE S1 (2026-10-23): the last-batch retraction gauge the pinned build
#: exports from server start. Present with no retraction counter, it means no
#: retraction since the server started (the counter is created lazily).
_LAZY_COUNTER_WITNESS = "sglang:num_retracted_reqs"


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
            if not samples:
                # CAGE A7 (2026-10-21): declared, no sample = no retraction since
                # the server started. SGLang 0.5.10.post1 creates the labeled
                # child only when it retracts (scheduler.py 2574: increment only
                # when len(retracted_reqs) > 0), so the count is 0, not unknown.
                # An undeclared counter stays absent (fail-closed above).
                samples = [({}, 0.0)]
        else:
            target = SGLANG_TO_VLLM.get(name, name)
        out.setdefault(target, []).extend(samples)
    if retraction_source is None and families.get(_LAZY_COUNTER_WITNESS):
        # CAGE S1 (2026-10-23, live): SGLang 0.5.10.post1 creates the counter's
        # labeled child, and with it the whole family in multiprocess mode, only
        # at its first retraction (scheduler.py 2574-2575, the one increment
        # site). The gauge sampled with no counter therefore means 0 so far.
        # Neither present stays absent.
        out.setdefault(_PREEMPT_TARGET, []).append(({}, 0.0))
    return out
