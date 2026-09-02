"""Prospective KV-transfer connector capture (T4.5).

WHAT: pins ``Snapshot.transfer_counters`` — the engine's capture of every
Prometheus family matching ``^vllm:(kv_transfer|nixl|kv_connector)`` or the
doc-recorded bare ``nixl_`` prefix (docs/VLLM_COMPATIBILITY.md §8.2/§8.4/§8.5),
as ``{family_name: value summed across label sets}``.

WHY: contrasts #18 (distribution's transfer price) and #19 (dedup over the
wire) need REAL connector counters, but the exact family names at the pinned
(vLLM, NIXL wheel, UCX) triple are a live-only fact
[VERIFY-LIVE at Run-C-prime preflight]. So the capture is PROSPECTIVE:
capture-what-exists, verbatim names, and NEVER fabricate. Pinned here:

* ABSENT: no matching family in the scrape -> ``transfer_counters is None``,
  never ``{}`` and never a fabricated entry (E2b absence doctrine, the
  ``preemptions_total`` / raw prefix-counter precedent);
* PRESENT: families surface under their scraped names, summed across label
  sets like every other counter in the snapshot; a genuine 0.0 survives;
* BOUNDARIES: the match is a case-sensitive PREFIX match —
  ``vllm:kv_transfer_bytes_total`` in, ``vllm:kv_cache_usage_perc`` out;
* SERIALIZATION: ``snapshot_to_dict`` carries the dict / the None;
* SGLANG NEGATIVE PIN: the dialect translation maps NOTHING into these
  families (no SGLang transfer metric is documented) — a translated
  fabrication would poison contrast #18 with counterfeit transfer data.
"""

from __future__ import annotations

from cage_stats.metrics.engine import MetricsEngine
from cage_stats.metrics.parse import parse_metrics
from cage_stats.metrics.sglang_dialect import (
    SGLANG_TO_VLLM,
    translate_sglang_families,
)
from cage_stats.metrics.state import snapshot_to_dict

BASE = 'vllm:num_requests_running{model_name="m",engine="0"} 0.0\n'

WITH_TRANSFER = (
    BASE
    + 'vllm:kv_transfer_bytes_total{model_name="m",engine="0"} 4096.0\n'
    + 'vllm:nixl_transfers_total{model_name="m",engine="0"} 7.0\n'
    + 'vllm:kv_connector_send_seconds_sum{model_name="m",engine="0"} 1.25\n'
)


def _derive(text: str):
    return MetricsEngine().derive(parse_metrics(text), now=0.0)


# ---------------------------------------------------------------------------
# Absence stays absence
# ---------------------------------------------------------------------------

def test_none_when_no_transfer_family_in_scrape():
    # Today's real scrapes look like this (only a simulator produces transfer
    # data): "connector metrics missing" must surface as None, not {} and not
    # a fabricated zero entry — else contrast #18 reads "connector idle".
    snap = _derive(BASE)
    assert snap.transfer_counters is None


def test_all_three_vllm_prefixes_captured_verbatim():
    snap = _derive(WITH_TRANSFER)
    assert snap.transfer_counters == {
        "vllm:kv_transfer_bytes_total": 4096.0,
        "vllm:nixl_transfers_total": 7.0,
        "vllm:kv_connector_send_seconds_sum": 1.25,
    }


def test_genuine_zero_counter_survives_as_zero_entry():
    # A REAL 0.0 counter (connector up, nothing transferred yet) is data, not
    # absence: the dict exists and carries 0.0.
    text = BASE + 'vllm:kv_transfer_bytes_total{model_name="m"} 0.0\n'
    snap = _derive(text)
    assert snap.transfer_counters == {"vllm:kv_transfer_bytes_total": 0.0}
    assert snap.transfer_counters is not None


def test_doc_recorded_bare_nixl_prefix_captured():
    # docs/VLLM_COMPATIBILITY.md §8.4's presence check is `grep nixl_` — the
    # doc records the BARE prefix, so the unprefixed spelling is captured too.
    # [VERIFY-LIVE at Run-C-prime preflight]: exact names on the pinned triple.
    text = BASE + 'nixl_transfer_bytes_total{model_name="m"} 512.0\n'
    snap = _derive(text)
    assert snap.transfer_counters == {"nixl_transfer_bytes_total": 512.0}


# ---------------------------------------------------------------------------
# Summing across label sets
# ---------------------------------------------------------------------------

def test_summed_across_label_sets():
    # Data-parallel / multi-rail scrape: aggregate the same way every other
    # counter in the snapshot does, so per-window deltas stay comparable to
    # the token totals.
    text = (
        BASE
        + 'vllm:kv_transfer_bytes_total{model_name="m",engine="0"} 100.0\n'
        + 'vllm:kv_transfer_bytes_total{model_name="m",engine="1"} 40.0\n'
    )
    snap = _derive(text)
    assert snap.transfer_counters == {"vllm:kv_transfer_bytes_total": 140.0}


# ---------------------------------------------------------------------------
# Prefix-match boundaries
# ---------------------------------------------------------------------------

def test_prefix_boundaries_exclude_non_transfer_kv_families():
    # `vllm:kv_cache_usage_perc` shares the `vllm:kv_` stem but is occupancy,
    # not transfer — it must NOT leak into the capture (nor any other family).
    text = (
        WITH_TRANSFER
        + 'vllm:kv_cache_usage_perc{model_name="m",engine="0"} 0.5\n'
        + 'vllm:prefix_cache_queries_total{model_name="m",engine="0"} 10.0\n'
    )
    snap = _derive(text)
    assert snap.transfer_counters is not None
    assert "vllm:kv_cache_usage_perc" not in snap.transfer_counters
    assert "vllm:prefix_cache_queries_total" not in snap.transfer_counters
    assert set(snap.transfer_counters) == {
        "vllm:kv_transfer_bytes_total",
        "vllm:nixl_transfers_total",
        "vllm:kv_connector_send_seconds_sum",
    }
    # the excluded families still flow through their own snapshot fields
    assert snap.kv_usage == 0.5
    assert snap.prefix_cache_queries_total == 10.0


def test_prefix_match_is_case_sensitive():
    # Prometheus names are case-sensitive; a case-mangled lookalike is NOT one
    # of the known connector families and matching it would claim knowledge we
    # do not have.
    text = BASE + 'vllm:KV_TRANSFER_bytes_total{model_name="m"} 9.0\n'
    assert _derive(text).transfer_counters is None


def test_prefix_must_anchor_at_name_start():
    # `kv_transfer` appearing mid-name (wrong engine prefix) must not match:
    # the anchor is the vllm:/nixl_ spelling, not the substring.
    text = BASE + 'other:kv_transfer_bytes_total{model_name="m"} 9.0\n'
    assert _derive(text).transfer_counters is None


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def test_snapshot_dict_carries_transfer_counters():
    d = snapshot_to_dict(_derive(WITH_TRANSFER))
    assert d["transfer_counters"] == {
        "vllm:kv_transfer_bytes_total": 4096.0,
        "vllm:nixl_transfers_total": 7.0,
        "vllm:kv_connector_send_seconds_sum": 1.25,
    }
    d_absent = snapshot_to_dict(_derive(BASE))
    assert d_absent["transfer_counters"] is None


# ---------------------------------------------------------------------------
# SGLang dialect negative pin
# ---------------------------------------------------------------------------

_SGLANG_TEXT = (
    'sglang:token_usage{model_name="m"} 0.5\n'
    'sglang:num_running_reqs{model_name="m"} 1.0\n'
    'sglang:prompt_tokens_total{model_name="m"} 100.0\n'
    'sglang:generation_tokens_total{model_name="m"} 50.0\n'
)


def test_sglang_translation_table_maps_nothing_into_transfer_families():
    # Structural pin: no documented SGLang name translates into a family the
    # transfer capture would match. No SGLang transfer metric is documented,
    # so ANY such mapping would be a fabricated connector counter.
    from cage_stats.metrics.engine import _TRANSFER_DOC_PREFIXES, _TRANSFER_FAMILY_RE

    # BOTH dialect translation paths are pinned: the direct rename map AND the
    # retraction-candidate mapping target (2026-09-01 verifier minor — the
    # second path was uncovered).
    for target in list(SGLANG_TO_VLLM.values()) + ["vllm:num_preemptions_total"]:
        assert not _TRANSFER_FAMILY_RE.match(target), target
        assert not target.startswith(_TRANSFER_DOC_PREFIXES), target


def test_sglang_scrape_end_to_end_yields_none():
    # parse -> translate -> derive on a documented SGLang scrape: no transfer
    # family may appear, so transfer_counters must be None end-to-end.
    fam = translate_sglang_families(parse_metrics(_SGLANG_TEXT))
    snap = MetricsEngine().derive(fam, now=0.0)
    assert snap.transfer_counters is None


def test_sglang_prefixed_lookalike_passes_through_uncaptured():
    # Even a transfer-LOOKING sglang name (no such metric is documented, but a
    # fork could invent one) passes through the dialect untranslated and must
    # NOT be captured: capturing it would attribute vLLM-connector semantics
    # to an unknown quantity and poison contrast #18.
    fam = translate_sglang_families(
        parse_metrics(_SGLANG_TEXT + 'sglang:kv_transfer_bytes{model_name="m"} 7.0\n')
    )
    assert "sglang:kv_transfer_bytes" in fam  # passthrough preserved it...
    snap = MetricsEngine().derive(fam, now=0.0)
    assert snap.transfer_counters is None  # ...but capture refused it
