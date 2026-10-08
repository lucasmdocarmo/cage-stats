"""ADR-0148 Batch C (CAGE, 2026-10-08): absence is None, never 0.0.

Pinned here:

1. The concurrency gauges (running, waiting) and the four rates (gen_tps,
   prompt_tps, req_rate, preempt_rate) are None when their family is absent
   from the scrape, and a family that reads 0 stays 0.0. Before this the
   engine read ``sum_value(...) or 0.0`` and fed the rate trackers a constant
   zero, so a missing counter produced a fabricated 0.0 rate on every tick.
2. A rate divides the counter delta by the gap between the two ``now``
   instants it was derived at.
3. The SGLang request counter feeds ``req_rate`` through the dialect map;
   without it the rate is None, not 0.
4. The session accounting does not advance on a tick without token counters
   and attributes no time on a tick without the running gauge.
5. The text dashboard and the fleet table render a snapshot with None fields
   without raising and without printing the word None.
6. ``api.fetch_snapshot`` derives the two polls at their wall-clock instants:
   ``ts`` is the second poll's epoch time and the rate uses the measured gap.
   The mock path keeps all its numbers.

No server, no network.
"""

from __future__ import annotations

import time

import pytest

from cage_stats import api
from cage_stats.metrics.engine import MetricsEngine
from cage_stats.metrics.parse import parse_metrics
from cage_stats.metrics.sglang_dialect import translate_sglang_families
from cage_stats.metrics.state import FleetSnapshot, Instance, Snapshot
from cage_stats.providers.vllm import ModelInfo, RawText
from cage_stats.ui import render
from cage_stats.ui.text import render_dashboard

L = 'model_name="m",engine="0"'


def _text(*, running=None, waiting=None, gen=None, prompt=None, req=None, preempt=None) -> str:
    lines = ["# TYPE vllm:kv_cache_usage_perc gauge", f"vllm:kv_cache_usage_perc{{{L}}} 0.4"]
    for name, kind, value in (
        ("vllm:num_requests_running", "gauge", running),
        ("vllm:num_requests_waiting", "gauge", waiting),
        ("vllm:generation_tokens_total", "counter", gen),
        ("vllm:prompt_tokens_total", "counter", prompt),
        ("vllm:request_success_total", "counter", req),
        ("vllm:num_preemptions_total", "counter", preempt),
    ):
        if value is not None:
            lines += [f"# TYPE {name} {kind}", f"{name}{{{L}}} {value}"]
    return "\n".join(lines) + "\n"


def _derive_twice(text0: str, text1: str, *, t0: float = 0.0, t1: float = 1.0) -> Snapshot:
    eng = MetricsEngine()
    eng.derive(parse_metrics(text0), now=t0)
    return eng.derive(parse_metrics(text1), now=t1)


# 1. absence is None, a measured zero stays zero -------------------------------


def test_absent_gauges_and_counters_are_none():
    snap = _derive_twice(_text(), _text())
    for field in ("running", "waiting", "gen_tps", "prompt_tps", "req_rate", "preempt_rate"):
        assert getattr(snap, field) is None, field
    assert snap.preemptions_total is None
    assert snap.kv_usage == pytest.approx(0.4)  # the gauge that was scraped stays a number


def test_present_zero_stays_zero():
    zero = _text(running=0.0, waiting=0.0, gen=0.0, prompt=0.0, req=0.0, preempt=0.0)
    snap = _derive_twice(zero, zero)
    for field in ("running", "waiting", "gen_tps", "prompt_tps", "req_rate", "preempt_rate"):
        value = getattr(snap, field)
        assert value == 0.0 and value is not None, field


# 2. the rate uses the gap between the two instants ----------------------------


def test_rates_divide_by_the_measured_gap():
    snap = _derive_twice(
        _text(running=1.0, gen=100.0, prompt=50.0, req=4.0, preempt=0.0),
        _text(running=1.0, gen=160.0, prompt=80.0, req=5.0, preempt=2.0),
        t0=10.0, t1=10.5,
    )
    assert snap.gen_tps == pytest.approx(120.0)
    assert snap.prompt_tps == pytest.approx(60.0)
    assert snap.req_rate == pytest.approx(2.0)
    assert snap.preempt_rate == pytest.approx(4.0)
    assert snap.running == 1.0


# 3. the SGLang request family ---------------------------------------------------


def _sglang(*, requests: float | None) -> str:
    lbl = 'model_name="m"'
    lines = [
        "# TYPE sglang:token_usage gauge", f"sglang:token_usage{{{lbl}}} 0.5",
        "# TYPE sglang:num_running_reqs gauge", f"sglang:num_running_reqs{{{lbl}}} 2.0",
        "# TYPE sglang:prompt_tokens_total counter", f"sglang:prompt_tokens_total{{{lbl}}} 100.0",
        "# TYPE sglang:generation_tokens_total counter",
        f"sglang:generation_tokens_total{{{lbl}}} 50.0",
    ]
    if requests is not None:
        lines += ["# TYPE sglang:num_requests_total counter",
                  f"sglang:num_requests_total{{{lbl}}} {requests}"]
    return "\n".join(lines) + "\n"


def test_sglang_request_counter_feeds_req_rate_and_absence_stays_none():
    eng = MetricsEngine()
    eng.derive(translate_sglang_families(parse_metrics(_sglang(requests=10.0))), now=0.0)
    snap = eng.derive(translate_sglang_families(parse_metrics(_sglang(requests=14.0))), now=2.0)
    assert snap.req_rate == pytest.approx(2.0)
    assert snap.running == 2.0
    eng2 = MetricsEngine()
    eng2.derive(translate_sglang_families(parse_metrics(_sglang(requests=None))), now=0.0)
    snap2 = eng2.derive(translate_sglang_families(parse_metrics(_sglang(requests=None))), now=1.0)
    assert snap2.req_rate is None
    assert snap2.gen_tps == 0.0                 # the token counters were scraped, unchanged


# 4. the session accounting ------------------------------------------------------


def test_session_accounting_pauses_without_counters_and_running():
    eng = MetricsEngine()
    full = _text(running=1.0, gen=100.0, prompt=50.0, req=1.0)
    eng.derive(parse_metrics(full), now=0.0)                      # baseline
    paused = eng.derive(parse_metrics(_text(running=1.0)), now=1.0)
    assert paused.session_active_s == 0.0 and paused.session_requests == 0
    later = eng.derive(parse_metrics(_text(running=1.0, gen=200.0, prompt=100.0, req=2.0)), now=2.0)
    assert later.session_active_s == pytest.approx(2.0)          # 0 -> 2 while running
    assert later.avg_decode_tps == pytest.approx(50.0)           # 100 tokens over 2 s
    assert later.session_requests == 1
    unknown = eng.derive(parse_metrics(_text(gen=260.0, prompt=130.0, req=3.0)), now=3.0)
    assert unknown.running is None
    assert unknown.session_active_s == pytest.approx(2.0)        # no attribution for that second
    assert unknown.session_idle_s == 0.0
    assert unknown.session_gen_tokens == pytest.approx(160.0)    # the totals still count


# 5. the renderers -----------------------------------------------------------------


def test_dashboard_and_fleet_render_none_fields_without_raising():
    snap = Snapshot(ts=1.0, connected=True)
    text = render_dashboard(snap, url="http://x")
    assert "CONCURRENCY" in text and "THROUGHPUT" in text
    assert "None" not in text
    fleet = FleetSnapshot(ts=1.0, items=[(Instance(name="a", url="http://x"), snap)])
    table = render.fleet_overview(fleet, selected=0)
    assert "None" not in table and "a" in table


# 6. the api path --------------------------------------------------------------------


class _FakeProvider:
    texts: list[str] = []
    times: list[float] = []

    def __init__(self, **_kw) -> None:
        pass

    async def fetch_model_info(self) -> ModelInfo:
        return ModelInfo(model_names=["m"], max_model_len=None, root=None)

    async def fetch_metrics(self) -> RawText:
        _FakeProvider.times.append(time.time())
        return RawText(text=_FakeProvider.texts[len(_FakeProvider.times) - 1], fetched_ok=True)

    async def aclose(self) -> None:
        return None


def test_api_derives_at_the_measured_instants(monkeypatch):
    import cage_stats.providers.vllm as vllm_provider

    _FakeProvider.texts = [
        _text(running=1.0, gen=100.0, prompt=50.0, req=1.0),
        _text(running=1.0, gen=110.0, prompt=55.0, req=2.0),
    ]
    _FakeProvider.times = []
    monkeypatch.setattr(vllm_provider, "VllmProvider", _FakeProvider)
    snap = api.fetch_snapshot("http://x", interval=0.05)
    t0, t1 = _FakeProvider.times
    assert snap.ts == pytest.approx(t1, abs=0.02)
    assert snap.gen_tps == pytest.approx(10.0 / (t1 - t0), rel=0.05)
    assert snap.req_rate == pytest.approx(1.0 / (t1 - t0), rel=0.05)


def test_mock_path_keeps_its_numbers():
    snap = api.fetch_snapshot(mock=True)
    for field in ("running", "waiting", "gen_tps", "prompt_tps", "req_rate"):
        assert getattr(snap, field) is not None, field
