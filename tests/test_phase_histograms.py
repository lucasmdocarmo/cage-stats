"""vLLM 0.11 per-phase request-time histograms + raw preemption counter.

The CAGE memory-pressure sweep reads the mechanism (prefix-cache eviction under
shrinking KV capacity) through the phase decomposition: cumulative
``vllm:request_{prefill,decode,inference,queue}_time_seconds`` SUM/COUNT pairs and
the raw ``vllm:num_preemptions_total`` counter. These must surface in the snapshot
(and thus ``snapshot_to_dict``) as monotonic values so per-trial deltas are exact,
and must be None -- never a fabricated 0.0 -- when the series is absent.
"""

import time

from cage_stats.metrics.engine import MetricsEngine
from cage_stats.metrics.parse import parse_metrics
from cage_stats.metrics.state import snapshot_to_dict

BASE = 'vllm:num_requests_running{model_name="m",engine="0"} 0.0\n'


def _hist(base: str, *, sum_v: float, count_v: float, engine: str = "0") -> str:
    lbl = f'model_name="m",engine="{engine}"'
    return (
        f"# TYPE {base} histogram\n"
        f'{base}_bucket{{le="1.0",{lbl}}} {count_v - 1}\n'
        f'{base}_bucket{{le="+Inf",{lbl}}} {count_v}\n'
        f"{base}_sum{{{lbl}}} {sum_v}\n"
        f"{base}_count{{{lbl}}} {count_v}\n"
    )


FULL = (
    BASE
    + _hist("vllm:request_prefill_time_seconds", sum_v=12.5, count_v=5.0)
    + _hist("vllm:request_decode_time_seconds", sum_v=40.0, count_v=5.0)
    + _hist("vllm:request_inference_time_seconds", sum_v=52.5, count_v=5.0)
    + _hist("vllm:request_queue_time_seconds", sum_v=0.75, count_v=5.0)
    + 'vllm:num_preemptions_total{model_name="m",engine="0"} 7.0\n'
)


def _derive(text: str):
    return MetricsEngine().derive(parse_metrics(text), now=time.time())


def test_phase_histogram_sums_and_counts_exposed():
    snap = _derive(FULL)
    assert snap.prefill_time_sum == 12.5
    assert snap.prefill_time_count == 5.0
    assert snap.decode_time_sum == 40.0
    assert snap.decode_time_count == 5.0
    assert snap.inference_time_sum == 52.5
    assert snap.inference_time_count == 5.0
    assert snap.queue_time_sum == 0.75
    assert snap.queue_time_count == 5.0


def test_preemptions_total_raw_counter_exposed():
    assert _derive(FULL).preemptions_total == 7.0


def test_absent_series_are_none_not_zero():
    # A scrape without the phase histograms / preemption counter must yield None so
    # "metric missing" is never recorded as a genuine zero downstream.
    snap = _derive(BASE)
    for field in (
        "prefill_time_sum",
        "prefill_time_count",
        "decode_time_sum",
        "decode_time_count",
        "inference_time_sum",
        "inference_time_count",
        "queue_time_sum",
        "queue_time_count",
        "preemptions_total",
    ):
        assert getattr(snap, field) is None, field


def test_phase_histograms_summed_across_label_sets():
    text = (
        BASE
        + _hist("vllm:request_prefill_time_seconds", sum_v=10.0, count_v=4.0, engine="0")
        + _hist("vllm:request_prefill_time_seconds", sum_v=2.5, count_v=1.0, engine="1")
    )
    snap = _derive(text)
    assert snap.prefill_time_sum == 12.5
    assert snap.prefill_time_count == 5.0


def test_snapshot_dict_carries_phase_fields():
    d = snapshot_to_dict(_derive(FULL))
    assert d["prefill_time_sum"] == 12.5
    assert d["queue_time_count"] == 5.0
    assert d["preemptions_total"] == 7.0
