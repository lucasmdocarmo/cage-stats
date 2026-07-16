"""
Core data classes for cage_stats.

All state flowing through the application is expressed as frozen or mutable
dataclasses defined here so that every layer (metrics engine, fleet, UI) shares
the same type vocabulary.

Key types
---------
``Quantiles``
    A p50 / p90 / p99 / mean snapshot of a latency distribution.

``GpuSample``
    Per-GPU hardware counters (util%, VRAM, temp, power, clocks, fan).

``GpuSnapshot``
    Host-level GPU state: a list of ``GpuSample`` objects plus availability
    metadata.  Remote instances carry ``available=False, source="remote"``.

``Instance``
    Configuration for a single vLLM server endpoint (URL, auth, GPU mapping, …).

``FleetSnapshot``
    One timestamped poll cycle across all monitored instances.

``Snapshot``
    Complete, derived metrics for one vLLM instance at one point in time.
    Holds concurrency, throughput, latency quantiles, KV-cache state,
    speculative-decode stats, efficiency counters, and the attached GPU snapshot.

``snapshot_to_dict``
    Serialises a ``Snapshot`` to a plain dict (for ``--json`` output), grouping
    the KV fields under a nested ``"kv"`` key for readability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Quantiles:
    p50: float | None = None
    p90: float | None = None
    p99: float | None = None
    mean: float | None = None


@dataclass
class GpuSample:
    index: int
    name: str
    vendor: str = ""
    util_gpu: float | None = None
    mem_used: int | None = None
    mem_total: int | None = None
    temp_c: float | None = None
    power_w: float | None = None
    power_limit_w: float | None = None
    fan_pct: float | None = None
    fan_rpm: int | None = None
    clock_sm_mhz: int | None = None
    clock_mem_mhz: int | None = None


@dataclass
class GpuSnapshot:
    available: bool = False
    source: str = "none"
    gpus: list[GpuSample] = field(default_factory=list)
    error: str | None = None


@dataclass
class Instance:
    name: str
    url: str
    metrics_path: str = "/metrics"
    api_key: str | None = None
    gpus: tuple[int, ...] = ()
    locality: str = "local"
    logs: str | None = None


@dataclass
class FleetSnapshot:
    ts: float
    items: list[tuple[Instance, Snapshot]] = field(default_factory=list)
    gpu: GpuSnapshot = field(default_factory=GpuSnapshot)


@dataclass
class Snapshot:
    ts: float
    connected: bool
    error: str | None = None
    model_names: list[str] = field(default_factory=list)
    engine_count: int = 0
    max_num_seqs: int | None = None
    running: float = 0.0
    waiting: float = 0.0
    preempt_rate: float = 0.0
    gen_tps: float = 0.0
    prompt_tps: float = 0.0
    req_rate: float = 0.0
    tokens_per_iter: float | None = None
    session_active_s: float = 0.0
    session_idle_s: float = 0.0
    session_active_frac: float | None = None
    avg_decode_tps: float | None = None
    avg_prefill_tps: float | None = None
    session_requests: int = 0
    session_gen_tokens: float = 0.0
    session_prompt_tokens: float = 0.0
    avg_gen_tokens_per_req: float | None = None
    prefix_hit_window: float | None = None
    prefix_hit_lifetime: float | None = None
    src_compute: float | None = None
    src_cache_hit: float | None = None
    src_external: float | None = None
    # None (not 0.0) when vLLM does not expose the counter, so an ABSENT series is
    # distinguishable from a genuine zero. The CAGE sampler's numeric filter drops None,
    # preventing a fabricated 0.0 from entering vllm_telemetry.json / metrics.json.
    cached_tokens_total: float | None = None
    recomputed_tokens_total: float | None = None
    # None when NEITHER vLLM external-KV metric family is in the scrape (vLLM 0.11.0
    # exposes none) -- a fabricated False here made "connector idle" indistinguishable
    # from "metric missing" (CAGE audit 2026-07-16 SANITY-7).
    external_kv_active: bool | None = None
    # vLLM 0.11 per-phase request-time histograms (cumulative SUM/COUNT per series,
    # summed across label sets). Monotonic counters: downstream consumers (CAGE's
    # memory-pressure sweep) diff first/last samples for per-trial phase-time deltas.
    # None (not 0.0) when the series is absent, so "metric missing" never reads as zero.
    prefill_time_sum: float | None = None
    prefill_time_count: float | None = None
    decode_time_sum: float | None = None
    decode_time_count: float | None = None
    inference_time_sum: float | None = None
    inference_time_count: float | None = None
    queue_time_sum: float | None = None
    queue_time_count: float | None = None
    # Raw cumulative vllm:num_preemptions_total (the EWMA preempt_rate above hides
    # low-frequency eviction/preemption events; the raw counter makes deltas exact).
    preemptions_total: float | None = None
    kv_usage: float = 0.0
    kv_capacity_tokens: int | None = None
    kv_used_tokens: int | None = None
    kv_dtype: str | None = None
    kv_ratio: float | None = None
    kv_ratio_kind: str = "none"
    kv_fp16_equiv_tokens: int | None = None
    kv_fp16_full_ctx_gb: float | None = None
    ttft: Quantiles = field(default_factory=Quantiles)
    tpot: Quantiles = field(default_factory=Quantiles)
    e2e: Quantiles = field(default_factory=Quantiles)
    queue: Quantiles = field(default_factory=Quantiles)
    spec_active: bool = False
    spec_acceptance: float | None = None
    spec_accepted_per_draft: float | None = None
    spec_per_pos: list[float] = field(default_factory=list)
    eff_active: bool = False
    gflops: float | None = None
    gbps: float | None = None
    mfu: float | None = None
    bw_util: float | None = None
    gpu: GpuSnapshot = field(default_factory=GpuSnapshot)


def snapshot_to_dict(s: Snapshot) -> dict:
    d = asdict(s)
    d["kv"] = {
        "dtype": s.kv_dtype,
        "capacity_tokens": s.kv_capacity_tokens,
        "used_tokens": s.kv_used_tokens,
        "ratio": s.kv_ratio,
        "ratio_kind": s.kv_ratio_kind,
        "fp16_equiv_tokens": s.kv_fp16_equiv_tokens,
        "fp16_full_ctx_gb": s.kv_fp16_full_ctx_gb,
        "usage": s.kv_usage,
    }
    return d
