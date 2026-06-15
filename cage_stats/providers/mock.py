"""
Mock provider for ``--mock`` mode and unit testing.

Generates deterministic, sinusoidally-varying Prometheus metrics so the full
dashboard (including braille plots and sparklines) can be exercised without a
live vLLM server.

``MockProvider``
    Produces Prometheus text with tick-varying counters and gauges.  Each call
    to ``metrics_text()`` advances an internal tick counter and returns a fresh
    body covering: concurrency, throughput, cache queries/hits, KV usage, token
    sources, iteration stats, speculative-decode counters, TTFT / TPOT latency
    histograms, and placeholder efficiency counters.

``MockVllmProvider``
    Thin async wrapper around ``MockProvider`` that satisfies the same interface
    as ``VllmProvider``, making it a drop-in replacement inside
    ``InstanceRuntime``.

``mock_fleet(names)``
    Convenience factory for building a multi-instance test fleet.  Each runtime
    gets its own ``MockProvider`` so the sinusoidal patterns are out of phase.

``mock_gpu_snapshot(tick)``
    Two synthetic NVIDIA RTX 4090 GPUs with tick-driven utilisation to populate
    the GPU panel in ``--mock`` mode.
"""

from __future__ import annotations

import math

from cage_stats.fleet.fleet import Fleet, InstanceRuntime
from cage_stats.metrics.state import GpuSample, GpuSnapshot, Instance
from cage_stats.providers.vllm import ModelInfo, RawText

_M = "mock-7b"


def mock_gpu_snapshot(tick: int = 0) -> GpuSnapshot:
    def _g(i: int, name: str, total: int) -> GpuSample:
        util = 50.0 + 40.0 * (math.sin((tick + i * 3) / 4.0) + 1.0) / 2.0
        return GpuSample(
            index=i,
            name=name,
            util_gpu=round(util),
            mem_used=int(total * 0.86),
            mem_total=total,
            temp_c=58.0 + i * 3,
            power_w=140.0 + i * 20,
            power_limit_w=300.0,
            fan_pct=42.0 + i * 5,
            clock_sm_mhz=2520,
            clock_mem_mhz=9501,
        )

    return GpuSnapshot(
        available=True,
        source="mock",
        gpus=[
            _g(0, "NVIDIA RTX 4090", 24_000_000_000),
            _g(1, "NVIDIA RTX 4090", 24_000_000_000),
        ],
    )


_E = 'engine="0",model_name="mock-7b"'


class MockProvider:
    def __init__(self) -> None:
        self._tick = 0
        self._gen = 1_000_000.0
        self._prompt = 3_000_000.0
        self._q = 5_000_000.0
        self._h = 1_500_000.0

    def metrics_text(self) -> str:
        self._tick += 1
        t = self._tick
        running = 2 + int(2 * (math.sin(t / 3) + 1))
        waiting = max(0, int(3 * math.sin(t / 5)))
        self._gen += 120 + 40 * math.sin(t / 2)
        self._prompt += 300 + 80 * math.cos(t / 2)
        self._q += 1000
        self._h += 400 + 50 * math.sin(t / 4)
        kv = 0.10 + 0.05 * (math.sin(t / 6) + 1)
        ttft_buckets = self._hist("vllm:time_to_first_token_seconds", base=0.05, n=t)
        tpot_buckets = self._hist("vllm:request_time_per_output_token_seconds", base=0.01, n=t)
        e = _E
        src = "vllm:prompt_tokens_by_source_total"
        cc = (
            "vllm:cache_config_info{"
            'block_size="16",cache_dtype="fp8_e4m3",'
            'num_gpu_blocks="20000",enable_prefix_caching="True",engine="0"'
            "} 1.0"
        )
        return (
            f"# TYPE vllm:num_requests_running gauge\n"
            f"vllm:num_requests_running{{{e}}} {running}.0\n"
            f"# TYPE vllm:num_requests_waiting gauge\n"
            f"vllm:num_requests_waiting{{{e}}} {waiting}.0\n"
            f"# TYPE vllm:num_preemptions_total counter\n"
            f"vllm:num_preemptions_total{{{e}}} 0.0\n"
            f"# TYPE vllm:generation_tokens_total counter\n"
            f"vllm:generation_tokens_total{{{e}}} {self._gen:.1f}\n"
            f"# TYPE vllm:prompt_tokens_total counter\n"
            f"vllm:prompt_tokens_total{{{e}}} {self._prompt:.1f}\n"
            f"# TYPE vllm:request_success_total counter\n"
            f"vllm:request_success_total{{{e}}} {t * 4}.0\n"
            f"# TYPE vllm:prefix_cache_queries_total counter\n"
            f"vllm:prefix_cache_queries_total{{{e}}} {self._q:.1f}\n"
            f"# TYPE vllm:prefix_cache_hits_total counter\n"
            f"vllm:prefix_cache_hits_total{{{e}}} {self._h:.1f}\n"
            f"# TYPE vllm:prompt_tokens_cached_total counter\n"
            f"vllm:prompt_tokens_cached_total{{{e}}} {self._h:.1f}\n"
            f"# TYPE vllm:prompt_tokens_recomputed_total counter\n"
            f"vllm:prompt_tokens_recomputed_total{{{e}}} 12.0\n"
            f"# TYPE vllm:prompt_tokens_by_source_total counter\n"
            f'{src}{{{e},source="local_compute"}} {self._prompt * 0.7:.1f}\n'
            f'{src}{{{e},source="local_cache_hit"}} {self._prompt * 0.3:.1f}\n'
            f'{src}{{{e},source="external_kv_transfer"}} 0.0\n'
            f"# TYPE vllm:kv_cache_usage_perc gauge\n"
            f"vllm:kv_cache_usage_perc{{{e}}} {kv:.4f}\n"
            f"# TYPE vllm:cache_config_info gauge\n"
            f"{cc}\n"
            f"# TYPE vllm:iteration_tokens_total histogram\n"
            f"vllm:iteration_tokens_total_sum{{{e}}} {t * 1024}.0\n"
            f"vllm:iteration_tokens_total_count{{{e}}} {t}.0\n"
            f"# TYPE vllm:spec_decode_num_drafts_total counter\n"
            f"vllm:spec_decode_num_drafts_total{{{e}}} {t * 100}.0\n"
            f"# TYPE vllm:spec_decode_num_draft_tokens_total counter\n"
            f"vllm:spec_decode_num_draft_tokens_total{{{e}}} {t * 500}.0\n"
            f"# TYPE vllm:spec_decode_num_accepted_tokens_total counter\n"
            f"vllm:spec_decode_num_accepted_tokens_total{{{e}}} {t * 210}.0\n"
            f"# TYPE vllm:estimated_flops_per_gpu_total counter\n"
            f"vllm:estimated_flops_per_gpu_total{{{e}}} 0.0\n"
            f"{ttft_buckets}\n"
            f"{tpot_buckets}\n"
        )

    def _hist(self, name: str, *, base_le: float = 0.05, n: int = 1, **kw) -> str:
        base_le = kw.get("base", base_le)
        les = [base_le * m for m in (1, 2, 4, 8, 16, 32)]
        lines = [f"# TYPE {name} histogram"]
        cum = 0.0
        for le in les:
            cum += n
            lines.append(f'{name}_bucket{{le="{le}"}} {cum:.1f}')
        lines.append(f'{name}_bucket{{le="+Inf"}} {cum + n:.1f}')
        lines.append(f"{name}_count {cum + n:.1f}")
        lines.append(f"{name}_sum {cum * base_le:.3f}")
        return "\n".join(lines)


class MockVllmProvider:
    def __init__(self, mock: MockProvider) -> None:
        self._mock = mock

    async def fetch_metrics(self) -> RawText:
        return RawText(text=self._mock.metrics_text(), fetched_ok=True)

    async def fetch_model_info(self) -> ModelInfo:
        return ModelInfo(model_names=["mock-model"], max_model_len=None, root=None)

    async def aclose(self) -> None:
        pass


def mock_fleet(
    names: tuple[str, ...] = ("qwen-30b", "llama-70b", "mixtral"),
) -> Fleet:
    rts = []
    for i, name in enumerate(names):
        inst = Instance(
            name=name,
            url=f"http://localhost:{8000 + i}",
            gpus=(i,),
            locality="local",
        )
        rts.append(InstanceRuntime(inst, provider=MockVllmProvider(MockProvider())))
    return Fleet([], runtimes=rts)
