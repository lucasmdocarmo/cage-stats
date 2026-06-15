"""
Fleet orchestration: concurrent polling of multiple vLLM instances.

``InstanceRuntime``
    Wraps a single vLLM endpoint with all the state needed to produce a
    ``Snapshot`` on every poll cycle: an HTTP provider, a ``MetricsEngine``,
    a ``History`` ring-buffer, and a ``TeeBuffer`` for proxy / log events.

    On the first call to ``poll()`` the runtime fetches model info
    (``/v1/models``) to populate KV-cache dimension data and model name, then
    delegates all subsequent metric derivation to the engine.  Failures surface
    as a disconnected ``Snapshot`` rather than an exception.

    ``reset_session()`` clears the engine's accumulated session statistics; the
    next ``poll()`` re-baselines.  Exposed via the ``r`` keybinding in the TUI.

``Fleet``
    Owns a list of ``InstanceRuntime`` objects and polls them concurrently with
    ``asyncio.gather``.  Per-runtime exceptions are caught and converted to
    disconnected snapshots so one failing instance never brings down the others.

    GPU metrics are sliced per-instance: if an ``Instance`` declares a ``gpus``
    tuple the host ``GpuSnapshot`` is filtered to just those indices; an empty
    tuple means "show all GPUs" (single-instance behaviour).  Remote instances
    always receive ``available=False, source="remote"``.

``build_fleet``
    Convenience factory for constructing a ``Fleet`` from a list of
    ``Instance`` configs.

``slice_gpu``
    Helper that filters a ``GpuSnapshot`` to a subset of GPU indices.
"""

from __future__ import annotations

import asyncio
from typing import Any

from cage_stats.metrics.engine import MetricsEngine
from cage_stats.metrics.kv import load_model_dims
from cage_stats.metrics.parse import parse_metrics
from cage_stats.metrics.state import FleetSnapshot, GpuSnapshot, Instance, Snapshot
from cage_stats.metrics.timeseries import History
from cage_stats.providers.tee import TeeBuffer
from cage_stats.providers.vllm import VllmProvider


def slice_gpu(host: GpuSnapshot, gpus: tuple[int, ...]) -> GpuSnapshot:
    if not host.available:
        return GpuSnapshot(available=False, source=host.source)
    if not gpus:
        return host
    want = set(gpus)
    sub = [g for g in host.gpus if g.index in want]
    return GpuSnapshot(available=bool(sub), source=host.source, gpus=sub)


class InstanceRuntime:
    def __init__(self, instance: Instance, *, provider: Any = None) -> None:
        self.instance = instance
        self._provider: Any = provider or VllmProvider(
            base_url=instance.url,
            metrics_path=instance.metrics_path,
            api_key=instance.api_key,
        )
        self._engine = MetricsEngine()
        self.history: History = History()
        self.tee: TeeBuffer = TeeBuffer()
        self.snapshot: Snapshot | None = None
        self.model_names: list[str] = []
        self._dims_loaded = False

    async def _ensure_dims(self) -> None:
        if self._dims_loaded:
            return
        self._dims_loaded = True
        info = await self._provider.fetch_model_info()
        md = load_model_dims(info.root, info.max_model_len)
        self._engine = MetricsEngine(dims=md.dims, max_model_len=md.max_model_len)
        self.model_names = info.model_names

    async def poll(self, now: float) -> Snapshot:
        await self._ensure_dims()
        raw = await self._provider.fetch_metrics()
        if raw.fetched_ok and raw.text:
            snap = self._engine.derive(parse_metrics(raw.text), now=now)
        else:
            prev = self.snapshot
            snap = prev if prev is not None else Snapshot(ts=now, connected=False, error=raw.error)
            snap.connected = False
            snap.error = raw.error
        self.snapshot = snap
        self._push_history(snap)
        return snap

    def _push_history(self, s: Snapshot) -> None:
        self.history.push("running", s.running)
        self.history.push("waiting", s.waiting)
        self.history.push("gen_tps", s.gen_tps)
        self.history.push("prompt_tps", s.prompt_tps)
        if s.prefix_hit_window is not None:
            self.history.push("prefix_hit", s.prefix_hit_window)

    def reset_session(self) -> None:
        self._engine.reset_session()

    async def aclose(self) -> None:
        await self._provider.aclose()


class Fleet:
    def __init__(
        self,
        instances: list[Instance],
        *,
        runtimes: list[InstanceRuntime] | None = None,
    ) -> None:
        self.runtimes: list[InstanceRuntime] = (
            runtimes if runtimes is not None else [InstanceRuntime(i) for i in instances]
        )

    async def poll(self, host_gpu: GpuSnapshot, now: float) -> FleetSnapshot:
        results: list[Any] = list(
            await asyncio.gather(*(rt.poll(now) for rt in self.runtimes), return_exceptions=True)
        )
        items: list[tuple[Instance, Snapshot]] = []
        for rt, res in zip(self.runtimes, results, strict=True):
            if isinstance(res, BaseException):
                prev = rt.snapshot
                res = (
                    prev if prev is not None else Snapshot(ts=now, connected=False, error=str(res))
                )
                res.connected = False
            if rt.instance.locality == "local":
                res.gpu = slice_gpu(host_gpu, rt.instance.gpus)
            else:
                res.gpu = GpuSnapshot(available=False, source="remote")
            items.append((rt.instance, res))
        return FleetSnapshot(ts=now, items=items, gpu=host_gpu)

    async def aclose(self) -> None:
        await asyncio.gather(*(rt.aclose() for rt in self.runtimes), return_exceptions=True)


def build_fleet(instances: list[Instance]) -> Fleet:
    return Fleet(instances)
