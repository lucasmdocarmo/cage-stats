"""
Command-line entry point and top-level orchestration.

``main(argv, env)``
    The single entry point exposed by the ``cage_stats`` console script.

    1. Parses CLI arguments and environment variables into a ``Config`` object.
    2. Calls ``resolve_instances`` to fill ``cfg.instances`` from TOML config
       files, Docker discovery, and ``--url`` flags.
    3. Dispatches to one of three execution paths:
       - ``--once --json``: single-poll snapshot printed as JSON, then exits.
       - TUI mode: launches the interactive ``CageStatsApp`` dashboard.

``resolve_instances(cfg, env)``
    Mutates ``cfg.instances`` in-place by merging all instance sources:
    TOML config file (searched via ``find_config``), ``--discover-docker``,
    and CLI ``--url`` flags.  Also propagates TOML global keys (``interval``,
    ``gpu``) into the config when the corresponding CLI flag is at its default.

``run_once_json(cfg)``
    For a single instance: fetches metrics twice (with a short sleep between
    samples) to prime the EWMA rates, then dumps the resulting ``Snapshot``
    as JSON.  Delegates to ``_run_once_fleet`` for multi-instance configs.

``_run_once_fleet(cfg)``
    Async variant that polls all instances concurrently and emits a JSON array
    of per-instance snapshots.
"""

from __future__ import annotations

import json
import os
import sys
import time

from cage_stats.config.config import Config, find_config, load_config
from cage_stats.metrics.engine import MetricsEngine
from cage_stats.metrics.kv import load_model_dims
from cage_stats.metrics.parse import parse_metrics
from cage_stats.metrics.state import snapshot_to_dict
from cage_stats.providers.mock import MockProvider


def resolve_instances(cfg: Config, env: dict[str, str]) -> Config:
    from cage_stats.fleet.resolve import instance_from_dict, local_hostnames, resolve_fleet
    from cage_stats.providers.docker import discover_docker

    local_names = local_hostnames()
    config_instances = []
    config_globals: dict = {}
    path = find_config(cfg.config_path, env)
    if path:
        try:
            raw, config_globals = load_config(path)
            config_instances = [
                instance_from_dict(
                    r,
                    defaults_api_key=cfg.api_key,
                    defaults_metrics_path=cfg.metrics_path,
                    local_names=local_names,
                )
                for r in raw
            ]
        except (OSError, ValueError) as e:
            print(f"cage_stats: ignoring config {path}: {e}", file=sys.stderr)
    interval = config_globals.get("interval")
    if isinstance(interval, bool):
        interval = None
    if cfg.interval == 1.0 and isinstance(interval, (int, float)):
        cfg.interval = float(interval)
    gpu = config_globals.get("gpu")
    if cfg.gpu is True and isinstance(gpu, bool):
        cfg.gpu = gpu
    docker_instances = discover_docker() if cfg.discover_docker else []
    cfg.instances = resolve_fleet(
        config_instances,
        docker_instances,
        cfg.urls,
        defaults_api_key=cfg.api_key,
        defaults_metrics_path=cfg.metrics_path,
        local_names=local_names,
    )
    return cfg


def _poll_fleet(cfg: Config):
    """Poll all instances twice (priming rates) and return a FleetSnapshot."""
    import asyncio

    from cage_stats.fleet.fleet import Fleet, InstanceRuntime
    from cage_stats.metrics.state import GpuSnapshot

    async def go():
        if cfg.mock:
            from cage_stats.providers.mock import MockVllmProvider

            rts = [
                InstanceRuntime(i, provider=MockVllmProvider(MockProvider())) for i in cfg.instances
            ]
        else:
            rts = [InstanceRuntime(i) for i in cfg.instances]
        fleet = Fleet([], runtimes=rts)
        await fleet.poll(GpuSnapshot(), 0.0)
        time.sleep(min(cfg.interval, 1.0))
        fs = await fleet.poll(GpuSnapshot(), 1.0)
        await fleet.aclose()
        return fs

    return asyncio.run(go())


def _run_once_fleet(cfg: Config) -> int:
    fs = _poll_fleet(cfg)
    out = [
        {
            "name": inst.name,
            "url": inst.url,
            "locality": inst.locality,
            "snapshot": snapshot_to_dict(snap),
        }
        for inst, snap in fs.items
    ]
    print(json.dumps(out, default=str))
    return 0


def run_once_json(cfg: Config) -> int:
    if len(cfg.instances) > 1:
        return _run_once_fleet(cfg)
    if cfg.mock:
        eng = MetricsEngine(dims=None, max_model_len=None)
        mp = MockProvider()
        eng.derive(parse_metrics(mp.metrics_text()), now=0.0)
        snap = eng.derive(parse_metrics(mp.metrics_text()), now=1.0)
    else:
        import asyncio

        from cage_stats.providers.vllm import VllmProvider

        async def _go():
            inst = cfg.instances[0] if cfg.instances else None
            url = inst.url if inst else cfg.url
            metrics_path = inst.metrics_path if inst else cfg.metrics_path
            api_key = inst.api_key if inst else cfg.api_key
            p = VllmProvider(base_url=url, metrics_path=metrics_path, api_key=api_key)
            info = await p.fetch_model_info()
            r0 = await p.fetch_metrics()
            time.sleep(min(cfg.interval, 1.0))
            r1 = await p.fetch_metrics()
            await p.aclose()
            return info, r0, r1

        info, r0, r1 = asyncio.run(_go())
        if not r1.fetched_ok:
            print(json.dumps({"error": r1.error}), file=sys.stderr)
            return 1
        # Fail closed on a 200 body that carries NO vLLM metrics (wrong metrics_path, a
        # proxy/error page, or a non-vLLM endpoint). Mirrors api.fetch_snapshot's guard so
        # this CLI JSON path (CAGE's `cage-stats --once --json` fallback) cannot emit an
        # all-zero fabricated snapshot that a downstream consumer reads as real telemetry.
        if "vllm:" not in (r1.text or ""):
            print(
                json.dumps({"error": "/metrics returned no vLLM metrics "
                            "(check metrics_path or that this endpoint is a vLLM server)"}),
                file=sys.stderr,
            )
            return 1
        # Mirror the api-path guard for the PRIMING poll (r0): a bad first poll would
        # zero-prime the rates and corrupt window deltas rather than fail loud.
        if not r0.fetched_ok:
            print(json.dumps({"error": r0.error or "failed first/priming /metrics poll"}),
                  file=sys.stderr)
            return 1
        if "vllm:" not in (r0.text or ""):
            print(
                json.dumps({"error": "first /metrics poll returned no vLLM metrics "
                            "(priming poll must be valid)"}),
                file=sys.stderr,
            )
            return 1
        md = load_model_dims(info.root, info.max_model_len)
        eng = MetricsEngine(dims=md.dims, max_model_len=md.max_model_len)
        eng.derive(parse_metrics(r0.text), now=0.0)
        snap = eng.derive(parse_metrics(r1.text), now=1.0)
    print(json.dumps(snapshot_to_dict(snap), default=str))
    return 0


def run_once_text(cfg: Config) -> int:
    """One-shot static terminal dashboard (no TUI, no Textual)."""
    from cage_stats.api import fetch_snapshot
    from cage_stats.ui.text import render_dashboard, render_fleet

    if len(cfg.instances) > 1:
        print(render_fleet(_poll_fleet(cfg), interval=cfg.interval))
        return 0
    inst = cfg.instances[0] if cfg.instances else None
    url = inst.url if inst else cfg.url
    try:
        snap = fetch_snapshot(
            url,
            metrics_path=(inst.metrics_path if inst else cfg.metrics_path),
            api_key=(inst.api_key if inst else cfg.api_key),
            interval=cfg.interval,
            mock=cfg.mock,
        )
    except Exception as e:
        print(f"cage-stats: {e}", file=sys.stderr)
        return 1
    print(render_dashboard(snap, url=url, interval=cfg.interval))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    env = dict(os.environ)
    cfg = Config.from_sources(argv, env)
    resolve_instances(cfg, env)
    if cfg.once and cfg.json:
        return run_once_json(cfg)
    if cfg.once:  # one-shot static terminal dashboard
        return run_once_text(cfg)
    from cage_stats.ui.app import run_app

    return run_app(cfg)
