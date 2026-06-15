"""
CLI argument parsing and TOML configuration file loading.

This module owns the complete configuration surface for cage_stats:

  - ``Config`` — the single dataclass that holds every runtime option.  It is
    populated by ``Config.from_sources(argv, env)`` which merges argparse output
    with environment variables.  A ``Config.instances`` list is filled in later
    by ``cli.resolve_instances`` once TOML / Docker discovery has run.

  - ``parse_config`` / ``load_config`` — parse a TOML file that may declare an
    ``[[instance]]`` array and optional top-level globals (``interval``, ``gpu``).

  - ``find_config`` — search for a config file in the standard locations:
    explicit ``--config`` flag → ``CAGE_STATS_CONFIG`` env var →
    ``./cage-stats.toml`` → ``~/.config/cage-stats/config.toml``.

TOML schema (all fields optional)::

    interval = 2.0          # poll interval in seconds
    gpu = true              # enable GPU stats

    [[instance]]
    url = "http://host:8000"
    name = "my-server"      # display label (derived from URL when omitted)
    api_key = "sk-..."
    metrics_path = "/metrics"
    gpus = [0, 1]           # GPU indices assigned to this instance
    local = true            # override automatic locality detection
    logs = "docker:mycontainer"
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

from cage_stats import __version__
from cage_stats.metrics.state import Instance

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]


def parse_config(text: str) -> tuple[list[dict], dict]:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"invalid TOML: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("config root must be a table")
    raw = data.get("instance", [])
    if not isinstance(raw, list):
        raise ValueError("'instance' must be an array of tables ([[instance]])")
    globals_ = {k: v for k, v in data.items() if k != "instance"}
    return raw, globals_


def load_config(path: str) -> tuple[list[dict], dict]:
    return parse_config(Path(path).expanduser().read_text())


def find_config(
    explicit: str | None,
    env: dict[str, str],
    *,
    candidates: list[str] | None = None,
    exists=None,
) -> str | None:
    exists = exists or (lambda p: Path(p).expanduser().is_file())
    if explicit:
        return explicit
    if env.get("CAGE_STATS_CONFIG"):
        return env["CAGE_STATS_CONFIG"]
    for c in candidates or ["./cage-stats.toml", "~/.config/cage-stats/config.toml"]:
        if exists(c):
            return c
    return None


@dataclass
class Config:
    urls: list[str] = field(default_factory=list)
    metrics_path: str = "/metrics"
    interval: float = 1.0
    api_key: str | None = None
    gpu: bool = True
    mock: bool = False
    once: bool = False
    json: bool = False
    config_path: str | None = None
    discover_docker: bool = False
    instances: list[Instance] = field(default_factory=list)
    logs: str | None = None
    proxy: str | None = None

    @property
    def url(self) -> str:
        return self.urls[0] if self.urls else "http://localhost:8000"

    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(prog="cage-stats", description="nvtop for vLLM")
        p.add_argument("-u", "--url", action="append", dest="urls", default=None, metavar="URL")
        p.add_argument("--metrics-path", default="/metrics")
        p.add_argument("-i", "--interval", type=float, default=1.0)
        p.add_argument("--api-key", default=None)
        p.add_argument("--no-gpu", dest="gpu", action="store_false", default=True)
        p.add_argument("--mock", action="store_true", default=False)
        p.add_argument("--once", action="store_true", default=False)
        p.add_argument("--json", action="store_true", default=False)
        p.add_argument("--config", dest="config_path", default=None)
        p.add_argument(
            "--discover-docker", dest="discover_docker", action="store_true", default=False
        )
        p.add_argument("--logs", dest="logs", default=None)
        p.add_argument("--proxy", dest="proxy", default=None)
        p.add_argument("--version", action="version", version=f"cage_stats {__version__}")
        return p

    @classmethod
    def from_sources(cls, argv: list[str], env: dict[str, str]) -> Config:
        ns = cls.build_parser().parse_args(argv)
        api_key = ns.api_key or env.get("VLLM_API_KEY")
        return cls(
            urls=list(ns.urls or []),
            metrics_path=ns.metrics_path,
            interval=ns.interval,
            api_key=api_key,
            gpu=ns.gpu,
            mock=ns.mock,
            once=ns.once,
            json=ns.json,
            config_path=ns.config_path,
            discover_docker=ns.discover_docker,
            logs=ns.logs,
            proxy=ns.proxy,
        )
