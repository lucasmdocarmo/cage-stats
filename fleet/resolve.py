"""
Instance URL resolution, normalisation, and fleet construction.

This module converts raw URL strings and TOML dicts into ``Instance`` objects
and assembles a deduplicated, ordered fleet list from multiple sources.

``normalize_url(url)``
    Canonicalise a URL: strip whitespace, prepend ``http://`` when the scheme
    is missing, lowercase the hostname, restore IPv6 brackets, and strip a
    trailing slash from the path.  Query strings and fragments are dropped
    (a vLLM base URL never carries them).

``derive_name(url)``
    Return a short display name for a URL — ``hostname:port`` when a port is
    present, otherwise just the hostname.

``classify_locality(url, local_names)``
    Return ``"local"`` when the URL's hostname is in ``local_names``, else
    ``"remote"``.

``local_hostnames()``
    Build the set of names that resolve to the local machine: the literal
    loopback addresses plus the machine's own hostname, FQDN, and all IP
    addresses returned by ``getaddrinfo``.

``instance_from_dict(raw, ...)``
    Deserialise a TOML ``[[instance]]`` dict into an ``Instance``, filling in
    defaults for omitted fields.  Raises ``ValueError`` when ``url`` is absent.

``resolve_fleet(config_instances, docker_instances, url_flags, ...)``
    Merge instances from multiple sources (TOML config, Docker discovery, CLI
    ``--url`` flags), deduplicating by normalised URL.  If all sources are
    empty, a default ``http://localhost:8000`` instance is created.

``_dedupe_names``
    Append ``#2``, ``#3``, … suffixes to instances that share a display name.
"""

from __future__ import annotations

import socket
from dataclasses import replace
from urllib.parse import urlparse

from vllmstat.metrics.state import Instance

_LOCAL = {"localhost", "127.0.0.1", "::1", "0.0.0.0", ""}


def normalize_url(url: str) -> str:
    u = url.strip()
    if "://" not in u:
        u = "http://" + u
    p = urlparse(u)
    host = (p.hostname or "").lower()
    if ":" in host:
        host = f"[{host}]"
    port = f":{p.port}" if p.port else ""
    return f"{p.scheme}://{host}{port}{p.path.rstrip('/')}"


def derive_name(url: str) -> str:
    p = urlparse(normalize_url(url))
    return f"{p.hostname}:{p.port}" if p.port else (p.hostname or url)


def classify_locality(url: str, local_names: set[str]) -> str:
    host = (urlparse(normalize_url(url)).hostname or "").lower()
    return "local" if host in local_names else "remote"


def local_hostnames() -> set[str]:
    names: set[str] = set(_LOCAL)
    try:
        h = socket.gethostname()
        names |= {h.lower(), socket.getfqdn().lower()}
        for info in socket.getaddrinfo(h, None):
            names.add(str(info[4][0]).lower())
    except OSError:
        pass
    return names


def instance_from_dict(
    raw: dict,
    *,
    defaults_api_key: str | None = None,
    defaults_metrics_path: str = "/metrics",
    local_names: set[str],
) -> Instance:
    url = raw.get("url")
    if not url:
        raise ValueError("instance is missing required 'url'")
    locality = (
        ("local" if raw["local"] else "remote")
        if "local" in raw
        else classify_locality(url, local_names)
    )
    return Instance(
        name=raw.get("name") or derive_name(url),
        url=normalize_url(url),
        metrics_path=raw.get("metrics_path", defaults_metrics_path),
        api_key=raw.get("api_key", defaults_api_key),
        gpus=tuple(int(g) for g in raw.get("gpus", [])),
        locality=locality,
        logs=raw.get("logs"),
    )


def instance_from_url(url: str, **kw) -> Instance:
    return instance_from_dict({"url": url}, **kw)


def resolve_fleet(
    config_instances: list[Instance],
    docker_instances: list[Instance],
    url_flags: list[str],
    *,
    defaults_api_key: str | None = None,
    defaults_metrics_path: str = "/metrics",
    default_url: str = "http://localhost:8000",
    local_names: set[str],
) -> list[Instance]:
    by_url: dict[str, Instance] = {}
    order: list[str] = []

    def add(inst: Instance) -> None:
        key = normalize_url(inst.url)
        if key not in by_url:
            order.append(key)
            by_url[key] = inst

    for i in config_instances:
        add(i)
    for i in docker_instances:
        add(i)
    for u in url_flags:
        add(
            instance_from_url(
                u,
                defaults_api_key=defaults_api_key,
                defaults_metrics_path=defaults_metrics_path,
                local_names=local_names,
            )
        )
    if not order:
        add(
            instance_from_url(
                default_url,
                defaults_api_key=defaults_api_key,
                defaults_metrics_path=defaults_metrics_path,
                local_names=local_names,
            )
        )
    return _dedupe_names([by_url[k] for k in order])


def _dedupe_names(instances: list[Instance]) -> list[Instance]:
    seen: dict[str, int] = {}
    out: list[Instance] = []
    for inst in instances:
        if inst.name in seen:
            seen[inst.name] += 1
            inst = replace(inst, name=f"{inst.name}#{seen[inst.name]}")
        else:
            seen[inst.name] = 1
        out.append(inst)
    return out
