"""
Fleet package for cage_stats.

Manages the lifecycle of one or more monitored vLLM instances, handling
concurrent polling, exception isolation, and GPU metric slicing.

Modules
-------
``fleet``
    ``InstanceRuntime`` — per-instance wrapper holding a provider, metrics
    engine, history buffer, and tee event buffer.
    ``Fleet`` — concurrent poll coordinator across all runtimes.

``resolve``
    URL normalisation, locality detection, TOML instance deserialisation, and
    name deduplication for multi-instance fleets.
"""

from cage_stats.fleet.fleet import Fleet, InstanceRuntime, build_fleet
from cage_stats.fleet.resolve import (
    classify_locality,
    derive_name,
    instance_from_dict,
    instance_from_url,
    local_hostnames,
    normalize_url,
    resolve_fleet,
)

__all__ = [
    "Fleet",
    "InstanceRuntime",
    "build_fleet",
    "classify_locality",
    "derive_name",
    "instance_from_dict",
    "instance_from_url",
    "local_hostnames",
    "normalize_url",
    "resolve_fleet",
]
