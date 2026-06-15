"""
Configuration package for cage_stats.

Exposes the runtime Config dataclass (built from CLI args and environment variables)
and TOML config-file loading utilities. Everything the entry point needs to bootstrap
the application lives here.
"""

from cage_stats.config.config import Config, find_config, load_config, parse_config

__all__ = ["Config", "find_config", "load_config", "parse_config"]
