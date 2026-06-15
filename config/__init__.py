"""
Configuration package for vllmstat.

Exposes the runtime Config dataclass (built from CLI args and environment variables)
and TOML config-file loading utilities. Everything the entry point needs to bootstrap
the application lives here.
"""

from vllmstat.config.config import Config, find_config, load_config, parse_config

__all__ = ["Config", "find_config", "load_config", "parse_config"]
