# cage-stats

**`nvtop` for vLLM** — a zero-infrastructure interactive terminal dashboard for vLLM serving
performance and GPU/fleet monitoring. Built for live monitoring of the CAGE benchmarking
cluster during Phase-2/3 runs.

It scrapes a vLLM server's built-in Prometheus `/metrics` endpoint directly and renders
everything in your terminal — concurrency, throughput, latency percentiles, cache & KV-memory,
speculative-decoding acceptance, KV-compression ratio, and per-GPU stats — refreshing every
second. No Prometheus, no Grafana, no browser tab.

---

## Install

```bash
# from the repo root
pip install -e .
# optional content-tee proxy (adds aiohttp)
pip install -e '.[proxy]'
```

Requires Python ≥ 3.10. A GPU driver is optional (only the GPU panel needs it).

## Usage

```bash
cage-stats                                   # point at http://localhost:8000
cage-stats --url http://my-gpu-host:8000     # a different host/port
cage-stats --mock                            # synthetic data, no server needed
cage-stats --once --json                     # one snapshot as JSON, then exit (scripting)
cage-stats --url http://h1:8000 --url http://h2:8000   # fleet (repeatable --url)
```

Run `cage-stats --help` for the full flag list (config file, Docker discovery, log/proxy tee,
metrics path, refresh interval, API key, `--no-gpu`).

## What it shows

- **Concurrency** — running requests, waiting queue, preemptions (with sparklines).
- **Throughput** — generation/prompt tok/s, tokens per iteration, requests/sec.
- **Cache & KV memory** — prefix-cache hit rate, token-source breakdown (compute vs cache-hit
  vs external KV transfer), KV-cache utilisation, and — when a quantised KV dtype is detected —
  the dtype and effective compression ratio vs fp16.
- **Latency percentiles** — TTFT, TPOT, end-to-end, queue-wait at p50/p90/p99.
- **Speculative decoding** — acceptance rate and accepted-per-draft (when active).
- **Per-GPU** — utilisation, VRAM, temperature, power, clocks, fan (NVIDIA / AMD / Intel).
- **Fleet** — monitor many vLLM servers from one overview; drill into any instance.
- **Tee** — a live request feed from logs, or full prompts/responses in proxy mode.

## Project layout

```
cage_stats/
├── cli.py            # entry point (cage-stats command) + orchestration
├── config/           # CLI/TOML config
├── metrics/          # /metrics parsing, KV math, time-series, snapshot engine
├── providers/        # vLLM scraper, mock, docker discovery, log/proxy tee
├── gpu/              # NVIDIA / AMD / Intel GPU stat providers
├── fleet/            # multi-instance resolution + overview
└── ui/               # Textual TUI (app, render, widgets, display)
```

## Status

Reorganized into an installable `cage_stats` package (entry point `cage-stats`). Derived from
the vllmstat architecture. Roadmap: tie the `--once --json` snapshot into the CAGE experiment
harness as a per-trial telemetry sidecar (spec-decode acceptance + KV-compression ratio).
