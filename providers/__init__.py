"""
Data-source providers for vllmstat.

Each provider is responsible for fetching or generating raw data that the
metrics engine transforms into a ``Snapshot``.

Modules
-------
``vllm``
    Async HTTP client for a vLLM ``/metrics`` and ``/v1/models`` endpoint.

``mock``
    Deterministic synthetic metrics for ``--mock`` mode and unit tests.

``proxy``
    Streaming reverse-proxy with prompt/completion tee (requires aiohttp).

``tee``
    ``TeeEvent`` / ``TeeBuffer`` — the shared event model consumed by the proxy
    and log tailer and rendered by the dashboard's TEE panel.

``logsource``
    Async tail of Docker container logs or plain log files.

``docker``
    Docker-based vLLM instance discovery for ``--discover-docker``.
"""
