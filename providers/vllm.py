"""
vLLM HTTP provider: async client for the vLLM serving API.

``VllmProvider``
    Wraps an ``httpx.AsyncClient`` configured for a specific vLLM base URL.
    Provides two async methods:

    ``fetch_metrics()``
        GET ``<base_url><metrics_path>`` (default ``/metrics``).  Returns a
        ``RawText`` with the Prometheus exposition text on success, or an error
        message on any transport / HTTP failure.  Never raises.

    ``fetch_model_info()``
        GET ``<base_url>/v1/models``.  Extracts model names, the maximum
        context length, and the model root path (used for loading HuggingFace
        ``config.json`` to derive KV-cache dimensions).  Returns empty / None
        fields on failure rather than raising.

``RawText``
    Result of a metrics fetch: raw Prometheus text plus a success flag and
    optional error string.

``ModelInfo``
    Result of a model-info fetch: list of model IDs, max context length, and
    model root path.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass
class RawText:
    text: str
    fetched_ok: bool
    error: str | None = None


@dataclass
class ModelInfo:
    model_names: list[str]
    max_model_len: int | None
    root: str | None


class VllmProvider:
    def __init__(
        self,
        *,
        base_url: str,
        metrics_path: str = "/metrics",
        api_key: str | None = None,
        timeout: float = 5.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.metrics_path = metrics_path
        self._timeout = timeout
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.AsyncClient(base_url=self.base_url, headers=headers)

    async def fetch_metrics(self) -> RawText:
        try:
            r = await self._client.get(self.metrics_path, timeout=self._timeout)
            r.raise_for_status()
            return RawText(text=r.text, fetched_ok=True)
        except Exception as e:  # noqa: BLE001
            return RawText(text="", fetched_ok=False, error=str(e))

    async def fetch_model_info(self) -> ModelInfo:
        try:
            r = await self._client.get("/v1/models", timeout=self._timeout)
            r.raise_for_status()
            data = r.json().get("data", [])
        except Exception:  # noqa: BLE001
            return ModelInfo(model_names=[], max_model_len=None, root=None)
        names = [d.get("id") for d in data if d.get("id")]
        max_len = next((d.get("max_model_len") for d in data if d.get("max_model_len")), None)
        root = next((d.get("root") for d in data if d.get("root")), None)
        return ModelInfo(model_names=names, max_model_len=max_len, root=root)

    async def aclose(self) -> None:
        await self._client.aclose()
