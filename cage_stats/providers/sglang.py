"""SGLang HTTP provider: async client for the SGLang serving API.

Mirrors :class:`~cage_stats.providers.vllm.VllmProvider` — same ``RawText`` /
``ModelInfo`` contract, same never-raise fetch semantics — pointed at an
SGLang server launched with ``--enable-metrics``. SGLang serves an
OpenAI-compatible ``/v1/models`` (model id + optional context length), so the
model-info fetch degrades to ``None`` fields rather than raising when the
shape differs.

The scraped exposition text is SGLang-dialect (``sglang:``-prefixed); callers
feed it through
:func:`cage_stats.metrics.sglang_dialect.translate_sglang_families` before the
:class:`~cage_stats.metrics.engine.MetricsEngine` sees it.
"""

from __future__ import annotations

import httpx

from cage_stats.providers.vllm import ModelInfo, RawText

__all__ = ["SGLangProvider"]


class SGLangProvider:
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
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.AsyncClient(timeout=timeout, headers=headers)

    async def fetch_metrics(self) -> RawText:
        """GET ``<base_url><metrics_path>``; never raises."""
        url = f"{self.base_url}{self.metrics_path}"
        try:
            resp = await self._client.get(url)
        except httpx.HTTPError as exc:  # transport failure
            return RawText(text="", fetched_ok=False, error=f"{type(exc).__name__}: {exc}")
        if resp.status_code != 200:
            return RawText(
                text="",
                fetched_ok=False,
                error=f"HTTP {resp.status_code} from {url}",
            )
        return RawText(text=resp.text, fetched_ok=True)

    async def fetch_model_info(self) -> ModelInfo:
        """GET ``/v1/models``; empty/None fields on any failure (never raises)."""
        url = f"{self.base_url}/v1/models"
        try:
            resp = await self._client.get(url)
            data = resp.json()
        except (httpx.HTTPError, ValueError):
            return ModelInfo(model_names=[], max_model_len=None, root=None)
        names: list[str] = []
        max_len: int | None = None
        root: str | None = None
        for item in data.get("data", []) if isinstance(data, dict) else []:
            if not isinstance(item, dict):
                continue
            mid = item.get("id")
            if isinstance(mid, str) and mid:
                names.append(mid)
            if max_len is None:
                for key in ("max_model_len", "context_length"):
                    val = item.get(key)
                    if isinstance(val, int) and val > 0:
                        max_len = val
                        break
            if root is None and isinstance(item.get("root"), str):
                root = item["root"]
        return ModelInfo(model_names=names, max_model_len=max_len, root=root)

    async def aclose(self) -> None:
        await self._client.aclose()
