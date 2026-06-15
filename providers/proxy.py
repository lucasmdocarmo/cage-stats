"""
Streaming reverse-proxy with traffic tee for the optional ``--proxy`` feature.

``TeeProxy``
    An aiohttp-backed HTTP server that forwards every request to an upstream
    vLLM instance byte-for-byte while intercepting prompts and completions from
    OpenAI-compatible ``/v1/chat/completions`` and ``/v1/completions`` endpoints
    and emitting ``TeeEvent`` objects for display in the dashboard's TEE panel.

    The relay preserves streaming SSE responses faithfully: chunks are written
    to the client immediately as they arrive, and the ``SSEAccumulator`` builds
    the text side-channel without buffering or delaying any bytes.  Non-streaming
    JSON responses are buffered only until the upstream body is complete, then
    the accumulated text and token counts are attached to the event.

    Errors in the tee side-channel (event parsing, callback exceptions) are
    always silently swallowed — the relay itself can never be broken by an
    observer.

    Requires ``aiohttp`` (``pip install 'vllmstat[proxy]'``).  The application
    checks ``aiohttp_available()`` at startup and degrades gracefully.

``SSEAccumulator``
    Stateful SSE chunk parser that accumulates the ``content`` or ``text``
    deltas from OpenAI streaming responses.  Tolerant of chunk-split ``data:``
    lines.

``parse_proxy_addr(s)``
    Parse ``"host:port"`` or bare ``"port"`` into ``(host, int_port)``.

``aiohttp_available()``
    Return ``True`` when aiohttp can be imported.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

import httpx

from vllmstat.providers.tee import TeeEvent


def endpoint_for(path: str) -> str | None:
    if "chat/completions" in path:
        return "chat"
    if path.endswith("/completions"):
        return "completions"
    return None


def _content_text(content) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        ]
        joined = " ".join(t for t in parts if t)
        return joined or None
    return None


def extract_prompt(path: str, body: dict) -> str | None:
    ep = endpoint_for(path)
    if ep == "chat":
        msgs = body.get("messages") or []
        return _content_text(msgs[-1].get("content")) if msgs else None
    if ep == "completions":
        p = body.get("prompt")
        if isinstance(p, list):
            return " ".join(str(x) for x in p)
        return p if isinstance(p, str) else None
    return None


def parse_json_content(body: dict, endpoint: str) -> tuple[str, int | None, int | None]:
    choices = body.get("choices") or []
    text = ""
    if choices:
        c = choices[0]
        if endpoint == "chat":
            text = ((c.get("message") or {}).get("content")) or ""
        else:
            text = c.get("text") or ""
    usage = body.get("usage") or {}
    return text, usage.get("prompt_tokens"), usage.get("completion_tokens")


class SSEAccumulator:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self.text = ""
        self.done = False
        self._buf = ""

    def feed(self, chunk: str) -> None:
        self._buf += chunk
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._consume(line.strip())

    def _consume(self, line: str) -> None:
        if not line.startswith("data:"):
            return
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            self.done = True
            return
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            return
        choices = obj.get("choices") or []
        if not choices:
            return
        c = choices[0]
        if self.endpoint == "chat":
            self.text += (c.get("delta") or {}).get("content") or ""
        else:
            self.text += c.get("text") or ""


def parse_proxy_addr(s: str) -> tuple[str, int]:
    if ":" in s:
        host, _, port = s.rpartition(":")
        return (host or "0.0.0.0", int(port))
    return ("0.0.0.0", int(s))


def aiohttp_available() -> bool:
    try:
        import aiohttp  # noqa: F401

        return True
    except ImportError:
        return False


_HOP = {
    "host",
    "content-length",
    "transfer-encoding",
    "connection",
    "keep-alive",
    "accept-encoding",
}


class TeeProxy:
    def __init__(
        self,
        *,
        upstream_url: str,
        host: str,
        port: int,
        on_event: Callable[[TeeEvent], None],
        api_key: str | None = None,
    ) -> None:
        self.upstream_url = upstream_url.rstrip("/")
        self.host = host
        self.port = port
        self._on_event = on_event
        self._api_key = api_key
        self._client = httpx.AsyncClient(timeout=None)
        self._runner: object | None = None

    async def start(self) -> None:
        from aiohttp import web

        app = web.Application(client_max_size=0)
        app.router.add_route("*", "/{tail:.*}", self._handle)
        runner = web.AppRunner(app)
        self._runner = runner
        await runner.setup()
        await web.TCPSite(runner, self.host, self.port).start()

    async def stop(self) -> None:
        if self._runner is not None:
            from aiohttp import web

            assert isinstance(self._runner, web.AppRunner)
            await self._runner.cleanup()
            self._runner = None
        await self._client.aclose()

    async def _handle(self, request: Any) -> Any:
        from aiohttp import web

        body = await request.read()
        path = request.path
        endpoint = endpoint_for(path)
        streaming, prompt = False, None
        if endpoint and body:
            try:
                req = json.loads(body)
                prompt = extract_prompt(path, req)
                streaming = bool(req.get("stream"))
            except (ValueError, AttributeError):
                pass
        event = None
        if endpoint:
            event = TeeEvent(
                ts=time.time(),
                kind="exchange",
                endpoint=endpoint,
                prompt=prompt,
                response="",
                streaming=streaming,
                done=False,
            )
            try:
                self._on_event(event)
            except Exception:  # noqa: BLE001
                event = None

        fwd = {k: v for k, v in request.headers.items() if k.lower() not in _HOP}
        fwd["accept-encoding"] = "identity"
        if self._api_key and not any(k.lower() == "authorization" for k in fwd):
            fwd["Authorization"] = f"Bearer {self._api_key}"

        try:
            async with self._client.stream(
                request.method,
                self.upstream_url + path,
                content=body or None,
                headers=fwd,
                params=request.query_string or None,
            ) as up:
                out_headers = {k: v for k, v in up.headers.items() if k.lower() not in _HOP}
                resp = web.StreamResponse(status=up.status_code, headers=out_headers)
                await resp.prepare(request)
                acc = SSEAccumulator(endpoint) if (endpoint and streaming) else None
                raw = bytearray()
                async for chunk in up.aiter_raw():
                    await resp.write(chunk)
                    if event is not None:
                        if acc is not None:
                            acc.feed(chunk.decode("utf-8", "replace"))
                            event.response, event.done = acc.text, acc.done
                        elif not streaming:
                            raw += chunk
                await resp.write_eof()
                if event is not None and endpoint and not streaming:
                    try:
                        text, pt, ct = parse_json_content(json.loads(bytes(raw)), endpoint)
                        event.response, event.prompt_tokens, event.completion_tokens = text, pt, ct
                    except (ValueError, KeyError, TypeError):
                        pass
                    event.done = True
                return resp
        except Exception as e:  # noqa: BLE001
            if event is not None:
                event.response, event.done = f"[proxy error: {e}]", True
            return web.Response(status=502, text=f"vllmstat proxy: {e}")
