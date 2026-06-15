"""
Traffic-tee event model for the optional proxy and log-tailer features.

``TeeEvent``
    An immutable record of a single observable event in the HTTP traffic stream.
    The ``kind`` field selects the payload:

    - ``"http"``      — an HTTP request/response observed in the access log or
      proxy.  Populated: ``method``, ``path``, ``status``, ``client``.
    - ``"exchange"``  — a prompt/completion pair captured by the reverse proxy.
      Populated: ``endpoint``, ``prompt``, ``response``, ``prompt_tokens``,
      ``completion_tokens``, ``streaming``, ``done``.
    - ``"note"``      — a human-readable status message (proxy started, docker
      not found, etc.).  Populated: ``text``.

``TeeBuffer``
    A bounded deque (default 500 events) of ``TeeEvent`` objects.  ``push``
    appends; ``recent(n)`` returns up to the last ``n`` events as a list.
    Thread-safety is provided by the GIL — all callers run in the same asyncio
    event loop.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class TeeEvent:
    ts: float
    kind: str
    method: str | None = None
    path: str | None = None
    status: int | None = None
    client: str | None = None
    endpoint: str | None = None
    prompt: str | None = None
    response: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    streaming: bool = False
    done: bool = True
    text: str | None = None


class TeeBuffer:
    def __init__(self, maxlen: int = 500) -> None:
        self._events: deque[TeeEvent] = deque(maxlen=maxlen)

    def push(self, event: TeeEvent) -> None:
        self._events.append(event)

    def recent(self, n: int) -> list[TeeEvent]:
        return list(self._events)[-n:] if n > 0 else []

    def __len__(self) -> int:
        return len(self._events)
