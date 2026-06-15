"""
Async log tailer for the optional ``--logs`` / Docker log source feature.

``LogTailer``
    Tails a log source asynchronously, parses each line for HTTP access-log
    entries, and emits ``TeeEvent`` objects to the dashboard's TEE panel.

    Two source types are supported:

    - ``"docker:<name>"`` — runs ``docker logs -f --since 0s <name>`` as a
      subprocess and reads its merged stdout / stderr.  If ``docker`` is not
      on ``PATH``, emits a note event and exits cleanly.

    - Any other string — treated as a file path.  Opens the file, seeks to the
      end, then polls for new lines every 0.2 s.  Detects log rotation by
      checking whether the file has shrunk and re-opening when it has.
      Waits for the file to appear if it does not exist yet.

    All errors are converted to ``"note"`` events rather than propagating
    exceptions into the event loop.  ``terminate()`` provides a synchronous
    best-effort shutdown suitable for use in ``__del__`` or ``atexit`` handlers;
    ``stop()`` awaits the task for a clean async shutdown.

``parse_access_line(line)``
    Parse one uvicorn / gunicorn access-log line into a ``TeeEvent`` or
    ``None`` when the line does not match or the path is on the filter list
    (health checks, metrics, etc.).
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable
from pathlib import Path

from cage_stats.providers.tee import TeeEvent

_ACCESS = re.compile(r'"(?P<method>[A-Z]+)\s+(?P<path>\S+)\s+HTTP/[\d.]+"\s+(?P<status>\d{3})')
_CLIENT = re.compile(r'(?P<client>\d{1,3}(?:\.\d{1,3}){3}|\[[0-9a-fA-F:]+\]):\d+\s+-\s+"')
_FILTER = ("/health", "/metrics", "/ping", "/v1/models")


def parse_access_line(
    line: str, *, now: float | None = None, filter_paths: tuple[str, ...] = _FILTER
) -> TeeEvent | None:
    m = _ACCESS.search(line)
    if not m:
        return None
    path = m.group("path")
    if path.split("?", 1)[0] in filter_paths:
        return None
    cm = _CLIENT.search(line)
    client = cm.group("client").strip("[]") if cm else None
    return TeeEvent(
        ts=now if now is not None else time.time(),
        kind="http",
        method=m.group("method"),
        path=path,
        status=int(m.group("status")),
        client=client,
    )


class LogTailer:
    def __init__(
        self,
        source: str,
        *,
        on_event: Callable[[TeeEvent], None],
        parse: Callable[[str], TeeEvent | None] = parse_access_line,
    ) -> None:
        self.source = source
        self._on_event = on_event
        self._parse = parse
        self._task: asyncio.Task[None] | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._stop = False

    def start(self) -> None:
        self._task = asyncio.ensure_future(self._run())

    async def _run(self) -> None:
        try:
            if self.source.startswith("docker:"):
                await self._run_docker(self.source[len("docker:"):])
            else:
                await self._run_file(self.source)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            self._note(f"tee source error: {e}")

    async def _run_docker(self, name: str) -> None:
        try:
            self._proc = await asyncio.create_subprocess_exec(
                "docker",
                "logs",
                "-f",
                "--since",
                "0s",
                name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except FileNotFoundError:
            self._note("docker not found")
            return
        assert self._proc.stdout is not None
        while not self._stop:
            raw = await self._proc.stdout.readline()
            if not raw:
                break
            self._feed(raw.decode("utf-8", "replace").rstrip("\n"))

    async def _run_file(self, path: str) -> None:
        p = Path(path).expanduser()
        while not self._stop:
            try:
                with p.open("r", errors="replace") as f:
                    f.seek(0, 2)
                    while not self._stop:
                        line = f.readline()
                        if line:
                            self._feed(line.rstrip("\n"))
                        else:
                            await asyncio.sleep(0.2)
                            if p.stat().st_size < f.tell():
                                break
            except FileNotFoundError:
                self._note(f"waiting for {path}…")
                await asyncio.sleep(1.0)

    def _feed(self, line: str) -> None:
        ev = self._parse(line)
        if ev is not None:
            self._on_event(ev)

    def _note(self, text: str) -> None:
        self._on_event(TeeEvent(ts=time.time(), kind="note", text=text))

    def terminate(self) -> None:
        self._stop = True
        if self._proc is not None:
            try:
                self._proc.terminate()
            except OSError:  # noqa: BLE001
                pass
        if self._task is not None:
            self._task.cancel()

    async def stop(self) -> None:
        self.terminate()
        if self._task is not None:
            try:
                await self._task
            except asyncio.CancelledError:
                pass
