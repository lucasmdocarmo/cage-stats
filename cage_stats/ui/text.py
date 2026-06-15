"""Static, one-shot terminal dashboard rendering (no Textual).

``render_dashboard`` composes the plain-text panels from :mod:`cage_stats.ui.render`
into a single framed dashboard string for a one-shot ``cage-stats --once`` (or for
embedding via :func:`cage_stats.api.dashboard_text`). Unlike the live TUI it needs
no Textual and no event loop — just a derived :class:`Snapshot`.
"""

from __future__ import annotations

from cage_stats.metrics.state import FleetSnapshot, Snapshot
from cage_stats.metrics.timeseries import History
from cage_stats.ui import render


_SEP = "─" * 78


def render_dashboard(snap: Snapshot, *, url: str = "", interval: float = 1.0, uptime: str = "—") -> str:
    """Render a full static dashboard for one vLLM instance as a string.

    Each panel keeps its own header (CONCURRENCY, THROUGHPUT, …) and is separated
    by a rule — no box framing, so nothing is truncated.
    """
    h = History()
    # Seed one data point so the mini-plots render a value rather than blank.
    h.push("running", snap.running)
    h.push("waiting", snap.waiting)
    h.push("gen_tps", snap.gen_tps)
    h.push("prompt_tps", snap.prompt_tps)
    if snap.prefix_hit_window is not None:
        h.push("prefix_hit", snap.prefix_hit_window)

    out: list[str] = [render.header(snap, url=url, interval=interval, uptime=uptime)]
    if not snap.connected:
        out.append(f"  (not connected: {snap.error or 'no metrics'})")
        return "\n".join(out)

    panels = [
        render.concurrency(snap, h),
        render.throughput(snap, h),
        render.latency(snap),
        render.cache_kv(snap, h),
        render.session(snap),
        render.specdecode(snap),
        render.efficiency(snap),
        render.gpu(snap),
    ]
    for p in panels:
        if p and p.strip():
            out.append(_SEP)
            out.append(p)
    return "\n".join(out)


def render_fleet(fs: FleetSnapshot, *, interval: float = 1.0, uptime: str = "—") -> str:
    """Render a one-shot fleet overview table."""
    return render.fleet_overview(fs, selected=-1, interval=interval, uptime=uptime)
