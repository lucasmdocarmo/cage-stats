"""
Terminal display primitives: number formatting, sparklines, and braille plots.

Number formatters
-----------------
``fmt_si(n)``
    Format a number with SI magnitude suffixes: T, B, M, k.  One decimal place.
    Returns ``"—"`` for ``None``.

``fmt_bytes(n)``
    Format an integer byte count as ``"X.X GB"``.

``fmt_dur(seconds)``
    Format a duration: sub-second values in milliseconds (``"42ms"``),
    longer values in seconds (``"1.2s"``).

``fmt_pct(frac)``
    Format a 0–1 fraction as a percentage with one decimal place.

``fmt_dur_hms(seconds)``
    Compact h/m/s duration.  ``None`` → ``"—"``; ``<60`` → ``"42s"``;
    ``<3600`` → ``"12m03s"``; else ``"1h05m"``.

Sparkline
---------
``sparkline(values)``
    Map a sequence of floats to a string of Unicode block characters (▁–█).
    The range is normalised to the min/max of the input; a flat sequence
    returns all ``▁``.

Braille area plot
-----------------
``braille_plot(values, width, height, lo, hi)``
    Render ``values`` as a filled area plot using Unicode braille characters
    (U+2800–U+28FF).  Each braille cell packs 2 columns × 4 rows of dots,
    yielding effective pixel resolution of ``2×width`` × ``4×height``.

    Dot-bit map (``cx`` 0–1 left→right, ``cy`` 0–3 top→bottom)::

        (0,0)=0x01  (1,0)=0x08
        (0,1)=0x02  (1,1)=0x10
        (0,2)=0x04  (1,2)=0x20
        (0,3)=0x40  (1,3)=0x80

    Returns exactly ``height`` strings, each exactly ``width`` characters.
    The last ``2×width`` values are shown; shorter inputs are left-padded with
    the baseline.  Never raises.
"""

from __future__ import annotations

from collections.abc import Sequence

_SPARK = "▁▂▃▄▅▆▇█"

_DOT_BITS: dict[tuple[int, int], int] = {
    (0, 0): 0x01,
    (0, 1): 0x02,
    (0, 2): 0x04,
    (0, 3): 0x40,
    (1, 0): 0x08,
    (1, 1): 0x10,
    (1, 2): 0x20,
    (1, 3): 0x80,
}


def sparkline(values: Sequence[float]) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span <= 0:
        return _SPARK[0] * len(vals)
    out = []
    for v in vals:
        idx = int((v - lo) / span * (len(_SPARK) - 1))
        out.append(_SPARK[idx])
    return "".join(out)


def fmt_si(n: float | None) -> str:
    if n is None:
        return "—"
    n = float(n)
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.1f}{unit}"
    return f"{n:.0f}"


def fmt_bytes(n: int | None) -> str:
    if n is None:
        return "—"
    g = n / 1e9
    return f"{g:.1f} GB"


def fmt_dur(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.1f}s"


def fmt_pct(frac: float | None) -> str:
    if frac is None:
        return "—"
    return f"{frac * 100:.1f}%"


def fmt_dur_hms(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    if total < 3600:
        m, s = divmod(total, 60)
        return f"{m}m{s:02d}s"
    h, rem = divmod(total, 3600)
    m, _ = divmod(rem, 60)
    return f"{h}h{m:02d}m"


def braille_plot(
    values: Sequence[float],
    width: int,
    height: int = 4,
    lo: float | None = None,
    hi: float | None = None,
) -> list[str]:
    width = max(0, int(width))
    height = max(0, int(height))
    if width == 0 or height == 0:
        return [" " * width for _ in range(height)]

    dot_cols = width * 2
    dot_rows = height * 4
    blank = [" " * width for _ in range(height)]

    vals = [float(v) for v in values if v is not None]
    if not vals:
        return blank

    if lo is None:
        lo = 0.0 if min(vals) >= 0 else float(min(vals))
    else:
        lo = float(lo)
    if hi is None:
        hi = float(max(vals))
    else:
        hi = float(hi)
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo

    tail = list(vals[-dot_cols:])
    if len(tail) < dot_cols:
        tail = [lo] * (dot_cols - len(tail)) + tail

    col_d: list[int] = []
    top = dot_rows - 1
    for v in tail:
        d = round((v - lo) / span * (dot_rows - 1))
        if d < 0:
            d = 0
        elif d > top:
            d = top
        col_d.append(d)

    masks = [[0] * width for _ in range(height)]
    for dx in range(dot_cols):
        d = col_d[dx]
        cell_x = dx // 2
        cx = dx % 2
        for dy in range(dot_rows - 1 - d, dot_rows):
            cell_y = dy // 4
            cy = dy % 4
            masks[cell_y][cell_x] |= _DOT_BITS[(cx, cy)]

    return ["".join(chr(0x2800 + m) for m in row) for row in masks]
