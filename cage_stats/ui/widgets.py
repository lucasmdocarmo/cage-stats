"""
Custom Textual widgets for cage_stats.

``Panel``
    A ``Static`` subclass that stores the last value passed to ``update()`` on
    ``self.renderable`` so callers and tests can read back the panel's current
    text without going through Textual's internal Visual layer (which changed
    between Textual 7 and 8).
"""

from __future__ import annotations

from typing import Any

from textual.widgets import Static


class Panel(Static):
    renderable: Any = ""

    def update(self, content: Any = "", *, layout: bool = True) -> None:
        self.renderable = content
        super().update(content, layout=layout)
