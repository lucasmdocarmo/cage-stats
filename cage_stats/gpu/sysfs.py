"""
DRM vendor detection and shared sysfs read helpers.

Used by both the AMD and Intel GPU backends to locate cards and read raw sysfs
values.  All functions accept explicit paths so that tests can point them at a
fake ``/sys`` tree under a temp directory.  Every reader catches ``OSError`` and
returns ``None``/empty — a missing file or permission error degrades to ``—``
in the UI rather than crashing.

``Card``
    Immutable description of a DRM card: its sysfs path, card index (integer
    suffix of the ``cardN`` directory name), and vendor key.

``detect_cards(drm_root)``
    Enumerate real DRM cards under ``drm_root`` (default ``/sys/class/drm``).
    Skips connector nodes (``card0-DP-1``) and render nodes (``renderD128``).
    Returns cards sorted by index.

``read_text(path)``
    Return the stripped contents of ``path``, or ``None`` on any ``OSError``.

``read_int(path)``
    Return ``path`` parsed as a decimal or ``0x``-prefixed hex integer, or
    ``None`` on any error.

``pci_name(card_path)``
    Derive a human-readable GPU name from the PCI vendor/device IDs, falling
    back to ``"<vendor> GPU 0x<id>"`` for unknown devices.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

DEFAULT_DRM_ROOT = "/sys/class/drm"

_VENDOR_IDS = {
    0x10DE: "nvidia",
    0x1002: "amd",
    0x8086: "intel",
}

_DEVICE_NAMES = {
    (0x8086, 0xE223): "Intel Arc B-series (Battlemage)",
    (0x8086, 0xE20B): "Intel Arc B580 (Battlemage)",
    (0x8086, 0xE20C): "Intel Arc B570 (Battlemage)",
    (0x8086, 0x56A0): "Intel Arc A770 (Alchemist)",
    (0x8086, 0x56A1): "Intel Arc A750 (Alchemist)",
}

_CARD_RE = re.compile(r"^card(\d+)$")


@dataclass(frozen=True)
class Card:
    index: int
    path: str
    vendor: str


def read_text(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read().strip()
    except OSError:
        return None


def read_int(path: str) -> int | None:
    raw = read_text(path)
    if raw is None or not raw:
        return None
    try:
        return int(raw, 0) if raw.lower().startswith(("0x", "0o", "0b")) else int(raw)
    except ValueError:
        return None


def _vendor_key(vendor_id: int | None) -> str:
    if vendor_id is None:
        return "other"
    return _VENDOR_IDS.get(vendor_id, "other")


def detect_cards(drm_root: str = DEFAULT_DRM_ROOT) -> list[Card]:
    try:
        entries = os.listdir(drm_root)
    except OSError:
        return []
    cards: list[Card] = []
    for entry in entries:
        m = _CARD_RE.match(entry)
        if not m:
            continue
        card_path = os.path.join(drm_root, entry)
        vendor_id = read_int(os.path.join(card_path, "device", "vendor"))
        if vendor_id is None:
            continue
        cards.append(Card(index=int(m.group(1)), path=card_path, vendor=_vendor_key(vendor_id)))
    cards.sort(key=lambda c: c.index)
    return cards


def pci_name(card_path: str) -> str:
    vendor_id = read_int(os.path.join(card_path, "device", "vendor"))
    device_id = read_int(os.path.join(card_path, "device", "device"))
    vendor = _vendor_key(vendor_id)
    if vendor_id is not None and device_id is not None:
        name = _DEVICE_NAMES.get((vendor_id, device_id))
        if name is not None:
            return name
    if device_id is not None:
        return f"{vendor} GPU 0x{device_id:04x}"
    return f"{vendor} GPU"
