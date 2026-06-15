"""
Intel GPU backend for xe and i915 drivers, combining sysfs metrics and DRM
fdinfo client aggregation.

Sysfs readers (``read_intel_sysfs``, ``read_gtidle_util``, ``read_pci_vram_total``)
    Read hardware counters from the xe / i915 sysfs tree.  All are world-readable
    as non-root and never raise — missing files degrade to ``None`` fields.

    - Clocks from ``tile0/gt0/freq0/cur_freq``
    - Package temperature and power cap from hwmon
    - Power derived from ``energy1_input`` counter delta between two samples
    - GPU utilisation from per-GT idle-residency counters
      (``tile*/gt*/gtidle/idle_residency_ms``): ``util = 1 − Δidle / Δwall``
    - VRAM total from the largest prefetchable PCI BAR in ``device/resource``

    VRAM used and a fallback utilisation are obtained via ``read_fdinfo`` (below)
    which requires the GPU process to run as the same user or as root.

DRM fdinfo aggregation (``read_fdinfo``)
    Scans ``/proc/[pid]/fdinfo/*`` for DRM clients bound to a specific PCI
    device (matched by ``drm-pdev``), deduplicates by ``drm-client-id``, sums
    per-client ``drm-cycles-<eng>`` busy counters, and takes the max per-client
    ``drm-total-cycles-<eng>`` elapsed-cycle counter.  Utilisation is
    ``(Δbusy / Δtotal)`` per engine; we report the busiest engine.
    VRAM is the sum of per-client ``drm-resident-vram0`` (KiB → bytes).
    All state is caller-managed so tests can inject a fake ``/proc`` tree.

``pdev_for_card(card_path)``
    Resolve the PCI bus address backing a DRM card by reading the ``device``
    symlink, returning the basename of its realpath.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

from vllmstat.gpu.sysfs import pci_name, read_int, read_text
from vllmstat.metrics.state import GpuSample

EnergyState = tuple[int, float]

_IORESOURCE_PREFETCH = 0x2000
_DRM_DRIVERS = frozenset({"xe", "i915"})


def _hwmon_dir(card_path: str) -> str | None:
    base = os.path.join(card_path, "device", "hwmon")
    try:
        candidates = sorted(glob.glob(os.path.join(base, "hwmon*")))
    except OSError:
        return None
    if not candidates:
        return None
    for hw in candidates:
        if read_text(os.path.join(hw, "name")) in ("xe", "i915"):
            return hw
    return candidates[0]


def _div(path: str, denom: float) -> float | None:
    val = read_int(path)
    return (val / denom) if val is not None else None


def read_intel_sysfs(
    card_path: str,
    prev_energy: EnergyState | None,
    now: float,
) -> tuple[GpuSample, EnergyState | None]:
    dev = os.path.join(card_path, "device")
    clock_sm = read_int(os.path.join(dev, "tile0", "gt0", "freq0", "cur_freq"))

    temp_c = power_limit_w = None
    fan_rpm = None
    new_energy: EnergyState | None = None
    power_w: float | None = None

    hw = _hwmon_dir(card_path)
    if hw is not None:
        temp_c = _div(os.path.join(hw, "temp2_input"), 1000.0)
        if temp_c is None:
            temp_c = _div(os.path.join(hw, "temp1_input"), 1000.0)
        power_limit_w = _div(os.path.join(hw, "power1_cap"), 1e6)
        fan_rpm = read_int(os.path.join(hw, "fan1_input"))

        energy = read_int(os.path.join(hw, "energy1_input"))
        if energy is not None:
            new_energy = (energy, now)
            if prev_energy is not None:
                e_prev, t_prev = prev_energy
                dt = now - t_prev
                if dt > 0:
                    power_w = (energy - e_prev) / 1e6 / dt

    return (
        GpuSample(
            index=0,
            name=pci_name(card_path),
            vendor="intel",
            util_gpu=None,
            mem_used=None,
            mem_total=None,
            temp_c=temp_c,
            power_w=power_w,
            power_limit_w=power_limit_w,
            fan_rpm=fan_rpm,
            clock_sm_mhz=clock_sm,
        ),
        new_energy,
    )


def read_gtidle_util(
    card_path: str,
    prev_idle: dict[str, int] | None,
    now: float,
    prev_now: float | None,
) -> tuple[float | None, dict[str, int], float]:
    pattern = os.path.join(card_path, "device", "tile*", "gt*", "gtidle", "idle_residency_ms")
    try:
        paths = sorted(glob.glob(pattern))
    except OSError:
        paths = []

    new_idle: dict[str, int] = {}
    best: float | None = None
    have_prev = prev_idle is not None and prev_now is not None
    dwall_ms = (now - prev_now) * 1000.0 if prev_now is not None else 0.0

    for path in paths:
        idle = read_int(path)
        if idle is None:
            continue
        new_idle[path] = idle
        if not have_prev or dwall_ms <= 0:
            continue
        assert prev_idle is not None
        prev = prev_idle.get(path)
        if prev is None:
            continue
        didle = idle - prev
        util_gt = 100.0 * (1.0 - didle / dwall_ms)
        util_gt = max(0.0, min(100.0, util_gt))
        if best is None or util_gt > best:
            best = util_gt

    return best, new_idle, now


def read_pci_vram_total(card_path: str) -> int | None:
    raw = read_text(os.path.join(card_path, "device", "resource"))
    if not raw:
        return None
    best: int | None = None
    for line in raw.splitlines():
        cols = line.split()
        if len(cols) < 3:
            continue
        try:
            start = int(cols[0], 16)
            end = int(cols[1], 16)
            flags = int(cols[2], 16)
        except ValueError:
            continue
        if not (flags & _IORESOURCE_PREFETCH) or end <= start:
            continue
        size = end - start + 1
        if best is None or size > best:
            best = size
    return best


def pdev_for_card(card_path: str) -> str | None:
    try:
        target = os.path.realpath(os.path.join(card_path, "device"))
    except OSError:
        return None
    name = os.path.basename(target)
    return name or None


@dataclass(frozen=True)
class FdinfoStats:
    util_pct: float | None
    vram_used_bytes: int | None
    clients: int


def _parse_fdinfo(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, val = line.partition(":")
        if sep:
            out[key.strip()] = val.strip()
    return out


def _first_int(val: str) -> int | None:
    parts = val.split()
    if not parts:
        return None
    try:
        return int(parts[0])
    except ValueError:
        return None


def read_fdinfo(
    pdev: str,
    *,
    proc_root: str = "/proc",
    prev_busy: dict[str, int] | None,
    prev_total: dict[str, int] | None,
    now: float,
) -> tuple[FdinfoStats, dict[str, int], dict[str, int]]:
    del now

    busy: dict[str, int] = {}
    total: dict[str, int] = {}
    vram_kib = 0
    seen_clients: set[str] = set()

    try:
        paths = glob.glob(os.path.join(proc_root, "[0-9]*", "fdinfo", "*"))
    except OSError:
        paths = []

    for path in paths:
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        if "drm-" not in text:
            continue
        kv = _parse_fdinfo(text)
        if kv.get("drm-driver") not in _DRM_DRIVERS:
            continue
        if kv.get("drm-pdev") != pdev:
            continue
        client_id = kv.get("drm-client-id")
        if client_id is None or client_id in seen_clients:
            continue
        seen_clients.add(client_id)

        for key, raw in kv.items():
            if key.startswith("drm-total-cycles-"):
                eng = key[len("drm-total-cycles-"):]
                n = _first_int(raw)
                if n is not None and n > total.get(eng, 0):
                    total[eng] = n
            elif key.startswith("drm-cycles-"):
                eng = key[len("drm-cycles-"):]
                n = _first_int(raw)
                if n is not None:
                    busy[eng] = busy.get(eng, 0) + n
        resident = _first_int(kv.get("drm-resident-vram0", ""))
        if resident is not None:
            vram_kib += resident

    clients = len(seen_clients)
    vram_used_bytes = vram_kib * 1024 if clients else None

    util_pct: float | None = None
    if prev_busy is not None and prev_total is not None:
        best: float | None = None
        for eng, total_now in total.items():
            dtotal = total_now - prev_total.get(eng, 0)
            if dtotal <= 0:
                continue
            dbusy = busy.get(eng, 0) - prev_busy.get(eng, 0)
            frac = dbusy / dtotal
            frac = max(0.0, min(1.0, frac))
            if best is None or frac > best:
                best = frac
        if best is not None:
            util_pct = 100.0 * best

    stats = FdinfoStats(util_pct=util_pct, vram_used_bytes=vram_used_bytes, clients=clients)
    return stats, busy, total
