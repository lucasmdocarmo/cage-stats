"""
GPU monitoring package for cage_stats.

Supports NVIDIA (via NVML / nvidia-smi), AMD (via amd-smi / rocm-smi / amdgpu
sysfs), and Intel (via xe / i915 sysfs and DRM fdinfo).  All backends degrade
gracefully when tools or permissions are unavailable — missing metrics appear as
``None`` in ``GpuSample`` fields and render as ``—`` in the dashboard.

Modules
-------
``sysfs``
    DRM card detection and low-level sysfs read helpers shared across GPU
    vendors.  Pure functions, no state.

``amd``
    AMD GPU backend: sysfs reader plus amd-smi / rocm-smi JSON parsers.

``intel``
    Intel xe / i915 backend: sysfs reader, GT idle-residency utilisation,
    PCI BAR VRAM detection, and DRM fdinfo aggregation.

``provider``
    ``GpuProvider`` — the single entry point used by the application.  Detects
    which vendor cards are present, dispatches to the right backend, and returns
    a ``GpuSnapshot``.
"""

from cage_stats.gpu.provider import GpuProvider

__all__ = ["GpuProvider"]
