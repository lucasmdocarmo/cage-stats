"""
Terminal UI package for cage_stats.

Built on the Textual framework.  All display logic — from raw number formatting
to full panel rendering — lives here so the rest of the application has no UI
dependencies.

Modules
-------
``display``
    Number formatters (SI prefixes, bytes, durations, percentages), Unicode
    sparkline generator, and braille-based area plot renderer.

``widgets``
    ``Panel`` — a Textual ``Static`` subclass with an accessible ``renderable``
    attribute for testing.

``render``
    High-level panel rendering functions that combine a ``Snapshot`` (and
    optionally a ``History``) into a formatted string ready to pass to
    ``Panel.update()``.

``app``
    ``CageStatsApp`` — the Textual application class.  ``run_app(cfg)`` is the
    entry point called by the CLI.
"""
