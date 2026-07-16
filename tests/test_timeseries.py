"""Rate EWMA priming regression tests (2026-07-15 review, finding B1).

The two-poll snapshot path (api.fetch_snapshot / cli --once) computes exactly ONE
delta. A 0-seeded EWMA reported 0.3x the true rate for every gen_tps/prompt_tps/
req_rate in every run (verified against the 2026-07-15 smoke data:
prompt_tps_peak 96.9 == 0.3 * 323). Dependency-free: imports only the module.
"""
from __future__ import annotations

import pytest

from cage_stats.metrics.timeseries import Rate


def test_two_poll_path_reports_true_instantaneous_rate():
    # The exact api.py access pattern: derive(now=0.0) then derive(now=1.0).
    r = Rate(alpha=0.3)
    assert r.update(1000.0, 0.0) == 0.0          # first sample only sets the baseline
    assert r.update(1323.0, 1.0) == pytest.approx(323.0)   # NOT 0.3 * 323 = 96.9


def test_ewma_blends_after_priming():
    r = Rate(alpha=0.3)
    r.update(0.0, 0.0)
    assert r.update(100.0, 1.0) == pytest.approx(100.0)            # primed
    assert r.update(300.0, 2.0) == pytest.approx(0.3 * 200 + 0.7 * 100)  # blended


def test_counter_reset_does_not_unprime_or_spike():
    r = Rate(alpha=0.3)
    r.update(500.0, 0.0)
    r.update(600.0, 1.0)                          # primed at 100
    v = r.update(10.0, 2.0)                       # server restart: counter reset
    assert v == pytest.approx(100.0)              # holds last value, no negative/spike
    assert r.update(110.0, 3.0) == pytest.approx(0.3 * 100 + 0.7 * 100)


def test_non_positive_dt_is_ignored():
    r = Rate(alpha=0.3)
    r.update(100.0, 5.0)
    r.update(200.0, 6.0)                          # primed at 100
    assert r.update(999.0, 6.0) == pytest.approx(100.0)   # dt == 0 -> unchanged
