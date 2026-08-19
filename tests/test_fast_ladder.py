"""A narrow, fast ladder that fits inside one receiver capture.

The Ku-band default steps 250 MHz between rungs, so a receiver has to retune for
each one and every rung carries a different tuning-dependent bias. A ladder whose
whole span fits inside the capture bandwidth is heard in a single tuning: every
rung shares one rx_lo, so that bias becomes common mode, and many cycles fit in
one listen.

Measured control cost on a Pi 4 at 1 MHz SPI: retune 0.84 ms, key 0.075 ms, so
about 1 ms per rung. u = 10 ms leaves 10x headroom.
"""
import time
import unittest

from . import context  # noqa: F401
from adf5355.ladder import (_digits_for, _sleep_until, check_schedule_feasible,
                            control_overhead_s, format_ladder, make_ladder)


class TestFeasibility(unittest.TestCase):
    def test_a_comfortable_fast_schedule_is_accepted(self):
        # 20 rungs, u = 10 ms, 4.2 s cycle.
        steps = make_ladder(11_000_000_000, 11_001_710_000, 20, 4.2)
        self.assertAlmostEqual(steps[0].on_s, 0.010, places=9)
        check_schedule_feasible(steps)          # must not raise

    def test_a_schedule_at_the_control_floor_is_refused(self):
        steps = make_ladder(11_000_000_000, 11_002_000_000, 20, 0.4)
        with self.assertRaises(ValueError) as caught:
            check_schedule_feasible(steps)
        self.assertIn("control overhead", str(caught.exception))

    def test_the_guard_scales_with_the_shortest_burst(self):
        floor = control_overhead_s(make_ladder(steps=4, total_s=1.0))
        # u just above 4x the floor passes, just below fails.
        n = 20
        ok = make_ladder(11_000_000_000, 11_001_000_000, n, 4.1 * floor * n * (n + 1))
        check_schedule_feasible(ok)
        bad = make_ladder(11_000_000_000, 11_001_000_000, n, 3.9 * floor * n * (n + 1))
        with self.assertRaises(ValueError):
            check_schedule_feasible(bad)

    def test_the_ku_default_is_feasible(self):
        check_schedule_feasible(make_ladder())


class TestDeadlineAccuracy(unittest.TestCase):
    def test_sleep_until_lands_within_a_millisecond(self):
        """time.sleep alone resolves to ~1 ms, which is 10% of a 10 ms unit."""
        for wait in (0.010, 0.025, 0.100):
            target = time.monotonic() + wait
            _sleep_until(target)
            overshoot = time.monotonic() - target
            self.assertGreaterEqual(overshoot, 0.0, "returned before the deadline")
            self.assertLess(overshoot, 0.001, f"overshot by {overshoot*1e3:.3f} ms")

    def test_a_deadline_already_passed_returns_at_once(self):
        start = time.monotonic()
        _sleep_until(start - 1.0)
        self.assertLess(time.monotonic() - start, 0.005)


class TestNarrowLadderRendering(unittest.TestCase):
    def test_every_rung_renders_distinctly(self):
        """90 kHz steps all print as the same number at three decimals."""
        steps = make_ladder(11_000_000_000, 11_001_710_000, 20, 4.2)
        text = format_ladder(steps)
        rendered = [line.split()[1] for line in text.splitlines()
                    if line.strip().startswith(tuple(str(i) for i in range(1, 10)))
                    and len(line.split()) > 3]
        self.assertEqual(len(set(rendered)), len(rendered),
                         "rungs are not distinguishable in the printed table")

    def test_narrow_spacing_reported_in_khz(self):
        text = format_ladder(make_ladder(11_000_000_000, 11_001_710_000, 20, 4.2))
        self.assertIn("spacing 90.000 kHz", text)

    def test_wide_spacing_still_reported_in_mhz(self):
        self.assertIn("spacing 250.000000 MHz", format_ladder(make_ladder()))

    def test_digits_scale_with_step_size(self):
        self.assertLess(_digits_for(250e6, 1e9), _digits_for(90e3, 1e9))


if __name__ == "__main__":
    unittest.main()
