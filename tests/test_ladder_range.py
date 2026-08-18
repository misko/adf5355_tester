"""The ladder script's user-selectable frequency range.

The defaults are pinned: raspberry_pi_adf5355_ku_ladder_guide.pdf documents the
9-rung 10.700-12.700 GHz / 18.000 s pattern, so a run with no range arguments
has to produce exactly that.  Everything else here is the new --start-ghz /
--stop-ghz behaviour and its rejection paths.
"""
import contextlib
import io
import os
import unittest

from . import context  # noqa: F401
from .test_ladder_parity import LADDER_PATH, load_ladder

HAVE_LADDER = os.path.exists(LADDER_PATH)


class LadderRangeTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ladder = load_ladder()

    def run_main(self, *argv):
        """Run main() in dry-run (no SPI, no RF); return (rc, stdout, stderr)."""
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = self.ladder.main(["--ref-mhz", "40", "--dry-run", *argv])
        return rc, out.getvalue(), err.getvalue()


@unittest.skipUnless(HAVE_LADDER, "ladder script not present")
class TestDefaultsPinned(LadderRangeTestCase):
    def test_make_ladder_defaults_unchanged(self):
        steps = self.ladder.make_ladder()
        self.assertEqual(len(steps), 9)
        self.assertEqual(steps[0].freq_hz, 10_700_000_000)
        self.assertEqual(steps[-1].freq_hz, 12_700_000_000)
        self.assertAlmostEqual(steps[-1].end_s, 18.0, places=9)
        # u = total / (N*(N+1)) = 18 / 90 = 0.2 s; rung n is ON=n*u, OFF=n*u.
        for step in steps:
            self.assertAlmostEqual(step.on_s, step.index * 0.2, places=9)
            self.assertAlmostEqual(step.off_s, step.index * 0.2, places=9)

    def test_cli_defaults_reproduce_the_default_ladder(self):
        args = self.ladder.parse_args(["--ref-mhz", "40"])
        self.assertEqual(args.start_ghz, 10.7)
        self.assertEqual(args.stop_ghz, 12.7)
        self.assertEqual(args.steps, 9)
        self.assertEqual(args.total_s, 18.0)
        # The GHz->Hz conversion must land on the documented integers exactly.
        self.assertEqual(self.ladder.ghz_to_hz(args.start_ghz), 10_700_000_000)
        self.assertEqual(self.ladder.ghz_to_hz(args.stop_ghz), 12_700_000_000)
        cli = self.ladder.make_ladder(
            start_hz=self.ladder.ghz_to_hz(args.start_ghz),
            stop_hz=self.ladder.ghz_to_hz(args.stop_ghz),
            steps=args.steps, total_s=args.total_s,
        )
        self.assertEqual(cli, self.ladder.make_ladder())

    def test_default_run_is_the_documented_pattern(self):
        rc, out, err = self.run_main()
        self.assertEqual(rc, 0)
        self.assertEqual(err, "")
        for expected in ("   1    10.700  0.200   0.200",
                         "   9    12.700  1.800   1.800",
                         "Total coded interval: 18.000 s"):
            self.assertIn(expected, out)

    def test_explicit_default_range_matches_implicit(self):
        _, implicit, _ = self.run_main()
        _, explicit, _ = self.run_main("--start-ghz", "10.7",
                                       "--stop-ghz", "12.7", "--steps", "9",
                                       "--total-s", "18.0")
        self.assertEqual(implicit, explicit)


@unittest.skipUnless(HAVE_LADDER, "ladder script not present")
class TestCustomRange(LadderRangeTestCase):
    def test_bin_count_and_spacing(self):
        steps = self.ladder.make_ladder(8_000_000_000, 9_000_000_000, steps=5,
                                        total_s=6.0)
        self.assertEqual(len(steps), 5)
        self.assertEqual([s.freq_hz for s in steps],
                         [8_000_000_000, 8_250_000_000, 8_500_000_000,
                          8_750_000_000, 9_000_000_000])

    def test_spacing_is_span_over_steps_minus_one(self):
        for start, stop, n in ((6_800_000_000, 13_600_000_000, 2),
                               (7_000_000_000, 12_000_000_000, 11),
                               (10_000_000_000, 10_001_000_000, 3)):
            with self.subTest(start=start, stop=stop, steps=n):
                steps = self.ladder.make_ladder(start, stop, steps=n)
                self.assertEqual(len(steps), n)
                self.assertEqual(steps[0].freq_hz, start)
                self.assertEqual(steps[-1].freq_hz, stop)
                expected = (stop - start) / (n - 1)
                for i in range(1, n):
                    self.assertAlmostEqual(
                        steps[i].freq_hz - steps[i - 1].freq_hz, expected,
                        delta=1.0)

    def test_frequencies_are_monotonic(self):
        for start, stop, n in ((6_800_000_000, 13_600_000_000, 17),
                               (10_700_000_000, 12_700_000_000, 9),
                               (11_000_000_000, 11_000_000_000, 4)):
            with self.subTest(start=start, stop=stop, steps=n):
                freqs = [s.freq_hz for s in
                         self.ladder.make_ladder(start, stop, steps=n)]
                self.assertEqual(freqs, sorted(freqs))
                self.assertTrue(all(6_800_000_000 <= f <= 13_600_000_000
                                    for f in freqs))

    def test_timing_budget_still_sums_to_total_s(self):
        for n, total in ((2, 1.0), (5, 6.0), (9, 18.0), (16, 42.5)):
            with self.subTest(steps=n, total_s=total):
                steps = self.ladder.make_ladder(7_000_000_000, 8_000_000_000,
                                                steps=n, total_s=total)
                unit = total / (n * (n + 1))
                self.assertAlmostEqual(steps[-1].end_s, total, places=9)
                self.assertAlmostEqual(sum(s.on_s + s.off_s for s in steps),
                                       total, places=9)
                for step in steps:
                    self.assertAlmostEqual(step.on_s, step.index * unit,
                                           places=12)
                    self.assertAlmostEqual(step.off_s, step.index * unit,
                                           places=12)
                # The timeline is contiguous: each rung starts where the last ended.
                for previous, current in zip(steps, steps[1:]):
                    self.assertAlmostEqual(current.start_s, previous.end_s,
                                           places=12)

    def test_custom_range_through_the_cli(self):
        rc, out, err = self.run_main("--start-ghz", "8.0", "--stop-ghz", "9.0",
                                     "--steps", "5", "--total-s", "6")
        self.assertEqual(rc, 0)
        self.assertEqual(err, "")
        self.assertIn("in 5 bins, spacing 250.000000 MHz", out)
        self.assertIn("Total coded interval: 6.000 s", out)
        self.assertIn("8.000000-9.000000 GHz", out)

    def test_summary_reports_the_derived_plan(self):
        steps = self.ladder.make_ladder()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.ladder.print_ladder(steps)
        text = out.getvalue()
        self.assertIn("in 9 bins, spacing 250.000000 MHz", text)
        self.assertIn("Unit time u = 0.200000 s", text)
        self.assertIn("Total coded interval: 18.000 s", text)
        # The existing column layout must survive.
        self.assertIn("step  RF GHz     ON s   OFF s   timeline s", text)


@unittest.skipUnless(HAVE_LADDER, "ladder script not present")
class TestRangeRejected(LadderRangeTestCase):
    def assert_rejected(self, argv, needle):
        rc, out, err = self.run_main(*argv)
        self.assertEqual(rc, 2, f"{argv} should have failed cleanly")
        self.assertIn("ERROR:", err)
        self.assertIn(needle, err)
        self.assertNotIn("Traceback", err)
        self.assertNotIn("Duration-coded RFOUTB ladder", out)

    def test_below_rfoutb_minimum(self):
        self.assert_rejected(["--start-ghz", "6.0"], "--start-ghz")

    def test_above_rfoutb_maximum(self):
        self.assert_rejected(["--stop-ghz", "14.0"], "--stop-ghz")

    def test_stop_below_start(self):
        self.assert_rejected(["--start-ghz", "12.7", "--stop-ghz", "10.7"],
                             "below --start-ghz")

    def test_too_few_steps(self):
        self.assert_rejected(["--steps", "1"], "at least 2 rungs")

    def test_non_positive_total(self):
        self.assert_rejected(["--total-s", "0"], "must be positive")

    def test_non_finite_frequency(self):
        self.assert_rejected(["--start-ghz", "nan"], "not a finite frequency")

    def test_out_of_band_raises_before_any_rf(self):
        # Validation must reject the range without needing the device object.
        for start, stop in ((6_000_000_000, 12_700_000_000),
                            (10_700_000_000, 14_000_000_000),
                            (0, 1_000_000_000)):
            with self.subTest(start=start, stop=stop):
                with self.assertRaises(ValueError):
                    self.ladder.validate_ladder_range(start, stop, 9)

    def test_band_edges_are_accepted(self):
        self.ladder.validate_ladder_range(6_800_000_000, 13_600_000_000, 9)

    def test_zero_span_needs_two_rungs(self):
        # Allowed with >= 2 rungs: the duration coding still distinguishes them.
        self.ladder.validate_ladder_range(11_000_000_000, 11_000_000_000, 2)
        with self.assertRaises(ValueError) as ctx:
            self.ladder.validate_ladder_range(11_000_000_000, 11_000_000_000, 1)
        self.assertIn("differ only in duration", str(ctx.exception))

    def test_zero_span_ladder_is_single_frequency(self):
        steps = self.ladder.make_ladder(11_000_000_000, 11_000_000_000, steps=3,
                                        total_s=18.0)
        self.assertEqual({s.freq_hz for s in steps}, {11_000_000_000})
        self.assertAlmostEqual(steps[-1].end_s, 18.0, places=9)


@unittest.skipUnless(HAVE_LADDER, "ladder script not present")
class TestSafetyPosture(LadderRangeTestCase):
    def test_satellite_band_warning_for_overlapping_ranges(self):
        for start, stop in ((10_700_000_000, 12_700_000_000),
                            (9_000_000_000, 11_000_000_000),
                            (12_000_000_000, 13_600_000_000)):
            with self.subTest(start=start, stop=stop):
                text = "\n".join(self.ladder.safety_notice(start, stop))
                self.assertIn("satellite downlink band", text)
                self.assertIn("Do not connect an antenna", text)

    def test_non_overlapping_range_still_says_bench_only(self):
        text = "\n".join(self.ladder.safety_notice(6_800_000_000,
                                                   9_000_000_000))
        self.assertIn("bench-only equipment", text)
        self.assertIn("do not connect an antenna", text.lower())
        self.assertIn("shielded", text)

    def test_rf_is_never_enabled_without_the_flag(self):
        rc, out, _ = self.run_main("--start-ghz", "8.0", "--stop-ghz", "9.0")
        self.assertEqual(rc, 0)
        self.assertIn("Dry run: RF is never enabled.", out)
        self.assertIn("SAFETY:", out)


if __name__ == "__main__":
    unittest.main()
