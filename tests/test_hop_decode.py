"""The hop decoder, against synthetic captures with a known answer.

Every number the decoder reports is checked against one that was injected:
a comb offset, an epoch, and a set of frequency points. Nothing here needs a
radio, and nothing here transmits -- the capture is built in memory from the
same schedule the transmitter would have run.

The failure that matters for this tool is a *confident wrong answer*, so the
last few tests feed it captures that hold no recoverable schedule (the wrong
seed, and pure noise) and assert that it says so.
"""
import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import unittest

import numpy as np

from . import context  # noqa: F401
from adf5355.hopper import (DEFAULT_HOP_START_HZ, DEFAULT_HOP_STOP_HZ,
                            DEFAULT_MIN_HOP_S, DEFAULT_POINTS, DEFAULT_SEED,
                            make_schedule, plan_frequencies)

DECODER_PATH = os.path.join(context.ROOT, "tools", "hop_decode.py")


def load_decoder():
    spec = importlib.util.spec_from_file_location("hop_decode", DECODER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


hd = load_decoder()

# One tuning hears the whole span: centre on the middle of it, less the LNB LO
# and the LO error the bench measured.
FS = 2.5e6
CENTRE = ((DEFAULT_HOP_START_HZ + DEFAULT_HOP_STOP_HZ) / 2
          - hd.DEFAULT_LO_HZ - hd.DEFAULT_LO_ERROR_HZ)
OFFSET = -106.0e3        # the comb offset to inject and then recover
EPOCH = 0.0371           # where in the schedule the capture starts
SECONDS = 0.45           # a bit over two periods at the defaults
PLAN = dict(fs=FS, centre_hz=CENTRE)


class TestPureHelpers(unittest.TestCase):
    def test_frame_bins_ascend_around_the_centre(self):
        bins = hd.frame_bins(512, FS, CENTRE)
        self.assertEqual(len(bins), 512)
        self.assertTrue(np.all(np.diff(bins) > 0))
        self.assertAlmostEqual(bins[256], CENTRE, places=6)
        self.assertAlmostEqual(bins[1] - bins[0], FS / 512, places=6)

    def test_slots_never_overlap(self):
        freqs = plan_frequencies(DEFAULT_HOP_START_HZ, DEFAULT_HOP_STOP_HZ,
                                 DEFAULT_POINTS)
        if_nom = np.array(freqs, dtype=float) - hd.DEFAULT_LO_HZ
        half = hd.slot_half_width(if_nom)
        self.assertLess(2 * half, float(np.min(np.diff(if_nom))))
        bins = hd.frame_bins(512, FS, CENTRE)
        slots = hd.point_slots(bins, if_nom, OFFSET, half)
        for (_, hi), (lo, _) in zip(slots, slots[1:]):
            self.assertLessEqual(hi, lo)

    def test_slot_width_is_capped_for_a_wide_plan(self):
        if_nom = np.array([1e9, 1e9 + 10e6])
        self.assertEqual(hd.slot_half_width(if_nom), hd.DEFAULT_MAX_SLOT_HZ)

    def test_envelope_db_is_relative_to_each_points_own_floor(self):
        env = np.vstack([np.full(100, 2.0), np.full(100, 2000.0)])
        env[:, :10] *= 100.0                      # a 20 dB excursion each
        db = hd.envelope_db(env)
        self.assertAlmostEqual(db[0, 0], 20.0, places=6)
        self.assertAlmostEqual(db[1, 0], 20.0, places=6)
        self.assertAlmostEqual(db[0, 50], 0.0, places=6)

    def test_capture_rejects_input_it_cannot_frame(self):
        with self.assertRaises(ValueError):
            hd.Capture(np.zeros(100, dtype=np.complex64), 512)
        with self.assertRaises(ValueError):
            hd.Capture(np.zeros((4, 512), dtype=np.complex64), 512)
        with self.assertRaises(ValueError):
            hd.Capture(np.zeros(4096, dtype=np.complex64), 4)

    def test_capture_blocks_cover_the_whole_capture_exactly_once(self):
        samples = np.arange(2048, dtype=np.complex64)
        capture = hd.Capture(samples, 512)
        self.assertEqual(capture.nframes, 4)
        seen = np.concatenate([capture.block(0, 3).ravel(),
                               capture.block(3, 3).ravel()])
        self.assertTrue(np.array_equal(seen, samples))


class TestCombOffset(unittest.TestCase):
    """Step 2 alone: slide a known comb over a spectrum built to hold it."""

    def setUp(self):
        self.freqs = plan_frequencies(DEFAULT_HOP_START_HZ,
                                      DEFAULT_HOP_STOP_HZ, DEFAULT_POINTS)
        self.if_nom = np.array(self.freqs, dtype=float) - hd.DEFAULT_LO_HZ
        self.bins = hd.frame_bins(512, FS, CENTRE)

    def _spectrum(self, offset_hz, noise=1.0):
        rng = np.random.default_rng(4)
        power = rng.uniform(0.5, 1.5, len(self.bins)) * noise
        for centre in self.if_nom + offset_hz:
            k = int(np.argmin(np.abs(self.bins - centre)))
            power[k] += 1000.0
        return power

    def test_a_planted_offset_comes_back(self):
        for planted in (-106e3, 0.0, +205e3):
            offset, sharpness = hd.comb_offset(self._spectrum(planted),
                                               self.bins, self.if_nom)
            self.assertLess(abs(offset - planted), FS / 512,
                            f"planted {planted}")
            self.assertGreater(sharpness, hd.MIN_COMB_SHARPNESS)

    def test_noise_alone_is_reported_as_unconvincing(self):
        rng = np.random.default_rng(9)
        _, sharpness = hd.comb_offset(rng.uniform(0.5, 1.5, len(self.bins)),
                                      self.bins, self.if_nom)
        self.assertLess(sharpness, hd.MIN_COMB_SHARPNESS)


class TestSyntheticDecode(unittest.TestCase):
    """The whole chain, once, against an injected offset and epoch."""

    @classmethod
    def setUpClass(cls):
        cls.samples = hd.synthesise(seconds=SECONDS, offset_hz=OFFSET,
                                    start_at_s=EPOCH, snr_db=10.0, **PLAN)
        cls.result = hd.decode(cls.samples, **PLAN)

    def test_every_point_is_recovered(self):
        self.assertEqual(self.result.recovered, DEFAULT_POINTS)
        self.assertEqual([r.point for r in self.result.rows],
                         list(range(DEFAULT_POINTS)))

    def test_the_comb_offset_lands_within_a_bin(self):
        self.assertLess(abs(self.result.comb_offset_hz - OFFSET), FS / 512)

    def test_the_epoch_lands_within_a_frame(self):
        error = abs(self.result.epoch_s - EPOCH) % self.result.period_s
        self.assertLess(min(error, self.result.period_s - error),
                        2 * self.result.frame_s)

    def test_every_point_reports_the_injected_offset(self):
        # Interpolated peaks, so the tolerance is a small fraction of a bin
        # rather than a bin: 500 Hz against a 4883 Hz bin.
        for row in self.result.rows:
            self.assertLess(abs(row.error_hz - OFFSET), 500.0,
                            f"point {row.point}")

    def test_the_summary_is_precise_and_tight(self):
        self.assertLess(abs(self.result.median_error_hz - OFFSET), 200.0)
        self.assertLess(self.result.spread_hz, 500.0)

    def test_confidence_is_reported_as_good(self):
        self.assertGreater(self.result.comb_sharpness, hd.MIN_COMB_SHARPNESS)
        self.assertGreater(self.result.epoch_sigma, hd.MIN_EPOCH_SIGMA)
        self.assertTrue(self.result.trustworthy)
        self.assertEqual(self.result.warnings, [])

    def test_alignment_labels_each_frame_with_the_schedules_point(self):
        """Identity comes from the seed, so the labels must BE the schedule.

        Compared frame by frame against the schedule that built the capture,
        for every frame that lies wholly inside one hop.
        """
        freqs = plan_frequencies(DEFAULT_HOP_START_HZ, DEFAULT_HOP_STOP_HZ,
                                 DEFAULT_POINTS)
        hops = make_schedule(DEFAULT_SEED, freqs, DEFAULT_MIN_HOP_S, 8)
        if_nom = np.array(freqs, dtype=float) - hd.DEFAULT_LO_HZ

        capture = hd.Capture(self.samples, hd.DEFAULT_FRAME)
        window = np.hanning(hd.DEFAULT_FRAME).astype(np.float32)
        bins = hd.frame_bins(hd.DEFAULT_FRAME, FS, CENTRE)
        slots = hd.point_slots(bins, if_nom, self.result.comb_offset_hz,
                               hd.slot_half_width(if_nom))
        env, _ = hd.point_envelopes(capture, window, bins, slots)
        fit = hd.align_epoch(hd.envelope_db(env), hops, DEFAULT_POINTS,
                             self.result.frame_s)

        ends = np.array([h.end_s for h in hops])
        points = np.array([h.point for h in hops])
        checked = wrong = 0
        for k in range(capture.nframes):
            begin = EPOCH + k * self.result.frame_s
            finish = begin + self.result.frame_s
            first = int(np.searchsorted(ends, begin))
            if first != int(np.searchsorted(ends, finish)):
                continue                       # straddles a hop boundary
            checked += 1
            wrong += int(fit.assigned[k] != points[first])
        self.assertGreater(checked, 1000)
        self.assertEqual(wrong, 0)

    def test_the_report_names_the_confidence_figures(self):
        text = hd.format_report(self.result)
        self.assertIn("comb offset", text)
        self.assertIn("sharpness", text)
        self.assertIn("sigma", text)
        self.assertIn(f"{DEFAULT_POINTS}/{DEFAULT_POINTS} recovered", text)

    def test_a_file_capture_decodes_the_same_as_an_array(self):
        with tempfile.TemporaryDirectory() as work:
            path = os.path.join(work, "synthetic.iq")
            hd.write_int16(path, self.samples)
            self.assertEqual(os.path.getsize(path), self.samples.size * 4)
            from_file = hd.decode(path, **PLAN)
        self.assertEqual(from_file.recovered, self.result.recovered)
        self.assertAlmostEqual(from_file.comb_offset_hz,
                               self.result.comb_offset_hz, places=6)
        self.assertAlmostEqual(from_file.epoch_s, self.result.epoch_s, places=9)
        self.assertLess(abs(from_file.median_error_hz
                            - self.result.median_error_hz), 50.0)


class TestItRefusesToBeConfidentlyWrong(unittest.TestCase):
    def test_the_wrong_seed_does_not_produce_a_measurement(self):
        samples = hd.synthesise(seconds=SECONDS, offset_hz=OFFSET,
                                start_at_s=EPOCH, snr_db=10.0, **PLAN)
        wrong = hd.decode(samples, seed=DEFAULT_SEED ^ 0xABCDEF, **PLAN)
        # The comb is still there -- it does not depend on the order -- but the
        # schedule cannot be aligned, so the points must not be reported.
        self.assertLess(wrong.epoch_sigma, hd.MIN_EPOCH_SIGMA)
        self.assertFalse(wrong.trustworthy)
        self.assertTrue(any("epoch" in w for w in wrong.warnings), wrong.warnings)

    def test_noise_alone_recovers_nothing_and_says_so(self):
        rng = np.random.default_rng(11)
        n = int(SECONDS * FS)
        noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)
                 ).astype(np.complex64)
        result = hd.decode(noise, **PLAN)
        self.assertEqual(result.recovered, 0)
        self.assertFalse(result.trustworthy)
        self.assertLess(result.comb_sharpness, hd.MIN_COMB_SHARPNESS)
        self.assertIn("CONFIDENCE IS POOR", hd.format_report(result))

    def test_a_frame_longer_than_a_dwell_is_called_out(self):
        samples = hd.synthesise(seconds=SECONDS, offset_hz=OFFSET,
                                start_at_s=EPOCH, snr_db=10.0, **PLAN)
        result = hd.decode(samples, frame=32768, **PLAN)
        self.assertTrue(any("frames" in w for w in result.warnings),
                        result.warnings)


class TestTheBannerAndTheExitStatusAgree(unittest.TestCase):
    """A scripted caller only sees the exit status; a human only sees the banner.

    If those two can disagree, the tool prints "do not use these numbers as a
    measurement" and hands the caller a zero anyway -- which is precisely the
    confident-wrong-answer failure this decoder exists to avoid.
    """

    def _result(self, **overrides):
        fields = dict(comb_offset_hz=-106e3, comb_sharpness=5000.0,
                      epoch_s=0.037, epoch_sigma=1000.0, period_s=0.2,
                      points=20, recovered=20, frame_s=2.048e-4, nframes=4882,
                      seconds=1.0, slot_half_hz=27e3,
                      rows=[hd.PointResult(p, 1.25e9, 1.25e9, 0.0, 200, 200,
                                           30.0) for p in range(20)],
                      warnings=[])
        fields.update(overrides)
        return hd.DecodeResult(**fields)

    def test_a_clean_result_is_trusted_and_prints_no_banner(self):
        result = self._result()
        self.assertTrue(result.trustworthy)
        self.assertNotIn("CONFIDENCE IS POOR", hd.format_report(result))

    def test_a_partial_recovery_is_not_quietly_reported_as_success(self):
        """15 of 20 points is a failure, not a measurement with a footnote."""
        result = self._result(recovered=15, rows=self._result().rows[:15],
                              warnings=["only 15 of 20 points recovered"])
        self.assertIn("CONFIDENCE IS POOR", hd.format_report(result))
        self.assertFalse(result.trustworthy)

    def test_a_configuration_warning_also_fails_the_run(self):
        result = self._result(warnings=["dwell spans 2.0 frames"])
        self.assertIn("CONFIDENCE IS POOR", hd.format_report(result))
        self.assertFalse(result.trustworthy)

    def test_every_warning_the_decoder_can_raise_fails_the_run(self):
        """Whatever earns the banner must set the exit status, without listing.

        Enumerated from the decoder itself rather than restated, so a warning
        added later cannot quietly come with a zero exit status attached.
        """
        for warning in ("comb sharpness", "epoch sigma", "only 3 of 20",
                        "dwell spans", "span 4.000 MHz exceeds"):
            with self.subTest(warning=warning):
                result = self._result(warnings=[warning])
                self.assertFalse(result.trustworthy)
                self.assertIn("CONFIDENCE IS POOR", hd.format_report(result))

    def test_main_exits_non_zero_on_a_capture_it_cannot_decode(self):
        """End to end through main(), which is what a shell actually sees."""
        rng = np.random.default_rng(5)
        n = int(0.3 * FS)
        noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        with tempfile.TemporaryDirectory() as work:
            path = os.path.join(work, "noise.iq")
            hd.write_int16(path, noise.astype(np.complex64))
            argv = ["--capture", path, "--fs", str(FS), "--seconds", "0.3"]
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                status = hd.main(argv)
        self.assertEqual(status, 1)
        self.assertIn("CONFIDENCE IS POOR", buffer.getvalue())

    def test_main_exits_zero_on_a_capture_it_can_decode(self):
        with tempfile.TemporaryDirectory() as work:
            path = os.path.join(work, "good.iq")
            samples = hd.synthesise(seconds=SECONDS, offset_hz=OFFSET,
                                    start_at_s=EPOCH, snr_db=10.0, **PLAN)
            hd.write_int16(path, samples)
            argv = ["--capture", path, "--fs", str(FS)]
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                status = hd.main(argv)
        self.assertEqual(status, 0)
        self.assertIn("confidence good", buffer.getvalue())


class TestBothEndsShareOneSchedule(unittest.TestCase):
    """The decoder must not carry its own copy of the schedule generator."""

    def test_the_schedule_comes_from_the_package(self):
        from adf5355 import hopper
        self.assertIs(hd.make_schedule, hopper.make_schedule)
        self.assertIs(hd.plan_frequencies, hopper.plan_frequencies)
        self.assertIs(hd.period_duration, hopper.period_duration)

    def test_defaults_are_the_packages_defaults(self):
        from adf5355 import hopper
        parser = hd.build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.seed, hopper.DEFAULT_SEED)
        self.assertEqual(args.points, hopper.DEFAULT_POINTS)
        self.assertEqual(args.min_hop_ms, hopper.DEFAULT_MIN_HOP_S * 1e3)
        self.assertEqual(args.jitter, hopper.DEFAULT_JITTER)
        self.assertEqual(args.period_cycles, hopper.DEFAULT_PERIOD_CYCLES)
        self.assertEqual(round(args.start_ghz * 1e9), hopper.DEFAULT_HOP_START_HZ)
        self.assertEqual(round(args.stop_ghz * 1e9), hopper.DEFAULT_HOP_STOP_HZ)

    def test_the_transmitters_flags_and_the_decoders_flags_agree(self):
        from adf5355 import cli
        tx = cli.build_parser().parse_args(["hop"])
        rx = hd.build_parser().parse_args([])
        for name in ("seed", "start_ghz", "stop_ghz", "points", "min_hop_ms",
                     "jitter", "period_cycles"):
            self.assertEqual(getattr(tx, name), getattr(rx, name), name)

    def test_importing_the_decoder_touches_no_hardware(self):
        """pyadi must stay lazy so --help and these tests run with no radio."""
        self.assertNotIn("adi", sys.modules)
        with open(DECODER_PATH, encoding="utf-8") as fh:
            source = fh.read()
        self.assertNotIn("\nimport adi", source)      # only inside a function
        self.assertIn("    import adi", source)


if __name__ == "__main__":
    unittest.main()
