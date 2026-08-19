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


# ===========================================================================
# The frequency estimator: what limits it, and what fixes it
# ===========================================================================
class TestTheFramedEstimatorsFloorIsNotNoise(unittest.TestCase):
    """The old estimator's spread is a fixed bias, so listening longer is futile.

    A parabola through three log-magnitude bins of a Hann-windowed frame is a
    biased fit, and the bias is a fixed function of where the tone sits inside
    its bin. Every point of the plan sits at its own fixed fractional bin
    position, so every point gets its own fixed error -- in every frame, of
    every visit, of every capture. That is why the bench saw the same 56-71 Hz
    spread whether it listened for 0.2 s or 32 s.

    These tests use NO noise at all. Whatever they measure is bias by
    construction.
    """

    FRAME = 512

    def _framed_peak(self, x):
        """One frame through exactly the arithmetic point_envelopes() uses."""
        x = x - x.mean()
        power = np.abs(np.fft.fftshift(np.fft.fft(x * np.hanning(self.FRAME))))**2
        bins = np.fft.fftshift(np.fft.fftfreq(self.FRAME, 1.0 / FS))
        bin_hz = bins[1] - bins[0]
        k = int(np.clip(np.argmax(power), 1, self.FRAME - 2))
        a, b, c = (np.log(power[k - 1]), np.log(power[k]), np.log(power[k + 1]))
        den = a - 2 * b + c
        d = 0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0
        return bins[k] + np.clip(d, -0.5, 0.5) * bin_hz

    def test_the_error_is_a_fixed_function_of_position_within_a_bin(self):
        n = np.arange(self.FRAME)
        bin_hz = FS / self.FRAME
        errors = []
        for frac in np.linspace(-0.5, 0.5, 41):
            f0 = 40 * bin_hz + frac * bin_hz
            errors.append(self._framed_peak(np.exp(2j * np.pi * f0 * n / FS)) - f0)
        errors = np.array(errors)
        self.assertGreater(np.abs(errors).max(), 50.0)      # tens of Hz, noiseless
        self.assertLess(np.abs(errors).max(), 100.0)

    def test_it_is_repeatable_to_the_last_bit_so_averaging_cannot_help(self):
        n = np.arange(self.FRAME)
        f0 = 40 * FS / self.FRAME + 0.3 * FS / self.FRAME
        first = self._framed_peak(np.exp(2j * np.pi * f0 * n / FS))
        for phase in (0.0, 1.0, 2.5):        # a different frame of the same tone
            again = self._framed_peak(np.exp(2j * np.pi * f0 * n / FS + 1j * phase))
            self.assertAlmostEqual(first, again, places=6)

    def test_the_twenty_real_points_spread_by_tens_of_hz_with_no_noise_present(self):
        """This is the ~58 Hz the bench measured, reproduced without any noise."""
        freqs = plan_frequencies(DEFAULT_HOP_START_HZ, DEFAULT_HOP_STOP_HZ,
                                 DEFAULT_POINTS)
        if_nom = np.array(freqs, dtype=float) - hd.DEFAULT_LO_HZ
        baseband = if_nom + OFFSET - CENTRE
        n = np.arange(self.FRAME)
        errors = np.array([self._framed_peak(np.exp(2j * np.pi * f * n / FS)) - f
                           for f in baseband])
        self.assertGreater(errors.std(), 40.0)
        self.assertLess(errors.std(), 80.0)

    def test_the_whole_dwell_estimator_has_no_such_bias(self):
        """The same sweep, through fine_ml: three orders of magnitude flatter."""
        m, fsd = 781, FS / hd.DEFAULT_DECIMATE
        n = np.arange(m)
        bin_hz = fsd / m
        errors = []
        for frac in np.linspace(-0.5, 0.5, 41):
            f0 = 10 * bin_hz + frac * bin_hz
            errors.append(hd.fine_ml(np.exp(2j * np.pi * f0 * n / fsd), fsd) - f0)
        self.assertLess(np.abs(errors).max(), 0.01)


class TestFineEstimatorsOnACleanTone(unittest.TestCase):
    """Each fine estimator on its own, against a planted frequency."""

    M = 781
    FSD = FS / hd.DEFAULT_DECIMATE

    def _tone(self, f0, snr_db=None, seed=0):
        y = np.exp(2j * np.pi * f0 * np.arange(self.M) / self.FSD)
        if snr_db is not None:
            rng = np.random.default_rng(seed)
            sigma = 10.0 ** (-snr_db / 20.0) / np.sqrt(2.0)
            y = y + (rng.standard_normal(self.M)
                     + 1j * rng.standard_normal(self.M)) * sigma
        return y

    def test_every_estimator_recovers_a_planted_frequency_without_noise(self):
        for name, fn in hd.FINE_ESTIMATORS.items():
            for f0 in (-1234.5, -7.25, 0.0, 311.0, 2500.75):
                with self.subTest(estimator=name, f0=f0):
                    self.assertLess(abs(fn(self._tone(f0), self.FSD) - f0), 0.05)

    def test_the_boxcar_decimation_does_not_move_the_frequency(self):
        """A boxcar is linear phase, so it can shift time but never bias a slope.

        Run through the shipped path with a zero mixing rate, which is exactly
        a boxcar decimation.
        """
        n = np.arange(25000)
        for f0 in (-900.0, 12.5, 3100.0):
            fine = np.exp(2j * np.pi * f0 * n / FS)
            coarse = hd.mix_and_decimate(fine, 0.0, 0, hd.DEFAULT_DECIMATE)
            self.assertEqual(coarse.size, 25000 // hd.DEFAULT_DECIMATE)
            got = hd.fine_ml(coarse, FS / hd.DEFAULT_DECIMATE)
            self.assertLess(abs(got - f0), 0.01, f"{f0} Hz")

    def test_folding_the_mixdown_into_the_decimation_changes_no_answer(self):
        """The fast path must equal the obvious one, not merely resemble it."""
        rng = np.random.default_rng(0)
        raw = (rng.standard_normal(25000)
               + 1j * rng.standard_normal(25000)).astype(np.complex64)
        cycles, start = 0.20281, 4_000_000
        n = np.arange(raw.size, dtype=np.float64) + start
        naive = raw.astype(np.complex128) * np.exp(
            -2j * np.pi * np.mod(cycles * n, 1.0))
        m = raw.size // hd.DEFAULT_DECIMATE
        naive = naive[:m * hd.DEFAULT_DECIMATE].reshape(
            m, hd.DEFAULT_DECIMATE).mean(axis=1)
        fast = hd.mix_and_decimate(raw, cycles, start, hd.DEFAULT_DECIMATE)
        np.testing.assert_allclose(fast, naive, atol=1e-8)

    def test_maximum_likelihood_reaches_the_cramer_rao_bound(self):
        """Not 'good enough': within 25% of the bound nothing can beat."""
        for snr_db in (10.0, 20.0):
            bound = hd.crb_hz(10 ** (snr_db / 10.0), self.M, 1.0 / self.FSD)
            errs = []
            for trial in range(120):
                f0 = (trial % 17) * 3.1 - 25.0
                errs.append(hd.fine_ml(self._tone(f0, snr_db, seed=trial),
                                       self.FSD) - f0)
            rms = float(np.sqrt(np.mean(np.square(errs))))
            self.assertLess(rms, 1.25 * bound, f"{snr_db} dB: {rms} vs {bound}")
            self.assertGreater(rms, 0.5 * bound, "beating the CRB means a bug")

    def test_the_bound_itself_matches_its_closed_form(self):
        # 6 / ((2 pi)^2 * snr * N * (N^2-1) * Ts^2), taken apart independently.
        n, ts, snr = 1000, 1e-5, 10.0
        want = np.sqrt(6.0 / (4 * np.pi ** 2 * snr * n * (n * n - 1) * ts * ts))
        self.assertAlmostEqual(hd.crb_hz(snr, n, ts), want, places=12)
        # Halving the noise improves the bound by sqrt(2); doubling the
        # observation improves it by 2^1.5, which is the whole reason a
        # whole-dwell fit beats a per-frame one.
        self.assertAlmostEqual(hd.crb_hz(2 * snr, n, ts) / hd.crb_hz(snr, n, ts),
                               1 / np.sqrt(2), places=6)
        self.assertAlmostEqual(hd.crb_hz(snr, 2 * n, ts) / hd.crb_hz(snr, n, ts),
                               2 ** -1.5, places=3)

    def test_an_unknown_estimator_is_refused_by_name(self):
        with self.assertRaises(ValueError):
            hd.measure_visits(hd.Capture(np.zeros(4096, dtype=np.complex64), 512),
                              [], np.array([1.0]), 0.0, 0.0, FS,
                              estimator="clairvoyance")


class TestTheNewEstimatorBeatsTheOldOnIdenticalInput(unittest.TestCase):
    """Same samples, both estimators. The comparison is the point."""

    @classmethod
    def setUpClass(cls):
        cls.samples = hd.synthesise(seconds=1.0, offset_hz=OFFSET,
                                    start_at_s=EPOCH, snr_db=10.0, **PLAN)
        cls.old = hd.decode(cls.samples, estimator="peak", **PLAN)
        cls.new = hd.decode(cls.samples, estimator="ml", **PLAN)

    def test_both_recover_every_point(self):
        self.assertEqual(self.old.recovered, DEFAULT_POINTS)
        self.assertEqual(self.new.recovered, DEFAULT_POINTS)

    def test_the_point_to_point_spread_improves_by_more_than_a_hundredfold(self):
        self.assertGreater(self.old.spread_hz, 30.0)
        self.assertLess(self.new.spread_hz, self.old.spread_hz / 100.0)

    def test_the_worst_single_point_improves_by_more_than_a_hundredfold(self):
        old_worst = float(np.abs(self.old.errors_hz - OFFSET).max())
        new_worst = float(np.abs(self.new.errors_hz - OFFSET).max())
        self.assertGreater(old_worst, 30.0)
        self.assertLess(new_worst, old_worst / 100.0)

    def test_the_default_is_the_better_one(self):
        self.assertEqual(hd.DEFAULT_ESTIMATOR, "ml")
        self.assertEqual(hd.build_parser().parse_args([]).estimator, "ml")
        self.assertEqual(hd.decode(self.samples[:len(self.samples)//4],
                                   **PLAN).estimator, "ml")

    def test_the_old_estimator_is_still_reachable_for_comparison(self):
        self.assertIn("peak", hd.ESTIMATOR_NAMES)
        self.assertEqual(self.old.estimator, "peak")
        self.assertEqual(self.old.visits, 0)      # it has no notion of a visit

    def test_only_the_new_one_improves_when_the_capture_gets_longer(self):
        """The old floor is bias, so it does not move. The new one is noise."""
        longer = hd.synthesise(seconds=2.5, offset_hz=OFFSET, start_at_s=EPOCH,
                               snr_db=10.0, **PLAN)
        old = hd.decode(longer, estimator="peak", **PLAN)
        new = hd.decode(longer, estimator="ml", **PLAN)
        self.assertGreater(old.spread_hz, 0.5 * self.old.spread_hz)
        self.assertLess(new.spread_hz, 0.75 * self.new.spread_hz)

    def test_every_fine_estimator_beats_the_framed_one(self):
        for name in hd.FINE_ESTIMATORS:
            with self.subTest(estimator=name):
                got = hd.decode(self.samples, estimator=name, **PLAN)
                self.assertEqual(got.recovered, DEFAULT_POINTS)
                self.assertLess(got.spread_hz, self.old.spread_hz / 50.0)


class TestRecoveryToTheImprovedTolerance(unittest.TestCase):
    """The injected offset comes back, now to a fraction of a hertz."""

    @classmethod
    def setUpClass(cls):
        cls.result = hd.decode(
            hd.synthesise(seconds=2.0, offset_hz=OFFSET, start_at_s=EPOCH,
                          snr_db=10.0, **PLAN), **PLAN)

    def test_every_point_lands_within_a_hertz_of_the_injected_offset(self):
        # The old estimator was tested at 500 Hz, and could not have passed
        # this at any capture length: its error is bias, not noise.
        for row in self.result.rows:
            self.assertLess(abs(row.error_hz - OFFSET), 1.0, f"point {row.point}")

    def test_the_centre_is_pinned_to_a_tenth_of_a_hertz(self):
        self.assertLess(abs(self.result.median_error_hz - OFFSET), 0.1)

    def test_the_spread_is_under_half_a_hertz(self):
        self.assertLess(self.result.spread_hz, 0.5)

    def test_each_point_reports_a_standard_error_that_is_not_a_fiction(self):
        """The quoted per-point error must actually bound the real error."""
        for row in self.result.rows:
            self.assertGreater(row.visits_used, 1)
            self.assertTrue(np.isfinite(row.stderr_hz))
            self.assertLess(abs(row.error_hz - OFFSET), 12.0 * row.stderr_hz,
                            f"point {row.point} understates its own error")

    def test_the_summary_separates_spread_from_the_error_on_the_centre(self):
        self.assertTrue(np.isfinite(self.result.stderr_hz))
        self.assertLess(self.result.stderr_hz, self.result.spread_hz)

    def test_it_is_still_unwilling_to_be_confidently_wrong(self):
        self.assertTrue(self.result.trustworthy)
        self.assertEqual(self.result.warnings, [])


class TestGuardsAgainstTheRetuneSettling(unittest.TestCase):
    """A coherent fit has no median to spit out the settling chirp for it."""

    @classmethod
    def setUpClass(cls):
        chirped = hd.synthesise(seconds=1.2, offset_hz=OFFSET,
                                start_at_s=EPOCH, snr_db=20.0,
                                settle_s=1.2e-3, settle_hz=2000.0, **PLAN)
        cls.naked = hd.decode(chirped, guard_start_s=0.0, **PLAN)
        cls.default = hd.decode(chirped, **PLAN)
        cls.longer = hd.decode(chirped, guard_start_s=3.0e-3, **PLAN)
        cls.clean = hd.decode(
            hd.synthesise(seconds=1.2, offset_hz=OFFSET, start_at_s=EPOCH,
                          snr_db=20.0, **PLAN), **PLAN)

    def test_without_a_head_guard_the_settling_biases_every_point(self):
        self.assertGreater(abs(self.naked.median_error_hz - OFFSET), 2.0)

    def test_the_default_guard_removes_most_of_it_and_a_longer_one_all(self):
        self.assertLess(abs(self.default.median_error_hz - OFFSET),
                        abs(self.naked.median_error_hz - OFFSET))
        self.assertLess(abs(self.longer.median_error_hz - OFFSET), 0.2)

    def test_the_half_split_diagnostic_notices_and_the_run_says_so(self):
        self.assertGreater(abs(self.naked.settling_hz),
                           hd.SETTLING_SIGMA * self.naked.settling_stderr_hz)
        self.assertTrue(any("halves of a dwell" in w
                            for w in self.naked.warnings), self.naked.warnings)
        self.assertFalse(self.naked.trustworthy)

    def test_a_clean_capture_reports_no_settling_and_earns_no_warning(self):
        self.assertLess(abs(self.clean.settling_hz),
                        hd.SETTLING_SIGMA * self.clean.settling_stderr_hz)
        self.assertFalse(any("halves of a dwell" in w
                             for w in self.clean.warnings))

    def test_the_guard_can_never_eat_a_whole_dwell(self):
        """A guard longer than the dwell must clamp, not starve the fit."""
        freqs = plan_frequencies(DEFAULT_HOP_START_HZ, DEFAULT_HOP_STOP_HZ, 4)
        hops = make_schedule(DEFAULT_SEED, freqs, 0.002, 3)
        windows = hd.visit_windows(hops, 4, 1, 0.0, 1.0,
                                   guard_start_s=1.0, guard_end_s=1.0)
        self.assertTrue(windows)
        for _, t0, t1 in windows:
            self.assertGreater(t1 - t0, 0.002 * (1 - 2 * hd.MAX_GUARD_FRACTION)
                               - 1e-12)


class TestTheCommonDriftFit(unittest.TestCase):
    """The LNB LO walks; a repeating schedule turns that into a per-point offset."""

    @classmethod
    def setUpClass(cls):
        cls.samples = hd.synthesise(seconds=2.0, offset_hz=OFFSET,
                                    start_at_s=EPOCH, snr_db=20.0,
                                    drift_hz_s=4.5, **PLAN)
        cls.fitted = hd.decode(cls.samples, **PLAN)
        cls.unfitted = hd.decode(cls.samples, fit_drift=False, **PLAN)
        clean = hd.synthesise(seconds=1.6, offset_hz=OFFSET, start_at_s=EPOCH,
                              snr_db=20.0, **PLAN)
        cls.clean_fitted = hd.decode(clean, fit_drift=True, **PLAN)
        cls.clean_unfitted = hd.decode(clean, fit_drift=False, **PLAN)

    def test_the_injected_drift_rate_comes_back(self):
        self.assertAlmostEqual(self.fitted.drift_hz_s, 4.5, delta=0.2)

    def test_removing_it_tightens_the_spread_by_an_order_of_magnitude(self):
        self.assertEqual(self.unfitted.drift_hz_s, 0.0)
        self.assertLess(self.fitted.spread_hz, self.unfitted.spread_hz / 10.0)

    def test_a_capture_with_no_drift_is_not_given_one(self):
        self.assertLess(abs(self.clean_fitted.drift_hz_s), 0.5)

    def test_fitting_a_drift_that_is_not_there_costs_almost_nothing(self):
        self.assertLess(self.clean_fitted.spread_hz,
                        2.0 * self.clean_unfitted.spread_hz)


class TestCrossVisitPhaseCoherence(unittest.TestCase):
    """Whether visits can be combined coherently is measured, never assumed.

    It would be worth a great deal -- 150x here -- and it is not available on
    this chain. Both halves of that are asserted, because a claim that a
    technique does not apply is only worth anything if the same code can show
    it working when it does.
    """

    @classmethod
    def setUpClass(cls):
        common = dict(seconds=2.0, offset_hz=OFFSET, start_at_s=EPOCH,
                      snr_db=20.0, **PLAN)
        coherent = hd.synthesise(visit_phase="coherent", **common)
        random = hd.synthesise(visit_phase="random", **common)
        drifting = hd.synthesise(visit_phase="coherent", drift_hz_s=4.5,
                                 **common)
        cls.plain_coherent = hd.decode(coherent, **PLAN)
        cls.plain_random = hd.decode(random, **PLAN)
        cls.plain_drifting = hd.decode(drifting, **PLAN)
        cls.joint_coherent = hd.decode(coherent, combine="coherent", **PLAN)
        cls.joint_random = hd.decode(random, combine="coherent", **PLAN)
        cls.short = hd.decode(
            hd.synthesise(seconds=0.45, offset_hz=OFFSET, start_at_s=EPOCH,
                          snr_db=20.0, **PLAN), **PLAN)

    @staticmethod
    def _coherence(result):
        return (float(np.median([r.coherence for r in result.rows])),
                float(np.median([r.coherence_null for r in result.rows])))

    def test_a_free_running_transmitter_reads_as_coherent(self):
        got, null = self._coherence(self.plain_coherent)
        self.assertGreater(got, 0.9)
        # The null cannot be scaled against, only cleared: at ten visits it
        # already sits near 0.56, and twice that is above the 1.0 the statistic
        # is bounded by. The margin is what matters.
        self.assertGreater(got - null, 0.3)

    def test_a_transmitter_that_relocks_each_visit_does_not(self):
        got, null = self._coherence(self.plain_random)
        self.assertLess(got, null)

    def test_an_lnb_drifting_at_45_hz_per_second_destroys_it_anyway(self):
        """Even a perfectly coherent transmitter is not coherent through this LNB.

        Phase accumulates as the integral of frequency, so a linear drift puts
        a quadratic phase across the capture that no single common frequency
        can absorb. Coherent combination is unavailable here twice over.
        """
        got, null = self._coherence(self.plain_drifting)
        self.assertLess(got, null)

    def test_coherent_combination_is_refused_when_the_phases_are_not_related(self):
        self.assertTrue(any("refused" in w for w in self.joint_random.warnings),
                        self.joint_random.warnings)
        self.assertFalse(self.joint_random.trustworthy)
        # Refused means it fell all the way back, not partway.
        np.testing.assert_allclose(self.joint_random.errors_hz,
                                   self.plain_random.errors_hz, atol=1e-12)

    def test_coherent_combination_is_enormously_better_when_it_does_apply(self):
        self.assertEqual(self.joint_coherent.recovered, DEFAULT_POINTS)
        self.assertLess(self.joint_coherent.spread_hz,
                        self.plain_coherent.spread_hz / 10.0)
        self.assertEqual(self.joint_coherent.warnings, [])

    def test_the_gate_is_decided_once_over_every_point_rather_than_per_point(self):
        """A per-point gate lets one point in twenty through, and one is enough.

        A single point combined coherently when it should not be lands a whole
        hertz out and ruins the spread the other nineteen just earned, so the
        decision is taken on the ensemble.
        """
        self.assertEqual(self.joint_random.recovered, DEFAULT_POINTS)
        self.assertLess(self.joint_random.spread_hz, 0.5)

    def test_too_few_visits_to_judge_is_reported_as_such_not_as_a_verdict(self):
        self.assertTrue(all(np.isnan(r.coherence) for r in self.short.rows))
        self.assertNotIn("coherence", hd.format_report(self.short))


class TestTheEstimatorIsSelectableEndToEnd(unittest.TestCase):
    def test_the_cli_offers_every_estimator_and_refuses_the_rest(self):
        parser = hd.build_parser()
        for name in hd.ESTIMATOR_NAMES:
            self.assertEqual(parser.parse_args(["--estimator", name]).estimator,
                             name)
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                parser.parse_args(["--estimator", "wishful"])

    def test_the_cli_offers_both_ways_of_combining_visits(self):
        parser = hd.build_parser()
        self.assertEqual(parser.parse_args([]).combine, "incoherent")
        self.assertEqual(parser.parse_args(["--combine", "coherent"]).combine,
                         "coherent")

    def test_the_report_names_the_estimator_and_its_uncertainty(self):
        result = hd.decode(
            hd.synthesise(seconds=1.0, offset_hz=OFFSET, start_at_s=EPOCH,
                          snr_db=20.0, **PLAN), **PLAN)
        text = hd.format_report(result)
        self.assertIn("estimator", text)
        self.assertIn("whole dwell", text)
        self.assertIn("standard error", text)
        self.assertIn("dwell halves", text)

    def test_a_run_through_main_defaults_to_the_new_estimator(self):
        with tempfile.TemporaryDirectory() as work:
            path = os.path.join(work, "good.iq")
            hd.write_int16(path, hd.synthesise(seconds=0.6, offset_hz=OFFSET,
                                               start_at_s=EPOCH, snr_db=20.0,
                                               **PLAN))
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                status = hd.main(["--capture", path, "--fs", str(FS)])
        self.assertEqual(status, 0)
        self.assertIn("estimator    : ml", buffer.getvalue())


class TestTheSynthesiserModelsWhatTheHardwareDoes(unittest.TestCase):
    """The offline proof is only worth the honesty of the signal it runs on."""

    def test_visits_default_to_independent_phase_because_a_pll_relocks(self):
        import inspect
        self.assertEqual(
            inspect.signature(hd.synthesise).parameters["visit_phase"].default,
            "random")

    def test_a_bad_phase_model_is_refused_rather_than_silently_ignored(self):
        with self.assertRaises(ValueError):
            hd.synthesise(seconds=0.1, visit_phase="hopeful", **PLAN)

    def test_the_settling_chirp_actually_appears_in_the_samples(self):
        plain = hd.synthesise(seconds=0.3, offset_hz=OFFSET, start_at_s=0.0,
                              snr_db=60.0, **PLAN)
        chirped = hd.synthesise(seconds=0.3, offset_hz=OFFSET, start_at_s=0.0,
                                snr_db=60.0, settle_s=1.2e-3,
                                settle_hz=2000.0, **PLAN)
        self.assertFalse(np.allclose(plain, chirped))

    def test_drift_moves_the_measured_offset_by_the_amount_injected(self):
        """Referred to the capture centre, so 4.5 Hz/s over 2 s reads +4.5 Hz."""
        result = hd.decode(
            hd.synthesise(seconds=2.0, offset_hz=OFFSET, start_at_s=EPOCH,
                          snr_db=20.0, drift_hz_s=4.5, **PLAN), **PLAN)
        self.assertAlmostEqual(result.median_error_hz - OFFSET, 4.5, delta=0.5)


class TestItChecksItselfAgainstTheBound(unittest.TestCase):
    """The decoder reports how close it got to the limit, on YOUR capture.

    Every precision claim made for this estimator was measured on synthetic
    signals, and nobody running it on a radio has to take those on trust: the
    weights are inverse variances from the Cramer-Rao bound, so the bound's
    prediction for the actual SNR and dwell of the actual capture comes out for
    free, and printing it beside the scatter that was really achieved turns the
    claim into something each run either confirms or contradicts.
    """

    @classmethod
    def setUpClass(cls):
        cls.clean = hd.decode(
            hd.synthesise(seconds=2.0, offset_hz=OFFSET, start_at_s=EPOCH,
                          snr_db=10.0, **PLAN), **PLAN)
        drifting = hd.synthesise(seconds=2.0, offset_hz=OFFSET,
                                 start_at_s=EPOCH, snr_db=20.0,
                                 drift_hz_s=4.5, **PLAN)
        cls.unmodelled = hd.decode(drifting, fit_drift=False, **PLAN)
        cls.modelled = hd.decode(drifting, fit_drift=True, **PLAN)

    def test_a_clean_capture_reports_sitting_on_the_bound(self):
        self.assertGreater(self.clean.excess_scatter, 0.6)
        self.assertLess(self.clean.excess_scatter, 1.6)

    def test_the_predicted_precision_actually_predicts_the_real_error(self):
        """The bound's promise, checked against the error we can see."""
        predicted = float(np.median([r.crb_hz for r in self.clean.rows]))
        actual = float(np.sqrt(np.mean((self.clean.errors_hz - OFFSET) ** 2)))
        self.assertLess(actual, 2.5 * predicted)
        self.assertGreater(actual, 0.4 * predicted)

    def test_something_unmodelled_shows_up_as_excess_scatter_and_warns(self):
        self.assertGreater(self.unmodelled.excess_scatter,
                           hd.MAX_EXCESS_SCATTER)
        self.assertTrue(any("scatter" in w for w in self.unmodelled.warnings),
                        self.unmodelled.warnings)
        self.assertFalse(self.unmodelled.trustworthy)

    def test_modelling_it_clears_both_the_ratio_and_the_warning(self):
        self.assertLess(self.modelled.excess_scatter, hd.MAX_EXCESS_SCATTER)
        self.assertFalse(any("scatter" in w for w in self.modelled.warnings))

    def test_the_ratio_is_withheld_when_there_are_too_few_visits_to_judge(self):
        short = hd.decode(
            hd.synthesise(seconds=0.45, offset_hz=OFFSET, start_at_s=EPOCH,
                          snr_db=20.0, **PLAN), **PLAN)
        self.assertTrue(np.isnan(short.excess_scatter))
        self.assertNotIn("vs the bound", hd.format_report(short))

    def test_a_standard_deviation_from_few_samples_is_corrected_for_its_bias(self):
        """c4(n), against the values every quality-control table prints."""
        for n, want in ((2, 0.7979), (5, 0.9400), (10, 0.9727), (25, 0.9896)):
            self.assertAlmostEqual(hd.sd_bias_factor(n), want, places=4)
        self.assertTrue(np.isnan(hd.sd_bias_factor(1)))
        # Monotone up to 1, and never over it.
        seen = [hd.sd_bias_factor(n) for n in range(2, 200)]
        self.assertTrue(all(b < a <= 1.0 for a, b in zip(seen[1:], seen)))
        self.assertAlmostEqual(hd.sd_bias_factor(100000), 1.0, places=5)

    def test_without_that_correction_it_would_look_like_beating_the_bound(self):
        """The reason the correction is there, stated as a test.

        Five visits give c4 = 0.94, and a raw sample sd reads about 30% low
        once the median over twenty points is taken -- which would print as
        0.7x the Cramer-Rao limit and invite exactly the wrong conclusion.
        """
        self.assertLess(hd.sd_bias_factor(5), 0.95)
        self.assertGreaterEqual(hd.MIN_VISITS_FOR_SCATTER, 8)


if __name__ == "__main__":
    unittest.main()
