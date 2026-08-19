"""The frequency lever arm: separating d_rx from d_lnb, and knowing how well.

Nothing here touches a radio. Two levels are tested, deliberately:

* the whole chain, from synthesised I/Q through the cluster decoder to the
  final fit, on a small but complete run -- so the parts cannot drift apart;
* the fit on its own, from records synthesised directly, where a hundred
  captures cost a millisecond instead of a minute and every error source can be
  injected one at a time and turned off again.

The failures that matter for this measurement are not noise, they are
CONFIDENT WRONG ANSWERS: a tuning bias that is not really being averaged, a
curved response mistaken for a clock error, a lever that is quietly tilted. So
several tests inject exactly those and assert that the report refuses the run
rather than reporting a tight number.
"""
import contextlib
import importlib.util
import io
import json
import os
import re
import sys
import tempfile
import unittest

import numpy as np

from . import context  # noqa: F401
from adf5355.hopper import (DEFAULT_BAND_EXTRA_S, DEFAULT_BLOCK,
                            DEFAULT_CLUSTER_POINTS, DEFAULT_CLUSTER_SPAN_HZ,
                            DEFAULT_SEED, GOLOMB_RULERS, cluster_centres,
                            cluster_hops, describe_clusters,
                            make_cluster_schedule, make_schedule,
                            period_duration, plan_clusters, run_hops)


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(context.ROOT, "tools", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


hd = _load("hop_decode")
lf = _load("lever_fit")
lr = _load("lever_run")

LO = 9.75e9
D_RX = 8.94e-6
D_LNB = 9.641e-6
CLUSTERS_IF = [0.95e9, 1.35e9, 1.75e9, 2.15e9]


def standard_plan(clusters=4, points=DEFAULT_CLUSTER_POINTS,
                  span=DEFAULT_CLUSTER_SPAN_HZ):
    return plan_clusters(cluster_centres(10.70e9, 11.90e9, clusters), points,
                         span)


# ---------------------------------------------------------------------------
# the schedule
# ---------------------------------------------------------------------------
class ClusterPlanTests(unittest.TestCase):
    def test_centres_span_the_whole_lever(self):
        centres = cluster_centres(10.7e9, 11.9e9, 4)
        self.assertEqual(centres[0], 10_700_000_000)
        self.assertEqual(centres[-1], 11_900_000_000)
        self.assertEqual(len(set(np.diff(centres))), 1)

    def test_one_cluster_is_the_midpoint(self):
        self.assertEqual(cluster_centres(10.7e9, 11.9e9, 1), [11_300_000_000])

    def test_points_are_centred_on_the_cluster(self):
        plan = standard_plan()
        freqs = plan.freqs(0)
        self.assertEqual(len(freqs), DEFAULT_CLUSTER_POINTS)
        self.assertEqual((freqs[0] + freqs[-1]) // 2, plan.centres_hz[0])
        self.assertEqual(plan.span_hz(), DEFAULT_CLUSTER_SPAN_HZ)

    def test_the_points_are_a_golomb_ruler_not_a_grid(self):
        # A regular comb has an alias one spacing away scoring (P-1)/P, and
        # a receiver that takes it mislabels every point. Distinct pairwise
        # differences remove the alias entirely.
        plan = standard_plan()
        offsets = np.array(plan.offsets)
        diffs = offsets[:, None] - offsets[None, :]
        upper = diffs[np.triu_indices(len(offsets), 1)]
        self.assertEqual(len(set(upper.tolist())), len(upper))
        self.assertLessEqual(plan.coincidence(), 1.0 / plan.points + 1e-9)

    def test_a_uniform_comb_would_have_been_ambiguous(self):
        uniform = np.arange(DEFAULT_CLUSTER_POINTS) * 80_000.0
        self.assertGreater(hd.comb_ambiguity(uniform, 5_000.0), 0.8)
        self.assertLess(hd.comb_ambiguity(np.array(standard_plan().offsets,
                                                   dtype=float), 5_000.0), 0.2)

    def test_too_many_points_for_the_span_is_refused_with_a_reason(self):
        with self.assertRaises(ValueError) as ctx:
            plan_clusters(cluster_centres(10.7e9, 11.9e9, 2), 40, 720_000)
        self.assertIn("Golomb ruler", str(ctx.exception))
        self.assertIn(str(max(GOLOMB_RULERS)), str(ctx.exception))

    def test_the_closest_pair_stays_wide_enough_to_resolve(self):
        plan = standard_plan()
        self.assertGreater(plan.min_gap_hz(), 40_000)

    def test_overlapping_clusters_are_refused(self):
        with self.assertRaises(ValueError):
            plan_clusters([10.70e9, 10.700_5e9], 6, 720_000)

    def test_duplicate_centres_are_refused(self):
        with self.assertRaises(ValueError):
            plan_clusters([10.7e9, 10.7e9], 6, 720_000)

    def test_lever_is_the_distance_between_the_outermost(self):
        self.assertEqual(standard_plan().lever_hz(), 1_200_000_000)

    def test_if_nom_subtracts_the_lo(self):
        plan = standard_plan()
        self.assertAlmostEqual(plan.if_nom(0, LO)[0], plan.freqs(0)[0] - LO)


# The cluster schedule is a PROTOCOL between two programs that never speak to
# each other: a transmitter that has been running for an hour and a receiver
# started afterwards, possibly from a different checkout. Structural tests --
# "every pair appears once", "blocks do not straddle clusters" -- are all
# satisfied by a schedule that is simply a different one, and a receiver
# regenerating a different schedule mislabels every point and returns a
# confident wrong answer. Only a recorded answer catches that, so here it is,
# for the shipped defaults: seed 0xC0FFEE, 4 clusters over 10.70-11.90 GHz, 6
# points on a 720 kHz Golomb ruler, 10 ms dwells in blocks of 3, 5 ms of band
# allowance. Changing any of these numbers is a protocol change and both ends
# have to move together.
REFERENCE_RULER_HZ = (0, 42000, 169000, 424000, 508000, 720000)
REFERENCE_CLUSTER_ORDER = (1, 1, 1, 3, 3, 3, 2, 2, 2, 0, 0, 0,
                           3, 3, 3, 2, 2, 2, 1, 1, 1, 0, 0, 0)
REFERENCE_POINT_ORDER = (1, 4, 0, 4, 3, 2, 3, 2, 4, 5, 0, 1,
                         0, 1, 5, 0, 5, 1, 3, 2, 5, 3, 2, 4)
REFERENCE_CLUSTER0_HZ = (10699640000, 10699682000, 10699809000,
                         10700064000, 10700148000, 10700360000)


class ClusterProtocolPinTests(unittest.TestCase):
    """Golden vectors. A different schedule that passes every other test is
    still a different schedule, and on the air it is a mislabelling."""

    def test_the_golomb_ruler_is_pinned(self):
        self.assertEqual(standard_plan().offsets, REFERENCE_RULER_HZ)

    def test_one_period_of_the_cluster_schedule_is_pinned(self):
        plan = standard_plan()
        hops = make_cluster_schedule(DEFAULT_SEED, plan, 0.010, 2,
                                     DEFAULT_BLOCK, 0.0, 1,
                                     DEFAULT_BAND_EXTRA_S)
        per = plan.clusters * plan.points
        self.assertEqual(tuple(h.cluster for h in hops[:per]),
                         REFERENCE_CLUSTER_ORDER)
        self.assertEqual(tuple(h.point for h in hops[:per]),
                         REFERENCE_POINT_ORDER)
        self.assertEqual(tuple(round(h.dwell_s, 9) for h in hops[:per]),
                         tuple(0.015 if i % DEFAULT_BLOCK == 0 else 0.010
                               for i in range(per)))

    def test_the_transmitted_frequencies_are_pinned(self):
        self.assertEqual(tuple(standard_plan().freqs(0)),
                         REFERENCE_CLUSTER0_HZ)


class ClusterScheduleTests(unittest.TestCase):
    def setUp(self):
        self.plan = standard_plan()
        self.hops = make_cluster_schedule(DEFAULT_SEED, self.plan, 0.010, 5)
        self.per_cycle = self.plan.clusters * self.plan.points

    def test_every_pair_appears_exactly_once_per_cycle(self):
        for cycle in range(3):
            window = self.hops[cycle * self.per_cycle:
                               (cycle + 1) * self.per_cycle]
            pairs = {(h.cluster, h.point) for h in window}
            self.assertEqual(len(pairs), self.per_cycle)

    def test_the_pattern_repeats_every_period(self):
        for i in range(self.per_cycle):
            a, b = self.hops[i], self.hops[i + self.per_cycle]
            self.assertEqual((a.cluster, a.point), (b.cluster, b.point))
            self.assertAlmostEqual(a.dwell_s, b.dwell_s)

    def test_the_same_seed_gives_the_same_schedule(self):
        again = make_cluster_schedule(DEFAULT_SEED, self.plan, 0.010, 5)
        self.assertEqual(self.hops, again)

    def test_a_different_seed_gives_a_different_schedule(self):
        other = make_cluster_schedule(DEFAULT_SEED + 1, self.plan, 0.010, 5)
        self.assertNotEqual([h.point for h in self.hops[:20]],
                            [h.point for h in other[:20]])

    def test_band_changes_only_at_block_boundaries(self):
        block = DEFAULT_BLOCK
        for i, hop in enumerate(self.hops[:self.per_cycle * 2]):
            self.assertEqual(hop.band_change, i % block == 0,
                             f"hop {i} band_change wrong")

    def test_a_block_never_changes_cluster_inside_itself(self):
        block = DEFAULT_BLOCK
        for start in range(0, self.per_cycle * 2, block):
            window = self.hops[start:start + block]
            self.assertEqual(len({h.cluster for h in window}), 1)

    def test_band_change_dwells_are_longer_by_exactly_the_allowance(self):
        plain = [h for h in self.hops[:self.per_cycle] if not h.band_change]
        changed = [h for h in self.hops[:self.per_cycle] if h.band_change]
        self.assertTrue(plain and changed)
        for h in changed:
            self.assertAlmostEqual(h.settle_s, DEFAULT_BAND_EXTRA_S)
            self.assertAlmostEqual(h.dwell_s,
                                   plain[0].dwell_s + DEFAULT_BAND_EXTRA_S)
        for h in plain:
            self.assertEqual(h.settle_s, 0.0)

    def test_every_cluster_is_visited_once_per_round(self):
        block = DEFAULT_BLOCK
        per_round = self.plan.clusters * block
        for start in range(0, self.per_cycle * 2, per_round):
            window = self.hops[start:start + per_round]
            self.assertEqual({h.cluster for h in window},
                             set(range(self.plan.clusters)))

    def test_no_cluster_is_ever_starved(self):
        # The worst gap between one cluster's dwells bounds how badly the
        # LNB's drift can differ between the clusters inside one capture.
        for c in range(self.plan.clusters):
            starts = np.array([h.start_s for h in cluster_hops(self.hops, c)])
            self.assertLess(np.max(np.diff(starts)), 0.30)

    def test_block_must_divide_the_points(self):
        with self.assertRaises(ValueError) as ctx:
            make_cluster_schedule(DEFAULT_SEED, self.plan, 0.010, 2, block=4)
        self.assertIn("does not divide", str(ctx.exception))

    def test_period_duration_needs_the_cycle_length(self):
        got = period_duration(self.hops, self.plan.points, 1, self.per_cycle)
        self.assertAlmostEqual(got, self.hops[self.per_cycle].start_s)

    def test_single_cluster_schedule_is_untouched(self):
        # The legacy path must keep every hop in one band and never ask for a
        # VCO search, or the proven single-cluster measurement would change.
        hops = make_schedule(DEFAULT_SEED, [11_000_000_000, 11_000_090_000],
                             0.010, 3)
        self.assertTrue(all(h.cluster == 0 and not h.band_change
                            and h.settle_s == 0.0 for h in hops))

    def test_describe_names_the_lever_and_the_duty(self):
        text = describe_clusters(self.plan, self.hops, DEFAULT_SEED, 0.010)
        self.assertIn("lever arm", text)
        self.assertIn("1/4 of the time", text)


class FakeDevice:
    """Records what a transmitter would have written, without a bus."""

    def __init__(self):
        self.fast = []
        self.autocal = []
        self.outputs = []

    def precompute(self, freqs, channel):
        return [("r1", f, "r0") for f in freqs]

    def set_frequency(self, freq, channel, autocal=True):
        self.autocal.append(freq)

    def apply_precomputed(self, triple):
        self.fast.append(triple[1])

    def retune(self, plan, autocal=True):
        self.autocal.append(plan)

    def set_output(self, channel, enabled):
        self.outputs.append(enabled)


class RunHopsTests(unittest.TestCase):
    def test_band_changes_get_a_real_retune_and_the_rest_do_not(self):
        plan = standard_plan(clusters=2, points=4)
        hops = make_cluster_schedule(DEFAULT_SEED, plan, 0.0005, 1, block=2)
        plans = {f: f for f in plan.all_freqs}
        dev = FakeDevice()
        run_hops(dev, hops, channel=None, settle_s=0.0, autocal_plans=plans)
        # The first hop is programmed before the loop; every later band change
        # is a full retune and every other hop is dividers only.
        wanted = [h.freq_hz for h in hops[1:] if h.band_change]
        self.assertEqual(dev.autocal[1:], wanted)
        self.assertEqual(dev.fast,
                         [h.freq_hz for h in hops[1:] if not h.band_change])
        self.assertEqual(dev.outputs, [True, False])

    def test_without_plans_nothing_autocals_mid_run(self):
        plan = standard_plan(clusters=2, points=4)
        hops = make_cluster_schedule(DEFAULT_SEED, plan, 0.0005, 1, block=2)
        dev = FakeDevice()
        run_hops(dev, hops, channel=None, settle_s=0.0)
        self.assertEqual(len(dev.autocal), 1)
        self.assertEqual(len(dev.fast), len(hops) - 1)


# ---------------------------------------------------------------------------
# one capture of one cluster
# ---------------------------------------------------------------------------
# Small on purpose: a real capture is 2.5 MS/s for three seconds, and testing
# at that size would cost minutes per assertion for nothing. The numbers below
# keep every ratio that matters -- points per passband, frames per dwell,
# periods per capture -- and shrink only the absolute scale.
FS = 500e3
FRAME = 128
SPAN = 120_000
POINTS = 4
BLOCK = 2
DWELL = 0.005
BAND_EXTRA = 0.002
SECONDS = 0.5
DECIMATE = 8


def small_plan(clusters=4):
    return plan_clusters(cluster_centres(10.70e9, 11.90e9, clusters), POINTS,
                         SPAN)


def tuning(plan, cluster, dither=0.0, d_rx=D_RX, d_lnb=D_LNB):
    if_c = float(plan.centres_hz[cluster]) - LO
    predicted = float(hd.reported_if_hz(if_c, lo_hz=LO, d_rx=d_rx,
                                        d_lnb=d_lnb)) - if_c
    return if_c + predicted + dither


class OneClusterDecodeTests(unittest.TestCase):
    def decode_one(self, cluster, *, dither=0.0, seconds=SECONDS,
                   noise_seed=3, **synth):
        plan = small_plan()
        centre = tuning(plan, cluster, dither)
        x = hd.synthesise_cluster(
            plan, cluster, fs=FS, centre_hz=centre, min_hop_s=DWELL,
            block=BLOCK, band_extra_s=BAND_EXTRA, lo_hz=LO, seconds=seconds,
            start_at_s=0.0371, d_rx=D_RX, d_lnb=D_LNB, snr_db=12.0,
            noise_seed=noise_seed, **synth)
        return plan, centre, hd.decode(
            x, fs=FS, centre_hz=centre, lo_hz=LO, cluster_plan=plan,
            cluster=cluster, min_hop_s=DWELL, block=BLOCK,
            band_extra_s=BAND_EXTRA, frame=FRAME, decimate=DECIMATE)

    def test_every_point_of_the_chosen_cluster_comes_back(self):
        for cluster in range(4):
            with self.subTest(cluster=cluster):
                plan, _, r = self.decode_one(cluster, noise_seed=7 + cluster)
                self.assertEqual(r.recovered, POINTS, r.warnings)
                self.assertGreater(r.comb_sharpness, hd.MIN_COMB_SHARPNESS)
                self.assertGreater(r.epoch_sigma, hd.MIN_EPOCH_SIGMA)
                self.assertEqual(r.warnings, [])
                self.assertEqual(r.cluster, cluster)
                self.assertAlmostEqual(r.cluster_centre_hz,
                                       plan.centres_hz[cluster])

    def test_the_offset_it_reports_is_the_injected_physics(self):
        plan, _, r = self.decode_one(2)
        want = float(hd.reported_if_hz(r.mean_if_hz, lo_hz=LO, d_rx=D_RX,
                                       d_lnb=D_LNB)) - r.mean_if_hz
        self.assertAlmostEqual(r.mean_error_hz, want, delta=2.0)

    def test_the_within_capture_slope_is_minus_d_rx(self):
        plan, _, r = self.decode_one(3)
        # A short synthetic dwell makes this a poor estimate of d_rx and a
        # perfectly good check that it is not the WRONG quantity.
        self.assertAlmostEqual(r.slope, -D_RX / (1 + D_RX), delta=3e-5)

    def test_a_dither_moves_the_tuning_and_not_the_answer(self):
        _, c0, near = self.decode_one(1, dither=0.0)
        _, c1, far = self.decode_one(1, dither=90e3)
        self.assertNotAlmostEqual(c0, c1, delta=1.0)
        self.assertAlmostEqual(near.mean_error_hz, far.mean_error_hz, delta=3.0)

    def test_other_clusters_are_never_confused_for_this_one(self):
        # Decoding a capture of cluster 0 while claiming it is cluster 1 must
        # not produce a confident answer -- that is the failure that would
        # silently corrupt a whole lever arm.
        plan = small_plan()
        centre = tuning(plan, 0)
        x = hd.synthesise_cluster(plan, 0, fs=FS, centre_hz=centre,
                                  min_hop_s=DWELL, block=BLOCK,
                                  band_extra_s=BAND_EXTRA, lo_hz=LO,
                                  seconds=SECONDS, start_at_s=0.0371,
                                  d_rx=D_RX, d_lnb=D_LNB, snr_db=12.0)
        r = hd.decode(x, fs=FS, centre_hz=centre, lo_hz=LO, cluster_plan=plan,
                      cluster=1, min_hop_s=DWELL, block=BLOCK,
                      band_extra_s=BAND_EXTRA, frame=FRAME, decimate=DECIMATE)
        self.assertTrue(r.warnings)
        self.assertFalse(r.trustworthy)

    def test_a_violent_band_change_transient_is_guarded_away(self):
        # 400 Hz of settling on every band-changing dwell, decaying over the
        # whole extra allowance. If the guard did not cover it the capture's
        # offset would be tens of hertz out.
        _, _, clean = self.decode_one(2, noise_seed=21)
        _, _, dirty = self.decode_one(2, noise_seed=21,
                                      band_settle_s=BAND_EXTRA,
                                      band_settle_hz=400.0)
        self.assertAlmostEqual(dirty.mean_error_hz, clean.mean_error_hz,
                               delta=2.0)
        self.assertEqual(dirty.warnings, [])

    def test_dropping_band_change_dwells_gives_the_same_answer(self):
        plan = small_plan()
        centre = tuning(plan, 1)
        common = dict(fs=FS, centre_hz=centre, lo_hz=LO, cluster_plan=plan,
                      cluster=1, min_hop_s=DWELL, block=BLOCK,
                      band_extra_s=BAND_EXTRA, frame=FRAME, decimate=DECIMATE)
        x = hd.synthesise_cluster(plan, 1, fs=FS, centre_hz=centre,
                                  min_hop_s=DWELL, block=BLOCK,
                                  band_extra_s=BAND_EXTRA, lo_hz=LO,
                                  seconds=SECONDS, start_at_s=0.0371,
                                  d_rx=D_RX, d_lnb=D_LNB, snr_db=12.0)
        kept = hd.decode(x, **common)
        dropped = hd.decode(x, drop_band_change=True, **common)
        self.assertLess(dropped.visits, kept.visits)
        self.assertAlmostEqual(dropped.mean_error_hz, kept.mean_error_hz,
                               delta=3.0)

    def test_dropping_band_changes_costs_whole_points_so_it_is_no_remedy(self):
        # The schedule repeats, so whichever point leads a block leads it every
        # period. Throwing those dwells away throws the point away with them --
        # points//block of them -- and the capture is disowned for partial
        # recovery. It is a diagnostic, and the docs must not offer it as the
        # fix; the fix is a longer --band-extra-ms at BOTH ends.
        plan = standard_plan()
        centre = float(np.mean(plan.if_nom(1, LO)))
        common = dict(fs=FS, centre_hz=centre, lo_hz=LO, cluster_plan=plan,
                      cluster=1, min_hop_s=DWELL, block=BLOCK,
                      band_extra_s=BAND_EXTRA, frame=FRAME, decimate=DECIMATE)
        x = hd.synthesise_cluster(plan, 1, fs=FS, centre_hz=centre,
                                  min_hop_s=DWELL, block=BLOCK,
                                  band_extra_s=BAND_EXTRA, lo_hz=LO,
                                  seconds=SECONDS, start_at_s=0.0371,
                                  d_rx=D_RX, d_lnb=D_LNB, snr_db=12.0)
        kept = hd.decode(x, **common)
        dropped = hd.decode(x, drop_band_change=True, **common)
        self.assertEqual(kept.recovered, plan.points)
        self.assertEqual(dropped.recovered, plan.points - plan.points // BLOCK)
        self.assertFalse(dropped.trustworthy)
        self.assertTrue(any("points recovered" in w for w in dropped.warnings),
                        dropped.warnings)
        # and the help says so, so an operator is not sent down that road
        self.assertIn("DIAGNOSTIC", hd.build_parser().format_help())

    def test_the_passband_warning_fires_when_the_dither_is_too_wide(self):
        _, _, r = self.decode_one(0, dither=180e3)
        self.assertTrue(any("half-passband" in w for w in r.warnings),
                        r.warnings)


class ReduceCaptureTests(unittest.TestCase):
    def rows(self, mean, slope, ifs):
        centre = float(np.mean(ifs))
        return [hd.PointResult(point=i, nominal_if_hz=f,
                               measured_if_hz=f + mean + slope * (f - centre),
                               error_hz=mean + slope * (f - centre),
                               frames_used=1, frames_assigned=1,
                               envelope_db=20.0, crb_hz=0.1)
                for i, f in enumerate(ifs)]

    def test_a_straight_line_comes_back_exactly(self):
        ifs = [1.0e9 + i * 80e3 for i in range(10)]
        got = hd.reduce_capture(self.rows(-1234.5, -9e-6, ifs))
        self.assertAlmostEqual(got["mean_error_hz"], -1234.5, places=6)
        self.assertAlmostEqual(got["slope"], -9e-6, places=12)
        self.assertAlmostEqual(got["mean_if_hz"], float(np.mean(ifs)), places=3)
        self.assertLess(got["chi2_scale"], 1e-6)

    def test_a_capture_that_disagrees_with_itself_widens_its_own_error_bars(self):
        ifs = [1.0e9 + i * 80e3 for i in range(10)]
        rows = self.rows(0.0, 0.0, ifs)
        for i, r in enumerate(rows):          # 1 Hz of scatter, 10x the CRB
            r.error_hz += (-1.0) ** i
        got = hd.reduce_capture(rows)
        self.assertGreater(got["chi2_scale"], 5.0)
        self.assertGreater(got["mean_error_stderr_hz"], 0.15)

    def test_too_few_points_report_nothing_rather_than_a_guess(self):
        self.assertEqual(hd.reduce_capture([]), {})


# ---------------------------------------------------------------------------
# the fit, on records synthesised directly
# ---------------------------------------------------------------------------
def run_records(**kw):
    kw.setdefault("clusters_if_hz", CLUSTERS_IF)
    kw.setdefault("lo_hz", LO)
    kw.setdefault("d_rx", D_RX)
    kw.setdefault("d_lnb", D_LNB)
    return lf.synthesise_run(**kw)


class ForwardModelTests(unittest.TestCase):
    def test_the_two_forward_models_agree(self):
        # The capture synthesiser and the fit must be describing the same
        # physics, or every test here would be checking one copy of a mistake
        # against another.
        kw = dict(lo_hz=LO, d_rx=D_RX, d_lnb=D_LNB, d_tx=1.5e-6,
                  tuning_bias_hz=37.0, drift_hz_s=4.5, t_rel_s=123.0)
        for if_nom in (0.95e9, 1.75e9, 2.15e9):
            self.assertAlmostEqual(float(hd.reported_if_hz(if_nom, **kw)),
                                   float(lf.reported_if_model(if_nom, **kw)),
                                   places=9)

    def test_the_inversion_is_exact_not_linearised(self):
        # The cross term d_rx*d_lnb*f_LO is about 0.85 Hz on this hardware,
        # twenty times the per-capture precision, so a linearised inversion
        # would be visible.
        ifs = np.array(CLUSTERS_IF)
        df = lf.reported_if_model(ifs, lo_hz=LO, d_rx=D_RX, d_lnb=D_LNB) - ifs
        slope, intercept = np.polyfit(ifs, df, 1)
        self.assertAlmostEqual(lf.d_rx_from_slope(slope), D_RX, places=13)
        self.assertAlmostEqual(
            lf.d_lnb_from_intercept(intercept, slope, LO), D_LNB, places=13)

    def test_a_linearised_inversion_would_have_been_wrong(self):
        ifs = np.array(CLUSTERS_IF)
        df = lf.reported_if_model(ifs, lo_hz=LO, d_rx=D_RX, d_lnb=D_LNB) - ifs
        slope, intercept = np.polyfit(ifs, df, 1)
        naive = -intercept / LO                      # forgetting the 1/(1+d_rx)
        self.assertGreater(abs(naive - D_LNB) * LO, 0.5)


class VisitPlanTests(unittest.TestCase):
    def test_every_sweep_holds_every_cluster_exactly_once(self):
        visits = lf.plan_visits(1234567, 4, 20)
        self.assertEqual(len(visits), 80)
        for s in range(20):
            got = [v.cluster for v in visits if v.sweep == s]
            self.assertEqual(sorted(got), [0, 1, 2, 3])

    def test_no_cluster_is_visited_twice_in_a_row(self):
        visits = lf.plan_visits(99, 4, 30)
        for a, b in zip(visits, visits[1:]):
            self.assertNotEqual(a.cluster, b.cluster)

    def test_the_order_is_redrawn_rather_than_fixed(self):
        visits = lf.plan_visits(1234567, 4, 30)
        orders = {tuple(v.cluster for v in visits if v.sweep == s)
                  for s in range(30)}
        self.assertGreater(len(orders), 6)

    def test_the_dither_covers_its_window_and_repeats_from_the_seed(self):
        visits = lf.plan_visits(1234567, 4, 40, dither_hz=450e3)
        d = np.array([v.dither_hz for v in visits])
        self.assertLessEqual(np.abs(d).max(), 450e3)
        self.assertGreater(np.abs(d).max(), 350e3)
        self.assertLess(abs(d.mean()), 90e3)
        again = lf.plan_visits(1234567, 4, 40, dither_hz=450e3)
        self.assertEqual([v.dither_hz for v in visits],
                         [v.dither_hz for v in again])

    def test_one_cluster_is_no_lever_arm_at_all(self):
        with self.assertRaises(ValueError):
            lf.plan_visits(1, 1, 10)


class CleanFitTests(unittest.TestCase):
    def test_both_come_back_exactly_when_nothing_is_wrong(self):
        recs = run_records(sweeps=10, sigma_bias_hz=0.0, sigma_est_hz=0.0,
                           sigma_slope=0.0, drift_hz_s=0.0)
        rep = lf.analyse(recs, lo_hz=LO, draws=60)
        self.assertAlmostEqual(rep.res.d_rx, D_RX, places=12)
        self.assertAlmostEqual(rep.res.d_lnb, D_LNB, places=12)
        self.assertTrue(rep.trustworthy, rep.warnings)

    def test_drift_is_fitted_out_and_reported(self):
        recs = run_records(sweeps=15, sigma_bias_hz=0.0, sigma_est_hz=0.0,
                           sigma_slope=0.0, drift_hz_s=4.5)
        rep = lf.analyse(recs, lo_hz=LO, draws=60)
        self.assertAlmostEqual(rep.res.d_rx, D_RX, places=12)
        self.assertAlmostEqual(rep.res.d_lnb, D_LNB, places=11)
        self.assertAlmostEqual(rep.res.drift_hz_s, 4.5, places=6)

    def test_d_rx_survives_a_drift_no_polynomial_could_follow(self):
        # A warm-up curve, not a polynomial: the sweep model has to be immune
        # to it by construction rather than by fitting it, and this is the
        # test that says so. d_lnb, which needs the drift MODELLED, is not
        # immune -- which is exactly why the two are fitted differently.
        recs = run_records(sweeps=20, sigma_bias_hz=0.0, sigma_est_hz=0.0,
                           sigma_slope=0.0, drift_hz_s=0.0)
        t = np.array([r.t_abs_s for r in recs])
        warm = 3000.0 * (1.0 - np.exp(-(t - t.min()) / 90.0))
        for r, extra in zip(recs, warm):
            r.mean_error_hz -= extra
        rep = lf.analyse(recs, lo_hz=LO, draws=60)
        # 3 kHz of warm-up, and what is left on d_rx is a few parts per
        # billion: the free offset per sweep removes all of it except the
        # curvature WITHIN one sweep, which the within-sweep slope then takes
        # most of as well.
        self.assertLess(abs(rep.est.d_rx_wide - D_RX), 1e-8)
        # d_lnb has no such protection -- it needs the drift modelled, and a
        # warm-up curve is not a quadratic. This is why the two are fitted
        # differently and why only d_rx is called immune.
        self.assertGreater(abs(rep.res.d_lnb - D_LNB) * LO, 50.0)


class TuningBiasTests(unittest.TestCase):
    def test_d_rx_and_d_lnb_both_survive_the_tuning_bias(self):
        recs = run_records(sweeps=25, sigma_bias_hz=127.0, drift_hz_s=4.5,
                           noise_seed=5)
        rep = lf.analyse(recs, lo_hz=LO, draws=200)
        self.assertTrue(rep.trustworthy, rep.warnings)
        self.assertLess(abs(rep.res.d_rx - D_RX), 3.0 * rep.res.d_rx_stderr)
        self.assertLess(abs(rep.res.d_rx - D_RX), 0.08e-6)
        self.assertLess(abs(rep.res.d_lnb - D_LNB),
                        3.0 * rep.res.d_lnb_stderr)

    def test_the_bias_size_is_measured_rather_than_assumed(self):
        recs = run_records(sweeps=40, sigma_bias_hz=127.0, noise_seed=9)
        rep = lf.analyse(recs, lo_hz=LO, draws=60)
        self.assertAlmostEqual(rep.est.sigma_bias_hz, 127.0, delta=25.0)

    def test_the_quoted_uncertainty_matches_what_the_design_allows(self):
        recs = run_records(sweeps=25, sigma_bias_hz=127.0, noise_seed=11)
        rep = lf.analyse(recs, lo_hz=LO, draws=300)
        self.assertLess(abs(np.log(rep.res.d_rx_stderr / rep.design_stderr)),
                        np.log(1.8))

    def test_precision_improves_as_the_square_root_of_the_sweeps(self):
        few = lf.analyse(run_records(sweeps=9, sigma_bias_hz=127.0,
                                     noise_seed=3), lo_hz=LO, draws=150)
        many = lf.analyse(run_records(sweeps=81, sigma_bias_hz=127.0,
                                      noise_seed=3), lo_hz=LO, draws=150)
        ratio = few.res.d_rx_stderr / many.res.d_rx_stderr
        self.assertGreater(ratio, 2.0)        # 3x is the ideal
        self.assertLess(ratio, 4.5)

    def test_a_covariance_that_believes_the_decoder_understates_it_tenfold(self):
        # This is the whole reason the uncertainty comes from a resample. The
        # decoder's own standard errors are the ESTIMATOR's, a few hundredths
        # of a hertz; the scatter that governs the answer is the tuning bias,
        # a hundred hertz. A covariance built on the former is not slightly
        # optimistic, it is optimistic by orders of magnitude.
        recs = run_records(sweeps=25, sigma_bias_hz=127.0, noise_seed=13)
        data = lf.Dataset.from_records(recs)
        naive = lf.fit_wide(data, model="sweep", sigma_bias_hz=0.0)
        rep = lf.analyse(recs, lo_hz=LO, draws=150)
        self.assertGreater(rep.res.d_rx_stderr / naive.slope_stderr, 10.0)

    def test_the_bootstrap_interval_actually_covers_the_truth(self):
        # A 95% interval that covers 60% of the time would be worse than no
        # interval at all, because it would be believed.
        hits = 0
        trials = 25
        for k in range(trials):
            recs = run_records(sweeps=25, sigma_bias_hz=127.0, drift_hz_s=4.5,
                               noise_seed=500 + k)
            rep = lf.analyse(recs, lo_hz=LO, draws=120, seed=17 + k)
            lo_ci, hi_ci = rep.res.d_rx_ci
            hits += lo_ci <= D_RX <= hi_ci
        # A wider study -- 120 runs of 25 sweeps -- measured 93% against a
        # nominal 95%, and 71% at 5 sweeps, which is why MIN_SWEEPS is 8.
        # Twenty-five trials cannot resolve 93 from 95, so this asserts only
        # that the interval is not grossly optimistic.
        self.assertGreaterEqual(hits, 20)


class ArithmeticThatNoOtherTestPinsTests(unittest.TestCase):
    """Small load-bearing choices that a wrong answer would slip past.

    Every test here was written because MUTATING the line it covers left all
    343 other tests passing. A rule with no test that can fail is a comment.
    """

    def test_the_quoted_uncertainty_is_the_larger_of_the_two_resamples(self):
        # The bootstrap is optimistic with few sweeps and the jackknife is
        # poor on a rough statistic; they are both run precisely so the WIDER
        # can be quoted. Quoting the narrower instead is a silent, plausible,
        # always-smaller answer.
        recs = run_records(sweeps=25, sigma_bias_hz=127.0, noise_seed=71)
        data = lf.Dataset.from_records(recs)
        res = lf.resample(data, LO, draws=200, seed=3)
        self.assertGreater(res.d_rx_bootstrap_stderr, 0.0)
        self.assertGreater(res.d_rx_jackknife_stderr, 0.0)
        self.assertAlmostEqual(
            res.d_rx_stderr,
            max(res.d_rx_bootstrap_stderr, res.d_rx_jackknife_stderr),
            places=12)
        self.assertGreaterEqual(res.d_lnb_stderr, res.d_lnb_jackknife_stderr)
        # ...and the two really do differ, so the max is not a no-op here.
        self.assertGreater(
            max(res.d_rx_bootstrap_stderr, res.d_rx_jackknife_stderr)
            / min(res.d_rx_bootstrap_stderr, res.d_rx_jackknife_stderr),
            1.02)

    def test_a_sweep_drawn_twice_becomes_two_sweeps(self):
        # The bootstrap draws sweeps with replacement, and the fit gives one
        # free offset per sweep. A duplicate that keeps its original label
        # would leave the fit with fewer parameters than the model it claims
        # to be, and the tuning-bias variance component divides by exactly
        # that count.
        recs = run_records(sweeps=12, sigma_bias_hz=127.0, noise_seed=73)
        data = lf.Dataset.from_records(recs)
        sweeps = np.unique(data.sweep)
        drawn = [sweeps[0], sweeps[0], sweeps[0]] + list(sweeps[1:])
        got = data.by_sweep(np.array(drawn))
        self.assertEqual(len(got), sum(int((data.sweep == s).sum())
                                       for s in drawn))
        self.assertEqual(np.unique(got.sweep).size, len(drawn))

    def test_the_variance_component_uses_the_model_s_degrees_of_freedom(self):
        # Method of moments: chi-square lands on its DOF, not on its point
        # count. Dividing by n instead reads the tuning bias low by
        # sqrt(n/dof) -- 14% at 25 sweeps -- and quietly shrinks the design
        # standard error the operator plans the next run with.
        rng = np.random.default_rng(7)
        n, dof = 400, 317
        resid = rng.normal(0.0, 100.0, n)
        got = lf._variance_component(resid, np.zeros(n), dof)
        self.assertAlmostEqual(got, float(np.sqrt((resid ** 2).sum() / dof)),
                               places=6)
        self.assertGreater(got / float(np.sqrt((resid ** 2).sum() / n)), 1.10)
        # and with a known per-point error it subtracts it rather than adding
        sv = np.full(n, 30.0)
        both = lf._variance_component(resid, sv, dof)
        self.assertAlmostEqual(
            both, float(np.sqrt(max((resid ** 2).sum() / dof - 900.0, 0.0))),
            places=6)

    def test_the_split_half_test_is_really_three_sigma(self):
        # Each half is fitted from half the sweeps, so it carries sqrt(2)
        # times the whole run's standard error; the halves are independent, so
        # their DIFFERENCE scatters by 2x it. Scaling by sqrt(2) instead turns
        # a nominal 3-sigma check into a 2.1-sigma one, which fails about one
        # blameless run in thirty. Measured on 120 synthetic runs at 25
        # sweeps: the difference scattered 1.97x the quoted sigma.
        self.assertAlmostEqual(lf.SPLIT_HALF_SCALE, 2.0, places=12)
        self.assertAlmostEqual(lf.SPLIT_HALF_SIGMA, 3.0, places=12)
        diffs = []
        for k in range(24):
            recs = run_records(sweeps=25, sigma_bias_hz=127.0,
                               noise_seed=900 + k)
            data = lf.Dataset.from_records(recs)
            _, _, d = lf.split_half(data, lf.DEFAULT_DRIFT_ORDER, False)
            diffs.append(d)
        quoted = lf.resample(lf.Dataset.from_records(
            run_records(sweeps=25, sigma_bias_hz=127.0, noise_seed=900)),
            LO, draws=200, seed=5).d_rx_stderr
        ratio = float(np.std(diffs, ddof=1)) / quoted
        self.assertGreater(ratio, 1.5)        # emphatically not sqrt(2)
        self.assertLess(ratio, 2.6)

    def test_a_bias_that_trends_with_the_tuning_lands_straight_on_d_rx(self):
        # The one systematic no resample here can see. The dither randomises
        # the tuning bias LOCALLY, over a few hundred kHz; a component that
        # varies smoothly across the 1.2 GHz of lever arm is not randomised at
        # all and enters d_rx one for one. The two-lever comparison is the
        # only check on it, and the report has to say what a pass is worth.
        base = run_records(sweeps=25, sigma_bias_hz=127.0, noise_seed=61)
        clean = lf.analyse(base, lo_hz=LO, draws=150)
        for k in (0.05e-6, 0.4e-6):
            tilted = run_records(sweeps=25, sigma_bias_hz=127.0, noise_seed=61)
            for r in tilted:
                r.mean_error_hz -= k * (r.rx_lo_hz - 1.55e9)
            rep = lf.analyse(tilted, lo_hz=LO, draws=150)
            moved = rep.res.d_rx - clean.res.d_rx
            self.assertAlmostEqual(moved, k, delta=0.05 * k)   # one for one
            caught = any("levers disagree" in w for w in rep.warnings)
            self.assertEqual(
                caught, k > lf.NARROW_AGREE_SIGMA * rep.agreement_denom,
                f"k={k*1e6:.3f} ppm against a threshold of "
                f"{lf.NARROW_AGREE_SIGMA*rep.agreement_denom*1e6:.3f} ppm")

    def test_the_report_quotes_the_accuracy_the_check_actually_applies(self):
        # The printed "certifies only to about X ppm" must be the threshold
        # the two-lever test really uses, not a second calculation of it.
        recs = run_records(sweeps=25, sigma_bias_hz=127.0, noise_seed=63)
        rep = lf.analyse(recs, lo_hz=LO, draws=100)
        text = lf.format_report(rep)
        want = lf.NARROW_AGREE_SIGMA * rep.agreement_denom * 1e6
        self.assertIn(f"{want:.3f} ppm", text)
        self.assertIn("READ THIS LINE FIRST", text)
        self.assertGreater(want, rep.res.d_rx_stderr * 1e6)
        self.assertAlmostEqual(rep.to_dict()["agreement_threshold_ppm"], want,
                               places=9)

    def test_the_comb_margin_floor_scales_with_what_the_plan_allows(self):
        # The mis-lock guard. A Golomb ruler of six points may be held to a
        # real threshold because its geometry allows 6x; a uniform comb of
        # twenty may not, because its geometry allows only 1.05x and holding
        # it to more would condemn it for something it cannot help.
        tol = 2.5e6 / 512
        golomb = np.asarray(standard_plan().if_nom(0, LO))
        uniform = np.linspace(0.95e9, 0.95e9 + 1.71e6, 20)
        floor_g = hd.comb_margin_floor(golomb, tol)
        floor_u = hd.comb_margin_floor(uniform, tol)
        self.assertAlmostEqual(floor_g, 1.0 + 0.3 * (6.0 - 1.0), delta=0.05)
        self.assertLess(floor_u, 1.05)
        self.assertGreater(floor_g, 2.0)      # a zeroed fraction gives 1.0
        # and a real capture clears its own floor with room to spare
        centre = float(np.mean(golomb))
        iq = hd.synthesise_cluster(standard_plan(), 0, fs=2.5e6,
                                   centre_hz=centre, seconds=1.2,
                                   snr_db=10.0, noise_seed=4)
        got = hd.decode(iq, fs=2.5e6, centre_hz=centre, lo_hz=LO,
                        cluster_plan=standard_plan(), cluster=0)
        self.assertGreater(got.comb_margin, floor_g)


class RefusalTests(unittest.TestCase):
    """The failures that would otherwise be confident wrong answers."""

    def test_a_curved_response_is_caught_by_the_linearity_check(self):
        # 300 Hz sitting on the middle two clusters and nothing on the ends.
        # No resample can find this -- it is a bias, not a scatter -- so the
        # only defence is that the clusters stop sitting on a line.
        recs = run_records(sweeps=25, sigma_bias_hz=127.0,
                           cluster_offset_hz=[0.0, 300.0, 300.0, 0.0],
                           noise_seed=17)
        rep = lf.analyse(recs, lo_hz=LO, draws=60)
        self.assertFalse(rep.trustworthy)
        self.assertTrue(any("straight line" in w for w in rep.warnings),
                        rep.warnings)

    def test_a_tilted_narrow_lever_is_caught_by_the_two_levers_disagreeing(self):
        recs = run_records(sweeps=25, sigma_bias_hz=127.0,
                           narrow_bias=5.0e-7, noise_seed=19)
        rep = lf.analyse(recs, lo_hz=LO, draws=60)
        self.assertFalse(rep.trustworthy)
        self.assertTrue(any("two levers disagree" in w for w in rep.warnings),
                        rep.warnings)
        # and the wide lever, which is what is quoted, is still right
        self.assertLess(abs(rep.res.d_rx - D_RX), 3 * rep.res.d_rx_stderr)

    def test_a_smooth_tuning_bias_is_caught_by_the_variogram(self):
        # A bias that varies smoothly with rx_lo instead of roughly: nearby
        # tunings then agree, dithering draws correlated biases rather than
        # independent ones, and the sqrt(N) being credited is not real.
        recs = run_records(sweeps=25, sigma_bias_hz=0.0, noise_seed=23)
        for r in recs:
            r.mean_error_hz -= 300.0 * np.sin(r.rx_lo_hz / 4.0e6)
        rep = lf.analyse(recs, lo_hz=LO, draws=60)
        self.assertTrue(any("SMOOTH" in w for w in rep.warnings), rep.warnings)

    def test_too_few_sweeps_is_said_out_loud(self):
        recs = run_records(sweeps=5, sigma_bias_hz=127.0)
        rep = lf.analyse(recs, lo_hz=LO, draws=60)
        self.assertTrue(any("sweeps is under" in w for w in rep.warnings),
                        rep.warnings)

    def test_a_receiver_that_drifts_between_the_halves_is_said_out_loud(self):
        recs = run_records(sweeps=25, sigma_bias_hz=20.0, noise_seed=29)
        # A clock that moves by a part in 10^6 halfway through the run.
        t = np.array([r.t_abs_s for r in recs])
        for r, late in zip(recs, t > np.median(t)):
            if late:
                r.mean_error_hz -= 1.0e-6 * r.mean_if_hz
        rep = lf.analyse(recs, lo_hz=LO, draws=120)
        self.assertTrue(any("moved between the halves" in w
                            for w in rep.warnings), rep.warnings)

    def test_captures_the_decoder_disowned_are_dropped_and_declared(self):
        recs = run_records(sweeps=25, sigma_bias_hz=127.0)
        recs[3].trustworthy = False
        recs[3].warnings = ["comb not found"]
        rep = lf.analyse(recs, lo_hz=LO, draws=60)
        self.assertEqual(rep.captures, len(recs) - 1)
        self.assertEqual(rep.rejected, 1)
        self.assertTrue(any("disowned" in w for w in rep.warnings))

    def test_a_run_with_one_cluster_cannot_separate_anything(self):
        recs = run_records(sweeps=12, sigma_bias_hz=10.0)
        only = [r for r in recs if r.cluster == 0]
        with self.assertRaises(ValueError) as ctx:
            lf.analyse(only, lo_hz=LO, draws=30)
        self.assertIn("no lever arm", str(ctx.exception))


class NarrowLeverTests(unittest.TestCase):
    def test_the_narrow_lever_is_a_witness_by_default(self):
        recs = run_records(sweeps=25, sigma_bias_hz=127.0, noise_seed=31)
        rep = lf.analyse(recs, lo_hz=LO, narrow="check", draws=60)
        self.assertFalse(rep.use_narrow)
        self.assertAlmostEqual(rep.res.d_rx, rep.est.d_rx_wide, places=12)
        self.assertLess(abs(rep.est.d_rx_narrow - D_RX), 1e-8)

    def test_combining_is_available_and_tightens_the_answer(self):
        recs = run_records(sweeps=25, sigma_bias_hz=127.0, noise_seed=31)
        witness = lf.analyse(recs, lo_hz=LO, narrow="check", draws=120)
        joined = lf.analyse(recs, lo_hz=LO, narrow="combine", draws=120)
        self.assertTrue(joined.use_narrow)
        self.assertLess(joined.res.d_rx_stderr, witness.res.d_rx_stderr)
        self.assertLess(abs(joined.res.d_rx - D_RX), 5 * joined.res.d_rx_stderr)

    def test_off_ignores_it_entirely(self):
        recs = run_records(sweeps=12, sigma_bias_hz=127.0)
        rep = lf.analyse(recs, lo_hz=LO, narrow="off", draws=30)
        self.assertFalse(rep.use_narrow)
        self.assertIn("switched off", rep.narrow_reason)

    def test_an_unknown_mode_is_refused(self):
        with self.assertRaises(ValueError):
            lf.analyse(run_records(sweeps=12), lo_hz=LO, narrow="maybe")


class RecordIoTests(unittest.TestCase):
    def test_records_survive_a_round_trip(self):
        recs = run_records(sweeps=6, sigma_bias_hz=127.0)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "run.jsonl")
            lf.write_records(path, {"lo_hz": LO, "note": "hello"}, recs)
            header, back = lf.read_records(path)
            self.assertEqual(header["lo_hz"], LO)
            self.assertEqual(header["note"], "hello")
            self.assertEqual(len(back), len(recs))
            for a, b in zip(recs, back):
                self.assertEqual(a.cluster, b.cluster)
                self.assertAlmostEqual(a.mean_error_hz, b.mean_error_hz)
                self.assertAlmostEqual(a.rx_lo_hz, b.rx_lo_hz)

    def test_unknown_fields_in_a_file_are_ignored_not_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "run.jsonl")
            rec = run_records(sweeps=2)[0]
            with open(path, "w") as fh:
                fh.write(json.dumps({"kind": "header", "lo_hz": LO}) + "\n")
                body = {"kind": "capture", "from_a_later_version": 1}
                body.update({k: getattr(rec, k)
                             for k in lf.CaptureRecord.__dataclass_fields__})
                fh.write(json.dumps(body) + "\n")
            _, back = lf.read_records(path)
            self.assertEqual(len(back), 1)

    def test_the_report_says_both_answers_and_its_own_limits(self):
        recs = run_records(sweeps=25, sigma_bias_hz=127.0, drift_hz_s=4.5)
        text = lf.format_report(lf.analyse(recs, lo_hz=LO, draws=60))
        for wanted in ("d_rx", "d_lnb", "lever arm", "tuning bias",
                       "variogram", "per cluster", "jackknife"):
            self.assertIn(wanted, text)

    def test_the_json_form_carries_the_numbers_a_script_would_want(self):
        recs = run_records(sweeps=25, sigma_bias_hz=127.0)
        got = lf.analyse(recs, lo_hz=LO, draws=60).to_dict()
        for key in ("d_rx", "d_rx_stderr", "d_rx_ci95", "d_lnb",
                    "d_lnb_stderr_hz", "sigma_bias_hz", "trustworthy"):
            self.assertIn(key, got)
        json.dumps(got)                     # must be serialisable as it stands


# ---------------------------------------------------------------------------
# the whole chain: I/Q in, d_rx and d_lnb out
# ---------------------------------------------------------------------------
class EndToEndTests(unittest.TestCase):
    """From synthesised captures, through the cluster decoder, to the answer.

    Small -- four clusters, four points, half-second captures at 500 kS/s --
    because a real run is a hundred captures of three seconds at 2.5 MS/s and
    testing at that scale would cost an hour. Every ratio that governs whether
    the chain works is preserved: points per passband, frames per dwell,
    periods per capture, clusters per sweep, captures per cluster. Only the
    absolute precision is worse, and the assertions are scaled to it.
    """

    SWEEPS = 6

    def do_run(self, *, drift_hz_s=0.0, bias_hz=0.0, clusters=4, sweeps=None,
               seed=4242, snr_db=12.0):
        plan = small_plan(clusters)
        visits = lf.plan_visits(1234567, clusters, sweeps or self.SWEEPS,
                                dither_hz=120e3)
        rng = np.random.default_rng(seed)
        overhead = 6.0
        synthetic = {
            "d_rx": D_RX, "d_lnb": D_LNB, "drift_hz_s": drift_hz_s,
            "snr_db": snr_db, "overhead_s": overhead,
            "t_ref_s": (len(visits) - 1) * (SECONDS + overhead) / 2.0
            + SECONDS / 2.0,
            "bias": (rng.normal(0.0, bias_hz, len(visits)) if bias_hz
                     else np.zeros(len(visits)))}
        records = lr.run(plan, visits, lo_hz=LO, fs=FS, seconds=SECONDS,
                         seed=DEFAULT_SEED, min_hop_s=DWELL, block=BLOCK,
                         jitter=0.0, period_cycles=1,
                         band_extra_s=BAND_EXTRA, frame=FRAME,
                         d_rx_guess=D_RX, lo_error_hz=D_LNB * LO,
                         synthetic=synthetic, decimate=DECIMATE,
                         progress=lambda *a, **k: None)
        return records

    def test_every_capture_decodes_and_both_errors_come_back(self):
        records = self.do_run()
        self.assertTrue(all(r.trustworthy for r in records),
                        [r.warnings for r in records if not r.trustworthy])
        rep = lf.analyse(records, lo_hz=LO, draws=120)
        self.assertEqual(rep.captures, 4 * self.SWEEPS)
        self.assertLess(abs(rep.res.d_rx - D_RX), 0.05e-6)
        self.assertLess(abs(rep.res.d_lnb - D_LNB) * LO, 30.0)

    def test_both_survive_the_lnb_drifting_under_the_measurement(self):
        records = self.do_run(drift_hz_s=4.5)
        self.assertTrue(all(r.trustworthy for r in records))
        rep = lf.analyse(records, lo_hz=LO, draws=120)
        self.assertLess(abs(rep.res.d_rx - D_RX), 0.05e-6)
        self.assertLess(abs(rep.res.d_lnb - D_LNB) * LO, 60.0)
        self.assertAlmostEqual(rep.res.drift_hz_s, 4.5, delta=0.6)

    def test_both_survive_a_per_capture_tuning_bias(self):
        # The measured 362 Hz peak to peak is about 127 Hz sd. With only six
        # sweeps the design allows 127/sqrt(6*8e17) = 0.058 ppm on d_rx, so
        # the assertion is scaled to that and NOT to what a full run gives.
        records = self.do_run(bias_hz=127.0, seed=77)
        rep = lf.analyse(records, lo_hz=LO, draws=200)
        self.assertLess(abs(rep.res.d_rx - D_RX), 3.0 * rep.res.d_rx_stderr)
        self.assertLess(abs(rep.res.d_rx - D_RX), 0.2e-6)
        self.assertLess(abs(rep.res.d_lnb - D_LNB),
                        3.0 * rep.res.d_lnb_stderr)
        self.assertGreater(rep.est.sigma_bias_hz, 60.0)
        self.assertLess(rep.est.sigma_bias_hz, 260.0)

    def test_drift_and_tuning_bias_together(self):
        records = self.do_run(drift_hz_s=4.5, bias_hz=127.0, seed=99)
        rep = lf.analyse(records, lo_hz=LO, draws=200)
        self.assertLess(abs(rep.res.d_rx - D_RX), 3.0 * rep.res.d_rx_stderr)
        self.assertLess(abs(rep.res.d_lnb - D_LNB),
                        3.5 * rep.res.d_lnb_stderr)

    def test_the_tuning_bias_makes_the_resample_far_wider_than_a_covariance(self):
        records = self.do_run(bias_hz=127.0, seed=77)
        data = lf.Dataset.from_records(records)
        naive = lf.fit_wide(data, model="sweep", sigma_bias_hz=0.0)
        rep = lf.analyse(records, lo_hz=LO, draws=200)
        self.assertGreater(rep.res.d_rx_stderr / naive.slope_stderr, 10.0)


# ---------------------------------------------------------------------------
# the operator-facing pieces
# ---------------------------------------------------------------------------
class TuningTests(unittest.TestCase):
    def test_the_receiver_is_centred_on_where_the_comb_will_be(self):
        # The comb sits about 100 kHz below the nominal IF, by an amount that
        # itself varies with the cluster. Centring on the prediction is what
        # leaves the dither the whole of the remaining passband to move in.
        for if_c in CLUSTERS_IF:
            rx = lr.tuning_for(if_c, 0.0, lo_hz=LO, d_rx_guess=D_RX,
                               lo_error_hz=D_LNB * LO)
            comb = float(lf.reported_if_model(if_c, lo_hz=LO, d_rx=D_RX,
                                              d_lnb=D_LNB))
            self.assertAlmostEqual(rx, comb, delta=1.0)

    def test_the_dither_moves_the_tuning_by_exactly_the_dither(self):
        a = lr.tuning_for(1.35e9, 0.0, lo_hz=LO, d_rx_guess=D_RX,
                          lo_error_hz=D_LNB * LO)
        b = lr.tuning_for(1.35e9, 137_000.0, lo_hz=LO, d_rx_guess=D_RX,
                          lo_error_hz=D_LNB * LO)
        self.assertAlmostEqual(b - a, 137_000.0, places=6)

    def test_a_wrong_guess_only_moves_the_tuning(self):
        # The guesses centre the receiver and nothing else. A guess that is
        # 5 ppm out must not change what is measured, only where it is heard.
        good = lr.tuning_for(2.15e9, 0.0, lo_hz=LO, d_rx_guess=D_RX,
                             lo_error_hz=D_LNB * LO)
        bad = lr.tuning_for(2.15e9, 0.0, lo_hz=LO, d_rx_guess=D_RX + 5e-6,
                            lo_error_hz=D_LNB * LO)
        self.assertLess(abs(good - bad), 12_000.0)


class RunnerCliTests(unittest.TestCase):
    def run_main(self, argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = lr.main(argv)
        return code, buf.getvalue()

    def test_the_default_is_a_dry_run_that_opens_nothing(self):
        args = lr.build_parser().parse_args([])
        self.assertFalse(args.open_radio)
        self.assertFalse(args.synthetic)
        code, text = self.run_main([])
        self.assertEqual(code, 0)
        self.assertIn("DRY RUN", text)
        self.assertIn("no radio was opened", text)

    def test_the_dry_run_shows_the_lever_and_the_dither(self):
        _, text = self.run_main([])
        self.assertIn("lever arm", text)
        self.assertIn("dither", text)
        self.assertIn("Golomb ruler", text)

    def test_it_warns_when_the_dither_would_push_points_out_of_band(self):
        _, text = self.run_main(["--dither-khz", "900"])
        self.assertIn("outside the passband", text)

    def test_it_warns_when_the_capture_is_under_two_periods(self):
        _, text = self.run_main(["--seconds", "0.2"])
        self.assertIn("under two periods", text)


class TransmitCliTests(unittest.TestCase):
    def run_cli(self, argv):
        from adf5355 import cli
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cli.main(argv)
        return code, buf.getvalue()

    def test_hop_lever_without_enable_rf_transmits_nothing(self):
        code, text = self.run_cli(["hop-lever"])
        self.assertEqual(code, 0)
        self.assertIn("RF disabled", text)
        self.assertIn("nothing was transmitted", text)

    def test_hop_lever_shouts_about_the_satellite_band(self):
        _, text = self.run_cli(["hop-lever"])
        self.assertIn("satellite downlink band", text)
        self.assertIn("do not radiate", text)

    def test_hop_lever_prints_everything_the_receiver_needs(self):
        _, text = self.run_cli(["hop-lever"])
        self.assertIn("the receiver needs only", text)
        for token in ("seed", "clusters", "points", "block", "band-extra",
                      "period-cycles"):
            self.assertIn(token, text)

    def test_hop_lever_checks_the_plan_across_the_whole_lever(self):
        _, text = self.run_cli(["hop-lever"])
        self.assertIn("plan error", text)

    def test_a_block_that_does_not_divide_the_points_is_refused(self):
        from adf5355 import cli
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), \
                contextlib.redirect_stderr(io.StringIO()):
            code = cli.main(["hop-lever", "--block", "4"])
        self.assertEqual(code, 2)

    def test_a_frequency_off_the_lnb_band_is_still_planned_or_refused_cleanly(self):
        from adf5355 import cli
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            code = cli.main(["hop-lever", "--low-ghz", "20", "--high-ghz", "21"])
        self.assertEqual(code, 2)


# ---------------------------------------------------------------------------
# the two operator scripts are one protocol
# ---------------------------------------------------------------------------
LEVER_TX = os.path.join(context.ROOT, "adf5355_rf_lever.sh")
LEVER_RX = os.path.join(context.ROOT, "sdr_lever.sh")

# The settings that define the schedule. Both ends must carry every one, at the
# same value, and must pass every one on. Nothing is transmitted between the
# two programs: this block IS the agreement.
LEVER_SCHEDULE_KEYS = ("SEED", "LOW_GHZ", "HIGH_GHZ", "CLUSTERS",
                       "CLUSTER_POINTS", "SPAN_KHZ", "HOP_MS", "BLOCK",
                       "BAND_EXTRA_MS", "JITTER", "PERIOD_CYCLES")
SETTING = re.compile(r'^([A-Z_][A-Z0-9_]*)="\$\{\1:-([^}]*)\}"', re.MULTILINE)


def read_script(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def script_settings(path):
    return dict(SETTING.findall(read_script(path)))


def exec_line(path):
    text = read_script(path).replace("\\\n", " ")
    for line in text.splitlines():
        if line.startswith("exec "):
            return " ".join(line.split())
    raise AssertionError(f"{path} has no exec line")


class LeverScriptTests(unittest.TestCase):
    def setUp(self):
        self.tx = script_settings(LEVER_TX)
        self.rx = script_settings(LEVER_RX)

    def test_both_scripts_define_every_schedule_setting(self):
        for key in LEVER_SCHEDULE_KEYS:
            self.assertIn(key, self.tx, f"{key} missing from the transmitter")
            self.assertIn(key, self.rx, f"{key} missing from the receiver")

    def test_every_shared_setting_has_the_same_default(self):
        for key in sorted(set(self.tx) & set(self.rx)):
            self.assertEqual(self.tx[key], self.rx[key],
                             f"{key} differs: transmitter {self.tx[key]!r} vs "
                             f"receiver {self.rx[key]!r}")

    def test_every_schedule_setting_reaches_both_ends(self):
        for key in LEVER_SCHEDULE_KEYS:
            self.assertIn(f'"${key}"', exec_line(LEVER_TX),
                          f"{key} never reaches adf5355 hop-lever")
            self.assertIn(f'"${key}"', exec_line(LEVER_RX),
                          f"{key} never reaches the runner")

    def test_the_transmitter_drives_hop_lever_with_rf_enabled(self):
        line = exec_line(LEVER_TX)
        self.assertIn('"$ADF" hop-lever', line)
        self.assertIn("--enable-rf", line)

    def test_the_receiver_drives_the_runner_and_passes_its_chain_on(self):
        line = exec_line(LEVER_RX)
        self.assertIn('"$RUNNER_PY"', line)
        for key in ("LO_HZ", "LO_ERROR_HZ", "FS", "SECONDS_LISTEN", "GAIN",
                    "URI", "SWEEPS", "DITHER_KHZ", "VISIT_SEED", "OUT"):
            self.assertIn(f'"${key}"', line, f"{key} never reaches the runner")

    def test_the_receiver_opens_no_radio_unless_told_to(self):
        text = read_script(LEVER_RX)
        self.assertIn("OPEN_RADIO", text)
        self.assertIn("--open-radio", text)
        # The flag must be conditional, never in the unconditional exec line.
        self.assertNotIn("--open-radio", exec_line(LEVER_RX))

    def test_the_defaults_match_the_package(self):
        self.assertEqual(int(self.tx["SEED"], 0), DEFAULT_SEED)
        self.assertEqual(int(self.tx["CLUSTER_POINTS"]), DEFAULT_CLUSTER_POINTS)
        self.assertEqual(round(float(self.tx["SPAN_KHZ"]) * 1e3),
                         DEFAULT_CLUSTER_SPAN_HZ)
        self.assertEqual(int(self.tx["BLOCK"]), DEFAULT_BLOCK)
        self.assertEqual(float(self.tx["BAND_EXTRA_MS"]) / 1e3,
                         DEFAULT_BAND_EXTRA_S)

    def test_the_defaults_match_both_command_lines(self):
        from adf5355 import cli
        tx = cli.build_parser().parse_args(["hop-lever"])
        self.assertEqual(int(self.tx["SEED"], 0), tx.seed)
        self.assertEqual(float(self.tx["LOW_GHZ"]), tx.low_ghz)
        self.assertEqual(float(self.tx["HIGH_GHZ"]), tx.high_ghz)
        self.assertEqual(int(self.tx["CLUSTERS"]), tx.clusters)
        self.assertEqual(float(self.tx["SPAN_KHZ"]), tx.span_khz)
        self.assertEqual(int(self.tx["BLOCK"]), tx.block)
        rx = lr.build_parser().parse_args([])
        self.assertEqual(int(self.rx["CLUSTERS"]), rx.clusters)
        self.assertEqual(float(self.rx["SPAN_KHZ"]), rx.span_khz)
        self.assertEqual(float(self.rx["FS"]), rx.fs)
        self.assertEqual(float(self.rx["SECONDS_LISTEN"]), rx.seconds)
        self.assertEqual(int(self.rx["SWEEPS"]), rx.sweeps)
        self.assertEqual(float(self.rx["DITHER_KHZ"]), rx.dither_khz)

    def test_the_block_divides_the_points(self):
        self.assertEqual(int(self.tx["CLUSTER_POINTS"]) % int(self.tx["BLOCK"]),
                         0)

    def test_the_cluster_and_its_dither_fit_the_passband(self):
        reach = (float(self.rx["SPAN_KHZ"]) / 2
                 + float(self.rx["DITHER_KHZ"])) * 1e3
        self.assertLess(reach, float(self.rx["FS"]) * 0.4)

    def test_the_capture_is_at_least_two_periods(self):
        plan = plan_clusters(
            cluster_centres(float(self.rx["LOW_GHZ"]) * 1e9,
                            float(self.rx["HIGH_GHZ"]) * 1e9,
                            int(self.rx["CLUSTERS"])),
            int(self.rx["CLUSTER_POINTS"]),
            round(float(self.rx["SPAN_KHZ"]) * 1e3))
        hop_s = float(self.rx["HOP_MS"]) / 1e3
        hops = make_cluster_schedule(int(self.rx["SEED"], 0), plan, hop_s,
                                     int(self.rx["PERIOD_CYCLES"]) + 1,
                                     int(self.rx["BLOCK"]),
                                     float(self.rx["JITTER"]),
                                     int(self.rx["PERIOD_CYCLES"]),
                                     float(self.rx["BAND_EXTRA_MS"]) / 1e3)
        period = period_duration(hops, plan.points,
                                 int(self.rx["PERIOD_CYCLES"]),
                                 plan.clusters * plan.points)
        self.assertGreater(float(self.rx["SECONDS_LISTEN"]), 2 * period)

    def test_the_transmitter_outlasts_the_whole_receiver_run(self):
        # The trap here is comparing against SECONDS_LISTEN. A capture costs
        # its own length plus a retune plus a DECODE, and on a Pi the decode is
        # twice the capture: 3 s of listening is about 10 s of wall clock. Held
        # against 3 s the check passes with a 7x margin it does not have, and a
        # transmitter that stops halfway through loses every capture after it.
        hops_per_cycle = (int(self.tx["CLUSTERS"])
                          * int(self.tx["CLUSTER_POINTS"]))
        blocks_per_cycle = (int(self.tx["CLUSTERS"])
                            * int(self.tx["CLUSTER_POINTS"])
                            // int(self.tx["BLOCK"]))
        cycle_s = (hops_per_cycle * float(self.tx["HOP_MS"])
                   + blocks_per_cycle * float(self.tx["BAND_EXTRA_MS"])) / 1e3
        run_s = int(self.tx["CYCLES"]) * cycle_s
        per_capture = (float(self.rx["SECONDS_LISTEN"])
                       + lr.PER_CAPTURE_OVERHEAD_S)
        for sweeps in (int(self.rx["SWEEPS"]), 50):
            need = sweeps * int(self.rx["CLUSTERS"]) * per_capture
            self.assertGreater(
                run_s, 2.0 * need,
                f"the transmitter runs {run_s/60:.0f} min, against "
                f"{need/60:.0f} min for a {sweeps}-sweep receiver run; "
                f"raise CYCLES in adf5355_rf_lever.sh")

    def test_the_two_ends_agree_on_what_a_capture_costs(self):
        # The transmitter's own printout quotes the receiver's per-capture
        # cost, so the two must come from one number rather than two.
        text = read_script(os.path.join(context.ROOT, "adf5355", "cli.py"))
        self.assertIn("about 10 s per capture", text)
        self.assertAlmostEqual(lr.PER_CAPTURE_OVERHEAD_S, 7.0, places=6)

    def test_the_sweep_count_is_enough_for_the_resample_to_mean_anything(self):
        self.assertGreaterEqual(int(self.rx["SWEEPS"]), lf.MIN_SWEEPS)

    def test_the_safety_banner_survives_on_the_transmitter(self):
        text = read_script(LEVER_TX)
        for phrase in ("CLOSED, CONDUCTED PATHS ONLY", "NEVER RADIATE",
                       "satellite downlink", "No antenna on either end",
                       "-100 dBm"):
            self.assertIn(phrase, text, f"safety banner lost {phrase!r}")

    def test_the_reminder_prints_before_the_transmitter_starts(self):
        text = read_script(LEVER_TX)
        self.assertLess(text.index("Do not connect an antenna"),
                        text.index('exec "$ADF"'))

    def test_the_receiver_carries_the_warning_too(self):
        text = read_script(LEVER_RX)
        for phrase in ("CLOSED, CONDUCTED PATHS ONLY", "NEVER RADIATE",
                       "No antenna"):
            self.assertIn(phrase, text, f"safety banner lost {phrase!r}")

    def test_the_runner_itself_carries_the_warning(self):
        text = read_script(os.path.join(context.ROOT, "tools", "lever_run.py"))
        for phrase in ("CLOSED, CONDUCTED PATHS ONLY", "NEVER RADIATE",
                       "No antenna"):
            self.assertIn(phrase, text, f"safety banner lost {phrase!r}")


class RunnerRobustnessTests(unittest.TestCase):
    """A twenty-minute run must not end because one capture would not decode."""

    def test_one_bad_capture_is_disowned_and_the_run_carries_on(self):
        plan = small_plan(2)
        visits = lf.plan_visits(1234567, 2, 3, dither_hz=120e3)
        synthetic = {"d_rx": D_RX, "d_lnb": D_LNB, "drift_hz_s": 0.0,
                     "snr_db": 12.0, "overhead_s": 6.0, "t_ref_s": 0.0,
                     "bias": np.zeros(len(visits))}
        # lever_run loads its own copy of the decoder, so patch that one.
        real = lr.hd.decode
        calls = {"n": 0}

        def flaky(*a, **kw):
            calls["n"] += 1
            if calls["n"] == 2:
                raise ValueError("planted failure")
            return real(*a, **kw)

        lr.hd.decode = flaky
        try:
            records = lr.run(plan, visits, lo_hz=LO, fs=FS, seconds=SECONDS,
                             seed=DEFAULT_SEED, min_hop_s=DWELL, block=BLOCK,
                             jitter=0.0, period_cycles=1,
                             band_extra_s=BAND_EXTRA, frame=FRAME,
                             d_rx_guess=D_RX, lo_error_hz=D_LNB * LO,
                             synthetic=synthetic, decimate=DECIMATE,
                             progress=lambda *a, **k: None)
        finally:
            lr.hd.decode = real
        self.assertEqual(len(records), len(visits))
        bad = [r for r in records if not r.trustworthy]
        self.assertEqual(len(bad), 1)
        self.assertIn("decode failed", bad[0].warnings[0])
        self.assertTrue(all(r.trustworthy for r in records
                            if r is not bad[0]))
