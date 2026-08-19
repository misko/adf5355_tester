"""The seed is a protocol between transmitter and receiver.

Both ends regenerate the schedule independently, so the generator is a
compatibility contract: if it ever changes, every receiver silently decodes
against the wrong schedule. These tests pin it.
"""
import time
import unittest

from . import context  # noqa: F401
from adf5355.hopper import (SplitMix64, cycle_duration, make_schedule,
                            plan_frequencies, run_hops)

# The visiting order this generator emits, recorded from the transmitter of
# record for the recommended 20-point plan. These are golden vectors, not
# descriptions: the identical tuples are pinned on the receive side in
# pluto-plus-utils (tests/test_seeded_hop.py, REFERENCE_ORDERS), and the two
# copies are the only thing that keeps two independent implementations of the
# same protocol honest. Every structural property below -- that a cycle is a
# permutation, that the pattern repeats, that the order is not sequential --
# survives a change that silently renumbers the whole schedule. These do not.
REFERENCE_ORDERS = {
    0: (10, 14, 4, 6, 18, 5, 12, 3, 9, 13, 7, 19, 8, 17, 0, 11, 2, 1, 16, 15),
    1: (1, 14, 10, 3, 19, 4, 6, 16, 15, 13, 2, 0, 11, 7, 18, 9, 17, 12, 8, 5),
    0xC0FFEE: (6, 18, 5, 1, 10, 11, 12, 4, 16, 7, 9, 13, 19, 17, 2, 8, 15, 3,
               0, 14),
    0xDEADBEEF: (18, 2, 9, 6, 4, 0, 19, 8, 1, 3, 10, 13, 17, 14, 5, 16, 12, 11,
                 15, 7),
}

# Eight points 20 kHz apart, jitter 0.5, two permutations per period: the same
# case pinned on the receive side, and the only one that exercises the dwell
# draw as well as the permutation draw.
REFERENCE_JITTER_ORDER = (6, 1, 7, 4, 0, 3, 5, 2, 6, 1, 0, 2, 3, 4, 5, 7)
REFERENCE_JITTER_DWELLS = (
    0.009368005567387911,
    0.005513646982068361,
    0.0081851616627713,
    0.006930070176986194,
    0.00657107527369603,
    0.007297794699275254,
)


class TestSplitMix64(unittest.TestCase):
    """Published SplitMix64 vectors -- the cross-language contract."""

    def test_known_vectors_for_seed_zero(self):
        rng = SplitMix64(0)
        self.assertEqual(rng.next_u64(), 0xE220A8397B1DCDAF)
        self.assertEqual(rng.next_u64(), 0x6E789E6AA1B965F4)
        self.assertEqual(rng.next_u64(), 0x06C45D188009454F)

    def test_output_stays_in_64_bits(self):
        rng = SplitMix64(0xDEADBEEFCAFEBABE)
        for _ in range(500):
            self.assertTrue(0 <= rng.next_u64() < (1 << 64))

    def test_uniform_is_in_range_and_spread(self):
        rng = SplitMix64(1)
        values = [rng.uniform() for _ in range(4000)]
        self.assertTrue(all(0.0 <= v < 1.0 for v in values))
        self.assertAlmostEqual(sum(values) / len(values), 0.5, delta=0.02)

    def test_below_is_in_range_and_unbiased(self):
        rng = SplitMix64(7)
        counts = [0] * 5
        for _ in range(5000):
            value = rng.below(5)
            self.assertTrue(0 <= value < 5)
            counts[value] += 1
        self.assertTrue(all(850 < c < 1150 for c in counts), counts)

    def test_below_rejects_nonpositive(self):
        with self.assertRaises(ValueError):
            SplitMix64(0).below(0)


class TestTheGeneratorIsPinned(unittest.TestCase):
    """Golden vectors, because every other test here passes on a broken one.

    The generator is a wire protocol. Swapping Fisher-Yates for its forward
    form, or truncating the frequency plan instead of rounding it, still yields
    a deterministic, repeating, non-sequential schedule in which every cycle
    visits every point exactly once -- and puts a completely different sequence
    on the air. Only a recorded answer catches that.
    """

    def test_the_visiting_order_is_pinned_for_several_seeds(self):
        freqs = plan_frequencies(11_000_000_000, 11_001_710_000, 20)
        for seed, order in REFERENCE_ORDERS.items():
            with self.subTest(seed=hex(seed)):
                hops = make_schedule(seed, freqs, 0.010, 1)
                self.assertEqual(tuple(h.point for h in hops), order)

    def test_the_frequency_plan_is_pinned(self):
        """Rounding, not truncation: the receiver rounds the same way."""
        freqs = plan_frequencies(11_000_000_000, 11_001_710_000, 20)
        self.assertEqual(freqs[:3], [11_000_000_000, 11_000_090_000,
                                     11_000_180_000])
        self.assertEqual(freqs[-1], 11_001_710_000)
        # 90 kHz does not divide evenly here, so truncation and rounding differ.
        self.assertEqual(plan_frequencies(11_000_000_000, 11_000_000_007, 3),
                         [11_000_000_000, 11_000_000_004, 11_000_000_007])

    def test_a_jittered_two_cycle_period_is_pinned(self):
        freqs = plan_frequencies(11_000_000_000, 11_000_140_000, 8)
        hops = make_schedule(0xC0FFEE, freqs, 0.005, 2, jitter=0.5,
                             period_cycles=2)
        self.assertEqual(tuple(h.point for h in hops), REFERENCE_JITTER_ORDER)
        self.assertEqual(tuple(h.dwell_s for h in hops[:6]),
                         REFERENCE_JITTER_DWELLS)

    def test_uniform_is_pinned_bit_for_bit(self):
        """The dwell draw. A different shift is still uniform, and still wrong."""
        rng = SplitMix64(0)
        self.assertEqual([rng.uniform() for _ in range(3)],
                         [0.8833108082136426, 0.43152799704850997,
                          0.026433771592597743])

    def test_below_is_pinned(self):
        """The permutation draw, independent of how it is consumed."""
        rng = SplitMix64(0xC0FFEE)
        self.assertEqual([rng.below(20) for _ in range(6)],
                         [14, 17, 11, 0, 4, 7])


class TestSchedule(unittest.TestCase):
    def setUp(self):
        self.freqs = plan_frequencies(11_000_000_000, 11_001_710_000, 20)

    def test_the_same_seed_gives_the_same_schedule(self):
        a = make_schedule(0xC0FFEE, self.freqs, 0.005, 5)
        b = make_schedule(0xC0FFEE, self.freqs, 0.005, 5)
        self.assertEqual([(h.point, h.dwell_s) for h in a],
                         [(h.point, h.dwell_s) for h in b])

    def test_a_different_seed_gives_a_different_schedule(self):
        a = make_schedule(0xC0FFEE, self.freqs, 0.005, 5)
        b = make_schedule(0xC0FFEF, self.freqs, 0.005, 5)
        self.assertNotEqual([h.point for h in a], [h.point for h in b])

    def test_the_pattern_repeats_every_period(self):
        """Periodicity bounds the receiver's epoch search to one period."""
        hops = make_schedule(0xC0FFEE, self.freqs, 0.005, 6)
        n = len(self.freqs)
        first = [h.point for h in hops[:n]]
        for cycle in range(1, 6):
            self.assertEqual([h.point for h in hops[cycle*n:(cycle+1)*n]], first)

    def test_a_longer_period_does_not_repeat_early(self):
        hops = make_schedule(0xC0FFEE, self.freqs, 0.005, 6, period_cycles=3)
        n = len(self.freqs)
        self.assertNotEqual([h.point for h in hops[n:2*n]],
                            [h.point for h in hops[:n]])
        self.assertEqual([h.point for h in hops[3*n:4*n]],
                         [h.point for h in hops[:n]])

    def test_each_cycle_visits_every_point_exactly_once(self):
        hops = make_schedule(42, self.freqs, 0.005, 8)
        for cycle in range(8):
            points = sorted(h.point for h in hops if h.cycle == cycle)
            self.assertEqual(points, list(range(len(self.freqs))))

    def test_dwell_is_fixed_by_default(self):
        hops = make_schedule(11, self.freqs, 0.005, 10)
        self.assertEqual({round(h.dwell_s, 12) for h in hops}, {0.005})

    def test_dwell_law_with_jitter(self):
        """dwell = min*(1 + jitter*rand*2), so [min, 3*min] at jitter 1.

        period_cycles is raised so the draws are distinct: the schedule repeats
        every period, so a single period only ever holds `points` dwells however
        many cycles are transmitted.
        """
        minimum = 0.005
        hops = make_schedule(11, self.freqs, minimum, 30, jitter=1.0,
                             period_cycles=30)
        dwells = [h.dwell_s for h in hops]
        self.assertGreaterEqual(min(dwells), minimum)
        self.assertLess(max(dwells), 3 * minimum)
        self.assertAlmostEqual(sum(dwells) / len(dwells), 2 * minimum,
                               delta=0.1 * minimum)

    def test_timeline_is_contiguous_with_no_gaps(self):
        hops = make_schedule(3, self.freqs, 0.005, 4)
        self.assertAlmostEqual(hops[0].start_s, 0.0, places=12)
        for previous, following in zip(hops, hops[1:]):
            self.assertAlmostEqual(previous.end_s, following.start_s, places=12)
            self.assertAlmostEqual(previous.end_s - previous.start_s,
                                   previous.dwell_s, places=12)

    def test_order_is_not_sequential(self):
        hops = make_schedule(0xC0FFEE, self.freqs, 0.005, 1)
        self.assertNotEqual([h.point for h in hops],
                            list(range(len(self.freqs))))

    def test_a_cycle_is_far_shorter_than_duration_coding(self):
        """20 points: hopping ~0.2 s against the ladder's 4.2 s."""
        hops = make_schedule(0xC0FFEE, self.freqs, 0.005, 1)
        self.assertLess(cycle_duration(hops, 0), 0.5)

    def test_rejects_bad_parameters(self):
        with self.assertRaises(ValueError):
            make_schedule(1, self.freqs, 0.0, 4)
        with self.assertRaises(ValueError):
            make_schedule(1, self.freqs, 0.005, 0)
        with self.assertRaises(ValueError):
            plan_frequencies(11_000_000_000, 11_001_000_000, 1)
        with self.assertRaises(ValueError):
            plan_frequencies(11_001_000_000, 11_000_000_000, 5)
        with self.assertRaises(ValueError):
            make_schedule(1, self.freqs, 0.005, 4, jitter=1.5)
        with self.assertRaises(ValueError):
            make_schedule(1, self.freqs, 0.005, 4, period_cycles=0)


class FakeDevice:
    """Records what the transmit loop would have programmed, and when."""

    def __init__(self):
        self.calls = []          # (elapsed_s, kind, payload)
        self.origin = None

    def _at(self):
        return 0.0 if self.origin is None else time.monotonic() - self.origin

    def set_frequency(self, freq_hz, channel, autocal=True):
        self.calls.append((self._at(), "freq", freq_hz, autocal))

    # run_hops solves every point once up front and then writes cached words,
    # so the fake models that split rather than a set_frequency per hop.
    def precompute(self, freqs, channel):
        self.precomputed = list(freqs)
        return [("words", int(f)) for f in freqs]

    def apply_precomputed(self, triple):
        kind, freq_hz = triple
        assert kind == "words", "apply_precomputed got something else"
        # A cached write never runs the band search; record it as autocal=False
        # so the existing assertions keep their meaning.
        self.calls.append((self._at(), "freq", freq_hz, False))

    def set_output(self, channel, on):
        if on and self.origin is None:
            self.origin = time.monotonic()
        self.calls.append((self._at(), "output", on, None))

    @property
    def frequencies(self):
        return [c[2] for c in self.calls if c[1] == "freq"]


class TestRunHops(unittest.TestCase):
    """The transmit loop is the one place the schedule becomes radio.

    Everything else here checks the schedule as data. If ``run_hops`` programs
    the wrong member of it, or programs the right one at the wrong moment, the
    data is still perfect and nothing on the air matches it -- and the failure
    presents as a receiver that cannot align, which reads like a hardware
    fault. So it is checked directly, against a fake device.
    """

    def setUp(self):
        self.freqs = plan_frequencies(11_000_000_000, 11_000_060_000, 4)
        self.hops = make_schedule(0xC0FFEE, self.freqs, 0.005, 1)
        self.dev = FakeDevice()
        run_hops(self.dev, self.hops, channel=None, settle_s=0.0)

    def test_every_hop_is_programmed_once_in_schedule_order(self):
        self.assertEqual(self.dev.frequencies,
                         [h.freq_hz for h in self.hops])

    def test_only_the_first_hop_runs_the_band_search(self):
        """Autocal on every hop would blank most of each dwell."""
        autocals = [c[3] for c in self.dev.calls if c[1] == "freq"]
        self.assertEqual(autocals, [True] + [False] * (len(self.hops) - 1))

    def test_the_output_is_keyed_once_and_muted_at_the_end(self):
        outputs = [(c[0], c[2]) for c in self.dev.calls if c[1] == "output"]
        self.assertEqual([on for _, on in outputs], [True, False])
        self.assertEqual(self.dev.calls[-1][1], "output")

    def test_each_retune_waits_for_the_dwell_it_follows_to_end(self):
        """Retuning at a hop's START would put every point one dwell early."""
        retunes = [c[0] for c in self.dev.calls if c[1] == "freq"][1:]
        for hop, at in zip(self.hops, retunes):
            self.assertGreaterEqual(at, hop.end_s - 1e-3,
                                    f"hop {hop.sequence} retuned early")
            self.assertLess(at, hop.end_s + 0.05,
                            f"hop {hop.sequence} retuned very late")

    def test_the_loop_lasts_as_long_as_the_schedule_says(self):
        self.assertGreaterEqual(self.dev.calls[-1][0],
                                self.hops[-1].end_s - 1e-3)


if __name__ == "__main__":
    unittest.main()


class TestGcIsHeldOffDuringTheLoop(unittest.TestCase):
    """A collection pause is the only thing that misses a hop deadline.

    Measured at 1600 hops/s: median jitter under a microsecond, but a pause of
    22.8 ms -- 36 dwells -- putting 43 of 4800 hops late by over half a dwell.
    Because pauses are stochastic the resulting failures move between runs,
    which is what made marginal hop rates look flaky rather than broken.
    """

    def test_collection_is_disabled_while_hopping_and_restored_after(self):
        import gc
        from adf5355.hopper import run_hops

        seen = []

        class Watcher(FakeDevice):
            def apply_precomputed(self, triple):
                seen.append(gc.isenabled())
                super().apply_precomputed(triple)

        freqs = plan_frequencies(11_000_000_000, 11_000_060_000, 4)
        hops = make_schedule(0xC0FFEE, freqs, 0.001, 1)
        self.assertTrue(gc.isenabled(), "precondition: gc on")
        run_hops(Watcher(), hops, channel=None, settle_s=0.0)
        self.assertTrue(seen, "no hops were applied")
        self.assertFalse(any(seen), "gc ran during the timed loop")
        self.assertTrue(gc.isenabled(), "gc was not restored")

    def test_gc_state_is_restored_even_if_the_loop_raises(self):
        import gc
        from adf5355.hopper import run_hops

        class Exploding(FakeDevice):
            def apply_precomputed(self, triple):
                raise RuntimeError("boom")

        freqs = plan_frequencies(11_000_000_000, 11_000_060_000, 4)
        hops = make_schedule(0xC0FFEE, freqs, 0.001, 1)
        with self.assertRaises(RuntimeError):
            run_hops(Exploding(), hops, channel=None, settle_s=0.0)
        self.assertTrue(gc.isenabled(), "gc left disabled after a failure")


class TestTraceStaysBounded(unittest.TestCase):
    """The write trace must not grow without bound during a long transmission.

    Three writes per hop at 1600 hops/s is about 17 million records an hour --
    roughly 2.3 GB -- which would exhaust the machine mid-run. The trace exists
    to show what just happened, so it keeps a window.
    """

    def test_a_long_run_does_not_grow_the_trace_without_limit(self):
        from adf5355 import Channel, SynthConfig
        from adf5355.device import TRACE_LIMIT, ADF5355

        cfg = SynthConfig(ref_hz=125_000_000, outb_enable=True)
        dev = ADF5355(cfg, dry_run=True)
        dev.set_frequency(11_000_000_000, Channel.B)
        cached = dev.precompute([11_000_000_000, 11_000_090_000], Channel.B)
        for i in range(TRACE_LIMIT * 3):
            dev.apply_precomputed(cached[i % 2])
        self.assertLessEqual(len(dev.trace), TRACE_LIMIT)

    def test_the_most_recent_writes_are_the_ones_kept(self):
        from adf5355 import Channel, SynthConfig
        from adf5355.device import TRACE_LIMIT, ADF5355

        cfg = SynthConfig(ref_hz=125_000_000, outb_enable=True)
        dev = ADF5355(cfg, dry_run=True)
        for i in range(TRACE_LIMIT + 50):
            dev._write(6, 0x35004836 | (i & 1))
        self.assertEqual(len(dev.trace), TRACE_LIMIT)
        self.assertEqual(dev.trace[-1].word & 1, (TRACE_LIMIT + 49) & 1)
