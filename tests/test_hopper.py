"""The seed is a protocol between transmitter and receiver.

Both ends regenerate the schedule independently, so the generator is a
compatibility contract: if it ever changes, every receiver silently decodes
against the wrong schedule. These tests pin it.
"""
import unittest

from . import context  # noqa: F401
from adf5355.hopper import (SplitMix64, cycle_duration, make_schedule,
                            plan_frequencies)


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

    def test_each_cycle_visits_every_point_exactly_once(self):
        hops = make_schedule(42, self.freqs, 0.005, 8)
        for cycle in range(8):
            points = sorted(h.point for h in hops if h.cycle == cycle)
            self.assertEqual(points, list(range(len(self.freqs))))

    def test_dwell_is_fixed_by_default(self):
        hops = make_schedule(11, self.freqs, 0.005, 10)
        self.assertEqual({round(h.dwell_s, 12) for h in hops}, {0.005})

    def test_dwell_law_with_jitter(self):
        """dwell = min*(1 + jitter*rand*2), so [min, 3*min] at jitter 1."""
        minimum = 0.005
        hops = make_schedule(11, self.freqs, minimum, 30, jitter=1.0)
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


if __name__ == "__main__":
    unittest.main()
