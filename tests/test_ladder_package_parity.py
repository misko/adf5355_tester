"""adf5355.ladder must reproduce the standalone script's schedule exactly.

The script is the artifact that has actually been run on the bench and matches
the published guide, so it is the reference for the pattern.
"""
import importlib.util
import os
import sys
import unittest

from . import context  # noqa: F401
from adf5355 import ladder as pkg

LADDER_PATH = os.path.join(context.ROOT, "adf5355_ladder.py")

CASES = [
    {},
    {"steps": 5},
    {"steps": 12},
    {"total_s": 30.0},
    {"start_hz": 7_000_000_000, "stop_hz": 13_000_000_000},
    {"start_hz": 8_000_000_000, "stop_hz": 9_000_000_000, "steps": 4,
     "total_s": 6.0},
]


def load_script():
    spec = importlib.util.spec_from_file_location("adf5355_ladder", LADDER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(os.path.exists(LADDER_PATH), "ladder script not present")
class TestLadderParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = load_script()

    def test_schedules_match_field_for_field(self):
        for case in CASES:
            with self.subTest(**case):
                mine = pkg.make_ladder(**case)
                theirs = self.script.make_ladder(**case)
                self.assertEqual(len(mine), len(theirs))
                for a, b in zip(mine, theirs):
                    self.assertEqual(a.index, b.index)
                    self.assertEqual(a.freq_hz, b.freq_hz)
                    for field in ("on_s", "off_s", "start_s", "on_end_s", "end_s"):
                        self.assertAlmostEqual(
                            getattr(a, field), getattr(b, field), places=9,
                            msg=f"{field} differs at rung {a.index} for {case}")

    def test_guide_defaults(self):
        """The published example: a 1.4 s burst is rung 7, so 12.200 GHz."""
        steps = pkg.make_ladder()
        self.assertEqual(len(steps), 9)
        self.assertEqual(steps[0].freq_hz, 10_700_000_000)
        self.assertEqual(steps[-1].freq_hz, 12_700_000_000)
        self.assertAlmostEqual(steps[-1].end_s, 18.0, places=9)
        rung7 = steps[6]
        self.assertEqual(rung7.index, 7)
        self.assertEqual(rung7.freq_hz, 12_200_000_000)
        self.assertAlmostEqual(rung7.on_s, 1.4, places=9)

    def test_every_rung_has_a_distinct_burst_length(self):
        """Duration coding only works if the lengths are unambiguous."""
        for case in CASES:
            with self.subTest(**case):
                lengths = [s.on_s for s in pkg.make_ladder(**case)]
                self.assertEqual(len(set(lengths)), len(lengths))

    def test_windows_sum_to_the_requested_total(self):
        for case in CASES:
            with self.subTest(**case):
                steps = pkg.make_ladder(**case)
                total = case.get("total_s", pkg.DEFAULT_TOTAL_S)
                self.assertAlmostEqual(
                    sum(s.on_s + s.off_s for s in steps), total, places=9)
                self.assertAlmostEqual(steps[-1].end_s, total, places=9)

    def test_rejects_degenerate_input(self):
        for bad in ({"steps": 1}, {"total_s": 0}, {"total_s": -1}):
            with self.subTest(**bad):
                with self.assertRaises(ValueError):
                    pkg.make_ladder(**bad)


if __name__ == "__main__":
    unittest.main()
