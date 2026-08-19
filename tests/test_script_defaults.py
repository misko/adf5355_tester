"""The two runner scripts are one protocol, so their defaults must agree.

Nothing is transmitted from the receiver to the transmitter and nothing is
transmitted back: the schedule is the entire agreement between them. If the two
settings blocks ever drift apart, the receiver regenerates a different schedule
from the one on the air, alignment fails, and the failure looks like a hardware
problem rather than an edit. These tests are the guard against that.

They parse the shell scripts as text -- deliberately, rather than sourcing
them -- so that what is checked is what a reader sees in the settings block.
"""
import os
import re
import unittest

from . import context  # noqa: F401
from adf5355 import cli
from adf5355 import hopper

TRANSMIT = os.path.join(context.ROOT, "adf5355_rf_hop.sh")
RECEIVE = os.path.join(context.ROOT, "sdr_listen.sh")

# The settings that define the schedule. Both ends must carry every one of
# them, at the same value, and must pass every one of them on.
SCHEDULE_KEYS = ("SEED", "START_GHZ", "STOP_GHZ", "POINTS", "HOP_MS", "JITTER",
                 "PERIOD_CYCLES")

SETTING = re.compile(r'^([A-Z_][A-Z0-9_]*)="\$\{\1:-([^}]*)\}"', re.MULTILINE)


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def settings(path):
    """Every NAME="${NAME:-default}" in the script, as {name: default}."""
    return dict(SETTING.findall(read(path)))


def exec_line(path):
    """The exec command, joined into one line with continuations removed."""
    text = read(path).replace("\\\n", " ")
    for line in text.splitlines():
        if line.startswith("exec "):
            return " ".join(line.split())
    raise AssertionError(f"{path} has no exec line")


class TestBothScriptsAgree(unittest.TestCase):
    def setUp(self):
        self.tx = settings(TRANSMIT)
        self.rx = settings(RECEIVE)

    def test_both_scripts_define_every_schedule_setting(self):
        for key in SCHEDULE_KEYS:
            self.assertIn(key, self.tx, f"{key} missing from the transmitter")
            self.assertIn(key, self.rx, f"{key} missing from the receiver")

    def test_every_shared_setting_has_the_same_default(self):
        shared = sorted(set(self.tx) & set(self.rx))
        self.assertTrue(set(SCHEDULE_KEYS) <= set(shared), shared)
        for key in shared:
            self.assertEqual(self.tx[key], self.rx[key],
                             f"{key} differs: transmitter {self.tx[key]!r} vs "
                             f"receiver {self.rx[key]!r}")

    def test_the_transmitter_drives_the_hop_command(self):
        line = exec_line(TRANSMIT)
        self.assertIn('"$ADF" hop', line)
        self.assertNotIn(" ladder ", line)
        self.assertIn("--enable-rf", line)

    def test_the_receiver_drives_the_hop_decoder(self):
        self.assertIn('"$DECODER"', exec_line(RECEIVE))
        self.assertIn("tools/hop_decode.py", read(RECEIVE))
        self.assertNotIn("freq_ladder_listen", read(RECEIVE))

    def test_every_schedule_setting_reaches_both_ends(self):
        """Agreeing on a default is worthless if it is never passed on."""
        for key in SCHEDULE_KEYS:
            self.assertIn(f'"${key}"', exec_line(TRANSMIT),
                          f"{key} never reaches adf5355 hop")
            self.assertIn(f'"${key}"', exec_line(RECEIVE),
                          f"{key} never reaches the decoder")

    def test_the_receiver_passes_its_whole_chain_on(self):
        line = exec_line(RECEIVE)
        for key in ("LO_HZ", "LO_ERROR_HZ", "FS", "FRAME", "SECONDS_LISTEN",
                    "GAIN", "URI"):
            self.assertIn(f'"${key}"', line, f"{key} never reaches the decoder")


class TestScriptsAgreeWithThePackage(unittest.TestCase):
    """A shell default that has drifted from the library default is a bug."""

    def setUp(self):
        self.tx = settings(TRANSMIT)
        self.rx = settings(RECEIVE)

    def test_schedule_defaults_match_the_hopper(self):
        self.assertEqual(int(self.tx["SEED"], 0), hopper.DEFAULT_SEED)
        self.assertEqual(int(self.tx["POINTS"]), hopper.DEFAULT_POINTS)
        self.assertEqual(float(self.tx["HOP_MS"]) / 1e3,
                         hopper.DEFAULT_MIN_HOP_S)
        self.assertEqual(float(self.tx["JITTER"]), hopper.DEFAULT_JITTER)
        self.assertEqual(int(self.tx["PERIOD_CYCLES"]),
                         hopper.DEFAULT_PERIOD_CYCLES)
        self.assertEqual(round(float(self.tx["START_GHZ"]) * 1e9),
                         hopper.DEFAULT_HOP_START_HZ)
        self.assertEqual(round(float(self.tx["STOP_GHZ"]) * 1e9),
                         hopper.DEFAULT_HOP_STOP_HZ)

    def test_schedule_defaults_match_the_transmit_cli(self):
        args = cli.build_parser().parse_args(["hop"])
        self.assertEqual(int(self.tx["SEED"], 0), args.seed)
        self.assertEqual(float(self.tx["START_GHZ"]), args.start_ghz)
        self.assertEqual(float(self.tx["STOP_GHZ"]), args.stop_ghz)
        self.assertEqual(int(self.tx["POINTS"]), args.points)
        self.assertEqual(float(self.tx["HOP_MS"]), args.min_hop_ms)
        self.assertEqual(float(self.tx["JITTER"]), args.jitter)
        self.assertEqual(int(self.tx["PERIOD_CYCLES"]), args.period_cycles)
        self.assertEqual(int(self.tx["CYCLES"]), args.cycles)
        self.assertEqual(int(self.tx["POWER"]), args.power)

    def test_receive_defaults_match_the_decoder(self):
        from tests.test_hop_decode import hd
        args = hd.build_parser().parse_args([])
        self.assertEqual(float(self.rx["LO_HZ"]), args.lo_hz)
        self.assertEqual(float(self.rx["LO_ERROR_HZ"]), args.lo_error_hz)
        self.assertEqual(float(self.rx["FS"]), args.fs)
        self.assertEqual(int(self.rx["FRAME"]), args.frame)
        self.assertEqual(float(self.rx["SECONDS_LISTEN"]), args.seconds)
        self.assertEqual(float(self.rx["GAIN"]), args.gain)
        self.assertEqual(self.rx["URI"], args.uri)

    def test_the_span_fits_the_sample_rate(self):
        span = (float(self.rx["STOP_GHZ"]) - float(self.rx["START_GHZ"])) * 1e9
        self.assertLess(span, float(self.rx["FS"]) * 0.8)

    def test_a_dwell_is_many_frames_long(self):
        frames = (float(self.rx["HOP_MS"]) / 1e3
                  / (int(self.rx["FRAME"]) / float(self.rx["FS"])))
        self.assertGreater(frames, 20)

    def test_the_transmitter_outlasts_the_capture(self):
        run_s = (int(self.tx["CYCLES"]) * int(self.tx["POINTS"])
                 * float(self.tx["HOP_MS"]) / 1e3)
        self.assertGreater(run_s, 4 * float(self.rx["SECONDS_LISTEN"]))


class TestTheSafetyBannerSurvives(unittest.TestCase):
    """The transmit script keys real RF into satellite downlink spectrum."""

    def test_the_banner_is_still_there(self):
        text = read(TRANSMIT)
        for phrase in ("CLOSED, CONDUCTED PATHS ONLY", "NEVER RADIATE",
                       "satellite downlink", "No antenna on either end",
                       "-100 dBm"):
            self.assertIn(phrase, text, f"safety banner lost {phrase!r}")

    def test_the_reminder_prints_before_the_transmitter_starts(self):
        text = read(TRANSMIT)
        self.assertLess(text.index("Do not connect an antenna"),
                        text.index('exec "$ADF"'))


if __name__ == "__main__":
    unittest.main()
