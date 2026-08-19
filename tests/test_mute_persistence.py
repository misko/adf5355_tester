"""An explicit mute must survive a retune.

program()/retune() write R6 straight from the plan, whose enable bits describe
the configuration rather than the live state. Before this was fixed, the ladder
called set_output(False) and then set_frequency() for the next rung, and the
retune re-keyed the transmitter -- so the output never went quiet between rungs
and the burst lengths no longer matched the published schedule. A receiver
decoding rung number from burst duration got the wrong answer.
"""
import unittest

from . import context  # noqa: F401
from adf5355 import Channel, SynthConfig, plan
from adf5355.device import ADF5355
from adf5355.registers import FIELDS

OUTA = FIELDS["rf_out_enable"]
OUTB_DIS = FIELDS["rf_outb_disable"]


def r6_writes(dev):
    return [w.word for w in dev.trace if w.reg == 6]


class TestMuteSurvivesRetune(unittest.TestCase):
    def _dev(self, channel):
        cfg = SynthConfig(ref_hz=125_000_000,
                          outa_enable=channel is Channel.A,
                          outb_enable=channel is Channel.B)
        return cfg, ADF5355(cfg, dry_run=True)

    def test_channel_b_retune_while_muted_stays_muted(self):
        cfg, dev = self._dev(Channel.B)
        dev.set_frequency(11_000_000_000, Channel.B)
        dev.set_output(Channel.B, False)
        dev.trace.clear()
        dev.set_frequency(11_300_000_000, Channel.B)
        self.assertTrue(r6_writes(dev), "retune should write R6")
        for word in r6_writes(dev):
            self.assertEqual(OUTB_DIS.decode(word), 1, "RFoutB re-keyed by retune")
            self.assertEqual(OUTA.decode(word), 0)

    def test_channel_a_retune_while_muted_stays_muted(self):
        cfg, dev = self._dev(Channel.A)
        dev.set_frequency(2_400_000_000, Channel.A)
        dev.set_output(Channel.A, False)
        dev.trace.clear()
        dev.set_frequency(3_500_000_000, Channel.A)
        for word in r6_writes(dev):
            self.assertEqual(OUTA.decode(word), 0, "RFoutA re-keyed by retune")

    def test_reenabling_clears_the_mute(self):
        cfg, dev = self._dev(Channel.B)
        dev.set_frequency(11_000_000_000, Channel.B)
        dev.set_output(Channel.B, False)
        dev.set_output(Channel.B, True)
        dev.trace.clear()
        dev.set_frequency(11_300_000_000, Channel.B)
        self.assertEqual(OUTB_DIS.decode(r6_writes(dev)[0]), 0,
                         "output should come back after an explicit enable")

    def test_cold_program_while_muted_stays_muted(self):
        cfg, dev = self._dev(Channel.B)
        dev.set_frequency(11_000_000_000, Channel.B)
        dev.mute()
        dev.trace.clear()
        dev._synced = False
        dev.program(plan(cfg, 11_300_000_000, Channel.B))
        for word in r6_writes(dev):
            self.assertEqual(OUTB_DIS.decode(word), 1)

    def test_ladder_never_keys_between_rungs(self):
        """Walk a real ladder and assert every OFF window is actually off."""
        from adf5355.ladder import make_ladder
        cfg = SynthConfig(ref_hz=125_000_000, outb_enable=True)
        dev = ADF5355(cfg, dry_run=True)
        steps = make_ladder(10_700_000_000, 11_600_000_000, 4, 8.0)

        dev.set_frequency(steps[0].freq_hz, Channel.B)
        dev.set_output(Channel.B, False)
        for i, step in enumerate(steps):
            dev.set_output(Channel.B, True)
            dev.trace.clear()
            dev.set_output(Channel.B, False)
            if i + 1 < len(steps):
                dev.set_frequency(steps[i + 1].freq_hz, Channel.B)
            for word in r6_writes(dev):
                self.assertEqual(
                    OUTB_DIS.decode(word), 1,
                    f"output keyed during the OFF window after rung {step.index}")


if __name__ == "__main__":
    unittest.main()
