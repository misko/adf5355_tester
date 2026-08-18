"""The package must reproduce adf5355_ladder.py's register images exactly.

The ladder script is the artifact that has actually been used on the bench, so
it is treated as a third independent implementation.  Agreement between it, the
ADI C reference and the package is what makes the register images trustworthy.
"""
import importlib.util
import os
import sys
import unittest

from . import context  # noqa: F401
from adf5355 import Channel, MuxOut, OutputPower, SynthConfig, plan
from adf5355.registers import N_REGS

LADDER_PATH = os.path.join(context.ROOT, "adf5355_ladder.py")


def load_ladder():
    spec = importlib.util.spec_from_file_location("adf5355_ladder", LADDER_PATH)
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations via sys.modules, so register first.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(os.path.exists(LADDER_PATH), "ladder script not present")
class TestLadderParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ladder = load_ladder()

    def _package_config(self, ref_hz, cp_ua=900):
        """SynthConfig matching the ladder script's hard-coded choices."""
        return SynthConfig(
            ref_hz=ref_hz, cp_ua=cp_ua,
            muxout=MuxOut.THREESTATE, muxout_3v3=False,
            outa_enable=False, outa_power=OutputPower.MINUS_4_DBM,
            outb_enable=False, mute_till_lock=False,
            negative_bleed=False, gated_bleed=False,
        )

    def _compare(self, ref_hz, freq_hz, cp_ua=900):
        synth = self.ladder.ADF5355(ref_hz=ref_hz, cp_ua=cp_ua, dry_run=True)
        synth._build_frequency_registers(freq_hz, b_enabled=False)
        p = plan(self._package_config(ref_hz, cp_ua), freq_hz, Channel.B)

        self.assertEqual(synth.fpfd_hz, p.config.fpfd_hz)
        self.assertEqual(synth.r_counter, p.config.r_divider)
        self.assertEqual(synth.delay_us, p.delay_us,
                         f"autocal delay ref={ref_hz}")
        for reg in range(N_REGS):
            self.assertEqual(
                (synth.regs[reg] | reg) & 0xFFFFFFFF, p.registers.word(reg),
                f"R{reg} ref={ref_hz} freq={freq_hz}")

    def test_ku_ladder_frequencies(self):
        steps = self.ladder.make_ladder()
        for ref in (10_000_000, 25_000_000, 50_000_000, 100_000_000):
            for step in steps:
                with self.subTest(ref=ref, freq=step.freq_hz):
                    self._compare(ref, step.freq_hz)

    def test_adc_divider_agrees_at_122_88_mhz(self):
        # f_PFD = 61.44 MHz is where the ADI C reference rounds the ADC divider
        # down and breaks the 100 kHz limit.  The ladder script computes this
        # step in floating point and lands on the correct value, so it is an
        # independent confirmation of the package's integer form.
        self._compare(122_880_000, 11_700_000_000)

    def test_charge_pump_variants(self):
        for cp_ua in (315, 900, 1575, 5040):
            with self.subTest(cp_ua=cp_ua):
                self._compare(10_000_000, 11_700_000_000, cp_ua=cp_ua)

    def test_ladder_timing_budget_is_consistent(self):
        steps = self.ladder.make_ladder(steps=9, total_s=18.0)
        self.assertEqual(len(steps), 9)
        self.assertAlmostEqual(steps[-1].end_s, 18.0, places=6)
        for step in steps:
            self.assertAlmostEqual(step.on_s, step.off_s, places=9)
        self.assertEqual(steps[0].freq_hz, 10_700_000_000)
        self.assertEqual(steps[-1].freq_hz, 12_700_000_000)


if __name__ == "__main__":
    unittest.main()
