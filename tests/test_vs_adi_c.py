"""Differential test: Python plan vs. Analog Devices' own C arithmetic.

tests/adi_reference.c is a verbatim transcription of the no-OS driver's
computation.  It is compiled on demand and every field is compared.  Where the
two disagree the reference is wrong; those cases are asserted explicitly rather
than tolerated.
"""
import os
import subprocess
import unittest

from . import context  # noqa: F401
from adf5355 import Channel, OutputPower, SynthConfig, plan
from adf5355.plan import MOD2_FIELD_MAX

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "adi_reference.c")
BINARY = os.path.join(HERE, "adi_reference")

REFS = [10_000_000, 12_800_000, 19_200_000, 25_000_000, 26_000_000,
        50_000_000, 100_000_000, 122_880_000]
FREQS_A = [100_000_000, 437_000_000, 1_000_000_000, 1_575_420_000,
           2_400_000_000, 3_400_000_000, 4_200_000_001, 5_800_000_000,
           6_800_000_000]
FREQS_B = [6_800_000_000, 8_400_000_000, 10_700_000_000, 11_700_000_000,
           12_700_000_000, 13_600_000_000]


def build_reference():
    if os.path.exists(BINARY) and \
            os.path.getmtime(BINARY) > os.path.getmtime(SOURCE):
        return True
    try:
        subprocess.run(["cc", "-O2", "-o", BINARY, SOURCE],
                       check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def run_reference(ref_hz, freq_hz, chan, cp_ua=900, doubler=0, div2=0):
    out = subprocess.run(
        [BINARY, str(ref_hz), str(freq_hz), str(chan), str(cp_ua),
         str(doubler), str(div2)],
        check=True, capture_output=True, text=True).stdout
    result = {}
    for line in out.strip().splitlines():
        key, _, value = line.partition("=")
        result[key] = int(value)
    return result


@unittest.skipUnless(build_reference(), "no C compiler available")
class TestAgainstADIReference(unittest.TestCase):

    def _config(self, ref_hz, chan, cp_ua=900, doubler=0, div2=0):
        return SynthConfig(
            ref_hz=ref_hz, cp_ua=cp_ua,
            ref_doubler=bool(doubler), ref_div2=bool(div2),
            outa_enable=(chan == 0), outb_enable=(chan == 1),
            outa_power=OutputPower.PLUS_5_DBM,
            mute_till_lock=True, negative_bleed=True,
        )

    def _compare(self, ref_hz, freq_hz, chan, cp_ua=900, doubler=0, div2=0):
        expected = run_reference(ref_hz, freq_hz, chan, cp_ua, doubler, div2)
        cfg = self._config(ref_hz, chan, cp_ua, doubler, div2)
        p = plan(cfg, freq_hz, Channel.A if chan == 0 else Channel.B)
        s, regs = p.solution, p.registers
        where = f"ref={ref_hz} freq={freq_hz} chan={chan}"

        self.assertEqual(cfg.fpfd_hz, expected["fpfd"], where)
        self.assertEqual(cfg.r_divider, expected["r_counter"], where)
        self.assertEqual(s.rf_divider_select, expected["rf_div_sel"], where)
        self.assertEqual(s.integer, expected["integer"], where)
        self.assertEqual(s.frac1, expected["frac1"], where)
        self.assertEqual(int(s.prescaler_89), expected["prescaler"], where)
        self.assertEqual(s.cp_bleed, expected["cp_bleed"], where)
        self.assertEqual(regs.get("negative_bleed"), expected["neg_bleed"], where)

        # The reference's ADC divider truncates before rounding up and can land
        # one step low, breaking the 100 kHz ceiling.  Compare only where the
        # reference is actually in spec; otherwise assert we corrected upward.
        adi_adc_clk = cfg.fpfd_hz / (4 * expected["adc_div"] + 2)
        ours = regs.get("adc_clk_divider")
        if adi_adc_clk <= 100_000:
            self.assertEqual(ours, expected["adc_div"], where)
            self.assertEqual(p.delay_us, expected["delay_us"], where)
        else:
            self.assertGreater(ours, expected["adc_div"],
                               f"{where}: should raise the divider above "
                               f"{expected['adc_div']}")
            self.assertLessEqual(cfg.fpfd_hz / (4 * ours + 2), 100_000, where)

        if expected["mod2"] > MOD2_FIELD_MAX:
            # Documented divergence: the reference would encode MOD2 as zero.
            self.assertLessEqual(s.mod2, MOD2_FIELD_MAX, where)
        else:
            self.assertEqual(s.mod2, expected["mod2"], where)
            self.assertEqual(s.frac2, expected["frac2"], where)
            compared = [0, 1, 2, 4, 6, 9]
            if adi_adc_clk <= 100_000:
                compared.append(10)   # R10 carries the ADC divider
            for reg in compared:
                self.assertEqual(regs.word(reg), expected[f"r{reg}"],
                                 f"R{reg} {where}: "
                                 f"0x{regs.word(reg):08X} != "
                                 f"0x{expected[f'r{reg}']:08X}")

    def test_channel_a_grid(self):
        for ref in REFS:
            for freq in FREQS_A:
                with self.subTest(ref=ref, freq=freq):
                    self._compare(ref, freq, 0)

    def test_channel_b_grid(self):
        for ref in REFS:
            for freq in FREQS_B:
                with self.subTest(ref=ref, freq=freq):
                    self._compare(ref, freq, 1)

    def test_charge_pump_settings(self):
        for cp_ua in (315, 630, 900, 1575, 3150, 5040):
            with self.subTest(cp_ua=cp_ua):
                self._compare(10_000_000, 2_400_000_000, 0, cp_ua=cp_ua)

    def test_reference_doubler_and_divider(self):
        for doubler, div2 in ((1, 0), (0, 1), (1, 1)):
            with self.subTest(doubler=doubler, div2=div2):
                self._compare(10_000_000, 2_400_000_000, 0,
                              doubler=doubler, div2=div2)

    def test_pseudorandom_sweep(self):
        import random
        rng = random.Random(20260819)
        for _ in range(400):
            ref = rng.choice(REFS)
            chan = rng.choice([0, 1])
            freq = (rng.randrange(60_000_000, 6_800_000_000) if chan == 0
                    else rng.randrange(6_800_000_000, 13_600_000_000))
            with self.subTest(ref=ref, freq=freq, chan=chan):
                self._compare(ref, freq, chan)


if __name__ == "__main__":
    unittest.main()
