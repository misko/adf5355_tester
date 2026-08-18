"""Solver and register-assembly behaviour."""
import random
import unittest
from fractions import Fraction

from . import context  # noqa: F401
from adf5355 import Channel, SynthConfig, plan
from adf5355.plan import (MAX_OUTA_HZ, MAX_OUTB_HZ, MAX_VCO_HZ, MIN_OUTA_HZ,
                          MIN_OUTB_HZ, MIN_VCO_HZ, MOD2_FIELD_MAX, MODULUS1,
                          solve_adi, solve_exact, vco_from_fields)

REF_10M = SynthConfig(ref_hz=10_000_000)


class TestReferencePath(unittest.TestCase):
    def test_r_counter_maximizes_pfd_under_75mhz(self):
        for ref, expect_r in ((10_000_000, 1), (100_000_000, 2),
                              (122_880_000, 2), (500_000_000, 7)):
            cfg = SynthConfig(ref_hz=ref)
            self.assertEqual(cfg.r_divider, expect_r, f"ref={ref}")
            self.assertLessEqual(cfg.fpfd_hz, 75_000_000)

    def test_doubler_and_div2_affect_pfd(self):
        self.assertEqual(SynthConfig(ref_hz=10_000_000, ref_doubler=True).fpfd_hz,
                         20_000_000)
        self.assertEqual(SynthConfig(ref_hz=10_000_000, ref_div2=True).fpfd_hz,
                         5_000_000)

    def test_out_of_range_reference_rejected(self):
        with self.assertRaises(ValueError):
            SynthConfig(ref_hz=700_000_000)

    def test_out_of_range_charge_pump_rejected(self):
        with self.assertRaises(ValueError):
            SynthConfig(ref_hz=10_000_000, cp_ua=100)


class TestSolvers(unittest.TestCase):
    def test_mod2_always_fits_the_14_bit_field(self):
        rng = random.Random(20260819)
        for _ in range(4000):
            fpfd = rng.randrange(1_000_000, 75_000_001)
            vco = rng.randrange(MIN_VCO_HZ, MAX_VCO_HZ)
            for solver in (solve_adi, solve_exact):
                _, f1, f2, m2 = solver(vco, fpfd)
                self.assertTrue(1 <= m2 <= MOD2_FIELD_MAX,
                                f"{solver.__name__} MOD2={m2} fpfd={fpfd}")
                self.assertLess(f2, m2 if m2 > 1 else 2)
                self.assertLess(f1, MODULUS1)

    def test_adi_divergence_mod2_16384_is_repaired(self):
        # f_PFD = 2**26 halves down to exactly 16384, which the ADI loop bound
        # allows and the 14-bit field encodes as zero.
        fpfd = 1 << 26
        vco = MIN_VCO_HZ + 12345
        _, _, _, mod2 = solve_adi(vco, fpfd)
        self.assertLessEqual(mod2, MOD2_FIELD_MAX)
        self.assertGreater(mod2, 0)

    def test_exact_solver_is_never_worse_than_adi(self):
        rng = random.Random(7)
        wins = 0
        for _ in range(500):
            fpfd = rng.choice([10_000_000, 25_000_000, 61_440_000, 50_000_000])
            vco = rng.randrange(MIN_VCO_HZ, MAX_VCO_HZ)
            err = []
            for solver in (solve_adi, solve_exact):
                i, f1, f2, m2 = solver(vco, fpfd)
                err.append(abs(vco_from_fields(i, f1, f2, m2, fpfd) - vco))
            self.assertLessEqual(err[1], err[0])
            if err[1] < err[0]:
                wins += 1
        self.assertGreater(wins, 0, "exact solver should sometimes do better")

    def test_integer_n_is_exact(self):
        # 5.85 GHz from a 10 MHz PFD is an exact integer-N case.
        i, f1, f2, m2 = solve_adi(5_850_000_000, 10_000_000)
        self.assertEqual((i, f1, f2), (585, 0, 0))
        self.assertEqual(vco_from_fields(i, f1, f2, m2, 10_000_000),
                         Fraction(5_850_000_000))


class TestChannelSelection(unittest.TestCase):
    def test_rf_divider_keeps_vco_in_band(self):
        for freq in (60_000_000, 100_000_000, 1_000_000_000,
                     2_400_000_000, 3_400_000_000, 6_800_000_000):
            p = plan(REF_10M, freq, Channel.A)
            vco = p.solution.vco_hz
            self.assertTrue(MIN_VCO_HZ <= vco <= MAX_VCO_HZ,
                            f"{freq} -> VCO {float(vco)}")
            self.assertLessEqual(p.solution.rf_divider_select, 6)

    def test_channel_b_doubles_the_vco(self):
        p = plan(REF_10M, 11_700_000_000, Channel.B)
        self.assertEqual(p.solution.vco_hz * 2, p.solution.achieved_hz)
        self.assertEqual(p.solution.rf_divider_select, 0)

    def test_out_of_band_requests_rejected(self):
        for freq, chan in ((MIN_OUTA_HZ - 1, Channel.A),
                           (MAX_OUTA_HZ + 1, Channel.A),
                           (MIN_OUTB_HZ - 1, Channel.B),
                           (MAX_OUTB_HZ + 1, Channel.B)):
            with self.assertRaises(ValueError, msg=f"{freq} {chan}"):
                plan(REF_10M, freq, chan)

    def test_prescaler_switches_at_n_75(self):
        low = plan(SynthConfig(ref_hz=100_000_000), 3_500_000_000, Channel.A)
        self.assertEqual(low.solution.integer, 70)
        self.assertFalse(low.solution.prescaler_89)
        high = plan(REF_10M, 3_500_000_000, Channel.A)
        self.assertTrue(high.solution.prescaler_89)


class TestAccuracy(unittest.TestCase):
    def test_error_stays_below_one_hertz_across_the_band(self):
        rng = random.Random(11)
        cfg = SynthConfig(ref_hz=10_000_000, solver="exact")
        for _ in range(300):
            freq = rng.randrange(100_000_000, MAX_OUTA_HZ)
            p = plan(cfg, freq, Channel.A)
            self.assertLess(abs(float(p.solution.error_hz)), 1.0,
                            f"{freq} Hz -> {float(p.solution.error_hz)} Hz error")

    def test_reported_frequency_is_exact_rational(self):
        p = plan(REF_10M, 2_400_000_000, Channel.A)
        self.assertIsInstance(p.solution.achieved_hz, Fraction)


class TestAssembly(unittest.TestCase):
    def test_adc_clock_never_exceeds_100khz(self):
        for ref in (10_000_000, 12_800_000, 25_000_000, 50_000_000,
                    100_000_000, 122_880_000, 500_000_000):
            cfg = SynthConfig(ref_hz=ref)
            p = plan(cfg, 2_400_000_000, Channel.A)
            adc_div = p.registers.get("adc_clk_divider")
            self.assertLessEqual(cfg.fpfd_hz / (4 * adc_div + 2), 100_000,
                                 f"ref={ref}")

    def test_adc_divider_is_the_smallest_value_that_meets_the_limit(self):
        # Minimal, because a larger divider slows VCO band select for nothing.
        for ref in (10_000_000, 12_800_000, 19_200_000, 25_000_000,
                    26_000_000, 40_000_000, 50_000_000, 61_440_000,
                    100_000_000, 122_880_000, 153_600_000, 500_000_000):
            cfg = SynthConfig(ref_hz=ref)
            p = plan(cfg, 2_400_000_000, Channel.A)
            div, fpfd = p.registers.get("adc_clk_divider"), cfg.fpfd_hz
            self.assertLessEqual(fpfd / (4 * div + 2), 100_000, f"ref={ref}")
            if div > 1:
                self.assertGreater(fpfd / (4 * (div - 1) + 2), 100_000,
                                   f"ref={ref}: divider {div} is not minimal")

    def test_autocal_set_and_muxout_configured_for_3v3_lock_detect(self):
        p = plan(REF_10M, 2_400_000_000, Channel.A)
        self.assertEqual(p.registers.get("autocal"), 1)
        self.assertEqual(p.registers.get("muxout"), 6)
        self.assertEqual(p.registers.get("mux_logic_3v3"), 1)

    def test_r7_is_the_corrected_adf5355_value(self):
        # Not 0x04000007, which is what the C reference emits for every part.
        p = plan(REF_10M, 2_400_000_000, Channel.A)
        self.assertEqual(p.registers.word(7), 0x12000067)

    def test_outb_enable_bit_is_active_low(self):
        on = plan(SynthConfig(ref_hz=10_000_000, outb_enable=True),
                  11_700_000_000, Channel.B)
        off = plan(SynthConfig(ref_hz=10_000_000, outb_enable=False),
                   11_700_000_000, Channel.B)
        self.assertEqual(on.registers.get("rf_outb_disable"), 0)
        self.assertEqual(off.registers.get("rf_outb_disable"), 1)

    def test_negative_bleed_gated_off_for_integer_n(self):
        p = plan(SynthConfig(ref_hz=10_000_000, negative_bleed=True),
                 5_850_000_000, Channel.A)
        self.assertTrue(p.solution.is_integer_n)
        self.assertEqual(p.registers.get("negative_bleed"), 0)


if __name__ == "__main__":
    unittest.main()
