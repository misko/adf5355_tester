"""The reference default must stay pinned to this board's X1 marking."""
import importlib.util
import os
import sys
import unittest

from . import context  # noqa: F401
from adf5355 import cli

LADDER_PATH = os.path.join(context.ROOT, "adf5355_ladder.py")


def load_ladder():
    spec = importlib.util.spec_from_file_location("adf5355_ladder", LADDER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestReferenceDefault(unittest.TestCase):
    def test_package_cli_defaults_to_125_mhz(self):
        args = cli.build_parser().parse_args(["dump", "--freq", "2.4G"])
        self.assertEqual(args.ref_mhz, 125.0)

    def test_package_cli_reference_is_still_overridable(self):
        args = cli.build_parser().parse_args(
            ["dump", "--freq", "2.4G", "--ref-mhz", "10"])
        self.assertEqual(args.ref_mhz, 10.0)

    def test_ladder_defaults_to_125_mhz(self):
        args = load_ladder().parse_args([])
        self.assertEqual(args.ref_mhz, 125.0)

    def test_ladder_reference_is_still_overridable(self):
        args = load_ladder().parse_args(["--ref-mhz", "40"])
        self.assertEqual(args.ref_mhz, 40.0)

    def test_default_produces_a_62_5_mhz_pfd(self):
        # 125 MHz with the R counter auto-selected to stay at or below 75 MHz.
        from adf5355 import SynthConfig
        cfg = SynthConfig(ref_hz=round(cli.DEFAULT_REF_MHZ * 1e6))
        self.assertEqual(cfg.r_divider, 2)
        self.assertEqual(cfg.fpfd_hz, 62_500_000)

    def test_default_reference_covers_the_whole_rfouta_band(self):
        from adf5355 import Channel, SynthConfig, plan
        cfg = SynthConfig(ref_hz=round(cli.DEFAULT_REF_MHZ * 1e6))
        for freq in (60_000_000, 1_000_000_000, 2_400_000_000,
                     3_400_000_000, 5_000_000_000, 6_800_000_000):
            p = plan(cfg, freq, Channel.A)
            self.assertLess(abs(float(p.solution.error_hz)), 1.0, f"{freq} Hz")


if __name__ == "__main__":
    unittest.main()


class TestRfGate(unittest.TestCase):
    """--enable-rf must gate every command that builds a register image."""

    def _r6(self, argv):
        from adf5355 import Channel, plan
        args = cli.build_parser().parse_args(argv)
        channel = Channel(args.channel)
        enable = args.enable_rf
        cfg = cli.build_config(args, enable and channel is Channel.A,
                               enable and channel is Channel.B)
        return plan(cfg, args.freq, channel).registers

    def test_dump_without_enable_rf_disables_both_outputs(self):
        regs = self._r6(["dump", "--freq", "2.4G"])
        self.assertEqual(regs.get("rf_out_enable"), 0)
        self.assertEqual(regs.get("rf_outb_disable"), 1)

    def test_dump_with_enable_rf_enables_channel_a(self):
        regs = self._r6(["dump", "--freq", "2.4G", "--enable-rf"])
        self.assertEqual(regs.get("rf_out_enable"), 1)

    def test_dump_with_enable_rf_enables_channel_b_active_low(self):
        regs = self._r6(["dump", "--freq", "11.7G", "--channel", "B", "--enable-rf"])
        self.assertEqual(regs.get("rf_outb_disable"), 0)
        self.assertEqual(regs.get("rf_out_enable"), 0)

    def test_dwell_defaults_to_ten_seconds_and_is_gated(self):
        args = cli.build_parser().parse_args(["dwell", "--freq", "2.4G"])
        self.assertEqual(args.dwell, 10.0)
        self.assertFalse(args.enable_rf)
