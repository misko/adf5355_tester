"""The reference default must stay pinned to this board's X1 marking."""
import contextlib
import importlib.util
import io
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
        channel = cli.resolve_channel(args)
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


class TestPowerDefault(unittest.TestCase):
    def test_cli_defaults_to_the_lowest_power_step(self):
        args = cli.build_parser().parse_args(["dwell", "--freq", "2.4G"])
        self.assertEqual(args.power, 0)
        self.assertEqual(cli.DEFAULT_POWER, 0)

    def test_library_default_matches_the_cli_default(self):
        from adf5355 import OutputPower, SynthConfig
        self.assertEqual(SynthConfig(ref_hz=125_000_000).outa_power,
                         OutputPower.MINUS_4_DBM)
        self.assertEqual(int(OutputPower.MINUS_4_DBM), cli.DEFAULT_POWER)

    def test_power_is_still_settable_to_maximum(self):
        from adf5355 import Channel, plan
        args = cli.build_parser().parse_args(
            ["dump", "--freq", "2.4G", "--enable-rf", "--power", "3"])
        cfg = cli.build_config(args, True, False)
        regs = plan(cfg, args.freq, Channel.A).registers
        self.assertEqual(regs.get("output_power"), 3)


class TestThreadExceptHook(unittest.TestCase):
    """Only lgpio's benign teardown error may be swallowed."""

    def setUp(self):
        import importlib
        self.mod = importlib.import_module("adf5355.__main__") \
            if "adf5355.__main__" in sys.modules else None

    def _hook_module(self):
        # Import the entry point's helpers without executing main().
        import types, pathlib
        src = pathlib.Path(context.ROOT, "adf5355", "entry.py").read_text()
        src = src.split("threading.excepthook = _thread_excepthook")[0]
        ns = types.ModuleType("entry_helpers")
        exec(compile(src, "__main__.py", "exec"), ns.__dict__)
        return ns

    def _args(self, exc_type, exc_value):
        class A:
            pass
        a = A(); a.exc_type = exc_type; a.exc_value = exc_value
        a.exc_traceback = None; a.thread = None
        return a

    def test_system_exit_is_ignored(self):
        ns = self._hook_module()
        self.assertTrue(ns._is_lgpio_teardown_noise(self._args(SystemExit, SystemExit())))

    def test_real_errors_are_not_ignored(self):
        ns = self._hook_module()
        for exc in (ValueError("boom"), RuntimeError("bad"), OSError("io")):
            self.assertFalse(
                ns._is_lgpio_teardown_noise(self._args(type(exc), exc)),
                f"{type(exc).__name__} must still be reported")

    def test_unrelated_lgpio_errors_are_not_ignored(self):
        ns = self._hook_module()

        class FakeLgpioError(Exception):
            pass
        FakeLgpioError.__module__ = "lgpio"
        FakeLgpioError.__name__ = "error"
        self.assertFalse(ns._is_lgpio_teardown_noise(
            self._args(FakeLgpioError, FakeLgpioError("GPIO busy"))))
        self.assertTrue(ns._is_lgpio_teardown_noise(
            self._args(FakeLgpioError, FakeLgpioError("unknown handle"))))


class TestChannelDefaults(unittest.TestCase):
    """argparse's set_defaults mutates the shared parent action in place.

    Giving `ladder` a channel default that way silently changed the default for
    every other subcommand, so `dump --freq 2.4G` started failing as RFoutB.
    """

    def test_ladder_defaults_to_b_without_contaminating_others(self):
        parser = cli.build_parser()
        ladder_args = parser.parse_args(["ladder"])
        self.assertEqual(cli.resolve_channel(ladder_args).value, "B")
        for argv in (["dump", "--freq", "2.4G"],
                     ["set", "--freq", "2.4G"],
                     ["dwell", "--freq", "2.4G"],
                     ["sweep", "--start", "1G", "--stop", "2G"]):
            with self.subTest(argv=argv[0]):
                self.assertEqual(
                    cli.resolve_channel(parser.parse_args(argv)).value, "A")

    def test_parsing_ladder_first_does_not_change_later_parses(self):
        parser = cli.build_parser()
        parser.parse_args(["ladder"])
        self.assertEqual(
            cli.resolve_channel(parser.parse_args(["dump", "--freq", "2.4G"])).value,
            "A")

    def test_explicit_channel_always_wins(self):
        parser = cli.build_parser()
        self.assertEqual(cli.resolve_channel(
            parser.parse_args(["ladder", "--channel", "A"])).value, "A")
        self.assertEqual(cli.resolve_channel(
            parser.parse_args(["dump", "--freq", "11.7G", "--channel", "B"])).value, "B")


class TestNoSharedDefaultMutation(unittest.TestCase):
    """Guard the whole class of bug, not just the --channel instance.

    Options declared on a `parents=[common]` parser are the SAME action objects
    in every subparser. argparse's set_defaults() writes through to
    `action.default`, so one subcommand setting a default for an inherited
    option silently changes it for all of them. That is how `ladder` flipped
    every other subcommand to channel B.
    """

    @staticmethod
    def _subparsers(parser):
        import argparse
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                return action.choices
        raise AssertionError("no subparsers found")

    def test_no_subcommand_sets_a_default_for_an_inherited_option(self):
        parser = cli.build_parser()
        subs = self._subparsers(parser)

        counts = {}
        for sub in subs.values():
            for action in sub._actions:
                entry = counts.setdefault(id(action), [action, 0])
                entry[1] += 1
        shared = {action.dest for action, n in counts.values() if n > 1}

        for name, sub in subs.items():
            for dest in getattr(sub, "_defaults", {}):
                self.assertNotIn(
                    dest, shared,
                    f"subcommand {name!r} calls set_defaults({dest}=...) for an "
                    f"option inherited from the shared parent parser. argparse "
                    f"writes that through to the shared action, changing the "
                    f"default for every other subcommand. Use a distinct dest "
                    f"(see channel_default / resolve_channel).")

    def test_the_six_point_eight_ghz_ambiguity_is_resolved_explicitly(self):
        """6.8 GHz is valid on both outputs -- the one silent failure case."""
        from adf5355 import Channel, plan
        parser = cli.build_parser()
        cfg_args = parser.parse_args(["dump", "--freq", "6.8G"])
        self.assertEqual(cli.resolve_channel(cfg_args), Channel.A)
        cfg = cli.build_config(cfg_args, True, False)
        for channel in (Channel.A, Channel.B):
            plan(cfg, 6_800_000_000, channel)   # both must be constructible


class TestEverySubcommandBuilds(unittest.TestCase):
    """`off` shipped broken because a refactor missed one call site.

    Every subcommand must parse, resolve a channel, and build a config without
    hardware -- the cheapest check that catches a missed rename.
    """

    ARGV = {
        "probe": [],
        "dump": ["--freq", "2.4G"],
        "set": ["--freq", "2.4G"],
        "dwell": ["--freq", "2.4G"],
        "hop": [],
        "ladder": [],
        "sweep": ["--start", "1G", "--stop", "2G"],
        "off": [],
    }

    def test_all_subcommands_are_covered_by_this_test(self):
        parser = cli.build_parser()
        subs = next(a for a in parser._actions
                    if hasattr(a, "choices") and isinstance(a.choices, dict))
        self.assertEqual(set(subs.choices), set(self.ARGV),
                         "a subcommand was added without a smoke case here")

    def test_each_subcommand_resolves_a_channel_and_builds_a_config(self):
        from adf5355 import Channel, plan
        parser = cli.build_parser()
        for name, extra in self.ARGV.items():
            with self.subTest(command=name):
                args = parser.parse_args([name] + extra)
                channel = cli.resolve_channel(args)
                self.assertIn(channel, (Channel.A, Channel.B))
                cfg = cli.build_config(args, False, False)
                freq = getattr(args, "freq", None)
                if freq is None:
                    freq = getattr(args, "start", None) or 2_400_000_000
                    if channel is Channel.B:
                        freq = 11_700_000_000
                plan(cfg, freq, channel)


class TestHopFlagsReachTheGenerator(unittest.TestCase):
    """Every schedule flag must arrive at ``make_schedule`` unchanged.

    A flag that parses, prints, and is then quietly dropped is worse than one
    that does not exist: the operator sets it on both ends, the transmitter
    ignores it, and the receiver aligns against a schedule that was never sent.
    ``--period-cycles`` was pinned to 1 here once already. Nothing noticed.
    """

    def _record(self, argv):
        recorded = {}
        real = cli.make_schedule

        def spy(seed, freqs, min_hop_s, cycles, jitter=0.0, period_cycles=1):
            recorded.update(seed=seed, freqs=list(freqs), min_hop_s=min_hop_s,
                            cycles=cycles, jitter=jitter,
                            period_cycles=period_cycles)
            return real(seed, freqs, min_hop_s, cycles, jitter, period_cycles)

        cli.make_schedule = spy
        try:
            args = cli.build_parser().parse_args(argv)
            with contextlib.redirect_stdout(io.StringIO()):
                status = cli.cmd_hop(args)         # no --enable-rf: no device
        finally:
            cli.make_schedule = real
        self.assertEqual(status, 0)
        self.assertTrue(recorded, "cmd_hop never built a schedule")
        return recorded

    def test_non_default_flags_all_arrive(self):
        from adf5355 import hopper
        got = self._record(["hop", "--seed", "0x1234", "--start-ghz", "11.0",
                            "--stop-ghz", "11.0002", "--points", "8",
                            "--min-hop-ms", "3.5", "--jitter", "0.5",
                            "--period-cycles", "4", "--cycles", "7"])
        self.assertEqual(got["seed"], 0x1234)
        self.assertEqual(got["freqs"],
                         hopper.plan_frequencies(11_000_000_000,
                                                 11_000_200_000, 8))
        self.assertEqual(got["min_hop_s"], 0.0035)
        self.assertEqual(got["jitter"], 0.5)
        self.assertEqual(got["period_cycles"], 4)
        self.assertEqual(got["cycles"], 7)

    def test_the_bare_command_uses_the_packages_defaults(self):
        from adf5355 import hopper
        got = self._record(["hop"])
        self.assertEqual(got["seed"], hopper.DEFAULT_SEED)
        self.assertEqual(got["min_hop_s"], hopper.DEFAULT_MIN_HOP_S)
        self.assertEqual(got["jitter"], hopper.DEFAULT_JITTER)
        self.assertEqual(got["period_cycles"], hopper.DEFAULT_PERIOD_CYCLES)
        self.assertEqual(got["cycles"], hopper.DEFAULT_CYCLES)
        self.assertEqual(got["freqs"],
                         hopper.plan_frequencies(hopper.DEFAULT_HOP_START_HZ,
                                                 hopper.DEFAULT_HOP_STOP_HZ,
                                                 hopper.DEFAULT_POINTS))

    def test_the_schedule_it_prints_is_the_one_it_would_transmit(self):
        """The preflight is the only view an operator gets before keying up."""
        from adf5355 import hopper
        args = cli.build_parser().parse_args(["hop", "--points", "6",
                                              "--period-cycles", "2"])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(cli.cmd_hop(args), 0)
        freqs = hopper.plan_frequencies(hopper.DEFAULT_HOP_START_HZ,
                                        hopper.DEFAULT_HOP_STOP_HZ, 6)
        hops = hopper.make_schedule(hopper.DEFAULT_SEED, freqs,
                                    hopper.DEFAULT_MIN_HOP_S,
                                    hopper.DEFAULT_CYCLES,
                                    hopper.DEFAULT_JITTER, 2)
        first = ", ".join(f"p{h.point}@{h.dwell_s*1e3:.1f}ms" for h in hops[:8])
        self.assertIn(first, out.getvalue())
        self.assertIn("period-cycles 2", out.getvalue())

