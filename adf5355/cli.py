"""Command line front end.

    python3 -m adf5355 dump  --freq 2.4G
    python3 -m adf5355 set   --freq 11.7G --channel B --enable-rf
    python3 -m adf5355 dwell --freq 2.4G --dwell 30 --enable-rf
    python3 -m adf5355 sweep --start 1G --stop 6G --points 51
    python3 -m adf5355 off

--ref-mhz defaults to this board's 125 MHz reference; pass it explicitly for
any board whose X1 is marked otherwise.

RF is never enabled without --enable-rf.
"""
from __future__ import annotations

import argparse
import signal
import sys
import time

from .device import (DEFAULT_CE_GPIO, DEFAULT_MUXOUT_GPIO, DEFAULT_SPI_HZ,
                     ADF5355, LockTimeout)
from .ladder import (make_ladder, format_ladder, run_ladder,
                     overlaps_satellite_band, check_schedule_feasible,
                     DEFAULT_START_HZ, DEFAULT_STOP_HZ, DEFAULT_STEPS,
                     DEFAULT_TOTAL_S)
from .hopper import (DEFAULT_BAND_EXTRA_S, DEFAULT_BLOCK,
                     DEFAULT_CLUSTER_POINTS, DEFAULT_CLUSTER_SPAN_HZ,
                     DEFAULT_CYCLES, DEFAULT_HOP_START_HZ,
                     DEFAULT_HOP_STOP_HZ, DEFAULT_JITTER, DEFAULT_MIN_HOP_S,
                     DEFAULT_PERIOD_CYCLES, DEFAULT_POINTS, DEFAULT_SEED,
                     cluster_centres, describe as describe_hops,
                     describe_clusters, make_cluster_schedule, make_schedule,
                     period_duration, plan_clusters, plan_frequencies,
                     run_hops)
from .plan import Channel, SynthConfig, plan
from .registers import MuxOut, OutputPower

# X1 on this board is marked R125.000.  Overridable, never guessed.
DEFAULT_REF_MHZ = 125.0

# Lowest of the four RFoutA steps (-4 dBm).  Least likely to damage
# whatever is on the other end of the cable; raise it deliberately.
DEFAULT_POWER = 0

# Ku-band satellite downlink; the ladder script carries the same warning.
SATBAND_LO_HZ = 10_700_000_000
SATBAND_HI_HZ = 12_700_000_000

_SUFFIXES = {"k": 1e3, "m": 1e6, "g": 1e9}


def parse_frequency(text: str) -> int:
    """Accept 2400000000, 2.4e9, 2400M, 11.7G, 53.125k."""
    text = text.strip().replace("_", "").replace("hz", "").replace("Hz", "")
    multiplier = 1.0
    if text and text[-1].lower() in _SUFFIXES:
        multiplier = _SUFFIXES[text[-1].lower()]
        text = text[:-1]
    try:
        return round(float(text) * multiplier)
    except ValueError:
        raise argparse.ArgumentTypeError(f"cannot parse frequency {text!r}")


def resolve_channel(args) -> Channel:
    """Explicit --channel wins; otherwise the subcommand's own fallback."""
    return Channel(args.channel or getattr(args, "channel_default", "A"))


def build_config(args, outa: bool, outb: bool) -> SynthConfig:
    return SynthConfig(
        ref_hz=round(args.ref_mhz * 1e6),
        ref_doubler=args.ref_doubler,
        ref_div2=args.ref_div2,
        ref_diff=args.ref_diff,
        r_counter=args.r_counter,
        cp_ua=args.cp_ua,
        muxout=MuxOut.DIGITAL_LOCK_DETECT,
        muxout_3v3=True,
        outa_enable=outa,
        outa_power=OutputPower(args.power),
        outb_enable=outb,
        mute_till_lock=not args.no_mute_till_lock,
        negative_bleed=not args.no_negative_bleed,
        solver=args.solver,
    )


def open_device(args, config: SynthConfig) -> ADF5355:
    return ADF5355(
        config, bus=args.bus, device=args.device, spi_hz=args.spi_hz,
        ce_gpio=None if args.ce_gpio < 0 else args.ce_gpio,
        muxout_gpio=None if args.muxout_gpio < 0 else args.muxout_gpio,
        dry_run=args.dry_run,
    ).open()


def report_lock(dev: ADF5355, args) -> bool:
    if not dev.can_detect_lock:
        dev.settle()
        print("  lock UNVERIFIED (MUXOUT not wired -- open loop)")
        return True
    if args.no_lock_check:
        return True
    try:
        waited = dev.wait_for_lock(args.lock_timeout)
    except LockTimeout as exc:
        print(f"  LOCK FAILED: {exc}", file=sys.stderr)
        return False
    print(f"  locked in {waited * 1e3:.1f} ms")
    return True


def cmd_dump(args) -> int:
    channel = resolve_channel(args)
    # Mirror the RF gate exactly: dump exists to preview the bytes that would
    # actually be written, so showing an output enabled when it would not be
    # defeats the point.
    enable = args.enable_rf
    config = build_config(args, enable and channel is Channel.A,
                          enable and channel is Channel.B)
    p = plan(config, args.freq, channel)
    print(p.summary())
    if not enable:
        print(f"  outputs DISABLED in this image "
              f"(rf_out_enable=0, rf_outb_disable=1); add --enable-rf to see "
              f"the transmitting image")
    print()
    print(p.registers.dump())
    print()
    print("Cold-start write order (R0 last, carrying autocal):")
    dev = ADF5355(config, dry_run=True)
    dev.program(p)
    print("  " + "\n  ".join(str(r) for r in dev.trace))
    return 0


def cmd_set(args) -> int:
    channel = resolve_channel(args)
    enable = args.enable_rf
    config = build_config(args, enable and channel is Channel.A,
                          enable and channel is Channel.B)
    p = plan(config, args.freq, channel)
    print(p.summary())
    if not enable and not args.dry_run:
        print("\nRF stays disabled; re-run with --enable-rf into a "
              "shielded/attenuated path.")

    dev = open_device(args, config)
    try:
        dev.program(p)
        print(f"\nprogrammed RFout{channel.value} "
              f"{float(p.solution.achieved_hz)/1e9:.9f} GHz")
        ok = report_lock(dev, args)
        if args.hold:
            print(f"holding for {args.hold:.3f} s (Ctrl-C to stop)")
            time.sleep(args.hold)
        else:
            return 0 if ok else 1
    finally:
        if args.hold or not args.enable_rf:
            dev.close()
    return 0


def cmd_dwell(args) -> int:
    """Hold one frequency for a fixed time, then mute and exit."""
    channel = resolve_channel(args)
    enable = args.enable_rf
    config = build_config(args, enable and channel is Channel.A,
                          enable and channel is Channel.B)
    p = plan(config, args.freq, channel)
    print(p.summary())

    if SATBAND_LO_HZ <= args.freq <= SATBAND_HI_HZ:
        print("\nSAFETY: this frequency is inside the 10.7-12.7 GHz satellite "
              "downlink band. Use a shielded, attenuated path; do not radiate.")

    dev = open_device(args, config)
    try:
        dev.program(p)
        if not report_lock(dev, args):
            print("not transmitting: never locked", file=sys.stderr)
            return 1

        achieved = float(p.solution.achieved_hz) / 1e9
        if not enable:
            print(f"\nprogrammed RFout{channel.value} {achieved:.9f} GHz but RF "
                  f"is disabled, so there is nothing to dwell on.\n"
                  f"Re-run with --enable-rf into a shielded/attenuated path.")
            return 0

        print(f"\ntransmitting RFout{channel.value} {achieved:.9f} GHz "
              f"for {args.dwell:g} s (Ctrl-C to stop early)")
        time.sleep(args.dwell)
    finally:
        dev.close()          # close() mutes both outputs on the way out
    print("dwell complete; output muted")
    return 0


def cmd_ladder(args) -> int:
    """Duration-coded ladder: rung n transmits for n*u, then is quiet for n*u."""
    channel = resolve_channel(args)
    enable = args.enable_rf
    try:
        steps = make_ladder(round(args.start_ghz * 1e9),
                            round(args.stop_ghz * 1e9),
                            args.steps, args.total_s)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        check_schedule_feasible(steps)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    config = build_config(args, enable and channel is Channel.A,
                          enable and channel is Channel.B)
    # Validate every rung up front: the per-frequency band check would
    # otherwise only fire once the ladder was already running.
    for step in steps:
        try:
            plan(config, step.freq_hz, channel)
        except ValueError as exc:
            print(f"error: rung {step.index} ({step.freq_hz/1e9:.3f} GHz): {exc}",
                  file=sys.stderr)
            return 2

    print(format_ladder(steps))
    if overlaps_satellite_band(steps):
        print("\nSAFETY: this range overlaps the 10.7-12.7 GHz satellite "
              "downlink band. Use a shielded, attenuated path; do not radiate.")
    else:
        print("\nSAFETY: bench equipment. Do not connect an antenna.")

    if not enable:
        print("\nRF disabled, so nothing was transmitted. Re-run with "
              "--enable-rf into a shielded/attenuated path.")
        return 0

    def check(dev, step):
        if not dev.can_detect_lock:
            return True
        try:
            dev.wait_for_lock(args.lock_timeout)
            return True
        except LockTimeout:
            print(f"  rung {step.index}: LOCK FAILED", file=sys.stderr)
            return False

    dev = open_device(args, config)
    try:
        failures = run_ladder(dev, steps, channel, args.loops,
                              None if args.no_lock_check else check,
                              verbose=steps[0].on_s >= 0.05)
    finally:
        dev.close()
    if failures:
        print(f"\n{failures} rung(s) failed to lock", file=sys.stderr)
        return 1
    print("\nladder complete; output muted")
    return 0


def cmd_hop(args) -> int:
    """Seeded pseudorandom frequency hopping; the receiver regenerates it."""
    channel = resolve_channel(args)
    enable = args.enable_rf
    try:
        freqs = plan_frequencies(round(args.start_ghz * 1e9),
                                 round(args.stop_ghz * 1e9), args.points)
        hops = make_schedule(args.seed, freqs, args.min_hop_ms / 1e3,
                             args.cycles, args.jitter, args.period_cycles)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Mute-till-lock would blank each dwell while the loop re-settles; hops
    # stay inside one VCO band so there is nothing to re-acquire.
    args.no_mute_till_lock = True
    config = build_config(args, enable and channel is Channel.A,
                          enable and channel is Channel.B)
    for freq in freqs:                       # validate before transmitting
        try:
            plan(config, freq, channel)
        except ValueError as exc:
            print(f"error: {freq/1e9:.6f} GHz: {exc}", file=sys.stderr)
            return 2

    print(describe_hops(hops, freqs, args.seed, args.min_hop_ms / 1e3,
                        args.jitter))
    print(f"\n  the receiver needs only: seed 0x{args.seed:X}, "
          f"{args.start_ghz}-{args.stop_ghz} GHz, {args.points} points, "
          f"hop {args.min_hop_ms:g} ms, jitter {args.jitter:g}, "
          f"period-cycles {args.period_cycles}")
    lo, hi = freqs[0], freqs[-1]
    if lo <= SATBAND_HI_HZ and hi >= SATBAND_LO_HZ:
        print("\nSAFETY: overlaps the 10.7-12.7 GHz satellite downlink band. "
              "Closed, attenuated path only; do not radiate.")
    else:
        print("\nSAFETY: bench equipment. Do not connect an antenna.")

    if not enable:
        print("\nRF disabled, so nothing was transmitted. Re-run with "
              "--enable-rf into a shielded/attenuated path.")
        return 0

    dev = open_device(args, config)
    try:
        # Program the first point BEFORE testing lock: nothing is loaded until
        # then, so MUXOUT sits at its power-on three-state and any lock check
        # here would always fail.
        dev.set_frequency(hops[0].freq_hz, channel)
        if dev.can_detect_lock and not args.no_lock_check:
            try:
                waited = dev.wait_for_lock(args.lock_timeout)
                print(f"  locked in {waited*1e3:.1f} ms")
            except LockTimeout as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
        start = time.monotonic()
        run_hops(dev, hops, channel)
        elapsed = time.monotonic() - start
    finally:
        dev.close()
    print(f"\ntransmitted {len(hops)} hops in {elapsed:.4f} s "
          f"(scheduled {hops[-1].end_s:.4f} s, "
          f"error {(elapsed - hops[-1].end_s)*1e3:+.1f} ms); output muted")
    return 0


def cmd_hop_lever(args) -> int:
    """Cluster hopping: the same seeded schedule, over a real frequency lever arm.

    One narrow cluster measures the receiver's total offset precisely and
    cannot say how much of it is the SDR's clock and how much is the LNB's LO.
    Spreading the same schedule over clusters hundreds of megahertz apart is
    what separates them, because only the SDR's clock error scales with the
    intermediate frequency.

    The transmitter is free-running and knows nothing about the receiver: it
    hops over every cluster forever, and a receiver tuned to any one of them
    sees a complete, decodable pattern for that cluster alone.
    """
    channel = resolve_channel(args)
    enable = args.enable_rf
    try:
        centres = cluster_centres(args.low_ghz * 1e9, args.high_ghz * 1e9,
                                  args.clusters)
        plan_c = plan_clusters(centres, args.cluster_points,
                               round(args.span_khz * 1e3))
        hops = make_cluster_schedule(args.seed, plan_c, args.min_hop_ms / 1e3,
                                     args.cycles, args.block, args.jitter,
                                     args.period_cycles,
                                     args.band_extra_ms / 1e3)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Within a cluster the span is well under a megahertz, so those hops stay
    # inside one VCO band and are written as dividers only. Cluster changes
    # get a full retune with the band search; mute-till-lock would blank the
    # dwell while that runs, so it stays off and the schedule pays for the
    # settling with a longer dwell instead.
    args.no_mute_till_lock = True
    config = build_config(args, enable and channel is Channel.A,
                          enable and channel is Channel.B)
    freqs = plan_c.all_freqs
    worst = 0.0
    for freq in freqs:
        try:
            p = plan(config, freq, channel)
        except ValueError as exc:
            print(f"error: {freq/1e9:.6f} GHz: {exc}", file=sys.stderr)
            return 2
        worst = max(worst, abs(float(p.solution.error_hz)))

    per_cycle = plan_c.clusters * plan_c.points
    print(describe_clusters(plan_c, hops, args.seed, args.min_hop_ms / 1e3,
                            args.block, args.lo_hz, args.period_cycles,
                            args.band_extra_ms / 1e3))
    print(f"  plan error: worst point is {worst*1e3:.4f} mHz from nominal "
          f"across the whole lever arm")
    print(f"  run time  : {hops[-1].end_s:.1f} s = {hops[-1].end_s/60:.0f} min "
          f"total over {args.cycles} cycles -- leave this running for the "
          f"whole receiver run, which is about 10 s per capture (not 3), so "
          f"25 sweeps of 4 clusters is ~17 min and 50 is ~33")
    print(f"\n  the receiver needs only: seed 0x{args.seed:X}, "
          f"{args.low_ghz}-{args.high_ghz} GHz in {args.clusters} clusters, "
          f"{args.cluster_points} points over {args.span_khz:g} kHz, "
          f"hop {args.min_hop_ms:g} ms, block {args.block}, band-extra "
          f"{args.band_extra_ms:g} ms, period-cycles {args.period_cycles}")

    lo, hi = min(freqs), max(freqs)
    if lo <= SATBAND_HI_HZ and hi >= SATBAND_LO_HZ:
        print("\nSAFETY: overlaps the 10.7-12.7 GHz satellite downlink band. "
              "Closed, attenuated path only; do not radiate.")
    else:
        print("\nSAFETY: bench equipment. Do not connect an antenna.")

    if not enable:
        print("\nRF disabled, so nothing was transmitted. Re-run with "
              "--enable-rf into a shielded/attenuated path.")
        return 0

    autocal_plans = {f: plan(config, f, channel) for f in freqs}
    dev = open_device(args, config)
    try:
        dev.set_frequency(hops[0].freq_hz, channel)
        if dev.can_detect_lock and not args.no_lock_check:
            try:
                waited = dev.wait_for_lock(args.lock_timeout)
                print(f"  locked in {waited*1e3:.1f} ms")
            except LockTimeout as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
        start = time.monotonic()
        run_hops(dev, hops, channel, autocal_plans=autocal_plans)
        elapsed = time.monotonic() - start
    finally:
        dev.close()
    print(f"\ntransmitted {len(hops)} hops in {elapsed:.4f} s "
          f"(scheduled {hops[-1].end_s:.4f} s, "
          f"error {(elapsed - hops[-1].end_s)*1e3:+.1f} ms); output muted")
    return 0


def cmd_sweep(args) -> int:
    channel = resolve_channel(args)
    config = build_config(args, args.enable_rf and channel is Channel.A,
                          args.enable_rf and channel is Channel.B)
    if args.points < 2:
        print("--points must be at least 2", file=sys.stderr)
        return 2
    step = (args.stop - args.start) / (args.points - 1)
    freqs = [round(args.start + step * i) for i in range(args.points)]

    dev = open_device(args, config)
    failures = 0
    try:
        for i, freq in enumerate(freqs):
            p = dev.set_frequency(freq, channel)
            s = p.solution
            marker = "  <- divider change" if i and \
                s.rf_divider_select != prev_div else ""
            prev_div = s.rf_divider_select
            print(f"{i + 1:>4}/{len(freqs)}  {freq / 1e9:12.6f} GHz  "
                  f"/{1 << s.rf_divider_select:<2}  "
                  f"err {float(s.error_hz):+9.3f} Hz{marker}")
            if not report_lock(dev, args):
                failures += 1
            time.sleep(args.dwell)
    finally:
        dev.close()
    if failures:
        print(f"\n{failures}/{len(freqs)} points failed to lock", file=sys.stderr)
        return 1
    print(f"\nall {len(freqs)} points locked")
    return 0


def cmd_off(args) -> int:
    config = build_config(args, False, False)
    dev = open_device(args, config)
    try:
        dev.program(plan(config, args.freq, resolve_channel(args)))
        dev.mute()
        print("outputs disabled")
    finally:
        dev.close()
    return 0


def cmd_probe(args) -> int:
    if args.muxout_gpio < 0:
        print("probe needs MUXOUT wired to a GPIO; pass --muxout-gpio N",
              file=sys.stderr)
        return 2
    config = build_config(args, False, False)
    dev = open_device(args, config)
    try:
        result = dev.probe()
    finally:
        dev.close()
    print(f"MUXOUT commanded high -> read {int(bool(result['high']))}")
    print(f"MUXOUT commanded low  -> read {int(bool(result['low']))}")
    if result["ok"]:
        print("PASS: the ADF5355 is receiving and acting on SPI writes")
        return 0
    print("FAIL: check CE is high, LE/CLK/DAT wiring, and board power",
          file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adf5355", description="ADF5355 synthesizer control")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ref-mhz", type=float, default=DEFAULT_REF_MHZ,
                        help=f"actual REFIN frequency on your board, in MHz "
                             f"(default {DEFAULT_REF_MHZ:g}, this board's X1). "
                             f"Clone boards differ -- read the marking rather "
                             f"than assuming")
    common.add_argument("--ref-doubler", action="store_true",
                        help="double REFIN before the R counter")
    common.add_argument("--ref-div2", action="store_true",
                        help="halve the reference after the R counter")
    common.add_argument("--ref-diff", action="store_true",
                        help="differential REFIN input")
    common.add_argument("--r-counter", type=int, default=None,
                        help="force the R divider instead of maximizing f_PFD")
    common.add_argument("--cp-ua", type=int, default=900,
                        help="charge pump current, 315-5040 uA (default 900)")
    common.add_argument("--power", type=int, default=DEFAULT_POWER,
                        choices=[0, 1, 2, 3],
                        help=f"RFoutA power: 0=-4 1=-1 2=+2 3=+5 dBm "
                             f"(default {DEFAULT_POWER}, the lowest setting)")
    common.add_argument("--channel", default=None, choices=["A", "B"],
                        help="A = VCO/2**d (53.125 MHz-6.8 GHz), "
                             "B = doubler (6.8-13.6 GHz). Defaults to A, "
                             "except ladder which defaults to B")
    common.add_argument("--solver", default="adi", choices=["adi", "exact"])
    common.add_argument("--no-mute-till-lock", action="store_true")
    common.add_argument("--no-negative-bleed", action="store_true")
    common.add_argument("--bus", type=int, default=0,
                        help="SPI bus number (default 0)")
    common.add_argument("--device", type=int, default=0,
                        help="SPI chip select driving LE (0 = CE0)")
    common.add_argument("--spi-hz", type=int, default=DEFAULT_SPI_HZ,
                        help=f"SPI clock in Hz (default {DEFAULT_SPI_HZ}). "
                             f"The part accepts up to 50 MHz; lower this if "
                             f"long jumper wires are marginal")
    common.add_argument("--ce-gpio", type=int, default=DEFAULT_CE_GPIO,
                        help="BCM pin driving CE; -1 if CE is strapped high")
    common.add_argument("--muxout-gpio", type=int, default=DEFAULT_MUXOUT_GPIO,
                        help=f"BCM pin reading MUXOUT (default "
                             f"{DEFAULT_MUXOUT_GPIO} = header pin 35); "
                             f"-1 if not wired, which leaves no lock detect "
                             f"and no way to confirm a write landed")
    common.add_argument("--lock-timeout", type=float, default=0.5,
                        help="seconds to wait for digital lock detect before "
                             "calling a point failed (default 0.5; a healthy "
                             "lock takes single-digit ms)")
    common.add_argument("--no-lock-check", action="store_true",
                        help="skip lock detect entirely and never fail on it")
    common.add_argument("--dry-run", action="store_true",
                        help="compute everything, touch no hardware")
    common.add_argument("--enable-rf", action="store_true",
                        help="required before any output is enabled")

    sub = parser.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("probe", parents=[common],
                        help="prove the chip receives writes, via MUXOUT")
    pr.set_defaults(func=cmd_probe)

    d = sub.add_parser("dump", parents=[common],
                       help="print the register image and write order")
    d.add_argument("--freq", type=parse_frequency, required=True)
    d.set_defaults(func=cmd_dump)

    s = sub.add_parser("set", parents=[common], help="program one frequency")
    s.add_argument("--freq", type=parse_frequency, required=True)
    s.add_argument("--hold", type=float, default=0.0,
                   help="seconds to hold the output before muting")
    s.set_defaults(func=cmd_set)

    dw = sub.add_parser("dwell", parents=[common],
                        help="transmit one frequency for a fixed time, then "
                             "mute and exit")
    dw.add_argument("--freq", type=parse_frequency, required=True)
    dw.add_argument("--dwell", type=float, default=10.0,
                    help="seconds to transmit before muting and exiting "
                         "(default 10)")
    dw.set_defaults(func=cmd_dwell)

    hop = sub.add_parser("hop", parents=[common],
                         help="seeded pseudorandom frequency hopping; a "
                              "receiver knowing the seed regenerates it exactly")
    hop.add_argument("--seed", type=lambda v: int(v, 0), default=DEFAULT_SEED,
                     help=f"shared schedule seed (default 0x{DEFAULT_SEED:X}); "
                          f"accepts 0x notation")
    hop.add_argument("--start-ghz", type=float,
                     default=DEFAULT_HOP_START_HZ / 1e9,
                     help=f"first frequency point in GHz (default "
                          f"{DEFAULT_HOP_START_HZ / 1e9:g})")
    hop.add_argument("--stop-ghz", type=float,
                     default=DEFAULT_HOP_STOP_HZ / 1e9,
                     help=f"last frequency point in GHz (default "
                          f"{DEFAULT_HOP_STOP_HZ / 1e9:g}); the whole span "
                          f"should fit the receiver's instantaneous bandwidth")
    hop.add_argument("--points", type=int, default=DEFAULT_POINTS,
                     help=f"number of frequency points (default {DEFAULT_POINTS})")
    hop.add_argument("--min-hop-ms", type=float,
                     default=DEFAULT_MIN_HOP_S * 1e3,
                     help=f"dwell in ms (default "
                          f"{DEFAULT_MIN_HOP_S*1e3:g}); fixed unless "
                          f"--jitter, which makes it min*(1 + jitter*rand*2)")
    hop.add_argument("--jitter", type=float, default=DEFAULT_JITTER,
                     help=f"dwell randomness 0..1 (default {DEFAULT_JITTER:g} = "
                          f"fixed). dwell = min*(1 + jitter*rand*2); identity "
                          f"comes from the frequency order, not from timing")
    hop.add_argument("--period-cycles", type=int,
                     default=DEFAULT_PERIOD_CYCLES,
                     help=f"permutations before the pattern repeats (default "
                          f"{DEFAULT_PERIOD_CYCLES}); the receiver searches "
                          f"one period for the epoch, so both ends must agree")
    hop.add_argument("--cycles", type=int, default=DEFAULT_CYCLES,
                     help=f"permutations to transmit (default "
                          f"{DEFAULT_CYCLES})")
    hop.set_defaults(func=cmd_hop, channel_default="B")

    lev = sub.add_parser("hop-lever", parents=[common],
                         help="seeded hopping over several widely separated "
                              "clusters, so a receiver can separate its own "
                              "clock error from the LNB's LO error")
    lev.add_argument("--seed", type=lambda v: int(v, 0), default=DEFAULT_SEED,
                     help=f"shared schedule seed (default 0x{DEFAULT_SEED:X})")
    lev.add_argument("--low-ghz", type=float, default=10.70,
                     help="lowest cluster centre in GHz (default 10.70, the "
                          "bottom of the LNB low band)")
    lev.add_argument("--high-ghz", type=float, default=11.90,
                     help="highest cluster centre in GHz (default 11.90); the "
                          "gap between these two IS the lever arm")
    lev.add_argument("--clusters", type=int, default=4,
                     help="how many clusters to spread across that range "
                          "(default 4). More gives a linearity check; fewer "
                          "gives each one more of the air time")
    lev.add_argument("--cluster-points", type=int,
                     default=DEFAULT_CLUSTER_POINTS,
                     help=f"points per cluster (default "
                          f"{DEFAULT_CLUSTER_POINTS})")
    lev.add_argument("--span-khz", type=float,
                     default=DEFAULT_CLUSTER_SPAN_HZ / 1e3,
                     help=f"how wide one cluster is, in kHz (default "
                          f"{DEFAULT_CLUSTER_SPAN_HZ/1e3:g}). The points sit "
                          f"on a Golomb ruler inside it, not on a regular "
                          f"grid. The whole cluster must fit the receiver's "
                          f"passband with room left over for its tuning "
                          f"dither")
    lev.add_argument("--min-hop-ms", type=float,
                     default=DEFAULT_MIN_HOP_S * 1e3,
                     help=f"dwell in ms (default {DEFAULT_MIN_HOP_S*1e3:g})")
    lev.add_argument("--block", type=int, default=DEFAULT_BLOCK,
                     help=f"consecutive dwells per cluster visit (default "
                          f"{DEFAULT_BLOCK}); must divide --cluster-points. "
                          f"Only the first dwell of a block changes band")
    lev.add_argument("--band-extra-ms", type=float,
                     default=DEFAULT_BAND_EXTRA_S * 1e3,
                     help=f"how much longer a band-changing dwell is, to pay "
                          f"for the VCO band search (default "
                          f"{DEFAULT_BAND_EXTRA_S*1e3:g} ms). The receiver "
                          f"skips exactly this much and reports whether it "
                          f"was enough")
    lev.add_argument("--jitter", type=float, default=DEFAULT_JITTER)
    lev.add_argument("--period-cycles", type=int,
                     default=DEFAULT_PERIOD_CYCLES,
                     help="cycles before the pattern repeats; bounds the "
                          "receiver's epoch search")
    lev.add_argument("--cycles", type=int, default=16000,
                     help="cycles to transmit (default 16000: at the default "
                          "dwell one cycle is 280 ms, so that is 75 minutes). "
                          "It must outlast the WHOLE receiver run, and the "
                          "receiver spends far more time decoding than "
                          "listening -- about 10 s per 3 s capture on a Pi -- "
                          "so 50 sweeps of 4 clusters is over half an hour. "
                          "The printed run time below is the number to "
                          "compare against, not the capture seconds")
    lev.add_argument("--lo-hz", type=float, default=9.75e9,
                     help="nominal LNB LO, printed so both ends agree on what "
                          "IF each cluster lands at")
    lev.set_defaults(func=cmd_hop_lever, channel_default="B")

    lad = sub.add_parser("ladder", parents=[common],
                         help="duration-coded ladder: rung n transmits for "
                              "n*u seconds, so burst length identifies the rung")
    lad.add_argument("--start-ghz", type=float, default=DEFAULT_START_HZ / 1e9,
                     help=f"first rung in GHz (default "
                          f"{DEFAULT_START_HZ / 1e9:g})")
    lad.add_argument("--stop-ghz", type=float, default=DEFAULT_STOP_HZ / 1e9,
                     help=f"last rung in GHz (default {DEFAULT_STOP_HZ / 1e9:g})")
    lad.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                     help=f"number of rungs (default {DEFAULT_STEPS})")
    lad.add_argument("--total-s", type=float, default=DEFAULT_TOTAL_S,
                     help=f"total coded interval in seconds (default "
                          f"{DEFAULT_TOTAL_S:g})")
    lad.add_argument("--loops", type=int, default=1,
                     help="number of complete ladders to run (default 1)")
    # The guide's pattern is RFoutB; 10.7-12.7 GHz is out of RFoutA's range.
    lad.set_defaults(func=cmd_ladder, channel_default="B")

    w = sub.add_parser("sweep", parents=[common],
                       help="step across a range, checking lock at each point")
    w.add_argument("--start", type=parse_frequency, required=True)
    w.add_argument("--stop", type=parse_frequency, required=True)
    w.add_argument("--points", type=int, default=51,
                   help="number of frequencies to step through, inclusive of "
                        "both ends (default 51)")
    w.add_argument("--dwell", type=float, default=0.02,
                   help="seconds to hold each point after it locks (default "
                        "0.02). Raise it to follow the sweep on an analyser")
    w.set_defaults(func=cmd_sweep)

    o = sub.add_parser("off", parents=[common], help="disable both outputs")
    o.add_argument("--freq", type=parse_frequency, default=2_400_000_000)
    o.set_defaults(func=cmd_off)
    return parser


def _shutdown_gpio() -> None:
    """Tear down gpiozero's pin factory before the interpreter finalizes.

    The lgpio factory keeps a daemon thread alive.  If the interpreter starts
    finalizing while it is still running, CPython intermittently aborts with
    "could not acquire lock for <_io.BufferedWriter name='<stderr>'> at
    interpreter shutdown", and the process exits 134 even though the command
    already succeeded -- which silently breaks anything checking exit status.
    Closing the factory here retires that thread while the runtime is healthy.

    Looked up through sys.modules so a --dry-run never imports gpiozero.
    """
    module = sys.modules.get("gpiozero")
    if module is None:
        return
    try:
        factory = module.Device.pin_factory
        if factory is not None:
            factory.close()
            module.Device.pin_factory = None
    except Exception:
        pass


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    def on_signal(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_signal)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted; outputs muted", file=sys.stderr)
        return 130
    except (ValueError, LockTimeout) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        _shutdown_gpio()


if __name__ == "__main__":
    raise SystemExit(main())
