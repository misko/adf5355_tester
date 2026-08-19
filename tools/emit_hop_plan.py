#!/usr/bin/env python3
"""Emit a binary hop plan for tools/hop_tx (the C transmitter).

The C side deliberately reimplements nothing: the seeded schedule and the
fractional-N register maths stay in Python, and C consumes the result. That
keeps one implementation of the protocol, so the two transmitters cannot drift.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adf5355 import Channel, SynthConfig, plan
from adf5355.hopper import make_period, plan_frequencies
from adf5355.registers import FIELDS

MAGIC = 0x36354441  # "AD56" little endian -- v4 adds per-hop autocal

# v3 exists because v2 stored the whole run. The schedule repeats, so the plan
# now carries a single period and the number of hops to play; the transmitter
# indexes it modulo the period length. Memory is then set by the period, not by
# how long the run lasts, which is what makes an indefinite loop possible.


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=lambda v: int(v, 0), default=0xC0FFEE)
    p.add_argument("--start-ghz", type=float, default=11.0)
    p.add_argument("--stop-ghz", type=float, default=11.00171)
    p.add_argument("--points", type=int, default=20)
    p.add_argument("--hop-ms", type=float, default=10.0)
    p.add_argument("--cycles", type=int, default=100,
                   help="how many permutations to play; ignored with --forever")
    p.add_argument("--forever", action="store_true",
                   help="loop until signalled -- total hop count is left at 0")
    p.add_argument("--period-cycles", type=int, default=1,
                   help="permutations per repeat of the pattern")
    p.add_argument("--autocal-every", action="store_true",
                   help="recalibrate the VCO band on every hop. Required "
                        "once the span exceeds one band; costs the settle "
                        "time, so the dwell has to be long enough for it")
    p.add_argument("--ref-mhz", type=float, default=125.0)
    p.add_argument("--power", type=int, default=0)
    p.add_argument("--channel", default="B", choices=["A", "B"])
    p.add_argument("--mute", action="store_true",
                   help="emit a plan that never keys the output, for "
                        "timing measurements with no RF")
    args = p.parse_args()

    channel = Channel(args.channel)
    freqs = plan_frequencies(round(args.start_ghz * 1e9),
                             round(args.stop_ghz * 1e9), args.points)
    period = make_period(args.seed, len(freqs), args.hop_ms / 1e3,
                         period_cycles=args.period_cycles)
    total_hops = 0 if args.forever else args.cycles * len(freqs)

    cfg_on = SynthConfig(ref_hz=round(args.ref_mhz * 1e6),
                         outa_enable=channel is Channel.A,
                         outb_enable=channel is Channel.B,
                         outa_power=args.power, mute_till_lock=False)
    cfg_off = SynthConfig(ref_hz=round(args.ref_mhz * 1e6),
                          outa_enable=False, outb_enable=False,
                          outa_power=args.power, mute_till_lock=False)

    autocal = FIELDS["autocal"].mask
    table = []
    for f in freqs:
        w = plan(cfg_on, f, channel).words
        table.append((w[1], w[2], w[0] & ~autocal, w[0] | autocal))

    r6_on = plan(cfg_off if args.mute else cfg_on,
                 freqs[0], channel).registers.word(6)
    r6_off = plan(cfg_off, freqs[0], channel).registers.word(6)

    seq = [point for point, _dwell in period]
    dwell_ns = int(round(args.hop_ms * 1e6))

    # The C side must be able to bring the part up on its own: without a cold
    # start with autocal at one of these frequencies the VCO keeps whatever band
    # the previous command calibrated, and writing dividers alone never locks.
    boot = plan(cfg_on, freqs[0], channel)
    boot_words = boot.words                      # R0..R12
    delay_us = boot.delay_us

    with open(args.out, "wb") as fh:
        fh.write(struct.pack("<IIIQQII", MAGIC, len(freqs), len(seq),
                             total_hops, dwell_ns, r6_on, r6_off))
        fh.write(struct.pack("<II", delay_us, 1 if args.autocal_every else 0))
        for w in boot_words:                     # 13 words, R0 first
            fh.write(struct.pack("<I", w))
        for r1, r2, r0, r0cal in table:
            fh.write(struct.pack("<IIII", r1, r2, r0, r0cal))
        for s in seq:
            fh.write(struct.pack("<H", s))

    span = "forever" if total_hops == 0 else f"{total_hops*args.hop_ms/1000:.2f} s"
    print(f"{args.out}: cold start {len(boot_words)} regs, "
          f"autocal settle {delay_us} us, "
          f"{len(freqs)} points, period {len(seq)} hops "
          f"({len(seq)*args.hop_ms/1000:.4f} s), plays {span}, "
          f"{args.hop_ms:g} ms dwell, "
          f"{'autocal every hop' if args.autocal_every else 'no per-hop autocal'}, "
          f"R6 on 0x{r6_on:08X} off 0x{r6_off:08X}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
