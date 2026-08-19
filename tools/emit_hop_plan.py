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
from adf5355.hopper import make_schedule, plan_frequencies
from adf5355.registers import FIELDS

MAGIC = 0x34354441  # "AD54" little endian -- v2 carries the cold start


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=lambda v: int(v, 0), default=0xC0FFEE)
    p.add_argument("--start-ghz", type=float, default=11.0)
    p.add_argument("--stop-ghz", type=float, default=11.00171)
    p.add_argument("--points", type=int, default=20)
    p.add_argument("--hop-ms", type=float, default=10.0)
    p.add_argument("--cycles", type=int, default=100)
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
    hops = make_schedule(args.seed, freqs, args.hop_ms / 1e3, args.cycles)

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
        table.append((w[1], w[2], w[0] & ~autocal))

    r6_on = plan(cfg_off if args.mute else cfg_on,
                 freqs[0], channel).registers.word(6)
    r6_off = plan(cfg_off, freqs[0], channel).registers.word(6)

    index = {f: i for i, f in enumerate(freqs)}
    seq = [index[h.freq_hz] for h in hops]
    dwell_ns = int(round(args.hop_ms * 1e6))

    # The C side must be able to bring the part up on its own: without a cold
    # start with autocal at one of these frequencies the VCO keeps whatever band
    # the previous command calibrated, and writing dividers alone never locks.
    boot = plan(cfg_on, freqs[0], channel)
    boot_words = boot.words                      # R0..R12
    delay_us = boot.delay_us

    with open(args.out, "wb") as fh:
        fh.write(struct.pack("<IIIQII", MAGIC, len(freqs), len(seq),
                             dwell_ns, r6_on, r6_off))
        fh.write(struct.pack("<I", delay_us))
        for w in boot_words:                     # 13 words, R0 first
            fh.write(struct.pack("<I", w))
        for r1, r2, r0 in table:
            fh.write(struct.pack("<III", r1, r2, r0))
        for s in seq:
            fh.write(struct.pack("<H", s))

    print(f"{args.out}: cold start {len(boot_words)} regs, "
          f"autocal settle {delay_us} us, "
          f"{len(freqs)} points, {len(seq)} hops, "
          f"{args.hop_ms:g} ms dwell, {len(seq)*args.hop_ms/1000:.2f} s, "
          f"R6 on 0x{r6_on:08X} off 0x{r6_off:08X}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
