#!/usr/bin/env python3
"""Emit a duration-coded ladder plan for tools/dwell_ladder (the C transmitter).

Each point carries its own dwell, so identity is encoded in duration. The
register maths stays in Python for the same reason as the fixed-dwell plan:
one implementation of the protocol, so the two transmitters cannot drift.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adf5355 import Channel, SynthConfig, plan
from adf5355.registers import FIELDS

MAGIC = 0x37354441  # "AD57" little endian


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--freqs", required=True,
                   help="comma-separated frequencies in Hz or with G/M suffix")
    p.add_argument("--dwells-ms", required=True,
                   help="comma-separated dwell per frequency, milliseconds")
    p.add_argument("--ref-mhz", type=float, default=125.0)
    p.add_argument("--power", type=int, default=3)
    p.add_argument("--channel", default="B", choices=["A", "B"])
    p.add_argument("--autocal-every", action="store_true")
    a = p.parse_args()

    def as_hz(tok: str) -> int:
        tok = tok.strip()
        mult = {"G": 1e9, "M": 1e6, "k": 1e3}.get(tok[-1:], None)
        return round(float(tok[:-1]) * mult) if mult else round(float(tok))

    freqs = [as_hz(t) for t in a.freqs.split(",")]
    dwells = [float(t) for t in a.dwells_ms.split(",")]
    if len(freqs) != len(dwells):
        p.error(f"{len(freqs)} frequencies but {len(dwells)} dwells")

    channel = Channel(a.channel)
    cfg_on = SynthConfig(ref_hz=round(a.ref_mhz * 1e6),
                         outa_enable=channel is Channel.A,
                         outb_enable=channel is Channel.B,
                         outa_power=a.power, mute_till_lock=False)
    cfg_off = SynthConfig(ref_hz=round(a.ref_mhz * 1e6),
                          outa_enable=False, outb_enable=False,
                          outa_power=a.power, mute_till_lock=False)

    autocal = FIELDS["autocal"].mask
    rows = []
    for f in freqs:
        w = plan(cfg_on, f, channel).words
        rows.append((w[1], w[2], w[0] & ~autocal, w[0] | autocal))

    r6_on = plan(cfg_on, freqs[0], channel).registers.word(6)
    r6_off = plan(cfg_off, freqs[0], channel).registers.word(6)
    boot = plan(cfg_on, freqs[0], channel)

    with open(a.out, "wb") as fh:
        fh.write(struct.pack("<IIIIII", MAGIC, len(freqs), r6_on, r6_off,
                             boot.delay_us, 1 if a.autocal_every else 0))
        for word in boot.words:
            fh.write(struct.pack("<I", word))
        for (r1, r2, r0, r0c), d in zip(rows, dwells):
            fh.write(struct.pack("<IIIIQ", r1, r2, r0, r0c,
                                 int(round(d * 1e6))))

    total = sum(dwells)
    print(f"{a.out}: {len(freqs)} points, cycle {total:g} ms, loops forever")
    for f, d in zip(freqs, dwells):
        print(f"    {f/1e9:.6f} GHz  hold {d:g} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
