#!/usr/bin/env python3
"""Find the ADF5355 tone in the LNB's IF by differencing TX-on against TX-off.

The LNB output already contains spurs and possibly live satellite carriers, so
the strongest peak is not necessarily ours.  Capturing the same centre with the
synthesiser muted and then transmitting, and subtracting the two spectra,
leaves only what we put there.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

import numpy as np

ADF = "/home/pi/.local/bin/adf5355"


def grab(sdr, centre, fs, n, avg=4):
    sdr.rx_lo = int(centre)
    sdr.rx_destroy_buffer()
    sdr.rx_buffer_size = n
    sdr.rx()                                   # discard after retune
    acc = None
    for _ in range(avg):
        x = np.asarray(sdr.rx())
        m = np.abs(np.fft.fftshift(np.fft.fft(x * np.hanning(len(x)))))
        acc = m if acc is None else acc + m
    return acc / avg


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rf-ghz", type=float, default=11.0)
    p.add_argument("--lo-ghz", type=float, default=9.75)
    p.add_argument("--span-mhz", type=float, default=90.0,
                   help="total IF window to search around the predicted IF")
    p.add_argument("--fs", type=float, default=40e6)
    p.add_argument("--n", type=int, default=1 << 16)
    p.add_argument("--uri", default="ip:192.168.2.1")
    args = p.parse_args()

    import adi
    sdr = adi.Pluto(uri=args.uri)
    sdr.sample_rate = int(args.fs)
    sdr.rx_rf_bandwidth = int(args.fs * 0.8)
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = 40

    usable = args.fs * 0.7
    predicted = (args.rf_ghz - args.lo_ghz) * 1e9
    centres = np.arange(predicted - args.span_mhz * 1e6 / 2,
                        predicted + args.span_mhz * 1e6 / 2 + 1, usable)
    print(f"RF {args.rf_ghz} GHz, assumed LO {args.lo_ghz} GHz "
          f"-> predicted IF {predicted/1e6:.1f} MHz")
    print(f"searching {len(centres)} window(s) of {usable/1e6:.1f} MHz")

    print("\ncapturing TX OFF baseline...")
    off = {c: grab(sdr, c, args.fs, args.n) for c in centres}

    print(f"enabling CW at {args.rf_ghz} GHz (RFoutB, minimum power)...")
    tx = subprocess.Popen([ADF, "dwell", "--freq", f"{args.rf_ghz}G",
                           "--channel", "B", "--dwell", "60",
                           "--power", "0", "--enable-rf"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3.0)

    best = None
    try:
        freqs_base = np.fft.fftshift(np.fft.fftfreq(args.n, 1.0 / args.fs))
        for c in centres:
            on = grab(sdr, c, args.fs, args.n)
            delta_db = 20 * np.log10((on + 1e-30) / (off[c] + 1e-30))
            edge = int(len(delta_db) * 0.12)          # ignore band edges
            core = slice(edge, len(delta_db) - edge)
            k = int(np.argmax(delta_db[core])) + edge
            rise = float(delta_db[k])
            f_abs = c + freqs_base[k]
            print(f"  centre {c/1e6:9.3f} MHz -> best rise {rise:5.1f} dB "
                  f"at {f_abs/1e6:12.6f} MHz")
            if best is None or rise > best[0]:
                best = (rise, f_abs, c)
    finally:
        tx.terminate(); tx.wait(timeout=10)
        subprocess.run([ADF, "off"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)

    rise, f_abs, c = best
    print(f"\nstrongest TX-attributable peak: {f_abs/1e6:.6f} MHz, +{rise:.1f} dB")
    if rise < 6:
        print("NOT FOUND -- no window rose meaningfully with TX on.")
        return 1
    implied_lo = args.rf_ghz * 1e9 - f_abs
    print(f"implied LNB LO = {args.rf_ghz} GHz - {f_abs/1e6:.6f} MHz "
          f"= {implied_lo/1e9:.9f} GHz")
    print(f"  vs nominal {args.lo_ghz} GHz -> error "
          f"{(implied_lo - args.lo_ghz*1e9)/1e3:+.3f} kHz "
          f"({(implied_lo/(args.lo_ghz*1e9) - 1)*1e6:+.2f} ppm)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
