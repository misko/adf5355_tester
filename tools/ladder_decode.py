#!/usr/bin/env python3
"""Decode a duration-coded ladder: identify each point by how long it lasts.

The fixed-dwell decoder in hop_decode.py must first find the comb offset, then
align the capture against a seeded schedule, before it can attribute any dwell
to any point. Both stages can fail while the signal is plainly present. Here
identity is carried by duration, so the receiver needs none of that: segment
the capture into bursts, measure each burst's length, and the length names the
point.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def segments(iq: np.ndarray, fs: float, frame: int, lo_hz: float, hi_hz: float,
             thr_db: float, tol_hz: float):
    """Frames -> (start_s, dur_s, freq_hz, snr_db) for each contiguous burst."""
    n = len(iq) // frame
    fb = np.fft.fftfreq(frame, 1.0 / fs)
    keep = (fb > lo_hz) & (fb < hi_hz)
    win = np.hanning(frame)
    pk = np.full(n, np.nan)
    snr = np.zeros(n)
    for i in range(n):
        sp = np.abs(np.fft.fft(iq[i*frame:(i+1)*frame] * win)) ** 2
        band = np.where(keep, sp, 0.0)
        k = int(np.argmax(band))
        noise = np.median(sp[keep])
        if noise > 0 and 10*np.log10(band[k]/noise) >= thr_db:
            # parabolic refine against the two neighbouring bins
            a, b, c = (np.log(sp[k-1] + 1e-30), np.log(sp[k] + 1e-30),
                       np.log(sp[(k+1) % frame] + 1e-30))
            d = 0.5*(a - c)/(a - 2*b + c) if (a - 2*b + c) != 0 else 0.0
            pk[i] = fb[k] + d*(fs/frame)
            snr[i] = 10*np.log10(band[k]/noise)
    out, i = [], 0
    while i < n:
        if np.isnan(pk[i]):
            i += 1
            continue
        j = i
        while j+1 < n and not np.isnan(pk[j+1]) and abs(pk[j+1] - pk[i]) <= tol_hz:
            j += 1
        out.append((i*frame/fs, (j-i+1)*frame/fs,
                    float(np.median(pk[i:j+1])), float(np.mean(snr[i:j+1]))))
        i = j + 1
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--capture", required=True)
    p.add_argument("--fs", type=float, default=2.5e6)
    p.add_argument("--tune-hz", type=float, required=True,
                   help="the rx_lo the capture was taken at")
    p.add_argument("--freqs", required=True, help="nominal RF, comma separated Hz")
    p.add_argument("--dwells-ms", required=True)
    p.add_argument("--lnb-lo", type=float, default=9.75e9)
    p.add_argument("--frame", type=int, default=2048)
    p.add_argument("--threshold-db", type=float, default=10.0)
    p.add_argument("--tol-khz", type=float, default=40.0)
    p.add_argument("--band-khz", default="150,1200",
                   help="baseband search window, lo,hi in kHz")
    a = p.parse_args()

    freqs = [float(t) for t in a.freqs.split(",")]
    dwells = [float(t)/1e3 for t in a.dwells_ms.split(",")]
    lo_khz, hi_khz = (float(v) for v in a.band_khz.split(","))

    raw = np.fromfile(a.capture, dtype="<i2")
    iq = raw[0::2].astype(np.float32) + 1j*raw[1::2].astype(np.float32)
    secs = len(iq)/a.fs
    segs = segments(iq, a.fs, a.frame, lo_khz*1e3, hi_khz*1e3,
                    a.threshold_db, a.tol_khz*1e3)
    res = a.frame/a.fs
    print(f"\n  capture   : {secs:.3f} s at {a.fs/1e6:g} MS/s, "
          f"frame {a.frame} = {res*1e3:.3f} ms (duration resolution)")
    print(f"  bursts    : {len(segs)} found, cycle is {sum(dwells)*1e3:g} ms "
          f"({secs/sum(dwells):.1f} cycles in the capture)")

    # classify each burst by nearest nominal duration
    print(f"\n  {'start s':>8} {'dur ms':>8} {'-> point':>9} {'baseband kHz':>13} {'SNR':>7}")
    buckets: dict[int, list] = {i: [] for i in range(len(dwells))}
    for st, du, fr, sn in segs:
        i = int(np.argmin([abs(du-d) for d in dwells]))
        # only accept if it is within 35% of that nominal duration
        if abs(du-dwells[i]) <= 0.35*dwells[i]:
            buckets[i].append((fr, du, sn))
            tag = f"P{i}"
        else:
            tag = "--"
        if st < 0.6:
            print(f"  {st:>8.3f} {du*1e3:>8.2f} {tag:>9} {fr/1e3:>13.1f} {sn:>6.1f}dB")

    print(f"\n  {'point':>5} {'nominal RF GHz':>15} {'held ms':>8} {'seen':>5} "
          f"{'measured IF MHz':>16} {'nominal IF MHz':>15} {'error kHz':>11}")
    errs = []
    for i, (f_rf, d) in enumerate(zip(freqs, dwells)):
        b = buckets[i]
        nom_if = f_rf - a.lnb_lo
        if not b:
            print(f"  {i:>5} {f_rf/1e9:>15.6f} {d*1e3:>8.2f} {0:>5} "
                  f"{'--':>16} {nom_if/1e6:>15.6f} {'--':>11}")
            continue
        meas_if = a.tune_hz + float(np.median([x[0] for x in b]))
        err = meas_if - nom_if
        errs.append((nom_if, err))
        print(f"  {i:>5} {f_rf/1e9:>15.6f} {d*1e3:>8.2f} {len(b):>5} "
              f"{meas_if/1e6:>16.6f} {nom_if/1e6:>15.6f} {err/1e3:>+11.3f}")

    if len(errs) >= 2:
        x = np.array([e[0] for e in errs]); y = np.array([e[1] for e in errs])
        slope, intercept = np.polyfit(x, y, 1)
        resid = y - (slope*x + intercept)
        print(f"\n  common offset : {np.mean(y)/1e3:+.3f} kHz "
              f"(all points shifted together -- LNB LO error plus receiver clock)")
        print(f"  slope         : {slope*1e6:+.3f} ppm across "
              f"{(x.max()-x.min())/1e3:.0f} kHz of lever arm")
        print(f"  residual      : {np.std(resid):.1f} Hz rms about the fit")
        print(f"\n  NOTE: {(x.max()-x.min())/1e3:.0f} kHz of lever arm is far too short to "
              f"separate\n        receiver-clock error from LNB LO error -- the slope above is "
              f"not\n        a usable ppm figure. This run demonstrates identification, "
              f"not calibration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
