#!/usr/bin/env python3
"""Blind ladder calibration: listen, then work out which rung was heard.

No control channel to the transmitter.  The receiver knows only the published
ladder parameters (range, rung count, total interval) -- never when a rung keys
or which one is currently up.  It listens at one tuning for longer than a full
coded interval, finds the burst, and divides its length by u to recover the rung
number, and hence that rung's frequency.  That identification is then checked
against the rung the tuning was aimed at, which is an end-to-end test of the
duration coding itself.

Streaming: buffers are reduced to (time, snr, peak frequency) as they arrive, so
a 20+ second listen costs kilobytes rather than gigabytes.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import numpy as np

ADF = "/home/pi/.local/bin/adf5355"


def buffer_summary(x, fs, centre, dc_notch_hz=20e3):
    x = x - np.mean(x)
    mag = np.abs(np.fft.fftshift(np.fft.fft(x * np.hanning(len(x)))))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(x), 1.0 / fs))
    mag[np.abs(freqs) < dc_notch_hz] = 0.0
    k = int(np.argmax(mag))
    med = float(np.median(mag)) + 1e-30
    snr = 20.0 * np.log10((mag[k] + 1e-30) / med)
    if 0 < k < len(mag) - 1:
        a, b, c = (np.log(mag[k-1]+1e-30), np.log(mag[k]+1e-30), np.log(mag[k+1]+1e-30))
        d = 0.5 * (a - c) / (a - 2*b + c)
    else:
        d = 0.0
    return snr, centre + freqs[k] + d * (freqs[1] - freqs[0])


def listen(sdr, centre, fs, nbuf, seconds):
    """Stream, reducing each buffer to (t, snr, f). Returns three arrays."""
    sdr.rx_lo = int(centre)
    sdr.rx_destroy_buffer()
    sdr.rx_buffer_size = nbuf
    sdr.rx()                                    # discard the retune transient
    ts, snrs, fss = [], [], []
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        x = np.asarray(sdr.rx())
        t = time.monotonic() - t0               # wall clock, so gaps do not lie
        snr, f = buffer_summary(x, fs, centre)
        ts.append(t); snrs.append(snr); fss.append(f)
    return np.array(ts), np.array(snrs), np.array(fss)


def find_bursts(ts, snrs, threshold_db):
    on = snrs > threshold_db
    bursts, start = [], None
    for i, flag in enumerate(on):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            bursts.append((start, i - 1)); start = None
    if start is not None:
        bursts.append((start, len(on) - 1))
    out = []
    for a, b in bursts:
        # Edge buffers straddle the transition; the true edge is within one
        # buffer either side, so bound the duration by the midpoints.
        lo = ts[a-1] if a > 0 else ts[a]
        hi = ts[b+1] if b < len(ts)-1 else ts[b]
        dur = 0.5 * ((ts[b] - ts[a]) + (hi - lo))
        out.append({"i0": a, "i1": b, "t0": ts[a], "dur": dur,
                    "n": b - a + 1})
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start-ghz", type=float, default=10.7)
    p.add_argument("--stop-ghz", type=float, default=11.9)
    p.add_argument("--steps", type=int, default=7)
    p.add_argument("--total-s", type=float, default=14.0)
    p.add_argument("--lo-ghz", type=float, default=9.75)
    p.add_argument("--lo-guess-hz", type=float, default=9.750094e9)
    p.add_argument("--fs", type=float, default=2.5e6)
    p.add_argument("--nbuf", type=int, default=1 << 15)
    p.add_argument("--tone-offset", type=float, default=400e3)
    p.add_argument("--threshold-db", type=float, default=35.0)
    p.add_argument("--uri", default="ip:192.168.2.1")
    args = p.parse_args()

    sys.path.insert(0, "/home/pi/adf5355_tester")
    from adf5355.ladder import make_ladder

    steps = make_ladder(round(args.start_ghz*1e9), round(args.stop_ghz*1e9),
                        args.steps, args.total_s)
    u = args.total_s / (args.steps * (args.steps + 1))
    f_lo_nom = args.lo_ghz * 1e9
    listen_s = args.total_s + 2 * u * args.steps + 2.0

    import adi
    sdr = adi.Pluto(uri=args.uri)
    sdr.sample_rate = int(args.fs)
    sdr.rx_rf_bandwidth = int(args.fs * 0.8)
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = 40

    print(f"published ladder: {args.steps} rungs, {args.start_ghz}-{args.stop_ghz} GHz, "
          f"{args.total_s:g} s  ->  u = {u:.4f} s")
    print(f"buffer {args.nbuf} @ {args.fs/1e6:g} MS/s = {args.nbuf/args.fs*1e3:.1f} ms "
          f"time resolution; listening {listen_s:.1f} s per tuning\n")

    # Loop the ladder continuously for the whole session; the receiver never
    # learns anything about its phase.
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    loops = int(np.ceil((listen_s * args.steps + 30) / args.total_s)) + 1
    tx = subprocess.Popen(
        [ADF, "ladder", "--start-ghz", str(args.start_ghz),
         "--stop-ghz", str(args.stop_ghz), "--steps", str(args.steps),
         "--total-s", str(args.total_s), "--loops", str(loops),
         "--power", "0", "--enable-rf"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    time.sleep(3.0)

    rows, correct = [], 0
    try:
        for aimed in steps:
            centre = aimed.freq_hz - args.lo_guess_hz - args.tone_offset
            ts, snrs, fss = listen(sdr, centre, args.fs, args.nbuf, listen_s)
            bursts = find_bursts(ts, snrs, args.threshold_db)
            # Keep the longest fully-interior burst.
            interior = [b for b in bursts if b["i0"] > 0 and b["i1"] < len(ts)-1]
            if not interior:
                print(f"  aimed at rung {aimed.index}: no complete burst heard")
                continue
            b = max(interior, key=lambda z: z["dur"])
            n_est = b["dur"] / u
            n = int(round(n_est))
            n = min(max(n, 1), args.steps)
            heard = steps[n-1]
            f_meas = float(np.median(fss[b["i0"]:b["i1"]+1]))
            f_if_nom = heard.freq_hz - f_lo_nom
            df = f_meas - f_if_nom
            ok = (n == aimed.index)
            correct += ok
            rows.append((n, heard.freq_hz, f_if_nom, f_meas, df,
                         float(np.median(snrs[b["i0"]:b["i1"]+1])), b["dur"]))
            print(f"  heard burst {b['dur']:.3f} s -> {b['dur']:.3f}/{u:.4f} = "
                  f"{n_est:.2f} -> RUNG {n}  ({'ok' if ok else 'MISMATCH, aimed at '+str(aimed.index)})"
                  f"  = {heard.freq_hz/1e9:.3f} GHz   Df {df/1e3:+8.3f} kHz")
    finally:
        try:
            tx.terminate(); tx.wait(timeout=10)
        except Exception:
            pass
        subprocess.run([ADF, "off"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f"\nrung identification from burst length alone: {correct}/{len(steps)} correct")
    if len(rows) < 3:
        print("not enough rungs heard to fit")
        return 1

    f_if = np.array([r[2] for r in rows], float)
    df = np.array([r[4] for r in rows], float)
    distinct = len(set(np.round(f_if).tolist()))
    if distinct < 3:
        print(f"only {distinct} distinct rung frequency(ies) identified -- cannot "
              f"separate slope from intercept. Need >= 3.")
        return 1
    fc, fsd = f_if.mean(), f_if.std()
    A = np.column_stack([(f_if - fc)/fsd, np.ones_like(f_if)])
    coef, *_ = np.linalg.lstsq(A, df, rcond=None)
    res = df - A @ coef
    a = coef[0]/fsd
    c = coef[1] - a*fc
    print("\n" + "="*64)
    print(f"  Pluto clock error : {-a*1e6:+.3f} ppm")
    print(f"  LNB LO error      : {-c/1e3:+.3f} kHz ({-c/f_lo_nom*1e6:+.3f} ppm)")
    print(f"  residual rms      : {np.sqrt(np.mean(res**2)):.0f} Hz")
    print("="*64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
