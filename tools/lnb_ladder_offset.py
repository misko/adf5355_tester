#!/usr/bin/env python3
"""Measure a PlutoSDR's clock error and an LNB's LO error from one ladder run.

    ADF5355 (125 MHz ref)  ->  LNB (free-running LO)  ->  PlutoSDR

For each rung, with d the fractional error of a reference:

    reported ~= (f_RF - f_LO_nom*(1 + d_lnb)) * (1 - d_rx)
    Df = reported - f_IF_nom ~= -d_rx * f_IF_nom - d_lnb * f_LO_nom

A straight line fitted to Df against f_IF_nom gives

    slope     = -d_rx                 -> the Pluto's clock error
    intercept = -d_lnb * f_LO_nom     -> the LNB's LO error

Rungs are identified the way any receiver would: the ladder announces each rung
as it keys, and the burst lengths are distinct, so no shared time base is
needed.  Retuning happens inside the previous rung's OFF window.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

import numpy as np

ADF = "/home/pi/.local/bin/adf5355"
RUNG_RE = re.compile(r"rung (\d+): ([\d.]+) GHz ON ([\d.]+) s")


def estimate_tone(x, fs, centre, dc_notch_hz=20e3):
    x = x - np.mean(x)                       # kill the DC term outright
    w = np.hanning(len(x))
    mag = np.abs(np.fft.fftshift(np.fft.fft(x * w)))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(x), 1.0 / fs))
    mag[np.abs(freqs) < dc_notch_hz] = 0.0   # and ignore the LO-leakage spur
    k = int(np.argmax(mag))
    if k == 0 or k == len(mag) - 1:
        return None, 0.0
    a, b, c = (np.log(mag[k - 1] + 1e-30), np.log(mag[k] + 1e-30),
               np.log(mag[k + 1] + 1e-30))
    delta = 0.5 * (a - c) / (a - 2 * b + c)
    bin_hz = freqs[1] - freqs[0]
    snr = 20 * np.log10((mag[k] + 1e-30) / (np.median(mag) + 1e-30))
    return centre + freqs[k] + delta * bin_hz, snr


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start-ghz", type=float, default=10.7)
    p.add_argument("--stop-ghz", type=float, default=11.7)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--total-s", type=float, default=42.0)
    p.add_argument("--lo-ghz", type=float, default=9.75, help="NOMINAL LNB LO")
    p.add_argument("--lo-guess-hz", type=float, default=9.750104736e9,
                   help="best estimate, used only to centre the receiver")
    p.add_argument("--fs", type=float, default=4e6)
    p.add_argument("--n", type=int, default=1 << 18)
    p.add_argument("--uri", default="ip:192.168.2.1")
    p.add_argument("--csv", default=None, help="write raw measurements here")
    p.add_argument("--tone-offset", type=float, default=500e3,
                   help="park the tone this far from baseband DC (Hz). The "
                        "Pluto has an LO-leakage spur at DC that pulls the "
                        "interpolated peak when a tone sits near it.")
    p.add_argument("--loops", type=int, default=1,
                   help="ladder repeats; >1 lets drift be fitted out")
    args = p.parse_args()

    sys.path.insert(0, "/home/pi/adf5355_tester")
    from adf5355.ladder import make_ladder

    steps = make_ladder(round(args.start_ghz * 1e9), round(args.stop_ghz * 1e9),
                        args.steps, args.total_s)
    f_lo_nom = args.lo_ghz * 1e9

    import adi
    sdr = adi.Pluto(uri=args.uri)
    sdr.sample_rate = int(args.fs)
    sdr.rx_rf_bandwidth = int(args.fs * 0.8)
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = 40

    print(f"ladder: {args.steps} rungs {args.start_ghz}-{args.stop_ghz} GHz "
          f"over {args.total_s:g} s (u = {args.total_s/(args.steps*(args.steps+1)):.3f} s)")
    print(f"nominal LNB LO {args.lo_ghz} GHz; {args.fs/args.n:.2f} Hz/bin\n")

    env = dict(os.environ, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(
        [ADF, "ladder", "--start-ghz", str(args.start_ghz),
         "--stop-ghz", str(args.stop_ghz), "--steps", str(args.steps),
         "--total-s", str(args.total_s), "--loops", str(args.loops),
         "--power", "0", "--enable-rf"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env)

    results = []
    try:
        # Pre-tune to rung 1 so its (shortest) window is not spent retuning.
        sdr.rx_lo = int(steps[0].freq_hz - args.lo_guess_hz
                        - args.tone_offset)
        sdr.rx_buffer_size = args.n

        for line in proc.stdout:
            m = RUNG_RE.search(line)
            if not m:
                continue
            n = int(m.group(1))
            step = steps[n - 1]
            f_if_nom = step.freq_hz - f_lo_nom

            sdr.rx_destroy_buffer()
            sdr.rx_buffer_size = args.n
            sdr.rx()                                   # flush
            x = np.asarray(sdr.rx())
            centre = float(sdr.rx_lo)
            f_meas, snr = estimate_tone(x, args.fs, centre)

            t_now = time.monotonic()
            if f_meas is None:
                print(f"  rung {n}: no peak")
            else:
                df = f_meas - f_if_nom
                results.append((n, step.freq_hz, f_if_nom, f_meas, df, snr,
                                step.on_s, t_now))
                print(f"  rung {n} ({step.on_s:.1f} s burst)  RF {step.freq_hz/1e9:7.3f} GHz"
                      f"  IF_nom {f_if_nom/1e6:9.3f}  meas {f_meas/1e6:12.6f} MHz"
                      f"  Df {df/1e3:+9.3f} kHz  snr {snr:4.1f} dB")

            # Retune for the next rung during this rung's OFF window.
            nxt = steps[n % len(steps)]           # wraps for the next loop
            sdr.rx_lo = int(nxt.freq_hz - args.lo_guess_hz
                            - args.tone_offset)
        proc.wait(timeout=20)
    finally:
        try:
            proc.terminate(); proc.wait(timeout=10)
        except Exception:
            pass
        subprocess.run([ADF, "off"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)

    if args.csv:
        with open(args.csv, "w") as fh:
            fh.write("rung,f_rf_hz,f_if_nom_hz,f_meas_hz,df_hz,snr_db,on_s,t_s\n")
            for r in results:
                fh.write(",".join(f"{float(v):.6f}" for v in r) + "\n")
        print(f"\nraw measurements -> {args.csv}")

    good = [r for r in results if r[5] > 15]
    print(f"\n{len(good)}/{len(steps)*args.loops} measurements usable (SNR > 15 dB)")
    if len(good) < 4:
        print("need at least 4 measurements")
        return 1

    f_if = np.array([r[2] for r in good], dtype=float)
    df = np.array([r[4] for r in good], dtype=float)
    t = np.array([r[7] for r in good], dtype=float)
    t = t - t.mean()

    def fit(A, y):
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        dof = max(1, len(y) - A.shape[1])
        s2 = float(resid @ resid) / dof
        cov = s2 * np.linalg.pinv(A.T @ A)
        return coef, np.sqrt(np.diag(cov)), resid

    f_lo_nom_local = f_lo_nom
    # Model 1: frequency only (what a single monotonic pass can fit).
    A1 = np.column_stack([f_if, np.ones_like(f_if)])
    c1, e1, r1 = fit(A1, df)
    # Model 2: frequency + linear time drift.
    A2 = np.column_stack([f_if, t, np.ones_like(f_if)])
    c2, e2, r2 = fit(A2, df)

    print("\n" + "=" * 72)
    print("Model 1  Df = a*f_IF + c            (drift confounded with slope)")
    print(f"  a = {c1[0]:+.4e} +/- {e1[0]:.1e}   -> Pluto {-c1[0]*1e6:+.3f} ppm")
    print(f"  c = {c1[1]/1e3:+.3f} +/- {e1[1]/1e3:.3f} kHz -> LNB LO "
          f"{-c1[1]/1e3:+.3f} kHz")
    print(f"  residual rms {np.sqrt(np.mean(r1**2)):.1f} Hz")
    print("\nModel 2  Df = a*f_IF + b*t + c      (drift fitted out)")
    print(f"  a = {c2[0]:+.4e} +/- {e2[0]:.1e}   -> Pluto {-c2[0]*1e6:+.3f} "
          f"+/- {e2[0]*1e6:.3f} ppm")
    print(f"  b = {c2[1]:+.2f} +/- {e2[1]:.2f} Hz/s   (LNB/system drift)")
    print(f"  c = {c2[2]/1e3:+.3f} +/- {e2[2]/1e3:.3f} kHz -> LNB LO "
          f"{-c2[2]/1e3:+.3f} +/- {e2[2]/1e3:.3f} kHz "
          f"({-c2[2]/f_lo_nom_local*1e6:+.3f} ppm)")
    print(f"  residual rms {np.sqrt(np.mean(r2**2)):.1f} Hz")
    print("=" * 72)

    d_rx, d_lnb = -c2[0], -c2[2] / f_lo_nom_local
    print(f"\nPLUTO REFERENCE: {abs(d_rx)*1e6:.3f} +/- {e2[0]*1e6:.3f} ppm "
          f"{'HIGH' if d_rx > 0 else 'LOW'}")
    print(f"  xo_correction 40000000 -> {40e6*(1+d_rx):.1f} Hz  "
          f"(re-measure afterwards to confirm the sign)")
    print(f"LNB LO: {f_lo_nom_local*(1+d_lnb)/1e9:.9f} GHz "
          f"({d_lnb*1e6:+.2f} ppm, {-c2[2]/1e3:+.1f} kHz)")
    print(f"DRIFT: {c2[1]:+.1f} Hz/s over {t.max()-t.min():.0f} s observed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
