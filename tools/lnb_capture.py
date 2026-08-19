#!/usr/bin/env python3
"""Capture an ADF5355 ladder through an LNB and recover the receiver's errors.

Chain:  ADF5355 (125 MHz ref)  ->  LNB (own free-running LO)  ->  PlutoSDR

With f_IF = f_RF - f_LO, and writing d for a reference's fractional error, the
frequency the Pluto reports for a rung comes out as

    reported  ~=  (f_RF - f_LO_nom*(1 + d_lnb)) * (1 - d_rx)

so the discrepancy against the nominal IF separates into two terms:

    Df  =  reported - f_IF_nom  ~=  -d_rx * f_IF_nom  -  d_lnb * f_LO_nom
                                    \_______________/    \_______________/
                                     scales with IF        constant

A straight-line fit of Df against f_IF_nom therefore yields the Pluto's clock
error as the SLOPE and the LNB's LO error as the INTERCEPT.  One tone cannot
separate them; the ladder can, which is the whole point of stepping frequency.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def capture(sdr, centre_hz: float, n: int) -> np.ndarray:
    sdr.rx_lo = int(centre_hz)
    sdr.rx_destroy_buffer()
    sdr.rx_buffer_size = n
    for _ in range(2):          # discard the first buffer after retuning
        data = sdr.rx()
    return np.asarray(data)


def tone_estimate(x: np.ndarray, fs: float, centre_hz: float):
    """Interpolated FFT peak.  Returns (absolute_hz, snr_db, power_db)."""
    w = np.hanning(len(x))
    spec = np.fft.fftshift(np.fft.fft(x * w))
    mag = np.abs(spec)
    freqs = np.fft.fftshift(np.fft.fftfreq(len(x), 1.0 / fs))
    k = int(np.argmax(mag))
    if k in (0, len(mag) - 1):
        return None, 0.0, -999.0

    # Quadratic interpolation on the log-magnitude peak.
    a, b, c = (np.log(mag[k - 1] + 1e-30), np.log(mag[k] + 1e-30),
               np.log(mag[k + 1] + 1e-30))
    delta = 0.5 * (a - c) / (a - 2 * b + c)
    bin_hz = freqs[1] - freqs[0]
    baseband = freqs[k] + delta * bin_hz

    power_db = 20 * np.log10(mag[k] + 1e-30)
    noise = np.median(mag)
    snr_db = 20 * np.log10((mag[k] + 1e-30) / (noise + 1e-30))
    return centre_hz + baseband, snr_db, power_db


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--uri", default="ip:192.168.2.1")
    p.add_argument("--centre", type=float, required=True, help="IF centre in Hz")
    p.add_argument("--fs", type=float, default=4e6, help="sample rate in Hz")
    p.add_argument("--n", type=int, default=1 << 18, help="samples per capture")
    p.add_argument("--gain", default="slow_attack")
    p.add_argument("--repeat", type=int, default=1)
    args = p.parse_args()

    import adi
    sdr = adi.Pluto(uri=args.uri)
    sdr.sample_rate = int(args.fs)
    sdr.rx_rf_bandwidth = int(args.fs * 0.8)
    sdr.gain_control_mode_chan0 = args.gain

    print(f"uri={args.uri}  centre={args.centre/1e6:.3f} MHz  fs={args.fs/1e6:g} MS/s  "
          f"n={args.n}  ({args.n/args.fs*1e3:.1f} ms, {args.fs/args.n:.2f} Hz/bin)")
    for i in range(args.repeat):
        x = capture(sdr, args.centre, args.n)
        f, snr, pwr = tone_estimate(x, args.fs, args.centre)
        rms = float(np.sqrt(np.mean(np.abs(x) ** 2)))
        if f is None:
            print(f"  [{i}] no usable peak; rms={rms:.1f}")
        else:
            print(f"  [{i}] peak {f/1e6:12.6f} MHz  snr {snr:5.1f} dB  "
                  f"offset {f - args.centre:+10.1f} Hz  rms {rms:7.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
