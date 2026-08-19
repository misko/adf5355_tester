#!/usr/bin/env python3
"""Count transmission blips in a capture, without knowing the schedule.

The decoder in hop_decode.py is the wrong instrument for this question: it is
told the full comb up front and reports confidence only when it recovers every
point. Here the receiver is deliberately ignorant -- the transmitter sweeps a
span far wider than the window, so any one capture hears an arbitrary sparse
subset, and the quantity of interest is simply how many bursts arrived.

The measurement has to survive being run at different sample rates and still be
comparable, which forces two choices:

*   Frames are a fixed duration in TIME, not a fixed number of samples. Bin
    width is then the same in Hz at every rate, so a tone lands in one bin
    regardless of fs and the detector is equally sensitive across the sweep.
*   The threshold is on peak-to-noise within each frame's own spectrum, never
    an absolute level. A wider window admits proportionally more total noise;
    thresholding on absolute power would make the wide captures look worse and
    would confound exactly the comparison being made.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np


@dataclass
class BlipStats:
    bursts: int
    frames_occupied: int
    frames_total: int
    unique_freqs: int
    median_burst_s: float
    freqs_hz: np.ndarray

    @property
    def occupancy(self) -> float:
        return self.frames_occupied / max(1, self.frames_total)


def count_blips(iq: np.ndarray, fs: float, dwell_s: float,
                frames_per_dwell: int = 8, threshold_db: float = 10.0,
                min_frames: int = 2, freq_tol_hz: float = 50e3,
                dc_notch_hz: float = 100e3,
                edge_frac: float = 0.8) -> BlipStats:
    """Detect bursts by per-frame peak-to-noise, then group adjacent frames.

    Grouping requires the peak to stay put: consecutive occupied frames only
    join the same burst if their peak frequencies agree to within freq_tol_hz.
    Without that, two different points landing back to back would merge into
    one long burst and the count would silently run low at high hop rates.

    Bins within dc_notch_hz of baseband zero are excluded from both the peak
    search and the noise estimate. The receiver's own LO leakage sits there
    permanently and is far stronger than any received tone, so without the
    notch every frame reports the same "signal" and the count measures the
    spur rather than the transmitter -- as a first run of this experiment
    duly demonstrated, returning one unique frequency and 50% occupancy.

    Bins beyond edge_frac of Nyquist are excluded for the same reason at the
    other end of the band: aliasing piles up against +/-fs/2 and produced a
    22 dB artifact sitting exactly on the edge, which read as a permanently
    occupied channel and pinned occupancy at 100%.
    """
    frame = max(16, int(round(dwell_s * fs / frames_per_dwell)))
    nframes = len(iq) // frame
    if nframes == 0:
        return BlipStats(0, 0, 0, 0, 0.0, np.array([]))

    win = np.hanning(frame)
    fbin = np.fft.fftfreq(frame, 1.0 / fs)
    keep = (np.abs(fbin) >= dc_notch_hz) & (np.abs(fbin) <= edge_frac * fs / 2)
    if not keep.any():
        raise ValueError(f"dc_notch_hz {dc_notch_hz:g} rejects the whole "
                         f"{fs/frame:g} Hz-per-bin spectrum; lower it")
    occupied = np.zeros(nframes, dtype=bool)
    peak_hz = np.zeros(nframes)

    for i in range(nframes):
        spec = np.abs(np.fft.fft(iq[i * frame:(i + 1) * frame] * win)) ** 2
        spec = np.where(keep, spec, 0.0)
        k = int(np.argmax(spec))
        noise = np.median(spec[keep])
        if noise > 0 and 10.0 * np.log10(spec[k] / noise) >= threshold_db:
            occupied[i] = True
            peak_hz[i] = fbin[k]

    bursts, freqs, run_start = 0, [], None
    for i in range(nframes + 1):
        live = i < nframes and occupied[i]
        if live and run_start is None:
            run_start = i
        elif live and run_start is not None:
            if abs(peak_hz[i] - peak_hz[i - 1]) > freq_tol_hz:
                if i - run_start >= min_frames:
                    bursts += 1
                    freqs.append(float(np.median(peak_hz[run_start:i])))
                run_start = i
        elif not live and run_start is not None:
            if i - run_start >= min_frames:
                bursts += 1
                freqs.append(float(np.median(peak_hz[run_start:i])))
            run_start = None

    f = np.array(freqs)
    uniq = 0
    if f.size:
        s = np.sort(f)
        uniq = 1 + int(np.sum(np.diff(s) > freq_tol_hz))
    return BlipStats(bursts, int(occupied.sum()), nframes, uniq,
                     float(dwell_s), f)


def synth(fs: float, seconds: float, span_hz: float, hop_rate: float,
          seed: int = 7, snr_db: float = 20.0) -> np.ndarray:
    """Transmitter sweeping span_hz uniformly; receiver sees only fs of it.

    The point of the synthetic case is that ground truth is known: the number
    of blips inside the window is hop_rate * seconds * min(1, fs/span_hz).
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    dwell = 1.0 / hop_rate
    per = int(round(dwell * fs))
    x = (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(2)
    amp = 10 ** (snr_db / 20.0)
    t = np.arange(per) / fs
    for start in range(0, n - per, per):
        f = rng.uniform(-span_hz / 2, span_hz / 2)
        if abs(f) < fs / 2 * 0.9:                  # inside the window
            ph = rng.uniform(0, 2 * np.pi)
            x[start:start + per] += amp * np.exp(2j * np.pi * f * t + 1j * ph)
    return x


def self_test() -> int:
    print("\nsynthetic check -- transmitter spans 100 MHz, window varies\n")
    span, rate, secs = 100e6, 1600.0, 0.4
    print(f"  span {span/1e6:g} MHz, {rate:g} hops/s, {secs:g} s "
          f"-> {int(rate*secs)} blips transmitted\n")
    print(f"  {'window':>9} {'predicted':>10} {'counted':>8} {'unique f':>9} {'ratio':>7}")
    base = None
    ok = True
    for fs in (2.5e6, 5e6, 10e6, 20e6):
        iq = synth(fs, secs, span, rate)
        st = count_blips(iq, fs, 1.0 / rate)
        pred = rate * secs * min(1.0, fs * 0.9 / span)
        if base is None:
            base = st.bursts or 1
        ratio = st.bursts / base
        print(f"  {fs/1e6:>6g} MHz {pred:>10.1f} {st.bursts:>8} "
              f"{st.unique_freqs:>9} {ratio:>6.2f}x")
        if pred > 5 and not (0.5 * pred <= st.bursts <= 1.6 * pred):
            ok = False
    print("\n  counts should track the window linearly; ratio column ~1/2/4/8")
    print("  RESULT:", "consistent with N proportional to bandwidth" if ok
          else "MISMATCH -- detector is not scaling as predicted")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--capture", help="interleaved int16 IQ file")
    p.add_argument("--fs", type=float, required=False)
    p.add_argument("--dwell-ms", type=float, default=0.625)
    p.add_argument("--threshold-db", type=float, default=12.0)
    p.add_argument("--dc-notch-hz", type=float, default=100e3,
                   help="ignore bins this close to baseband zero (LO leakage)")
    p.add_argument("--edge-frac", type=float, default=0.8,
                   help="keep only |f| below this fraction of Nyquist")
    p.add_argument("--frames-per-dwell", type=int, default=8)
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        return self_test()
    if not a.capture or not a.fs:
        p.error("--capture and --fs are required unless --self-test")
    raw = np.fromfile(a.capture, dtype="<i2")
    iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
    st = count_blips(iq, a.fs, a.dwell_ms / 1e3,
                     frames_per_dwell=a.frames_per_dwell,
                     threshold_db=a.threshold_db,
                     dc_notch_hz=a.dc_notch_hz, edge_frac=a.edge_frac)
    print(f"  bursts      : {st.bursts}")
    print(f"  unique freqs: {st.unique_freqs}")
    print(f"  occupancy   : {st.occupancy*100:.2f}% of {st.frames_total} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
