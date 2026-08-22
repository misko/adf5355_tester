#!/usr/bin/env python3
"""Identify which LNB band is live, from two tones one band apart.

    ####################################################################
    #  CLOSED, CONDUCTED PATH ONLY.  NEVER RADIATE.                    #
    #  10.7-12.75 GHz is satellite downlink spectrum. Coax into an     #
    #  attenuator and a load, or a shielded enclosure. No antenna.     #
    ####################################################################

A universal LNB has two local oscillators, 9.75 and 10.6 GHz, chosen by a
22 kHz tone on the coax and readable back from nothing at all. The IF cannot
tell you which is running, because the LO is free-running with of order
100 kHz of error and either band can explain any given IF.

So do not ask which frequency is closer. Pick two RF tones that the LNB's own
950-2150 MHz output filter answers for you: 11.30 GHz passes only through the
9.75 LO (1550 MHz; the 10.6 LO puts it at 700 MHz, below the filter), and
12.20 GHz passes only through the 10.6 LO (1600 MHz; the 9.75 LO puts it at
2450 MHz, above it). Exactly one tone survives, and which one names the band.
That is a presence test, so the LO error cannot reach it.

Stage 2 then reuses the surviving tone to measure the total offset, from four
tunings rather than one, because the receiver's tuning-dependent bias lands on
the answer one for one and is invisible from a single tuning.

    tools/lnb_band_id.py                    # print the plan, open nothing
    tools/lnb_band_id.py --open-radio       # measure
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time

import numpy as np

#: The LNB output filter. Everything here rests on it.
IF_MIN_HZ, IF_MAX_HZ = 950e6, 2150e6
LO_LOW_HZ, LO_HIGH_HZ = 9.75e9, 10.60e9

DEFAULT_LOW_RF_HZ = 11.30e9
DEFAULT_HIGH_RF_HZ = 12.20e9
DEFAULT_FS = 2.5e6
DEFAULT_SECONDS = 4.0
DEFAULT_URI = "ip:192.168.2.1"

#: Fixed, not random. The four tone positions are then 100 kHz apart, so at
#: most one can sit on 0 Hz IF and the best is always >=150 kHz clear of it.
#: Uniform random cannot make that guarantee and a fixed seed destroys it by
#: replaying the same bad draw every run.
DEFAULT_DITHERS_HZ = (-150e3, -50e3, 50e3, 150e3)

#: Stage 1 only has to see whether the tone is there at all, so it needs one
#: tuning, not four -- but not a tuning that puts the tone on DC. 50 kHz is
#: clear of the guard band and nothing else about it matters.
STAGE1_OFFSET_HZ = 50e3

#: Where to park the tone in baseband during a drift run. Far enough from DC
#: that the guard band is irrelevant, far enough from fs*0.4 that the tone can
#: drift for the whole run without approaching the edge.
MONITOR_TONE_HZ = 300e3

#: One number, used both to blank the receiver's LO leakage and to declare a
#: try too close to DC to believe. These were once 60 kHz and 20 kHz, and a
#: tone that landed at +33 kHz was silently swallowed by the wider notch while
#: the narrower threshold said the try was fine. They must be the same number.
DC_GUARD_HZ = 20e3

COARSE_FRAME = 8192
ON_THRESHOLD_DB = 15.0
BAND_MARGIN_DB = 10.0


def band_hypotheses(low_rf_hz: float, high_rf_hz: float) -> list[dict]:
    """The two hypotheses, with the arithmetic that makes them separable."""
    out = []
    for name, lo, rf in (("LOW", LO_LOW_HZ, low_rf_hz),
                         ("HIGH", LO_HIGH_HZ, high_rf_hz)):
        other = LO_HIGH_HZ if lo == LO_LOW_HZ else LO_LOW_HZ
        out.append({"band": name, "lo_hz": lo, "rf_hz": rf,
                    "if_hz": rf - lo, "if_other_hz": rf - other})
    return out


def check_pair(hyps: list[dict]) -> list[str]:
    """Refuse a tone pair that cannot decide anything. Cheap, and it catches
    a mistyped frequency before four minutes of capture do."""
    bad = []
    for h in hyps:
        if not IF_MIN_HZ <= h["if_hz"] <= IF_MAX_HZ:
            bad.append(f"{h['rf_hz']/1e9:.3f} GHz gives IF "
                       f"{h['if_hz']/1e6:.0f} MHz under its own {h['band']} "
                       f"hypothesis, outside the {IF_MIN_HZ/1e6:.0f}-"
                       f"{IF_MAX_HZ/1e6:.0f} MHz filter: it would never be "
                       f"heard at all")
        if IF_MIN_HZ <= h["if_other_hz"] <= IF_MAX_HZ:
            bad.append(f"{h['rf_hz']/1e9:.3f} GHz also lands in band under "
                       f"the other LO ({h['if_other_hz']/1e6:.0f} MHz): it "
                       f"survives either way and decides nothing")
    return bad


def measure(iq, fs: float, frame: int = COARSE_FRAME) -> dict:
    """Find the tone, then refine it over the longest continuous dwell.

    Two passes because they want different resolutions. The coarse pass only
    has to say *when* the tone is on, and 8192 samples at 2.5 MS/s is 305 Hz,
    which is plenty for that. The fine pass then transforms that whole stretch
    at once, so its resolution is set by the dwell -- about 1 Hz for a second
    -- rather than by the coarse frame.
    """
    n = len(iq) // frame
    fb = np.fft.fftfreq(frame, 1.0 / fs)
    keep = (np.abs(fb) > DC_GUARD_HZ) & (np.abs(fb) < fs * 0.4)
    win = np.hanning(frame)
    snr = np.zeros(n)
    for i in range(n):
        sp = np.abs(np.fft.fft(iq[i * frame:(i + 1) * frame] * win)) ** 2
        noise = np.median(sp[keep])
        if noise > 0:
            snr[i] = 10 * np.log10(np.where(keep, sp, 0.0).max() / noise)
    on = snr > ON_THRESHOLD_DB
    result = {"peak_snr_db": float(snr.max()) if n else 0.0,
              "tone_hz": None, "dwell_s": 0.0, "near_dc": False}
    if not on.any():
        return result

    best_len = best_start = cur = 0
    for i, v in enumerate(list(on) + [False]):
        if v:
            cur += 1
        else:
            if cur > best_len:
                best_len, best_start = cur, i - cur
            cur = 0
    seg = iq[best_start * frame:(best_start + best_len) * frame]
    if len(seg) < frame * 4:
        return result

    N = 1 << int(np.floor(np.log2(len(seg))))
    sp = np.abs(np.fft.fft(seg[:N] * np.hanning(N))) ** 2
    f = np.fft.fftfreq(N, 1.0 / fs)
    ok = (np.abs(f) > DC_GUARD_HZ) & (np.abs(f) < fs * 0.4)
    k = int(np.where(ok, sp, 0).argmax())
    y0, y1, y2 = (np.log(sp[k - 1] + 1e-30), np.log(sp[k] + 1e-30),
                  np.log(sp[(k + 1) % N] + 1e-30))
    den = y0 - 2 * y1 + y2
    delta = 0.5 * (y0 - y2) / den if den else 0.0
    tone = float(f[k] + delta * fs / N)
    result.update(tone_hz=tone, dwell_s=float(N / fs),
                  fine_bin_hz=float(fs / N),
                  # The estimate belongs to the middle of the stretch it was
                  # taken over, not to the start of the capture. Over a 60 s
                  # run that is a couple of seconds of timing error per point,
                  # which a drift slope would otherwise absorb.
                  t_centre_s=float((best_start * frame + N / 2) / fs),
                  # The guard blanks a band the tone could legitimately be in.
                  # If the peak sits right against that edge the real tone may
                  # be inside it and this is a spur, so say so rather than
                  # quietly returning the spur.
                  near_dc=bool(abs(tone) < 2 * DC_GUARD_HZ))
    return result


class Receiver:
    """One Pluto, retuned per capture, single-shot.

    Single-shot and not streaming: a streaming capture on this hardware only
    sustains about 40% of real time, so a multi-second window arrives torn and
    the dwell structure the coarse pass looks for is not there any more.
    """

    def __init__(self, uri: str, fs: float, seconds: float, chan: int) -> None:
        import adi                                            # noqa: PLC0415
        self.sdr = adi.ad9361(uri=uri)
        # The context timeout has to cover the capture, or a slow radio and a
        # wedged radio look identical from here.
        self.sdr._ctx.set_timeout(int(seconds * 2000) + 30_000)
        self.sdr.sample_rate = int(fs)
        self.sdr.rx_rf_bandwidth = int(min(fs * 0.8, 2e6))
        setattr(self.sdr, f"gain_control_mode_chan{chan}", "slow_attack")
        self.fs, self.seconds, self.chan = fs, seconds, chan
        self._tuned_to = None

    def capture(self, tune_hz: float):
        """One contiguous capture at `tune_hz`.

        The discarded buffer exists to throw away the retune transient, so it
        is only paid on an actual retune. Holding one tuning -- which is what
        a drift run does -- it would double the cost of every point for
        nothing, and halve how many points fit in the run.
        """
        want = int(round(tune_hz))
        if want != self._tuned_to:
            self.sdr.rx_destroy_buffer()
            self.sdr.rx_enabled_channels = [self.chan]
            self.sdr.rx_lo = want
            self.sdr._rxadc.set_kernel_buffers_count(1)
            self.sdr.rx_buffer_size = int(self.seconds * self.fs)
            self.sdr.rx()                    # discard the retune transient
            self._tuned_to = want
        x = np.asarray(self.sdr.rx())
        return x[0] if x.ndim > 1 else x


def monitor(rx, tune_hz: float, seconds: float, nominal_if_hz: float,
            out=None, progress=print) -> list[dict]:
    """Park on one tuning and re-measure the tone until `seconds` have passed.

    Fixed tuning, deliberately: the receiver's tuning-dependent bias is worth
    about 90 Hz here, and dithering during a drift run would inject all of it
    as scatter on the slope. Held still it is a constant, and a constant is
    invisible to a slope.
    """
    rows = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        t_start = time.monotonic() - t0
        r = measure(rx.capture(tune_hz), rx.fs)
        if r["tone_hz"] is None:
            progress(f"    {t_start:6.1f} s  no tone")
            continue
        # Stamp the estimate at the middle of the stretch it came from.
        t = t_start + r["t_centre_s"]
        row = {"t_s": t, "if_hz": tune_hz + r["tone_hz"],
               "offset_hz": tune_hz + r["tone_hz"] - nominal_if_hz,
               "snr_db": r["peak_snr_db"], "tune_hz": tune_hz}
        rows.append(row)
        if out is not None:
            out.write(json.dumps({"kind": "monitor", **row}) + "\n")
        progress(f"    {t:6.1f} s  IF {row['if_hz']/1e6:15.6f} MHz  "
                 f"offset {row['offset_hz']/1e3:+9.3f} kHz  "
                 f"{r['peak_snr_db']:5.1f} dB")
    return rows


def fit_drift(rows: list[dict]) -> dict | None:
    """Least squares line through (t, f). Slope is the drift."""
    if len(rows) < 3:
        return None
    t = np.array([r["t_s"] for r in rows])
    f = np.array([r["if_hz"] for r in rows])
    slope, intercept = np.polyfit(t - t.mean(), f, 1)
    resid = f - (slope * (t - t.mean()) + intercept)
    # Standard error of the slope, which is what says whether the drift is
    # real or just the scatter arranging itself into a line.
    dof = len(t) - 2
    s_err = float(np.sqrt((resid @ resid) / dof / ((t - t.mean()) ** 2).sum()))
    return {"slope_hz_s": float(slope), "slope_stderr_hz_s": s_err,
            "intercept_hz": float(intercept), "span_s": float(np.ptp(t)),
            "resid_rms_hz": float(np.sqrt(resid @ resid / len(resid))),
            "n": len(t), "t": t, "f": f, "resid": resid}


def plot_drift(fit: dict, path: str, nominal_if_hz: float, title: str) -> bool:
    try:
        import matplotlib                                     # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt                       # noqa: PLC0415
    except ImportError:
        return False
    t, f, resid = fit["t"], fit["f"], fit["resid"]
    khz = (f - nominal_if_hz) / 1e3
    line = (fit["slope_hz_s"] * (t - t.mean()) + fit["intercept_hz"]
            - nominal_if_hz) / 1e3

    fig, (a, b) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1]})
    a.plot(t, khz, "o", ms=6, color="#1b6ca8", label="measured")
    a.plot(t, line, "-", lw=1.5, color="#d1495b",
           label=f"fit {fit['slope_hz_s']:+.2f} Hz/s")
    a.set_ylabel(f"IF offset from {nominal_if_hz/1e6:.0f} MHz  (kHz)")
    a.set_title(title)
    a.grid(alpha=0.3); a.legend(loc="best")
    b.axhline(0, color="#888", lw=1)
    b.plot(t, resid, "o", ms=5, color="#1b6ca8")
    b.set_ylabel("residual (Hz)"); b.set_xlabel("time (s)")
    b.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return True


def report_drift(fit: dict, rows: list[dict], a, nominal: float,
                 title: str, out=None, ppb_ref: float | None = None,
                 ppb_label: str = "carrier") -> int:
    print(f"\n    drift {fit['slope_hz_s']:+.3f} +/- "
          f"{fit['slope_stderr_hz_s']:.3f} Hz/s over {fit['span_s']:.1f} s "
          f"({fit['n']} points)")
    print(f"    residual rms {fit['resid_rms_hz']:.1f} Hz")
    sigma = (abs(fit["slope_hz_s"]) / fit["slope_stderr_hz_s"]
             if fit["slope_stderr_hz_s"] else float("inf"))
    print(f"    that is {sigma:.1f} sigma, so the drift is "
          + ("real, not scatter fitting itself to a line" if sigma >= 3 else
             "NOT separable from the scatter over this span"))
    # Against the carrier itself, so the two paths can be compared even though
    # they sit at wildly different frequencies.
    ref = ppb_ref if ppb_ref else nominal
    print(f"    as a fraction of the {ref/1e9:.3f} GHz {ppb_label}: "
          f"{fit['slope_hz_s']/ref*1e9:+.3f} ppb/s")
    if out:
        out.write(json.dumps({"kind": "drift", "nominal_hz": nominal, **{
            k: v for k, v in fit.items()
            if k not in ("t", "f", "resid")}}) + "\n")
    if a.plot:
        ok = plot_drift(fit, a.plot, nominal, title)
        print(f"    plot written to {a.plot}" if ok else
              "    plot skipped: matplotlib not available")
    return 0


def run_monitor_only(rx, a, out) -> int:
    """Drift on a path with no LNB in it, so no band and no filter test.

    Still needs to find the tone before it can park on it: the offset here is
    the ADF reference against the receiver's clock, and neither is known in
    advance. Acquire at the nominal, then re-park so the tone sits at
    +300 kHz baseband with room to move.
    """
    nominal = a.monitor_only_ghz * 1e9
    print(f"  MONITOR ONLY -- no band test, tone expected at "
          f"{nominal/1e9:.6f} GHz on RX{a.rx_channel}")
    for probe in (nominal, nominal + MONITOR_TONE_HZ):
        r = measure(rx.capture(probe), a.fs)
        if r["tone_hz"] is not None and not r["near_dc"]:
            break
        print(f"    tuning {probe/1e6:.3f} MHz: "
              + ("no tone" if r["tone_hz"] is None else "tone too near DC")
              + ", trying another")
    else:
        print("\n  NOT FOUND: nothing usable at either acquisition tuning.")
        return 1
    offset = probe + r["tone_hz"] - nominal
    print(f"    acquired at {probe/1e6:.3f} MHz: tone "
          f"{r['tone_hz']/1e3:+.1f} kHz -> offset {offset/1e3:+.3f} kHz, "
          f"{r['peak_snr_db']:.1f} dB")

    park = nominal + offset - MONITOR_TONE_HZ
    print(f"\n  DRIFT -- {a.monitor_s:g} s on one tuning ({park/1e6:.3f} MHz)")
    rows = monitor(rx, park, a.monitor_s, nominal, out=out)
    fit = fit_drift(rows)
    if fit is None:
        print(f"\n    only {len(rows)} usable point(s); need 3 to fit a slope")
        return 1
    return report_drift(fit, rows, a, nominal,
                        f"RX{a.rx_channel} conducted, {nominal/1e9:.3f} GHz",
                        out=out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--low-rf-ghz", type=float, default=DEFAULT_LOW_RF_HZ / 1e9,
                   help="tone that survives only the 9.75 GHz LO")
    p.add_argument("--high-rf-ghz", type=float,
                   default=DEFAULT_HIGH_RF_HZ / 1e9,
                   help="tone that survives only the 10.6 GHz LO")
    p.add_argument("--fs", type=float, default=DEFAULT_FS,
                   help="2.5 MS/s covers a +/-500 kHz worst-case LNB error")
    p.add_argument("--seconds", type=float, default=DEFAULT_SECONDS,
                   help="must exceed dwell + cycle, or a whole dwell is not "
                        "guaranteed to fall inside the window")
    p.add_argument("--rx-channel", type=int, default=0,
                   help="0 = RX1, which is where the LNB is")
    p.add_argument("--uri", default=DEFAULT_URI)
    p.add_argument("--dithers-khz", default=",".join(
        f"{d/1e3:g}" for d in DEFAULT_DITHERS_HZ))
    p.add_argument("--monitor-only-ghz", type=float, default=None,
                   help="skip the band and offset stages and monitor a tone "
                        "expected at this frequency. For a path where the "
                        "filter test means nothing: no LNB in the chain, or a "
                        "band already known by other means")
    p.add_argument("--monitor-s", type=float, default=0.0,
                   help="after the offset, hold one tuning for this long and "
                        "fit the drift. 0 disables it")
    p.add_argument("--plot", default=None,
                   help="write the drift plot here (needs matplotlib)")
    p.add_argument("--out", default=None, help="append JSON lines here")
    p.add_argument("--open-radio", action="store_true",
                   help="actually open the PlutoSDR. Without this nothing is "
                        "opened and nothing is captured")
    a = p.parse_args(argv)

    hyps = band_hypotheses(a.low_rf_ghz * 1e9, a.high_rf_ghz * 1e9)
    print("  tone pair and what the LNB filter does with it\n")
    print(f"    {'RF GHz':>9} {'via LO 9.75':>14} {'via LO 10.6':>14}   decides")
    print("    " + "-" * 60)
    for h in hyps:
        ifs = {}
        for lo in (LO_LOW_HZ, LO_HIGH_HZ):
            v = h["rf_hz"] - lo
            ifs[lo] = (f"{v/1e6:8.0f} MHz" +
                       ("  " if IF_MIN_HZ <= v <= IF_MAX_HZ else " x"))
        print(f"    {h['rf_hz']/1e9:>9.3f} {ifs[LO_LOW_HZ]:>14} "
              f"{ifs[LO_HIGH_HZ]:>14}   {h['band']}")
    for problem in check_pair(hyps):
        print(f"\n  REFUSING: {problem}")
    if check_pair(hyps):
        return 2

    dithers = [float(t) * 1e3 for t in a.dithers_khz.split(",")]
    spacing = min(abs(x - y) for i, x in enumerate(dithers)
                  for y in dithers[i + 1:]) if len(dithers) > 1 else 0.0
    print(f"\n  capture     : {a.seconds:g} s single-shot at {a.fs/1e6:g} MS/s "
          f"= {int(a.seconds*a.fs):,} samples, kernel buffers 1")
    print(f"  dithers     : " + ", ".join(f"{d/1e3:+g}" for d in dithers)
          + f" kHz, spaced {spacing/1e3:g} kHz")
    print(f"  DC guard    : +/-{DC_GUARD_HZ/1e3:g} kHz, blanked and flagged "
          f"with the same number")
    if spacing and spacing < 2 * DC_GUARD_HZ:
        print(f"  WARNING: dithers closer together than the guard band; more "
              f"than one try can be lost to DC at once")

    if not a.open_radio:
        print("\n  DRY RUN: no radio was opened and nothing was captured. "
              "Add --open-radio to measure.\n")
        return 0

    print("\n  SAFETY: closed, conducted path only. Do not connect an "
          "antenna.\n")
    rx = Receiver(a.uri, a.fs, a.seconds, a.rx_channel)
    out = open(a.out, "w", buffering=1) if a.out else None

    try:
        if a.monitor_only_ghz is not None:
            return run_monitor_only(rx, a, out)
        print("  discarding one capture to let the AGC settle")
        rx.capture(hyps[0]["if_hz"])

        print("\n  STAGE 1 -- which band?")
        for h in hyps:
            r = measure(rx.capture(h["if_hz"] + STAGE1_OFFSET_HZ), a.fs)
            h["snr_db"] = r["peak_snr_db"]
            print(f"    {h['band']:>4}: tune "
                  f"{(h['if_hz']+STAGE1_OFFSET_HZ)/1e6:9.3f} MHz -> peak "
                  f"{r['peak_snr_db']:5.1f} dB")
        win = max(hyps, key=lambda h: h["snr_db"])
        margin = abs(hyps[0]["snr_db"] - hyps[1]["snr_db"])
        print(f"    => {win['band']} band  (margin {margin:.1f} dB)")
        if out:
            out.write(json.dumps({"kind": "band", "band": win["band"],
                                  "margin_db": margin}) + "\n")
        if margin < BAND_MARGIN_DB:
            print(f"\n  INCONCLUSIVE: {margin:.1f} dB is under the "
                  f"{BAND_MARGIN_DB:g} dB this test should give. Check the "
                  f"transmitter is running and the LNB is powered.")
            return 1

        nominal = win["if_hz"]
        print(f"\n  STAGE 2 -- offset over {len(dithers)} dithers, tone at "
              f"{win['rf_hz']/1e9:.2f} GHz, IF {nominal/1e6:.0f} MHz")
        print(f"    {'dither':>9} {'tune MHz':>11} {'tone @ bb':>12} "
              f"{'implied IF':>15} {'offset':>11} {'SNR':>8} {'bin':>8}")
        rows = []
        for d in dithers:
            tune = nominal + d
            r = measure(rx.capture(tune), a.fs)
            if r["tone_hz"] is None:
                print(f"    {d/1e3:>+8.0f}k {tune/1e6:>11.3f} {'no tone':>12}")
                continue
            if_meas = tune + r["tone_hz"]
            r.update(dither_hz=d, tune_hz=tune, if_meas_hz=if_meas,
                     offset_hz=if_meas - nominal, band=win["band"])
            rows.append(r)
            if out:
                out.write(json.dumps({"kind": "try", **r}) + "\n")
            print(f"    {d/1e3:>+8.0f}k {tune/1e6:>11.3f} "
                  f"{r['tone_hz']/1e3:>+11.1f}k {if_meas/1e6:>15.6f} "
                  f"{r['offset_hz']/1e3:>+10.3f}k {r['peak_snr_db']:>7.1f}dB "
                  f"{r['fine_bin_hz']:>7.2f}Hz"
                  + ("   <-- near DC, rejected" if r["near_dc"] else ""))

        good = [r for r in rows if not r["near_dc"]]
        if len(good) < 2:
            print("\n  too few usable tunings to quote a spread")
            return 1
        offs = [r["offset_hz"] for r in good]
        # Median rejection as well as the DC flag: a spur can be far from DC.
        med = st.median(offs)
        keep = [o for o in offs if abs(o - med) < 10e3]
        print(f"\n    mean offset {st.mean(keep)/1e3:+.4f} kHz   spread "
              f"{st.pstdev(keep):.1f} Hz across {len(keep)} tunings spanning "
              f"{(max(r['tune_hz'] for r in good) - min(r['tune_hz'] for r in good))/1e3:.0f} kHz")
        if len(keep) < len(offs):
            print(f"    {len(offs)-len(keep)} further tuning(s) rejected as "
                  f"outliers more than 10 kHz from the median")
        print("\n    the spread is the tuning-dependent receiver bias. It is "
              "the accuracy\n    floor of this measurement, and it is a "
              "systematic, not statistics.")
        offset = st.mean(keep)
        if out:
            out.write(json.dumps({"kind": "result", "band": win["band"],
                                  "offset_hz": offset,
                                  "spread_hz": st.pstdev(keep),
                                  "n": len(keep)}) + "\n")

        if a.monitor_s <= 0:
            return 0

        # Park where the tone sits clear of DC and of the passband edge. The
        # offset just measured is what says where that is; without it a 300 kHz
        # wander would be enough to put the tone somewhere unhelpful.
        park = nominal + offset - MONITOR_TONE_HZ
        print(f"\n  STAGE 3 -- drift, {a.monitor_s:g} s on one tuning "
              f"({park/1e6:.3f} MHz, tone parked at "
              f"{MONITOR_TONE_HZ/1e3:+.0f} kHz baseband)")
        rows = monitor(rx, park, a.monitor_s, nominal, out=out)
        fit = fit_drift(rows)
        if fit is None:
            print(f"\n    only {len(rows)} usable point(s); need 3 to fit a "
                  f"slope")
            return 1
        return report_drift(
            fit, rows, a, nominal,
            f"ADF5355 {win['rf_hz']/1e9:.2f} GHz through the "
            f"{win['band']} band LNB", out=out,
            ppb_ref=win["lo_hz"], ppb_label="LNB LO")
    finally:
        if out:
            out.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
