#!/usr/bin/env python3
"""Drive a whole lever-arm run: visit every cluster, many times, and record it.

    ####################################################################
    #  CLOSED, CONDUCTED PATHS ONLY.  NEVER RADIATE.                   #
    #  10.7-11.9 GHz is satellite downlink spectrum. Coax into an      #
    #  attenuator and a load, or a shielded enclosure. No antenna.     #
    ####################################################################

The transmitter is already hopping over every cluster on its own schedule and
knows nothing about this program. All this does is retune the receiver, capture,
decode, and append one line per capture -- so it can be stopped and restarted,
and a run that dies at capture 87 still has 86 usable ones on disk.

Three things about the order it works in are not arbitrary:

* **Sweeps.** One capture of every cluster, then again, and again. Every sweep
  holds every cluster, which is what lets d_rx be fitted with a free offset per
  sweep and so be immune to the LNB's drift whatever shape that drift has.
* **A fresh random cluster order every sweep.** A fixed order would put each
  cluster at a fixed phase of the sweep, and the drift inside a sweep would then
  land on the clusters in a fixed pattern -- the monotonic-ladder mistake, one
  level up.
* **A random tuning dither on every capture.** The receiver's tuning-dependent
  bias is the dominant error of the whole measurement, at 362 Hz peak to peak
  on this hardware. Dithering the rx_lo makes that bias an independent draw per
  capture instead of a fixture, so it averages down as sqrt(N). Without it,
  more captures buy nothing at all.

Nothing here opens a radio unless ``--open-radio`` is passed. The default is a
dry run that prints the plan, and ``--synthetic`` runs the entire pipeline --
schedule, capture, decode, record -- against captures built in memory from a
known d_rx and d_lnb, which is how it is tested.

    tools/lever_run.py                                  # print the plan only
    tools/lever_run.py --synthetic --out /tmp/fake.jsonl # whole pipeline, no radio
    tools/lever_run.py --open-radio --out run.jsonl      # the real thing
    tools/lever_fit.py run.jsonl                         # then fit it
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adf5355.hopper import (DEFAULT_BAND_EXTRA_S, DEFAULT_BLOCK,  # noqa: E402
                            DEFAULT_CLUSTER_POINTS,
                            DEFAULT_CLUSTER_SPAN_HZ, DEFAULT_JITTER,
                            DEFAULT_MIN_HOP_S, DEFAULT_PERIOD_CYCLES,
                            DEFAULT_SEED, cluster_centres, describe_clusters,
                            make_cluster_schedule, plan_clusters,
                            period_duration)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, str(Path(__file__).resolve().parent / f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


hd = _load("hop_decode")
lf = _load("lever_fit")

DEFAULT_LOW_GHZ = 10.70          # the LNB low band, less its edges
DEFAULT_HIGH_GHZ = 11.90
DEFAULT_CLUSTERS = 4
DEFAULT_SECONDS = 3.0
DEFAULT_FS = 2.5e6
DEFAULT_GAIN = 40.0
DEFAULT_URI = "ip:192.168.2.1"
# Starting guesses, used ONLY to centre the receiver on the comb. They cannot
# bias the answer: the decoder measures the comb's real position and the fit
# never sees these numbers.
DEFAULT_D_RX_GUESS = 8.94e-6
DEFAULT_LO_ERROR_HZ = 94_000.0
# What one capture costs BESIDES its own length: the retune and settle, the
# write, and the decode -- which is the big one. Measured on a Pi 4: 5.8 s to
# decode one 3 s capture of a 4-cluster schedule, plus about 0.4 s for
# everything else. This is not cosmetic; it is what says whether the
# transmitter's --cycles outlasts the receiver's --sweeps, and a run whose
# transmitter stops halfway loses every capture after that point.
PER_CAPTURE_OVERHEAD_S = 7.0


def tuning_for(cluster_if_hz: float, dither_hz: float, *, lo_hz: float,
               d_rx_guess: float, lo_error_hz: float) -> float:
    """Where to tune for one capture: on the comb, plus the dither.

    The comb does not sit at the nominal IF -- it sits about 100 kHz below it,
    and by an amount that itself varies with the cluster. Centring on the
    prediction rather than on the nominal leaves the dither the whole of the
    remaining passband to move in, which is what makes a wide dither possible
    at all.
    """
    predicted = (lf.reported_if_model(cluster_if_hz, lo_hz=lo_hz,
                                      d_rx=d_rx_guess,
                                      d_lnb=lo_error_hz / lo_hz)
                 - cluster_if_hz)
    return float(cluster_if_hz + predicted + dither_hz)


class PlutoReceiver:
    """One Pluto, retuned per capture. Imported lazily so nothing else needs it.

    One device object for the whole run, not one per capture: the retune is the
    operation whose bias this measurement is built around, so it wants to be
    the ordinary one the radio does, not a side effect of re-opening the
    context a hundred times.
    """

    #: Buffers thrown away after a retune, before anything is kept. The LO has
    #: to settle and the receiver's DC-offset tracking has to reconverge, and a
    #: capture that starts inside that is not a capture of the schedule.
    SETTLE_BUFFERS = 2

    def __init__(self, uri: str, fs: float, gain: float, nbuf: int) -> None:
        import adi                                            # noqa: PLC0415
        self.sdr = adi.Pluto(uri=uri)
        self.sdr.sample_rate = int(fs)
        self.sdr.rx_rf_bandwidth = int(fs * 0.8)
        self.sdr.gain_control_mode_chan0 = "manual"
        self.sdr.rx_hardwaregain_chan0 = gain
        self.sdr.rx_destroy_buffer()
        self.sdr.rx_buffer_size = nbuf
        self.fs = fs
        self.nbuf = nbuf

    def capture(self, path: Path, if_hz: float, seconds: float) -> tuple[int, float]:
        # Tear the buffer down around the retune. Whatever is already in flight
        # was sampled at the old tuning, and letting it through would put a
        # slice of the wrong frequency at the head of the capture.
        self.sdr.rx_destroy_buffer()
        self.sdr.rx_lo = int(round(if_hz))
        self.sdr.rx_buffer_size = self.nbuf
        for _ in range(self.SETTLE_BUFFERS):
            self.sdr.rx()
        want = int(seconds * self.fs)
        total = 0
        started = time.monotonic()
        with open(path, "wb", buffering=1 << 22) as fh:
            while total < want:
                block = np.asarray(self.sdr.rx())
                out = np.empty(block.size * 2, dtype="<i2")
                out[0::2] = np.real(block).astype("<i2")
                out[1::2] = np.imag(block).astype("<i2")
                fh.write(out.tobytes())
                total += block.size
        return total, time.monotonic() - started


def run(plan, visits, *, lo_hz: float, fs: float, seconds: float,
        seed: int, min_hop_s: float, block: int, jitter: float,
        period_cycles: int, band_extra_s: float, frame: int,
        d_rx_guess: float, lo_error_hz: float, receiver=None,
        synthetic: dict | None = None, out=None, workdir: str = "/tmp",
        estimator: str = hd.DEFAULT_ESTIMATOR,
        decimate: int = hd.DEFAULT_DECIMATE,
        guard_start_s: float = hd.DEFAULT_GUARD_START_S,
        progress=print) -> list:
    """Capture and decode every planned visit, appending each record as it lands.

    ``receiver`` is a :class:`PlutoReceiver` or None. With ``synthetic`` given
    the capture is built in memory from a known truth instead, which exercises
    exactly the same decode and record path -- there is no separate offline
    code path to rot.
    """
    records = []
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for v in visits:
        if_c = float(plan.centres_hz[v.cluster]) - lo_hz
        rx_lo = tuning_for(if_c, v.dither_hz, lo_hz=lo_hz,
                           d_rx_guess=d_rx_guess, lo_error_hz=lo_error_hz)
        t_abs = time.time() - t0 if synthetic is None else v.index * (
            seconds + synthetic.get("overhead_s", 9.0))
        common = dict(fs=fs, centre_hz=rx_lo, seed=seed, min_hop_s=min_hop_s,
                      jitter=jitter, period_cycles=period_cycles, lo_hz=lo_hz,
                      cluster_plan=plan, cluster=v.cluster, block=block,
                      band_extra_s=band_extra_s)
        tmp = None
        try:
            if synthetic is not None:
                source = hd.synthesise_cluster(
                    plan, v.cluster, fs=fs, centre_hz=rx_lo, seed=seed,
                    min_hop_s=min_hop_s, block=block, jitter=jitter,
                    period_cycles=period_cycles, lo_hz=lo_hz, seconds=seconds,
                    start_at_s=synthetic.get("start_at_s", 0.037)
                    + 0.911 * v.index,
                    t_abs_s=t_abs - synthetic["t_ref_s"],
                    band_extra_s=band_extra_s,
                    noise_seed=synthetic.get("noise_seed", 1) + v.index,
                    d_rx=synthetic["d_rx"], d_lnb=synthetic["d_lnb"],
                    drift_hz_s=synthetic["drift_hz_s"],
                    tuning_bias_hz=float(synthetic["bias"][v.index]),
                    snr_db=synthetic["snr_db"])
            elif receiver is not None:
                tmp = work / f"lever-{uuid.uuid4().hex[:12]}.iq"
                count, elapsed = receiver.capture(tmp, rx_lo, seconds)
                live = count / fs / elapsed
                if live < 0.98:
                    progress(f"  WARNING: capture {v.index} fell behind real "
                             f"time ({live*100:.1f}%); its timeline is broken")
                source = str(tmp)
            else:
                raise RuntimeError("no receiver and not synthetic")
            result = hd.decode(source, frame=frame, estimator=estimator,
                               decimate=decimate, guard_start_s=guard_start_s,
                               t_abs_s=t_abs, **common)
        except Exception as exc:                             # noqa: BLE001
            # One capture that cannot be decoded must not end a run that is
            # twenty minutes long. Record it as disowned -- the fit drops it
            # and says how many it dropped -- and carry on.
            progress(f"  {v.index:4d}  sweep {v.sweep:3d}  cluster {v.cluster}"
                     f"  FAILED: {exc}")
            rec = lf.CaptureRecord(
                index=v.index, sweep=v.sweep, cluster=v.cluster,
                cluster_centre_hz=float(plan.centres_hz[v.cluster]),
                rx_lo_hz=rx_lo, dither_hz=v.dither_hz, t_abs_s=t_abs,
                mean_if_hz=if_c, mean_error_hz=float("nan"),
                mean_error_stderr_hz=float("nan"), slope=float("nan"),
                slope_stderr=float("nan"), trustworthy=False,
                warnings=[f"decode failed: {exc}"])
            records.append(rec)
            if out is not None:
                out.write(json.dumps({"kind": "capture",
                                      **lf.asdict(rec)}) + "\n")
                out.flush()
            continue
        finally:
            if tmp is not None and tmp.exists():
                tmp.unlink()
        rec = lf.CaptureRecord.from_decode(result, index=v.index,
                                           sweep=v.sweep, rx_lo_hz=rx_lo,
                                           dither_hz=v.dither_hz)
        records.append(rec)
        if out is not None:
            out.write(json.dumps({"kind": "capture",
                                  **lf.asdict(rec)}) + "\n")
            out.flush()
        flag = "" if rec.trustworthy else "   <-- DISOWNED: " + \
            "; ".join(rec.warnings)
        progress(f"  {v.index:4d}  sweep {v.sweep:3d}  cluster {v.cluster}  "
                 f"rx_lo {rx_lo/1e6:9.3f} MHz  offset "
                 f"{rec.mean_error_hz:+10.2f} Hz  slope "
                 f"{rec.slope*1e6:+8.4f} ppm  {rec.recovered}/{rec.points} pts"
                 f"{flag}")
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sched = p.add_argument_group(
        "schedule (must match adf5355_rf_lever.sh exactly)")
    sched.add_argument("--seed", type=lambda v: int(v, 0), default=DEFAULT_SEED)
    sched.add_argument("--low-ghz", type=float, default=DEFAULT_LOW_GHZ)
    sched.add_argument("--high-ghz", type=float, default=DEFAULT_HIGH_GHZ)
    sched.add_argument("--clusters", type=int, default=DEFAULT_CLUSTERS)
    sched.add_argument("--cluster-points", type=int,
                       default=DEFAULT_CLUSTER_POINTS)
    sched.add_argument("--span-khz", type=float,
                       default=DEFAULT_CLUSTER_SPAN_HZ / 1e3,
                       help="how wide one cluster is; the points sit on a "
                            "Golomb ruler inside it")
    sched.add_argument("--min-hop-ms", type=float,
                       default=DEFAULT_MIN_HOP_S * 1e3)
    sched.add_argument("--block", type=int, default=DEFAULT_BLOCK)
    sched.add_argument("--jitter", type=float, default=DEFAULT_JITTER)
    sched.add_argument("--period-cycles", type=int,
                       default=DEFAULT_PERIOD_CYCLES)
    sched.add_argument("--band-extra-ms", type=float,
                       default=DEFAULT_BAND_EXTRA_S * 1e3,
                       help="how much longer a band-changing dwell is, and how "
                            "much of it the receiver skips")

    rx = p.add_argument_group("receive chain")
    rx.add_argument("--lo-hz", type=float, default=lf.DEFAULT_LO_HZ)
    rx.add_argument("--lo-error-hz", type=float, default=DEFAULT_LO_ERROR_HZ)
    rx.add_argument("--d-rx-guess", type=float, default=DEFAULT_D_RX_GUESS)
    rx.add_argument("--fs", type=float, default=DEFAULT_FS)
    rx.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    rx.add_argument("--frame", type=int, default=hd.DEFAULT_FRAME)
    rx.add_argument("--gain", type=float, default=DEFAULT_GAIN)
    rx.add_argument("--uri", default=DEFAULT_URI)
    rx.add_argument("--nbuf", type=int, default=hd.DEFAULT_NBUF)
    rx.add_argument("--estimator", choices=hd.ESTIMATOR_NAMES,
                    default=hd.DEFAULT_ESTIMATOR)
    rx.add_argument("--decimate", type=int, default=hd.DEFAULT_DECIMATE)
    rx.add_argument("--guard-start-ms", type=float,
                    default=hd.DEFAULT_GUARD_START_S * 1e3,
                    help="skipped at the head of every ordinary dwell; a "
                         "band-changing dwell skips its whole extra as well")

    order = p.add_argument_group("the run")
    order.add_argument("--sweeps", type=int, default=lf.DEFAULT_SWEEPS,
                       help="one capture of every cluster per sweep; the "
                            "answer's precision goes as sqrt(this)")
    order.add_argument("--dither-khz", type=float,
                       default=lf.DEFAULT_DITHER_HZ / 1e3,
                       help="tuning dither half-range. This is the single most "
                            "important setting here: it is what turns the "
                            "receiver's tuning bias from a fixture into "
                            "something that averages away")
    order.add_argument("--visit-seed", type=lambda v: int(v, 0), default=1234567,
                       help="seeds the cluster order and the dither, so the "
                            "whole run is reproducible")
    order.add_argument("--out", default=None,
                       help="append records here as JSON lines")
    order.add_argument("--workdir", default="/tmp")
    order.add_argument("--open-radio", action="store_true",
                       help="actually open the PlutoSDR and capture. Without "
                            "this nothing is opened and nothing is captured")
    order.add_argument("--synthetic", action="store_true",
                       help="build every capture in memory from a known d_rx "
                            "and d_lnb and run the whole pipeline on it")
    order.add_argument("--inject-d-rx", type=float, default=8.94e-6)
    order.add_argument("--inject-d-lnb", type=float, default=9.641e-6)
    order.add_argument("--inject-drift", type=float, default=4.5)
    order.add_argument("--inject-bias", type=float, default=127.0)
    order.add_argument("--snr-db", type=float, default=10.0)
    order.add_argument("--fit", action="store_true",
                       help="fit the run as soon as it finishes")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    centres = cluster_centres(args.low_ghz * 1e9, args.high_ghz * 1e9,
                              args.clusters)
    plan = plan_clusters(centres, args.cluster_points,
                         round(args.span_khz * 1e3))
    min_hop_s = args.min_hop_ms / 1e3
    band_extra_s = args.band_extra_ms / 1e3
    hops = make_cluster_schedule(args.seed, plan, min_hop_s,
                                 args.period_cycles + 1, args.block,
                                 args.jitter, args.period_cycles, band_extra_s)
    period = period_duration(hops, plan.points, args.period_cycles,
                             plan.clusters * plan.points)
    print(describe_clusters(plan, hops, args.seed, min_hop_s, args.block,
                            args.lo_hz, args.period_cycles, band_extra_s))

    visits = lf.plan_visits(args.visit_seed, plan.clusters, args.sweeps,
                            args.dither_khz * 1e3)
    per_capture = args.seconds + PER_CAPTURE_OVERHEAD_S
    print(f"\n  run          : {len(visits)} captures = {args.sweeps} sweeps "
          f"of {plan.clusters} clusters, {args.seconds:g} s each")
    print(f"  listening    : {args.seconds/period:.1f} periods per capture "
          f"({period*1e3:.0f} ms period; two is the floor)")
    print(f"  dither       : +/-{args.dither_khz:g} kHz on rx_lo, redrawn every "
          f"capture -- this is what makes the tuning bias average away")
    print(f"  expected     : about {args.sweeps} captures per cluster, so the "
          f"tuning bias falls by sqrt({args.sweeps}) = "
          f"{np.sqrt(args.sweeps):.1f}x")
    reach = plan.span_hz() / 2 + args.dither_khz * 1e3 + 30e3
    print(f"  passband     : the comb reaches {reach/1e3:.0f} kHz from the "
          f"tuning, against {args.fs*0.4/1e3:.0f} kHz of half-passband")
    if reach > args.fs * 0.4:
        print("  WARNING: the dither plus the cluster span will push points "
              "outside the passband;\n           narrow --dither-khz or "
              "--span-khz, or raise --fs")
    if args.seconds < 2 * period:
        print(f"  WARNING: {args.seconds:g} s is under two periods "
              f"({2*period:.2f} s); some points may never be heard")
    print(f"  rough time   : {len(visits)*per_capture/60:.0f} min including "
          f"decode ({args.seconds:g} s capture + {PER_CAPTURE_OVERHEAD_S:g} s "
          f"retune, write and decode, measured on a Pi 4)")
    print(f"  transmitter  : must still be hopping {len(visits)*per_capture/60:.0f} "
          f"min from now -- check adf5355_rf_lever.sh's CYCLES covers it")

    if not (args.open_radio or args.synthetic):
        print("\n  DRY RUN: no radio was opened and nothing was captured. "
              "Add --synthetic to run\n  the whole pipeline offline, or "
              "--open-radio to measure for real.\n")
        for v in visits[:12]:
            if_c = float(plan.centres_hz[v.cluster]) - args.lo_hz
            rx = tuning_for(if_c, v.dither_hz, lo_hz=args.lo_hz,
                            d_rx_guess=args.d_rx_guess,
                            lo_error_hz=args.lo_error_hz)
            print(f"    {v.index:4d}  sweep {v.sweep:3d}  cluster {v.cluster}"
                  f"  rx_lo {rx/1e6:9.3f} MHz  (dither "
                  f"{v.dither_hz/1e3:+7.1f} kHz)")
        if len(visits) > 12:
            print(f"    ... and {len(visits)-12} more")
        return 0

    print("\n  SAFETY: closed, conducted path only. Do not connect an "
          "antenna.\n")
    header = {"lo_hz": args.lo_hz, "seed": args.seed,
              "centres_hz": list(plan.centres_hz), "points": plan.points,
              "span_hz": plan.span_hz(), "block": args.block,
              "min_hop_s": min_hop_s, "band_extra_s": band_extra_s,
              "period_cycles": args.period_cycles, "fs": args.fs,
              "seconds": args.seconds, "sweeps": args.sweeps,
              "dither_hz": args.dither_khz * 1e3, "visit_seed": args.visit_seed,
              "synthetic": bool(args.synthetic)}
    synthetic = None
    receiver = None
    if args.synthetic:
        rng = np.random.default_rng(4242)
        header |= {"injected_d_rx": args.inject_d_rx,
                   "injected_d_lnb": args.inject_d_lnb,
                   "injected_drift_hz_s": args.inject_drift,
                   "injected_bias_hz": args.inject_bias}
        overhead = PER_CAPTURE_OVERHEAD_S
        synthetic = {
            "d_rx": args.inject_d_rx, "d_lnb": args.inject_d_lnb,
            "drift_hz_s": args.inject_drift, "snr_db": args.snr_db,
            "overhead_s": overhead,
            "t_ref_s": (len(visits) - 1) * (args.seconds + overhead) / 2.0,
            "bias": rng.normal(0.0, args.inject_bias, len(visits))}
        print("  SYNTHETIC: every capture is built in memory. No radio is "
              "opened and nothing is transmitted.")
    else:
        receiver = PlutoReceiver(args.uri, args.fs, args.gain, args.nbuf)

    out = open(args.out, "w") if args.out else None
    try:
        if out is not None:
            out.write(json.dumps({"kind": "header", **header}) + "\n")
            out.flush()
        records = run(plan, visits, lo_hz=args.lo_hz, fs=args.fs,
                      seconds=args.seconds, seed=args.seed,
                      min_hop_s=min_hop_s, block=args.block,
                      jitter=args.jitter, period_cycles=args.period_cycles,
                      band_extra_s=band_extra_s, frame=args.frame,
                      d_rx_guess=args.d_rx_guess,
                      lo_error_hz=args.lo_error_hz, receiver=receiver,
                      synthetic=synthetic, out=out, workdir=args.workdir,
                      estimator=args.estimator, decimate=args.decimate,
                      guard_start_s=args.guard_start_ms / 1e3)
    finally:
        if out is not None:
            out.close()

    kept = sum(1 for r in records if r.trustworthy)
    print(f"\n  {kept} of {len(records)} captures usable"
          + (f"; written to {args.out}" if args.out else ""))
    if not args.fit:
        if args.out:
            print(f"  now fit it:  tools/lever_fit.py {args.out}")
        return 0 if kept == len(records) else 1
    report = lf.analyse(records, lo_hz=args.lo_hz)
    print(lf.format_report(report))
    if args.synthetic:
        print(f"\n  recovered d_rx  {report.res.d_rx*1e6:+.4f} ppm against "
              f"{args.inject_d_rx*1e6:+.4f} injected: error "
              f"{(report.res.d_rx-args.inject_d_rx)*1e6:+.4f} ppm")
        print(f"  recovered d_lnb {report.res.d_lnb*1e6:+.4f} ppm against "
              f"{args.inject_d_lnb*1e6:+.4f} injected: error "
              f"{(report.res.d_lnb-args.inject_d_lnb)*args.lo_hz:+.2f} Hz")
    return 0 if report.trustworthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
