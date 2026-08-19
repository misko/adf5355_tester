#!/usr/bin/env python3
"""Listen once at a fixed tuning and decode a whole fast ladder from the capture.

Captures IQ to a SigMF-style data file, wraps it in an ArtifactSummary, and runs
pluto-plus-utils' shipped ``freq_ladder`` analyzer over it -- the same code path
as ``pluto analyze``, without needing plutod (which is blocked on this radio
until it has been through the canonical AD9361/2R2T setup).

The receiver is never told when a rung keys. Every rung is identified purely by
burst duration against the published schedule.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/pi/pluto-plus-utils/src")


def capture(sdr, path: Path, seconds: float, fs: float, nbuf: int) -> int:
    """Stream raw int16 I/Q to disk. Capture-only keeps up with real time."""
    sdr.rx_destroy_buffer()
    sdr.rx_buffer_size = nbuf
    sdr.rx()                                    # discard the retune transient
    total = 0
    want = int(seconds * fs)
    with path.open("wb", buffering=1 << 22) as fh:
        while total < want:
            x = np.asarray(sdr.rx())
            out = np.empty(x.size * 2, dtype="<i2")
            out[0::2] = np.real(x).astype("<i2")
            out[1::2] = np.imag(x).astype("<i2")
            fh.write(out.tobytes())
            total += x.size
    return total


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--if-hz", type=float, required=True,
                   help="tune here; the whole ladder must fit in the passband")
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--fs", type=float, default=2.5e6)
    p.add_argument("--nbuf", type=int, default=1 << 16)
    p.add_argument("--gain", type=float, default=40.0)
    p.add_argument("--uri", default="ip:192.168.2.1")
    p.add_argument("--rung-start-hz", type=float, required=True)
    p.add_argument("--rung-stop-hz", type=float, required=True)
    p.add_argument("--rung-count", type=int, required=True)
    p.add_argument("--total-seconds", type=float, required=True)
    p.add_argument("--lo-hz", type=float, default=9.75e9)
    p.add_argument("--frame-size", type=int, default=2048,
                   help="must be well under the shortest burst")
    p.add_argument("--threshold-db", type=float, default=25.0)
    p.add_argument("--search-half-width-hz", type=float, default=400e3)
    p.add_argument("--keep", action="store_true", help="do not delete the capture")
    p.add_argument("--workdir", default="/tmp/claude-1000/-home-pi/"
                                        "1d5cfe25-0154-411c-a9b2-1aebb3a736e0/scratchpad/caps")
    args = p.parse_args()

    from pluto_plus.analysis import FreqLadderAnalyzer
    from pluto_plus.models import ArtifactSummary

    u = args.total_seconds / (args.rung_count * (args.rung_count + 1))
    frame_s = args.frame_size / args.fs
    print(f"ladder: {args.rung_count} rungs, "
          f"{args.rung_start_hz/1e9:.6f}-{args.rung_stop_hz/1e9:.6f} GHz, "
          f"cycle {args.total_seconds:g} s, u = {u*1e3:.1f} ms")
    print(f"listening {args.seconds:g} s at {args.if_hz/1e6:.3f} MHz, "
          f"{args.fs/1e6:g} MS/s  ({args.seconds/args.total_seconds:.1f} cycles)")
    print(f"frame {args.frame_size} = {frame_s*1e3:.2f} ms "
          f"({u/frame_s:.1f} frames in the shortest burst)\n")
    if u / frame_s < 4:
        print("WARNING: shortest burst spans under 4 frames; lower --frame-size")

    import adi
    sdr = adi.Pluto(uri=args.uri)
    sdr.sample_rate = int(args.fs)
    sdr.rx_rf_bandwidth = int(args.fs * 0.8)
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = args.gain
    sdr.rx_lo = int(args.if_hz)

    work = Path(args.workdir); work.mkdir(parents=True, exist_ok=True)
    artifact_id = uuid.uuid4().hex[:16]
    data = work / f"{artifact_id}.sigmf-data"

    t0 = time.monotonic()
    count = capture(sdr, data, args.seconds, args.fs, args.nbuf)
    elapsed = time.monotonic() - t0
    print(f"captured {count} samples in {elapsed:.2f} s "
          f"({count/args.fs/elapsed*100:.1f}% of real time, "
          f"{data.stat().st_size/1e6:.0f} MB)")
    if count / args.fs / elapsed < 0.98:
        print("WARNING: capture fell behind real time; durations may be wrong")

    digest = hashlib.sha256()
    with data.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)

    artifact = ArtifactSummary(
        artifact_id=artifact_id, radio_id="local", created_at=datetime.now(timezone.utc),
        path=str(work), sample_count=count, receiver_count=1,
        sample_rate_hz=args.fs, center_frequency_hz=args.if_hz,
        sha256=digest.hexdigest(), label="freq-ladder listen")

    result = FreqLadderAnalyzer().run(artifact, {
        "rung_start_hz": args.rung_start_hz, "rung_stop_hz": args.rung_stop_hz,
        "rung_count": args.rung_count, "total_seconds": args.total_seconds,
        "lo_hz": args.lo_hz, "frame_size": args.frame_size,
        "threshold_db": args.threshold_db,
        "search_half_width_hz": args.search_half_width_hz,
    })

    bursts = result.get("bursts", [])
    ident = [b for b in bursts if b.get("rung") is not None]
    print(f"\nbursts detected: {len(bursts)}   identified: {len(ident)}")
    seen: dict[int, list] = {}
    for b in ident:
        seen.setdefault(b["rung"], []).append(b)
    for rung in sorted(seen):
        rows = seen[rung]
        dfs = [r["frequency_error_hz"] for r in rows if r.get("frequency_error_hz") is not None]
        if not dfs:
            continue
        print(f"  rung {rung:2d}  n={len(rows):3d}  "
              f"burst {np.median([r['duration_seconds'] for r in rows])*1e3:7.2f} ms  "
              f"Df {np.median(dfs)/1e3:+9.3f} kHz  (sd {np.std(dfs):6.1f} Hz)")
    for key in ("identification", "fit"):
        if result.get(key):
            print(f"\n{key}:")
            for k, v in result[key].items():
                print(f"    {k}: {v}")
    if not args.keep:
        data.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
