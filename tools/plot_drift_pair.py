#!/usr/bin/env python3
"""Plot two drift runs side by side, each against its own mean.

The two LNBs sit 613 kHz apart, so a shared absolute axis would show two flat
lines a long way from each other and nothing else. Subtracting each run's own
mean puts them on the same scale and makes the thing worth looking at visible:
the points do not scatter around the fitted line, they wander away from it and
back.

    tools/plot_drift_pair.py rx0.jsonl rx1.jsonl --out pair.png
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

BLUE, RED, GREY = "#1b6ca8", "#d1495b", "#8a8f98"


def load(path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    rows, fit = [], None
    for line in open(path):
        r = json.loads(line)
        if r.get("kind") == "monitor":
            rows.append(r)
        elif r.get("kind") == "drift":
            fit = r
    if not rows:
        raise SystemExit(f"{path}: no monitor rows")
    return (np.array([r["t_s"] for r in rows]),
            np.array([r["if_hz"] for r in rows]), fit or {})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs=2, help="two drift JSONL files")
    p.add_argument("--labels", default="RX0  (LNB #1),RX1  (LNB #2)")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    import matplotlib                                          # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt                            # noqa: PLC0415

    labels = a.labels.split(",")
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7), sharex=True, sharey=True)
    for ax, path, label in zip(axes, a.runs, labels):
        t, f, fit = load(path)
        dev = f - f.mean()
        slope = fit.get("slope_hz_s", 0.0)
        err = fit.get("slope_stderr_hz_s", float("nan"))
        rms = fit.get("resid_rms_hz", float(np.std(dev)))
        sigma = abs(slope) / err if err else float("inf")

        # The band the fit says the scatter should live in. Points leaving it
        # and coming back is what "wandering" looks like.
        ax.axhspan(-rms, rms, color=GREY, alpha=0.15,
                   label=f"±1 residual rms ({rms:.0f} Hz)")
        ax.axhline(0, color=GREY, lw=1)
        ax.plot(t, slope * (t - t.mean()), "-", lw=1.6, color=RED,
                label=f"fit {slope:+.3f} ± {err:.3f} Hz/s  ({sigma:.1f}σ)")
        ax.plot(t, dev, "o-", ms=6, lw=0.8, color=BLUE, alpha=0.9,
                label="measured")
        ax.set_ylabel("deviation from run mean (Hz)")
        ax.set_title(label, loc="left", fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    # Headroom so the legend never sits on top of a point. Shared y, so one
    # limit covers both panels.
    lo, hi = axes[0].get_ylim()
    axes[0].set_ylim(lo, hi + 0.45 * (hi - lo))
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Two LNBs, same ADF5355 tone: the frequency wanders, "
                 "it does not ramp", fontsize=12)
    fig.tight_layout()
    fig.savefig(a.out, dpi=140)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
