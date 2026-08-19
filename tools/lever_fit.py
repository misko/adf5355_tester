#!/usr/bin/env python3
"""Combine single-cluster measurements into d_rx and d_lnb, separately.

What one cluster can and cannot do
----------------------------------
A capture of one narrow cluster measures the total offset superbly and cannot
take it apart. Write d for a fractional frequency error; the receiver reports

    Df(f_IF) = -d_rx * f_IF  -  d_lnb * f_LO_nom  -  b(rx_lo)  -  g * (t - t0)
               \___________/   \_______________/   \________/   \___________/
                scales with     constant in Hz      tuning       LNB warming
                the IF                              bias         up

and across one 0.8 MHz cluster the first term moves by 7 Hz for a 9 ppm clock.
That is why the number today is a *blend*: precise, and not separable.

The lever arm
-------------
The LNB low band gives 0.95 to 2.15 GHz of IF. Measuring the same precise local
offset at clusters spread across that range makes the first term move by
10.7 kHz between the ends, which is enormous, and the fit separates the two.
The receiver hears about 2 MHz at once, so clusters are visited by retuning --
and that is where the difficulty is.

The tuning bias is the whole problem
------------------------------------
The same unmoving tone, measured from eight different rx_lo settings, spanned
362 Hz on this hardware. That bias is additive and constant across one capture:
it cancels out of anything measured *inside* a capture and it does not cancel
at all between captures at different clusters -- which is exactly the contrast
the lever arm is made of. Left alone it would put roughly
sqrt(2)*127/1.2e9 = 0.15 ppm on d_rx, only twice better than the +/-0.3 ppm the
single-cluster method already quotes, and the whole exercise would be pointless.

Four things are done about it, in order of importance:

1. **Dither the tuning.** Every capture is taken at an rx_lo drawn at random
   from a window several hundred kHz wide. Whether the bias is a deterministic
   function of the requested LO or a fresh draw on every retune -- and it
   cannot be told which from the outside -- dithering makes it an independent
   draw per capture, so N captures of one cluster average it down as
   sqrt(N). Without the dither, re-tuning to the same number would reproduce
   the same bias and averaging would buy nothing.
2. **Many short captures rather than few long ones.** The bias per capture does
   not care how long the capture is, and it is 3000x larger than the
   estimator's noise. Precision therefore tracks the NUMBER of captures, not
   the total listening time. Two seconds is enough to decode; anything longer
   is spent buying precision that is already free.
3. **Measure it rather than assume it.** The scatter of one cluster's captures
   about the fit IS the tuning bias, so its size is estimated from the run and
   fed into the answer's uncertainty. And the variogram below -- that scatter
   as a function of how far apart the two tunings were -- says whether the
   dither is actually decorrelating it. If the bias were smooth over the dither
   window, close tunings would agree and distant ones would not; a flat
   variogram is the evidence that the averaging is real.
4. **Resample, do not trust a covariance.** Every uncertainty quoted here comes
   from refitting resampled data, because the covariance of the fit knows only
   the errors it was told about.

Drift, and why the answer for d_rx is fitted differently from d_lnb
-------------------------------------------------------------------
The LNB LO walks about 4.5 Hz/s, so over a ten-minute run it moves by kilohertz
-- far more than the tuning bias. Captures are therefore taken in SWEEPS: one
capture of every cluster, in a fresh random order each time. Then

* **d_rx** is fitted with a free offset per sweep. That absorbs the LNB's LO
  error, its drift, and any other purely time-dependent common term, of ANY
  shape, exactly. d_rx survives because it is the only term that varies with
  IF, and every sweep contains every cluster. This is the deliverable, and it
  is the one number here that no drift model can corrupt.
* **d_lnb** cannot be fitted that way: it IS a constant in Hz, so free per-sweep
  offsets would swallow it. It is fitted instead against an explicit smooth
  drift in time, and its uncertainty is dominated by that model choice rather
  than by noise. It is also only meaningful at an instant, since the thing
  being measured is moving at 4.5 Hz/s; the report says which instant.

The narrow lever, as an independent check
-----------------------------------------
Each capture also yields a slope across its own cluster, over 0.8 MHz instead
of 1.2 GHz. A thousand times less lever arm, but no tuning bias at all, and no
dependence on the drift model. The two estimates of d_rx are vulnerable to
completely different things, so they are both reported and compared. They agree
or something is wrong, and that is worth more than either alone.

    tools/lever_fit.py run.jsonl              # fit a run taken on hardware
    tools/lever_fit.py --self-test            # no radio, no capture, known answer
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adf5355.hopper import SplitMix64  # noqa: E402

DEFAULT_LO_HZ = 9.75e9
DEFAULT_SWEEPS = 25
DEFAULT_DITHER_HZ = 450_000.0
DITHER_QUANTUM_HZ = 1_000          # rx_lo is programmed in whole Hz anyway
DEFAULT_DRIFT_ORDER = 2
DEFAULT_BOOTSTRAP = 400
# The narrow (within-capture) lever is a WITNESS, not a contributor, and the
# default reflects that. It is far tighter than the wide lever, so folding it
# in would hand it the answer -- and the only evidence that it is unbiased is
# that it agrees with the wide lever, which is to say, evidence good to about
# the wide lever's own precision. Quoting an uncertainty smaller than that
# would be quoting an assumption. '--narrow combine' does it anyway, for
# someone who accepts that a slope measured inside one capture cannot be
# tilted by anything but the sample clock.
NARROW_MODES = ("check", "combine", "off")
NARROW_AGREE_SIGMA = 3.0
# How many sigma the two halves of a run may differ by before the answer is
# called unsteady, and the SCALE that converts the quoted uncertainty into the
# sigma of that difference. Each half is fitted from half the sweeps, so its
# own standard error is sqrt(2) times the whole run's; the two halves are
# independent, so their difference scatters by sqrt(2)*sqrt(2) = 2 times the
# whole run's. Using sqrt(2) here instead -- as if the halves were the run --
# turns a nominal 3-sigma test into a 2.1-sigma one and fails roughly one
# blameless run in thirty. Measured directly: over 120 synthetic runs at 25
# sweeps the split-half difference scattered 1.97x the quoted sigma.
SPLIT_HALF_SIGMA = 3.0
SPLIT_HALF_SCALE = 2.0
# Below this many sweeps a resample over sweeps has too few blocks to be
# believed. Measured on independently synthesised runs, a nominal 95% interval
# covered 93% at 25 sweeps and 93% at 8, but only 86% at 5 -- and at 5 the
# d_lnb interval came apart entirely (quoted sigma 4.5x its true scatter) and
# the variogram raised a false alarm on 13 runs in 100, against 1 in 100 at
# eight. Eight is where everything comes back into line.
MIN_SWEEPS = 8
# Below this a residual scatter is float dust rather than a measurement.
NOTHING_TO_TEST_HZ = 1e-6
# A variogram whose near bin sits this far under its far bin means the tuning
# bias is SMOOTH across the dither window -- nearby tunings agree -- so the
# dither is not decorrelating it and the averaging is not real.
VARIOGRAM_RATIO = 0.5


# ---------------------------------------------------------------------------
# what order to visit the clusters in
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PlannedVisit:
    index: int
    sweep: int
    cluster: int
    dither_hz: float


def plan_visits(seed: int, clusters: int, sweeps: int,
                dither_hz: float = DEFAULT_DITHER_HZ,
                quantum_hz: int = DITHER_QUANTUM_HZ) -> list[PlannedVisit]:
    """The order the operator's receiver visits clusters in, and at what tuning.

    One sweep is one capture of every cluster, in an order redrawn every sweep.
    Two properties are being bought, and both matter more than they look:

    * **Every sweep holds every cluster.** That is what lets d_rx be fitted
      with a free offset per sweep, which makes it immune to drift of any
      shape rather than merely to the shape someone modelled.
    * **The order is a cyclic Latin square, redrawn every ``clusters``
      sweeps.** Over any run of that many sweeps each cluster occupies each
      POSITION in the sweep exactly once. A fixed order would put each cluster
      at a fixed phase of the sweep, so the LNB's drift within a sweep -- about
      200 Hz across a minute -- would land on the clusters in a fixed pattern,
      which is precisely the monotonic-ladder mistake one level up. A plain
      reshuffle would leave that imbalance at 1/sqrt(sweeps), worth 0.01 to
      0.02 ppm of d_rx, which is the size of the whole answer. A Latin square
      leaves none of it.

    The tuning dither is drawn per capture from the same seeded generator, so
    the whole run is reproducible from one integer.
    """
    if clusters < 2:
        raise ValueError("a lever arm needs at least 2 clusters")
    if sweeps < 1:
        raise ValueError("need at least one sweep")
    if dither_hz < 0:
        raise ValueError("dither must not be negative")
    rng = SplitMix64(seed)
    out: list[PlannedVisit] = []
    last = -1
    base: list[int] = []
    for s in range(sweeps):
        if s % clusters == 0:                    # a fresh Latin square
            for _ in range(64):
                base = list(range(clusters))
                for i in range(clusters - 1, 0, -1):
                    j = rng.below(i + 1)
                    base[i], base[j] = base[j], base[i]
                if base[0] != last:
                    break
        # Row s of a cyclic Latin square on that permutation. Over any run of
        # ``clusters`` sweeps every cluster then occupies every POSITION in
        # the sweep exactly once -- which is what makes the LNB's drift within
        # a sweep cancel out of the lever arm exactly, instead of aliasing on
        # to it by however much a random order happened to be unbalanced. A
        # plain reshuffle leaves that imbalance at about 1/sqrt(sweeps), and
        # at 4.5 Hz/s over a ten-minute run that is worth 0.01 to 0.02 ppm of
        # d_rx -- the same size as the answer's whole uncertainty.
        k = s % clusters
        order = [base[(i + k) % clusters] for i in range(clusters)]
        last = order[-1]
        for c in order:
            d = (rng.uniform() * 2.0 - 1.0) * dither_hz
            if quantum_hz > 0:
                d = round(d / quantum_hz) * quantum_hz
            out.append(PlannedVisit(len(out), s, c, float(d)))
    return out


# ---------------------------------------------------------------------------
# the record one capture contributes
# ---------------------------------------------------------------------------
@dataclass
class CaptureRecord:
    """One decoded capture, reduced to what the fit consumes."""

    index: int
    sweep: int
    cluster: int
    cluster_centre_hz: float
    rx_lo_hz: float
    dither_hz: float
    t_abs_s: float
    mean_if_hz: float
    mean_error_hz: float
    mean_error_stderr_hz: float
    slope: float
    slope_stderr: float
    chi2_scale: float = float("nan")
    recovered: int = 0
    points: int = 0
    comb_sharpness: float = float("nan")
    epoch_sigma: float = float("nan")
    visits: int = 0
    drift_hz_s: float = float("nan")
    settling_hz: float = float("nan")
    excess_scatter: float = float("nan")
    trustworthy: bool = True
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_decode(cls, result, *, index: int, sweep: int, rx_lo_hz: float,
                    dither_hz: float) -> "CaptureRecord":
        return cls(
            index=index, sweep=sweep, cluster=result.cluster,
            cluster_centre_hz=result.cluster_centre_hz, rx_lo_hz=rx_lo_hz,
            dither_hz=dither_hz, t_abs_s=result.t_abs_s,
            mean_if_hz=result.mean_if_hz, mean_error_hz=result.mean_error_hz,
            mean_error_stderr_hz=result.mean_error_stderr_hz,
            slope=result.slope, slope_stderr=result.slope_stderr,
            chi2_scale=result.chi2_scale, recovered=result.recovered,
            points=result.points, comb_sharpness=result.comb_sharpness,
            epoch_sigma=result.epoch_sigma, visits=result.visits,
            drift_hz_s=result.drift_hz_s, settling_hz=result.settling_hz,
            excess_scatter=result.excess_scatter,
            trustworthy=result.trustworthy, warnings=list(result.warnings))


def write_records(path, header: dict, records: list[CaptureRecord]) -> None:
    with open(path, "w") as fh:
        fh.write(json.dumps({"kind": "header", **header}) + "\n")
        for r in records:
            fh.write(json.dumps({"kind": "capture", **asdict(r)}) + "\n")


def read_records(path) -> tuple[dict, list[CaptureRecord]]:
    header: dict = {}
    records: list[CaptureRecord] = []
    fields = set(CaptureRecord.__dataclass_fields__)
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("kind") == "header":
                header = {k: v for k, v in obj.items() if k != "kind"}
            elif obj.get("kind") == "capture":
                records.append(CaptureRecord(
                    **{k: v for k, v in obj.items() if k in fields}))
    return header, records


# ---------------------------------------------------------------------------
# the fit
# ---------------------------------------------------------------------------
@dataclass
class Dataset:
    """A run as columns, which is what resampling wants it to be.

    Resampling means refitting hundreds of times, so the records are unpacked
    into arrays once and every resample after that is index arithmetic. It also
    keeps the fit honest about what it uses: exactly these nine columns, and
    nothing else in the record.
    """

    x: np.ndarray            # mean_if_hz: the lever arm
    y: np.ndarray            # mean_error_hz: the capture's offset at x
    sv: np.ndarray           # the estimator's standard error on y
    t: np.ndarray            # absolute time of the capture's middle
    sweep: np.ndarray        # which sweep it belongs to
    cluster: np.ndarray
    slope: np.ndarray        # the within-capture slope, the narrow lever
    slope_se: np.ndarray
    rx_lo: np.ndarray        # what the tuning actually was, for the variogram

    @classmethod
    def from_records(cls, records: list[CaptureRecord]) -> "Dataset":
        def col(name, default=np.nan):
            return np.array([getattr(r, name) for r in records], dtype=float)
        sv = col("mean_error_stderr_hz")
        return cls(
            x=col("mean_if_hz"), y=col("mean_error_hz"),
            sv=np.where(np.isfinite(sv) & (sv > 0), sv, 0.0),
            t=col("t_abs_s"),
            sweep=np.array([r.sweep for r in records], dtype=int),
            cluster=np.array([r.cluster for r in records], dtype=int),
            slope=col("slope"), slope_se=col("slope_stderr"),
            rx_lo=col("rx_lo_hz"))

    def __len__(self) -> int:
        return self.x.size

    def take(self, idx: np.ndarray,
             sweep_labels: np.ndarray | None = None) -> "Dataset":
        return Dataset(
            x=self.x[idx], y=self.y[idx], sv=self.sv[idx], t=self.t[idx],
            sweep=self.sweep[idx] if sweep_labels is None else sweep_labels,
            cluster=self.cluster[idx], slope=self.slope[idx],
            slope_se=self.slope_se[idx], rx_lo=self.rx_lo[idx])

    def by_sweep(self, sweeps) -> "Dataset":
        """Rebuild from a list of sweeps, which may repeat.

        A sweep drawn twice becomes two DISTINCT sweeps. For the SLOPE that
        makes no numerical difference -- two identical copies centred on one
        shared mean and centred on two separate means come to the same thing,
        and that was checked rather than assumed. What it does buy is an
        honest parameter count: the bootstrap fit then has as many free
        offsets as it has sweeps, so its degrees of freedom, and through them
        the fitted tuning-bias variance component, match the model that is
        actually being fitted instead of a smaller one.
        """
        idx, labels = [], []
        for label, s in enumerate(sweeps):
            where = np.flatnonzero(self.sweep == s)
            idx.append(where)
            labels.append(np.full(where.size, label))
        if not idx:
            raise ValueError("no sweeps selected")
        return self.take(np.concatenate(idx), np.concatenate(labels))


@dataclass
class WideFit:
    """One weighted least-squares fit of the offsets against IF and time."""

    slope: float                 # d(Df)/d(f_IF), dimensionless
    slope_stderr: float          # formal, from the covariance -- see below
    intercept_hz: float          # Df at f_IF = 0 and t = t_ref, in Hz
    intercept_stderr_hz: float
    drift_hz_s: float            # first-order term of the fitted time trend
    sigma_bias_hz: float         # the tuning bias, estimated from the scatter
    chi2: float
    dof: int
    model: str
    npoints: int
    t_ref_s: float
    residuals: np.ndarray = field(default_factory=lambda: np.zeros(0))


def _weights(sigma_v: np.ndarray, sigma_bias: float) -> np.ndarray:
    """Inverse variance, floored so a noiseless point cannot become infinite.

    Weights only matter up to a common scale, so when nothing is known to
    differ in quality -- a synthetic run with no noise injected at all -- the
    honest answer is that every capture counts the same, not that every
    capture counts infinitely.
    """
    var = np.asarray(sigma_v, dtype=float) ** 2 + sigma_bias ** 2
    var = np.where(np.isfinite(var) & (var > 0), var, 0.0)
    top = float(var.max()) if var.size else 0.0
    if top <= 0.0:
        return np.ones_like(var)
    return 1.0 / np.maximum(var, top * 1e-12)


def _solve(design: np.ndarray, y: np.ndarray, w: np.ndarray):
    """Weighted normal equations. Small, well conditioned, and refit hundreds
    of times by the resample, so an SVD per call is not worth its cost."""
    a = design * np.sqrt(w)[:, None]
    ata = a.T @ a
    try:
        cov = np.linalg.inv(ata)
    except np.linalg.LinAlgError:
        n = design.shape[1]
        return np.zeros(n), np.full((n, n), np.nan), y.copy()
    coef = cov @ (design.T @ (w * y))
    return coef, cov, y - design @ coef


def _sweep_means(values: np.ndarray, w: np.ndarray, inverse: np.ndarray,
                 nsweeps: int) -> np.ndarray:
    """Weighted mean of ``values`` within each sweep, broadcast back per row."""
    totals = np.bincount(inverse, weights=w * values, minlength=nsweeps)
    norm = np.bincount(inverse, weights=w, minlength=nsweeps)
    return (totals / np.where(norm > 0, norm, 1.0))[inverse]


def _fit_sweep_model(data: Dataset, w: np.ndarray, inverse: np.ndarray,
                     nsweeps: int, t_ref: float, xs: float, ts: float):
    """The free-offset-per-sweep fit, in closed form, plus a drift inside it.

    Sweeping out one free constant per sweep is exactly a within-sweep
    centring, so there is nothing to invert: subtract each sweep's weighted
    mean from x, from t and from y, and what is left is a two-parameter
    regression. Having it in three lines rather than in a 26-column design
    matrix is not only faster under a bootstrap -- it is what the model MEANS,
    and it makes the claim "d_rx cannot be touched by any drift that is
    constant within a sweep" checkable by eye.

    The second parameter is the drift WITHIN a sweep, and it is not optional.
    A sweep takes about a minute and the LNB moves 4.5 Hz/s, so the captures
    inside one sweep are hundreds of hertz apart in time alone. A free
    constant per sweep does not touch that; it aliases on to whichever cluster
    happened to be visited early against late. The visit plan is balanced so
    that imbalance is zero by construction, and this term removes whatever
    imbalance is left over anyway -- two independent defences against the same
    0.02 ppm, because it is the same size as the answer.
    """
    dx = (data.x - _sweep_means(data.x, w, inverse, nsweeps)) / xs
    dt = (data.t - t_ref - _sweep_means(data.t - t_ref, w, inverse,
                                        nsweeps)) / ts
    dy = data.y - _sweep_means(data.y, w, inverse, nsweeps)
    sxx = float((w * dx * dx).sum())
    stt = float((w * dt * dt).sum())
    sxt = float((w * dx * dt).sum())
    sxy = float((w * dx * dy).sum())
    sty = float((w * dt * dy).sum())
    det = sxx * stt - sxt * sxt
    if sxx <= 0 or not np.isfinite(det):
        return float("nan"), float("inf"), data.y - data.y.mean()
    if stt <= 0 or det <= 0:                   # no time spread inside a sweep
        slope = sxy / sxx
        return slope / xs, float(np.sqrt(1.0 / sxx)) / xs, dy - slope * dx
    slope = (stt * sxy - sxt * sty) / det
    drift = (sxx * sty - sxt * sxy) / det
    return (slope / xs, float(np.sqrt(stt / det)) / xs,
            dy - slope * dx - drift * dt)


def _variance_component(resid: np.ndarray, sv: np.ndarray, dof: int) -> float:
    """The extra scatter that makes the weighted chi-square land on its dof.

    Method of moments, solved by bisection because chi2(sigma) is monotone.
    This is where the tuning bias enters the arithmetic: the decoder's own
    standard errors are fractions of a hertz and the residuals are hundreds,
    and the difference is not noise the model forgot -- it is the bias, and it
    belongs in the weights and in the answer's uncertainty.
    """
    if dof < 1:
        return 0.0
    r2 = np.asarray(resid, dtype=float) ** 2
    s2 = np.asarray(sv, dtype=float) ** 2
    s2 = np.where(np.isfinite(s2) & (s2 >= 0), s2, 0.0)
    total = float(r2.sum())
    if total <= 0.0:
        return 0.0                       # nothing scattered; nothing to add
    if not np.any(s2 > 0):
        return math.sqrt(total / dof)    # every capture equally uncertain
    if float(s2.max() - s2.min()) <= 0.0:
        # Every capture equally precise, which is the usual case here: the
        # solution is then a subtraction rather than a search.
        return math.sqrt(max(total / dof - float(s2[0]), 0.0))
    lo, hi = 0.0, max(float(r2.mean()), 1e-30)
    for _ in range(80):                  # widen until the root is bracketed
        if float((r2 / (s2 + hi)).sum()) <= dof:
            break
        hi *= 2.0
    for _ in range(40):                  # halvings of a variance, in v = s^2
        mid = 0.5 * (lo + hi)
        if float((r2 / (s2 + mid)).sum()) > dof:
            lo = mid
        else:
            hi = mid
    return math.sqrt(0.5 * (lo + hi))


def _design(data: Dataset, drift_order: int, t_ref: float, x0: float,
            xs: float, ts: float) -> np.ndarray:
    """Columns of the 'poly' model: the IF slope, a constant, and a time trend.

    Only this model needs a design matrix. The 'sweep' model is a within-sweep
    centring, which :func:`_fit_sweep_model` does in three lines without ever
    building the twenty-odd dummy columns it would otherwise need.
    """
    tt = (data.t - t_ref) / ts
    return np.column_stack([(data.x - x0) / xs, np.ones_like(data.x)]
                           + [tt ** j for j in range(1, drift_order + 1)])


def fit_wide(data: Dataset, *, model: str = "sweep",
             drift_order: int = DEFAULT_DRIFT_ORDER,
             t_ref_s: float | None = None,
             sigma_bias_hz: float | None = None,
             iterations: int = 3) -> WideFit:
    """Fit offset against IF across clusters, with a tuning-bias variance term.

    The per-capture standard error the decoder reports is the *estimator's*
    error, which here is a fraction of a hertz. The scatter that actually
    matters is the tuning bias, which is a hundred times larger and is not in
    that number at all. So the weights carry both -- ``1/(sigma_v^2 +
    sigma_b^2)`` -- and ``sigma_b`` is estimated from this run's own residuals.

    That makes the weighting honest. It does NOT make the covariance-derived
    standard error honest, because sigma_b came out of the same residuals and
    the model itself might be wrong; the numbers that get quoted come from
    :func:`resample`.
    """
    n = len(data)
    if n < 3:
        raise ValueError("need at least three captures to fit anything")
    if t_ref_s is None:
        t_ref_s = float(data.t.mean())
    x0, xs = float(data.x.mean()), max(float(data.x.std()), 1.0)
    ts = max(float(np.abs(data.t - t_ref_s).max()), 1.0)
    if model == "sweep":
        design = None
        nparam = int(np.unique(data.sweep).size) + 2
    elif model == "poly":
        design = _design(data, drift_order, t_ref_s, x0, xs, ts)
        nparam = design.shape[1]
    else:
        raise ValueError(f"unknown model {model!r}; use 'sweep' or 'poly'")
    dof = n - nparam
    if dof < 1:
        raise ValueError(f"model '{model}' has no degrees of freedom left "
                         f"({n} captures, {nparam} parameters); "
                         f"take more sweeps")

    inverse = None
    if model == "sweep":
        uniq, inverse = np.unique(data.sweep, return_inverse=True)
        nsweeps = uniq.size

    sigma_b = 0.0 if sigma_bias_hz is None else float(sigma_bias_hz)
    for _ in range(1 if sigma_bias_hz is not None else iterations):
        w = _weights(data.sv, sigma_b)
        if model == "sweep":
            _, _, resid = _fit_sweep_model(data, w, inverse, nsweeps,
                                           t_ref_s, xs, ts)
        else:
            _, _, resid = _solve(design, data.y, w)
        if sigma_bias_hz is None:
            sigma_b = _variance_component(resid, data.sv, dof)
    w = _weights(data.sv, sigma_b)
    if model == "sweep":
        slope, slope_se, resid = _fit_sweep_model(data, w, inverse, nsweeps,
                                                  t_ref_s, xs, ts)
        coef = cov = None
    else:
        coef, cov, resid = _solve(design, data.y, w)
        slope = float(coef[0]) / xs
        slope_se = float(np.sqrt(abs(cov[0, 0]))) / xs
    chi2 = float((w * resid ** 2).sum())

    if model == "poly":
        intercept = float(coef[1]) - float(coef[0]) * x0 / xs
        k = np.array([-x0 / xs, 1.0])
        sub = cov[np.ix_([0, 1], [0, 1])]
        intercept_se = float(np.sqrt(abs(k @ sub @ k)))
        drift = float(coef[2]) / ts if drift_order >= 1 else 0.0
    else:
        intercept = intercept_se = drift = float("nan")
    return WideFit(slope=slope, slope_stderr=slope_se, intercept_hz=intercept,
                   intercept_stderr_hz=intercept_se, drift_hz_s=drift,
                   sigma_bias_hz=float(sigma_b), chi2=chi2, dof=int(dof),
                   model=model, npoints=n, t_ref_s=float(t_ref_s),
                   residuals=resid)


def fit_narrow(data: Dataset, iterations: int = 3
               ) -> tuple[float, float, float]:
    """Combine the within-capture slopes. Returns (slope, stderr, excess sd).

    Every capture measures the same -d_rx/(1+d_rx) over its own cluster's
    sub-megahertz, with no tuning bias and no drift model involved. They are
    combined by inverse variance with a variance component of their own, so a
    run whose slopes scatter wider than their error bars says so instead of
    averaging a disagreement into a confident number. That excess IS the
    interesting quantity: it is the only handle this design has on whether
    anything in the receiver depends on baseband frequency.
    """
    s, e = data.slope, data.slope_se
    ok = np.isfinite(s) & np.isfinite(e) & (e > 0)
    s, e = s[ok], e[ok]
    if s.size < 3:
        return float("nan"), float("nan"), float("nan")
    tau = 0.0
    mean = float(np.mean(s))
    for _ in range(iterations):
        w = 1.0 / (e ** 2 + tau ** 2)
        mean = float((w * s).sum() / w.sum())
        tau = _variance_component(s - mean, e, s.size - 1)
    w = 1.0 / (e ** 2 + tau ** 2)
    mean = float((w * s).sum() / w.sum())
    return mean, float(np.sqrt(1.0 / w.sum())), float(tau)


# ---------------------------------------------------------------------------
# from fitted coefficients to the two clock errors
# ---------------------------------------------------------------------------
def d_rx_from_slope(slope: float) -> float:
    """Exact, not linearised: Df/f_IF is -d_rx/(1+d_rx), so invert it."""
    return -slope / (1.0 + slope)


def d_lnb_from_intercept(intercept_hz: float, slope: float,
                         lo_hz: float) -> float:
    """d_lnb at the fit's reference instant. (1+d_rx) = 1/(1+slope)."""
    return -intercept_hz / (lo_hz * (1.0 + slope))


def drift_from_coefficient(k_hz_s: float, slope: float) -> float:
    return -k_hz_s / (1.0 + slope)


@dataclass
class Estimate:
    """One complete answer from one set of captures."""

    slope_wide: float
    slope_wide_stderr: float
    slope_narrow: float
    slope_narrow_stderr: float
    slope_used: float
    intercept_hz: float
    intercept_stderr_hz: float
    drift_coef_hz_s: float
    sigma_bias_hz: float
    narrow_tau: float
    chi2: float
    dof: int

    @property
    def d_rx(self) -> float:
        return d_rx_from_slope(self.slope_used)

    @property
    def d_rx_wide(self) -> float:
        return d_rx_from_slope(self.slope_wide)

    @property
    def d_rx_narrow(self) -> float:
        return d_rx_from_slope(self.slope_narrow)

    def d_lnb(self, lo_hz: float) -> float:
        return d_lnb_from_intercept(self.intercept_hz, self.slope_used, lo_hz)

    @property
    def drift_hz_s(self) -> float:
        return drift_from_coefficient(self.drift_coef_hz_s, self.slope_used)


def estimate(data: Dataset, *, drift_order: int = DEFAULT_DRIFT_ORDER,
             t_ref_s: float | None = None,
             use_narrow: bool = False) -> Estimate:
    """Both fits and, if asked, their combination -- one answer.

    ``use_narrow`` is off by default and that is a considered position, not
    laziness. See the note on NARROW_MODES: combining is only defensible if the
    within-capture slope is assumed unbiased, and the run cannot show that to
    better than the wide lever's own precision.
    """
    wide = fit_wide(data, model="sweep", t_ref_s=t_ref_s)
    poly = fit_wide(data, model="poly", drift_order=drift_order,
                    t_ref_s=t_ref_s)
    m_narrow, se_narrow, tau = fit_narrow(data)
    used = wide.slope
    if use_narrow and np.isfinite(m_narrow) and se_narrow > 0 \
            and wide.slope_stderr > 0:
        wn, ww = 1.0 / se_narrow ** 2, 1.0 / wide.slope_stderr ** 2
        used = (ww * wide.slope + wn * m_narrow) / (ww + wn)
    return Estimate(
        slope_wide=wide.slope, slope_wide_stderr=wide.slope_stderr,
        slope_narrow=m_narrow, slope_narrow_stderr=se_narrow,
        slope_used=used, intercept_hz=poly.intercept_hz,
        intercept_stderr_hz=poly.intercept_stderr_hz,
        drift_coef_hz_s=poly.drift_hz_s, sigma_bias_hz=wide.sigma_bias_hz,
        narrow_tau=tau, chi2=wide.chi2, dof=wide.dof)


# ---------------------------------------------------------------------------
# uncertainty, by refitting rather than by believing a covariance
# ---------------------------------------------------------------------------
@dataclass
class Resample:
    d_rx: float
    d_rx_stderr: float
    d_rx_ci: tuple[float, float]
    d_lnb: float
    d_lnb_stderr: float
    d_lnb_ci: tuple[float, float]
    drift_hz_s: float
    drift_stderr_hz_s: float
    d_rx_jackknife_stderr: float
    d_lnb_jackknife_stderr: float
    d_rx_bootstrap_stderr: float
    draws: int
    failures: int


def resample(data: Dataset, lo_hz: float, *,
             drift_order: int = DEFAULT_DRIFT_ORDER,
             use_narrow: bool = False, draws: int = DEFAULT_BOOTSTRAP,
             seed: int = 20260819) -> Resample:
    """Bootstrap and jackknife over SWEEPS, and quote the wider of the two.

    The sweep is the resampling unit because it is the exchangeable one: every
    sweep holds every cluster, its captures share a moment in the LNB's warm-up
    and nothing else, and each carries its own independent draw of the tuning
    bias. Resampling individual captures would break the sweeps up and let the
    drift leak in; resampling clusters would destroy the lever arm the fit is
    made of.

    Both a bootstrap and a leave-one-sweep-out jackknife are run because they
    fail differently -- the bootstrap is optimistic with few sweeps, the
    jackknife is poor with a rough statistic -- and the larger of the two
    standard errors is quoted. The covariance standard error is computed as
    well, and printed beside these, only so the gap is visible.
    """
    t_ref = float(data.t.mean())
    base = estimate(data, drift_order=drift_order, t_ref_s=t_ref,
                    use_narrow=use_narrow)
    sweeps = np.unique(data.sweep)
    rng = np.random.default_rng(seed)

    def one(subset: Dataset):
        e = estimate(subset, drift_order=drift_order, t_ref_s=t_ref,
                     use_narrow=use_narrow)
        return e.d_rx, e.d_lnb(lo_hz), e.drift_hz_s

    boot, failures = [], 0
    for _ in range(draws):
        picked = sweeps[rng.integers(0, sweeps.size, sweeps.size)]
        try:
            boot.append(one(data.by_sweep(picked)))
        except (ValueError, np.linalg.LinAlgError):
            failures += 1
    arr = np.array(boot) if boot else np.zeros((0, 3))

    jack = []
    for drop in sweeps:
        try:
            jack.append(one(data.by_sweep(sweeps[sweeps != drop])))
        except (ValueError, np.linalg.LinAlgError):
            pass
    jarr = np.array(jack) if jack else np.zeros((0, 3))
    n = len(jarr)
    jse = (np.sqrt((n - 1) / n * ((jarr - jarr.mean(axis=0)) ** 2).sum(axis=0))
           if n > 1 else np.full(3, np.nan))

    def pick(i: int) -> tuple[float, tuple[float, float]]:
        if arr.shape[0] < 8:
            return float("nan"), (float("nan"), float("nan"))
        col = arr[:, i]
        return (float(col.std(ddof=1)),
                (float(np.percentile(col, 2.5)),
                 float(np.percentile(col, 97.5))))

    se_rx, ci_rx = pick(0)
    se_lnb, ci_lnb = pick(1)
    se_drift, _ = pick(2)
    return Resample(
        d_rx=base.d_rx, d_rx_stderr=float(np.nanmax([se_rx, jse[0]])),
        d_rx_ci=ci_rx, d_lnb=base.d_lnb(lo_hz),
        d_lnb_stderr=float(np.nanmax([se_lnb, jse[1]])), d_lnb_ci=ci_lnb,
        drift_hz_s=base.drift_hz_s,
        drift_stderr_hz_s=float(np.nanmax([se_drift, jse[2]])),
        d_rx_jackknife_stderr=float(jse[0]),
        d_lnb_jackknife_stderr=float(jse[1]), d_rx_bootstrap_stderr=se_rx,
        draws=int(arr.shape[0]), failures=failures)


# ---------------------------------------------------------------------------
# diagnostics: the checks that can actually fail
# ---------------------------------------------------------------------------
def variogram(data: Dataset, residuals: np.ndarray,
              bins: int = 3) -> list[tuple[float, float, int]]:
    """Is the tuning bias really decorrelated by the dither?

    For every pair of captures of the SAME cluster, half the squared difference
    of their residuals, against how far apart the two tunings were. If the bias
    is rough on the scale of the dither window -- which is what the bench saw,
    362 Hz across 3 MHz of tuning -- then near pairs disagree as much as far
    pairs and this comes out flat at sigma_b^2. If instead the bias were smooth,
    near pairs would agree, the near bin would sit low, and the dither would not
    be buying the sqrt(N) it is being credited with. Returns
    (mean separation, gamma, pairs) per bin.
    """
    n = len(data)
    a, b = np.triu_indices(n, k=1)
    same = data.cluster[a] == data.cluster[b]
    a, b = a[same], b[same]
    if a.size < 2 * bins:
        return []
    h = np.abs(data.rx_lo[a] - data.rx_lo[b])
    g = 0.5 * (residuals[a] - residuals[b]) ** 2
    edges = np.quantile(h, np.linspace(0, 1, bins + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    out = []
    for i in range(bins):
        m = (h >= edges[i]) & (h < edges[i + 1])
        if m.any():
            out.append((float(h[m].mean()), float(g[m].mean()), int(m.sum())))
    return out


def split_half(data: Dataset, drift_order: int,
               use_narrow: bool) -> tuple[float, float, float]:
    """d_rx from the first half of the sweeps against the second half.

    The Pluto's own reference moves with temperature, and this run takes
    minutes. Nothing inside a single fit can see that -- the fit assumes one
    d_rx -- so it is checked from outside. A difference well outside the quoted
    uncertainty means the answer is an average over something that is moving,
    which is a fact about the hardware rather than a bug in the fit, but it has
    to be known before the number is written down.

    What "well outside" means is set by SPLIT_HALF_SCALE, and it is not
    sqrt(2). Each half is fitted from half the sweeps, so each carries about
    sqrt(2) times the whole run's standard error, and the two are independent:
    their difference scatters by 2 times it. Measured directly on 120
    synthetic runs at 25 sweeps: 1.97x.
    """
    sweeps = np.unique(data.sweep)
    if sweeps.size < 2 * MIN_SWEEPS:
        return float("nan"), float("nan"), float("nan")
    cut = sweeps[sweeps.size // 2]
    try:
        a = estimate(data.take(np.flatnonzero(data.sweep < cut)),
                     drift_order=drift_order, use_narrow=use_narrow).d_rx
        b = estimate(data.take(np.flatnonzero(data.sweep >= cut)),
                     drift_order=drift_order, use_narrow=use_narrow).d_rx
    except (ValueError, np.linalg.LinAlgError):
        return float("nan"), float("nan"), float("nan")
    return a, b, b - a


def cluster_leave_one_out(data: Dataset, drift_order: int,
                          use_narrow: bool) -> list[tuple[int, float]]:
    """d_rx with each cluster dropped in turn -- does one cluster carry it?"""
    clusters = np.unique(data.cluster)
    if clusters.size < 3:
        return []
    out = []
    for drop in clusters:
        try:
            out.append((int(drop), estimate(
                data.take(np.flatnonzero(data.cluster != drop)),
                drift_order=drift_order, use_narrow=use_narrow).d_rx))
        except (ValueError, np.linalg.LinAlgError):
            continue
    return out


def cluster_residuals(data: Dataset, residuals: np.ndarray
                      ) -> list[tuple[int, float, float, int]]:
    """Mean residual per cluster: a straight line in IF either fits or it does not.

    A response that curves with frequency -- an LNB band edge, a filter, a
    reflection -- would show here as clusters sitting consistently off the
    line. No amount of resampling would find that, because it is a bias and
    not a scatter, and it lands squarely on d_rx.
    """
    out = []
    for c in np.unique(data.cluster):
        vals = residuals[data.cluster == c]
        if vals.size < 2:
            continue
        out.append((int(c), float(vals.mean()),
                    float(vals.std(ddof=1) / np.sqrt(vals.size)),
                    int(vals.size)))
    return out


# ---------------------------------------------------------------------------
# the whole answer
# ---------------------------------------------------------------------------
@dataclass
class LeverReport:
    lo_hz: float
    captures: int
    rejected: int
    sweeps: int
    clusters: int
    lever_hz: float
    span_s: float
    t_ref_s: float
    use_narrow: bool
    narrow_reason: str
    est: Estimate
    res: Resample
    design_stderr: float          # what the design allows, given sigma_bias
    covariance_stderr: float      # what a covariance alone would have claimed
    agreement_sigma: float
    # The denominator of that sigma, in slope units. Multiplied by
    # NARROW_AGREE_SIGMA it IS the threshold the two-lever check applies, and
    # therefore the size of the one systematic no resample here can see: a
    # receiver bias that trends with rx_lo lands on d_rx one for one and only
    # this check can refuse it. Carried on the report so the printed accuracy
    # and the applied test cannot drift apart.
    agreement_denom: float = float("nan")
    variogram: list = field(default_factory=list)
    per_cluster: list = field(default_factory=list)
    leave_one_out: list = field(default_factory=list)
    split: tuple = (float("nan"),) * 3
    warnings: list[str] = field(default_factory=list)

    @property
    def trustworthy(self) -> bool:
        return not self.warnings

    def to_dict(self) -> dict:
        return {
            "lo_hz": self.lo_hz, "captures": self.captures,
            "rejected": self.rejected, "sweeps": self.sweeps,
            "clusters": self.clusters, "lever_hz": self.lever_hz,
            "span_s": self.span_s, "t_ref_s": self.t_ref_s,
            "d_rx": self.res.d_rx, "d_rx_stderr": self.res.d_rx_stderr,
            "d_rx_ci95": list(self.res.d_rx_ci),
            "d_rx_ppm": self.res.d_rx * 1e6,
            "d_rx_stderr_ppm": self.res.d_rx_stderr * 1e6,
            "d_lnb": self.res.d_lnb, "d_lnb_stderr": self.res.d_lnb_stderr,
            "d_lnb_ci95": list(self.res.d_lnb_ci),
            "d_lnb_hz": self.res.d_lnb * self.lo_hz,
            "d_lnb_stderr_hz": self.res.d_lnb_stderr * self.lo_hz,
            "lnb_drift_hz_s": self.res.drift_hz_s,
            "lnb_drift_stderr_hz_s": self.res.drift_stderr_hz_s,
            "d_rx_wide": self.est.d_rx_wide,
            "d_rx_narrow": self.est.d_rx_narrow,
            "narrow_excess_sd": self.est.narrow_tau,
            "agreement_sigma": self.agreement_sigma,
            "agreement_threshold_ppm": (NARROW_AGREE_SIGMA
                                        * self.agreement_denom * 1e6),
            "sigma_bias_hz": self.est.sigma_bias_hz,
            "design_stderr": self.design_stderr,
            "covariance_stderr": self.covariance_stderr,
            "bootstrap_stderr": self.res.d_rx_bootstrap_stderr,
            "jackknife_stderr": self.res.d_rx_jackknife_stderr,
            "use_narrow": self.use_narrow,
            "narrow_reason": self.narrow_reason,
            "split_half": list(self.split),
            "variogram": [list(v) for v in self.variogram],
            "per_cluster": [list(v) for v in self.per_cluster],
            "leave_one_out": [list(v) for v in self.leave_one_out],
            "warnings": self.warnings,
            "trustworthy": self.trustworthy,
        }


def analyse(records: list[CaptureRecord], *, lo_hz: float = DEFAULT_LO_HZ,
            drift_order: int = DEFAULT_DRIFT_ORDER,
            narrow: str = "check", draws: int = DEFAULT_BOOTSTRAP,
            seed: int = 20260819) -> LeverReport:
    """Everything, from a run's records to d_rx and d_lnb with real error bars."""
    if narrow not in NARROW_MODES:
        raise ValueError(f"narrow must be one of {NARROW_MODES}")
    good = [r for r in records if r.trustworthy]
    rejected = len(records) - len(good)
    if len(good) < 3:
        raise ValueError(f"only {len(good)} usable captures; the decoder "
                         f"disowned {rejected} of {len(records)}")
    data = Dataset.from_records(good)
    if np.unique(data.cluster).size < 2:
        raise ValueError(
            "only one cluster was measured, so there is no lever arm: d_rx "
            "and d_lnb enter every capture in the same fixed combination and "
            "no fit can separate them. That is the whole problem this tool "
            "exists to solve -- measure at least two clusters, as far apart "
            "in IF as the LNB band allows")
    t_ref = float(data.t.mean())

    # The narrow lever is admitted or refused ONCE, over the whole run. A
    # per-capture or per-cluster test would let one in by luck, and the narrow
    # lever is tight enough that a single bad admission would drag the answer.
    plain = estimate(data, drift_order=drift_order, t_ref_s=t_ref,
                     use_narrow=False)
    denom = math.hypot(plain.slope_wide_stderr, plain.slope_narrow_stderr)
    agree = (abs(plain.slope_wide - plain.slope_narrow) / denom
             if denom > 0 and np.isfinite(plain.slope_narrow) else float("nan"))
    if narrow == "off":
        use_narrow, reason = False, "switched off"
    elif not np.isfinite(agree):
        use_narrow, reason = False, "no usable within-capture slopes"
    elif narrow == "combine":
        use_narrow = True
        reason = ("folded in on request -- the answer is now only as good as "
                  "the assumption that nothing but the sample clock can tilt "
                  "a slope measured inside one capture")
    else:
        use_narrow = False
        reason = ("kept as an independent witness, not folded in: it is much "
                  "tighter than the wide lever, and the only evidence it is "
                  "unbiased is that the two agree -- which is evidence good "
                  "to about the wide lever's own precision")

    est = estimate(data, drift_order=drift_order, t_ref_s=t_ref,
                   use_narrow=use_narrow)
    res = resample(data, lo_hz, drift_order=drift_order,
                   use_narrow=use_narrow, draws=draws, seed=seed)
    wide = fit_wide(data, model="sweep", t_ref_s=t_ref)

    # What the design allows: the tuning bias divided by the lever arm the
    # sweeps actually built. This is the number the resample should land on,
    # and the number to plan a longer run with.
    sxx = 0.0
    for s in np.unique(data.sweep):
        m = data.sweep == s
        sxx += float(((data.x[m] - data.x[m].mean()) ** 2).sum())
    design_se = est.sigma_bias_hz / math.sqrt(sxx) if sxx > 0 else float("nan")
    cov_se = wide.slope_stderr
    if use_narrow and est.slope_narrow_stderr > 0 and cov_se > 0:
        cov_se = 1.0 / math.sqrt(1.0 / cov_se ** 2
                                 + 1.0 / est.slope_narrow_stderr ** 2)

    report = LeverReport(
        lo_hz=lo_hz, captures=len(good), rejected=rejected,
        sweeps=int(np.unique(data.sweep).size),
        clusters=int(np.unique(data.cluster).size),
        lever_hz=float(data.x.max() - data.x.min()),
        span_s=float(data.t.max() - data.t.min()), t_ref_s=t_ref,
        use_narrow=use_narrow, narrow_reason=reason, est=est, res=res,
        design_stderr=design_se, covariance_stderr=cov_se,
        agreement_sigma=agree, agreement_denom=denom,
        variogram=variogram(data, wide.residuals),
        per_cluster=cluster_residuals(data, wide.residuals),
        leave_one_out=cluster_leave_one_out(data, drift_order, use_narrow),
        split=split_half(data, drift_order, use_narrow))

    w = report.warnings
    if rejected:
        w.append(f"{rejected} of {len(records)} captures were disowned by the "
                 f"decoder and are not in this fit -- read their warnings "
                 f"before believing the ones that are left")
    if report.sweeps < MIN_SWEEPS:
        w.append(f"{report.sweeps} sweeps is under {MIN_SWEEPS}; a resample "
                 f"over sweeps has almost nothing to resample, so the "
                 f"uncertainty below is itself unreliable")
    if len(report.variogram) >= 2:
        near, far = report.variogram[0][1], report.variogram[-1][1]
        if far > 0 and near < VARIOGRAM_RATIO * far:
            w.append(
                f"the tuning bias is SMOOTH across the dither window: captures "
                f"{report.variogram[0][0]/1e3:.0f} kHz apart scatter "
                f"{math.sqrt(max(near,0)):.0f} Hz against "
                f"{math.sqrt(max(far,0)):.0f} Hz at "
                f"{report.variogram[-1][0]/1e3:.0f} kHz. Dithering is then not "
                f"drawing independent biases, and the sqrt(N) credited below "
                f"is not being earned -- widen the dither, which means raising "
                f"--fs or narrowing the cluster span to make room for it")
    if np.isfinite(agree) and agree > NARROW_AGREE_SIGMA:
        w.append(
            f"the two levers disagree by {agree:.1f} sigma: "
            f"{plain.d_rx_wide*1e6:+.4f} ppm across the clusters against "
            f"{plain.d_rx_narrow*1e6:+.4f} ppm inside them. One of them is "
            f"biased -- a tuning bias that the dither is not randomising, or "
            f"something in the receiver that depends on baseband frequency -- "
            f"and the fit cannot say which")
    if len(report.per_cluster) >= 3:
        # Pooled scale, not each cluster's own: a per-cluster standard error
        # from a handful of captures is itself so noisy that a quiet cluster
        # would flag every time. With no scatter at all there is nothing to
        # test and the check stands down rather than dividing by zero.
        pooled = float(np.std(wide.residuals, ddof=1)) if len(good) > 2 else 0.0
        r_mean = np.array([v[1] for v in report.per_cluster])
        counts = np.array([v[3] for v in report.per_cluster], dtype=float)
        # A residual scale below a microhertz is arithmetic dust, not a
        # measurement: dividing by it would make float noise look like a
        # frequency-dependent bias. Nothing real lives down there -- the
        # estimator's own bound is a hundredth of a hertz.
        ok = (counts > 1) & (pooled > NOTHING_TO_TEST_HZ)
        if ok.sum() >= 3:
            chi2 = float(((r_mean[ok] * np.sqrt(counts[ok]) / pooled) ** 2).sum())
            if chi2 > 4.0 * ok.sum():
                w.append(
                    f"the clusters do not sit on a straight line in IF "
                    f"(chi-square {chi2:.0f} on {int(ok.sum())} clusters) -- "
                    f"something in the chain depends on frequency in a way "
                    f"this model does not have, and it is sitting on d_rx")
    first, second, diff = report.split
    if np.isfinite(diff) and res.d_rx_stderr > 0 and \
            abs(diff) > SPLIT_HALF_SIGMA * SPLIT_HALF_SCALE * res.d_rx_stderr:
        w.append(
            f"d_rx moved between the halves of the run: {first*1e6:+.4f} then "
            f"{second*1e6:+.4f} ppm. The answer is then an average over "
            f"something that is drifting, most likely the Pluto's own "
            f"reference warming up -- let it settle and run again, or quote "
            f"the answer against the run's midpoint and say so")
    return report


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def format_report(rep: LeverReport) -> str:
    r, e, lo = rep.res, rep.est, rep.lo_hz
    lines = [
        "",
        f"  run          : {rep.captures} captures, {rep.sweeps} sweeps of "
        f"{rep.clusters} clusters over {rep.span_s/60:.1f} min",
        f"  lever arm    : {rep.lever_hz/1e6:.1f} MHz of IF between the "
        f"outermost clusters",
        "",
        "  ================  THE ANSWER  ================",
        f"  d_rx         : {r.d_rx*1e6:+.4f} +/- {r.d_rx_stderr*1e6:.4f} ppm"
        f"   (95% {r.d_rx_ci[0]*1e6:+.4f} to {r.d_rx_ci[1]*1e6:+.4f})",
        "                 the SDR's clock error, referred to the "
        "transmitter's own reference",
        f"  d_lnb        : {r.d_lnb*1e6:+.4f} +/- {r.d_lnb_stderr*1e6:.4f} ppm"
        f"  = {r.d_lnb*lo:+.1f} +/- {r.d_lnb_stderr*lo:.1f} Hz on "
        f"{lo/1e9:g} GHz",
        f"                 at the run's midpoint, and moving at "
        f"{r.drift_hz_s:+.2f} +/- {r.drift_stderr_hz_s:.2f} Hz/s, so it is "
        f"only true at an instant",
        "  ==============================================",
        "",
        f"  d_rx by lever: wide, across clusters  {e.d_rx_wide*1e6:+.4f} ppm "
        f"(+/-{e.slope_wide_stderr*1e6:.4f} formal)",
        f"                 narrow, inside each    {e.d_rx_narrow*1e6:+.4f} ppm "
        f"(+/-{e.slope_narrow_stderr*1e6:.4f} formal)",
        f"                 they differ by {rep.agreement_sigma:.1f} sigma; "
        f"narrow lever {'FOLDED IN' if rep.use_narrow else 'not folded in'}",
        f"                 -- {rep.narrow_reason}",
        f"                 READ THIS LINE FIRST. It is the only check on the "
        f"one error the resample",
        f"                 cannot see: a receiver bias that TRENDS with rx_lo "
        f"across the lever lands",
        f"                 on d_rx one for one, and the dither cannot touch "
        f"it. Passing certifies",
        f"                 that bias only to about "
        f"{NARROW_AGREE_SIGMA*rep.agreement_denom*1e6:.3f}"
        f" ppm, which is the accuracy here --",
        f"                 the {r.d_rx_stderr*1e6:.4f} ppm above is the "
        f"precision.",
        f"  tuning bias  : {e.sigma_bias_hz:.1f} Hz sd per capture, fitted "
        f"from this run's own scatter",
        f"  slope scatter: {e.narrow_tau*1e6:.4f} ppm sd beyond the estimator's "
        f"own error -- anything here depends on baseband frequency",
        f"  uncertainty  : quoted {r.d_rx_stderr*1e6:.4f}  |  bootstrap "
        f"{r.d_rx_bootstrap_stderr*1e6:.4f}  |  jackknife "
        f"{r.d_rx_jackknife_stderr*1e6:.4f}  |  design "
        f"{rep.design_stderr*1e6:.4f}  |  covariance "
        f"{rep.covariance_stderr*1e6:.4f} ppm",
        f"                 {r.draws} bootstrap draws over sweeps"
        + (f", {r.failures} refused" if r.failures else ""),
    ]
    if rep.covariance_stderr > 0 and np.isfinite(r.d_rx_stderr):
        ratio = r.d_rx_stderr / rep.covariance_stderr
        lines.append(
            f"                 the resample is {ratio:.1f}x the covariance's "
            f"claim" + (" -- which is why the covariance is not what gets "
                        "quoted" if ratio > 1.2 else ""))
    if rep.split[0] == rep.split[0]:
        lines.append(
            f"  split half   : {rep.split[0]*1e6:+.4f} then "
            f"{rep.split[1]*1e6:+.4f} ppm (difference "
            f"{rep.split[2]*1e6:+.4f}) -- is d_rx itself steady?")
    if rep.variogram:
        lines += ["",
                  "  tuning-bias variogram: how much two captures of ONE "
                  "cluster disagree, against how",
                  "  far apart their tunings were. Flat means the dither is "
                  "decorrelating the bias,",
                  "  which is the whole basis for averaging it down."]
        for h, g, n in rep.variogram:
            lines.append(f"     {h/1e3:7.1f} kHz apart : "
                         f"{math.sqrt(max(g, 0)):8.1f} Hz  from {n} pairs")
    if rep.per_cluster:
        lines += ["", "  per cluster (residual from the straight line; a "
                      "pattern here is a bias, not noise)"]
        for c, mean, se, n in rep.per_cluster:
            lines.append(f"     cluster {c}: {mean:+9.2f} +/- {se:6.2f} Hz "
                         f"over {n} captures")
    if rep.leave_one_out:
        lines += ["", "  d_rx with each cluster dropped in turn (ppm): " +
                  ", ".join(f"-c{c} {v*1e6:+.4f}"
                            for c, v in rep.leave_one_out)]
    if rep.warnings:
        bar = "  " + "!" * 68
        lines += ["", bar,
                  "  CONFIDENCE IS POOR -- do not use these numbers as a "
                  "measurement:"]
        lines += [f"    * {w}" for w in rep.warnings]
        lines += [bar]
    else:
        lines += ["", "  confidence good: every capture decoded, the two "
                      "levers agree, the clusters sit on a",
                  "  straight line, and the tuning bias is being averaged "
                  "rather than assumed away"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# a whole run, synthesised: the fit is verifiable with no radio and no capture
# ---------------------------------------------------------------------------
def reported_if_model(if_nom_hz, *, lo_hz: float, d_rx: float = 0.0,
                      d_lnb: float = 0.0, d_tx: float = 0.0,
                      tuning_bias_hz: float = 0.0, drift_hz_s: float = 0.0,
                      t_rel_s: float = 0.0):
    """The exact forward model, shared with the capture synthesiser.

    Three clocks are involved and only two can ever be separated:

        reported_IF = ( f_rf(1+d_tx) - f_lo(1+d_lnb) ) / (1 + d_rx) - b

    and since f_rf = f_IF + f_lo, the d_tx term splits into a piece that scales
    with f_IF and a piece that does not -- exactly the two slots d_rx and d_lnb
    occupy. So what is measured is (d_rx - d_tx) and (d_lnb - d_tx): everything
    is referred to the transmitter's 125 MHz reference, and that has to be said
    out loud, because "the SDR's clock error" only means something against
    something else.

    Written out rather than linearised because the cross term d_rx*d_lnb*f_LO
    is about 0.85 Hz at the numbers this hardware actually has, twenty times
    the per-capture precision the estimator now reaches.
    """
    if_nom = np.asarray(if_nom_hz, dtype=float)
    f_rf = (if_nom + lo_hz) * (1.0 + d_tx)
    f_lo = lo_hz * (1.0 + d_lnb) + drift_hz_s * t_rel_s
    return (f_rf - f_lo) / (1.0 + d_rx) - tuning_bias_hz


def synthesise_run(*, clusters_if_hz, sweeps: int = DEFAULT_SWEEPS,
                   lo_hz: float = DEFAULT_LO_HZ, d_rx: float = 8.94e-6,
                   d_lnb: float = 9.641e-6, drift_hz_s: float = 0.0,
                   sigma_bias_hz: float = 127.0, sigma_est_hz: float = 0.05,
                   sigma_slope: float = 5.0e-8, narrow_bias: float = 0.0,
                   cluster_offset_hz=None, capture_s: float = 3.0,
                   overhead_s: float = 9.0,
                   dither_hz: float = DEFAULT_DITHER_HZ,
                   seed: int = 0xC0FFEE, noise_seed: int = 11
                   ) -> list[CaptureRecord]:
    """The records a run WOULD have produced, from a known d_rx and d_lnb.

    This is the level the fit consumes, so it is the level the fit is tested
    at: no I/Q, no decoding, a hundred captures in a millisecond, and every
    error source the design worries about injectable one at a time.
    ``cluster_offset_hz`` puts a fixed per-cluster error in and ``narrow_bias``
    tilts the within-capture slopes -- the two failures no amount of resampling
    can find, put there so that the linearity check and the two-lever
    comparison can be shown to catch them.
    """
    rng = np.random.default_rng(noise_seed)
    visits = plan_visits(seed, len(clusters_if_hz), sweeps, dither_hz)
    per = capture_s + overhead_s
    # The reference instant is the mean of the CAPTURE midpoints, because that
    # is what the fit uses; anything else would make d_lnb come back offset by
    # the drift times the difference, which at 4.5 Hz/s is not small.
    t_ref = (len(visits) - 1) * per / 2.0 + capture_s / 2.0
    out: list[CaptureRecord] = []
    for v in visits:
        t = v.index * per + capture_s / 2.0
        if_c = float(clusters_if_hz[v.cluster])
        bias = rng.normal(0.0, sigma_bias_hz) if sigma_bias_hz > 0 else 0.0
        true = float(reported_if_model(if_c, lo_hz=lo_hz, d_rx=d_rx,
                                       d_lnb=d_lnb, drift_hz_s=drift_hz_s,
                                       t_rel_s=t - t_ref)) - if_c
        extra = (0.0 if cluster_offset_hz is None
                 else float(cluster_offset_hz[v.cluster]))
        y = true - bias + extra + rng.normal(0.0, sigma_est_hz)
        slope = (-d_rx / (1.0 + d_rx) + narrow_bias
                 + rng.normal(0.0, sigma_slope))
        out.append(CaptureRecord(
            index=v.index, sweep=v.sweep, cluster=v.cluster,
            cluster_centre_hz=if_c + lo_hz, rx_lo_hz=if_c + v.dither_hz,
            dither_hz=v.dither_hz, t_abs_s=t, mean_if_hz=if_c,
            mean_error_hz=y, mean_error_stderr_hz=sigma_est_hz,
            slope=slope, slope_stderr=sigma_slope, recovered=10, points=10,
            visits=40, trustworthy=True))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", nargs="?", default=None,
                   help="the JSON-lines file tools/lever_run.py wrote")
    p.add_argument("--lo-hz", type=float, default=DEFAULT_LO_HZ,
                   help="nominal LNB LO (default 9.75e9, low band)")
    p.add_argument("--drift-order", type=int, default=DEFAULT_DRIFT_ORDER,
                   help="polynomial order of the LNB drift. It is used ONLY "
                        f"for d_lnb (default {DEFAULT_DRIFT_ORDER}); d_rx is "
                        "fitted with a free offset per sweep and never sees it")
    p.add_argument("--narrow", choices=NARROW_MODES, default="check",
                   help="what to do with each capture's own within-cluster "
                        "slope. 'check' (default) reports it as an independent "
                        "witness; 'combine' folds it into the answer, which is "
                        "only defensible if nothing but the sample clock can "
                        "tilt a slope inside one capture; 'off' ignores it")
    p.add_argument("--draws", type=int, default=DEFAULT_BOOTSTRAP,
                   help="bootstrap resamples over sweeps")
    p.add_argument("--seed", type=lambda v: int(v, 0), default=20260819,
                   help="resampling seed, so a quoted number is reproducible")
    p.add_argument("--json", action="store_true")
    p.add_argument("--self-test", action="store_true",
                   help="synthesise a run with a known d_rx and d_lnb and "
                        "recover them; no radio, no capture, no file")
    p.add_argument("--inject-drift", type=float, default=4.5,
                   help="self-test only: LNB drift in Hz/s")
    p.add_argument("--inject-bias", type=float, default=127.0,
                   help="self-test only: tuning bias sd in Hz")
    p.add_argument("--sweeps", type=int, default=DEFAULT_SWEEPS,
                   help="self-test only: how many sweeps to synthesise")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        d_rx, d_lnb = 8.94e-6, 9.641e-6
        clusters = [0.95e9, 1.35e9, 1.75e9, 2.15e9]
        print("\n  SELF TEST: synthesising a run. Nothing is captured and no "
              "radio is opened.")
        print(f"  injected     : d_rx {d_rx*1e6:+.4f} ppm, d_lnb "
              f"{d_lnb*1e6:+.4f} ppm ({d_lnb*args.lo_hz:+.1f} Hz), drift "
              f"{args.inject_drift:+.2f} Hz/s, tuning bias "
              f"{args.inject_bias:.0f} Hz sd")
        records = synthesise_run(clusters_if_hz=clusters, sweeps=args.sweeps,
                                 lo_hz=args.lo_hz, d_rx=d_rx, d_lnb=d_lnb,
                                 drift_hz_s=args.inject_drift,
                                 sigma_bias_hz=args.inject_bias)
        header = {"lo_hz": args.lo_hz, "synthetic": True,
                  "injected_d_rx": d_rx, "injected_d_lnb": d_lnb}
    elif args.run:
        header, records = read_records(args.run)
        print(f"\n  reading      : {args.run} ({len(records)} captures)")
    else:
        build_parser().print_help()
        return 2

    lo_hz = float(header.get("lo_hz", args.lo_hz))
    report = analyse(records, lo_hz=lo_hz, drift_order=args.drift_order,
                     narrow=args.narrow, draws=args.draws, seed=args.seed)
    print(format_report(report))
    if args.self_test:
        got, want = report.res.d_rx, header["injected_d_rx"]
        gl, wl = report.res.d_lnb, header["injected_d_lnb"]
        print(f"\n  recovered d_rx  {got*1e6:+.4f} ppm against "
              f"{want*1e6:+.4f} injected: error {(got-want)*1e6:+.4f} ppm "
              f"({abs(got-want)/max(report.res.d_rx_stderr, 1e-30):.2f} sigma)")
        print(f"  recovered d_lnb {gl*1e6:+.4f} ppm against "
              f"{wl*1e6:+.4f} injected: error {(gl-wl)*lo_hz:+.2f} Hz "
              f"({abs(gl-wl)/max(report.res.d_lnb_stderr, 1e-30):.2f} sigma)")
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=float))
    return 0 if report.trustworthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
