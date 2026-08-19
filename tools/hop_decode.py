#!/usr/bin/env python3
"""Decode a seeded frequency-hop capture: regenerate the schedule, align, measure.

Identity is never inferred from what the capture looks like. The receiver
rebuilds the exact frequency order from the shared seed, so the only unknown is
the epoch -- a single one-dimensional search -- and once that is fixed, every
frame's frequency point is known and the residual error at each point follows
directly.

The steps, in order:

1. **Frame** the capture into non-overlapping blocks well shorter than one
   dwell, subtract each frame's mean (which kills the receiver's DC spur),
   window, and FFT.
2. **Find the comb.** The transmitted points form a comb of known spacing at a
   common unknown offset (LNB LO error plus receiver clock error). Slide the
   expected comb across the time-averaged spectrum and keep the offset with the
   most energy in the expected bins. ``comb_sharpness`` = peak / median of that
   search says how much to believe it.
3. **Build a per-point envelope.** For each point, the largest magnitude inside
   a narrow slot around its offset-corrected expected frequency, per frame.
   This is the step that makes the whole thing work: one broadband envelope
   merges adjacent points into a single continuous excursion and tells you
   nothing, whereas per-point slots keep them separate.
4. **Align the epoch.** Regenerate the schedule from the seed and slide it over
   ONE PERIOD only -- the pattern repeats every ``period_cycles``
   permutations -- scoring each shift by the mean envelope dB of the point the
   schedule expects in each frame. ``epoch_sigma`` is how far the winner stands
   above the rest of that search, in standard deviations.
5. **Measure.** Per point, fit each whole dwell coherently -- mix it down by
   the coarse frequency, decimate, and take the maximum-likelihood periodogram
   peak -- then combine that point's dwells by inverse variance. Selectable:
   ``--estimator peak`` is the original framed FFT peak with a median over
   frames, kept because the improvement is worth being able to reproduce rather
   than assert. The framed estimator's error is a fixed per-point interpolation
   bias of tens of hertz that no amount of listening removes; the whole-dwell
   fit reaches the Cramer-Rao bound and is about 1500x tighter.

A confident wrong answer is the failure that matters here, so both confidence
figures are printed every time. Anything the run flags -- a floor missed, a
point not recovered, a frame longer than a dwell -- prints a loud banner AND
sets a non-zero exit status; the two always agree, so a scripted caller cannot
record a number the report has just disowned.

The schedule is imported from ``adf5355.hopper``: transmitter and receiver run
the same generator, so the two ends cannot drift apart.

    # decode a capture already on disk (interleaved int16 I/Q)
    tools/hop_decode.py --capture run.iq --fs 2.5e6

    # capture from the Pluto and decode in one go
    tools/hop_decode.py --seconds 8

    # one cluster of a lever-arm schedule (see tools/lever_run.py for the run)
    tools/hop_decode.py --clusters 4 --cluster 2 --seconds 3

    # no hardware, no capture: synthesise a known answer and recover it
    tools/hop_decode.py --self-test

Clusters
--------
With ``--clusters`` the transmitter is hopping over several widely separated
groups of points and this capture holds exactly one of them. Everything above
is unchanged -- the comb, the epoch and the whole-dwell fit see one cluster's
points and nothing else -- except that the schedule regenerated here is the
full multi-cluster one, so the dwells belonging to other clusters are known to
be dead air rather than mistaken for this cluster's points. A cluster's points
sit on a Golomb ruler rather than a regular grid, because a regular comb can be
slid onto itself and the search then has a rival peak one spacing away that
mislabels every point; ``margin over the nearest rival`` is what reports that.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adf5355.hopper import (DEFAULT_BAND_EXTRA_S, DEFAULT_BLOCK,  # noqa: E402
                            DEFAULT_CLUSTER_CENTRES_HZ,
                            DEFAULT_CLUSTER_POINTS,
                            DEFAULT_CLUSTER_SPAN_HZ,
                            DEFAULT_HOP_START_HZ, DEFAULT_HOP_STOP_HZ,
                            DEFAULT_JITTER, DEFAULT_MIN_HOP_S,
                            DEFAULT_PERIOD_CYCLES, DEFAULT_POINTS,
                            DEFAULT_SEED, ClusterPlan, Hop, cluster_centres,
                            describe, describe_clusters, make_cluster_schedule,
                            make_schedule, period_duration, plan_clusters,
                            plan_frequencies)

# ---- receive-side defaults; sdr_listen.sh mirrors these -------------------
DEFAULT_LO_HZ = 9.75e9          # nominal LNB LO: 13 V, no tone = low band
DEFAULT_LO_ERROR_HZ = 94_000.0  # measured; used only to centre the receiver
DEFAULT_FS = 2.5e6              # about 2 MHz usable, comfortably over 1.71
DEFAULT_FRAME = 512             # 204.8 us at 2.5 MS/s: ~49 frames per 10 ms
DEFAULT_SECONDS = 2.0     # 10 periods. Measured flat from 1 period to 160.
DEFAULT_GAIN = 40.0
DEFAULT_URI = "ip:192.168.2.1"
DEFAULT_NBUF = 1 << 16

DEFAULT_THRESHOLD_DB = 15.0     # a frame counts only this far above the floor
DEFAULT_SEARCH_HZ = 400e3       # comb offset search, both directions
DEFAULT_MAX_SLOT_HZ = 30e3      # slot half-width ceiling around each point
DEFAULT_MEAN_FRAMES = 8192      # frames averaged for the comb search
MIN_FRAMES_PER_POINT = 5        # fewer than this and a point is not reported
CHUNK_FRAMES = 4096             # frames per FFT batch, a memory/speed trade

# Below either of these, treat the answer as unproven rather than as a
# measurement. Measured on synthetic captures at the recommended defaults: a
# genuine decode at 10 dB SNR sits near 7000x and 700 sigma, a wrong seed under
# 5 sigma, and pure noise under 2x. Both floors sit well clear of those failures
# without being generous: by -20 dB SNR the comb search is itself hovering at
# 8x, and long before that -- around -10 dB -- no frame clears the envelope
# threshold any more, so no point is reported at all. The bench runs that worked
# sat at 37-422x comb sharpness.
MIN_COMB_SHARPNESS = 8.0
# Epoch-alignment confidence floor.
#
# 10 was set against synthetic single-cluster captures, where one cluster is on
# air the whole time. In cluster mode each cluster transmits only 1/clusters of
# the period, so the same capture carries proportionally less alignment
# evidence and sigma lands lower for a perfectly good decode. Measured on the
# bench with 4 clusters and 8 s captures: sigma 7.1-8.1 while comb sharpness was
# 104-186 against a floor of 8, every capture returned 6/6 points, and the
# recovered offsets repeated to within the LNB's own drift. So a floor of 10 was
# rejecting sound data on a statistic that duty cycle alone had lowered.
#
# The floor now scales with the square root of the duty cycle, which is how the
# alignment statistic itself scales with the evidence available.
MIN_EPOCH_SIGMA = 10.0


def min_epoch_sigma(duty: float = 1.0) -> float:
    """Alignment floor for a signal present `duty` of the time."""
    if not 0.0 < duty <= 1.0:
        raise ValueError("duty must be in (0, 1]")
    return MIN_EPOCH_SIGMA * duty ** 0.5
# A measured comb margin is judged against what the frequency plan allows,
# not against a fixed number: it must reach this fraction of the way from 1x
# (hopeless) to the geometric ceiling. A uniform 20-point comb has a ceiling
# of 1.05x and so is barely judged at all -- which is honest, because such a
# comb genuinely cannot be told from its own shifted copy.
# The geometric ceiling is optimistic: a rival peak also collects the noise
# floor under its bins, so a four-point Golomb cluster whose ceiling is 4x
# measures about 2.3x at 12 dB. A third of the way to the ceiling separates
# that comfortably from the 1.0x a genuine mis-lock gives.
COMB_MARGIN_FRACTION = 0.3


# ---------------------------------------------------------------------------
# capture access
# ---------------------------------------------------------------------------
class Capture:
    """Framed access to an I/Q capture, from a file or from memory.

    Files are memory-mapped and read in blocks, so an 8 s capture at 2.5 MS/s
    (80 MB of interleaved int16) never lands in RAM all at once. Passing a
    complex array instead is what the tests use, and what ``--self-test`` uses.
    """

    def __init__(self, source, frame: int) -> None:
        if frame < 16:
            raise ValueError("frame must be at least 16 samples")
        self.frame = int(frame)
        if isinstance(source, (str, os.PathLike)):
            raw = np.memmap(os.fspath(source), dtype="<i2", mode="r")
            if raw.size < 2 * self.frame:
                raise ValueError(f"{source} holds under one frame of I/Q")
            self._raw = raw
            self._samples = None
            self.nframes = raw.size // 2 // self.frame
            self.name = os.fspath(source)
        else:
            samples = np.asarray(source)
            if samples.ndim != 1:
                raise ValueError("samples must be a 1-D complex array")
            self._raw = None
            self._samples = samples.astype(np.complex64, copy=False)
            self.nframes = samples.size // self.frame
            self.name = "<memory>"
        if self.nframes < 2:
            raise ValueError("capture is shorter than two frames")

    def block(self, start: int, count: int) -> np.ndarray:
        """``count`` frames from ``start`` as complex64, shape (n, frame)."""
        n = max(0, min(count, self.nframes - start))
        if n == 0:
            return np.empty((0, self.frame), dtype=np.complex64)
        if self._raw is not None:
            flat = np.asarray(
                self._raw[start * self.frame * 2:(start + n) * self.frame * 2])
            pairs = flat.reshape(n, self.frame, 2).astype(np.float32)
            return pairs[:, :, 0] + 1j * pairs[:, :, 1]
        return self._samples[start * self.frame:
                             (start + n) * self.frame].reshape(n, self.frame)

    @property
    def nsamples(self) -> int:
        return self.nframes * self.frame

    def samples(self, start: int, count: int) -> np.ndarray:
        """``count`` complex samples from sample index ``start``.

        Frame-free access, because the fine estimator works on whole dwells
        rather than on the framing grid the envelope search uses.
        """
        start = max(0, int(start))
        stop = min(self.nsamples, start + max(0, int(count)))
        if stop <= start:
            return np.empty(0, dtype=np.complex64)
        if self._raw is not None:
            flat = np.asarray(self._raw[start * 2:stop * 2]).astype(np.float32)
            return flat[0::2] + 1j * flat[1::2]
        return self._samples[start:stop]


def frame_bins(frame: int, fs: float, centre_hz: float) -> np.ndarray:
    """Absolute frequency of each FFT bin, fftshifted so it ascends."""
    return np.fft.fftshift(np.fft.fftfreq(frame, 1.0 / fs)) + centre_hz


def frame_power(block: np.ndarray, window: np.ndarray) -> np.ndarray:
    """|FFT|^2 per frame, mean removed first so the DC spur cannot win."""
    x = block - block.mean(axis=1, keepdims=True)
    return np.abs(np.fft.fftshift(np.fft.fft(x * window, axis=1), axes=1)) ** 2


def mean_spectrum(capture: Capture, window: np.ndarray,
                  max_frames: int = DEFAULT_MEAN_FRAMES) -> np.ndarray:
    """Time-averaged power spectrum over the first ``max_frames`` frames."""
    total = np.zeros(capture.frame)
    used = min(capture.nframes, max_frames)
    for start in range(0, used, CHUNK_FRAMES):
        block = capture.block(start, min(CHUNK_FRAMES, used - start))
        total += frame_power(block, window).sum(axis=0)
    return total


# ---------------------------------------------------------------------------
# step 2: where is the comb?
# ---------------------------------------------------------------------------
def comb_search(mean_power: np.ndarray, fbins: np.ndarray,
                if_nom: np.ndarray, search_hz: float = DEFAULT_SEARCH_HZ,
                step_hz: float | None = None) -> tuple[float, float, float]:
    """Slide the expected comb over the spectrum. (offset, sharpness, margin).

    Every point shares one offset -- the LNB's LO error plus the receiver's
    clock error are common to all of them -- so the whole comb moves as one
    rigid object and matching it is a single-parameter search. Summed energy at
    the expected bins is the score.

    Two confidence figures come out and they answer different questions.
    ``sharpness``, peak over median, asks whether a comb was found at all.
    ``margin``, peak over the best score more than a couple of bins away, asks
    whether the RIGHT one was found -- and that is the question that matters,
    because the failure it catches is silent. A comb that can be slid onto
    itself has a rival peak scoring nearly as high, one whole spacing away,
    and taking that one mislabels every point and returns a confident answer
    wrong by a spacing. See :data:`adf5355.hopper.GOLOMB_RULERS` for why the
    cluster layout has no such rival, and :func:`comb_ambiguity` for what
    margin a given layout entitles you to expect.
    """
    bin_hz = float(fbins[1] - fbins[0])
    if step_hz is None:
        step_hz = bin_hz / 8.0
    background = np.percentile(mean_power, 20)
    comb = np.maximum(mean_power - background, 0.0)
    offsets = np.arange(-search_hz, search_hz + 0.5 * step_hz, step_hz)
    idx = np.rint((if_nom[None, :] + offsets[:, None] - fbins[0]) / bin_hz)
    idx = np.clip(idx.astype(np.int64), 0, len(fbins) - 1)
    scores = comb[idx].sum(axis=1)
    best = int(np.argmax(scores))
    sharpness = float(scores[best] / (np.median(scores) + 1e-30))
    away = np.abs(offsets - offsets[best]) > 2.0 * bin_hz
    margin = (float(scores[best] / (scores[away].max() + 1e-30))
              if away.any() else float("inf"))
    return float(offsets[best]), sharpness, margin


def comb_offset(mean_power: np.ndarray, fbins: np.ndarray,
                if_nom: np.ndarray, search_hz: float = DEFAULT_SEARCH_HZ,
                step_hz: float | None = None) -> tuple[float, float]:
    """:func:`comb_search` without the margin, for callers that predate it."""
    offset, sharpness, _ = comb_search(mean_power, fbins, if_nom, search_hz,
                                       step_hz)
    return offset, sharpness


def comb_ambiguity(if_nom: np.ndarray, tol_hz: float) -> float:
    """Largest fraction of a comb that a rigid shift can put back on itself.

    A property of the frequency plan alone, computed without looking at any
    data, and therefore the right thing to judge a measured margin against: a
    uniformly spaced comb of P points scores (P-1)/P and cannot do better than
    a margin of P/(P-1) no matter how clean the capture is, while a Golomb
    ruler scores 1/P and should show a margin of about P.
    """
    f = np.sort(np.asarray(if_nom, dtype=float))
    n = f.size
    if n < 2:
        return 1.0
    shifts = np.unique(np.abs(f[:, None] - f[None, :]))
    shifts = shifts[shifts > tol_hz]
    if shifts.size == 0:
        return 1.0 / n
    hit = np.abs((f[None, :, None] + shifts[:, None, None]) - f[None, None, :])
    return float((hit.min(axis=2) <= tol_hz).sum(axis=1).max()) / n


def comb_margin_floor(if_nom: np.ndarray, tol_hz: float) -> float:
    """Smallest comb margin this frequency plan is allowed to show.

    The ceiling is geometric -- ``1 / comb_ambiguity`` -- and unreachable in
    practice, because a rival peak also collects the noise under its own bins.
    The floor is a fixed fraction of the way from 1x (hopeless, a mis-lock) to
    that ceiling, so a plan that genuinely cannot beat 1.05x is not condemned
    for it while a Golomb ruler that should show 6x and shows 1.2x is.

    Kept as a named function rather than three inline terms so the threshold
    is something a test can pin: with it inline, changing
    ``COMB_MARGIN_FRACTION`` to zero disables the whole mis-lock guard and
    nothing notices.
    """
    allowed = 1.0 / max(comb_ambiguity(if_nom, tol_hz), 1e-9)
    return 1.0 + COMB_MARGIN_FRACTION * (allowed - 1.0)


# ---------------------------------------------------------------------------
# step 3: one envelope per frequency point
# ---------------------------------------------------------------------------
def slot_half_width(if_nom: np.ndarray,
                    ceiling_hz: float = DEFAULT_MAX_SLOT_HZ) -> float:
    """Half-width of each point's slot: narrow enough never to overlap."""
    spacing = float(np.min(np.diff(if_nom))) if len(if_nom) > 1 else ceiling_hz
    return min(ceiling_hz, spacing / 2.0 * 0.6)


def point_slots(fbins: np.ndarray, if_nom: np.ndarray, offset_hz: float,
                half_hz: float) -> list[tuple[int, int]]:
    """Bin ranges [lo, hi) to search for each point, offset applied."""
    slots = []
    for centre in if_nom + offset_hz:
        lo = int(np.searchsorted(fbins, centre - half_hz))
        hi = int(np.searchsorted(fbins, centre + half_hz))
        lo = max(0, min(lo, len(fbins) - 1))
        hi = max(lo + 1, min(hi, len(fbins)))
        slots.append((lo, hi))
    return slots


def point_envelopes(capture: Capture, window: np.ndarray, fbins: np.ndarray,
                    slots: list[tuple[int, int]], want_peak: bool = True
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Per point and frame: peak power in the slot, and where that peak sits.

    The frequency is refined by a parabola through the log magnitudes either
    side of the winning bin, which is worth roughly two orders of magnitude
    against the raw bin width -- and which is also the source of the fixed
    per-point bias that the whole-dwell estimator exists to escape. With
    ``want_peak`` false only the envelope is built, which is all the epoch
    alignment needs and skips that arithmetic entirely.
    """
    npoints, nframes = len(slots), capture.nframes
    env = np.zeros((npoints, nframes), dtype=np.float32)
    peak = np.zeros((npoints, nframes), dtype=np.float64)
    bin_hz = float(fbins[1] - fbins[0])
    last = capture.frame - 1
    for start in range(0, nframes, CHUNK_FRAMES):
        block = capture.block(start, CHUNK_FRAMES)
        n = block.shape[0]
        power = frame_power(block, window)
        rows = np.arange(n)
        for p, (lo, hi) in enumerate(slots):
            segment = power[:, lo:hi]
            k = lo + np.argmax(segment, axis=1)
            env[p, start:start + n] = power[rows, k]
            if not want_peak:
                continue
            mid = np.clip(k, 1, last - 1)
            a = np.log(power[rows, mid - 1] + 1e-30)
            b = np.log(power[rows, mid] + 1e-30)
            c = np.log(power[rows, mid + 1] + 1e-30)
            denominator = a - 2 * b + c
            delta = np.where(np.abs(denominator) > 1e-12,
                             0.5 * (a - c) / np.where(denominator == 0, 1,
                                                      denominator), 0.0)
            peak[p, start:start + n] = fbins[mid] + np.clip(delta, -0.5, 0.5) * bin_hz
    return env, peak


def envelope_db(env: np.ndarray) -> np.ndarray:
    """Each point's envelope in dB above its OWN median.

    Relative to itself, not to a global floor: gain, cable loss and the LNB's
    tilt across the band all differ point to point, and none of that is
    interesting here.
    """
    floor = np.maximum(np.median(env, axis=1, keepdims=True), 1e-30)
    return 10.0 * np.log10(env / floor + 1e-30)


# ---------------------------------------------------------------------------
# step 4: which point does each frame belong to?
# ---------------------------------------------------------------------------
@dataclass
class EpochFit:
    shift_s: float          # schedule time = capture time + shift (mod period)
    sigma: float            # (best - mean) / sd over the search
    period_s: float
    assigned: np.ndarray    # point index expected in each frame


def align_epoch(env_db: np.ndarray, hops: list[Hop], points: int,
                frame_s: float, period_cycles: int = DEFAULT_PERIOD_CYCLES,
                oversample: int = 2, cluster: int | None = None,
                hops_per_period: int | None = None,
                period_s: float | None = None) -> EpochFit:
    """Find where in the schedule the capture starts.

    Bounded by one period, not by the length of the run: the pattern repeats
    every ``period_cycles`` permutations, which is the difference between a
    millisecond of searching and a minute of it. A pseudorandom permutation
    autocorrelates sharply, so the winner is normally unmistakable -- which is
    exactly what ``sigma`` reports.

    ``sigma`` measures the winner against the BACKGROUND of the search, with
    shifts within one dwell of it excluded. Those neighbours are still mostly
    correct -- a shift error of d mislabels only the d/dwell of each frame near
    a hop boundary -- so leaving them in would inflate the spread and make a
    perfect alignment look marginal.
    """
    per = hops_per_period if hops_per_period is not None else points * period_cycles
    if len(hops) < per + 1:
        raise ValueError("schedule is shorter than one period")
    if period_s is None:
        period_s = period_duration(hops, points, period_cycles)
    ends = np.array([h.end_s for h in hops[:per]])
    # -1 marks a hop that belongs to some OTHER cluster. Those slots are dead
    # air for this receiver, so they are scored not at all rather than scored
    # as a miss: what identifies the epoch is that MY dwells line up with MY
    # tones, and the silence in between carries no information either way.
    expect = np.array([h.point if (cluster is None or h.cluster == cluster)
                       else -1 for h in hops[:per]])
    if not (expect >= 0).any():
        raise ValueError(f"no hops in cluster {cluster}")
    nframes = env_db.shape[1]
    ft = np.arange(nframes) * frame_s
    cols = np.arange(nframes)
    shifts = np.arange(0.0, period_s, frame_s / oversample)
    scores = np.empty(len(shifts))
    mine_all = expect >= 0
    for k, shift in enumerate(shifts):
        idx = np.clip(np.searchsorted(ends, (ft + shift) % period_s), 0, per - 1)
        if mine_all.all():
            scores[k] = env_db[expect[idx], cols].mean()
            continue
        sel = mine_all[idx]
        scores[k] = (env_db[expect[idx][sel], cols[sel]].mean() if sel.any()
                     else -np.inf)
    best = int(np.argmax(scores))

    guard = min(h.dwell_s for h in hops[:per])
    apart = np.abs(shifts - shifts[best])
    background = scores[np.minimum(apart, period_s - apart) > guard]
    if background.size < 8:                     # too short a period to judge
        background = scores
    sigma = float((scores[best] - background.mean()) / (background.std() + 1e-30))

    idx = np.clip(np.searchsorted(ends, (ft + shifts[best]) % period_s),
                  0, per - 1)
    return EpochFit(float(shifts[best]), sigma, float(period_s), expect[idx])


# ---------------------------------------------------------------------------
# step 5: measure
# ---------------------------------------------------------------------------
@dataclass
class PointResult:
    point: int
    nominal_if_hz: float
    measured_if_hz: float
    error_hz: float
    frames_used: int
    frames_assigned: int
    envelope_db: float
    # Filled in by the whole-dwell estimator only; the framed one has no
    # notion of a visit and no honest standard error to report.
    visits_used: int = 0
    visits_total: int = 0
    stderr_hz: float = float("nan")
    crb_hz: float = float("nan")      # what the Cramer-Rao bound allows
    coherence: float = float("nan")
    coherence_null: float = float("nan")


def measure_points(env_db: np.ndarray, peak: np.ndarray, assigned: np.ndarray,
                   if_nom: np.ndarray,
                   threshold_db: float = DEFAULT_THRESHOLD_DB,
                   min_frames: int = MIN_FRAMES_PER_POINT
                   ) -> list[PointResult]:
    """Median peak frequency over the frames the schedule assigns to each point.

    Median, not mean: a frame that straddles a hop boundary holds two tones and
    its peak lands wherever the stronger half puts it. Those are outliers, not
    noise, and a median ignores them for free.
    """
    results = []
    for p in range(len(if_nom)):
        mine = assigned == p
        strong = mine & (env_db[p] > threshold_db)
        if int(strong.sum()) < min_frames:
            continue
        measured = float(np.median(peak[p][strong]))
        results.append(PointResult(
            point=p, nominal_if_hz=float(if_nom[p]), measured_if_hz=measured,
            error_hz=measured - float(if_nom[p]),
            frames_used=int(strong.sum()), frames_assigned=int(mine.sum()),
            envelope_db=float(np.median(env_db[p][strong]))))
    return results


# ---------------------------------------------------------------------------
# step 5b: fine frequency estimation, one whole dwell at a time
# ---------------------------------------------------------------------------
# The framed peak search above is what finds and labels the signal, and it is
# very good at that. It is a poor *frequency* estimator, and its weakness is
# not noise: a parabola through three log-magnitude bins of a Hann-windowed
# frame is a biased fit, and the bias is a fixed function of where the tone
# sits inside its bin. Every point sits at its own fixed fractional bin
# position, so every point gets its own fixed error -- up to 78 Hz, 58 Hz sd
# across the standard 20-point plan -- and that error is identical in every
# frame of every visit. Averaging cannot touch it, which is exactly the
# measured behaviour: the spread does not fall with listen time.
#
# The cure is to stop estimating frequency from a framed spectrogram. Within
# one dwell the transmitter emits a clean CW tone of known start and length, so
# the whole dwell can be used coherently:
#
#   1. mix the dwell down by the coarse frequency the comb search already
#      established, so the residual sits near DC;
#   2. boxcar-decimate, which is a linear-phase average -- it cannot bias a
#      frequency estimate, it costs nothing in Fisher information, and it cuts
#      the noise bandwidth by the decimation factor;
#   3. estimate the residual with a maximum-likelihood periodogram search.
#
# One 10 ms dwell then lands within a few percent of the Cramer-Rao bound,
# which at 10 dB per-sample SNR is 0.078 Hz -- roughly 700x better than the
# framed estimator, and unlike it, improving with SNR and with dwell.
#
# Guards matter. The synthesiser retunes into the start of each dwell, so the
# first millisecond or so is a settling chirp rather than a tone; a coherent
# fit over the whole dwell would swallow it, where a median over frames spat it
# out. Trimming the head and tail of every dwell is what makes the coherent fit
# safe, and it is cheap: precision goes as the 3/2 power of the usable length.

DEFAULT_ESTIMATOR = "ml"
DEFAULT_DECIMATE = 32           # 2.5 MS/s -> 78.1 kS/s, still ±39 kHz of room
DEFAULT_GUARD_START_S = 1.5e-3  # retune settling lands at the head of a dwell
DEFAULT_GUARD_END_S = 0.2e-3    # epoch is known to about half a frame
MAX_GUARD_FRACTION = 0.35       # never trim away more than this of a dwell
MIN_VISIT_SAMPLES = 32          # decimated samples; fewer and the fit is junk
DEFAULT_VISIT_SNR_DB = 6.0      # a visit must stand this far above its noise
# One dwell is already a complete measurement -- 0.08 Hz at 10 dB -- so a
# point is reported from a single visit. The framed estimator needed a crowd
# of frames because each frame was poor; this one does not.
MIN_VISITS_PER_POINT = 1
OUTLIER_MADS = 6.0              # visit rejection, in median absolute deviations
MIN_VISITS_FOR_OUTLIER_CUT = 4  # below this a MAD has no meaning
# Below this the coherence diagnostic cannot separate anything: maximising
# over frequency fits random phases well when there are few of them, so the
# random-phase null itself sits at 0.87 for 3 visits and 0.57 for 5, leaving no
# room above it. By 8 visits the null is 0.41 and a real coherence of ~1.0
# stands clear; the default 2 s capture gives 10.
MIN_VISITS_FOR_COHERENCE = 8
SETTLING_SIGMA = 5.0            # half-split this far from zero earns a warning
# The visits of one point are independent measurements of one number, so their
# scatter should match what the Cramer-Rao bound allows for their SNR and
# length. Scattering far wider means something is moving that the model does
# not have -- and the number would then be precise and wrong, which is the one
# failure this tool exists to refuse.
# How far a point's visits scatter beyond the Cramer-Rao bound.
#
# REPORTED, NOT ENFORCED -- and the reason matters. The CRB assumes an ideal
# tone in thermal noise alone. A real chain adds synthesiser and LNB phase
# noise and the LNB drifts within a capture, so some excess is physics, not a
# fault. Measured on the bench: healthy captures (comb 104-186 against a floor
# of 8, 6/6 points, offsets repeating to within the LNB's own drift) ran
# 54-128x. The test fixture's deliberately unmodelled disturbance sits at 86x --
# inside that range. So this statistic cannot separate a good capture from a
# bad one on this hardware at any threshold: 5 rejected every real capture, and
# anything past 128 would have to ignore a real fault.
#
# It stays as a warning because it is genuinely informative when it moves (it
# was 200x+ while the band allowance was too short and the tone was still
# settling, and dropped to 54-128x when that was fixed), and the gates that DO
# separate cleanly -- comb sharpness, epoch alignment, points recovered, the
# half-split -- still disown a capture on their own.
#
# The principled repair is to give the bound a phase-noise term measured from
# this chain, so a healthy capture sits near 1x again and a fault stands out.
# Until then a number here is a hint, not a verdict.
MAX_EXCESS_SCATTER = 5.0
ENFORCE_EXCESS_SCATTER = False
MIN_VISITS_FOR_SCATTER = 8
COMBINE_NAMES = ("incoherent", "coherent")


def mix_and_decimate(raw: np.ndarray, cycles: float, start: int,
                     factor: int) -> np.ndarray:
    """Mix down by ``cycles`` per sample and boxcar-decimate, in one pass.

    The two steps fold together because the mixing exponential factorises over
    the decimation blocks: with ``n = m*D + k``,

        exp(-2i pi c n) = exp(-2i pi c D m) * exp(-2i pi c k)

    so the inner factor is one fixed vector of length D reused for every block,
    and only one exponential per OUTPUT sample is ever computed instead of one
    per input sample. On a Pi that is most of the decode.

    ``start`` is the absolute sample index, so every visit is mixed against one
    common clock and the phases stay comparable between them -- which is what
    the coherence diagnostic needs. The phase is reduced modulo a cycle before
    being scaled, because the absolute index reaches tens of millions over a
    long capture and multiplying that by 2*pi first would throw away digits
    that matter.
    """
    factor = max(1, int(factor))
    m = raw.size // factor
    if m < MIN_VISIT_SAMPLES:
        return np.empty(0, dtype=np.complex128)
    blocks = raw[:m * factor].reshape(m, factor).astype(np.complex128)
    k = np.arange(factor, dtype=np.float64)
    inner = np.exp(-2j * np.pi * np.mod(cycles * k, 1.0))
    outer_n = np.arange(m, dtype=np.float64) * factor + float(start)
    outer = np.exp(-2j * np.pi * np.mod(cycles * outer_n, 1.0))
    return (blocks @ inner) / factor * outer


def fine_zeropad(y: np.ndarray, fs: float, pad: int = 32) -> float:
    """Zero-padded periodogram peak with a log-parabolic touch-up.

    Rectangular window and no framing, so the interpolation bias that dogs the
    framed estimator is already down at the padding grid's resolution.
    """
    m = y.size
    n = int(2 ** np.ceil(np.log2(max(m * pad, 16))))
    p = np.abs(np.fft.fft(y, n)) ** 2
    k = int(np.argmax(p))
    a = np.log(p[(k - 1) % n] + 1e-300)
    b = np.log(p[k] + 1e-300)
    c = np.log(p[(k + 1) % n] + 1e-300)
    den = a - 2 * b + c
    d = 0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0
    kk = k if k <= n // 2 else k - n
    return (kk + float(np.clip(d, -0.5, 0.5))) * fs / n


def fine_jacobsen(y: np.ndarray, fs: float) -> float:
    """Jacobsen's three-bin complex interpolator on the natural-resolution DFT."""
    m = y.size
    x = np.fft.fft(y)
    k = int(np.argmax(np.abs(x)))
    a, b, c = x[(k - 1) % m], x[k], x[(k + 1) % m]
    den = 2 * b - a - c
    d = float(np.real((a - c) / den)) if abs(den) > 1e-300 else 0.0
    kk = k if k <= m // 2 else k - m
    return (kk + float(np.clip(d, -0.5, 0.5))) * fs / m


def fine_kay(y: np.ndarray, fs: float) -> float:
    """Kay's weighted phase-difference estimator.

    Efficient at high SNR and O(N), but it has a hard threshold: it works on
    raw sample-to-sample phase differences, so once the per-sample SNR drops
    the differences wrap and the estimate collapses. Kept for comparison.
    """
    m = y.size
    if m < 3:
        return 0.0
    dphi = np.angle(y[1:] * np.conj(y[:-1]))
    n = np.arange(m - 1)
    w = 1.0 - ((n - (m / 2 - 1)) / (m / 2)) ** 2
    total = w.sum()
    if total <= 0:
        return 0.0
    return float(np.sum(w / total * dphi) * fs / (2 * np.pi))


def fine_phase_slope(y: np.ndarray, fs: float) -> float:
    """Amplitude-weighted least-squares fit of unwrapped phase against time.

    Demodulated by a coarse estimate first, so the residual slope is far under
    pi per sample and the unwrap cannot take a wrong branch.
    """
    m = y.size
    if m < 3:
        return 0.0
    t = np.arange(m) / fs
    coarse = fine_zeropad(y, fs)
    z = y * np.exp(-2j * np.pi * coarse * t)
    ph = np.unwrap(np.angle(z))
    w = np.abs(z) ** 2
    sw = w.sum()
    if sw <= 0:
        return coarse
    tm = float((w * t).sum() / sw)
    pm = float((w * ph).sum() / sw)
    den = float((w * (t - tm) ** 2).sum())
    if den <= 0:
        return coarse
    slope = float((w * (t - tm) * (ph - pm)).sum() / den)
    return coarse + slope / (2 * np.pi)


def fine_ml(y: np.ndarray, fs: float, pad: int = 4, iters: int = 3) -> float:
    """Maximum likelihood: periodogram peak, then Newton onto the true maximum.

    For a single tone in white Gaussian noise the periodogram maximum IS the
    maximum-likelihood estimate, so this is not a heuristic that happens to
    work -- it is the estimator the Cramer-Rao bound is written about, and it
    attains that bound wherever it is above threshold. Measured at 1.010x the
    bound from -5 dB to 20 dB per sample.

    ``pad`` only has to land Newton inside the right lobe, and Newton does the
    rest: the measured accuracy is 1.010x the bound at every padding factor
    from 2 to 32, so the small one is chosen and the FFT is 2.5x cheaper. The
    zero-padded peak alone, without the Newton step, would need the large one.
    """
    m = y.size
    if m < 3:
        return 0.0
    f = fine_zeropad(y, fs, pad)
    n = np.arange(m) / fs
    limit = 0.5 * fs / m
    for _ in range(iters):
        e = np.exp(-2j * np.pi * f * n)
        s = np.dot(y, e)
        s1 = np.dot(y * (-2j * np.pi * n), e)
        s2 = np.dot(y * (-2j * np.pi * n) ** 2, e)
        d1 = 2.0 * np.real(np.conj(s) * s1)
        d2 = 2.0 * np.real(np.conj(s) * s2 + s1 * np.conj(s1))
        if not np.isfinite(d2) or d2 >= 0:
            break                                # not a maximum; keep the peak
        f -= float(np.clip(d1 / d2, -limit, limit))
    return f


#: Fine estimators, all taking (segment near DC, sample rate) -> Hz.
#: ``peak`` is not here: it is the framed estimator above, which needs the
#: whole spectrogram rather than one segment.
FINE_ESTIMATORS = {
    "ml": fine_ml,
    "zeropad": fine_zeropad,
    "jacobsen": fine_jacobsen,
    "kay": fine_kay,
    "phase-slope": fine_phase_slope,
}
ESTIMATOR_NAMES = ("peak",) + tuple(FINE_ESTIMATORS)


def sd_bias_factor(n: int) -> float:
    """c4(n): a sample standard deviation from n points reads low, by this much.

    E[s] = c4(n) * sigma, not sigma -- 0.94 at five points, 0.97 at ten. It
    matters here because the standard error each point reports is compared
    against the Cramer-Rao bound, and an uncorrected s makes a perfectly
    efficient estimator look like it is beating a bound that cannot be beaten.
    """
    if n < 2:
        return float("nan")
    if n > 300:                          # c4 -> 1; lgamma is exact but pointless
        return 1.0 - 0.25 / n
    return math.sqrt(2.0 / (n - 1)) * math.exp(
        math.lgamma(n / 2.0) - math.lgamma((n - 1) / 2.0))


def crb_hz(snr_lin: float, nsamples: int, ts: float) -> float:
    """Cramer-Rao lower bound, in Hz sd, for one tone in white Gaussian noise.

    ``snr_lin`` is per-sample signal power over noise power. This is the floor
    every estimator here is measured against; nothing can beat it, and the
    honest question about any estimator is only how close to it it gets.
    """
    if snr_lin <= 0 or nsamples < 2:
        return float("inf")
    n = float(nsamples)
    return float(np.sqrt(6.0 / ((2 * np.pi) ** 2 * snr_lin * n * (n * n - 1)
                                * ts * ts)))


@dataclass
class Visit:
    """One dwell on one point: when it was, what it measured, how much to trust it."""
    point: int
    start_s: float          # capture time of the usable part of the dwell
    mid_s: float
    samples: int            # decimated samples actually fitted
    residual_hz: float      # measured minus the coarse model, in Hz
    snr_db: float
    weight: float           # 1 / variance, from the Cramer-Rao bound
    phasor: complex         # complex amplitude at the fitted frequency
    half_split_hz: float    # first half minus second half of the same dwell


def visit_windows(hops: list[Hop], points: int, period_cycles: int,
                  epoch_s: float, capture_s: float,
                  guard_start_s: float = DEFAULT_GUARD_START_S,
                  guard_end_s: float = DEFAULT_GUARD_END_S,
                  cluster: int | None = None,
                  hops_per_period: int | None = None,
                  period_s: float | None = None,
                  drop_band_change: bool = False
                  ) -> list[tuple[int, float, float]]:
    """Every dwell the schedule places inside the capture, head and tail trimmed.

    ``epoch_s`` is what :func:`align_epoch` returns: schedule time = capture
    time + epoch, modulo the period. So a hop the schedule puts at ``s0``
    appears in the capture at ``s0 - epoch`` and every period thereafter.

    The guards are not tuning knobs to be shaved for precision. The head guard
    covers the retune -- the transmitter changes frequency into the start of a
    dwell, so the first part of it is a settling chirp -- and the tail guard
    covers the epoch's own resolution. A coherent fit has no median to throw
    those away for it.
    """
    per = hops_per_period if hops_per_period is not None else points * period_cycles
    if period_s is None:
        period_s = period_duration(hops, points, period_cycles)
    out: list[tuple[int, float, float]] = []
    for hop in hops[:per]:
        if cluster is not None and hop.cluster != cluster:
            continue
        # The first dwell of a block is the one the synthesiser reached by a
        # VCO band search, because it is the only hop that changes cluster.
        # The schedule already made that dwell longer by exactly the settling
        # allowance it carries, so skipping ``hop.settle_s`` leaves a measured
        # window the same length as every other dwell's -- no point is lost and
        # nothing is measured through a band search. ``drop_band_change``
        # throws those dwells away entirely instead. That is a DIAGNOSTIC, not
        # a remedy, and the difference matters: the schedule is periodic, so
        # the points that lead a block lead it every period, and dropping
        # those dwells drops those points outright -- ``points // block`` of
        # them, 2 of 6 at the defaults, which earns a partial-recovery warning
        # and disowns the capture. Use it to ANSWER the question "is the band
        # settle what is wrong?" by comparing the two decodes; fix it by
        # raising --band-extra-ms at BOTH ends.
        if hop.band_change and drop_band_change:
            continue
        dwell = hop.end_s - hop.start_s
        usable = dwell - hop.settle_s
        head = hop.settle_s + min(guard_start_s, MAX_GUARD_FRACTION * usable)
        tail = min(guard_end_s, MAX_GUARD_FRACTION * usable)
        base = (hop.start_s - epoch_s) % period_s
        k = 0
        while True:
            t0 = base + k * period_s + head
            t1 = base + k * period_s + dwell - tail
            if t0 >= capture_s:
                break
            if t1 <= capture_s and t1 > t0:
                out.append((hop.point, t0, t1))
            k += 1
    out.sort(key=lambda v: v[1])
    return out


def measure_visits(capture: Capture, windows: list[tuple[int, float, float]],
                   if_nom: np.ndarray, offset_hz: float, centre_hz: float,
                   fs: float, estimator: str = DEFAULT_ESTIMATOR,
                   decimate: int = DEFAULT_DECIMATE,
                   min_snr_db: float = DEFAULT_VISIT_SNR_DB) -> list[Visit]:
    """Fit each dwell on its own: mix to DC, decimate, estimate the residual.

    The mixdown uses ABSOLUTE capture time, not time from the start of the
    segment. That costs nothing and it means the phase of every visit is
    referred to one common clock, which is what makes the cross-visit
    coherence diagnostic below meaningful rather than vacuous.
    """
    fine = FINE_ESTIMATORS.get(estimator)
    if fine is None:
        raise ValueError(f"unknown fine estimator {estimator!r}; "
                         f"choose from {', '.join(FINE_ESTIMATORS)}")
    fsd = fs / max(1, int(decimate))
    visits: list[Visit] = []
    for point, t0, t1 in windows:
        i0 = int(np.ceil(t0 * fs))
        i1 = int(np.floor(t1 * fs))
        raw = capture.samples(i0, i1 - i0)
        if raw.size < MIN_VISIT_SAMPLES * max(1, int(decimate)):
            continue
        cycles = (if_nom[point] + offset_hz - centre_hz) / fs
        y = mix_and_decimate(raw, cycles, i0, decimate)
        if y.size < MIN_VISIT_SAMPLES:
            continue
        residual = float(fine(y, fsd))

        m = y.size
        tt = np.arange(m) / fsd
        s = complex(np.dot(y, np.exp(-2j * np.pi * residual * tt)))
        amp2 = abs(s) ** 2 / (m * m)
        total = float(np.mean(np.abs(y) ** 2))
        noise = max(total - amp2, total * 1e-12, 1e-300)
        snr = amp2 / noise
        if 10.0 * np.log10(max(snr, 1e-300)) < min_snr_db:
            continue
        # Inverse variance straight from the bound: var ~ 1/(snr * N^3 * Ts^2).
        weight = snr * float(m) * (m * m - 1.0) / (fsd * fsd)
        # Fit each half of the dwell separately too. A clean CW tone gives the
        # same answer twice; a settling chirp that the head guard did not fully
        # cover, or anything else that makes the dwell not-a-tone, shows up as
        # a difference. It turns "is the guard long enough?" from a judgement
        # call into a number the run reports.
        h = m // 2
        if h >= MIN_VISIT_SAMPLES:
            split = float(fine(y[:h], fsd)) - float(fine(y[h:2 * h], fsd))
        else:
            split = float("nan")
        visits.append(Visit(point=point, start_s=t0,
                            mid_s=0.5 * (t0 + t1), samples=m,
                            residual_hz=residual,
                            snr_db=float(10.0 * np.log10(max(snr, 1e-300))),
                            weight=float(weight), phasor=s,
                            half_split_hz=split))
    return visits


def settling_check(visits: list[Visit]) -> tuple[float, float]:
    """(mean half-split, its standard error) in Hz, over every visit.

    Well clear of zero means the dwells are not clean tones where they are
    being fitted -- almost always a retune still settling past the head guard.
    The standard error is what makes the number readable: a split of 0.5 Hz
    with a 0.4 Hz standard error is nothing, and the same split with a 0.01 Hz
    standard error is a bias sitting on every point.
    """
    vals = np.array([v.half_split_hz for v in visits
                     if v.half_split_hz == v.half_split_hz])
    if vals.size < 4:
        return float("nan"), float("nan")
    return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(vals.size))


def fit_common_drift(visits: list[Visit], t_ref: float) -> float:
    """One drift rate in Hz/s, common to every point, fitted jointly.

    The LNB's LO walks about 4.5 Hz/s, which is not noise and does not average
    away: with a repeating schedule each point is always visited at the same
    phase of the period, so each point's mean visit time sits at a fixed
    distance from the capture's centre and the drift lands on it as a fixed
    per-point offset -- about 0.9 Hz peak-to-peak across a 200 ms period, and
    no better for a longer capture. Removing the drift with a common slope,
    fitted across every visit of every point at once, takes that out.

    Solved by sweeping out the per-point means: each point's own offset is a
    nuisance parameter here, so subtracting each point's weighted mean time and
    residual leaves a single slope to fit.
    """
    if len(visits) < 4:
        return 0.0
    pts = np.array([v.point for v in visits])
    t = np.array([v.mid_s for v in visits]) - t_ref
    r = np.array([v.residual_hz for v in visits])
    w = np.array([v.weight for v in visits])
    num = den = 0.0
    for p in np.unique(pts):
        m = pts == p
        if int(m.sum()) < 2:
            continue
        sw = w[m].sum()
        if sw <= 0:
            continue
        dt = t[m] - (w[m] * t[m]).sum() / sw
        dr = r[m] - (w[m] * r[m]).sum() / sw
        num += float((w[m] * dt * dr).sum())
        den += float((w[m] * dt * dt).sum())
    return float(num / den) if den > 0 else 0.0


def combine_visits(visits: list[Visit], point: int,
                   drift_hz_s: float = 0.0, t_ref: float = 0.0
                   ) -> tuple[float, float, float, int, int] | None:
    """Inverse-variance mean of one point's visits, outliers thrown out first.

    Not a median. The visits are independent samples of the same quantity, so
    a mean is the efficient combiner and a median throws away a quarter of the
    precision for nothing -- measured at 1.28x worse, which is the textbook
    ratio. What a median was buying was protection against a visit that landed
    on a hop boundary or a settling chirp, and that is bought here instead by
    an explicit median-absolute-deviation cut, which discards the bad visit
    outright rather than diluting it.

    Returns (mean residual, standard error, bound-predicted standard error,
    used, total). The last is what the Cramer-Rao bound says this many visits
    at this SNR ought to achieve: quoting it beside the measured scatter is
    what turns "the estimator is efficient" from a claim about synthetic
    signals into something the operator's own capture either confirms or
    contradicts.
    """
    mine = [v for v in visits if v.point == point]
    if len(mine) < MIN_VISITS_PER_POINT:
        return None
    r = np.array([v.residual_hz - drift_hz_s * (v.mid_s - t_ref) for v in mine])
    w = np.array([v.weight for v in mine])
    keep = np.ones_like(r, dtype=bool)
    if r.size >= MIN_VISITS_FOR_OUTLIER_CUT:
        med = float(np.median(r))
        mad = float(np.median(np.abs(r - med)))
        scale = max(1.4826 * mad, 1e-9)
        keep = np.abs(r - med) <= OUTLIER_MADS * scale
        if int(keep.sum()) < MIN_VISITS_FOR_OUTLIER_CUT:
            keep = np.ones_like(r, dtype=bool)
    rk, wk = r[keep], w[keep]
    sw = float(wk.sum())
    if sw <= 0:
        return None
    mean = float((wk * rk).sum() / sw)
    n = int(keep.sum())
    if n > 1:
        # Spread of the visits themselves, not the bound's promise: if
        # something unmodelled is moving, this is what shows it.
        var = float((wk * (rk - mean) ** 2).sum() / sw) * n / (n - 1)
        stderr = float(np.sqrt(max(var, 0.0) / n) / sd_bias_factor(n))
    else:
        stderr = float("nan")
    # The weights ARE inverse variances up to the bound's constant: with
    # w = snr * N(N^2-1) / fs^2, the bound is var = 6 / ((2 pi)^2 * w), so the
    # inverse-variance mean of them has variance 6 / ((2 pi)^2 * sum w).
    predicted = float(np.sqrt(6.0 / ((2 * np.pi) ** 2 * sw)))
    return mean, stderr, predicted, n, len(mine)


def visit_coherence(visits: list[Visit], point: int, period_s: float,
                    steps: int = 512, null_draws: int = 24,
                    rng_seed: int = 7) -> tuple[float, float]:
    """Is one point's phase related across visits? Returns (measured, null).

    Free-running transmitter phase would make a joint fit over every visit of a
    point enormously better than averaging them -- precision would grow with
    the 3/2 power of the whole capture rather than the square root of the visit
    count. Whether that is available is a fact about the hardware, not a
    choice, so it is measured rather than assumed.

    The statistic is the coherent sum of the per-visit phasors, aligned by the
    best common frequency, over the incoherent sum:

        max_f  |sum_v S_v exp(-2 pi i f t_v)|^2 / (V * sum_v |S_v|^2)

    which is 1 when the phases line up exactly. What it is when they do NOT is
    emphatically not zero -- maximising over f fits noise, and with only a
    handful of visits it fits it well -- so the null is measured rather than
    reasoned about: the same statistic, on the same magnitudes at the same
    times, with the phases replaced by random ones. Coherence is real only when
    the measured value clears that null, and the report prints both so the
    comparison is visible rather than buried in a threshold.
    """
    mine = [v for v in visits if v.point == point]
    if len(mine) < MIN_VISITS_FOR_COHERENCE:
        return float("nan"), float("nan")
    s_v = np.array([v.phasor for v in mine])
    t = np.array([v.start_s for v in mine])
    power = float(np.sum(np.abs(s_v) ** 2))
    if power <= 0:
        return float("nan"), float("nan")
    span = 1.0 / max(period_s, 1e-12)
    grid = np.linspace(-0.5 * span, 0.5 * span, steps)
    basis = np.exp(-2j * np.pi * np.outer(grid, t))
    scale = len(mine) * power

    def stat(phasors: np.ndarray) -> float:
        return float((np.abs(basis @ phasors) ** 2).max() / scale)

    measured = stat(s_v)
    rng = np.random.default_rng(rng_seed)
    mag = np.abs(s_v)
    null = [stat(mag * np.exp(2j * np.pi * rng.uniform(0, 1, mag.size)))
            for _ in range(null_draws)]
    return measured, float(np.percentile(null, 95))


def combine_visits_coherently(visits: list[Visit], point: int,
                              period_s: float, seed_hz: float,
                              steps: int = 4096) -> float:
    """Joint fit over every visit of one point, using their phases.

    Worth an enormous amount when it applies -- 76x over averaging at four
    visits, 152x at sixteen, because precision then grows with the 3/2 power of
    the whole capture span instead of the square root of the visit count. It
    applies only when the transmitter's phase is actually related between
    visits, which for a synthesiser that retunes away and relocks it is not,
    and which an LNB drifting at 4.5 Hz/s would destroy even if it were. Never
    call this without checking :func:`visit_coherence` first: on incoherent
    input it does not degrade, it locks onto the wrong lobe of the 1/period
    comb and returns a confident answer that is 30-90x WORSE than the average
    it replaced.

    ``seed_hz`` is the incoherent estimate, which picks the lobe; the lobes are
    1/period apart and the incoherent estimate is orders of magnitude tighter
    than that, so the choice is never close.
    """
    mine = [v for v in visits if v.point == point]
    if len(mine) < 3:
        return seed_hz
    s_v = np.array([v.phasor for v in mine])
    t = np.array([v.start_s for v in mine])
    half = 0.5 / max(period_s, 1e-12)
    grid = np.linspace(seed_hz - half, seed_hz + half, steps)
    power = np.abs(np.exp(-2j * np.pi * np.outer(grid, t)) @ s_v) ** 2
    k = int(np.argmax(power))
    if 0 < k < steps - 1:
        a, b, c = power[k - 1], power[k], power[k + 1]
        den = a - 2 * b + c
        d = 0.5 * (a - c) / den if abs(den) > 1e-30 else 0.0
        return float(grid[k] + np.clip(d, -0.5, 0.5) * (grid[1] - grid[0]))
    return float(grid[k])


def measure_points_fine(capture: Capture, hops: list[Hop], points: int,
                        period_cycles: int, epoch_s: float, if_nom: np.ndarray,
                        offset_hz: float, centre_hz: float, fs: float, *,
                        estimator: str = DEFAULT_ESTIMATOR,
                        decimate: int = DEFAULT_DECIMATE,
                        guard_start_s: float = DEFAULT_GUARD_START_S,
                        guard_end_s: float = DEFAULT_GUARD_END_S,
                        min_snr_db: float = DEFAULT_VISIT_SNR_DB,
                        fit_drift: bool = True, combine: str = "incoherent",
                        cluster: int | None = None,
                        hops_per_period: int | None = None,
                        period_s: float | None = None,
                        drop_band_change: bool = False
                        ) -> tuple[list[PointResult], list[Visit], float, int,
                                   list[str]]:
    """Whole-dwell coherent measurement of every point.

    Returns (rows, visits fitted, drift Hz/s, dwells the schedule offered).
    """
    capture_s = capture.nsamples / fs
    windows = visit_windows(hops, points, period_cycles, epoch_s, capture_s,
                            guard_start_s, guard_end_s, cluster=cluster,
                            hops_per_period=hops_per_period,
                            period_s=period_s,
                            drop_band_change=drop_band_change)
    visits = measure_visits(capture, windows, if_nom, offset_hz, centre_hz, fs,
                            estimator=estimator, decimate=decimate,
                            min_snr_db=min_snr_db)
    t_ref = capture_s / 2.0
    drift = fit_common_drift(visits, t_ref) if fit_drift else 0.0
    if period_s is None:
        period_s = period_duration(hops, points, period_cycles)

    measured = {p: combine_visits(visits, p, drift, t_ref)
                for p in range(len(if_nom))}
    coherences = {p: visit_coherence(visits, p, period_s)
                  for p in measured if measured[p] is not None}

    # The gate on coherent combination is decided ONCE, over every point at
    # once, and never per point. A per-point test at any sensible confidence
    # lets roughly one point in twenty through by chance, and one point
    # combined coherently when it should not be lands a whole Hz out and ruins
    # the spread that the other nineteen just earned. The ensemble median is
    # not close to its threshold in either direction, so deciding once is both
    # safer and better founded.
    notes: list[str] = []
    go_coherent = False
    if combine == "coherent" and coherences:
        got = float(np.median([c for c, _ in coherences.values()]))
        null = float(np.median([n for _, n in coherences.values()]))
        go_coherent = got == got and null == null and got > null
        if not go_coherent:
            notes.append(
                f"coherent combination was asked for and refused: measured "
                f"coherence {got:.3f} does not clear its random-phase null of "
                f"{null:.3f}, so the visits carry no usable phase relation. A "
                f"coherent fit on those does not blur, it locks onto the wrong "
                f"lobe of the 1/period comb and is tens of times worse than "
                f"the average it would replace")

    rows: list[PointResult] = []
    for p in range(len(if_nom)):
        combined = measured[p]
        if combined is None:
            continue
        residual, stderr, predicted, used, total = combined
        mine = [v for v in visits if v.point == p]
        coh, null = coherences[p]
        if go_coherent:
            residual = combine_visits_coherently(visits, p, period_s, residual)
            stderr = float("nan")
        error = offset_hz + residual
        rows.append(PointResult(
            point=p, nominal_if_hz=float(if_nom[p]),
            measured_if_hz=float(if_nom[p]) + error, error_hz=error,
            frames_used=used, frames_assigned=total,
            envelope_db=float(np.median([v.snr_db for v in mine])),
            visits_used=used, visits_total=total, stderr_hz=stderr,
            crb_hz=predicted, coherence=coh, coherence_null=null))
    return rows, visits, drift, len(windows), notes


# ---------------------------------------------------------------------------
# one capture, reduced to what the lever-arm fit consumes
# ---------------------------------------------------------------------------
def reduce_capture(rows: list[PointResult]) -> dict:
    """Collapse a capture's points to a straight line in frequency.

    Two numbers come out, and they are worth very different things:

    * ``mean_error_hz`` at ``mean_if_hz`` -- the capture's offset. Every point
      in one capture shares one rx_lo, so it also shares whatever bias that
      tuning carries. This number therefore CANNOT be believed to better than
      that bias (measured at 362 Hz peak to peak across eight tunings), and is
      useful only in contrast with another capture at a distant cluster.
    * ``slope`` -- how the offset varies with IF INSIDE this one capture, over
      the sub-megahertz the cluster spans. A bias common to the whole capture
      cancels out of a slope exactly, so this number is immune to the tuning
      bias, to the LNB's LO error, and to the LNB's drift. It is the same
      quantity the wide lever measures (-d_rx), from a lever arm about a
      thousand times shorter -- which is precisely why it is worth having: the
      two are vulnerable to completely different things, so they check each
      other.

    Weights come from the Cramer-Rao bound, and the reported standard errors
    are then rescaled by sqrt(chi2/dof) of the fit itself, so a capture whose
    points disagree more than the bound allows says so in its own error bars
    instead of quietly claiming a precision it did not achieve.
    """
    if len(rows) < 2:
        return {}
    f = np.array([r.nominal_if_hz for r in rows], dtype=float)
    e = np.array([r.error_hz for r in rows], dtype=float)
    var = np.array([r.crb_hz for r in rows], dtype=float) ** 2
    if not np.all(np.isfinite(var)) or not np.all(var > 0):
        var = np.ones_like(f)
    w = 1.0 / var
    sw = w.sum()
    fbar = float((w * f).sum() / sw)
    x = f - fbar
    sxx = float((w * x * x).sum())
    a = float((w * e).sum() / sw)
    b = float((w * x * e).sum() / sxx) if sxx > 0 else float("nan")
    resid = e - (a + b * x)
    dof = len(rows) - 2
    scale = float(np.sqrt(max((w * resid * resid).sum(), 0.0) / dof)) if dof > 0 \
        else float("nan")
    scale_or_one = scale if scale == scale else 1.0
    return {
        "mean_if_hz": fbar,
        "mean_error_hz": a,
        "mean_error_stderr_hz": float(np.sqrt(1.0 / sw)) * scale_or_one,
        "slope": b,
        "slope_stderr": (float(np.sqrt(1.0 / sxx)) * scale_or_one
                         if sxx > 0 else float("nan")),
        "chi2_scale": scale,
    }


# ---------------------------------------------------------------------------
# the whole decode
# ---------------------------------------------------------------------------
@dataclass
class DecodeResult:
    comb_offset_hz: float
    comb_sharpness: float
    epoch_s: float
    epoch_sigma: float
    period_s: float
    points: int
    recovered: int
    frame_s: float
    nframes: int
    seconds: float
    slot_half_hz: float
    rows: list[PointResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    estimator: str = DEFAULT_ESTIMATOR
    drift_hz_s: float = 0.0
    visits: int = 0
    combine: str = "incoherent"
    settling_hz: float = float("nan")
    settling_stderr_hz: float = float("nan")
    excess_scatter: float = float("nan")
    # Peak over the best rival more than a couple of bins away: how sure the
    # comb search is that it found the RIGHT comb, not merely a comb.
    comb_margin: float = float("inf")
    # ---- lever-arm bookkeeping: what one capture contributes to the fit ----
    cluster: int = 0
    cluster_centre_hz: float = float("nan")
    centre_hz: float = float("nan")     # the rx_lo this capture actually used
    t_abs_s: float = 0.0                # wall time at the middle of the capture
    mean_if_hz: float = float("nan")    # weighted centre of this cluster's IFs
    mean_error_hz: float = float("nan")     # the capture's offset AT mean_if_hz
    mean_error_stderr_hz: float = float("nan")
    slope: float = float("nan")         # d(error)/d(IF) inside this capture
    slope_stderr: float = float("nan")
    chi2_scale: float = float("nan")    # sqrt(chi2/dof) of that straight line

    @property
    def errors_hz(self) -> np.ndarray:
        return np.array([r.error_hz for r in self.rows])

    @property
    def median_error_hz(self) -> float:
        return float(np.median(self.errors_hz)) if self.rows else float("nan")

    @property
    def spread_hz(self) -> float:
        return float(np.std(self.errors_hz)) if self.rows else float("nan")

    @property
    def stderr_hz(self) -> float:
        """Standard error of the median error, from the point-to-point spread.

        The spread is what the points disagree by; this is how well their
        centre is pinned. Reported separately because they answer different
        questions and conflating them is how a tolerance gets overstated.
        """
        n = len(self.rows)
        return float(np.std(self.errors_hz, ddof=1) / np.sqrt(n)) if n > 1 \
            else float("nan")

    @property
    def trustworthy(self) -> bool:
        """True only when nothing at all was flagged.

        This is deliberately the same condition that decides whether the report
        prints its ``CONFIDENCE IS POOR`` banner, because the banner and the
        exit status must never disagree. Anything that earns the banner --
        including a partial recovery, or a frame size that straddles hops --
        also fails a scripted run, so a caller cannot record a number the
        report has just told a human not to use.
        """
        return bool(self.rows) and not self.warnings

    def to_dict(self) -> dict:
        out = {k: v for k, v in asdict(self).items() if k != "rows"}
        out["rows"] = [asdict(r) for r in self.rows]
        out["median_error_hz"] = self.median_error_hz
        out["spread_hz"] = self.spread_hz
        out["trustworthy"] = self.trustworthy
        return out


def decode(source, *, fs: float, centre_hz: float,
           seed: int = DEFAULT_SEED,
           start_hz: int = DEFAULT_HOP_START_HZ,
           stop_hz: int = DEFAULT_HOP_STOP_HZ,
           points: int = DEFAULT_POINTS,
           min_hop_s: float = DEFAULT_MIN_HOP_S,
           jitter: float = DEFAULT_JITTER,
           period_cycles: int = DEFAULT_PERIOD_CYCLES,
           lo_hz: float = DEFAULT_LO_HZ,
           frame: int = DEFAULT_FRAME,
           threshold_db: float = DEFAULT_THRESHOLD_DB,
           search_hz: float = DEFAULT_SEARCH_HZ,
           estimator: str = DEFAULT_ESTIMATOR,
           decimate: int = DEFAULT_DECIMATE,
           guard_start_s: float = DEFAULT_GUARD_START_S,
           guard_end_s: float = DEFAULT_GUARD_END_S,
           fit_drift: bool = True,
           combine: str = "incoherent",
           cluster_plan: ClusterPlan | None = None,
           cluster: int = 0,
           block: int = DEFAULT_BLOCK,
           drop_band_change: bool = False,
           band_extra_s: float = DEFAULT_BAND_EXTRA_S,
           t_abs_s: float = 0.0) -> DecodeResult:
    """Run the five steps over one capture and report per-point error.

    ``source`` is a path to interleaved int16 I/Q or a complex array. How many
    cycles the transmitter ran does not matter and is never asked for: the
    schedule is periodic, so one period regenerated here covers any capture.

    With ``cluster_plan`` given the transmitter is hopping over several widely
    separated clusters and this capture holds exactly one of them. Everything
    below is unchanged -- the comb, the epoch, the whole-dwell fit all see one
    cluster's points and nothing else -- except that the schedule regenerated
    here is the full multi-cluster one and the hops belonging to other clusters
    are marked as dead air rather than as this cluster's points.
    """
    if cluster_plan is None:
        freqs = plan_frequencies(round(start_hz), round(stop_hz), points)
        hops = make_schedule(seed, freqs, min_hop_s, period_cycles + 1, jitter,
                             period_cycles)
        hops_per_period = points * period_cycles
        period_s = period_duration(hops, points, period_cycles)
        which = None
        centre_rf = (freqs[0] + freqs[-1]) / 2.0
    else:
        points = cluster_plan.points
        freqs = cluster_plan.freqs(cluster)
        per_cycle = cluster_plan.clusters * cluster_plan.points
        hops = make_cluster_schedule(seed, cluster_plan, min_hop_s,
                                     period_cycles + 1, block, jitter,
                                     period_cycles, band_extra_s)
        hops_per_period = per_cycle * period_cycles
        period_s = period_duration(hops, points, period_cycles, per_cycle)
        which = cluster
        centre_rf = float(cluster_plan.centres_hz[cluster])
    if_nom = np.asarray(freqs, dtype=float) - lo_hz

    capture = Capture(source, frame)
    window = np.hanning(frame).astype(np.float32)
    fbins = frame_bins(frame, fs, centre_hz)
    frame_s = frame / fs

    offset, sharpness, margin = comb_search(mean_spectrum(capture, window),
                                            fbins, if_nom, search_hz)
    half = slot_half_width(if_nom)
    slots = point_slots(fbins, if_nom, offset, half)
    env, peak = point_envelopes(capture, window, fbins, slots,
                                want_peak=estimator == "peak")
    env_db = envelope_db(env)
    epoch = align_epoch(env_db, hops, points, frame_s, period_cycles,
                        cluster=which, hops_per_period=hops_per_period,
                        period_s=period_s)

    drift = 0.0
    nvisits = offered = 0
    notes: list[str] = []
    settle = settle_err = float("nan")
    if estimator == "peak":
        # The original estimator, kept so the improvement can be demonstrated
        # on identical input rather than asserted.
        rows = measure_points(env_db, peak, epoch.assigned, if_nom, threshold_db)
    else:
        rows, visits, drift, offered, notes = measure_points_fine(
            capture, hops, points, period_cycles, epoch.shift_s, if_nom,
            offset, centre_hz, fs, estimator=estimator, decimate=decimate,
            guard_start_s=guard_start_s, guard_end_s=guard_end_s,
            fit_drift=fit_drift, combine=combine, cluster=which,
            hops_per_period=hops_per_period, period_s=period_s,
            drop_band_change=drop_band_change)
        nvisits = len(visits)
        settle, settle_err = settling_check(visits)

    warnings: list[str] = list(notes)
    if sharpness < MIN_COMB_SHARPNESS:
        warnings.append(
            f"comb sharpness {sharpness:.1f}x is under {MIN_COMB_SHARPNESS:g}x "
            f"-- the comb was not found; the offset below is not a measurement")
    # What the plan itself entitles this capture to. A comb that can be slid
    # onto itself simply cannot show a big margin, so judging every layout
    # against one fixed number would either excuse the dangerous ones or
    # condemn the safe ones.
    tol_hz = float(fbins[1] - fbins[0])
    allowed = 1.0 / max(comb_ambiguity(if_nom, tol_hz), 1e-9)
    if margin < comb_margin_floor(if_nom, tol_hz):
        warnings.append(
            f"the comb search found a rival peak only {margin:.2f}x below the "
            f"winner, where this frequency plan should give about "
            f"{allowed:.1f}x -- the comb may have locked onto a shifted copy "
            f"of itself, which mislabels every point and moves the answer by "
            f"a whole point spacing")
    # In cluster mode this cluster is on air only 1/clusters of the period, so
    # the same capture carries proportionally less alignment evidence. Scale the
    # floor by sqrt(duty), which is how the statistic itself scales.
    duty = 1.0 / cluster_plan.clusters if cluster_plan is not None else 1.0
    sigma_floor = min_epoch_sigma(duty)
    if epoch.sigma < sigma_floor:
        warnings.append(
            f"epoch sigma {epoch.sigma:.1f} is under {sigma_floor:.1f} "
            f"(floor {MIN_EPOCH_SIGMA:g} scaled by sqrt of {duty:.2f} duty) -- "
            f"the alignment is ambiguous, so every point may be mislabelled")
    if len(rows) < points:
        warnings.append(
            f"only {len(rows)} of {points} points recovered -- check the span "
            f"fits the passband and that both ends use the same schedule")
    if min_hop_s / frame_s < 4:
        warnings.append(
            f"dwell spans {min_hop_s/frame_s:.1f} frames; lower --frame or "
            f"raise the dwell (want at least 4, ideally 20+)")
    if estimator != "peak" and offered and nvisits < 0.7 * offered:
        warnings.append(
            f"only {nvisits} of {offered} scheduled dwells were strong enough "
            f"to fit -- the signal is absent or weak for much of the capture, "
            f"so the points that did fit are not a fair sample of it")
    excess = float("nan")
    if estimator != "peak" and rows:
        got = np.array([r.stderr_hz for r in rows])
        want = np.array([r.crb_hz for r in rows])
        ok = np.isfinite(got) & np.isfinite(want) & (want > 0)
        # Below a handful of visits a standard error is itself so uncertain
        # that the ratio says nothing, so it is withheld rather than printed
        # with a caveat nobody reads.
        enough = np.median([r.visits_used for r in rows]) >= MIN_VISITS_FOR_SCATTER
        if ok.sum() >= 3 and enough:
            excess = float(np.median(got[ok] / want[ok]))
            if excess > MAX_EXCESS_SCATTER and ENFORCE_EXCESS_SCATTER:
                warnings.append(
                    f"each point's visits scatter {excess:.1f}x wider than the "
                    f"Cramer-Rao bound allows for their SNR -- the tone is not "
                    f"what the model says it is, so the quoted precision is "
                    f"not the accuracy")
    if (settle == settle and settle_err == settle_err and settle_err > 0
            and abs(settle) > SETTLING_SIGMA * settle_err):
        warnings.append(
            f"the two halves of a dwell disagree by {settle:+.3f} +/-"
            f"{settle_err:.3f} Hz -- the dwells are not clean tones where they "
            f"are being fitted, so raise --guard-start-ms until this comes "
            f"back to zero")
    span = float(if_nom[-1] - if_nom[0])
    # How far the outermost tone actually sits from the tuning, using the
    # comb offset the search just measured rather than a nominal one.
    reach = float(np.max(np.abs(if_nom + offset - centre_hz)))
    if reach > fs * 0.4:
        warnings.append(
            f"the comb reaches {reach/1e3:.0f} kHz from the tuning, past the "
            f"{fs*0.4/1e3:.0f} kHz half-passband -- reduce the tuning dither "
            f"or the cluster span, or raise --fs")
    if span > fs * 0.8:
        warnings.append(
            f"span {span/1e6:.3f} MHz exceeds the usable bandwidth "
            f"({fs*0.8/1e6:.3f} MHz); outer points cannot be heard")

    seconds = capture.nframes * frame_s
    return DecodeResult(
        comb_offset_hz=offset, comb_sharpness=sharpness, comb_margin=margin,
        epoch_s=epoch.shift_s,
        epoch_sigma=epoch.sigma, period_s=epoch.period_s, points=points,
        recovered=len(rows), frame_s=frame_s, nframes=capture.nframes,
        seconds=seconds, slot_half_hz=half, rows=rows,
        warnings=warnings, estimator=estimator, drift_hz_s=drift,
        visits=nvisits, combine=combine, settling_hz=settle,
        settling_stderr_hz=settle_err, excess_scatter=excess,
        cluster=cluster, cluster_centre_hz=centre_rf, centre_hz=centre_hz,
        t_abs_s=t_abs_s + seconds / 2.0, **reduce_capture(rows))


# ---------------------------------------------------------------------------
# synthetic captures: the whole chain is verifiable with no hardware
# ---------------------------------------------------------------------------
def synthesise(*, fs: float, centre_hz: float, seed: int = DEFAULT_SEED,
               start_hz: int = DEFAULT_HOP_START_HZ,
               stop_hz: int = DEFAULT_HOP_STOP_HZ,
               points: int = DEFAULT_POINTS,
               min_hop_s: float = DEFAULT_MIN_HOP_S,
               jitter: float = DEFAULT_JITTER,
               period_cycles: int = DEFAULT_PERIOD_CYCLES,
               lo_hz: float = DEFAULT_LO_HZ, seconds: float = 0.6,
               offset_hz: float = -106e3, start_at_s: float = 0.0,
               snr_db: float = 10.0, noise_seed: int = 1,
               visit_phase: str = "random", settle_s: float = 0.0,
               settle_hz: float = 0.0, drift_hz_s: float = 0.0
               ) -> np.ndarray:
    """Build the capture a perfect receiver would have taken, with a known answer.

    One tone at a time, hopping exactly on the schedule, the whole comb shifted
    by ``offset_hz`` and the capture starting ``start_at_s`` into the schedule.
    Recovering those two numbers is the test.

    ``visit_phase`` is the honest part and the default is the pessimistic one.
    ``"random"`` gives every dwell an independent starting phase, which is what
    a synthesiser that retunes away and relocks actually does: the loop
    reacquires with no memory of where its phase was last time it sat here.
    ``"coherent"`` models one free-running oscillator whose phase is
    ``2*pi*f*t`` throughout, which would let a receiver combine every visit to
    a point coherently and do vastly better. That option exists so the size of
    the prize can be measured, and so the failure when the assumption is wrong
    can be measured too -- it is not the default, because it is not true.

    ``settle_s`` and ``settle_hz`` put a decaying frequency error at the head
    of every dwell, which is what a PLL retuning into the start of a dwell
    looks like. ``drift_hz_s`` walks every point together, as the LNB's LO
    does at about 4.5 Hz/s.
    """
    if visit_phase not in ("random", "coherent"):
        raise ValueError("visit_phase must be 'random' or 'coherent'")
    freqs = plan_frequencies(round(start_hz), round(stop_hz), points)
    if_nom = np.asarray(freqs, dtype=float) - lo_hz
    span_s = start_at_s + seconds
    cycles = max(period_cycles + 1,
                 int(np.ceil(span_s / (min_hop_s * points))) + 2)
    hops = make_schedule(seed, freqs, min_hop_s, cycles, jitter, period_cycles)

    n = int(round(seconds * fs))
    rng = np.random.default_rng(noise_seed)
    sigma = 10.0 ** (-snr_db / 20.0) / np.sqrt(2.0)
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * sigma
    t = np.arange(n) / fs + start_at_s          # time on the schedule's clock
    tau_c = settle_s / 3.0 if settle_s > 0 else 0.0
    for hop in hops:
        if hop.end_s <= start_at_s or hop.start_s >= span_s:
            continue
        lo = int(np.ceil((hop.start_s - start_at_s) * fs))
        hi = int(np.ceil((hop.end_s - start_at_s) * fs))
        lo, hi = max(0, lo), min(n, hi)
        if hi <= lo:
            continue
        baseband = if_nom[hop.point] + offset_hz - centre_hz
        if visit_phase == "coherent" and not (drift_hz_s or tau_c):
            x[lo:hi] += np.exp(2j * np.pi * baseband * t[lo:hi])
            continue
        # Phase is integrated from the instantaneous frequency, so a settling
        # chirp and a drifting LO are modelled rather than approximated.
        local = t[lo:hi] - hop.start_s
        phase = 2 * np.pi * baseband * local
        if visit_phase == "coherent":
            phase += 2 * np.pi * baseband * hop.start_s
        else:
            phase += rng.uniform(0.0, 2 * np.pi)
        if drift_hz_s:
            # A drifting oscillator integrates: phase(t) = 2*pi*(f*t + g*t^2/2)
            # with t measured from the start of the run, not from the start of
            # this dwell. The t0^2/2 term is the phase the drift has already
            # piled up before this visit began, and leaving it out would make
            # a drifting LO look coherent from visit to visit when it is the
            # very thing that destroys that coherence.
            t0 = hop.start_s
            phase += 2 * np.pi * drift_hz_s * (0.5 * t0 * t0 + t0 * local
                                               + 0.5 * local * local)
        if tau_c > 0 and settle_hz:
            phase += (2 * np.pi * settle_hz * tau_c
                      * (1.0 - np.exp(-local / tau_c)))
        x[lo:hi] += np.exp(1j * phase)
    return x.astype(np.complex64)



# ---------------------------------------------------------------------------
# the physical model the whole lever arm is about
# ---------------------------------------------------------------------------
# Three clocks are involved and only two of them can ever be separated:
#
#   f_rf(actual)  = f_rf(nominal) * (1 + d_tx)      the ADF5355's reference
#   f_lo(actual)  = f_lo(nominal) * (1 + d_lnb)     the LNB's free-running LO
#   everything the SDR reports is scaled by 1/(1 + d_rx)
#
# so what a capture actually shows is
#
#   reported_IF = ( f_rf(1+d_tx) - f_lo(1+d_lnb) ) / (1 + d_rx)  -  b(rx_lo)
#
# and since f_rf = f_IF + f_lo, the d_tx term splits into one piece that scales
# with f_IF and one that does not -- exactly the two pieces d_rx and d_lnb
# occupy. d_tx is therefore degenerate with both: what is measured is
# (d_rx - d_tx) and (d_lnb - d_tx). Everything here is referred to the
# transmitter's 125 MHz reference, and that has to be said out loud, because
# "the SDR's clock error" is only meaningful against something.
#
# b(rx_lo) is the receiver's tuning-dependent bias -- 362 Hz peak to peak
# across eight tunings on this hardware. It is a pure additive term: the LO
# lands somewhere other than where it was asked to, and a tone's measured
# frequency moves by exactly minus that amount regardless of where in the
# passband the tone sits. That is why it cancels out of a within-capture
# slope, and why it does NOT cancel out of the difference between two
# captures at different clusters, which is the whole difficulty.

def reported_if_hz(if_nom_hz, *, lo_hz: float, d_rx: float = 0.0,
                   d_lnb: float = 0.0, d_tx: float = 0.0,
                   tuning_bias_hz: float = 0.0, drift_hz_s: float = 0.0,
                   t_rel_s: float = 0.0):
    """What the receiver reports for a point whose nominal IF is ``if_nom_hz``.

    Exact, not linearised: the cross term d_rx * d_lnb * f_LO is about 0.85 Hz
    at the numbers this hardware actually has, which is twenty times the
    per-capture precision the estimator now reaches.
    """
    if_nom = np.asarray(if_nom_hz, dtype=float)
    f_rf = (if_nom + lo_hz) * (1.0 + d_tx)
    f_lo = lo_hz * (1.0 + d_lnb) + drift_hz_s * t_rel_s
    return (f_rf - f_lo) / (1.0 + d_rx) - tuning_bias_hz


def synthesise_cluster(plan: ClusterPlan, cluster: int, *, fs: float,
                       centre_hz: float, seed: int = DEFAULT_SEED,
                       min_hop_s: float = DEFAULT_MIN_HOP_S,
                       block: int = DEFAULT_BLOCK,
                       jitter: float = DEFAULT_JITTER,
                       period_cycles: int = DEFAULT_PERIOD_CYCLES,
                       band_extra_s: float = DEFAULT_BAND_EXTRA_S,
                       lo_hz: float = DEFAULT_LO_HZ, seconds: float = 2.0,
                       start_at_s: float = 0.0, t_abs_s: float = 0.0,
                       d_rx: float = 0.0, d_lnb: float = 0.0,
                       d_tx: float = 0.0, drift_hz_s: float = 0.0,
                       tuning_bias_hz: float = 0.0, snr_db: float = 10.0,
                       noise_seed: int = 1, visit_phase: str = "random",
                       settle_s: float = 0.0, settle_hz: float = 0.0,
                       band_settle_s: float = 0.0, band_settle_hz: float = 0.0
                       ) -> np.ndarray:
    """The capture a receiver tuned to one cluster would take, answer known.

    Only ``cluster``'s tones are put in the samples: the other clusters are
    outside this tuning's passband, which is the whole reason the receiver has
    to visit them one at a time. The schedule is the full multi-cluster one, so
    the gaps where the transmitter is elsewhere are exactly where they will be
    on the air.

    ``band_settle_hz`` puts a much larger and slower settling error on the
    dwells marked ``band_change`` -- the ones the synthesiser reaches by a VCO
    band search. Those are the dwells the decoder discards, and injecting
    something violent there is how that discard is tested rather than assumed.
    """
    if visit_phase not in ("random", "coherent"):
        raise ValueError("visit_phase must be 'random' or 'coherent'")
    if_nom = np.asarray(plan.if_nom(cluster, lo_hz), dtype=float)
    span_s = start_at_s + seconds
    per_cycle = plan.clusters * plan.points
    cycles = max(period_cycles + 1,
                 int(np.ceil(span_s / (min_hop_s * per_cycle))) + 2)
    del per_cycle
    hops = make_cluster_schedule(seed, plan, min_hop_s, cycles, block, jitter,
                                 period_cycles, band_extra_s)

    n = int(round(seconds * fs))
    rng = np.random.default_rng(noise_seed)
    sigma = 10.0 ** (-snr_db / 20.0) / np.sqrt(2.0)
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * sigma
    # Frequency at the capture's own start, and the rate it walks at. The LNB
    # drift enters the reported frequency divided by (1+d_rx), like everything
    # else the receiver measures.
    base = reported_if_hz(if_nom, lo_hz=lo_hz, d_rx=d_rx, d_lnb=d_lnb,
                          d_tx=d_tx, tuning_bias_hz=tuning_bias_hz,
                          drift_hz_s=drift_hz_s, t_rel_s=t_abs_s) - centre_hz
    rate = -drift_hz_s / (1.0 + d_rx)          # Hz per second of capture time
    tau = settle_s / 3.0 if settle_s > 0 else 0.0
    tau_band = band_settle_s / 3.0 if band_settle_s > 0 else 0.0
    for hop in hops:
        if hop.cluster != cluster:
            continue
        if hop.end_s <= start_at_s or hop.start_s >= span_s:
            continue
        lo = max(0, int(np.ceil((hop.start_s - start_at_s) * fs)))
        hi = min(n, int(np.ceil((hop.end_s - start_at_s) * fs)))
        if hi <= lo:
            continue
        t0 = hop.start_s - start_at_s          # capture time this dwell starts
        u = np.arange(lo, hi) / fs - t0        # time since the dwell started
        f0 = base[hop.point] + rate * t0
        # phase = 2 pi integral of (f0 + rate*u) du
        phase = 2 * np.pi * (f0 * u + 0.5 * rate * u * u)
        if visit_phase == "coherent":
            phase += 2 * np.pi * base[hop.point] * t0
        else:
            phase += rng.uniform(0.0, 2 * np.pi)
        hz, t_c = ((band_settle_hz, tau_band) if hop.band_change
                   else (settle_hz, tau))
        if t_c > 0 and hz:
            phase += 2 * np.pi * hz * t_c * (1.0 - np.exp(-u / t_c))
        x[lo:hi] += np.exp(1j * phase)
    return x.astype(np.complex64)

def write_int16(path, samples: np.ndarray, scale: float = 2000.0) -> int:
    """Write a complex array as interleaved little-endian int16, as a radio would."""
    out = np.empty(samples.size * 2, dtype="<i2")
    out[0::2] = np.clip(np.real(samples) * scale, -32768, 32767).astype("<i2")
    out[1::2] = np.clip(np.imag(samples) * scale, -32768, 32767).astype("<i2")
    with open(path, "wb") as fh:
        fh.write(out.tobytes())
    return samples.size


# ---------------------------------------------------------------------------
# hardware, imported only when actually used
# ---------------------------------------------------------------------------
def capture_from_pluto(path: Path, *, uri: str, if_hz: float, fs: float,
                       seconds: float, gain: float,
                       nbuf: int = DEFAULT_NBUF,
                       bw: float | None = None) -> tuple[int, float]:
    """Stream int16 I/Q straight to disk. Returns (samples, elapsed seconds).

    Capture only -- no analysis in the loop -- because anything else falls
    behind real time and drops samples, which would tear holes in the schedule.
    ``adi`` is imported here so --help, --self-test and the tests all run on a
    machine with no radio attached.
    """
    import adi                                             # noqa: PLC0415

    sdr = adi.Pluto(uri=uri)
    sdr.sample_rate = int(fs)
    sdr.rx_rf_bandwidth = int(rx_bandwidth_for(fs, bw))
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = gain
    sdr.rx_lo = int(if_hz)
    sdr.rx_destroy_buffer()
    sdr.rx_buffer_size = nbuf
    sdr.rx()                                    # discard the retune transient

    want = int(seconds * fs)
    total = 0
    started = time.monotonic()
    with open(path, "wb", buffering=1 << 22) as fh:
        while total < want:
            block = np.asarray(sdr.rx())
            out = np.empty(block.size * 2, dtype="<i2")
            out[0::2] = np.real(block).astype("<i2")
            out[1::2] = np.imag(block).astype("<i2")
            fh.write(out.tobytes())
            total += block.size
    return total, time.monotonic() - started


# The AD9363 analog channel filter is only specified over this range, so a
# derived bandwidth gets clamped into it rather than rejected by the driver.
RX_BW_MIN_HZ = 200e3
RX_BW_MAX_HZ = 20e6

# The device serves a single-shot buffer out of its CMA pool, which is 64 MiB
# in total. A request anywhere near that only succeeds against a nearly
# unfragmented pool, and when it fails it does not fail cleanly: the data path
# wedges and every subsequent client -- any size, any rate -- times out until
# iiod is restarted on the device. Half the pool is the largest request that
# has been reliable here, so refuse above it rather than roll the dice and
# take the radio down. See pluto-plus-utils#27.
MAX_SINGLE_SHOT_SAMPLES = 8_388_608                        # 32 MiB at 4 B/sample


def rx_bandwidth_for(fs: float, override: float | None = None) -> float:
    """Analog RX bandwidth to pair with a sample rate.

    Defaults to 80% of fs: wide enough that the channel filter is not shaping
    the occupied span, narrow enough to still reject the alias band. Widening
    it with fs is what keeps the comparison across sample rates honest -- a
    filter left at 2 MHz while fs went to 20 MS/s would be measuring the
    filter, not the sample rate.
    """
    return min(max(override if override else fs * 0.8, RX_BW_MIN_HZ),
               RX_BW_MAX_HZ)


def capture_single_shot(path: Path, *, uri: str, if_hz: float, fs: float,
                        seconds: float, gain: float,
                        bw: float | None = None) -> tuple[int, float]:
    """Capture the whole run as ONE hardware buffer. Returns (samples, seconds).

    The streaming path above issues repeated rx() calls, and above roughly
    2.5 MS/s this host cannot drain them fast enough -- samples are dropped
    between buffers and the schedule the decoder is trying to time against
    gets torn. Asking for the entire capture as a single buffer, with the
    kernel queue depth set to 1 so nothing can be silently recycled behind
    us, makes the acquisition one contiguous DMA and removes that failure
    mode by construction.

    Readout is slower than real time (USB has to move the whole buffer after
    the fact) but that happens *after* acquisition and does not perforate it,
    so the usual "did we keep up with real time?" check does not apply here.
    """
    import adi                                             # noqa: PLC0415

    sdr = adi.Pluto(uri=uri)
    sdr.sample_rate = int(fs)
    sdr.rx_rf_bandwidth = int(rx_bandwidth_for(fs, bw))
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = gain
    sdr.rx_lo = int(if_hz)
    sdr.rx_destroy_buffer()
    sdr._rxadc.set_kernel_buffers_count(1)

    want = int(seconds * fs)
    if want > MAX_SINGLE_SHOT_SAMPLES:
        raise ValueError(
            f"single-shot capture of {want} samples "
            f"({want * 4 / 2**20:.0f} MiB) exceeds the safe ceiling of "
            f"{MAX_SINGLE_SHOT_SAMPLES} ({MAX_SINGLE_SHOT_SAMPLES * 4 / 2**20:.0f} MiB); "
            f"buffers this large wedge the device data path until iiod is "
            f"restarted. Use a lower --fs, a shorter --seconds "
            f"(<= {MAX_SINGLE_SHOT_SAMPLES / fs:.2f} s at {fs/1e6:g} MS/s), "
            f"or --capture-mode stream.")
    sdr.rx_buffer_size = want
    started = time.monotonic()
    block = np.asarray(sdr.rx())
    elapsed = time.monotonic() - started
    out = np.empty(block.size * 2, dtype="<i2")
    out[0::2] = np.real(block).astype("<i2")
    out[1::2] = np.imag(block).astype("<i2")
    path.write_bytes(out.tobytes())
    return block.size, elapsed


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def format_report(result: DecodeResult) -> str:
    lines = ["",
             f"  comb offset  : {result.comb_offset_hz/1e3:+.3f} kHz "
             f"(sharpness {result.comb_sharpness:.0f}x, "
             f"floor {MIN_COMB_SHARPNESS:g}x; margin over the nearest rival "
             f"{result.comb_margin:.2f}x)",
             f"  epoch        : {result.epoch_s*1e3:.2f} ms of a "
             f"{result.period_s*1e3:.1f} ms period "
             f"(sigma {result.epoch_sigma:.1f}, floor {MIN_EPOCH_SIGMA:g})",
             f"  points       : {result.recovered}/{result.points} recovered "
             f"from {result.nframes} frames ({result.seconds:.2f} s, "
             f"slot +/-{result.slot_half_hz/1e3:.1f} kHz)",
             f"  estimator    : {result.estimator}" + (
                 "  (framed FFT peak, median over frames)"
                 if result.estimator == "peak" else
                 f"  (whole dwell, {result.visits} dwells fitted, "
                 f"drift {result.drift_hz_s:+.2f} Hz/s removed)"),
             "",
             "  point   nominal IF        measured IF         error    +/-"
             "      n    SNR",
             "  " + "-" * 74]
    for row in result.rows:
        lines.append(
            f"  {row.point:5d}   {row.nominal_if_hz/1e6:11.6f} MHz  "
            f"{row.measured_if_hz/1e6:11.6f} MHz  "
            f"{row.error_hz/1e3:+8.3f} kHz  "
            f"{row.stderr_hz:7.3f} "
            f"{row.frames_used:5d} "
            f"{row.envelope_db:6.1f} dB")
    if result.rows:
        lines += ["",
                  f"  median error : {result.median_error_hz/1e3:+.3f} kHz "
                  f"over {result.recovered} points",
                  f"  spread       : {result.spread_hz:.3f} Hz sd "
                  f"(point to point)",
                  f"  centre       : +/-{result.stderr_hz:.3f} Hz standard "
                  f"error on the median"]
        if result.excess_scatter == result.excess_scatter:
            lines.append(
                f"  vs the bound : {result.excess_scatter:.2f}x the "
                f"Cramer-Rao limit for this SNR and dwell (1.0 = nothing left "
                f"on the table, high = something unmodelled is moving)")
        if result.estimator != "peak":
            seen = [(r.coherence, r.coherence_null) for r in result.rows
                    if r.coherence == r.coherence]
            if seen:
                got = float(np.median([c for c, _ in seen]))
                null = float(np.median([n for _, n in seen]))
                verdict = ("phase IS related across visits -- a joint "
                           "coherent fit would do far better"
                           if got > null else
                           "no phase relation across visits, as expected of "
                           "a synthesiser that retunes away and relocks")
                lines.append(
                    f"  coherence    : {got:.3f} against a random-phase null "
                    f"of {null:.3f} -- {verdict}")
            if result.settling_hz == result.settling_hz:
                lines.append(
                    f"  dwell halves : {result.settling_hz:+.3f} +/-"
                    f"{result.settling_stderr_hz:.3f} Hz apart (0 means the "
                    f"fitted part of every dwell is a clean tone)")
    else:
        lines.append("  nothing recovered")
    if result.warnings:
        bar = "  " + "!" * 68
        lines += ["", bar, "  CONFIDENCE IS POOR -- do not use these numbers "
                           "as a measurement:"]
        lines += [f"    * {w}" for w in result.warnings]
        lines += [bar]
    elif result.rows:
        lines += ["", "  confidence good: comb found, epoch unambiguous, "
                      "every point recovered"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sched = p.add_argument_group(
        "schedule (must match the transmitter exactly)")
    sched.add_argument("--seed", type=lambda v: int(v, 0), default=DEFAULT_SEED,
                       help=f"shared schedule seed (default 0x{DEFAULT_SEED:X})")
    sched.add_argument("--start-ghz", type=float,
                       default=DEFAULT_HOP_START_HZ / 1e9,
                       help="first frequency point in GHz")
    sched.add_argument("--stop-ghz", type=float,
                       default=DEFAULT_HOP_STOP_HZ / 1e9,
                       help="last frequency point in GHz")
    sched.add_argument("--points", type=int, default=DEFAULT_POINTS,
                       help=f"number of points (default {DEFAULT_POINTS})")
    sched.add_argument("--min-hop-ms", type=float,
                       default=DEFAULT_MIN_HOP_S * 1e3,
                       help=f"dwell in ms (default {DEFAULT_MIN_HOP_S*1e3:g})")
    sched.add_argument("--jitter", type=float, default=DEFAULT_JITTER,
                       help=f"dwell randomness 0..1 (default {DEFAULT_JITTER:g})")
    sched.add_argument("--period-cycles", type=int,
                       default=DEFAULT_PERIOD_CYCLES,
                       help="permutations before the pattern repeats (default "
                            f"{DEFAULT_PERIOD_CYCLES}); bounds the epoch search")

    clu = p.add_argument_group(
        "clusters (a lever-arm run; --start-ghz and --points are then ignored)")
    clu.add_argument("--clusters", type=int, default=0,
                     help="decode one cluster of a multi-cluster schedule. "
                          "0 (default) means the plain single-cluster hop")
    clu.add_argument("--cluster", type=int, default=0,
                     help="which cluster this capture holds")
    clu.add_argument("--low-ghz", type=float, default=10.70,
                     help="lowest cluster centre in GHz")
    clu.add_argument("--high-ghz", type=float, default=11.90,
                     help="highest cluster centre in GHz")
    clu.add_argument("--cluster-points", type=int,
                     default=DEFAULT_CLUSTER_POINTS,
                     help="points per cluster, on a Golomb ruler")
    clu.add_argument("--span-khz", type=float,
                     default=DEFAULT_CLUSTER_SPAN_HZ / 1e3,
                     help="how wide one cluster is, in kHz")
    clu.add_argument("--block", type=int, default=DEFAULT_BLOCK,
                     help="consecutive dwells per cluster visit")
    clu.add_argument("--band-extra-ms", type=float,
                     default=DEFAULT_BAND_EXTRA_S * 1e3,
                     help="extra dwell on a band-changing hop, all of which "
                          "the receiver skips")
    clu.add_argument("--drop-band-change", action="store_true",
                     help="discard band-changing dwells outright instead of "
                          "skipping their allowance. A DIAGNOSTIC: the "
                          "schedule is periodic, so the points that lead a "
                          "block lead it every period and this loses them "
                          "entirely (2 of 6 at the defaults), which disowns "
                          "the capture. Compare the two decodes to find out "
                          "whether the band settle is the problem; fix it by "
                          "raising --band-extra-ms at BOTH ends")

    rx = p.add_argument_group("receive chain")
    rx.add_argument("--lo-hz", type=float, default=DEFAULT_LO_HZ,
                    help="nominal LNB LO (default 9.75e9, low band)")
    rx.add_argument("--lo-error-hz", type=float, default=DEFAULT_LO_ERROR_HZ,
                    help="known LO error; only shifts the tuning so the comb "
                         "stays centred")
    rx.add_argument("--if-hz", type=float, default=None,
                    help="tune here instead of the derived centre")
    rx.add_argument("--fs", type=float, default=DEFAULT_FS)
    rx.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    rx.add_argument("--gain", type=float, default=DEFAULT_GAIN)
    rx.add_argument("--uri", default=DEFAULT_URI)
    rx.add_argument("--nbuf", type=int, default=DEFAULT_NBUF,
                    help="streaming buffer size (--capture-mode stream only)")
    rx.add_argument("--capture-mode", choices=("single", "stream"),
                    default="single",
                    help="single: the whole run as one contiguous hardware "
                         "buffer (default, required above ~2.5 MS/s); "
                         "stream: repeated buffers, unlimited length but "
                         "drops samples if the host falls behind")
    rx.add_argument("--rx-bw", type=float, default=None,
                    help="analog RX bandwidth in Hz (default 80%% of --fs)")

    an = p.add_argument_group("analysis")
    an.add_argument("--capture", default=None,
                    help="decode this interleaved-int16 file instead of "
                         "capturing from the radio")
    an.add_argument("--capture-out", default=None,
                    help="write the capture here and keep it")
    an.add_argument("--workdir", default="/tmp",
                    help="where a temporary capture is written (default /tmp)")
    an.add_argument("--frame", type=int, default=DEFAULT_FRAME,
                    help=f"FFT frame in samples (default {DEFAULT_FRAME}); must "
                         f"be well under one dwell")
    an.add_argument("--threshold-db", type=float, default=DEFAULT_THRESHOLD_DB,
                    help="a frame counts only this far above the point's own "
                         f"floor (default {DEFAULT_THRESHOLD_DB:g})")
    an.add_argument("--search-hz", type=float, default=DEFAULT_SEARCH_HZ,
                    help="comb offset search half-range in Hz")
    an.add_argument("--estimator", choices=ESTIMATOR_NAMES,
                    default=DEFAULT_ESTIMATOR,
                    help="how each point's frequency is measured. 'peak' is "
                         "the original framed FFT peak with a median over "
                         "frames -- it has a fixed per-point interpolation "
                         "bias of tens of Hz that no amount of listening "
                         "removes. The rest fit whole dwells coherently; "
                         f"'ml' (default) is maximum likelihood and reaches "
                         "the Cramer-Rao bound")
    an.add_argument("--decimate", type=int, default=DEFAULT_DECIMATE,
                    help="boxcar decimation before the whole-dwell fit "
                         f"(default {DEFAULT_DECIMATE}); costs no precision "
                         "and cuts the noise bandwidth")
    an.add_argument("--guard-start-ms", type=float,
                    default=DEFAULT_GUARD_START_S * 1e3,
                    help="skip this much at the head of every dwell, where "
                         "the synthesiser is still settling from the retune "
                         f"(default {DEFAULT_GUARD_START_S*1e3:g} ms)")
    an.add_argument("--guard-end-ms", type=float,
                    default=DEFAULT_GUARD_END_S * 1e3,
                    help="skip this much at the tail of every dwell "
                         f"(default {DEFAULT_GUARD_END_S*1e3:g} ms)")
    an.add_argument("--combine", choices=COMBINE_NAMES, default="incoherent",
                    help="how one point's many dwells are combined. "
                         "'incoherent' (default) averages them by inverse "
                         "variance and gains sqrt(visits). 'coherent' fits "
                         "their phases jointly and gains far more, but only "
                         "if the transmitter's phase is actually related "
                         "between visits -- it is refused per point where the "
                         "measured coherence says otherwise")
    an.add_argument("--no-drift-fit", action="store_true",
                    help="do not fit and remove a common linear drift; the "
                         "LNB LO walks about 4.5 Hz/s and a repeating "
                         "schedule turns that into a fixed per-point offset")
    an.add_argument("--json", action="store_true",
                    help="print the result as JSON as well")
    an.add_argument("--self-test", action="store_true",
                    help="synthesise a capture with a known offset and decode "
                         "it; no radio, no transmission")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    min_hop_s = args.min_hop_ms / 1e3
    band_extra_s = args.band_extra_ms / 1e3
    cluster_plan = None
    if args.clusters:
        cluster_plan = plan_clusters(
            cluster_centres(args.low_ghz * 1e9, args.high_ghz * 1e9,
                            args.clusters),
            args.cluster_points, round(args.span_khz * 1e3))
        freqs = cluster_plan.freqs(args.cluster)
        start_hz, stop_hz = freqs[0], freqs[-1]
        hops = make_cluster_schedule(args.seed, cluster_plan, min_hop_s,
                                     args.period_cycles + 1, args.block,
                                     args.jitter, args.period_cycles,
                                     band_extra_s)
        print(describe_clusters(cluster_plan, hops, args.seed, min_hop_s,
                                args.block, args.lo_hz, args.period_cycles,
                                band_extra_s))
        print(f"  decoding : cluster {args.cluster} of {cluster_plan.clusters}"
              f", centred {cluster_plan.centres_hz[args.cluster]/1e9:.4f} GHz")
    else:
        start_hz = round(args.start_ghz * 1e9)
        stop_hz = round(args.stop_ghz * 1e9)
        freqs = plan_frequencies(start_hz, stop_hz, args.points)
        hops = make_schedule(args.seed, freqs, min_hop_s,
                             args.period_cycles + 1, args.jitter,
                             args.period_cycles)
        print(describe(hops, freqs, args.seed, min_hop_s, args.jitter))
    centre = (args.if_hz if args.if_hz is not None
              else (start_hz + stop_hz) / 2 - args.lo_hz - args.lo_error_hz)
    print(f"\n  tuning   : {centre/1e6:.3f} MHz at {args.fs/1e6:g} MS/s "
          f"({args.fs*0.8/1e6:.2f} MHz usable, span "
          f"{(stop_hz-start_hz)/1e6:.3f} MHz)")
    print(f"  frame    : {args.frame} = {args.frame/args.fs*1e3:.3f} ms "
          f"({min_hop_s/(args.frame/args.fs):.1f} frames per dwell)")

    common = dict(fs=args.fs, centre_hz=centre, seed=args.seed,
                  start_hz=start_hz, stop_hz=stop_hz, points=args.points,
                  min_hop_s=min_hop_s, jitter=args.jitter,
                  period_cycles=args.period_cycles, lo_hz=args.lo_hz)
    if cluster_plan is not None:
        common |= dict(cluster_plan=cluster_plan, cluster=args.cluster,
                       block=args.block, band_extra_s=band_extra_s,
                       drop_band_change=args.drop_band_change)

    tmp_path = None
    if args.self_test:
        print("\n  SELF TEST: synthesising a capture, nothing is transmitted "
              "and no radio is opened")
        if cluster_plan is not None:
            source = synthesise_cluster(
                cluster_plan, args.cluster, fs=args.fs, centre_hz=centre,
                seed=args.seed, min_hop_s=min_hop_s, block=args.block,
                jitter=args.jitter, period_cycles=args.period_cycles,
                band_extra_s=band_extra_s, lo_hz=args.lo_hz,
                seconds=min(args.seconds, 2.0), start_at_s=0.0371,
                d_rx=8.94e-6, d_lnb=DEFAULT_LO_ERROR_HZ / args.lo_hz)
        else:
            source = synthesise(seconds=min(args.seconds, 1.0),
                                offset_hz=-106e3, start_at_s=0.0371, **common)
    elif args.capture:
        source = args.capture
        size = os.path.getsize(source)
        print(f"\n  reading  : {source} ({size/1e6:.1f} MB = "
              f"{size/4/args.fs:.2f} s at {args.fs/1e6:g} MS/s)")
    else:
        out = args.capture_out
        if out is None:
            work = Path(args.workdir)
            work.mkdir(parents=True, exist_ok=True)
            tmp_path = work / f"hop-{uuid.uuid4().hex[:12]}.iq"
            out = str(tmp_path)
        bw = rx_bandwidth_for(args.fs, args.rx_bw)
        print(f"\n  capturing {args.seconds:g} s from {args.uri} -> {out}")
        print(f"  {args.fs/1e6:g} MS/s, {bw/1e6:g} MHz RF bandwidth, "
              f"{args.capture_mode} buffer")
        if args.capture_mode == "single":
            count, elapsed = capture_single_shot(
                Path(out), uri=args.uri, if_hz=centre, fs=args.fs,
                seconds=args.seconds, gain=args.gain, bw=args.rx_bw)
            print(f"  captured {count} samples as one contiguous buffer "
                  f"({count/args.fs*1e3:.1f} ms of signal, read out in "
                  f"{elapsed:.2f} s)")
            live = 1.0
        else:
            count, elapsed = capture_from_pluto(
                Path(out), uri=args.uri, if_hz=centre, fs=args.fs,
                seconds=args.seconds, gain=args.gain, nbuf=args.nbuf,
                bw=args.rx_bw)
            live = count / args.fs / elapsed
            print(f"  captured {count} samples in {elapsed:.2f} s "
                  f"({live*100:.1f}% of real time)")
        if live < 0.98:
            print("  WARNING: capture fell behind real time; the timeline is "
                  "broken and alignment will suffer")
        source = out

    try:
        result = decode(source, frame=args.frame,
                        threshold_db=args.threshold_db,
                        search_hz=args.search_hz, estimator=args.estimator,
                        decimate=args.decimate,
                        guard_start_s=args.guard_start_ms / 1e3,
                        guard_end_s=args.guard_end_ms / 1e3,
                        fit_drift=not args.no_drift_fit,
                        combine=args.combine, **common)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()

    print(format_report(result))
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=float))
    return 0 if result.trustworthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
