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
5. **Measure.** Per point, the median interpolated peak frequency over the
   frames the schedule assigns to it, keeping only frames whose envelope stands
   clear of that point's own noise floor.

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

    # no hardware, no capture: synthesise a known answer and recover it
    tools/hop_decode.py --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adf5355.hopper import (DEFAULT_HOP_START_HZ, DEFAULT_HOP_STOP_HZ,  # noqa: E402
                            DEFAULT_JITTER, DEFAULT_MIN_HOP_S,
                            DEFAULT_PERIOD_CYCLES, DEFAULT_POINTS,
                            DEFAULT_SEED, Hop, describe, make_schedule,
                            period_duration, plan_frequencies)

# ---- receive-side defaults; sdr_listen.sh mirrors these -------------------
DEFAULT_LO_HZ = 9.75e9          # nominal LNB LO: 13 V, no tone = low band
DEFAULT_LO_ERROR_HZ = 94_000.0  # measured; used only to centre the receiver
DEFAULT_FS = 2.5e6              # about 2 MHz usable, comfortably over 1.71
DEFAULT_FRAME = 512             # 204.8 us at 2.5 MS/s: ~49 frames per 10 ms
DEFAULT_SECONDS = 8.0
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
MIN_EPOCH_SIGMA = 10.0


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
def comb_offset(mean_power: np.ndarray, fbins: np.ndarray,
                if_nom: np.ndarray, search_hz: float = DEFAULT_SEARCH_HZ,
                step_hz: float | None = None) -> tuple[float, float]:
    """Slide the expected comb over the spectrum; return (offset, sharpness).

    Every point shares one offset -- the LNB's LO error plus the receiver's
    clock error are common to all of them -- so the whole comb moves as one
    rigid object and matching it is a single-parameter search. Summed energy at
    the expected bins is the score; peak over median is how far the winner
    stands above the field.
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
    return float(offsets[best]), sharpness


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
                    slots: list[tuple[int, int]]
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Per point and frame: peak power in the slot, and where that peak sits.

    The frequency is refined by a parabola through the log magnitudes either
    side of the winning bin, which is worth roughly two orders of magnitude
    against the raw bin width.
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
                oversample: int = 2) -> EpochFit:
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
    per = points * period_cycles
    if len(hops) < per + 1:
        raise ValueError("schedule is shorter than one period")
    period_s = period_duration(hops, points, period_cycles)
    ends = np.array([h.end_s for h in hops[:per]])
    expect = np.array([h.point for h in hops[:per]])
    nframes = env_db.shape[1]
    ft = np.arange(nframes) * frame_s
    cols = np.arange(nframes)
    shifts = np.arange(0.0, period_s, frame_s / oversample)
    scores = np.empty(len(shifts))
    for k, shift in enumerate(shifts):
        idx = np.clip(np.searchsorted(ends, (ft + shift) % period_s), 0, per - 1)
        scores[k] = env_db[expect[idx], cols].mean()
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
           search_hz: float = DEFAULT_SEARCH_HZ) -> DecodeResult:
    """Run the five steps over one capture and report per-point error.

    ``source`` is a path to interleaved int16 I/Q or a complex array. How many
    cycles the transmitter ran does not matter and is never asked for: the
    schedule is periodic, so one period regenerated here covers any capture.
    """
    freqs = plan_frequencies(round(start_hz), round(stop_hz), points)
    hops = make_schedule(seed, freqs, min_hop_s, period_cycles + 1, jitter,
                         period_cycles)
    if_nom = np.asarray(freqs, dtype=float) - lo_hz

    capture = Capture(source, frame)
    window = np.hanning(frame).astype(np.float32)
    fbins = frame_bins(frame, fs, centre_hz)
    frame_s = frame / fs

    offset, sharpness = comb_offset(mean_spectrum(capture, window), fbins,
                                    if_nom, search_hz)
    half = slot_half_width(if_nom)
    slots = point_slots(fbins, if_nom, offset, half)
    env, peak = point_envelopes(capture, window, fbins, slots)
    env_db = envelope_db(env)
    epoch = align_epoch(env_db, hops, points, frame_s, period_cycles)
    rows = measure_points(env_db, peak, epoch.assigned, if_nom, threshold_db)

    warnings: list[str] = []
    if sharpness < MIN_COMB_SHARPNESS:
        warnings.append(
            f"comb sharpness {sharpness:.1f}x is under {MIN_COMB_SHARPNESS:g}x "
            f"-- the comb was not found; the offset below is not a measurement")
    if epoch.sigma < MIN_EPOCH_SIGMA:
        warnings.append(
            f"epoch sigma {epoch.sigma:.1f} is under {MIN_EPOCH_SIGMA:g} -- the "
            f"alignment is ambiguous, so every point may be mislabelled")
    if len(rows) < points:
        warnings.append(
            f"only {len(rows)} of {points} points recovered -- check the span "
            f"fits the passband and that both ends use the same schedule")
    if min_hop_s / frame_s < 4:
        warnings.append(
            f"dwell spans {min_hop_s/frame_s:.1f} frames; lower --frame or "
            f"raise the dwell (want at least 4, ideally 20+)")
    span = float(if_nom[-1] - if_nom[0])
    if span > fs * 0.8:
        warnings.append(
            f"span {span/1e6:.3f} MHz exceeds the usable bandwidth "
            f"({fs*0.8/1e6:.3f} MHz); outer points cannot be heard")

    return DecodeResult(
        comb_offset_hz=offset, comb_sharpness=sharpness, epoch_s=epoch.shift_s,
        epoch_sigma=epoch.sigma, period_s=epoch.period_s, points=points,
        recovered=len(rows), frame_s=frame_s, nframes=capture.nframes,
        seconds=capture.nframes * frame_s, slot_half_hz=half, rows=rows,
        warnings=warnings)


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
               snr_db: float = 10.0, noise_seed: int = 1) -> np.ndarray:
    """Build the capture a perfect receiver would have taken, with a known answer.

    One tone at a time, hopping exactly on the schedule, the whole comb shifted
    by ``offset_hz`` and the capture starting ``start_at_s`` into the schedule.
    Recovering those two numbers is the test.
    """
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
    for hop in hops:
        if hop.end_s <= start_at_s or hop.start_s >= span_s:
            continue
        lo = int(np.ceil((hop.start_s - start_at_s) * fs))
        hi = int(np.ceil((hop.end_s - start_at_s) * fs))
        lo, hi = max(0, lo), min(n, hi)
        if hi <= lo:
            continue
        baseband = if_nom[hop.point] + offset_hz - centre_hz
        x[lo:hi] += np.exp(2j * np.pi * baseband * t[lo:hi])
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
                       nbuf: int = DEFAULT_NBUF) -> tuple[int, float]:
    """Stream int16 I/Q straight to disk. Returns (samples, elapsed seconds).

    Capture only -- no analysis in the loop -- because anything else falls
    behind real time and drops samples, which would tear holes in the schedule.
    ``adi`` is imported here so --help, --self-test and the tests all run on a
    machine with no radio attached.
    """
    import adi                                             # noqa: PLC0415

    sdr = adi.Pluto(uri=uri)
    sdr.sample_rate = int(fs)
    sdr.rx_rf_bandwidth = int(fs * 0.8)
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


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def format_report(result: DecodeResult) -> str:
    lines = ["",
             f"  comb offset  : {result.comb_offset_hz/1e3:+.3f} kHz "
             f"(sharpness {result.comb_sharpness:.0f}x, "
             f"floor {MIN_COMB_SHARPNESS:g}x)",
             f"  epoch        : {result.epoch_s*1e3:.2f} ms of a "
             f"{result.period_s*1e3:.1f} ms period "
             f"(sigma {result.epoch_sigma:.1f}, floor {MIN_EPOCH_SIGMA:g})",
             f"  points       : {result.recovered}/{result.points} recovered "
             f"from {result.nframes} frames ({result.seconds:.2f} s, "
             f"slot +/-{result.slot_half_hz/1e3:.1f} kHz)",
             "",
             "  point   nominal IF        measured IF         error   frames"
             "   env",
             "  " + "-" * 68]
    for row in result.rows:
        lines.append(
            f"  {row.point:5d}   {row.nominal_if_hz/1e6:11.6f} MHz  "
            f"{row.measured_if_hz/1e6:11.6f} MHz  "
            f"{row.error_hz/1e3:+8.3f} kHz  "
            f"{row.frames_used:4d}/{row.frames_assigned:<4d} "
            f"{row.envelope_db:5.1f} dB")
    if result.rows:
        lines += ["",
                  f"  median error : {result.median_error_hz/1e3:+.3f} kHz "
                  f"over {result.recovered} points",
                  f"  spread       : {result.spread_hz:.0f} Hz sd "
                  f"(point to point)"]
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
    rx.add_argument("--nbuf", type=int, default=DEFAULT_NBUF)

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
    an.add_argument("--json", action="store_true",
                    help="print the result as JSON as well")
    an.add_argument("--self-test", action="store_true",
                    help="synthesise a capture with a known offset and decode "
                         "it; no radio, no transmission")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    start_hz = round(args.start_ghz * 1e9)
    stop_hz = round(args.stop_ghz * 1e9)
    min_hop_s = args.min_hop_ms / 1e3
    centre = (args.if_hz if args.if_hz is not None
              else (start_hz + stop_hz) / 2 - args.lo_hz - args.lo_error_hz)

    freqs = plan_frequencies(start_hz, stop_hz, args.points)
    hops = make_schedule(args.seed, freqs, min_hop_s, args.period_cycles + 1,
                         args.jitter, args.period_cycles)
    print(describe(hops, freqs, args.seed, min_hop_s, args.jitter))
    print(f"\n  tuning   : {centre/1e6:.3f} MHz at {args.fs/1e6:g} MS/s "
          f"({args.fs*0.8/1e6:.2f} MHz usable, span "
          f"{(stop_hz-start_hz)/1e6:.3f} MHz)")
    print(f"  frame    : {args.frame} = {args.frame/args.fs*1e3:.3f} ms "
          f"({min_hop_s/(args.frame/args.fs):.1f} frames per dwell)")

    common = dict(fs=args.fs, centre_hz=centre, seed=args.seed,
                  start_hz=start_hz, stop_hz=stop_hz, points=args.points,
                  min_hop_s=min_hop_s, jitter=args.jitter,
                  period_cycles=args.period_cycles, lo_hz=args.lo_hz)

    tmp_path = None
    if args.self_test:
        print("\n  SELF TEST: synthesising a capture, nothing is transmitted "
              "and no radio is opened")
        source = synthesise(seconds=min(args.seconds, 1.0), offset_hz=-106e3,
                            start_at_s=0.0371, **common)
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
        print(f"\n  capturing {args.seconds:g} s from {args.uri} -> {out}")
        count, elapsed = capture_from_pluto(
            Path(out), uri=args.uri, if_hz=centre, fs=args.fs,
            seconds=args.seconds, gain=args.gain, nbuf=args.nbuf)
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
                        search_hz=args.search_hz, **common)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()

    print(format_report(result))
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=float))
    return 0 if result.trustworthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
