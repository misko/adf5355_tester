"""Seeded pseudorandom frequency hopping, reproducible at both ends.

A transmitter hops among a set of frequencies in an order, and for dwells, that
are derived entirely from a shared seed. A receiver that knows the seed and the
frequency plan regenerates the identical schedule, so it never has to *infer*
which point it is hearing -- it only has to find the epoch, after which every
hop is known.

Why this beats duration coding
------------------------------
* Identity comes from the schedule, not from measuring a burst length. Duration
  estimation was the fragile step: it needs hysteresis, gap merging and
  tolerance, and it fails outright when adjacent points share the capture band.
* Random order decorrelates frequency from time. A monotonic ladder steps
  frequency in lockstep with time, so oscillator drift lands squarely on the
  frequency-dependent term being measured. Randomising makes drift orthogonal
  to it instead of confounded with it.
* A pseudorandom pattern autocorrelates sharply, so epoch alignment is
  unambiguous. A monotonic ramp correlates broadly and aligns poorly.
* It is much faster. Duration coding costs u*N*(N+1) per cycle and spends half
  of that muted; hopping costs sum(dwell) with no gaps at all.

Dwell law
---------
``dwell = min_hop * (1 + jitter * random(0,1) * 2)``.

``jitter = 0`` gives a fixed dwell, which is the default and is what a receiver
wants: identity comes entirely from the seeded *frequency* order, so nothing
needs to be encoded in time, and a uniform grid makes alignment a single
one-dimensional search instead of a per-burst duration estimate. Duration
estimation was the fragile step -- it needs hysteresis, gap merging and
tolerance, and it degrades badly when the signal is marginal.

``jitter = 1`` gives ``[min_hop, 3*min_hop]``, i.e. uniform with mean
``2*min_hop``, for cases where varying dwell is wanted as well.

Reproducibility
---------------
Python's ``random`` is not a stable cross-language contract, so the generator
here is SplitMix64 -- about ten lines, identical in any language, and the same
sequence for a given seed forever. Both ends must agree on seed, the frequency
plan, ``min_hop`` and the hop count; nothing else is transmitted.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

_MASK = (1 << 64) - 1
DEFAULT_SEED = 0xC0FFEE
DEFAULT_POINTS = 20
DEFAULT_MIN_HOP_S = 0.005
DEFAULT_JITTER = 0.0     # fixed dwell: the frequency order carries identity


class SplitMix64:
    """Deterministic 64-bit generator, trivially portable to other languages."""

    __slots__ = ("_state",)

    def __init__(self, seed: int) -> None:
        self._state = seed & _MASK

    def next_u64(self) -> int:
        self._state = (self._state + 0x9E3779B97F4A7C15) & _MASK
        z = self._state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
        return z ^ (z >> 31)

    def uniform(self) -> float:
        """Uniform in [0, 1) from the top 53 bits, as IEEE doubles do it."""
        return (self.next_u64() >> 11) * (2.0 ** -53)

    def below(self, bound: int) -> int:
        """Unbiased integer in [0, bound) by rejection."""
        if bound <= 0:
            raise ValueError("bound must be positive")
        limit = _MASK - (_MASK % bound)
        while True:
            value = self.next_u64()
            if value <= limit:
                return value % bound


@dataclass(frozen=True)
class Hop:
    sequence: int        # index within the whole run
    cycle: int           # which permutation this came from
    point: int           # 0-based index into the frequency plan
    freq_hz: int
    dwell_s: float
    start_s: float       # offset from the start of the run
    end_s: float


def plan_frequencies(start_hz: int, stop_hz: int, points: int) -> list[int]:
    if points < 2:
        raise ValueError("need at least 2 frequency points")
    if stop_hz < start_hz:
        raise ValueError("stop must not be below start")
    step = (stop_hz - start_hz) / (points - 1)
    return [round(start_hz + step * i) for i in range(points)]


def _permutation(rng: SplitMix64, n: int) -> list[int]:
    """Fisher-Yates, so every point is visited exactly once per cycle."""
    order = list(range(n))
    for i in range(n - 1, 0, -1):
        j = rng.below(i + 1)
        order[i], order[j] = order[j], order[i]
    return order


def make_schedule(seed: int, freqs: list[int], min_hop_s: float,
                  cycles: int, jitter: float = DEFAULT_JITTER,
                  period_cycles: int = 1) -> list[Hop]:
    """Regenerate the exact hop sequence from the shared parameters.

    The pattern repeats every ``period_cycles`` permutations. Periodicity is
    what makes the receiver cheap: it need only search one period for the
    epoch, instead of the whole run. A repeated random permutation still
    autocorrelates sharply, so alignment stays unambiguous; raising
    ``period_cycles`` trades a longer search for a longer unique sequence.
    """
    if min_hop_s <= 0:
        raise ValueError("min_hop_s must be positive")
    if cycles < 1:
        raise ValueError("need at least one cycle")
    if not 0.0 <= jitter <= 1.0:
        raise ValueError("jitter must be between 0 and 1")
    if period_cycles < 1:
        raise ValueError("period_cycles must be at least 1")
    rng = SplitMix64(seed)

    # One period of (point, dwell), then repeated.
    period: list[tuple[int, float]] = []
    for _ in range(period_cycles):
        for point in _permutation(rng, len(freqs)):
            period.append((point, min_hop_s * (1.0 + jitter * rng.uniform() * 2.0)))

    hops: list[Hop] = []
    t = 0.0
    for seq in range(cycles * len(freqs)):
        point, dwell = period[seq % len(period)]
        hops.append(Hop(sequence=seq, cycle=seq // len(freqs), point=point,
                        freq_hz=freqs[point], dwell_s=dwell,
                        start_s=t, end_s=t + dwell))
        t += dwell
    return hops


def period_duration(hops: list[Hop], points: int, period_cycles: int = 1) -> float:
    """Wall time of one repeat of the pattern -- the receiver's search range."""
    n = points * period_cycles
    return hops[n].start_s if len(hops) > n else hops[-1].end_s


def cycle_duration(hops: list[Hop], cycle: int) -> float:
    inside = [h for h in hops if h.cycle == cycle]
    return inside[-1].end_s - inside[0].start_s if inside else 0.0


def describe(hops: list[Hop], freqs: list[int], seed: int,
             min_hop_s: float, jitter: float = DEFAULT_JITTER) -> str:
    dwells = [h.dwell_s for h in hops]
    cycles = hops[-1].cycle + 1
    spacing = (freqs[-1] - freqs[0]) / (len(freqs) - 1)
    lines = [
        "",
        f"Seeded frequency hop schedule  (seed 0x{seed:X})",
        f"  points   : {len(freqs)}  from {freqs[0]/1e9:.6f} to "
        f"{freqs[-1]/1e9:.6f} GHz  (spacing {spacing/1e3:.1f} kHz, "
        f"span {(freqs[-1]-freqs[0])/1e6:.3f} MHz)",
        (f"  dwell    : fixed {min_hop_s*1e3:.2f} ms" if jitter == 0 else
         f"  dwell    : {min_hop_s*1e3:.1f} to {min_hop_s*(1+2*jitter)*1e3:.1f}"
         f" ms (jitter {jitter:g}, mean {sum(dwells)/len(dwells)*1e3:.2f} ms)"),
        f"  hops     : {len(hops)} over {cycles} cycles, "
        f"total {hops[-1].end_s:.3f} s",
        f"  cycle    : {cycle_duration(hops, 0):.4f} s for the first "
        f"permutation (varies with the draw)",
        f"  first 8  : " + ", ".join(f"p{h.point}@{h.dwell_s*1e3:.1f}ms"
                                     for h in hops[:8]),
    ]
    return "\n".join(lines)


def _sleep_until(deadline: float, spin_margin_s: float = 0.002) -> None:
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        if remaining > spin_margin_s:
            time.sleep(remaining - spin_margin_s)


def run_hops(dev, hops: list[Hop], channel, settle_s: float = 0.05) -> None:
    """Transmit the schedule, hopping continuously with no muted gaps.

    Identity comes from the schedule rather than from an on/off pattern, so
    there is nothing to gate: the synthesiser simply retunes. The retune itself
    (about 0.84 ms) is the only discontinuity, and it lands inside the dwell it
    precedes rather than eating a coded window.
    """
    # Cold-start once with autocal so the VCO band is chosen for this span,
    # then hop without it. Re-running the band search on every hop would blank
    # the output through mute-till-lock for a large part of each dwell, which
    # at 5 ms leaves almost nothing radiated.
    dev.set_frequency(hops[0].freq_hz, channel)
    if settle_s:
        time.sleep(settle_s)
    dev.set_output(channel, True)
    t0 = time.monotonic()
    for index, hop in enumerate(hops):
        _sleep_until(t0 + hop.end_s)
        nxt = hops[index + 1] if index + 1 < len(hops) else None
        if nxt is not None:
            dev.set_frequency(nxt.freq_hz, channel, autocal=False)
    dev.set_output(channel, False)
