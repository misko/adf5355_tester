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

import gc
import time
from bisect import bisect_left
from dataclasses import dataclass
from functools import lru_cache

_MASK = (1 << 64) - 1
DEFAULT_SEED = 0xC0FFEE
DEFAULT_POINTS = 20
# Precision tracks dwell, because dwell is integration time. Measured on the
# bench over 8 s captures, 20 points: 2 ms gave 2946 Hz sd, 5 ms 1361 Hz,
# 10 ms 730 Hz. 10 ms it is; both ends default to the same number.
DEFAULT_MIN_HOP_S = 0.010
DEFAULT_JITTER = 0.0     # fixed dwell: the frequency order carries identity
DEFAULT_PERIOD_CYCLES = 1
# 300 permutations of 20 points at 10 ms is 60 s: long enough to start the
# transmitter, walk to the receiver, and still capture 8 s of it.
DEFAULT_CYCLES = 300

# A span narrow enough to sit inside a 2.5 MS/s receiver's instantaneous
# bandwidth, so one tuning hears every point: 20 points, 90 kHz apart.
DEFAULT_HOP_START_HZ = 11_000_000_000
DEFAULT_HOP_STOP_HZ = 11_001_710_000


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
    point: int           # 0-based index into the frequency plan OF ITS CLUSTER
    freq_hz: int
    dwell_s: float
    start_s: float       # offset from the start of the run
    end_s: float
    cluster: int = 0     # which cluster this point belongs to
    band_change: bool = False   # the hop before this one was in another cluster
    settle_s: float = 0.0       # head of the dwell that is not yet a clean tone


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


def make_period(seed: int, points: int, min_hop_s: float,
                jitter: float = DEFAULT_JITTER,
                period_cycles: int = DEFAULT_PERIOD_CYCLES
                ) -> list[tuple[int, float]]:
    """One period of (point index, dwell) -- the run is this, repeated.

    The whole schedule is periodic by construction, so anything that only
    needs to *play* it needs this and nothing more. Materialising the full
    run instead costs memory linear in its length, which for a transmitter
    meant to loop indefinitely is unbounded: a 200-point schedule asked to
    run 2,000,000 cycles is 400 million entries, and the Python transmitter
    duly tried to allocate all of them before emitting a single hop.
    """
    if min_hop_s <= 0:
        raise ValueError("min_hop_s must be positive")
    if not 0.0 <= jitter <= 1.0:
        raise ValueError("jitter must be between 0 and 1")
    if period_cycles < 1:
        raise ValueError("period_cycles must be at least 1")
    rng = SplitMix64(seed)
    period: list[tuple[int, float]] = []
    for _ in range(period_cycles):
        for point in _permutation(rng, points):
            period.append((point, min_hop_s * (1.0 + jitter * rng.uniform() * 2.0)))
    return period


def make_schedule(seed: int, freqs: list[int], min_hop_s: float,
                  cycles: int, jitter: float = DEFAULT_JITTER,
                  period_cycles: int = DEFAULT_PERIOD_CYCLES) -> list[Hop]:
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
    period = make_period(seed, len(freqs), min_hop_s, jitter, period_cycles)

    hops: list[Hop] = []
    t = 0.0
    for seq in range(cycles * len(freqs)):
        point, dwell = period[seq % len(period)]
        hops.append(Hop(sequence=seq, cycle=seq // len(freqs), point=point,
                        freq_hz=freqs[point], dwell_s=dwell,
                        start_s=t, end_s=t + dwell))
        t += dwell
    return hops


def period_duration(hops: list[Hop], points: int,
                    period_cycles: int = DEFAULT_PERIOD_CYCLES,
                    hops_per_cycle: int | None = None) -> float:
    """Wall time of one repeat of the pattern -- the receiver's search range.

    ``hops_per_cycle`` defaults to ``points``, which is right for a
    single-cluster schedule where one cycle is one permutation of the points.
    A cluster schedule emits ``clusters * points`` hops per cycle, and passing
    that here is the only difference: the period is still one whole cycle (or
    ``period_cycles`` of them), it is simply longer.
    """
    n = (points if hops_per_cycle is None else hops_per_cycle) * period_cycles
    return hops[n].start_s if len(hops) > n else hops[-1].end_s



# ---------------------------------------------------------------------------
# clusters: the same schedule, spread over a real frequency lever arm
# ---------------------------------------------------------------------------
# One narrow cluster measures the TOTAL offset beautifully and cannot take it
# apart. The offset is
#
#     Df(f_IF) = -d_rx * f_IF  -  d_lnb * f_LO_nom
#
# and across one cluster of well under a megahertz the first term moves by a
# few hertz for a 9 ppm clock, which is nothing. Separating the two needs the same precise local measurement made at
# several widely separated IFs, so that the term that scales with f_IF can be
# told apart from the term that does not. The LNB low band gives 0.95 to
# 2.15 GHz of IF, and that whole range is the lever.
#
# The receiver hears about 2 MHz at once, so it visits one cluster at a time.
# Everything below exists to make each cluster's sub-pattern decodable ON ITS
# OWN, from a capture that contains nothing else:
#
# * Every cluster is visited once per ROUND, in a random order that is redrawn
#   each round. So each cluster's hops are spread evenly through the cycle --
#   no cluster is ever starved, and no cluster sits at a fixed phase of the
#   period, which is what would let the LNB's drift correlate with which
#   cluster is being heard.
# * Within a cluster the point order is an independent seeded permutation, so
#   the sub-pattern one receiver sees is itself pseudorandom and autocorrelates
#   sharply. Alignment stays a one-dimensional search over one period.
# * Hops are emitted in BLOCKS of consecutive same-cluster points. Only the
#   first hop of a block changes cluster, and only that hop needs the VCO band
#   search: within a cluster the span is under a megahertz, far inside one
#   band, so the rest hop with dividers alone exactly as the single-cluster
#   schedule does. ``band_change`` marks those hops, and the schedule makes
#   them longer by ``band_extra_s`` so the receiver can skip the re-acquisition
#   without losing the dwell.
# * The points inside a cluster sit on a Golomb ruler, not on a regular grid.
#   See GOLOMB_RULERS below: a regular comb has an alias one spacing away that
#   a receiver's comb search can and does lock on to.
#
# The period is one whole cycle: every cluster's every point, once. With C
# clusters that is C times longer than the single-cluster period, and a
# receiver on one cluster hears its own points 1/C of the time. That duty
# factor is the price of the lever arm and it is unavoidable with one
# transmitter -- the fix is simply to listen C times as long.

DEFAULT_CLUSTER_CENTRES_HZ = (10_700_000_000, 11_100_000_000,
                              11_500_000_000, 11_900_000_000)
DEFAULT_CLUSTER_POINTS = 6
# What one cluster spans, which is what has to fit the receiver's passband
# with room left over for the tuning dither the receiver needs.
DEFAULT_CLUSTER_SPAN_HZ = 720_000
# Points per block, i.e. consecutive dwells in one cluster. Every block costs
# one VCO band search on its first dwell, and that dwell is longer to pay for
# it, so a larger block wastes less. Too large and a cluster's dwells clump in
# time. 3 of 6 keeps the overhead near a sixth while visiting every cluster
# twice per cycle.
DEFAULT_BLOCK = 3
# A band change costs a VCO band search, and the dwell it lands on is not a
# clean tone while that runs. Rather than throw the dwell away -- which would
# silently cost whichever point drew it, every period, forever -- the schedule
# makes that one dwell LONGER by this much and the receiver skips exactly the
# extra. Every measured window is then the same length whether or not the band
# moved, no point is ever lost, and how long is long enough stops being a
# judgement call: the receiver's half-split check reports it.
DEFAULT_BAND_EXTRA_S = 0.005

# Optimal Golomb rulers. Every pairwise difference of the marks is DISTINCT,
# which is exactly the property a receiver's comb search needs and a uniform
# comb does not have.
#
# Slide a uniformly spaced comb of P points by one spacing and P-1 of its
# points land on neighbours: the comb search then has an alias scoring
# (P-1)/P of the truth, one whole spacing away, and when noise tips it over
# every point is mislabelled and the capture returns a confident answer wrong
# by a spacing. That is not hypothetical -- it happened, on synthetic captures,
# to a four-point cluster whose alias scored 3/4. With distinct differences no
# shift can realign more than ONE pair, so the best alias scores 1/P and there
# is nothing to lock on to but the truth.
#
# The price is that a Golomb ruler of P marks is longer than P-1 units, so for
# a fixed span the smallest gap shrinks as the ruler's length. That is what
# bounds the useful number of points per cluster: at 720 kHz of span, six
# points leave 42 kHz between the closest pair, and twelve would leave 8 kHz,
# which is under two FFT bins and too close to separate.
GOLOMB_RULERS = {
    2: (0, 1),
    3: (0, 1, 3),
    4: (0, 1, 4, 6),
    5: (0, 1, 4, 9, 11),
    6: (0, 1, 4, 10, 12, 17),
    7: (0, 1, 4, 10, 18, 23, 25),
    8: (0, 1, 4, 9, 15, 22, 32, 34),
    9: (0, 1, 5, 12, 25, 27, 35, 41, 44),
    10: (0, 1, 6, 10, 23, 26, 34, 41, 53, 55),
    11: (0, 1, 4, 13, 28, 33, 47, 54, 64, 70, 72),
    12: (0, 2, 6, 24, 29, 40, 43, 55, 68, 75, 76, 85),
}
LAYOUT_QUANTUM_HZ = 1_000


def _coincidence(offsets, tol_hz: float) -> float:
    """Worst fraction of points a rigid shift can put back on top of the comb.

    The diagnostic behind the ruler: a uniform comb scores (P-1)/P and a
    Golomb ruler scores 1/P. It is reported rather than assumed, because the
    marks are scaled and rounded to whole kilohertz on the way to the
    synthesiser and rounding can, in principle, put two differences back on
    top of each other.
    """
    pts = sorted(offsets)
    n = len(pts)
    shifts = sorted({b - a for i, a in enumerate(pts) for b in pts[i:]
                     if b - a > tol_hz})
    worst = 0.0
    for shift in shifts:
        hit = 0
        for x in pts:
            target = x + shift
            i = bisect_left(pts, target - tol_hz)
            if i < n and pts[i] <= target + tol_hz:
                hit += 1
        worst = max(worst, hit / n)
    return worst


@lru_cache(maxsize=256)
def _ruler_offsets(points: int, span_hz: int) -> tuple[int, ...]:
    """A Golomb ruler of ``points`` marks, scaled to exactly ``span_hz``."""
    marks = GOLOMB_RULERS.get(points)
    if marks is None:
        raise ValueError(
            f"no Golomb ruler tabulated for {points} points per cluster "
            f"(2 to {max(GOLOMB_RULERS)} are). More points is not the way to "
            f"more precision here anyway: the total time on air per cluster is "
            f"what sets it, and splitting that among more points leaves each "
            f"one shorter")
    scale = span_hz / marks[-1]
    out = [round(m * scale / LAYOUT_QUANTUM_HZ) * LAYOUT_QUANTUM_HZ
           for m in marks]
    out[0], out[-1] = 0, int(span_hz)
    if len(set(out)) != len(out) or any(b <= a for a, b in zip(out, out[1:])):
        raise ValueError(f"span {span_hz} Hz is too narrow for {points} points; "
                         f"the ruler collapses onto itself")
    return tuple(out)


@dataclass(frozen=True)
class ClusterPlan:
    """C clusters of P points each, laid out on the usable lever arm."""

    centres_hz: tuple[int, ...]
    points: int
    span_hz_: int

    @property
    def clusters(self) -> int:
        return len(self.centres_hz)

    @property
    def offsets(self) -> tuple[int, ...]:
        return _ruler_offsets(self.points, self.span_hz_)

    def freqs(self, cluster: int) -> list[int]:
        """The absolute frequencies of one cluster, low to high.

        Not evenly spaced, and that is the point: see GOLOMB_RULERS above. The
        same ruler is used in every cluster, so nothing about the layout has to
        be transmitted -- both ends build it from the point count and the span.
        """
        if not 0 <= cluster < self.clusters:
            raise ValueError(f"cluster {cluster} outside 0..{self.clusters-1}")
        base = self.centres_hz[cluster] - self.span_hz_ // 2
        return [base + o for o in self.offsets]

    @property
    def all_freqs(self) -> list[int]:
        return [f for c in range(self.clusters) for f in self.freqs(c)]

    def span_hz(self) -> int:
        """Cluster span: what has to fit inside the receiver's passband."""
        return int(self.span_hz_)

    def min_gap_hz(self) -> int:
        """Closest pair in a cluster: what sets the receiver's slot width."""
        o = self.offsets
        return min(b - a for a, b in zip(o, o[1:]))

    def coincidence(self, tol_hz: float = 5_000.0) -> float:
        """Fraction of a cluster a rigid shift could realign. 1/P is ideal."""
        return _coincidence(self.offsets, tol_hz)

    def lever_hz(self) -> int:
        """Distance between the outermost cluster centres -- the whole point."""
        return max(self.centres_hz) - min(self.centres_hz)

    def if_nom(self, cluster: int, lo_hz: float) -> list[float]:
        return [f - lo_hz for f in self.freqs(cluster)]


def plan_clusters(centres_hz, points: int = DEFAULT_CLUSTER_POINTS,
                  span_hz: int = DEFAULT_CLUSTER_SPAN_HZ) -> ClusterPlan:
    centres = tuple(int(round(c)) for c in centres_hz)
    if len(centres) < 1:
        raise ValueError("need at least one cluster")
    if len(set(centres)) != len(centres):
        raise ValueError("cluster centres must be distinct")
    if points < 2:
        raise ValueError("need at least 2 points per cluster")
    if span_hz <= 0:
        raise ValueError("span must be positive")
    plan = ClusterPlan(centres, int(points), int(span_hz))
    plan.offsets                                  # validates points vs span
    ordered = sorted(centres)
    if len(ordered) > 1 and min(b - a for a, b in zip(ordered, ordered[1:])) \
            <= plan.span_hz():
        raise ValueError("clusters overlap; widen the centres or narrow the "
                         "span, or one capture would hear two of them")
    return plan


def cluster_centres(low_hz: float, high_hz: float, clusters: int) -> list[int]:
    """``clusters`` centres spread evenly from ``low_hz`` to ``high_hz``.

    Even spacing is not the tightest possible slope estimate -- piling half the
    clusters at each end would be about 1.3x better -- but the intermediate
    ones are what turn "the fit has two points and therefore no residual" into
    a linearity check that can actually fail. That check is worth more than the
    1.3x, because the failure it catches is a bias and the 1.3x is only noise.
    """
    if clusters < 1:
        raise ValueError("need at least one cluster")
    if clusters == 1:
        return [int(round((low_hz + high_hz) / 2))]
    step = (high_hz - low_hz) / (clusters - 1)
    return [int(round(low_hz + step * i)) for i in range(clusters)]


def make_cluster_schedule(seed: int, plan: ClusterPlan, min_hop_s: float,
                          cycles: int, block: int = DEFAULT_BLOCK,
                          jitter: float = DEFAULT_JITTER,
                          period_cycles: int = DEFAULT_PERIOD_CYCLES,
                          band_extra_s: float = DEFAULT_BAND_EXTRA_S
                          ) -> list[Hop]:
    """Seeded schedule over several clusters, decodable one cluster at a time.

    One cycle emits every (cluster, point) pair exactly once, as
    ``points/block`` rounds of ``clusters`` blocks of ``block`` consecutive
    same-cluster dwells. ``block`` must divide ``points`` so every round is
    full and every cluster gets identical treatment.
    """
    if min_hop_s <= 0:
        raise ValueError("min_hop_s must be positive")
    if cycles < 1:
        raise ValueError("need at least one cycle")
    if not 0.0 <= jitter <= 1.0:
        raise ValueError("jitter must be between 0 and 1")
    if period_cycles < 1:
        raise ValueError("period_cycles must be at least 1")
    if block < 1:
        raise ValueError("block must be at least 1")
    if band_extra_s < 0:
        raise ValueError("band_extra_s must not be negative")
    if plan.points % block:
        raise ValueError(f"block {block} does not divide {plan.points} points; "
                         f"a partial round would give one cluster fewer dwells "
                         f"than the others and bias the fit toward it")
    rng = SplitMix64(seed)
    freqs = [plan.freqs(c) for c in range(plan.clusters)]
    rounds = plan.points // block

    period: list[tuple[int, int, float, bool, float]] = []
    for _ in range(period_cycles):
        orders = [_permutation(rng, plan.points) for _ in range(plan.clusters)]
        for r in range(rounds):
            for c in _permutation(rng, plan.clusters):
                for j in range(block):
                    dwell = min_hop_s * (1.0 + jitter * rng.uniform() * 2.0)
                    extra = band_extra_s if j == 0 else 0.0
                    period.append((c, orders[c][r * block + j],
                                   dwell + extra, j == 0, extra))

    per_cycle = plan.clusters * plan.points
    hops: list[Hop] = []
    t = 0.0
    for seq in range(cycles * per_cycle):
        cluster, point, dwell, change, extra = period[seq % len(period)]
        hops.append(Hop(sequence=seq, cycle=seq // per_cycle, point=point,
                        freq_hz=freqs[cluster][point], dwell_s=dwell,
                        start_s=t, end_s=t + dwell,
                        cluster=cluster, band_change=change, settle_s=extra))
        t += dwell
    return hops


def cluster_hops(hops: list[Hop], cluster: int) -> list[Hop]:
    """Just one cluster's dwells -- what a receiver tuned there can hear."""
    return [h for h in hops if h.cluster == cluster]


def describe_clusters(plan: ClusterPlan, hops: list[Hop], seed: int,
                      min_hop_s: float, block: int = DEFAULT_BLOCK,
                      lo_hz: float = 9.75e9,
                      period_cycles: int = DEFAULT_PERIOD_CYCLES,
                      band_extra_s: float = DEFAULT_BAND_EXTRA_S) -> str:
    per_cycle = plan.clusters * plan.points
    period = period_duration(hops, plan.points, period_cycles, per_cycle)
    lines = [
        "",
        f"Cluster hop schedule  (seed 0x{seed:X})",
        f"  clusters : {plan.clusters} at " + ", ".join(
            f"{c/1e9:.4f}" for c in plan.centres_hz) + " GHz",
        f"  lever arm: {plan.lever_hz()/1e6:.1f} MHz of IF "
        f"({(plan.centres_hz[0]-lo_hz)/1e9:.3f} to "
        f"{(plan.centres_hz[-1]-lo_hz)/1e9:.3f} GHz at LO {lo_hz/1e9:g} GHz)",
        f"  points   : {plan.points} per cluster on a Golomb ruler, span "
        f"{plan.span_hz()/1e3:.0f} kHz, closest pair "
        f"{plan.min_gap_hz()/1e3:.0f} kHz  <-- must fit one capture",
        f"  comb      : no shift can realign more than "
        f"{plan.coincidence():.2f} of it (1/{plan.points} is the floor; an "
        f"evenly spaced comb would be {(plan.points-1)/plan.points:.2f})",
        f"  blocks   : {block} dwells per cluster visit, "
        f"{plan.points//block} rounds per cycle, "
        f"{plan.clusters*plan.points//block} band changes per cycle "
        f"(each {band_extra_s*1e3:.1f} ms longer, and the extra is skipped)",
        f"  dwell    : {min_hop_s*1e3:.2f} ms, {per_cycle} hops per cycle",
        f"  period   : {period*1e3:.1f} ms -- the receiver's whole search range",
        f"  duty     : one cluster is on air 1/{plan.clusters} of the time, so "
        f"listen {plan.clusters}x as long per cluster",
        f"  first 10 : " + ", ".join(
            f"c{h.cluster}p{h.point}" + ("*" if h.band_change else "")
            for h in hops[:10]) + "   (* = band change, longer dwell)",
    ]
    return "\n".join(lines)

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


def run_hops(dev, hops: list[Hop], channel, settle_s: float = 0.05,
             autocal_plans: dict | None = None) -> None:
    """Transmit the schedule, hopping continuously with no muted gaps.

    Identity comes from the schedule rather than from an on/off pattern, so
    there is nothing to gate: the synthesiser simply retunes. The retune itself
    (about 0.84 ms) is the only discontinuity, and it lands inside the dwell it
    precedes rather than eating a coded window.

    ``autocal_plans`` maps frequency to a full :class:`Plan` and is what makes
    a multi-cluster schedule possible. Hops inside a cluster move by well under
    a megahertz, which stays inside one VCO band, so they need dividers only --
    that is the fast path and it is what the single-cluster schedule has always
    used. A hop that CHANGES cluster moves by hundreds of megahertz, which does
    not, and writing dividers alone would simply fail to lock. Those hops are
    marked ``band_change`` in the schedule and get a real retune with the VCO
    band search, and the schedule has already made their dwell longer to pay
    for it.
    """
    # Cold-start once with autocal so the VCO band is chosen for this span,
    # then hop without it. Re-running the band search on every hop would blank
    # the output through mute-till-lock for a large part of each dwell, which
    # at 5 ms leaves almost nothing radiated.
    # Solve every point once. Rebuilding a Plan per hop costs about 139 us and
    # would itself set the maximum hop rate well below what the part can do.
    order = sorted({h.freq_hz for h in hops})
    cached = dict(zip(order, dev.precompute(order, channel)))

    dev.set_frequency(hops[0].freq_hz, channel)
    if settle_s:
        time.sleep(settle_s)

    # A collection pause is the one thing that can miss a hop deadline. Median
    # jitter is under a microsecond, but a pause measured 22.8 ms -- 36 dwells
    # at 1600 hops/s -- and put 43 of 4800 hops late by over half a dwell. The
    # emitted pattern then stops matching the schedule the receiver assumes, and
    # because pauses are stochastic the failure moves between runs, which is
    # what made marginal hop rates look flaky rather than simply broken.
    # Disabled here, max jitter is 3.6 us and nothing is late.
    collecting = gc.isenabled()
    gc.disable()
    try:
        dev.set_output(channel, True)
        t0 = time.monotonic()
        for index, hop in enumerate(hops):
            _sleep_until(t0 + hop.end_s)
            nxt = hops[index + 1] if index + 1 < len(hops) else None
            if nxt is None:
                continue
            if nxt.band_change and autocal_plans is not None:
                dev.retune(autocal_plans[nxt.freq_hz], autocal=True)
            else:
                dev.apply_precomputed(cached[nxt.freq_hz])
        dev.set_output(channel, False)
    finally:
        if collecting:
            gc.enable()
