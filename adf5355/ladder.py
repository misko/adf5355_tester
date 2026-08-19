"""Duration-coded frequency ladder.

The pattern documented in ``raspberry_pi_adf5355_ku_ladder_guide.pdf``: N CW
carriers stepped evenly across a range, where rung n transmits for ``n * u``
seconds and then stays quiet for ``n * u``.  Because every rung has a distinct
burst length, the duration of a burst identifies which rung produced it, and
therefore its frequency -- a 1.4 s burst in the default pattern is rung 7, so
12.200 GHz.

``u`` is derived from the requested total, since the windows sum to
``2 * u * (1 + 2 + ... + N) = u * N * (N + 1)``.

The defaults reproduce the guide exactly: 10.700-12.700 GHz in nine 250 MHz
steps over 18.000 s.  tests/test_ladder_package_parity.py asserts this module
and the standalone adf5355_ladder.py agree step for step.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

DEFAULT_START_HZ = 10_700_000_000
DEFAULT_STOP_HZ = 12_700_000_000
DEFAULT_STEPS = 9
DEFAULT_TOTAL_S = 18.0

# Ku-band satellite downlink.
SATBAND_LO_HZ = 10_700_000_000
SATBAND_HI_HZ = 12_700_000_000


@dataclass(frozen=True)
class LadderStep:
    index: int          # 1-based rung number, which is also the duration code
    freq_hz: int
    on_s: float
    off_s: float
    start_s: float      # offsets from the start of the coded interval
    on_end_s: float
    end_s: float


def make_ladder(start_hz: int = DEFAULT_START_HZ,
                stop_hz: int = DEFAULT_STOP_HZ,
                steps: int = DEFAULT_STEPS,
                total_s: float = DEFAULT_TOTAL_S) -> list[LadderStep]:
    """Build the duration-coded schedule."""
    if steps < 2:
        raise ValueError("need at least 2 ladder rungs")
    if total_s <= 0:
        raise ValueError("total pattern time must be positive")

    unit_s = total_s / (steps * (steps + 1))
    result: list[LadderStep] = []
    t = 0.0
    for i in range(steps):
        n = i + 1
        freq = round(start_hz + (stop_hz - start_hz) * i / (steps - 1))
        width = n * unit_s
        on_end = t + width
        end = on_end + width
        result.append(LadderStep(index=n, freq_hz=freq, on_s=width,
                                 off_s=width, start_s=t, on_end_s=on_end,
                                 end_s=end))
        t = end
    return result


def format_ladder(steps: list[LadderStep]) -> str:
    lines = ["", "Duration-coded ladder",
             "step  RF GHz     ON s   OFF s   timeline s",
             "----  --------  -----  ------  ----------------"]
    for s in steps:
        lines.append(f"{s.index:>4}  {s.freq_hz/1e9:8.3f}  {s.on_s:5.3f}  "
                     f"{s.off_s:6.3f}  {s.start_s:5.1f}-{s.end_s:5.1f}")
    span = steps[-1].freq_hz - steps[0].freq_hz
    spacing = span / (len(steps) - 1) if len(steps) > 1 else 0
    lines.append(f"Range: {steps[0].freq_hz/1e9:.6f}-{steps[-1].freq_hz/1e9:.6f} "
                 f"GHz in {len(steps)} bins, spacing {spacing/1e6:.6f} MHz")
    lines.append(f"Unit time u = {steps[0].on_s:.6f} s; rung n is ON=n*u then "
                 f"OFF=n*u ({steps[0].on_s:.3f} s to {steps[-1].on_s:.3f} s "
                 f"per phase)")
    lines.append(f"Total coded interval: {steps[-1].end_s:.3f} s")
    return "\n".join(lines)


def overlaps_satellite_band(steps: list[LadderStep]) -> bool:
    return any(SATBAND_LO_HZ <= s.freq_hz <= SATBAND_HI_HZ for s in steps)


def _sleep_until(deadline: float) -> None:
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(remaining)


def run_ladder(dev, steps: list[LadderStep], channel, loops: int = 1,
               on_lock_failure=None) -> int:
    """Transmit the schedule.  Returns the number of rungs that failed to lock.

    Each rung's frequency is programmed during the previous rung's OFF window,
    so retuning never eats into a coded ON window and the burst lengths stay
    true to the schedule.
    """
    failures = 0
    dev.set_frequency(steps[0].freq_hz, channel)
    dev.set_output(channel, False)
    time.sleep(0.050)

    for loop_no in range(1, loops + 1):
        if loops > 1:
            print(f"ladder {loop_no}/{loops}")
        t0 = time.monotonic()
        for i, step in enumerate(steps):
            dev.set_output(channel, True)
            if on_lock_failure is not None and not on_lock_failure(dev, step):
                failures += 1
            print(f"  rung {step.index}: {step.freq_hz/1e9:.3f} GHz "
                  f"ON {step.on_s:.3f} s")
            _sleep_until(t0 + step.on_end_s)

            dev.set_output(channel, False)
            if i + 1 < len(steps):
                dev.set_frequency(steps[i + 1].freq_hz, channel)
            _sleep_until(t0 + step.end_s)

        elapsed = time.monotonic() - t0
        print(f"  coded interval complete in {elapsed:.3f} s")
        if loop_no < loops:
            dev.set_frequency(steps[0].freq_hz, channel)
            dev.set_output(channel, False)
            time.sleep(0.050)
    return failures
