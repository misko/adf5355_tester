#!/usr/bin/env python3
"""
ADF5355 RFOUTB duration-coded ladder for a shielded/attenuated bench test.

Default pattern:
  9 RF steps, 10.700 GHz through 12.700 GHz inclusive
  step n uses ON=n*0.200 s, then OFF=n*0.200 s
  total pattern time = 18.000 s

The frequency range, rung count and total time are selectable with
--start-ghz/--stop-ghz/--steps/--total-s. The defaults above are pinned and
unchanged: they are the pattern the accompanying guide PDF documents.

IMPORTANT:
  - This is intended only for a closed, shielded or otherwise controlled RF
    test path. 10.7-12.7 GHz overlaps satellite downlink allocations in many
    places. Do not connect an antenna and radiate the sweep into free space.
  - ADF5355 RFOUTB is used; RFOUTA+/- are disabled.
  - The board reference clock is REQUIRED because clone boards use different
    oscillators. Read the frequency marked on X1 or determine it from the
    board documentation. Do not guess.

The register calculations and write sequence are a Python port of the public
Analog Devices no-OS ADF5355 driver conventions.
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from dataclasses import dataclass
from math import gcd
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# ADF5355 register definitions (from Analog Devices no-OS adf5355.h)
# ---------------------------------------------------------------------------
MOD1 = 16_777_216
MAX_MOD2 = 16_384
MAX_PFD_HZ = 75_000_000
MIN_OUTB_HZ = 6_800_000_000
MAX_OUTB_HZ = 13_600_000_000
MIN_INT_PRESCALER_89 = 75

# Pinned defaults: the pattern documented in the guide PDF. Changing any of
# these changes what the guide describes, so they are named rather than typed
# twice (make_ladder signature and argparse both use them).
DEFAULT_START_HZ = 10_700_000_000
DEFAULT_STOP_HZ = 12_700_000_000
DEFAULT_STEPS = 9
DEFAULT_TOTAL_S = 18.0

# Satellite downlink allocations that the default range sits inside.
SATBAND_LO_HZ = 10_700_000_000
SATBAND_HI_HZ = 12_700_000_000

REG5_DEFAULT = 0x00800025
REG6_DEFAULT = 0x14000006
REG7_DEFAULT = 0x10000007
REG8_DEFAULT = 0x102D0428
REG10_DEFAULT = 0x00C0000A
REG11_DEFAULT = 0x0061300B
REG12_DEFAULT = 0x0000041C


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def div_round_up(n: int, d: int) -> int:
    return (n + d - 1) // d


def div_round_closest(n: int, d: int) -> int:
    return (n + d // 2) // d


def r0_int(x: int) -> int:
    return (x & 0xFFFF) << 4


def r0_prescaler(x: int) -> int:
    return (x & 1) << 20


def r0_autocal(x: int) -> int:
    return (x & 1) << 21


def r1_fract(x: int) -> int:
    return (x & 0xFFFFFF) << 4


def r2_mod2(x: int) -> int:
    return (x & 0x3FFF) << 4


def r2_frac2(x: int) -> int:
    return (x & 0x3FFF) << 18


def r4_counter_reset(x: int) -> int:
    return (x & 1) << 4


def r4_cp_threestate(x: int) -> int:
    return (x & 1) << 5


def r4_power_down(x: int) -> int:
    return (x & 1) << 6


def r4_pd_polarity_pos(x: int) -> int:
    return (x & 1) << 7


def r4_mux_logic_3v3(x: int) -> int:
    return (x & 1) << 8


def r4_refin_diff(x: int) -> int:
    return (x & 1) << 9


def r4_cp_current(x: int) -> int:
    return (x & 0xF) << 10


def r4_double_buffer(x: int) -> int:
    return (x & 1) << 14


def r4_r_counter(x: int) -> int:
    return (x & 0x3FF) << 15


def r4_rdiv2(x: int) -> int:
    return (x & 1) << 25


def r4_ref_doubler(x: int) -> int:
    return (x & 1) << 26


def r4_muxout(x: int) -> int:
    return (x & 0x7) << 27


def r6_outa_power(x: int) -> int:
    return (x & 0x3) << 4


def r6_outa_enable(x: int) -> int:
    return (x & 1) << 6


def r6_outb_disable_bit(disabled: bool) -> int:
    # ADF5355 DB10 is inverted: 0 = RFOUTB enabled, 1 = RFOUTB disabled.
    return (1 if disabled else 0) << 10


def r6_mute_till_lock(x: int) -> int:
    return (x & 1) << 11


def r6_cp_bleed(x: int) -> int:
    return (x & 0xFF) << 13


def r6_rf_div_sel(x: int) -> int:
    return (x & 0x7) << 21


def r6_feedback_fund(x: int) -> int:
    return (x & 1) << 24


def r6_neg_bleed(x: int) -> int:
    return (x & 1) << 29


def r6_gated_bleed(x: int) -> int:
    return (x & 1) << 30


def r7_ld_int_n(x: int) -> int:
    return (x & 1) << 4


def r7_frac_ld_precision(x: int) -> int:
    return (x & 0x3) << 5


def r7_lol_mode(x: int) -> int:
    return (x & 1) << 7


def r7_ld_cycle_count(x: int) -> int:
    return (x & 0x3) << 8


def r7_le_synced_refin(x: int) -> int:
    return (x & 1) << 25


def r9_synth_lock_timeout(x: int) -> int:
    return (x & 0x1F) << 4


def r9_alc_timeout(x: int) -> int:
    return (x & 0x1F) << 9


def r9_timeout(x: int) -> int:
    return (x & 0x3FF) << 14


def r9_vco_band_div(x: int) -> int:
    return (x & 0xFF) << 24


def r10_adc_enable(x: int) -> int:
    return (x & 1) << 4


def r10_adc_convert(x: int) -> int:
    return (x & 1) << 5


def r10_adc_clk_div(x: int) -> int:
    return (x & 0xFF) << 6


def r12_phase_resync_clk_div(x: int) -> int:
    return (x & 0xFFFF) << 16


@dataclass
class PllParams:
    integer: int
    fract1: int
    fract2: int
    mod2: int
    prescaler_89: bool
    cp_bleed: int


@dataclass
class LadderStep:
    index: int
    freq_hz: int
    on_s: float
    off_s: float
    start_s: float
    on_end_s: float
    end_s: float


class ADF5355:
    """Small RFOUTB-only ADF5355 controller for Raspberry Pi spidev."""

    def __init__(
        self,
        ref_hz: int,
        cp_ua: int = 900,
        bus: int = 0,
        device: int = 0,
        spi_hz: int = 1_000_000,
        dry_run: bool = False,
    ) -> None:
        if not (1_000_000 <= ref_hz <= 600_000_000):
            raise ValueError("Reference frequency must be between 1 MHz and 600 MHz")
        if not (315 <= cp_ua <= 5_040):
            raise ValueError("Charge-pump current must be between 315 and 5040 uA")

        self.ref_hz = int(ref_hz)
        self.cp_ua = int(cp_ua)
        self.bus = bus
        self.device = device
        self.spi_hz = spi_hz
        self.dry_run = dry_run
        self.spi = None
        self.regs: List[int] = [0] * 13
        self.initialized = False

        # Conservative/reference-driver defaults for this simple setup.
        self.ref_doubler = False
        self.ref_div2 = False
        self.ref_diff = False
        self.mux_3v3 = False
        self.mux_sel = 0  # three-state MUXOUT; not used by this script
        self.outa_enable = False
        self.outa_power = 0
        self.mute_till_lock = False
        self.cp_neg_bleed = False
        self.cp_gated_bleed = False

        self.r_counter, self.fpfd_hz = self._choose_r_counter()
        self.delay_us = self._build_static_registers()

        if not dry_run:
            try:
                import spidev  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "Python spidev is not installed. Run: sudo apt install python3-spidev"
                ) from exc
            self.spi = spidev.SpiDev()
            self.spi.open(bus, device)
            self.spi.max_speed_hz = spi_hz
            self.spi.mode = 0
            self.spi.lsbfirst = False
            self.spi.bits_per_word = 8

    def _choose_r_counter(self) -> Tuple[int, int]:
        r = 1
        while True:
            fpfd = (self.ref_hz * (2 if self.ref_doubler else 1)) // (
                r * (2 if self.ref_div2 else 1)
            )
            if fpfd <= MAX_PFD_HZ:
                return r, fpfd
            r += 1
            if r > 1023:
                raise ValueError("Unable to keep PFD at or below 75 MHz")

    def _build_static_registers(self) -> int:
        # Analog Devices no-OS uses round((CP_uA - 315) / 315), clamped 0..15.
        cp_code = div_round_closest(max(0, self.cp_ua - 315), 315)
        cp_code = clamp(cp_code, 0, 15)

        self.regs[3] = 0
        self.regs[4] = (
            r4_counter_reset(0)
            | r4_cp_threestate(0)
            | r4_power_down(0)
            | r4_pd_polarity_pos(1)
            | r4_mux_logic_3v3(1 if self.mux_3v3 else 0)
            | r4_refin_diff(1 if self.ref_diff else 0)
            | r4_cp_current(cp_code)
            | r4_double_buffer(1)
            | r4_r_counter(self.r_counter)
            | r4_rdiv2(1 if self.ref_div2 else 0)
            | r4_ref_doubler(1 if self.ref_doubler else 0)
            | r4_muxout(self.mux_sel)
        )
        self.regs[5] = REG5_DEFAULT
        self.regs[7] = (
            r7_ld_int_n(0)
            | r7_frac_ld_precision(3)
            | r7_lol_mode(0)
            | r7_ld_cycle_count(0)
            | r7_le_synced_refin(1)
            | REG7_DEFAULT
        )
        self.regs[8] = REG8_DEFAULT

        timeout = clamp(div_round_up(self.fpfd_hz, 20_000 * 30), 1, 1023)
        synth_timeout = clamp(
            div_round_up(self.fpfd_hz * 2, 100_000 * timeout), 1, 31
        )
        alc_timeout = clamp(
            div_round_up(self.fpfd_hz * 5, 100_000 * timeout), 1, 31
        )
        vco_band_div = clamp(div_round_up(self.fpfd_hz, 2_400_000), 1, 255)
        self.regs[9] = (
            r9_timeout(timeout)
            | r9_synth_lock_timeout(synth_timeout)
            | r9_alc_timeout(alc_timeout)
            | r9_vco_band_div(vco_band_div)
        )

        # ADI formula; clamp to the documented 8-bit field.
        adc_expr = (self.fpfd_hz / 100_000.0 - 2.0) / 4.0
        adc_div = clamp(int(math.ceil(adc_expr)), 1, 255)
        adc_clk_hz = self.fpfd_hz / (4 * adc_div + 2)
        delay_us = max(1, int(math.ceil(16_000_000.0 / adc_clk_hz)))
        self.regs[10] = (
            r10_adc_enable(1)
            | r10_adc_convert(1)
            | r10_adc_clk_div(adc_div)
            | REG10_DEFAULT
        )
        self.regs[11] = REG11_DEFAULT
        self.regs[12] = r12_phase_resync_clk_div(1) | REG12_DEFAULT
        return delay_us

    @staticmethod
    def _pll_params(vco_hz: int, pfd_hz: int, cp_ua: int) -> PllParams:
        integer, rem = divmod(vco_hz, pfd_hz)
        tmp = rem * MOD1
        fract1, fract2 = divmod(tmp, pfd_hz)
        mod2 = pfd_hz

        while mod2 > MAX_MOD2:
            mod2 >>= 1
            fract2 >>= 1

        g = gcd(fract2, mod2)
        if g:
            fract2 //= g
            mod2 //= g

        if mod2 == 0:
            mod2 = 1

        cp_bleed = div_round_up(400 * cp_ua, max(1, integer) * 375)
        cp_bleed = clamp(cp_bleed, 1, 255)
        return PllParams(
            integer=integer,
            fract1=fract1,
            fract2=fract2,
            mod2=mod2,
            prescaler_89=(integer >= MIN_INT_PRESCALER_89),
            cp_bleed=cp_bleed,
        )

    def _write_reg(self, addr: int, data: int) -> None:
        word = (data | (addr & 0xF)) & 0xFFFFFFFF
        if self.dry_run:
            return
        assert self.spi is not None
        buf = [
            (word >> 24) & 0xFF,
            (word >> 16) & 0xFF,
            (word >> 8) & 0xFF,
            word & 0xFF,
        ]
        # SPI CE0 is wired to ADF5355 LE. CE0 goes high at the end of xfer2,
        # providing the LE rising edge that latches the 32-bit word.
        self.spi.xfer2(buf)

    def _sleep_delay(self) -> None:
        time.sleep(self.delay_us / 1_000_000.0)

    def _build_frequency_registers(self, outb_hz: int, b_enabled: bool) -> PllParams:
        if not (MIN_OUTB_HZ <= outb_hz <= MAX_OUTB_HZ):
            raise ValueError("RFOUTB must be between 6.8 and 13.6 GHz")

        # RFOUTB is 2 x fundamental VCO on the ADF5355.
        vco_hz = outb_hz // 2
        p = self._pll_params(vco_hz, self.fpfd_hz, self.cp_ua)

        self.regs[0] = (
            r0_int(p.integer)
            | r0_prescaler(1 if p.prescaler_89 else 0)
            | r0_autocal(1)
        )
        self.regs[1] = r1_fract(p.fract1)
        self.regs[2] = r2_mod2(p.mod2) | r2_frac2(p.fract2)

        # A disabled; B enabled/disabled using inverted DB10 semantics.
        self.regs[6] = (
            r6_outa_power(self.outa_power)
            | r6_outa_enable(0)
            | r6_outb_disable_bit(not b_enabled)
            | r6_mute_till_lock(1 if self.mute_till_lock else 0)
            | r6_cp_bleed(p.cp_bleed)
            | r6_rf_div_sel(0)
            | r6_feedback_fund(1)
            | r6_neg_bleed(1 if self.cp_neg_bleed else 0)
            | r6_gated_bleed(1 if self.cp_gated_bleed else 0)
            | REG6_DEFAULT
        )
        return p

    def initialize_muted(self, outb_hz: int) -> PllParams:
        p = self._build_frequency_registers(outb_hz, b_enabled=False)
        # Full initialization: R12 down through R1, wait >16 ADC clocks, then R0.
        for addr in range(12, 0, -1):
            self._write_reg(addr, self.regs[addr])
        self._sleep_delay()
        self._write_reg(0, self.regs[0])
        self.initialized = True
        return p

    def set_frequency_muted(self, outb_hz: int) -> PllParams:
        if not self.initialized:
            return self.initialize_muted(outb_hz)

        p = self._build_frequency_registers(outb_hz, b_enabled=False)

        # Frequency-update sequence follows the ADI no-OS driver:
        # R10, R6, R4(reset), R2, R1, R0(no autocal), R4, delay, R0(autocal).
        self._write_reg(10, self.regs[10])
        self._write_reg(6, self.regs[6])
        self._write_reg(4, self.regs[4] | r4_counter_reset(1))
        self._write_reg(2, self.regs[2])
        self._write_reg(1, self.regs[1])
        self._write_reg(0, self.regs[0] & ~r0_autocal(1))
        self._write_reg(4, self.regs[4])
        self._sleep_delay()
        self._write_reg(0, self.regs[0])
        return p

    def set_output_b(self, enabled: bool) -> None:
        if not self.initialized:
            raise RuntimeError("Synthesizer has not been initialized")
        if enabled:
            self.regs[6] &= ~(1 << 10)
        else:
            self.regs[6] |= (1 << 10)
        self._write_reg(6, self.regs[6])

    def mute(self) -> None:
        if self.initialized:
            try:
                self.set_output_b(False)
            except Exception:
                pass

    def close(self) -> None:
        self.mute()
        if self.spi is not None:
            self.spi.close()
            self.spi = None

    def register_dump(self) -> List[str]:
        return [f"R{i:02d} = 0x{(self.regs[i] | i) & 0xFFFFFFFF:08X}" for i in range(12, -1, -1)]


def make_ladder(
    start_hz: int = DEFAULT_START_HZ,
    stop_hz: int = DEFAULT_STOP_HZ,
    steps: int = DEFAULT_STEPS,
    total_s: float = DEFAULT_TOTAL_S,
) -> List[LadderStep]:
    if steps < 2:
        raise ValueError("Need at least 2 ladder steps")
    if total_s <= 0:
        raise ValueError("Total pattern time must be positive")

    # Each step n has ON=n*u and OFF=n*u. Therefore:
    # total = 2*u*sum(1..N) = u*N*(N+1), so u=total/[N(N+1)].
    unit_s = total_s / (steps * (steps + 1))
    result: List[LadderStep] = []
    t = 0.0
    for i in range(steps):
        n = i + 1
        freq = round(start_hz + (stop_hz - start_hz) * i / (steps - 1))
        width = n * unit_s
        start = t
        on_end = start + width
        end = on_end + width
        result.append(
            LadderStep(
                index=n,
                freq_hz=freq,
                on_s=width,
                off_s=width,
                start_s=start,
                on_end_s=on_end,
                end_s=end,
            )
        )
        t = end
    return result


def ghz_to_hz(ghz: float) -> int:
    """GHz from the command line to integer Hz, rounded like --ref-mhz is."""
    if not math.isfinite(ghz):
        raise ValueError(f"{ghz} GHz is not a finite frequency")
    return round(ghz * 1_000_000_000)


def validate_ladder_range(start_hz: int, stop_hz: int, steps: int) -> None:
    """Check a requested range before any hardware is touched.

    _build_frequency_registers enforces the RFOUTB limits one frequency at a
    time, which is too late to be useful: the early rungs would already have
    been keyed on before an out-of-band rung was reached.
    """
    for option, hz in (("--start-ghz", start_hz), ("--stop-ghz", stop_hz)):
        if not (MIN_OUTB_HZ <= hz <= MAX_OUTB_HZ):
            raise ValueError(
                f"{option} {hz/1e9:.6f} GHz is outside the ADF5355 RFOUTB "
                f"range {MIN_OUTB_HZ/1e9:.1f}-{MAX_OUTB_HZ/1e9:.1f} GHz"
            )
    if stop_hz < start_hz:
        raise ValueError(
            f"--stop-ghz {stop_hz/1e9:.6f} GHz is below --start-ghz "
            f"{start_hz/1e9:.6f} GHz; the ladder only climbs"
        )
    if steps < 2:
        # A zero-span range is legal - the rungs are then told apart by their
        # duration coding alone - but that still needs at least two of them.
        if start_hz == stop_hz:
            raise ValueError(
                "--start-ghz equals --stop-ghz, so the rungs differ only in "
                f"duration; that needs --steps 2 or more, not {steps}"
            )
        raise ValueError(
            f"--steps {steps} is too few; a ladder needs at least 2 rungs, "
            "one at --start-ghz and one at --stop-ghz"
        )


def safety_notice(start_hz: int, stop_hz: int) -> List[str]:
    """Bench-only warning lines. Only the satellite text depends on the range."""
    lines = [
        "SAFETY: closed/shielded/attenuated bench test only; do not radiate this sweep.",
        "RFOUTA+/- are disabled; RFOUTB/OB is the only RF output used.",
    ]
    band = f"{SATBAND_LO_HZ/1e9:.1f}-{SATBAND_HI_HZ/1e9:.1f} GHz"
    swept = f"{start_hz/1e9:.3f}-{stop_hz/1e9:.3f} GHz"
    if start_hz <= SATBAND_HI_HZ and stop_hz >= SATBAND_LO_HZ:
        lines.append(
            f"SAFETY: {swept} overlaps the {band} satellite downlink band. "
            "Do not connect an antenna."
        )
    else:
        lines.append(
            f"SAFETY: {swept} is outside the {band} satellite downlink band, "
            "but this is bench-only equipment either way: keep the OB path "
            "shielded and attenuated, and do not connect an antenna."
        )
    return lines


def print_ladder(steps: List[LadderStep]) -> None:
    print("\nDuration-coded RFOUTB ladder")
    print("step  RF GHz     ON s   OFF s   timeline s")
    print("----  --------  -----  ------  ----------------")
    for s in steps:
        print(
            f"{s.index:>4}  {s.freq_hz/1e9:8.3f}  {s.on_s:5.3f}  {s.off_s:6.3f}  "
            f"{s.start_s:5.1f}-{s.end_s:5.1f}"
        )
    # Everything below is derived from the requested range and timing budget.
    span_hz = steps[-1].freq_hz - steps[0].freq_hz
    spacing_hz = span_hz / (len(steps) - 1) if len(steps) > 1 else 0.0
    # Rung n is ON=n*u and OFF=n*u, so rung 1's ON time is the unit time itself.
    unit_s = steps[0].on_s
    print(
        f"Range: {steps[0].freq_hz/1e9:.6f}-{steps[-1].freq_hz/1e9:.6f} GHz "
        f"in {len(steps)} bins, spacing {spacing_hz/1e6:.6f} MHz"
    )
    print(
        f"Unit time u = {unit_s:.6f} s; rung n is ON=n*u then OFF=n*u "
        f"({steps[0].on_s:.3f} s to {steps[-1].on_s:.3f} s per phase)"
    )
    print(f"Total coded interval: {steps[-1].end_s:.3f} s\n")


def sleep_until(deadline: float) -> None:
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        # One sleep is normally enough; looping handles signals/scheduler wakeups.
        time.sleep(remaining)


def run_ladder(synth: ADF5355, steps: List[LadderStep], loops: int) -> None:
    # Program the first frequency and settle before the 18 s coded interval starts.
    p = synth.initialize_muted(steps[0].freq_hz)
    print(
        f"PFD={synth.fpfd_hz/1e6:.6f} MHz, R={synth.r_counter}, "
        f"first N={p.integer}+fraction, ADC delay={synth.delay_us} us"
    )
    time.sleep(0.050)

    for loop_no in range(1, loops + 1):
        print(f"Starting ladder {loop_no}/{loops}")
        t0 = time.monotonic()

        for i, step in enumerate(steps):
            # The current frequency was programmed during the previous OFF slot.
            synth.set_output_b(True)
            print(
                f"  {step.index}: {step.freq_hz/1e9:.3f} GHz ON "
                f"{step.on_s:.3f} s"
            )
            sleep_until(t0 + step.on_end_s)

            synth.set_output_b(False)

            # Reprogram the next frequency while muted, inside this step's OFF time.
            if i + 1 < len(steps):
                synth.set_frequency_muted(steps[i + 1].freq_hz)

            sleep_until(t0 + step.end_s)

        synth.mute()
        elapsed = time.monotonic() - t0
        print(f"Completed coded interval in {elapsed:.3f} s; RFOUTB muted")

        # If more than one loop was explicitly requested, pre-program the first
        # frequency while muted before the next interval.
        if loop_no < loops:
            synth.set_frequency_muted(steps[0].freq_hz)
            time.sleep(0.050)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "ADF5355 duration-coded frequency ladder (RFOUTB, bench test only); "
            f"defaults to {DEFAULT_START_HZ/1e9:.1f}-{DEFAULT_STOP_HZ/1e9:.1f} GHz"
        )
    )
    p.add_argument(
        "--ref-mhz",
        type=float,
        required=True,
        help="Actual reference oscillator frequency on your board in MHz (required)",
    )
    p.add_argument("--cp-ua", type=int, default=900, help="Charge-pump current in uA (default 900)")
    p.add_argument("--bus", type=int, default=0, help="SPI bus (default 0)")
    p.add_argument("--device", type=int, default=0, help="SPI device/CE (default 0 = CE0)")
    p.add_argument("--spi-hz", type=int, default=1_000_000, help="SPI clock rate (default 1 MHz)")
    p.add_argument(
        "--start-ghz",
        type=float,
        default=DEFAULT_START_HZ / 1e9,
        help=f"First ladder frequency in GHz (default {DEFAULT_START_HZ/1e9:.1f})",
    )
    p.add_argument(
        "--stop-ghz",
        type=float,
        default=DEFAULT_STOP_HZ / 1e9,
        help=f"Last ladder frequency in GHz (default {DEFAULT_STOP_HZ/1e9:.1f})",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Number of ladder rungs (default {DEFAULT_STEPS})",
    )
    p.add_argument(
        "--total-s",
        type=float,
        default=DEFAULT_TOTAL_S,
        help=f"Total coded interval (default {DEFAULT_TOTAL_S} s)",
    )
    p.add_argument("--loops", type=int, default=1, help="Number of complete ladders (default 1; max 100)")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Calculate/print schedule and first register set; do not open SPI or enable RF",
    )
    p.add_argument(
        "--enable-rf",
        action="store_true",
        help="Required to actually enable RFOUTB. Use only in a shielded/attenuated bench setup.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if not (1 <= args.loops <= 100):
        print("ERROR: --loops must be 1..100", file=sys.stderr)
        return 2

    ref_hz = round(args.ref_mhz * 1_000_000)
    try:
        start_hz = ghz_to_hz(args.start_ghz)
        stop_hz = ghz_to_hz(args.stop_ghz)
        validate_ladder_range(start_hz, stop_hz, args.steps)
        steps = make_ladder(
            start_hz=start_hz,
            stop_hz=stop_hz,
            steps=args.steps,
            total_s=args.total_s,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print_ladder(steps)

    for line in safety_notice(start_hz, stop_hz):
        print(line)

    synth = ADF5355(
        ref_hz=ref_hz,
        cp_ua=args.cp_ua,
        bus=args.bus,
        device=args.device,
        spi_hz=args.spi_hz,
        dry_run=args.dry_run,
    )

    # Build the first frequency so dry-run can show a useful register set.
    p0 = synth._build_frequency_registers(steps[0].freq_hz, b_enabled=False)
    print(
        f"Reference={ref_hz/1e6:.6f} MHz -> PFD={synth.fpfd_hz/1e6:.6f} MHz "
        f"(R={synth.r_counter}); first VCO={steps[0].freq_hz/2/1e9:.6f} GHz"
    )
    print(
        f"First PLL: INT={p0.integer}, FRAC1={p0.fract1}, "
        f"FRAC2={p0.fract2}, MOD2={p0.mod2}"
    )

    if args.dry_run:
        print("\nDry run: RF is never enabled. First muted register image:")
        for line in synth.register_dump():
            print("  " + line)
        return 0

    if not args.enable_rf:
        print(
            "\nNo RF generated. Re-run with --enable-rf only after the OB path is "
            "shielded/attenuated and the reference clock value has been verified."
        )
        synth.close()
        return 0

    # Mute on Ctrl-C / termination.
    stopping = False

    def handle_signal(signum, frame):  # type: ignore[no-untyped-def]
        nonlocal stopping
        if not stopping:
            stopping = True
            print("\nSignal received: muting RFOUTB...", file=sys.stderr)
            synth.mute()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        run_ladder(synth, steps, args.loops)
    except KeyboardInterrupt:
        print("Stopped; RFOUTB muted.")
        return 130
    finally:
        synth.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
