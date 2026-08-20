"""Frequency -> register plan for the ADF5355.  Pure computation, no hardware.

The arithmetic follows Analog Devices' no-OS driver so that register images can
be compared bit-for-bit against the reference implementation (see
tests/test_vs_adi_c.py).  Three deliberate divergences from that driver are
marked DIVERGENCE below; each is a defect in the C reference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from math import gcd

from .registers import MuxOut, OutputPower, RegisterFile

# --- Device limits (adf5355.h) ---------------------------------------------
MODULUS1 = 16_777_216            # 2**24, the fixed primary modulus
ADI_MAX_MODULUS2 = 16_384        # loop bound used by the ADI driver
MOD2_FIELD_MAX = 16_383          # what actually fits the 14-bit MOD2 field
MAX_PFD_HZ = 75_000_000
MAX_REFIN_HZ = 600_000_000
MIN_REFIN_HZ = 10_000_000
MIN_VCO_HZ = 3_400_000_000
MAX_VCO_HZ = 6_800_000_000
MIN_OUTA_HZ = MIN_VCO_HZ // 64   # 53.125 MHz
MAX_OUTA_HZ = MAX_VCO_HZ
MIN_OUTB_HZ = MIN_VCO_HZ * 2     # 6.8 GHz
MAX_OUTB_HZ = MAX_VCO_HZ * 2     # 13.6 GHz
MIN_INT_PRESCALER_89 = 75
MAX_R_COUNTER = 1023
CP_STEP_UA = 315


class Channel(Enum):
    """RFoutA is VCO/2**d; RFoutB is the frequency doubler, 2*VCO."""
    A = "A"
    B = "B"


def clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def div_round_up(n: int, d: int) -> int:
    return -(-n // d)


def div_round_closest(n: int, d: int) -> int:
    return (n + d // 2) // d


# --- Fractional-N solving ---------------------------------------------------

def solve_adi(vco_hz: int, fpfd_hz: int) -> tuple[int, int, int, int]:
    """The reference algorithm: MOD2 starts at f_PFD and is halved into range.

    Returns (int, frac1, frac2, mod2).
    """
    integer, remainder = divmod(vco_hz, fpfd_hz)
    scaled = remainder * MODULUS1
    frac1, frac2 = divmod(scaled, fpfd_hz)

    mod2 = fpfd_hz
    while mod2 > ADI_MAX_MODULUS2:
        mod2 >>= 1
        frac2 >>= 1

    divisor = gcd(frac2, mod2)
    if divisor:
        mod2 //= divisor
        frac2 //= divisor

    # DIVERGENCE 1: the ADI loop bound permits MOD2 == 16384, which does not
    # fit the 14-bit field and encodes as MOD2 = 0 -- a divide-by-zero in the
    # sigma-delta modulator.  Reachable whenever f_PFD is 16384 * 2**k and the
    # reduced FRAC2 is odd (e.g. f_PFD = 67.108864 MHz).  Halve once more.
    while mod2 > MOD2_FIELD_MAX:
        mod2 >>= 1
        frac2 >>= 1

    return integer, frac1, frac2, max(1, mod2)


def solve_exact(vco_hz: int, fpfd_hz: int) -> tuple[int, int, int, int]:
    """Minimum-error alternative: best FRAC2/MOD2 rational under the field cap.

    The ADI algorithm truncates FRAC2 on every halving step, so it is not
    optimal.  This picks the closest representable fraction instead.
    """
    ratio = Fraction(vco_hz, fpfd_hz)
    integer = ratio.numerator // ratio.denominator
    scaled = (ratio - integer) * MODULUS1
    frac1 = scaled.numerator // scaled.denominator
    residual = scaled - frac1

    best = residual.limit_denominator(MOD2_FIELD_MAX)
    frac2, mod2 = best.numerator, best.denominator

    if frac2 >= mod2:                 # residual rounded up to a whole LSB
        frac2, mod2 = 0, 1
        frac1 += 1
        if frac1 >= MODULUS1:
            frac1 = 0
            integer += 1

    return integer, frac1, frac2, mod2


SOLVERS = {"adi": solve_adi, "exact": solve_exact}


def vco_from_fields(integer: int, frac1: int, frac2: int, mod2: int,
                    fpfd_hz: int) -> Fraction:
    """Exact VCO frequency implied by the divider fields."""
    n = (Fraction(integer)
         + Fraction(frac1, MODULUS1)
         + Fraction(frac2, mod2 * MODULUS1))
    return n * fpfd_hz


# --- Configuration ----------------------------------------------------------

@dataclass(frozen=True)
class SynthConfig:
    """Everything that does not change when you retune."""
    ref_hz: int
    ref_doubler: bool = False
    ref_div2: bool = False
    ref_diff: bool = False
    r_counter: int | None = None          # None -> maximize f_PFD
    cp_ua: int = 900
    pd_polarity_positive: bool = True
    muxout: MuxOut = MuxOut.DIGITAL_LOCK_DETECT
    muxout_3v3: bool = True               # Pi GPIO wants 3.3 V, not 1.8 V
    outa_enable: bool = False
    outa_power: OutputPower = OutputPower.MINUS_4_DBM
    outb_enable: bool = False
    mute_till_lock: bool = True
    negative_bleed: bool = True
    gated_bleed: bool = False
    solver: str = "adi"

    def __post_init__(self):
        if not MIN_REFIN_HZ <= self.ref_hz <= MAX_REFIN_HZ:
            raise ValueError(
                f"REFIN {self.ref_hz/1e6:.6f} MHz outside "
                f"{MIN_REFIN_HZ/1e6:g}-{MAX_REFIN_HZ/1e6:g} MHz"
            )
        if not CP_STEP_UA <= self.cp_ua <= CP_STEP_UA * 16:
            raise ValueError(f"charge pump {self.cp_ua} uA outside 315-5040 uA")
        if self.solver not in SOLVERS:
            raise ValueError(f"unknown solver {self.solver!r}")

    @property
    def r_divider(self) -> int:
        """R counter actually used, auto-selected to maximize f_PFD if unset."""
        if self.r_counter is not None:
            if not 1 <= self.r_counter <= MAX_R_COUNTER:
                raise ValueError(f"R counter {self.r_counter} outside 1-1023")
            return self.r_counter
        r = 1
        while self._pfd_for(r) > MAX_PFD_HZ:
            r += 1
            if r > MAX_R_COUNTER:
                raise ValueError("cannot bring f_PFD to 75 MHz or below")
        return r

    def _pfd_for(self, r: int) -> int:
        numerator = self.ref_hz * (2 if self.ref_doubler else 1)
        return numerator // (r * (2 if self.ref_div2 else 1))

    @property
    def fpfd_hz(self) -> int:
        return self._pfd_for(self.r_divider)

    @property
    def fpfd_exact(self) -> Fraction:
        """f_PFD without the integer truncation the hardware maths assumes."""
        return Fraction(self.ref_hz * (2 if self.ref_doubler else 1),
                        self.r_divider * (2 if self.ref_div2 else 1))


# --- Solution and plan ------------------------------------------------------

@dataclass(frozen=True)
class Solution:
    channel: Channel
    requested_hz: int
    integer: int
    frac1: int
    frac2: int
    mod2: int
    rf_divider_select: int
    prescaler_89: bool
    cp_bleed: int
    fpfd_hz: int

    @property
    def vco_hz(self) -> Fraction:
        return vco_from_fields(self.integer, self.frac1, self.frac2,
                               self.mod2, self.fpfd_hz)

    @property
    def achieved_hz(self) -> Fraction:
        if self.channel is Channel.B:
            return self.vco_hz * 2
        return self.vco_hz / (1 << self.rf_divider_select)

    @property
    def error_hz(self) -> Fraction:
        return self.achieved_hz - self.requested_hz

    @property
    def error_ppb(self) -> float:
        return float(self.error_hz / self.requested_hz) * 1e9

    @property
    def is_integer_n(self) -> bool:
        return self.frac1 == 0 and self.frac2 == 0


@dataclass(frozen=True)
class Plan:
    config: SynthConfig
    solution: Solution
    registers: RegisterFile
    delay_us: int

    @property
    def words(self) -> list[int]:
        return self.registers.words

    def summary(self) -> str:
        s = self.solution
        return (
            f"REFIN {self.config.ref_hz/1e6:.6f} MHz  R={self.config.r_divider}"
            f"  f_PFD={s.fpfd_hz/1e6:.6f} MHz\n"
            f"RFout{s.channel.value} requested {s.requested_hz/1e9:.9f} GHz\n"
            f"  VCO {float(s.vco_hz)/1e9:.9f} GHz"
            f"  RFdiv=/{1 << s.rf_divider_select}\n"
            f"  INT={s.integer} FRAC1={s.frac1} FRAC2={s.frac2} MOD2={s.mod2}"
            f"  prescaler={'8/9' if s.prescaler_89 else '4/5'}\n"
            f"  achieved {float(s.achieved_hz)/1e9:.9f} GHz"
            f"  error {float(s.error_hz):+.6f} Hz ({s.error_ppb:+.3f} ppb)\n"
            f"  autocal settle delay {self.delay_us} us"
        )


def _select_channel_a(freq_hz: int) -> tuple[int, int]:
    """Return (vco_hz, rf_divider_select) for RFoutA."""
    if not MIN_OUTA_HZ <= freq_hz <= MAX_OUTA_HZ:
        raise ValueError(
            f"RFoutA {freq_hz/1e6:.3f} MHz outside "
            f"{MIN_OUTA_HZ/1e6:.3f}-{MAX_OUTA_HZ/1e9:.1f} GHz"
        )
    vco, div_sel = freq_hz, 0
    while vco < MIN_VCO_HZ:
        vco <<= 1
        div_sel += 1
    return vco, div_sel


def plan(config: SynthConfig, freq_hz: int,
         channel: Channel = Channel.A) -> Plan:
    """Build the complete register image for one output frequency."""
    freq_hz = int(freq_hz)
    fpfd = config.fpfd_hz

    if channel is Channel.A:
        vco_hz, rf_div_sel = _select_channel_a(freq_hz)
    else:
        if not MIN_OUTB_HZ <= freq_hz <= MAX_OUTB_HZ:
            raise ValueError(
                f"RFoutB {freq_hz/1e9:.3f} GHz outside "
                f"{MIN_OUTB_HZ/1e9:.1f}-{MAX_OUTB_HZ/1e9:.1f} GHz"
            )
        vco_hz, rf_div_sel = freq_hz // 2, 0

    integer, frac1, frac2, mod2 = SOLVERS[config.solver](vco_hz, fpfd)

    if integer >= (1 << 16):
        raise ValueError(f"INT={integer} overflows the 16-bit field")

    prescaler_89 = integer >= MIN_INT_PRESCALER_89
    cp_bleed = clamp(div_round_up(400 * config.cp_ua, integer * 375), 1, 255)

    # ADI gates negative bleed off for integer-N and for very fast PFDs.
    neg_bleed = config.negative_bleed
    if fpfd > 100_000_000 or (frac1 == 0 and frac2 == 0):
        neg_bleed = False

    solution = Solution(
        channel=channel, requested_hz=freq_hz, integer=integer, frac1=frac1,
        frac2=frac2, mod2=mod2, rf_divider_select=rf_div_sel,
        prescaler_89=prescaler_89, cp_bleed=cp_bleed, fpfd_hz=fpfd,
    )

    regs, delay_us = _assemble(config, solution, neg_bleed)
    return Plan(config=config, solution=solution, registers=regs,
                delay_us=delay_us)


def _assemble(config: SynthConfig, s: Solution,
              neg_bleed: bool) -> tuple[RegisterFile, int]:
    fpfd = s.fpfd_hz
    regs = RegisterFile()

    # R0 -- divider and calibration.  autocal is set here; the caller clears it
    # for the intermediate write in the retune sequence.
    regs.update(int=s.integer, prescaler=int(s.prescaler_89), autocal=1)

    # R1, R2 -- fractional part.
    regs.update(frac1=s.frac1, mod2=s.mod2, frac2=s.frac2)

    # R4 -- reference path, charge pump, MUXOUT.
    _assemble_r4(config, regs)

    # R6 -- output stage and RF divider.  DB10 is active-low on the ADF5355:
    # 0 enables RFoutB, 1 disables it.
    regs.update(
        output_power=int(config.outa_power),
        rf_out_enable=int(config.outa_enable),
        rf_outb_disable=int(not config.outb_enable),
        mute_till_lock=int(config.mute_till_lock),
        cp_bleed_current=s.cp_bleed,
        rf_divider_select=s.rf_divider_select,
        feedback_fundamental=1,
        negative_bleed=int(neg_bleed),
        gated_bleed=int(config.gated_bleed),
    )

    # R7 -- lock detect.
    # DIVERGENCE 2: the ADI C driver builds R7 as `a | b | ... | cond ? X : Y`.
    # `?:` binds looser than `|`, so the whole left side becomes the condition
    # and the driver always emits the ADF5356 default (0x04000007) -- even for
    # an ADF5355.  Built correctly here: 0x12000067.
    regs.update(ld_mode_int_n=0, frac_n_ld_precision=3, lol_mode=0,
                ld_cycle_count=0, le_synced_refin=1)

    # R9 -- calibration timeouts, all derived from f_PFD.
    timeout = clamp(div_round_up(fpfd, 20_000 * 30), 1, 1023)
    regs.update(
        timeout=timeout,
        synth_lock_timeout=clamp(div_round_up(fpfd * 2, 100_000 * timeout), 1, 31),
        alc_timeout=clamp(div_round_up(fpfd * 5, 100_000 * timeout), 1, 31),
        vco_band_division=clamp(div_round_up(fpfd, 2_400_000), 1, 255),
    )

    # R10 -- the ADC that drives VCO band select.  The datasheet requires its
    # clock to stay at or below 100 kHz:
    #
    #     f_PFD / (4 * adc_div + 2) <= 100 kHz
    #  => adc_div >= (f_PFD - 200000) / 400000
    #
    # DIVERGENCE 3: the C reference computes this as
    # DIV_ROUND_UP(fpfd / 100000U - 2, 4), where the inner integer division
    # truncates before the ceiling and so can round the divider *down* by one.
    # At f_PFD = 61.44 MHz (a 122.88 MHz reference, common on clone boards)
    # that yields adc_div = 153 and a 100,065 Hz ADC clock -- over the limit.
    # Solving the inequality directly in integers avoids the lost remainder.
    adc_div = clamp(div_round_up(max(1, fpfd - 200_000), 400_000), 1, 255)
    adc_clk_hz = fpfd // (4 * adc_div + 2)
    if adc_clk_hz > 100_000:
        raise ValueError(
            f"ADC clock {adc_clk_hz} Hz exceeds the 100 kHz limit "
            f"(f_PFD={fpfd} Hz needs adc_div>255)"
        )
    regs.update(adc_enable=1, adc_conversion_enable=1, adc_clk_divider=adc_div)

    # R12 -- phase resync clock divider.
    regs.update(phase_resync_clk_divider=1)

    # Autocal needs more than 16 ADC clock cycles of settling.
    delay_us = div_round_up(16_000_000, adc_clk_hz)
    return regs, delay_us


def _assemble_r4(config: SynthConfig, regs: RegisterFile,
                 muxout: MuxOut | None = None) -> RegisterFile:
    """R4: reference path, charge pump and MUXOUT.  Depends only on config."""
    cp_code = clamp(div_round_closest(config.cp_ua - CP_STEP_UA, CP_STEP_UA),
                    0, 15)
    regs.update(
        counter_reset=0, cp_threestate=0, power_down=0,
        pd_polarity_positive=int(config.pd_polarity_positive),
        mux_logic_3v3=int(config.muxout_3v3),
        refin_mode_diff=int(config.ref_diff),
        charge_pump_current=cp_code,
        double_buffer=1,
        r_counter=config.r_divider,
        rdiv2=int(config.ref_div2),
        ref_doubler=int(config.ref_doubler),
        muxout=int(config.muxout if muxout is None else muxout),
    )
    return regs


def reference_word(config: SynthConfig, muxout: MuxOut | None = None) -> int:
    """The R4 word on its own.

    The ADF5355 cannot be read back, so MUXOUT is the only return path from the
    chip.  R4 selects what MUXOUT reports and is meaningful before the PLL has
    been configured at all, which makes it usable as a link test.
    """
    return _assemble_r4(config, RegisterFile(), muxout).word(4)
