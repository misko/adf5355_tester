"""Differential test against the ADI Linux driver's own arithmetic.

Every other test in this suite checks the solver against this project's model
of the part. That is circular: it cannot catch a shared misreading of the
datasheet, and it passes just as happily when no hardware exists. This one
compares against an independent implementation -- adf5355_pll_fract_n_compute()
and adf5355_set_freq() from drivers/iio/frequency/adf5355.c in the Analog
Devices kernel fork -- transliterated below.

It is deliberately transliterated rather than imported: the point is a second
opinion, so it must not share any code with the thing under test. Keep it a
faithful copy of the driver, not a tidied one.
"""
from __future__ import annotations

from math import gcd

import pytest

from adf5355 import Channel, SynthConfig, plan

MOD1 = 1 << 24
MIN_VCO_HZ = 3_400_000_000
MAX_MODULUS2 = 16384


def driver_fract_n_compute(vco: int, pfd: int) -> tuple[int, int, int, int]:
    """adf5355_pll_fract_n_compute(), line for line."""
    integer, rem = divmod(vco, pfd)
    tmp = rem * MOD1
    fract1, fract2 = divmod(tmp, pfd)
    mod2 = pfd
    while mod2 > MAX_MODULUS2:
        mod2 >>= 1
        fract2 >>= 1
    div = gcd(fract2, mod2) or 1
    return integer, fract1, fract2 // div, mod2 // div


def driver_set_freq(freq: int, pfd: int, channel: int):
    """adf5355_set_freq(): channel 0 is RFoutA, channel 1 is RFoutB."""
    rf_div_sel = 0
    if channel == 1:
        freq >>= 1
    else:
        while freq < MIN_VCO_HZ:
            freq <<= 1
            rf_div_sel += 1
    return (*driver_fract_n_compute(freq, pfd), rf_div_sel)


def _config(channel: Channel) -> SynthConfig:
    return SynthConfig(ref_hz=125_000_000,
                       outa_enable=channel is Channel.A,
                       outb_enable=channel is Channel.B,
                       outa_power=3, mute_till_lock=False)


@pytest.mark.parametrize("channel,lo,hi", [
    (Channel.A, 60_000_000, 6_800_000_000),
    (Channel.B, 6_800_000_000, 13_600_000_000),
])
def test_matches_adi_driver(channel, lo, hi, subtests):
    cfg = _config(channel)
    pfd = int(cfg.fpfd_hz)
    chan = 0 if channel is Channel.A else 1
    span = hi - lo
    for i in range(400):
        freq = lo + (span * i) // 399
        try:
            ours = plan(cfg, freq, channel).solution
        except ValueError:
            continue                       # outside this channel's reach
        d_int, d_f1, _d_f2, _d_m2, d_div = driver_set_freq(freq, pfd, chan)
        with subtests.test(freq=freq):
            assert ours.integer == d_int, f"INT at {freq}"
            assert ours.frac1 == d_f1, f"FRAC1 at {freq}"
            if channel is Channel.A:
                assert ours.rf_divider_select == d_div, f"divider at {freq}"


def test_fpfd_is_read_not_assumed():
    """A wrong fPFD makes every vector wrong while looking entirely plausible.

    Assuming 125 MHz here instead of reading 62.5 MHz once produced a
    confident, wholly incorrect diagnosis of a 2x frequency bug.
    """
    cfg = _config(Channel.B)
    assert cfg.fpfd_hz == 62_500_000, (
        f"fPFD is {cfg.fpfd_hz/1e6:g} MHz; any hand-computed test vector "
        f"must be derived from this value, not from ref_hz")
