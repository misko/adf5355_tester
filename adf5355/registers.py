"""ADF5355 register map -- the single source of truth for bit positions.

Field definitions mirror Analog Devices' own no-OS driver
(drivers/frequency/adf5355/adf5355.h), which is the authoritative encoding of
the datasheet register map.  Nothing above this module touches raw hex.

The ADF5355 has 13 registers, R0..R12.  Each is a 32-bit word whose low four
bits carry the register address; the word is shifted MSB-first and latched on
the rising edge of LE.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

N_REGS = 13
SPI_WORD_BYTES = 4


class MuxOut(IntEnum):
    """R4[29:27] -- what the MUXOUT pin reports."""
    THREESTATE = 0
    DVDD = 1
    GND = 2
    R_DIV_OUT = 3
    N_DIV_OUT = 4
    ANALOG_LOCK_DETECT = 5
    DIGITAL_LOCK_DETECT = 6


class OutputPower(IntEnum):
    """R6[5:4] -- RFoutA output power."""
    MINUS_4_DBM = 0
    MINUS_1_DBM = 1
    PLUS_2_DBM = 2
    PLUS_5_DBM = 3


@dataclass(frozen=True)
class Field:
    """One contiguous bitfield inside one register."""
    name: str
    reg: int
    lsb: int
    width: int

    @property
    def mask(self) -> int:
        return ((1 << self.width) - 1) << self.lsb

    def encode(self, value: int) -> int:
        value = int(value)
        if not 0 <= value < (1 << self.width):
            raise ValueError(
                f"{self.name}={value} does not fit in {self.width} bits "
                f"(max {(1 << self.width) - 1})"
            )
        return value << self.lsb

    def decode(self, word: int) -> int:
        return (word & self.mask) >> self.lsb


def _f(name, reg, lsb, width):
    return Field(name, reg, lsb, width)


# --- Field table ------------------------------------------------------------
# name                              reg  lsb  width
FIELDS = {f.name: f for f in (
    # R0
    _f("int",                         0,   4, 16),
    _f("prescaler",                   0,  20,  1),   # 0 = 4/5, 1 = 8/9
    _f("autocal",                     0,  21,  1),
    # R1
    _f("frac1",                       1,   4, 24),
    # R2
    _f("mod2",                        2,   4, 14),
    _f("frac2",                       2,  18, 14),
    # R3
    _f("phase",                       3,   4, 24),
    _f("phase_adjust",                3,  28,  1),
    _f("phase_resync",                3,  29,  1),
    _f("exact_sdload_reset",          3,  30,  1),
    # R4
    _f("counter_reset",               4,   4,  1),
    _f("cp_threestate",               4,   5,  1),
    _f("power_down",                  4,   6,  1),
    _f("pd_polarity_positive",        4,   7,  1),
    _f("mux_logic_3v3",               4,   8,  1),   # 0 = 1.8 V, 1 = 3.3 V
    _f("refin_mode_diff",             4,   9,  1),
    _f("charge_pump_current",         4,  10,  4),
    _f("double_buffer",               4,  14,  1),
    _f("r_counter",                   4,  15, 10),
    _f("rdiv2",                       4,  25,  1),
    _f("ref_doubler",                 4,  26,  1),
    _f("muxout",                      4,  27,  3),
    # R6
    _f("output_power",                6,   4,  2),
    _f("rf_out_enable",               6,   6,  1),
    _f("rf_outb_disable",             6,  10,  1),   # active-low on ADF5355
    _f("mute_till_lock",              6,  11,  1),
    _f("cp_bleed_current",            6,  13,  8),
    _f("rf_divider_select",           6,  21,  3),
    _f("feedback_fundamental",        6,  24,  1),
    _f("negative_bleed",              6,  29,  1),
    _f("gated_bleed",                 6,  30,  1),
    # R7
    _f("ld_mode_int_n",               7,   4,  1),
    _f("frac_n_ld_precision",         7,   5,  2),
    _f("lol_mode",                    7,   7,  1),
    _f("ld_cycle_count",              7,   8,  2),
    _f("le_synced_refin",             7,  25,  1),
    # R9
    _f("synth_lock_timeout",          9,   4,  5),
    _f("alc_timeout",                 9,   9,  5),
    _f("timeout",                     9,  14, 10),
    _f("vco_band_division",           9,  24,  8),
    # R10
    _f("adc_enable",                 10,   4,  1),
    _f("adc_conversion_enable",      10,   5,  1),
    _f("adc_clk_divider",            10,   6,  8),
    # R12
    _f("phase_resync_clk_divider",   12,  16, 16),
)}


# Base words: reserved bits that must be written as-is, plus the address
# nibble.  R5, R8 and R11 are wholly reserved -- they have no user fields.
BASE_WORDS = {
    0:  0x00000000,
    1:  0x00000001,
    2:  0x00000002,
    3:  0x00000003,
    4:  0x00000004,
    5:  0x00800025,   # reserved, fixed
    6:  0x14000006,   # reserved bits 26 and 28 set
    7:  0x10000007,   # reserved bit 28 set
    8:  0x102D0428,   # reserved, fixed
    9:  0x00000009,
    10: 0x00C0000A,   # reserved bits 22 and 23 set
    11: 0x0061300B,   # reserved, fixed
    12: 0x0000041C,   # reserved bits 2,3,4,10 set
}

FIXED_REGISTERS = (5, 8, 11)


class RegisterFile:
    """Accumulates field assignments into the 13 register words."""

    def __init__(self):
        self._words = dict(BASE_WORDS)

    def set(self, name: str, value) -> "RegisterFile":
        try:
            field = FIELDS[name]
        except KeyError:
            raise KeyError(f"unknown ADF5355 field {name!r}") from None
        self._words[field.reg] |= field.encode(int(value))
        return self

    def update(self, **kwargs) -> "RegisterFile":
        for name, value in kwargs.items():
            self.set(name, value)
        return self

    def word(self, reg: int) -> int:
        # The address nibble is re-applied unconditionally; a base word can
        # never be written without it.
        return (self._words[reg] | reg) & 0xFFFFFFFF

    @property
    def words(self) -> list[int]:
        return [self.word(i) for i in range(N_REGS)]

    def get(self, name: str) -> int:
        field = FIELDS[name]
        return field.decode(self.word(field.reg))

    def describe(self, reg: int) -> str:
        parts = [
            f"{f.name}={f.decode(self.word(reg))}"
            for f in FIELDS.values() if f.reg == reg
        ]
        tail = "  " + " ".join(parts) if parts else "  (reserved)"
        return f"R{reg:<2} 0x{self.word(reg):08X}{tail}"

    def dump(self) -> str:
        return "\n".join(self.describe(i) for i in range(N_REGS))


def to_bytes(word: int) -> list[int]:
    """32-bit word -> 4 bytes, MSB first, as the ADF5355 shifts them."""
    return [(word >> 24) & 0xFF, (word >> 16) & 0xFF,
            (word >> 8) & 0xFF, word & 0xFF]
