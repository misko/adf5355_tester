"""Bitfield table integrity -- no hardware, no reference needed."""
import unittest

from . import context  # noqa: F401
from adf5355.registers import (BASE_WORDS, FIELDS, FIXED_REGISTERS, N_REGS,
                               Field, RegisterFile, to_bytes)


class TestFieldTable(unittest.TestCase):
    def test_no_overlapping_fields_within_a_register(self):
        for reg in range(N_REGS):
            used = 0
            for f in FIELDS.values():
                if f.reg != reg:
                    continue
                self.assertEqual(used & f.mask, 0,
                                 f"{f.name} overlaps another field in R{reg}")
                used |= f.mask

    def test_fields_never_collide_with_the_address_nibble(self):
        for f in FIELDS.values():
            self.assertGreaterEqual(f.lsb, 4,
                                    f"{f.name} would corrupt the R{f.reg} address")

    def test_fields_never_collide_with_reserved_base_bits(self):
        for f in FIELDS.values():
            base = BASE_WORDS[f.reg] & ~0xF
            self.assertEqual(base & f.mask, 0,
                             f"{f.name} overlaps reserved bits of R{f.reg}")

    def test_fields_fit_in_32_bits(self):
        for f in FIELDS.values():
            self.assertLessEqual(f.lsb + f.width, 32, f.name)

    def test_encode_decode_round_trip(self):
        for f in FIELDS.values():
            for value in (0, 1, (1 << f.width) - 1):
                self.assertEqual(f.decode(f.encode(value)), value, f.name)

    def test_encode_rejects_overflow(self):
        f = Field("demo", 0, 4, 3)
        with self.assertRaises(ValueError):
            f.encode(8)


class TestRegisterFile(unittest.TestCase):
    def test_address_nibble_always_present(self):
        regs = RegisterFile()
        for i in range(N_REGS):
            self.assertEqual(regs.word(i) & 0xF, i, f"R{i} address nibble")

    def test_reserved_registers_match_the_datasheet_constants(self):
        regs = RegisterFile()
        expected = {5: 0x00800025, 8: 0x102D0428, 11: 0x0061300B}
        for reg, word in expected.items():
            self.assertEqual(regs.word(reg), word, f"R{reg}")

    def test_reserved_registers_have_no_writable_fields(self):
        for reg in FIXED_REGISTERS:
            owned = [f.name for f in FIELDS.values() if f.reg == reg]
            self.assertEqual(owned, [], f"R{reg} should be wholly reserved")

    def test_set_is_idempotent_per_field_and_composes(self):
        regs = RegisterFile().set("int", 585).set("prescaler", 1).set("autocal", 1)
        self.assertEqual(regs.word(0), 0x00302490)
        self.assertEqual(regs.get("int"), 585)

    def test_unknown_field_is_rejected(self):
        with self.assertRaises(KeyError):
            RegisterFile().set("not_a_field", 1)

    def test_to_bytes_is_msb_first(self):
        self.assertEqual(to_bytes(0x12345678), [0x12, 0x34, 0x56, 0x78])
        self.assertEqual(len(to_bytes(0)), 4)


if __name__ == "__main__":
    unittest.main()
