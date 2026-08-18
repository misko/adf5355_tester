# adf5355_tester

ADF5355 wideband synthesizer (54 MHz – 13.6 GHz) driven from a Raspberry Pi.

## Wiring

| Raspberry Pi | Header pin | Board | Purpose |
|---|---|---|---|
| GPIO10 / MOSI | 19 | DAT | Serial data |
| GPIO11 / SCLK | 23 | CLK | Serial clock |
| GPIO8 / CE0 | 24 | LE | Load enable / latch |
| GPIO23 | 16 | CE | Chip enable — held low at startup, driven high to run |
| GPIO19 | 35 | MUX | MUXOUT — lock detect and link probe |
| GND | 6 | GND | Common ground |

SPI mode 0, MSB first, 1 MHz by default (the part accepts up to 50 MHz).

**CE0 is the correct pin for LE by construction.** The Pi holds CE0 low for the
whole transfer and releases it high afterwards, so the 32 bits shift in with LE
low and the trailing rising edge latches them. This requires all four bytes to
go out in a *single* transfer — never one byte at a time.

Enable SPI once with `sudo raspi-config nonint do_spi 0` (or uncomment
`dtparam=spi=on` in `/boot/firmware/config.txt`). `sudo dtoverlay spi0-2cs`
loads it immediately without a reboot.

Power the board from its own supply, not the Pi's 3V3 pin: the ADF5355 draws
several hundred mA and the Pi rail is both current-limited and electrically
noisy, which you would pay for in phase noise. Tie the grounds together.

## The chip cannot be read back

DAT is an input, there is no MISO, and no register is readable. MUXOUT is the
only signal the ADF5355 can drive back. Two of its settings are static logic
levels, so commanding it high and then low and watching the GPIO follow proves
the entire chain — CE, CLK, DAT, LE and the chip's own register decode:

```bash
python3 -m adf5355 probe --ref-mhz 40
```

Without MUXOUT wired (`--muxout-gpio -1`) the part is entirely open loop: writes
cannot be confirmed, lock cannot be detected, and the only way to know the
synthesizer did what you asked is to measure the RF output.

## Usage

```bash
python3 -m adf5355 dump  --ref-mhz 40 --freq 2.4G          # registers, no hardware
python3 -m adf5355 set   --ref-mhz 40 --freq 2.4G --enable-rf --hold 5
python3 -m adf5355 set   --ref-mhz 40 --freq 11.7G --channel B --enable-rf
python3 -m adf5355 sweep --ref-mhz 40 --start 1G --stop 6G --points 51
python3 -m adf5355 off   --ref-mhz 40
```

`--ref-mhz` is required and never guessed: clone boards ship different
oscillators, and every register depends on it. Read the marking on X1.

RF is never enabled without `--enable-rf`. `--dry-run` computes everything and
touches no hardware.

## Layout

Everything except `device.py` is pure computation and is tested with no
hardware attached.

| File | Role |
|---|---|
| `adf5355/registers.py` | Bit positions — the single source of truth |
| `adf5355/plan.py` | Frequency → field values (solver + register assembly) |
| `adf5355/device.py` | spidev + GPIO, write ordering, autocal, lock detect |
| `adf5355/cli.py` | Command line front end |
| `adf5355_ladder.py` | Standalone Ku-band duration-coded ladder (bench test) |

### Frequency solving

```
f_PFD = f_REF × (1 + D) / (R × (1 + T))
f_VCO = f_out × 2^d          d = RF divider select, f_VCO ∈ [3.4, 6.8] GHz
N     = f_VCO / f_PFD = INT + (FRAC1 + FRAC2/MOD2) / 2^24
```

Two solvers are available. `adi` reproduces the reference driver exactly.
`exact` (`--solver exact`) picks the closest representable FRAC2/MOD2 with
`Fraction.limit_denominator` instead of truncating on each halving step, and is
never worse. Achieved frequency and error are reported as exact rationals, so
the residual is real rather than a float artifact.

## Testing

```bash
python3 -m unittest discover -s tests -t .
```

41 tests, no hardware required. The register images are checked three ways:

1. **`tests/test_registers.py`** — the bitfield table itself: no field overlaps
   another, collides with reserved bits, or corrupts the address nibble.
2. **`tests/test_vs_adi_c.py`** — a differential test against
   `tests/adi_reference.c`, a verbatim transcription of the arithmetic in
   Analog Devices' no-OS driver, compiled on demand and compared field by
   field across a grid of references, frequencies and charge-pump settings.
3. **`tests/test_ladder_parity.py`** — the package must reproduce
   `adf5355_ladder.py`'s register images exactly. That script has been used on
   the bench, so it counts as a third independent implementation.

### Known defects in the Analog Devices reference

The C driver is the authority on bit positions, but its arithmetic has three
bugs. Each is corrected here and marked `DIVERGENCE` in `plan.py`.

1. **MOD2 can overflow its field.** The loop bound permits `MOD2 == 16384`,
   which does not fit the 14-bit field and encodes as `MOD2 = 0` — a
   divide-by-zero in the sigma-delta modulator. Reachable when f_PFD is
   `16384 × 2^k` and the reduced FRAC2 is odd, e.g. f_PFD = 67.108864 MHz.

2. **R7 is always wrong.** The driver builds it as
   `a | b | … | cond ? X : Y`. `?:` binds looser than `|`, so the whole left
   side becomes the condition, which is always truthy — the driver emits the
   ADF5356 default `0x04000007` for every part, including the ADF5355. The
   correct ADF5355 value is `0x12000067`.

3. **The ADC clock divider can exceed its limit.** The driver computes
   `DIV_ROUND_UP(fpfd / 100000U - 2, 4)`; the inner integer division truncates
   before the ceiling and can round the divider down by one. At f_PFD =
   61.44 MHz (a 122.88 MHz reference, common on clone boards) that gives
   `adc_div = 153` and a 100,065 Hz ADC clock, over the datasheet's 100 kHz
   ceiling for VCO band select. Solving the inequality directly in integers
   avoids the lost remainder.

## Safety

`adf5355_ladder.py` sweeps 10.7–12.7 GHz, which overlaps satellite downlink
allocations. Use a closed, shielded, attenuated path. Do not radiate.
