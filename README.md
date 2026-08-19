# adf5355_tester

Control an ADF5355 wideband synthesizer (54 MHz – 13.6 GHz) from a Raspberry Pi.

> ## ⚠️ Closed RF paths only — never radiate
>
> Everything in this repository is a **bench procedure into a closed, shielded,
> attenuated path**. The ladder and the calibration described below are
> theoretical exercises for conducted test setups only.
>
> **Never connect an antenna. Never transmit any of this over the air.**
>
> The default ladder occupies 10.7–12.7 GHz, which is *satellite downlink*
> spectrum. Terrestrial transmission there is prohibited in essentially every
> jurisdiction, and even a few milliwatts near a dish or LNB can wipe out
> reception well beyond your own bench. Keep the synthesiser output conducted:
> coax into a load, an attenuator, or a shielded enclosure.

Provides a single `adf5355` command for bring-up, register inspection, single-tone
transmission, band sweeps with lock verification, and a duration-coded frequency
ladder that a downstream receiver can use to measure its own clock error.

---

## Install

```bash
uv tool install --editable ~/adf5355_tester
```

That puts `adf5355` on your PATH (`~/.local/bin`). Everything below also works
without installing, as `python3 -m adf5355 <cmd>` from the repository root.

SPI must be enabled once:

```bash
sudo raspi-config nonint do_spi 0     # or uncomment dtparam=spi=on in
sudo reboot                           # /boot/firmware/config.txt
```

`sudo dtoverlay spi0-2cs` loads it immediately without rebooting.

---

## Wiring

```
   Raspberry Pi 4B                                   ADF5355 board
   40-pin header
  ┌────────────────────────┐                      ┌───────────────────────────┐
  │ pin 19  GPIO10 / MOSI  │──────── DAT ────────▶│ DAT   serial data         │
  │ pin 23  GPIO11 / SCLK  │──────── CLK ────────▶│ CLK   serial clock        │
  │ pin 24  GPIO8  / CE0   │──────── LE  ────────▶│ LE    latch enable        │
  │ pin 16  GPIO23         │──────── CE  ────────▶│ CE    chip enable         │
  │ pin 35  GPIO19         │◀─────── MUX ─────────│ MUX   lock detect         │
  │ pin 6   GND            │──────── GND ─────────│ GND                       │
  └────────────────────────┘                      │                           │
                                                  │ X1 = 125.000 MHz          │
   external 3.3 V supply ────── VDD ─────────────▶│ VDD                       │
              └───────────────── common GND ─────▶│ GND                       │
                                                  │                           │
                                                  │ OUT+  RFoutA   53 MHz–6.8 GHz
                                                  │ OUT−  RFoutA   (50 Ω load)│
                                                  │ OB    RFoutB   6.8–13.6 GHz
                                                  │ REF+/REF−  external ref in│
                                                  └───────────────────────────┘
```

| Pi signal | Header pin | Board | Purpose |
|---|---|---|---|
| GPIO10 / MOSI | 19 | DAT | Serial data |
| GPIO11 / SCLK | 23 | CLK | Serial clock |
| GPIO8 / CE0 | 24 | LE | Load enable / latch |
| GPIO23 | **16** | CE | Chip enable — note: *pin 16*, not pin 23 |
| GPIO19 | 35 | MUX | Lock detect and link probe |
| GND | 6 | GND | Common ground |

SPI mode 0, MSB first, 1 MHz by default (the part accepts up to 50 MHz).

**Which port emits what.** `OUT+`/`OUT−` are the RFoutA differential pair
(53.125 MHz – 6.8 GHz); take the signal off `OUT+` and terminate `OUT−` into
50 Ω, since leaving it open degrades the match. `OB` is RFoutB, the frequency
doubler, and produces **only** 6.8–13.6 GHz — it is silent below that.
`REF+`/`REF−` are *inputs* for an external reference; leave them unconnected
when using the onboard oscillator.

**CE0 is the correct pin for LE by construction.** The Pi holds CE0 low for the
whole transfer and releases it high afterwards, so the 32 bits shift in with LE
low and the trailing rising edge latches them. This requires all four bytes to
go out in a *single* transfer — never one byte at a time.

**Power the board from its own supply**, not the Pi's 3V3 pin. The ADF5355
draws several hundred mA and the Pi rail is both current-limited and
electrically noisy, which you pay for in phase noise. Tie the grounds together.

### The chip cannot be read back

DAT is an input, there is no MISO, and no register is readable. MUXOUT is the
only signal the ADF5355 can drive back. Two of its settings are static logic
levels, so commanding it high and then low and watching the GPIO follow proves
the entire chain — CE, CLK, DAT, LE and the chip's own register decode:

```bash
adf5355 probe
```

```
MUXOUT commanded high -> read 1
MUXOUT commanded low  -> read 0
PASS: the ADF5355 is receiving and acting on SPI writes
```

Run this first whenever something does not work. `PASS` means the link is fine
and any failure is downstream (reference, loop, lock). `FAIL` means the chip is
not hearing you at all — check board power, CE, and the DAT/CLK/LE wiring.

Without MUXOUT wired (`--muxout-gpio -1`) the part is entirely open loop: writes
cannot be confirmed, lock cannot be detected, and the only way to know the
synthesizer did what you asked is to measure the RF output.

---

## Commands

```bash
adf5355 probe                                   # is the chip receiving writes?
adf5355 dump   --freq 2.4G                      # registers, no hardware touched
adf5355 set    --freq 2.4G --enable-rf --hold 5
adf5355 dwell  --freq 2.4G --dwell 30 --enable-rf
adf5355 ladder --enable-rf                      # duration-coded Ku-band pattern
adf5355 sweep  --start 1G --stop 6G --points 51
adf5355 off
```

| Command | What it does |
|---|---|
| `probe` | Commands MUXOUT high then low and watches the GPIO follow. The only readback this part has. |
| `dump` | Solves a frequency and prints the full register image with every field named, plus the write order. Touches no hardware. |
| `set` | Programs one frequency, optionally holding it with `--hold`. |
| `dwell` | Transmits one frequency for `--dwell` seconds, then mutes and exits. |
| `ladder` | Duration-coded ladder — see below. |
| `sweep` | Steps a range and checks lock at every point. Exits non-zero if any point fails. |
| `off` | Disables both outputs. |

**RF is never enabled without `--enable-rf`.** Without it the PLL is fully
programmed and locks, but both outputs stay off — which is how you verify lock
across a whole band without transmitting. `--dry-run` computes everything and
touches no hardware at all.

### Reference frequency

This board's reference is **125.000 MHz** (marked `R125.000` on X1), and that is
the `--ref-mhz` default, so the examples omit it. Pass `--ref-mhz` explicitly for
any board whose X1 is marked otherwise.

Getting this wrong is the single most likely reason a sweep runs cleanly but
never locks. The registers are computed against the assumed reference, so a
wrong value scales every VCO target by the ratio of real to assumed reference;
only the points that happen to land back inside 3.4–6.8 GHz will lock. **If you
see scattered `LOCK FAILED` lines with no frequency pattern, check this first.**

### Output power

`--power` selects one of four RFoutA steps: `0` = −4 dBm (**default**), `1` = −1,
`2` = +2, `3` = +5 dBm. It defaults to the lowest step so that forgetting the
flag cannot damage whatever is on the other end of the cable — +5 dBm is already
past a PlutoSDR front end's limit. There is no finer control on this part; use
an external attenuator. RFoutB has enable only, so `--power` does nothing on
`--channel B`.

### Pacing a sweep

`--dwell` is how long each point is held after it locks (default `0.02` s) and
`--points` is the resolution. To follow a sweep on an analyser:

```bash
adf5355 sweep --start 1G --stop 6G --points 51 --dwell 1
```

`--lock-timeout` (default `0.5` s) is how long to wait before calling a point
failed; a healthy lock takes single-digit milliseconds.

---

## The duration-coded ladder

```bash
adf5355 ladder --enable-rf                            # the guide's pattern
adf5355 ladder --start-ghz 8 --stop-ghz 9 --steps 4 --total-s 6
adf5355 ladder --loops 3 --enable-rf
```

N carriers are stepped evenly across a range. **Rung *n* transmits for *n*·u
seconds, then stays quiet for *n*·u seconds**, where the unit time follows from
the requested total:

```
    u = total_s / (N · (N + 1))
```

because the windows sum to `2·u·(1 + 2 + … + N) = u·N·(N+1)`.

Every rung therefore has a **distinct burst length**, and the length of a burst
alone identifies which rung produced it — and so its frequency. No handshake,
no shared clock, no agreement on a start time is required between transmitter
and receiver.

Rung *n* occupies, measured from the start of the coded interval:

```
    ON   from  u·n·(n−1)   to  u·n²
    OFF  from  u·n²        to  u·n·(n+1)
    burst length = u·n
    frequency    = f_start + (f_stop − f_start) · (n−1)/(N−1)
```

The defaults reproduce `raspberry_pi_adf5355_ku_ladder_guide.pdf` exactly —
nine carriers, 10.700 to 12.700 GHz in 250 MHz steps, 18.000 s total, u = 0.200 s:

| rung | frequency | burst | ON window (s) | OFF until |
|---:|---:|---:|---:|---:|
| 1 | 10.700 GHz | 0.2 s | 0.00 – 0.20 | 0.40 |
| 2 | 10.950 GHz | 0.4 s | 0.40 – 0.80 | 1.20 |
| 3 | 11.200 GHz | 0.6 s | 1.20 – 1.80 | 2.40 |
| 4 | 11.450 GHz | 0.8 s | 2.40 – 3.20 | 4.00 |
| 5 | 11.700 GHz | 1.0 s | 4.00 – 5.00 | 6.00 |
| 6 | 11.950 GHz | 1.2 s | 6.00 – 7.20 | 8.40 |
| 7 | 12.200 GHz | 1.4 s | 8.40 – 9.80 | 11.20 |
| 8 | 12.450 GHz | 1.6 s | 11.20 – 12.80 | 14.40 |
| 9 | 12.700 GHz | 1.8 s | 14.40 – 16.20 | 18.00 |

A 1.4 s burst is rung 7, and therefore 12.200 GHz.

`ladder` defaults to `--channel B`, since 10.7–12.7 GHz is out of RFoutA's
range. Each rung's frequency is programmed during the *previous* rung's OFF
window, so retuning never eats into a coded ON window and the burst lengths stay
true to the schedule.

---

## Running a calibration end to end

> **Conducted path only.** This example is written for a closed bench setup: the
> synthesiser reaches the LNB through coax and attenuation, inside shielding.
> Do not radiate it, and do not put an antenna on either end. The frequencies
> involved are satellite downlink allocations.
>
> Note also that an LNB front end expects roughly −100 dBm. Feeding one directly
> from the synthesiser without heavy attenuation will saturate it and can damage
> it, quite apart from the licensing question.

A complete worked example against a PlutoSDR behind a 13 V (low band) universal
LNB, whose nominal LO is 9.750 GHz.

**Frequency plan.** Five rungs, 10.7 to 11.5 GHz in 200 MHz steps, one pass
every 30 s, so `u = 30 / (5 x 6) = 1.0 s` and the bursts are 1, 2, 3, 4 and 5 s.
Subtracting the LNB LO gives the IF each rung lands on:

| rung | burst | RF | IF (RF − 9.75 GHz) |
|---:|---:|---:|---:|
| 1 | 1 s | 10.700 GHz | 950 MHz |
| 2 | 2 s | 10.900 GHz | 1150 MHz |
| 3 | 3 s | 11.100 GHz | 1350 MHz |
| 4 | 4 s | 11.300 GHz | 1550 MHz |
| 5 | 5 s | 11.500 GHz | 1750 MHz |

Those IFs sit inside the LNB's low-band output passband (roughly 950–2150 MHz).
13 V with no 22 kHz tone selects low band; high band would use a 10.6 GHz LO and
move every rung.

### 1. Transmit (this repository, on the Pi driving the ADF5355)

Start the ladder first and leave it looping for the whole receive session. Twelve
passes is 6 minutes, comfortably longer than five 40 s captures:

```bash
adf5355 ladder --start-ghz 10.7 --stop-ghz 11.5                --steps 5 --total-s 30                --loops 12 --power 0 --enable-rf
```

The receiver is never told when a rung keys. It recovers that from burst length.

### 2. Receive and solve (pluto-plus-utils, on the host with the radio)

One capture per rung, retuned to that rung's nominal IF. Each capture must be
longer than one full pass plus the longest burst, so the rung is guaranteed to
appear complete rather than clipped at an edge:

```bash
for IF in 950000000 1150000000 1350000000 1550000000 1750000000; do
    uv run pluto radio settings set RADIO --frequency $IF
    uv run pluto capture start RADIO --duration 40
done
```

Then hand the artifact IDs to the calibrator, telling it the *published* ladder
parameters — the same numbers used above:

```bash
uv run pluto calibrate freq-ladder ART_1 ART_2 ART_3 ART_4 ART_5     --rung-start-hz 10.7e9 --rung-stop-hz 11.5e9 --rung-count 5     --total-seconds 30 --lo-hz 9.75e9
```

It identifies each burst by duration, maps it back to that rung's frequency, and
fits the receiver's clock error (slope) and the LNB's LO error (intercept). A
single capture can be inspected on its own with
`pluto analyze ARTIFACT_ID --analyzer freq_ladder`.

### 3. Apply

```
    xo_new = xo_current x (1 + d_rx)
```

Re-run the calibration afterwards and confirm the fitted slope collapses toward
zero — that also confirms the sign.

### Choosing the parameters

- **Span as wide as the LNB IF allows.** The slope is the receiver's clock error,
  so its precision is roughly (measurement error) / (IF span). A 2 GHz span is
  worth far more than a 200 MHz one.
- **At least 3 rungs**, or slope and intercept cannot be separated. Five or more
  leaves room for leave-one-out uncertainty.
- **Several passes.** A single monotonic pass confounds LO drift with the slope.
- **Keep `total_s` generous** so the shortest burst is comfortably longer than a
  capture frame; `u = total_s / (N(N+1))`.

## Using the ladder to measure a receiver's clock offset

This is what the ladder is for. A downstream SDR can recover both its **clock
error** and any **fixed frequency offset** in its receive chain, without being
told which rung is which and without sharing a time base.

### Why it works

The synthesizer's output is derived entirely from its 125 MHz reference. A
receiver measures that tone using its own reference. Writing δ for a
reference's fractional error, the measured frequency of a rung comes out as

```
    f_measured  ≈  f_nominal · (1 + δ_tx − δ_rx)
```

so the fractional discrepancy is the **difference between the two references**:

```
    Δf / f_nominal  =  δ_tx − δ_rx  ≡  ε
```

If you trust the ADF5355's reference, ε is the receiver's clock error, and its
sign tells you the direction: a tone measuring **high** means the receiver's
clock is running **slow**.

### Why a ladder rather than a single tone

A single tone cannot distinguish a clock error from a constant offset. The two
have different signatures across frequency:

- a **clock error** scales with frequency — `Δf = ε · f`, a line through the origin
- a **fixed offset** (LO/NCO/IF error, a mis-set correction) does not — `Δf = c`

Measuring several rungs and fitting

```
    Δf(f)  =  ε · f  +  c
```

separates them: the **slope** is the clock error, the **intercept** is the fixed
offset. The default ladder spans 2 GHz, which is a long lever arm — at 1 ppm the
offset grows from 10,700 Hz at rung 1 to 12,700 Hz at rung 9, a 2,000 Hz spread
that is trivially measurable.

### Procedure

1. **Capture.** Tune the SDR to each rung's nominal frequency in turn, or use a
   span wide enough to see several. Rung 1 is the shortest burst at 0.2 s, so
   size the capture and FFT to resolve that.
2. **Segment and identify.** Detect burst edges — the OFF gaps make this
   unambiguous and also give you a noise-floor reference. Divide each burst
   length by `u` to get the rung number *n*, and hence the nominal frequency
   `f_start + 250 MHz · (n−1)`. Round to the nearest integer; the lengths are
   0.2 s apart, so identification is robust.
3. **Estimate each tone.** Interpolated FFT peak, or a phase-slope estimate over
   the burst. Longer rungs give better resolution — rung 9's 1.8 s burst gives
   about 0.55 Hz, which at 12.7 GHz is **0.04 ppb**. Conveniently the longest
   bursts are also the highest frequencies, where a given ε produces the largest
   offset, so precision improves at both ends of the product.
4. **Fit.** Compute `Δf_n = f_measured,n − f_nominal,n` and least-squares fit
   `Δf = ε·f + c` across the rungs.

### Worked example

Suppose three rungs measure:

| rung | nominal | measured | Δf |
|---:|---:|---:|---:|
| 1 | 10.700 GHz | 10.700010720 GHz | +10,720 Hz |
| 5 | 11.700 GHz | 11.700011721 GHz | +11,721 Hz |
| 9 | 12.700 GHz | 12.700012722 GHz | +12,722 Hz |

```
    ε = (12722 − 10720) / (12.7e9 − 10.7e9) = 2002 / 2e9 = +1.001e-6   →  +1.00 ppm
    c = 10720 − 1.001e-6 × 10.7e9                                      →  +9 Hz
```

The receiver's reference is **1.00 ppm low**, plus a 9 Hz fixed offset in its
chain. Had you measured only rung 5, you would have seen 11,721 Hz and been
unable to say how much of it was which.

### Applying the correction

For a PlutoSDR, the reference correction is `ad9361-phy,xo_correction` (nominally
40 MHz):

```
    xo_new = xo_current × (1 − ε)
```

Re-run the ladder afterwards and confirm the residual ε has collapsed toward
zero — that also confirms you got the sign right, which is easier than reasoning
about it.

### Caveats

- This measures the **difference** between the two references. It only gives the
  receiver's absolute error if the ADF5355's reference is the trusted one.
- Any real Doppler or drift folds into ε. Over a cable on a bench, both are zero.
- Keep the path attenuated. The ladder's default range is satellite downlink
  spectrum — see Safety.

---

## Layout

Everything except `device.py` is pure computation and is tested with no hardware
attached.

| File | Role |
|---|---|
| `adf5355/registers.py` | Bit positions — the single source of truth |
| `adf5355/plan.py` | Frequency → field values (solver + register assembly) |
| `adf5355/ladder.py` | Duration-coded ladder pattern and runner |
| `adf5355/device.py` | spidev + GPIO, write ordering, autocal, lock detect |
| `adf5355/cli.py` | Command line front end |
| `adf5355/entry.py` | Process entry point and shutdown handling |
| `adf5355_ladder.py` | Original standalone bench script, kept as a reference |

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

---

## Testing

```bash
python3 -m unittest discover -s tests -t .
```

90 tests, no hardware required. The register images are checked three ways:

1. **`tests/test_registers.py`** — the bitfield table itself: no field overlaps
   another, collides with reserved bits, or corrupts the address nibble.
2. **`tests/test_vs_adi_c.py`** — a differential test against
   `tests/adi_reference.c`, a verbatim transcription of the arithmetic in
   Analog Devices' no-OS driver, compiled on demand and compared field by field
   across a grid of references, frequencies and charge-pump settings.
3. **`tests/test_ladder_parity.py`** and **`tests/test_ladder_package_parity.py`** —
   the package must reproduce `adf5355_ladder.py`'s register images and ladder
   schedules exactly. That script has been used on the bench, so it counts as a
   third independent implementation.

### Known defects in the Analog Devices reference

The C driver is the authority on bit positions, but its arithmetic has three
bugs. Each is corrected here and marked `DIVERGENCE` in `plan.py`.

1. **MOD2 can overflow its field.** The loop bound permits `MOD2 == 16384`,
   which does not fit the 14-bit field and encodes as `MOD2 = 0` — a
   divide-by-zero in the sigma-delta modulator. Reachable when f_PFD is
   `16384 × 2^k` and the reduced FRAC2 is odd, e.g. f_PFD = 67.108864 MHz.

2. **R7 is always wrong.** The driver builds it as `a | b | … | cond ? X : Y`.
   `?:` binds looser than `|`, so the whole left side becomes the condition,
   which is always truthy — the driver emits the ADF5356 default `0x04000007`
   for every part, including the ADF5355. The correct value is `0x12000067`.

3. **The ADC clock divider can exceed its limit.** The driver computes
   `DIV_ROUND_UP(fpfd / 100000U - 2, 4)`; the inner integer division truncates
   before the ceiling and can round the divider down by one. At f_PFD =
   61.44 MHz (a 122.88 MHz reference, common on clone boards) that gives
   `adc_div = 153` and a 100,065 Hz ADC clock, over the datasheet's 100 kHz
   ceiling for VCO band select.

---

## Safety

**These are theoretical bench tests and must never be performed over open air.**

The default ladder sweeps 10.7–12.7 GHz, which overlaps satellite downlink
allocations. Transmitting there terrestrially is prohibited in essentially every
jurisdiction, requires a licence you almost certainly do not hold, and can
interfere with satellite reception far outside your own site.

Every procedure in this repository assumes a **closed, conducted path**: coax
from the synthesiser into an attenuator and a load, or into a shielded
enclosure. **Do not connect an antenna. Do not radiate.** If you cannot
guarantee the path is closed, do not pass `--enable-rf`.

`dump`, `probe` and any `--dry-run` invocation never key the output, so the
register work and the arithmetic can all be exercised with no RF at all.

`--power` defaults to the lowest step, but +5 dBm is available and will damage a
PlutoSDR front end (limit around +2.5 dBm). Pad the path 20–30 dB before
connecting a receiver.
