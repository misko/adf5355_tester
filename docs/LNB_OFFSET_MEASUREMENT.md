# Measuring a PlutoSDR's clock error through an LNB

Result of running the ADF5355 ladder into a 13 V (low band, vertical) universal
LNB feeding a PlutoSDR on RX0.

```
   ADF5355  ──RF──▶  LNB  ──IF──▶  PlutoSDR RX0
   125 MHz ref       LO ≈ 9.75 GHz   40 MHz ref
                     (free-running)
```

## Result

| Quantity | Value |
|---|---|
| **PlutoSDR reference error** | **+8.94 ppm high** (≈ +357 Hz on its 40 MHz XO) |
| **LNB LO error** | **+94.0 kHz high** (+9.64 ppm), actual LO ≈ 9.750094 GHz |
| System drift observed | −3 to −7 Hz/s over ~105 s |
| Realistic uncertainty | **±0.3 ppm**, systematic-limited (see below) |

Reproduced across four independent runs: +8.897, +8.938, +8.948, and +8.94 ppm
(mean of leave-one-out folds), with the LNB LO landing at +94.015, +94.031 and
+94.100 kHz. The run-to-run spread is far smaller than the systematic floor.

## Method

The LNB downconverts, so `f_IF = f_RF − f_LO`. Writing `δ` for a reference's
fractional error, the frequency the Pluto reports for a rung is

```
    reported  ≈  (f_RF − f_LO_nom·(1 + δ_lnb)) · (1 − δ_rx)
    Δf = reported − f_IF_nom  ≈  −δ_rx·f_IF_nom  −  δ_lnb·f_LO_nom
                                 \______________/   \______________/
                                  scales with IF      constant
```

The receiver's clock error scales with IF; the LNB's LO error does not. Fitting
`Δf = a·f_IF + b·t + c` therefore separates them:

- **slope `a`** → the Pluto's clock error, `δ_rx = −a`
- **intercept `c`** → the LNB's LO error, `δ_lnb = −c / f_LO_nom`
- **`b`** → a nuisance term for drift, which matters (see below)

A single tone cannot do this: it yields one number conflating both. That is the
reason to step frequency at all.

Rungs were identified the way any receiver would identify them — by burst
length — and the ladder was run at minimum power (−4 dBm).

## What the ladder run looked like

7 rungs, 10.7–11.9 GHz (IF 950–2150 MHz, inside the LNB's low-band IF passband),
28 s per loop, 4 loops, 28 measurements, all at 68–79 dB SNR.

```
  rung 1  RF 10.700 GHz  IF_nom  950.000  meas  949.897516  Δf −102.484 kHz
  rung 4  RF 11.300 GHz  IF_nom 1550.000  meas 1549.892393  Δf −107.607 kHz
  rung 7  RF 11.900 GHz  IF_nom 2150.000  meas 2149.886462  Δf −113.538 kHz
```

## Two corrections that mattered

**Drift is confounded with the answer.** The ladder steps frequency
monotonically in time, so any LO drift maps directly onto the slope being
measured. A single pass gave +9.011 ppm; adding a linear time term across
multiple loops gave +8.94 ppm, and revealed a real −3 to −7 Hz/s drift. Always
run several loops so drift is separable.

**The receiver's own tuning biases the answer.** Parking the ADF5355 on one
frequency and measuring it from eight different `rx_lo` settings gave answers
spanning **362 Hz** for a tone that never moved:

```
   baseband offset   measured absolute
      −900 kHz       1549.892142 MHz
      +1500 kHz      1549.892504 MHz
```

So the Pluto's absolute frequency readings carry a few hundred Hz of
tuning-dependent bias. That is the accuracy floor here, and it explains the
fit's residual exactly: residual rms is ~176 Hz with a reproducible
frequency-dependent shape (−117, −93, +31, +177, +232, +49, −279 Hz by rung,
repeating to within ~20 Hz across independent runs).

Because of that, the honest uncertainty is **±0.3 ppm** — the spread across
leave-one-rung-out folds — not the ±0.02 ppm the least-squares covariance
reports. Quoting the statistical error here would overstate the result by more
than tenfold.

A hypothesis that did *not* pan out: parking the tone away from baseband DC to
avoid the LO-leakage spur changed the residual not at all (176.3 vs 177.4 Hz).

## Applying the correction

```
    xo_new = xo_current × (1 + δ_rx) = 40e6 × (1 + 8.94e-6) ≈ 40,000,357.6 Hz
```

Re-run the ladder afterwards and confirm the fitted slope collapses toward zero.
That is also the cheapest way to confirm the sign.

**Not applied here** — changing `xo_correction` persists on the radio, so it is
left as the operator's call.

## Caveats

- This measures the **difference** between the two references. It gives the
  Pluto's absolute error only if the ADF5355's 125 MHz reference is trusted.
- The LNB LO figure comes from an intercept extrapolated from 950 MHz down to
  zero, so it is more sensitive to the systematic than the slope is.
- 13 V with no 22 kHz tone selects low band, LO 9.75 GHz. High band (10.6 GHz)
  needs the tone, and the rungs would move accordingly.

## Tools

| Script | Purpose |
|---|---|
| `tools/lnb_capture.py` | Single capture and interpolated peak estimate |
| `tools/lnb_acquire.py` | Finds the tone by differencing TX-on against TX-off |
| `tools/lnb_ladder_offset.py` | Runs the ladder, measures every rung, fits the model |

```bash
uv run python tools/lnb_ladder_offset.py \
    --start-ghz 10.7 --stop-ghz 11.9 --steps 7 --total-s 28 --loops 4
```
