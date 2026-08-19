#!/usr/bin/env bash
#
# TRANSMIT side of a frequency-ladder calibration.
#
#   ####################################################################
#   #  CLOSED, CONDUCTED PATHS ONLY.  NEVER RADIATE.                   #
#   #                                                                  #
#   #  These frequencies are satellite downlink spectrum. Terrestrial  #
#   #  transmission there is prohibited in essentially every           #
#   #  jurisdiction. Coax into an attenuator and a load, or a shielded #
#   #  enclosure. No antenna on either end.                            #
#   #                                                                  #
#   #  An LNB front end expects about -100 dBm. Feeding one directly   #
#   #  without heavy attenuation will saturate and can damage it.      #
#   ####################################################################
#
# Emits a duration-coded ladder narrow enough that a receiver sitting at ONE
# tuning hears every rung: rung n transmits for n*u seconds then is quiet for
# n*u, so a burst's length identifies which rung -- and therefore which
# frequency -- it was. No control channel to the receiver is needed.
#
# Run this first and leave it going, then run sdr_listen.sh in another shell.
#
# Override any of the settings below from the environment, e.g.
#     STEPS=24 TOTAL_S=6.0 ./adf5355_rf_ladder.sh
#
set -euo pipefail

# ---- ladder definition (keep in step with sdr_listen.sh) -------------------
START_GHZ="${START_GHZ:-11.0}"        # first rung
STOP_GHZ="${STOP_GHZ:-11.00171}"      # last rung; span must fit the receiver
STEPS="${STEPS:-20}"                  # number of rungs
TOTAL_S="${TOTAL_S:-4.2}"             # one complete cycle
# ---------------------------------------------------------------------------
LOOPS="${LOOPS:-10}"                  # cycles to transmit
POWER="${POWER:-0}"                   # 0 = -4 dBm, the lowest step
CHANNEL="${CHANNEL:-B}"               # B = 6.8-13.6 GHz doubler output (OB)

ADF="${ADF:-adf5355}"
command -v "$ADF" >/dev/null 2>&1 || {
    echo "error: '$ADF' not on PATH. Install with:" >&2
    echo "         uv tool install --editable ~/adf5355_tester" >&2
    exit 1
}

python3 - "$START_GHZ" "$STOP_GHZ" "$STEPS" "$TOTAL_S" "$LOOPS" <<'PY'
import sys
start, stop, steps, total, loops = (float(sys.argv[1]), float(sys.argv[2]),
                                    int(sys.argv[3]), float(sys.argv[4]),
                                    int(sys.argv[5]))
u = total / (steps * (steps + 1))
span = (stop - start) * 1e9
spacing = span / (steps - 1) if steps > 1 else 0
print(f"  rungs        : {steps}  from {start:.6f} to {stop:.6f} GHz")
print(f"  span         : {span/1e6:.3f} MHz  (spacing {spacing/1e3:.1f} kHz)")
print(f"  unit time u  : {u*1e3:.1f} ms   bursts {u*1e3:.0f} ms to {u*steps*1e3:.0f} ms")
print(f"  cycle        : {total:g} s   transmitting {loops} cycles = {total*loops:.1f} s")
PY

cat <<'WARN'

  SAFETY: closed, conducted path only. Do not connect an antenna.
          Satellite downlink spectrum -- never radiate this.

WARN

echo "starting transmitter (Ctrl-C to stop; outputs mute on exit)"
exec "$ADF" ladder \
    --start-ghz "$START_GHZ" \
    --stop-ghz  "$STOP_GHZ" \
    --steps     "$STEPS" \
    --total-s   "$TOTAL_S" \
    --loops     "$LOOPS" \
    --channel   "$CHANNEL" \
    --power     "$POWER" \
    --enable-rf
