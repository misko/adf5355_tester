#!/usr/bin/env bash
#
# TRANSMIT side of a seeded frequency-hop calibration.
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
# Hops among a set of frequencies in an order derived entirely from a shared
# seed. The receiver regenerates the identical schedule, so it never has to
# work out WHICH point it is hearing -- only WHEN the pattern started. Nothing
# is encoded in burst length, and no control channel to the receiver is needed.
#
# This replaces the duration-coded ladder, which never identified more than one
# burst in 95 on the bench. Seeded hopping identified 100% of points in every
# configuration tried.
#
# Run this first and leave it going, then run sdr_listen.sh in another shell.
#
# Override any of the settings below from the environment, e.g.
#     POINTS=40 HOP_MS=5 ./adf5355_rf_hop.sh
#
set -euo pipefail

# ---- schedule: EVERY LINE HERE MUST MATCH sdr_listen.sh --------------------
SEED="${SEED:-0xC0FFEE}"              # the whole protocol between the two ends
START_GHZ="${START_GHZ:-11.0}"        # first frequency point
STOP_GHZ="${STOP_GHZ:-11.00171}"      # last; span must fit the receiver
POINTS="${POINTS:-20}"                # frequency points, 90 kHz apart
HOP_MS="${HOP_MS:-10}"                # dwell per hop; precision tracks this
JITTER="${JITTER:-0}"                 # 0 = fixed dwell (measured no worse)
PERIOD_CYCLES="${PERIOD_CYCLES:-1}"   # permutations before the pattern repeats
# ---- transmit only --------------------------------------------------------
CYCLES="${CYCLES:-300}"               # permutations to transmit: 300 x 200 ms
POWER="${POWER:-0}"                   # 0 = -4 dBm, the lowest step
CHANNEL="${CHANNEL:-B}"               # B = 6.8-13.6 GHz doubler output (OB)
# ---------------------------------------------------------------------------

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ADF="${ADF:-adf5355}"
command -v "$ADF" >/dev/null 2>&1 || {
    echo "error: '$ADF' not on PATH. Install with:" >&2
    echo "         uv tool install --editable $REPO" >&2
    exit 1
}

# Preflight from the same generator the transmitter and the decoder both use,
# so what is printed here is the schedule, not a description of one.
REPO="$REPO" python3 - "$SEED" "$START_GHZ" "$STOP_GHZ" "$POINTS" "$HOP_MS" \
                       "$JITTER" "$PERIOD_CYCLES" "$CYCLES" <<'PY'
import os, sys
sys.path.insert(0, os.environ["REPO"])
from adf5355.hopper import (describe, make_schedule, period_duration,
                            plan_frequencies)

seed = int(sys.argv[1], 0)
start, stop = float(sys.argv[2]) * 1e9, float(sys.argv[3]) * 1e9
points, hop_ms, jitter = int(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
period_cycles, cycles = int(sys.argv[7]), int(sys.argv[8])

freqs = plan_frequencies(round(start), round(stop), points)
hops = make_schedule(seed, freqs, hop_ms / 1e3, cycles, jitter, period_cycles)
print(describe(hops, freqs, seed, hop_ms / 1e3, jitter))
print(f"  period   : {period_duration(hops, points, period_cycles)*1e3:.1f} ms"
      f"  -- the only interval the receiver has to search")
print(f"  run time : {hops[-1].end_s:.1f} s total -- leave this running while "
      f"sdr_listen.sh captures")
if hops[-1].end_s < 20:
    print(f"  WARNING: {hops[-1].end_s:.1f} s leaves little room to start the "
          f"receiver; raise CYCLES")
PY

cat <<'WARN'

  SAFETY: closed, conducted path only. Do not connect an antenna.
          Satellite downlink spectrum -- never radiate this.

WARN

echo "starting transmitter (Ctrl-C to stop; outputs mute on exit)"
exec "$ADF" hop \
    --seed          "$SEED" \
    --start-ghz     "$START_GHZ" \
    --stop-ghz      "$STOP_GHZ" \
    --points        "$POINTS" \
    --min-hop-ms    "$HOP_MS" \
    --jitter        "$JITTER" \
    --period-cycles "$PERIOD_CYCLES" \
    --cycles        "$CYCLES" \
    --channel       "$CHANNEL" \
    --power         "$POWER" \
    --spi-hz        "$SPI_HZ" \
    --enable-rf
