#!/usr/bin/env bash
#
# RECEIVE side of a LEVER-ARM calibration: measure d_rx and d_lnb separately.
#
#   ####################################################################
#   #  CLOSED, CONDUCTED PATHS ONLY.  NEVER RADIATE.                   #
#   #  10.7-11.9 GHz is satellite downlink spectrum. Coax into an      #
#   #  attenuator and a load, or a shielded enclosure. No antenna.     #
#   ####################################################################
#
# Visits every cluster in turn, over and over, retuning between them, and
# writes one record per capture. Then fits all of them at once:
#
#     Df(f_IF) = -d_rx * f_IF  -  d_lnb * f_LO_nom  -  b(rx_lo) - drift(t)
#
# Three things about the order it works in carry the whole measurement:
#
#   * SWEEPS. One capture of every cluster, then again. d_rx is then fitted
#     with a free offset per sweep, which absorbs the LNB's LO error and its
#     drift whatever shape they have, and leaves d_rx untouched because it is
#     the only term that varies with frequency.
#   * A FRESH RANDOM CLUSTER ORDER every sweep, so drift never lands on the
#     clusters in a fixed pattern.
#   * A RANDOM TUNING DITHER on every capture. The receiver's tuning bias --
#     362 Hz peak to peak across eight tunings on this hardware -- is the
#     dominant error of the whole measurement. Dithering makes it an
#     independent draw per capture so it averages down as sqrt(captures).
#     WITHOUT IT, MORE CAPTURES BUY NOTHING.
#
# Start adf5355_rf_lever.sh first and leave it running. The schedule block
# below must be IDENTICAL to that script's -- those numbers are the entire
# protocol between the two ends.
#
# Nothing here opens the radio unless OPEN_RADIO=1. The default prints the
# plan; SYNTHETIC=1 runs the whole pipeline against captures built in memory
# from a known answer, which is how to check the chain end to end with no
# hardware at all.
#
#     ./sdr_lever.sh                       # print the plan, touch nothing
#     SYNTHETIC=1 ./sdr_lever.sh           # whole pipeline offline, known answer
#     OPEN_RADIO=1 ./sdr_lever.sh          # the real measurement
#
set -euo pipefail

# ---- schedule: EVERY LINE HERE MUST MATCH adf5355_rf_lever.sh -------------
SEED="${SEED:-0xC0FFEE}"
LOW_GHZ="${LOW_GHZ:-10.70}"
HIGH_GHZ="${HIGH_GHZ:-11.90}"
CLUSTERS="${CLUSTERS:-4}"
CLUSTER_POINTS="${CLUSTER_POINTS:-6}"
SPAN_KHZ="${SPAN_KHZ:-720}"
HOP_MS="${HOP_MS:-10}"
BLOCK="${BLOCK:-3}"
BAND_EXTRA_MS="${BAND_EXTRA_MS:-5}"
JITTER="${JITTER:-0}"
PERIOD_CYCLES="${PERIOD_CYCLES:-1}"
# ---- receive chain --------------------------------------------------------
LO_HZ="${LO_HZ:-9.75e9}"              # NOMINAL LNB LO. 13 V, no 22 kHz tone
LO_ERROR_HZ="${LO_ERROR_HZ:-94000}"   # only used to centre the receiver
D_RX_GUESS="${D_RX_GUESS:-8.94e-6}"   # ditto; cannot bias the answer
FS="${FS:-2.5e6}"                     # 2.5 MS/s gives about +/-1 MHz usable
SECONDS_LISTEN="${SECONDS_LISTEN:-3}" # per capture. Two periods is the floor.
                                      # Longer captures do NOT tighten the
                                      # answer -- the tuning bias does not
                                      # care how long you listen. More
                                      # CAPTURES do. Prefer more sweeps.
FRAME="${FRAME:-512}"
GAIN="${GAIN:-40}"
URI="${URI:-ip:192.168.2.1}"
# ---- the run --------------------------------------------------------------
SWEEPS="${SWEEPS:-25}"                # one capture of each cluster per sweep;
                                      # precision goes as sqrt(SWEEPS)
DITHER_KHZ="${DITHER_KHZ:-450}"       # tuning dither half-range. The single
                                      # most important number here.
VISIT_SEED="${VISIT_SEED:-1234567}"   # makes the whole run reproducible
OUT="${OUT:-lever-run.jsonl}"
WORKDIR="${WORKDIR:-/tmp}"
# ---------------------------------------------------------------------------

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
RUNNER_PY="${RUNNER_PY:-$REPO/tools/lever_run.py}"
FIT_PY="${FIT_PY:-$REPO/tools/lever_fit.py}"
UTILS="${UTILS:-$HOME/pluto-plus-utils}"   # only for its pyadi-iio + numpy
[ -f "$RUNNER_PY" ] || { echo "error: $RUNNER_PY not found" >&2; exit 1; }

if [ -n "${PYTHON_RUN:-}" ]; then
    read -r -a RUNNER <<< "$PYTHON_RUN"
elif [ -d "$UTILS" ]; then
    RUNNER=(uv run --project "$UTILS" python)
else
    RUNNER=(python3)
fi

MODE=()
if [ "${OPEN_RADIO:-0}" = "1" ]; then
    MODE+=(--open-radio)
    cat <<'WARN'

  SAFETY: closed, conducted path only. Do not connect an antenna.
          Satellite downlink spectrum -- never radiate this.

WARN
elif [ "${SYNTHETIC:-0}" = "1" ]; then
    MODE+=(--synthetic)
fi
[ "${FIT:-1}" = "1" ] && MODE+=(--fit)

exec "${RUNNER[@]}" "$RUNNER_PY" \
    --seed            "$SEED" \
    --low-ghz         "$LOW_GHZ" \
    --high-ghz        "$HIGH_GHZ" \
    --clusters        "$CLUSTERS" \
    --cluster-points  "$CLUSTER_POINTS" \
    --span-khz        "$SPAN_KHZ" \
    --min-hop-ms      "$HOP_MS" \
    --block           "$BLOCK" \
    --band-extra-ms   "$BAND_EXTRA_MS" \
    --jitter          "$JITTER" \
    --period-cycles   "$PERIOD_CYCLES" \
    --lo-hz           "$LO_HZ" \
    --lo-error-hz     "$LO_ERROR_HZ" \
    --d-rx-guess      "$D_RX_GUESS" \
    --fs              "$FS" \
    --seconds         "$SECONDS_LISTEN" \
    --frame           "$FRAME" \
    --gain            "$GAIN" \
    --uri             "$URI" \
    --sweeps          "$SWEEPS" \
    --dither-khz      "$DITHER_KHZ" \
    --visit-seed      "$VISIT_SEED" \
    --out             "$OUT" \
    --workdir         "$WORKDIR" \
    "${MODE[@]}" "$@"

# After the run:   tools/lever_fit.py "$OUT"      (re-fit without re-capturing)
