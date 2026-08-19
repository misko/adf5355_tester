#!/usr/bin/env bash
#
# TRANSMIT side of a LEVER-ARM calibration: several clusters, one schedule.
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
# WHY THIS EXISTS, and how it differs from adf5355_rf_hop.sh
# ----------------------------------------------------------
# One narrow cluster measures the receiver's total frequency offset beautifully
# and cannot take it apart. The offset is
#
#     Df(f_IF) = -d_rx * f_IF  -  d_lnb * f_LO_nom
#
# and across a single cluster the first term barely moves, so the answer is a
# blend of the SDR's clock error and the LNB's LO error. Spreading the SAME
# seeded schedule over clusters more than a gigahertz apart in IF makes the
# first term move by kilohertz, and the receiver's fit separates the two.
#
# The transmitter is free-running and knows nothing about the receiver. It hops
# over every cluster forever; a receiver tuned to any one of them sees a
# complete, decodable pattern for THAT cluster and needs none of the others.
#
# Run this first and leave it going for the whole receiver run, then run
# sdr_lever.sh in another shell.
#
# Override any of the settings below from the environment, e.g.
#     CLUSTERS=6 SWEEP_MINUTES=40 ./adf5355_rf_lever.sh
#
set -euo pipefail

# ---- schedule: EVERY LINE HERE MUST MATCH sdr_lever.sh --------------------
SEED="${SEED:-0xC0FFEE}"              # the whole protocol between the two ends
LOW_GHZ="${LOW_GHZ:-10.70}"           # lowest cluster centre
HIGH_GHZ="${HIGH_GHZ:-11.90}"         # highest; the gap IS the lever arm
CLUSTERS="${CLUSTERS:-4}"             # clusters spread across that range
CLUSTER_POINTS="${CLUSTER_POINTS:-6}"    # points per cluster
SPAN_KHZ="${SPAN_KHZ:-720}"          # how wide one cluster is. The points sit
                                      # on a Golomb ruler inside it, never on a
                                      # regular grid: a regular comb has an
                                      # alias one spacing over that can and does
                                      # capture the receiver.  Must fit the
                                      # passband WITH room for the dither.
HOP_MS="${HOP_MS:-10}"                # dwell per hop
BLOCK="${BLOCK:-3}"                   # consecutive dwells per cluster visit;
                                      # must divide CLUSTER_POINTS
BAND_EXTRA_MS="${BAND_EXTRA_MS:-5}"   # extra dwell on a band-changing hop, to
                                      # pay for the VCO band search
JITTER="${JITTER:-0}"
PERIOD_CYCLES="${PERIOD_CYCLES:-1}"
LO_HZ="${LO_HZ:-9.75e9}"              # nominal LNB LO, for the printout only
# ---- transmit only --------------------------------------------------------
CYCLES="${CYCLES:-8000}"              # must outlast the whole receiver run
POWER="${POWER:-0}"                   # 0 = -4 dBm, the lowest step
CHANNEL="${CHANNEL:-B}"               # B = 6.8-13.6 GHz doubler output (OB)
SPI_HZ="${SPI_HZ:-1000000}"
# ---------------------------------------------------------------------------

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ADF="${ADF:-adf5355}"
command -v "$ADF" >/dev/null 2>&1 || {
    echo "error: '$ADF' not on PATH. Install with:" >&2
    echo "         uv tool install --editable $REPO" >&2
    exit 1
}

cat <<'WARN'

  SAFETY: closed, conducted path only. Do not connect an antenna.
          Satellite downlink spectrum -- never radiate this.

WARN

echo "starting cluster transmitter (Ctrl-C to stop; outputs mute on exit)"
exec "$ADF" hop-lever \
    --seed           "$SEED" \
    --low-ghz        "$LOW_GHZ" \
    --high-ghz       "$HIGH_GHZ" \
    --clusters       "$CLUSTERS" \
    --cluster-points "$CLUSTER_POINTS" \
    --span-khz       "$SPAN_KHZ" \
    --min-hop-ms     "$HOP_MS" \
    --block          "$BLOCK" \
    --band-extra-ms  "$BAND_EXTRA_MS" \
    --jitter         "$JITTER" \
    --period-cycles  "$PERIOD_CYCLES" \
    --cycles         "$CYCLES" \
    --lo-hz          "$LO_HZ" \
    --channel        "$CHANNEL" \
    --power          "$POWER" \
    --spi-hz         "$SPI_HZ" \
    --enable-rf
