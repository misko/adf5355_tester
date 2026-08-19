#!/usr/bin/env bash
#
# RECEIVE side of a seeded frequency-hop calibration.
#
# Listens ONCE at a fixed tuning -- the whole hop span sits inside the
# receiver's instantaneous bandwidth, so one tuning hears every point -- then
# regenerates the transmitter's schedule from the shared seed, aligns it to the
# capture, and reports each point's frequency error.
#
# Nothing is inferred from what the capture looks like: identity comes from the
# seed, so the only unknown is where the pattern started. The decoder prints a
# comb sharpness and an epoch sigma with every run, and shouts if either is
# poor, because a confident wrong answer is the failure that matters.
#
# Start adf5355_rf_hop.sh first, then run this. The schedule block below must
# be IDENTICAL to that script's -- those numbers are the entire protocol
# between the two ends.
#
# Override from the environment, e.g.
#     SECONDS_LISTEN=20 FS=3e6 ./sdr_listen.sh
#
# Any extra arguments are handed straight to the decoder, e.g.
#     ./sdr_listen.sh --capture-out run.iq --json
#
set -euo pipefail

# ---- schedule: EVERY LINE HERE MUST MATCH adf5355_rf_hop.sh ---------------
SEED="${SEED:-0xC0FFEE}"              # the whole protocol between the two ends
START_GHZ="${START_GHZ:-11.0}"        # first frequency point
STOP_GHZ="${STOP_GHZ:-11.00171}"      # last; span must fit the receiver
POINTS="${POINTS:-20}"                # frequency points, 90 kHz apart
HOP_MS="${HOP_MS:-10}"                # dwell per hop; precision tracks this
JITTER="${JITTER:-0}"                 # 0 = fixed dwell (measured no worse)
PERIOD_CYCLES="${PERIOD_CYCLES:-1}"   # permutations before the pattern repeats
# ---- receive chain --------------------------------------------------------
LO_HZ="${LO_HZ:-9.75e9}"              # NOMINAL LNB LO. 13 V, no 22 kHz tone
                                      # selects low band = 9.75 GHz.
LO_ERROR_HZ="${LO_ERROR_HZ:-94000}"   # measured LNB LO error, used only to
                                      # centre the receiver on the comb
SECONDS_LISTEN="${SECONDS_LISTEN:-2}" # 2 s = 10 periods. Measured: every
                                      # metric is flat from 1 period to 160,
                                      # so longer only costs decode time.
FS="${FS:-2.5e6}"                     # 2.5 MS/s gives about 2 MHz usable
FRAME="${FRAME:-512}"                 # must be well under one dwell
GAIN="${GAIN:-40}"
URI="${URI:-ip:192.168.2.1}"
# ---------------------------------------------------------------------------

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DECODER="${DECODER:-$REPO/tools/hop_decode.py}"
UTILS="${UTILS:-$HOME/pluto-plus-utils}"   # only for its pyadi-iio + numpy
[ -f "$DECODER" ] || { echo "error: decoder not found at $DECODER" >&2; exit 1; }

# The decoder needs numpy, and needs pyadi-iio only when it opens the radio.
if [ -n "${PYTHON_RUN:-}" ]; then
    read -r -a RUNNER <<< "$PYTHON_RUN"
elif [ -d "$UTILS" ]; then
    RUNNER=(uv run --project "$UTILS" python)
else
    RUNNER=(python3)
fi

# Centre on the middle of the span, shifted by the known LNB LO error so the
# whole comb stays inside the passband. The decoder derives the same number;
# computing it here as well is what makes the tuning visible before anything
# is captured.
IF_HZ="$(python3 - "$START_GHZ" "$STOP_GHZ" "$LO_HZ" "$LO_ERROR_HZ" <<'PY'
import sys
start, stop, lo, err = (float(sys.argv[1])*1e9, float(sys.argv[2])*1e9,
                        float(sys.argv[3]), float(sys.argv[4]))
print(int(round((start + stop) / 2 - lo - err)))
PY
)"

REPO="$REPO" python3 - "$SEED" "$START_GHZ" "$STOP_GHZ" "$POINTS" "$HOP_MS" \
        "$JITTER" "$PERIOD_CYCLES" "$FS" "$FRAME" "$SECONDS_LISTEN" "$IF_HZ" <<'PY'
import os, sys
sys.path.insert(0, os.environ["REPO"])
from adf5355.hopper import make_schedule, period_duration, plan_frequencies

seed = int(sys.argv[1], 0)
start, stop = float(sys.argv[2]) * 1e9, float(sys.argv[3]) * 1e9
points, hop_ms, jitter = int(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
period_cycles, fs, frame = int(sys.argv[7]), float(sys.argv[8]), int(sys.argv[9])
secs, if_hz = float(sys.argv[10]), float(sys.argv[11])

freqs = plan_frequencies(round(start), round(stop), points)
hops = make_schedule(seed, freqs, hop_ms / 1e3, period_cycles + 1, jitter,
                     period_cycles)
period = period_duration(hops, points, period_cycles)
span = stop - start
frame_s = frame / fs

print(f"  schedule     : seed 0x{seed:X}, {points} points "
      f"{start/1e9:.6f}-{stop/1e9:.6f} GHz "
      f"(spacing {span/(points-1)/1e3:.1f} kHz, span {span/1e6:.3f} MHz)")
print(f"                 dwell {hop_ms:g} ms, jitter {jitter:g}, "
      f"period {period*1e3:.1f} ms  <-- must match the transmitter exactly")
print(f"  tuning       : {if_hz/1e6:.3f} MHz at {fs/1e6:g} MS/s "
      f"({fs*0.8/1e6:.2f} MHz usable)")
print(f"  listening    : {secs:g} s = {secs/period:.1f} periods, "
      f"{int(secs/(hop_ms/1e3))} hops")
print(f"  frame        : {frame} = {frame_s*1e3:.3f} ms "
      f"({hop_ms/1e3/frame_s:.1f} frames per dwell, "
      f"{fs/frame/1e3:.2f} kHz per bin)")

if span > fs * 0.8:
    print(f"  WARNING: span {span/1e6:.3f} MHz exceeds the usable bandwidth "
          f"{fs*0.8/1e6:.3f} MHz;\n"
          f"           raise FS or narrow the span, or the outer points are "
          f"simply not there")
if hop_ms / 1e3 / frame_s < 4:
    print(f"  WARNING: a dwell spans only {hop_ms/1e3/frame_s:.1f} frames; "
          f"lower FRAME (want 20+)")
if frame_s > hop_ms / 1e3:
    print(f"  WARNING: a frame is longer than a dwell; every frame straddles "
          f"hops and nothing will align")
if secs < period:
    print(f"  WARNING: {secs:g} s is under ONE period ({period:.2f} s). Only "
          f"{secs/period*points:.0f} of {points} points are transmitted in that "
          f"time,\n           so the rest cannot be observed at all. Measured at "
          f"half a period: 3/20\n           points, 5.5 sigma, and an answer "
          f"90 kHz wrong -- the decoder refuses it.")
elif secs < 2 * period:
    print(f"  NOTE: {secs:g} s is {secs/period:.1f} periods. One period is the "
          f"hard floor and decodes\n        fine; two or more gives margin if a "
          f"hop is lost.")
if fs / frame > span / (points - 1) / 2:
    print(f"  WARNING: bin width {fs/frame/1e3:.1f} kHz is coarse against the "
          f"{span/(points-1)/1e3:.1f} kHz spacing; raise FRAME")
PY
echo

exec "${RUNNER[@]}" "$DECODER" \
    --seed          "$SEED" \
    --start-ghz     "$START_GHZ" \
    --stop-ghz      "$STOP_GHZ" \
    --points        "$POINTS" \
    --min-hop-ms    "$HOP_MS" \
    --jitter        "$JITTER" \
    --period-cycles "$PERIOD_CYCLES" \
    --lo-hz         "$LO_HZ" \
    --lo-error-hz   "$LO_ERROR_HZ" \
    --fs            "$FS" \
    --frame         "$FRAME" \
    --seconds       "$SECONDS_LISTEN" \
    --gain          "$GAIN" \
    --uri           "$URI" \
    "$@"
