#!/usr/bin/env bash
#
# RECEIVE side of a frequency-ladder calibration.
#
# Listens ONCE at a fixed tuning for long enough to hear at least one complete
# ladder cycle, then decodes it: each burst is identified purely by its
# duration against the published schedule, mapped back to that rung's
# frequency, and its frequency error recorded. The transmitter is never
# controlled or queried from here -- only observed.
#
# Start adf5355_rf_ladder.sh first, then run this. Keep the ladder settings
# below identical to that script's, since they are the "published parameters"
# the decoder works from.
#
# Override from the environment, e.g.
#     SECONDS_LISTEN=40 FS=3e6 ./sdr_listen.sh
#
set -euo pipefail

# ---- ladder definition (must match adf5355_rf_ladder.sh) ------------------
START_GHZ="${START_GHZ:-11.0}"
STOP_GHZ="${STOP_GHZ:-11.00171}"
STEPS="${STEPS:-20}"
TOTAL_S="${TOTAL_S:-4.2}"
# ---- receive chain --------------------------------------------------------
LO_HZ="${LO_HZ:-9.75e9}"              # NOMINAL LNB LO. 13 V, no 22 kHz tone
                                      # selects low band = 9.75 GHz.
LO_ERROR_HZ="${LO_ERROR_HZ:-94000}"   # measured LNB LO error, used only to
                                      # centre the receiver on the comb
SECONDS_LISTEN="${SECONDS_LISTEN:-20}"
FS="${FS:-2.5e6}"                     # 2.5 MS/s gives about 2 MHz usable
FRAME="${FRAME:-2048}"                # must be well under the shortest burst
GAIN="${GAIN:-40}"
URI="${URI:-ip:192.168.2.1}"

UTILS="${UTILS:-$HOME/pluto-plus-utils}"
LISTENER="${LISTENER:-$HOME/adf5355_tester/tools/freq_ladder_listen.py}"
[ -f "$LISTENER" ] || { echo "error: listener not found at $LISTENER" >&2; exit 1; }
[ -d "$UTILS" ]    || { echo "error: pluto-plus-utils not found at $UTILS" >&2; exit 1; }

# Centre on the middle of the comb, shifted by the known LNB LO error so the
# whole span stays inside the passband.
IF_HZ="$(python3 - "$START_GHZ" "$STOP_GHZ" "$LO_HZ" "$LO_ERROR_HZ" <<'PY'
import sys
start, stop, lo, err = (float(sys.argv[1])*1e9, float(sys.argv[2])*1e9,
                        float(sys.argv[3]), float(sys.argv[4]))
print(int(round((start + stop) / 2 - lo - err)))
PY
)"

python3 - "$START_GHZ" "$STOP_GHZ" "$STEPS" "$TOTAL_S" "$FS" "$FRAME" \
          "$SECONDS_LISTEN" "$IF_HZ" <<'PY'
import sys
start, stop, steps, total, fs, frame, secs, if_hz = (
    float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]),
    float(sys.argv[5]), int(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8]))
u = total / (steps * (steps + 1))
span = (stop - start) * 1e9
print(f"  tuning       : {if_hz/1e6:.3f} MHz at {fs/1e6:g} MS/s "
      f"({fs*0.8/1e6:.1f} MHz usable, ladder spans {span/1e6:.3f} MHz)")
print(f"  listening    : {secs:g} s = {secs/total:.1f} ladder cycles")
print(f"  frame        : {frame} = {frame/fs*1e3:.2f} ms "
      f"({u/(frame/fs):.1f} frames in the shortest burst)")
if secs < total * 1.2:
    print(f"  WARNING: {secs:g} s is under one full {total:g} s cycle plus the "
          f"longest burst.\n"
          f"           Raise SECONDS_LISTEN to at least "
          f"{total + u*steps:.1f} s or no rung is guaranteed to arrive whole.")
if span > fs * 0.8:
    print("  WARNING: ladder span exceeds the usable bandwidth; raise FS")
if u / (frame / fs) < 4:
    print("  WARNING: shortest burst spans under 4 frames; lower FRAME")
PY
echo

cd "$UTILS"
exec uv run python "$LISTENER" \
    --if-hz "$IF_HZ" \
    --seconds "$SECONDS_LISTEN" \
    --fs "$FS" \
    --frame-size "$FRAME" \
    --gain "$GAIN" \
    --uri "$URI" \
    --rung-start-hz "$(python3 -c "print(float('$START_GHZ')*1e9)")" \
    --rung-stop-hz  "$(python3 -c "print(float('$STOP_GHZ')*1e9)")" \
    --rung-count "$STEPS" \
    --total-seconds "$TOTAL_S" \
    --lo-hz "$LO_HZ" \
    "$@"
