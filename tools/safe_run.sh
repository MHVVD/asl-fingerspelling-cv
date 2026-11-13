#!/usr/bin/env bash
# Run a heavy job gently on a laptop that powers off when it overheats (journal 18).
#
#   tools/safe_run.sh NAME [--cpus N] -- command args...
#   tools/safe_run.sh bench_tune -- .venv/bin/python -m aslcv.benchmark score --split tune
#
# * CPU capped at N threads' worth (default 2 of 8) with a user systemd scope, low priority, and the thread pools of
#   NumPy/BLAS/TFLite limited to N, so the CPU never runs flat out for hours.
# * Memory capped at MEM (default 6G, no more than 1G of swap): a runaway job is stopped on its own instead of pushing
#   the whole laptop into swap until it freezes.
# * Heat guard: the job's scope is frozen (systemctl --user freeze, which pauses it and any child processes) when the
#   CPU package passes PAUSE_C, and thawed again below RESUME_C.
# * Detached (setsid + nohup): the job survives the terminal or the Claude session ending.
# * Temperature log, one line every 10 s, flushed to disk: outputs/logs/temps.csv
#   (time, package temp, hottest zone, 1-min load, job state). After an unexpected power-off, its last lines show
#   whether heat was the cause.
# * Output in outputs/logs/NAME.log; outputs/logs/NAME.done holds the exit code when it finishes.
set -euo pipefail
cd "$(dirname "$0")/.."

NAME=${1:?usage: tools/safe_run.sh NAME [--cpus N] -- command...}; shift
CPUS=2
while [[ $# -gt 0 && $1 != -- ]]; do
  case $1 in --cpus) CPUS=$2; shift 2 ;; *) echo "unknown option $1" >&2; exit 2 ;; esac
done
[[ ${1:-} == -- ]] && shift
(( $# > 0 )) || { echo "no command given" >&2; exit 2; }

MEM=${MEM:-6G}
PAUSE_C=${PAUSE_C:-88}
RESUME_C=${RESUME_C:-75}
LOGS=outputs/logs
mkdir -p "$LOGS"
rm -f "$LOGS/$NAME.done"

pkg_temp() {     # CPU package temperature in whole degrees C (falls back to the hottest zone)
  local z t max=0
  for z in /sys/class/thermal/thermal_zone*; do
    t=$(( $(cat "$z/temp" 2>/dev/null || echo 0) / 1000 ))
    [[ $(cat "$z/type" 2>/dev/null) == x86_pkg_temp ]] && { echo "$t"; return; }
    (( t > max )) && max=$t
  done
  echo "$max"
}

job() {
  export OMP_NUM_THREADS=$CPUS OPENBLAS_NUM_THREADS=$CPUS MKL_NUM_THREADS=$CPUS NUMEXPR_NUM_THREADS=$CPUS \
         TF_NUM_INTRAOP_THREADS=$CPUS TF_NUM_INTEROP_THREADS=1 PYTHONPATH=src
  local unit="aslcv-$NAME-$$"
  systemd-run --user --scope --quiet --unit="$unit" -p CPUQuota=$((CPUS * 100))% \
    -p MemoryMax="$MEM" -p MemorySwapMax=1G -- nice -n 10 "$@" \
    > "$LOGS/$NAME.log" 2>&1 &
  local pid=$! state=running t
  [[ -f $LOGS/temps.csv ]] || echo "time,pkg_c,max_zone_c,load1,job,state" > "$LOGS/temps.csv"
  while kill -0 "$pid" 2>/dev/null; do
    t=$(pkg_temp)
    if [[ $state == running ]] && (( t >= PAUSE_C )); then
      systemctl --user freeze "$unit.scope" 2>/dev/null && state=paused
    elif [[ $state == paused ]] && (( t <= RESUME_C )); then
      systemctl --user thaw "$unit.scope" 2>/dev/null && state=running
    fi
    echo "$(date +%FT%T),$t,$(( $(cat /sys/class/thermal/thermal_zone*/temp | sort -n | tail -1) / 1000 )),$(cut -d' ' -f1 /proc/loadavg),$NAME,$state" >> "$LOGS/temps.csv"
    sync -d "$LOGS/temps.csv" 2>/dev/null || sync
    sleep 10
  done
  wait "$pid"; echo $? > "$LOGS/$NAME.done"
}

export -f job pkg_temp
export NAME CPUS MEM PAUSE_C RESUME_C LOGS
setsid nohup bash -c 'job "$@"' _ "$@" > /dev/null 2>&1 < /dev/null &
echo "started $NAME (cap ${CPUS} CPUs, ${MEM} memory, pause at ${PAUSE_C}°C): log $LOGS/$NAME.log, temps $LOGS/temps.csv"
