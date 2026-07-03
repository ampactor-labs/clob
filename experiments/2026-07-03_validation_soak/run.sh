#!/usr/bin/env bash
# Long validation soak for Path B gradient/BPTT groundwork.
#
# Tunables:
#   DURATION_HOURS=6   wall-clock duration
#   SMOKE_EVERY=4      run SMOKE=1 first_run.sh every N iterations

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EXP="$ROOT/experiments/2026-07-03_validation_soak"
LOG="$EXP/soak.log"

: "${DURATION_HOURS:=6}"
: "${SMOKE_EVERY:=4}"

exec > >(tee -a "$LOG") 2>&1

cd "$ROOT"

echo "== validation soak =="
echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "git_commit=$(git rev-parse HEAD)"
echo "duration_hours=$DURATION_HOURS"
echo "smoke_every=$SMOKE_EVERY"
echo

deadline=$((SECONDS + DURATION_HOURS * 3600))
iter=0

while [ "$SECONDS" -lt "$deadline" ]; do
    iter=$((iter + 1))
    echo
    echo "==== iteration $iter start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="

    echo "-- cargo test --release learn::"
    cargo test --release learn::

    echo "-- cargo test --release"
    cargo test --release

    if [ "$SMOKE_EVERY" -gt 0 ] && [ $((iter % SMOKE_EVERY)) -eq 0 ]; then
        echo "-- SMOKE=1 first_run.sh"
        EXPERIMENT_NAME="validation_soak_smoke_${iter}" \
            SMOKE=1 \
            NICE=1 \
            MAX_TOKENS=1500 \
            EPOCHS=1 \
            ./scripts/first_run.sh
    fi

    echo "==== iteration $iter done $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
done

echo
echo "== validation soak done =="
echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "iterations=$iter"
