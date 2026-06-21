#!/usr/bin/env bash
# Ground-truth-first diagnostic run.
#
# Streams the planted-pattern synthetic corpus (data/synthetic/diagnostic.txt)
# through the full loop and surfaces the metrics. Run this BEFORE the real
# corpus: the synthetic corpus has *known* structure (documented in
# data/synthetic/expected.md), so it tells you whether the loop finds the
# structure it is supposed to find before you face the noise of real text.
#
# On a random-weight synth model the signals are flat by construction — the
# value here is the harness + a baseline. Re-run after training to watch the
# numbers move (and to check the modules against expected.md).
#
# Usage: [SEED=1] [CONFIG=small] [WORK=/path] ./scripts/diagnostic_run.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${SEED:=1}"
: "${CONFIG:=small}"
CORPUS="$ROOT/data/synthetic/diagnostic.txt"
WORK="${WORK:-$(mktemp -d -t clob_diag.XXXXXX)}"
mkdir -p "$WORK"

CLOB="$ROOT/target/release/clob"
if [ ! -x "$CLOB" ]; then
    echo "building clob (release)..."
    (cd "$ROOT" && cargo build --release)
fi

echo "== diagnostic run (seed=$SEED config=$CONFIG) =="
echo "corpus: $CORPUS"
echo "work:   $WORK"
echo

# Note: only `synth` is seeded — `ingest` and `crystal` derive no entropy of
# their own. Keep it that way unless those subcommands gain a --seed.
echo "[1/4] synth seed model"
"$CLOB" synth --seed "$SEED" --config "$CONFIG" --output "$WORK/seed.clob"

echo "[2/4] ingest (record novel episodes + metrics)"
"$CLOB" ingest --model "$WORK/seed.clob" \
    --input "$CORPUS" --memory-dir "$WORK/episodes" \
    --metrics-out "$WORK/metrics.jsonl" --metrics-window 200 --max-tokens 0

echo "[3/4] crystal (drain episodes into modules)"
"$CLOB" crystal --model "$WORK/seed.clob" \
    --memory-dir "$WORK/episodes" --modules-dir "$WORK/modules"

echo "[4/4] metrics dashboard"
"$CLOB" metrics --path "$WORK/metrics.jsonl"

echo
echo "modules crystallized:"
ls -1 "$WORK/modules" 2>/dev/null || echo "  (none)"
echo
echo "Compare against data/synthetic/expected.md."
echo "On random weights these are flat by construction — re-run after training."
echo "artifacts in: $WORK"
