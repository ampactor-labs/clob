#!/usr/bin/env bash
# Scaffold a new experiment directory under experiments/YYYY-MM-DD_<label>/.
# Pre-populates manifest.toml, run.sh, notes.md.
#
# Usage: ./scripts/new_experiment.sh <label>

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <label>" >&2
    exit 2
fi

LABEL="$1"
DATE="$(date -u +%Y-%m-%d)"
DIR="experiments/${DATE}_${LABEL}"

if [ -e "$DIR" ]; then
    echo "experiments dir already exists: $DIR" >&2
    exit 2
fi

mkdir -p "$DIR"

COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
KERNEL_VERSION="$(grep '^version' Cargo.toml | head -1 | cut -d'"' -f2)"

cat > "$DIR/manifest.toml" <<EOF
# Experiment manifest (top-level). Per-artifact sidecars live next to
# each artifact and record its inputs' sha256 — this file is the
# overarching experiment record.

label = "${LABEL}"
date = "${DATE}"
kernel_version = "${KERNEL_VERSION}"
git_commit = "${COMMIT}"

[run]
# Edit run.sh and fill in seed, config, corpus, etc. This section is a
# free-form free-text summary of what the experiment does. Keep it
# short; the truth is in run.sh.
summary = "TODO: describe the experiment in one sentence."

[expectations]
# What numbers do you expect? J/nat, mean_nll, modules-crystallized, etc.
# If you're running an ablation, say which cell you expect to win.
notes = "TODO"
EOF

cat > "$DIR/run.sh" <<EOF
#!/usr/bin/env bash
# Experiment: ${LABEL}
# Generated: ${DATE}, commit ${COMMIT}
#
# Rerun with: ./experiments/${DATE}_${LABEL}/run.sh
# Re-run must produce byte-identical artifacts given the same seed.

set -euo pipefail

: "\${SEED:=1}"
ROOT="\$(dirname "\$0")"

clob_bin="./target/release/clob"
if [ ! -x "\$clob_bin" ]; then
    cargo build --release
fi

# TODO: fill in the pipeline.
# Example:
#   "\$clob_bin" synth --seed "\$SEED" --output "\$ROOT/seed.clob" --config small
#   "\$clob_bin" calibrate-confidence --seed "\$SEED" \\
#       --model "\$ROOT/seed.clob" \\
#       --corpus data/corpus/train.txt \\
#       --output "\$ROOT/conf.bin"
#   ...

echo "Experiment ${LABEL}: edit \$ROOT/run.sh and fill in the pipeline."
exit 1
EOF

cat > "$DIR/notes.md" <<EOF
# ${LABEL}

**Date:** ${DATE}
**Commit:** ${COMMIT}

## Question

What is this experiment testing?

## Expectations

What do you expect to see?

## Results

Fill in after running.

## Conclusions

What does the result mean? Does it support or falsify any of the Part V
bets in ARCHITECTURE.md?
EOF

chmod +x "$DIR/run.sh"
echo "scaffolded $DIR"
echo "next steps:"
echo "  1. edit $DIR/run.sh and fill in the pipeline"
echo "  2. edit $DIR/manifest.toml expectations"
echo "  3. SEED=1 $DIR/run.sh"
