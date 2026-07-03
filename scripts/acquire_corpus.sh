#!/usr/bin/env bash
# Acquire a pinned, reproducible text corpus for clob's first real run.
#
# Writes (under data/corpus/):
#   train.txt    (~90% head)  + train.txt.sha256
#   holdout.txt  (~10% tail)  + holdout.txt.sha256
#
# Idempotent: if the artifacts and their .sha256 sidecars exist and verify,
# it re-verifies and exits without re-downloading. Run with FORCE=1 (or delete
# the files) to re-acquire.
#
# Corpus choice (CHANGE ME via env): defaults to a single public-domain prose
# book from Project Gutenberg — the simplest thing to pin and the cleanest
# license (public domain, no attribution). Swap CORPUS_URL/CORPUS_NAME for a
# different text, or point at a Wiki-40B slice for broader-domain coverage
# (note: Wikipedia text is CC-BY-SA — attribution required).
#
# NOTE: PLAN.md's original example ("A Brief History of Time") is NOT public
# domain (Hawking, 1988, in copyright). The default below is genuinely PD.

set -euo pipefail

# --- corpus configuration -------------------------------------------------
CORPUS_NAME="${CORPUS_NAME:-moby-dick-pg2701}"
CORPUS_URL="${CORPUS_URL:-https://www.gutenberg.org/files/2701/2701-0.txt}"
HOLDOUT_FRACTION="${HOLDOUT_FRACTION:-0.10}"
# --------------------------------------------------------------------------

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/data/corpus"
mkdir -p "$OUT"

TRAIN="$OUT/train.txt"
HOLDOUT="$OUT/holdout.txt"
RAW="$OUT/.raw_${CORPUS_NAME}.txt"

verify() {
    local f="$1"
    [ -f "$f" ] && [ -f "$f.sha256" ] || return 1
    if (cd "$OUT" && sha256sum -c "$(basename "$f").sha256") >/dev/null 2>&1; then
        return 0
    fi
    echo "checksum MISMATCH for $f — refusing to proceed; delete to re-acquire" >&2
    exit 1
}

if [ "${FORCE:-0}" != "1" ] && verify "$TRAIN" && verify "$HOLDOUT"; then
    echo "corpus present and verified: $TRAIN, $HOLDOUT"
    exit 0
fi

echo "acquiring corpus '$CORPUS_NAME' from $CORPUS_URL"
if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$CORPUS_URL" -o "$RAW"
elif command -v wget >/dev/null 2>&1; then
    wget -qO "$RAW" "$CORPUS_URL"
else
    echo "need curl or wget to download the corpus" >&2
    exit 1
fi

# Strip Project Gutenberg header/footer boilerplate if present, so the corpus
# is prose and the holdout tail isn't the license footer.
if grep -q '\*\*\* *START OF' "$RAW"; then
    awk '/\*\*\* *START OF/{f=1; next} /\*\*\* *END OF/{f=0} f' "$RAW" > "$RAW.body"
    [ -s "$RAW.body" ] && mv "$RAW.body" "$RAW" || rm -f "$RAW.body"
fi

# The default Moby-Dick file contains a table of contents whose chapter
# headings repeat later in the book. For split enforcement that looks like
# train/holdout leakage, so for this corpus start at the real first chapter.
if [ "$CORPUS_NAME" = "moby-dick-pg2701" ]; then
    awk '
        /^CHAPTER 1\. Loomings\.$/ { seen += 1 }
        seen >= 2 { print }
    ' "$RAW" > "$RAW.body"
    [ -s "$RAW.body" ] && mv "$RAW.body" "$RAW" || rm -f "$RAW.body"
fi

# Split: last HOLDOUT_FRACTION of lines become holdout, the rest train.
total=$(wc -l < "$RAW")
holdout_lines=$(awk -v t="$total" -v f="$HOLDOUT_FRACTION" 'BEGIN{printf "%d", t*f}')
train_lines=$((total - holdout_lines))
head -n "$train_lines" "$RAW" > "$TRAIN"
tail -n "$holdout_lines" "$RAW" > "$HOLDOUT"

# Pin checksums (trust-on-first-use). Re-runs verify against these.
(cd "$OUT" && sha256sum "$(basename "$TRAIN")" > "$(basename "$TRAIN").sha256")
(cd "$OUT" && sha256sum "$(basename "$HOLDOUT")" > "$(basename "$HOLDOUT").sha256")

echo "wrote $TRAIN ($train_lines lines) and $HOLDOUT ($holdout_lines lines)"
echo "checksums pinned in $OUT/*.sha256"
echo "next: ./scripts/tokenize_corpus.sh"
