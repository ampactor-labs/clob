#!/usr/bin/env bash
# Tokenize the acquired corpus into cached `.tokens` arrays.
#
# Steps (idempotent — skips a step if its output is newer than its inputs):
#   1. clob train-tokenizer --corpus train.txt   -> data/corpus/tokenizer.bin
#   2. clob encode train.txt                      -> data/corpus/train.tokens
#   3. clob encode holdout.txt                    -> data/corpus/holdout.tokens
#
# Requires scripts/acquire_corpus.sh to have run first. Then ingest with:
#   clob ingest --input data/corpus/train.tokens --tokenizer data/corpus/tokenizer.bin ...

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/data/corpus"
N_MERGES="${N_MERGES:-4000}"

TRAIN_TXT="$OUT/train.txt"
HOLDOUT_TXT="$OUT/holdout.txt"
TOKENIZER="$OUT/tokenizer.bin"

if [ ! -f "$TRAIN_TXT" ] || [ ! -f "$HOLDOUT_TXT" ]; then
    echo "corpus missing — run ./scripts/acquire_corpus.sh first" >&2
    exit 1
fi

CLOB="$ROOT/target/release/clob"
if [ ! -x "$CLOB" ]; then
    echo "building clob (release)..."
    (cd "$ROOT" && cargo build --release)
fi

newer() { [ "$1" -nt "$2" ]; }

if [ ! -f "$TOKENIZER" ] || newer "$TRAIN_TXT" "$TOKENIZER"; then
    echo "training BPE tokenizer ($N_MERGES merges) on $(basename "$TRAIN_TXT")"
    "$CLOB" train-tokenizer --corpus "$TRAIN_TXT" --output "$TOKENIZER" --n-merges "$N_MERGES"
else
    echo "tokenizer up to date: $TOKENIZER"
fi

for split in train holdout; do
    txt="$OUT/$split.txt"
    tok="$OUT/$split.tokens"
    if [ ! -f "$tok" ] || newer "$txt" "$tok" || newer "$TOKENIZER" "$tok"; then
        echo "encoding $split -> $(basename "$tok")"
        "$CLOB" encode --input "$txt" --output "$tok" --tokenizer "$TOKENIZER"
    else
        echo "$split tokens up to date: $tok"
    fi
done

echo "done. ingest with:"
echo "  $CLOB ingest --input $OUT/train.tokens --tokenizer $TOKENIZER ..."
