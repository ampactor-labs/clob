#!/usr/bin/env bash
# first_run.sh — the one-command, fire-and-forget overnight run.
#
# Does EVERYTHING end to end on the real corpus:
#   acquire → tokenize → synth → calibrate-confidence(+meta) → train-router
#   → ingest (crystallize, checkpointed) → crystal → bench-suite (ablation on
#   the held-out split) → metrics summary.
#
# Designed to be left running on an idle machine:
#   - re-execs itself under `nice`/`ionice` so it yields if anything else wakes
#   - logs everything to the experiment dir (safe under `nohup ... &`)
#   - `set -euo pipefail` + a clear FAILED/DONE banner with the failing step
#   - ingest is checkpointed, so a crash is resumable
#
# Launch (when you're done with other work):
#   nohup ./scripts/first_run.sh >/dev/null 2>&1 &
#   # ...or schedule it for tonight (atd is running):
#   echo 'cd '"$PWD"' && ./scripts/first_run.sh' | at 23:00
#
# Validate the whole chain fast first (seconds, no network, in-repo corpus):
#   SMOKE=1 ./scripts/first_run.sh
#
# Tunables (env): SEED, CONFIG (seed|small|tiny), N_MERGES, MAX_TOKENS (0=all),
#   EPOCHS, EXPERIMENT_NAME, NICE (1=idle prio, 0=normal), SMOKE (1=fast smoke).
#   Corpus overrides pass through to acquire_corpus.sh: CORPUS_NAME, CORPUS_URL.

set -euo pipefail

# --- idle priority: deprioritize self; children (clob) inherit --------------
# Best-effort and non-fatal: renice 0→19 and ionice idle class need no
# privileges for your own process on a modern kernel, but if either is denied
# we simply run at normal priority rather than aborting the overnight job.
if [ "${NICE:-1}" = "1" ]; then
    renice -n 19 -p "$$" >/dev/null 2>&1 || true
    ionice -c 3 -p "$$" >/dev/null 2>&1 || true
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SEED="${SEED:-1}"
SMOKE="${SMOKE:-0}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-first_run}"
DATE="$(date -u +%Y-%m-%d)"
EXP="$ROOT/experiments/${DATE}_${EXPERIMENT_NAME}"
mkdir -p "$EXP"
LOG="$EXP/run.log"

# Everything from here is tee'd to the run log (so a bare `&` still captures it).
exec > >(tee -a "$LOG") 2>&1

step() { echo; echo "==== [$(date -u +%H:%M:%S)] $* ===="; STEP="$*"; STEP_T=$SECONDS; }
done_step() { echo "---- done in $((SECONDS - STEP_T))s"; }
fail() { echo; echo "#### FAILED at: ${STEP:-startup} (after $((SECONDS))s). See $LOG"; exit 1; }
trap fail ERR

CLOB="$ROOT/target/release/clob"
if [ ! -x "$CLOB" ]; then
    step "build clob (release)"; cargo build --release; done_step
fi

# --- corpus/tokenizer selection (smoke vs real) -----------------------------
if [ "$SMOKE" = "1" ]; then
    CONFIG="${CONFIG:-small}"; MAX_TOKENS="${MAX_TOKENS:-1500}"; EPOCHS="${EPOCHS:-1}"
    TRAIN_TXT="$ROOT/tests/data/eval_corpus.txt"
    HOLDOUT_TXT="$TRAIN_TXT"
    INGEST_INPUT="$TRAIN_TXT"           # byte-level, text
    TOK_ARGS=()                          # no tokenizer ⇒ byte-level (vocab 260)
    CKPT_ARGS=()
    echo "== SMOKE run (in-repo corpus, byte-level, fast) =="
else
    CONFIG="${CONFIG:-seed}"; MAX_TOKENS="${MAX_TOKENS:-0}"; EPOCHS="${EPOCHS:-2}"
    N_MERGES="${N_MERGES:-3000}"
    # Pre-flight: BPE vocab must fit the model config, or the run dies later.
    case "$CONFIG" in tiny) CFG_VOCAB=256;; small) CFG_VOCAB=1024;; seed) CFG_VOCAB=4096;; *) CFG_VOCAB=0;; esac
    if [ "$CFG_VOCAB" -gt 0 ] && [ $((N_MERGES + 260)) -gt "$CFG_VOCAB" ]; then
        echo "FATAL: BPE vocab (~$((N_MERGES + 260))) exceeds model vocab ($CFG_VOCAB for config '$CONFIG')."
        echo "       Lower N_MERGES or raise CONFIG. Aborting before the long run."
        exit 2
    fi
    TOK="$ROOT/data/corpus/tokenizer.bin"
    TRAIN_TXT="$ROOT/data/corpus/train.txt"
    HOLDOUT_TXT="$ROOT/data/corpus/holdout.txt"
    INGEST_INPUT="$ROOT/data/corpus/train.tokens"   # fast pre-encoded cache
    TOK_ARGS=(--tokenizer "$TOK")
    CKPT_ARGS=(--checkpoint-every 50000 --checkpoint-dir "$EXP/checkpoints")

    step "acquire corpus"; N_MERGES="$N_MERGES" "$ROOT/scripts/acquire_corpus.sh"; done_step
    step "tokenize corpus (N_MERGES=$N_MERGES)"; N_MERGES="$N_MERGES" "$ROOT/scripts/tokenize_corpus.sh"; done_step
fi

echo
echo "experiment: $EXP"
echo "config=$CONFIG seed=$SEED max_tokens=$MAX_TOKENS epochs=$EPOCHS smoke=$SMOKE"

SEEDM="$EXP/seed.clob"
CONF="$EXP/conf.bin"; META="$EXP/meta.bin"; ROUTERS="$EXP/routers.bin"

step "synth seed model (--config $CONFIG)"
"$CLOB" synth --seed "$SEED" --config "$CONFIG" --output "$SEEDM"; done_step

step "calibrate-confidence (Head C + MetaCritic)"
"$CLOB" calibrate-confidence --seed "$SEED" --model "$SEEDM" \
    --corpus "$TRAIN_TXT" "${TOK_ARGS[@]}" \
    --output "$CONF" --meta-out "$META" --epochs "$EPOCHS"; done_step

step "train-router (MoE REINFORCE)"
"$CLOB" train-router --seed "$SEED" --model "$SEEDM" \
    --corpus "$TRAIN_TXT" "${TOK_ARGS[@]}" \
    --cf-rate 0.15 --max-tokens "$MAX_TOKENS" --output "$ROUTERS"; done_step

step "ingest (stream corpus, record episodes, metrics)"
"$CLOB" ingest --model "$SEEDM" --input "$INGEST_INPUT" "${TOK_ARGS[@]}" \
    --memory-dir "$EXP/episodes" \
    --confidence-head "$CONF" --meta-critic "$META" --routers "$ROUTERS" \
    --adaptive-compute --metrics-out "$EXP/metrics.jsonl" --metrics-window 1000 \
    --max-tokens "$MAX_TOKENS" "${CKPT_ARGS[@]}"; done_step

step "crystal (drain episodes into ternary modules)"
"$CLOB" crystal --model "$SEEDM" \
    --memory-dir "$EXP/episodes" --modules-dir "$EXP/modules"; done_step

step "bench-suite (ablation on the HELD-OUT split — Bet 2)"
"$CLOB" bench-suite --model "$SEEDM" --eval-corpus "$HOLDOUT_TXT" "${TOK_ARGS[@]}" \
    --seeds 1,2,3 \
    --confidence-head "$CONF" --meta-critic "$META" --routers-trained "$ROUTERS" \
    --modules-dirs "$EXP/modules" --out "$EXP/bench.tsv"; done_step

step "metrics dashboard"
"$CLOB" metrics --path "$EXP/metrics.jsonl"; done_step

trap - ERR
echo
echo "#### DONE in $((SECONDS))s — artifacts in $EXP"
echo "modules crystallized: $(ls -1 "$EXP/modules" 2>/dev/null | wc -l)"
echo "bench table:          $EXP/bench.tsv"
echo
echo "Judge bench.tsv against the pre-registered thresholds in ARCHITECTURE.md"
echo "(Part V → Falsification thresholds). Note: the core stays at synth init —"
echo "this run trains the heads/router and crystallizes modules; it is the"
echo "first real-data test of Bet 2 (crystallization gain), not core training."
