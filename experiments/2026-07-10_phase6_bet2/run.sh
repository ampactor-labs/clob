#!/usr/bin/env bash
# Phase 6 second half: Bet 2 on the trained dense substrate.
# Procedure registered by notes.md. Receipts (results.tsv, grid.tsv,
# verdict.txt, runinfo.txt) land in the experiment dir; substrate artifacts and
# logs stay in the workdir.
set -euo pipefail

REPO=/home/suds/Projects/clob
CLOB=$REPO/target/release/clob
TR=$REPO/data/corpus/train.tokens
HO=$REPO/data/corpus/holdout.tokens
TOK=$REPO/data/corpus/tokenizer.bin
EXP=$REPO/experiments/2026-07-10_phase6_bet2
W=${BET2_WORKDIR:-/tmp/claude-1000/-home-suds-Projects-clob/7e2d5167-0855-4cd6-8485-9b1ff20b630d/scratchpad/bet2}

CFG=small; WINDOW=32; SEED=1; SUBSTRATE_STEPS=15000

rm -rf "$W"; mkdir -p "$W"; cd "$W"

{
    echo "commit  $(git -C $REPO rev-parse HEAD)"
    echo "binary  $(sha256sum $CLOB | cut -d' ' -f1)"
    echo "date    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$EXP/runinfo.txt"

echo "== train the QAT substrate ($SUBSTRATE_STEPS steps, qat_B recipe) =="
$CLOB train --corpus "$TR" --tokenizer "$TOK" --config $CFG --window $WINDOW \
    --lr 2e-3 --weight-decay 0.0 --clip-norm 1.0 --steps $SUBSTRATE_STEPS \
    --seed $SEED --qat --output qat_substrate.dense 2>&1 | tail -1

echo "== train the f32-latent substrate (context; latent_B recipe) =="
$CLOB train --corpus "$TR" --tokenizer "$TOK" --config $CFG --window $WINDOW \
    --lr 2e-3 --weight-decay 0.0 --clip-norm 1.0 --steps $SUBSTRATE_STEPS \
    --seed $SEED --output latent_substrate.dense 2>&1 | tail -1

# crystal-dense wrapper -> emits the JSON line; jq-free field extraction.
field() { grep -oE "\"$1\":[-0-9.]+" | cut -d: -f2; }

run_cd() { # run_cd <label> <substrate> <view-flag> <n_clusters> <causal-flag> <act-thresh>
    local label=$1 sub=$2 view=$3 nc=$4 causal=$5 act=$6
    $CLOB crystal-dense --dense "$sub" --train "$TR" --holdout "$HO" --tokenizer "$TOK" \
        --window $WINDOW $view --error-threshold 0.5 --max-episodes 4000 \
        --n-clusters "$nc" $causal --causal-horizon 8 --activation-threshold "$act" \
        --max-eval-tokens 0 2>"${label}.err" | tee "${label}.json"
}

echo "== PRIMARY: qat substrate, n_clusters=16, causal ON, act=0.3 =="
run_cd primary qat_substrate.dense --ternary 16 --causal 0.3

PJSON=$(cat primary.json)
NC=$(echo "$PJSON" | field nll_cleared)
NL=$(echo "$PJSON" | field nll_loaded)
DP=$(echo "$PJSON" | field delta_pct)
MODS=$(echo "$PJSON" | field modules)
FRAC=$(echo "$PJSON" | field frac_activated)

printf 'metric\tvalue\n' > "$EXP/results.tsv"
{
    printf 'substrate\tqat_B step-%s ternary view\n' "$SUBSTRATE_STEPS"
    printf 'primary_nll_cleared\t%s\n' "$NC"
    printf 'primary_nll_loaded\t%s\n' "$NL"
    printf 'primary_delta_pct\t%s\n' "$DP"
    printf 'primary_modules\t%s\n' "$MODS"
    printf 'primary_frac_tokens_routed\t%s\n' "$FRAC"
    printf 'holdout_unigram\t4.776462\n'
} >> "$EXP/results.tsv"

echo "== SENSITIVITY GRID (reported, not verdict-bearing) =="
printf 'cell\tn_clusters\tcausal\tact_threshold\tsubstrate\tview\tmodules\tnll_cleared\tnll_loaded\tdelta_pct\tfrac_routed\n' > "$EXP/grid.tsv"
grid_row() { # grid_row <cell> <substrate> <view> <nc> <causal> <act>
    local cell=$1 sub=$2 view=$3 nc=$4 causal=$5 act=$6
    run_cd "$cell" "$sub" "$view" "$nc" "$causal" "$act" >/dev/null
    local j; j=$(cat "${cell}.json")
    local cflag="off"; [[ "$causal" == "--causal" ]] && cflag="on"
    local vflag="ternary"; [[ "$view" == "--latent" ]] && vflag="latent"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$cell" "$nc" "$cflag" "$act" "$(basename "$sub" .dense)" "$vflag" \
        "$(echo "$j" | field modules)" "$(echo "$j" | field nll_cleared)" \
        "$(echo "$j" | field nll_loaded)" "$(echo "$j" | field delta_pct)" \
        "$(echo "$j" | field frac_activated)" >> "$EXP/grid.tsv"
    echo "  [$cell] delta_pct=$(echo "$j" | field delta_pct)"
}
# Primary echoed into the grid for reference, then one-at-a-time variations.
grid_row primary       qat_substrate.dense    --ternary 16 --causal 0.3
grid_row nclusters_8   qat_substrate.dense    --ternary 8  --causal 0.3
grid_row nclusters_32  qat_substrate.dense    --ternary 32 --causal 0.3
grid_row causal_off    qat_substrate.dense    --ternary 16 ""        0.3
grid_row act_0p5       qat_substrate.dense    --ternary 16 --causal 0.5
grid_row act_0p7       qat_substrate.dense    --ternary 16 --causal 0.7
grid_row latent_ctx    latent_substrate.dense --latent  16 --causal 0.3

echo "== verdict =="
DP="$DP" NC="$NC" python3 - > "$EXP/verdict.txt" <<'PY'
import os
dp = float(os.environ['DP']); nc = float(os.environ['NC'])
uni = 4.776462
valid = nc < uni
if not valid:
    cap = 'INVALID (substrate cleared NLL did not beat the holdout unigram)'
elif dp >= 2.0:
    cap = 'PASS'
elif dp <= 0.0:
    cap = 'KILL'
else:
    cap = 'INDETERMINATE'
print(f"substrate cleared holdout NLL: {nc:.6f}  (unigram {uni:.6f})")
print(f"validity precondition (cleared < unigram): {'OK' if valid else 'FAILED'}")
print(f"primary delta_pct (loaded vs cleared): {dp:+.4f}%  (pass >=2.0, kill <=0.0)")
print(f"BET 2 CAPABILITY LINE: {cap}")
print("BET 2 DRIFT LINE: PASS-by-construction "
      "(untouched tokens scored from identical hidden in both arms; drift = 0)")
PY
cat "$EXP/verdict.txt"
echo "== grid =="; cat "$EXP/grid.tsv"
echo "== done =="
