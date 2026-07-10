#!/usr/bin/env bash
# Phase 6: real-corpus QAT vs matched f32 controls.
# The procedure registered by notes.md. Receipts (curves.tsv, results.tsv,
# verdict.txt, runinfo.txt) land in the experiment dir; training artifacts
# and logs stay in the workdir.
set -euo pipefail

REPO=/home/suds/Projects/clob
CLOB=$REPO/target/release/clob
TR=$REPO/data/corpus/train.tokens
HO=$REPO/data/corpus/holdout.tokens
TOK=$REPO/data/corpus/tokenizer.bin
EXP=$REPO/experiments/2026-07-10_phase6_real_qat
W=${PHASE6_WORKDIR:-/tmp/claude-1000/-home-suds-Projects-clob/7e2d5167-0855-4cd6-8485-9b1ff20b630d/scratchpad/phase6}

CFG=small; WINDOW=32; STEPS=20000; CKPT=2500; SEED=1

rm -rf "$W"; mkdir -p "$W"; cd "$W"

{
    echo "commit  $(git -C $REPO rev-parse HEAD)"
    echo "binary  $(sha256sum $CLOB | cut -d' ' -f1)"
    echo "date    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$EXP/runinfo.txt"

# Shared init artifact (init params depend only on dims + seed, not mode).
$CLOB train --corpus "$TR" --tokenizer "$TOK" --config $CFG --window $WINDOW \
    --steps 0 --seed $SEED --output init.dense >/dev/null 2>&1

curve() { # curve <name> <view: latent|ternary>
    local name=$1 view=$2 tflag=""
    [[ $view == ternary ]] && tflag="--ternary"
    printf 'step\tnll4k\tartifact\n' > "${name}_curve.tsv"
    local arts=(init.dense)
    for d in $(ls -d "ck_${name}"/checkpoint_* | sort); do arts+=("$d/trained.dense"); done
    for art in "${arts[@]}"; do
        local lbl
        if [[ $art == init.dense ]]; then lbl=0
        else lbl=$(echo "$art" | grep -oE 'checkpoint_0*([0-9]+)' | grep -oE '[0-9]+$'); fi
        local nll
        nll=$($CLOB eval-dense --dense "$art" --corpus "$HO" --tokenizer "$TOK" \
              --window $WINDOW --max-tokens 4000 $tflag 2>/dev/null \
              | grep -oE '"nll":[-0-9.]+' | cut -d: -f2)
        printf '%s\t%s\t%s\n' "$lbl" "$nll" "$art" >> "${name}_curve.tsv"
        echo "  [$name] step $lbl: nll4k=$nll"
    done
}

full_eval() { # full_eval <artifact> <view> -> "nll unigram"
    local art=$1 view=$2 tflag=""
    [[ $view == ternary ]] && tflag="--ternary"
    $CLOB eval-dense --dense "$art" --corpus "$HO" --tokenizer "$TOK" \
        --window $WINDOW --max-tokens 0 $tflag 2>/dev/null \
        | grep -oE '"(nll|unigram)":[-0-9.]+' | cut -d: -f2 | paste -sd' '
}

run_one() { # run_one <name> <lr> <wd> <view> [--qat]
    local name=$1 lr=$2 wd=$3 view=$4 modeflag=${5:-}
    echo "== $name: train (lr=$lr wd=$wd mode=${modeflag:-latent}) =="
    $CLOB train --corpus "$TR" --tokenizer "$TOK" --config $CFG --window $WINDOW \
        --lr "$lr" --weight-decay "$wd" --clip-norm 1.0 --steps $STEPS \
        --checkpoint-every $CKPT --checkpoint-dir "ck_${name}" --log-every 2500 \
        --seed $SEED $modeflag --output "${name}_final.dense" > "${name}_train.log" 2>&1
    echo "== $name: holdout curve ($view view) =="
    curve "$name" "$view"
}

run_one latent_A 5e-3 0.01 latent
run_one latent_B 2e-3 0.0  latent
run_one qat_A    5e-3 0.01 ternary --qat
run_one qat_B    2e-3 0.0  ternary --qat

# Selection: per run, lowest curve NLL; ties go to the lower step.
select_art() { # select_art <name> -> "step nll art"
    tail -n +2 "${1}_curve.tsv" | sort -t$'\t' -k2,2g -k1,1n | head -1 \
        | awk -F'\t' '{print $1, $2, $3}'
}

printf 'kind\trun\tmode\tlr\twd\tselected_step\tnll4k\tfull_nll\tnote\n' > "$EXP/results.tsv"
declare -A FULL SEL
for spec in "latent_A latent 5e-3 0.01" "latent_B latent 2e-3 0.0" \
            "qat_A ternary 5e-3 0.01" "qat_B ternary 2e-3 0.0"; do
    read -r name view lr wd <<< "$spec"
    read -r step nll4k art <<< "$(select_art "$name")"
    read -r fnll funi <<< "$(full_eval "$art" "$view")"
    mode=latent; [[ $view == ternary ]] && mode=qat
    printf 'run\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tselected artifact full-holdout (%s view)\n' \
        "$name" "$mode" "$lr" "$wd" "$step" "$nll4k" "$fnll" "$view" >> "$EXP/results.tsv"
    FULL[$name]=$fnll; SEL[$name]=$art
    echo "== $name: selected step $step, full-holdout nll=$fnll (unigram $funi)"
    UNIGRAM=$funi
done

# Arm representatives: lower full-holdout NLL per arm (tie -> recipe A).
rep_of() { # rep_of <nameA> <nameB>
    if awk -v a="${FULL[$1]}" -v b="${FULL[$2]}" 'BEGIN{exit !(b<a)}'; then echo "$2"; else echo "$1"; fi
}
LREP=$(rep_of latent_A latent_B)
QREP=$(rep_of qat_A qat_B)
echo "== representatives: latent=$LREP qat=$QREP"

# Context: regime of each representative (deployed view), and the latent
# representative's effective-ternary projection on the full holdout.
LLAM=$($CLOB regime --dense "${SEL[$LREP]}" --corpus "$HO" --tokenizer "$TOK" \
       --tokens 2000 --warmup 256 --seed $SEED 2>/dev/null \
       | grep -oE 'lambda1 = [-0-9.]+' | grep -oE '[-0-9.]+$' || echo nan)
QLAM=$($CLOB regime --dense "${SEL[$QREP]}" --ternary --corpus "$HO" --tokenizer "$TOK" \
       --tokens 2000 --warmup 256 --seed $SEED 2>/dev/null \
       | grep -oE 'lambda1 = [-0-9.]+' | grep -oE '[-0-9.]+$' || echo nan)
read -r LTERN _ <<< "$(full_eval "${SEL[$LREP]}" ternary)"
printf 'context\t%s\tlatent\t-\t-\t-\t-\t%s\tlambda1 of latent representative\n' "$LREP" "$LLAM" >> "$EXP/results.tsv"
printf 'context\t%s\tqat\t-\t-\t-\t-\t%s\tlambda1 of qat representative (ternary view)\n' "$QREP" "$QLAM" >> "$EXP/results.tsv"
printf 'context\t%s\tlatent\t-\t-\t-\t-\t%s\tlatent representative effective-ternary projection, full holdout\n' "$LREP" "$LTERN" >> "$EXP/results.tsv"

# Concatenated curves receipt.
printf 'run\tstep\tnll4k\n' > "$EXP/curves.tsv"
for name in latent_A latent_B qat_A qat_B; do
    tail -n +2 "${name}_curve.tsv" | awk -F'\t' -v r="$name" '{printf "%s\t%s\t%s\n", r, $1, $2}' >> "$EXP/curves.tsv"
done

# Verdict per the pre-registered rule (notes.md). The memory line is derived
# from the shapes here so the number is arithmetic, not assertion.
LFULL=${FULL[$LREP]} QFULL=${FULL[$QREP]} UNI=$UNIGRAM LR_=$LREP QR_=$QREP \
python3 - > "$EXP/verdict.txt" <<'PY'
import os
lat = float(os.environ['LFULL']); qat = float(os.environ['QFULL'])
uni = float(os.environ['UNI'])
lrep = os.environ['LR_']; qrep = os.environ['QR_']

# Memory accounting: small(3260), TernaryMatrix encoding (2 b/trit,
# 32-row tiling, f32 per-row scales) over the nine ternary families.
d, dinner, vocab, L = 64, 128, 3260, 3
xrows = 4 + 2*4*16
fams = [(d,d),(xrows,d),(d,d),(d,d),(d,d),(d,d),(dinner,d),(dinner,d),(d,dinner)]
packed = sum(((r+31)//32)*32 * ((c+3)//4) for r,c in fams) * L
scales = sum(r for r,_ in fams) * 4 * L
tern_params = sum(r*c for r,c in fams) * L
core_f32 = tern_params * 4
core_ratio = core_f32 / (packed + scales)
layer_rest = d + 64 + 4 + 4 + d + 3*d   # norms, a_log, d_param, dt_bias, biases
total_params = 2*vocab*d + d + L*layer_rest + tern_params
whole_f32 = total_params * 4
whole_dep = (packed + scales) + (total_params - tern_params) * 4
whole_ratio = whole_f32 / whole_dep

valid = lat < uni
gap = qat / lat
if not valid:
    cap = 'INVALID (f32 control did not beat the holdout unigram baseline)'
elif gap <= 1.25:
    cap = 'PASS'
elif gap > 1.5:
    cap = 'KILL'
else:
    cap = 'INDETERMINATE'
mem = 'PASS' if core_ratio >= 10 else ('KILL' if core_ratio < 8 else 'INDETERMINATE')

print(f"latent representative: {lrep}, full-holdout NLL {lat:.6f}")
print(f"qat representative:    {qrep}, full-holdout NLL {qat:.6f} (ternary view)")
print(f"holdout unigram baseline: {uni:.6f}")
print(f"validity precondition (latent < unigram): {'OK' if valid else 'FAILED'}")
print(f"capability gap = {qat:.6f} / {lat:.6f} = {gap:.4f}x  (pass <=1.25, kill >1.5)")
print(f"BET 1 CAPABILITY LINE: {cap}")
print(f"memory, core families: f32 {core_f32} B vs deployed {packed+scales} B "
      f"= {core_ratio:.2f}x smaller  (pass >=10x, kill <8x)")
print(f"memory, whole model (context): f32 {whole_f32} B vs deployed {whole_dep} B "
      f"= {whole_ratio:.2f}x")
print(f"BET 1 MEMORY LINE: {mem}")
PY

echo "== verdict =="
cat "$EXP/verdict.txt"
echo "== done =="
