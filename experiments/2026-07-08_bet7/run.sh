#!/usr/bin/env bash
set -euo pipefail
CLOB=/home/suds/Projects/clob/target/release/clob
TR=/home/suds/Projects/clob/data/corpus/train.tokens
HO=/home/suds/Projects/clob/data/corpus/holdout.tokens
TOK=/home/suds/Projects/clob/data/corpus/tokenizer.bin
W=/tmp/claude-1000/-home-suds-Projects-clob/c5f6f849-fbef-44d8-b0ae-c15890a459a9/scratchpad/bet7
rm -rf "$W"; mkdir -p "$W"; cd "$W"

CFG=small; WINDOW=32; LR=5e-3; STEPS=20000; CKPT=2500; SEED=1

echo "== init (steps 0) =="
$CLOB train --corpus "$TR" --tokenizer "$TOK" --config $CFG --window $WINDOW --steps 0 --seed $SEED --output init.dense >/dev/null 2>&1

echo "== train $STEPS steps, checkpoint every $CKPT =="
$CLOB train --corpus "$TR" --tokenizer "$TOK" --config $CFG --window $WINDOW --lr $LR \
    --steps $STEPS --checkpoint-every $CKPT --checkpoint-dir ck --log-every 5000 \
    --seed $SEED --output final.dense 2>&1 | grep -E "step|done"

# Collect checkpoint artifacts in step order: init, each ckpt, final.
declare -a ARTS=(init.dense)
for d in $(ls -d ck/checkpoint_* | sort); do ARTS+=("$d/trained.dense"); done
ARTS+=(final.dense)

echo -e "step\theldout_nll\tlambda1\thorizon" > sweep.tsv
step=0
for art in "${ARTS[@]}"; do
    # step label
    if [[ "$art" == "init.dense" ]]; then lbl=0
    elif [[ "$art" == "final.dense" ]]; then lbl=$STEPS
    else lbl=$(echo "$art" | grep -oE 'checkpoint_0*([0-9]+)' | grep -oE '[0-9]+$'); fi
    nll=$($CLOB eval-dense --dense "$art" --corpus "$HO" --tokenizer "$TOK" --window $WINDOW --max-tokens 4000 2>/dev/null | grep -oE '"nll":[-0-9.]+' | cut -d: -f2)
    lam=$($CLOB regime --dense "$art" --corpus "$HO" --tokenizer "$TOK" --tokens 2000 --warmup 256 --seed $SEED 2>/dev/null | grep -oE 'lambda1 = [-0-9.]+' | grep -oE '[-0-9.]+$')
    hor=$($CLOB regime --dense "$art" --corpus "$HO" --tokenizer "$TOK" --tokens 2000 --warmup 256 --seed $SEED 2>/dev/null | grep -oE 'horizon ≈ [0-9.]+' | grep -oE '[0-9.]+$' || echo "inf")
    echo -e "${lbl}\t${nll}\t${lam}\t${hor}" >> sweep.tsv
    echo "  step ${lbl}: heldout_nll=${nll} lambda1=${lam} horizon=${hor}"
done

echo "== sweep.tsv =="; cat sweep.tsv

python3 - <<'PY'
import csv
rows=[r for r in csv.DictReader(open('bet7/sweep.tsv'.replace('bet7/','')), delimiter='\t')]
lam=[float(r['lambda1']) for r in rows]
nll=[float(r['heldout_nll']) for r in rows]
cap=[-x for x in nll]  # capability = -held-out NLL
def spearman(a,b):
    def rank(x):
        order=sorted(range(len(x)), key=lambda i:x[i])
        rk=[0]*len(x)
        i=0
        while i<len(x):
            j=i
            while j+1<len(x) and x[order[j+1]]==x[order[i]]: j+=1
            avg=(i+j)/2.0
            for k in range(i,j+1): rk[order[k]]=avg
            i=j+1
        return rk
    ra,rb=rank(a),rank(b)
    n=len(a); ma=sum(ra)/n; mb=sum(rb)/n
    cov=sum((ra[i]-ma)*(rb[i]-mb) for i in range(n))
    va=sum((x-ma)**2 for x in ra)**0.5; vb=sum((x-mb)**2 for x in rb)**0.5
    return cov/(va*vb) if va*vb>0 else float('nan')
rho=spearman(lam,cap)
best_i=min(range(len(nll)), key=lambda i:nll[i])  # best capability = lowest NLL
best_lam=lam[best_i]
print(f"\n== Bet 7 verdict ==")
print(f"n_checkpoints={len(rows)}")
print(f"spearman(lambda1, capability=-NLL) = {rho:+.3f}")
print(f"best-capability checkpoint: step {rows[best_i]['step']}, NLL {nll[best_i]:.4f}, lambda1 {best_lam:+.4f}")
band = (-0.5 < best_lam <= 0.05)
if rho >= 0.6 and band:
    print("VERDICT: PASS (rho>=0.6 and best lambda1 in (-0.5, 0.05])")
elif abs(rho) < 0.2 or best_lam < -1.0:
    print("VERDICT: KILL (|rho|<0.2 or best lambda1 < -1)")
else:
    print(f"VERDICT: INDETERMINATE (rho={rho:+.3f}, best lambda1={best_lam:+.4f}, band={band})")
PY
