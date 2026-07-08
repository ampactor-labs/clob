#!/usr/bin/env bash
# Path B Phase 5 planted-pattern dense-core run.
#
# This script owns the run receipt. It trains on the synthetic diagnostic
# corpus with the pre-registered hyperparameters in manifest.toml, then writes:
#
#   results.tsv  per-scope baselines and dense NLLs
#   verdict.txt  PASS / KILL / INDETERMINATE under the registered rule
#
# Large artifacts are written under WORK and are intentionally not tracked.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EXP="$ROOT/experiments/2026-07-08_phase5_planted_patterns"
CLOB="$ROOT/target/release/clob"
CORPUS="$ROOT/data/synthetic/diagnostic.txt"

: "${SEED:=1}"
: "${CONFIG:=small}"
: "${WINDOW:=32}"
: "${LR:=5e-3}"
: "${WEIGHT_DECAY:=0.01}"
: "${CLIP_NORM:=1.0}"
: "${STEPS:=8000}"
: "${TERNARIZE_EVERY:=2000}"
: "${WORK:=/tmp/clob_phase5_planted_patterns}"

mkdir -p "$WORK"

if [ ! -x "$CLOB" ]; then
    (cd "$ROOT" && cargo build --release)
fi

echo "== Phase 5 planted-pattern dense training =="
echo "root:   $ROOT"
echo "work:   $WORK"
echo "corpus: $CORPUS"
echo "seed=$SEED config=$CONFIG window=$WINDOW lr=$LR steps=$STEPS"
echo

python3 - "$CORPUS" "$WORK/sections" <<'PY'
import pathlib
import re
import sys

corpus = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

sections = {}
current = None
buf = []
for line in corpus.read_text().splitlines(keepends=True):
    m = re.match(r"SECTION\s+(\d+)\s+(.+)$", line.strip())
    if m:
        if current is not None:
            sections[current] = "".join(buf).lstrip("\n")
        label = m.group(2).lower().replace(" ", "_")
        current = f"section_{int(m.group(1))}_{label}"
        buf = []
    else:
        buf.append(line)
if current is not None:
    sections[current] = "".join(buf).lstrip("\n")

for name, text in sections.items():
    (out / f"{name}.txt").write_text(text)
print(f"wrote {len(sections)} section files to {out}")
PY

echo "== random init artifact =="
"$CLOB" train \
    --corpus "$CORPUS" \
    --config "$CONFIG" \
    --window "$WINDOW" \
    --steps 0 \
    --seed "$SEED" \
    --output "$WORK/init.dense" \
    > "$WORK/init.log"

echo "== train =="
"$CLOB" train \
    --corpus "$CORPUS" \
    --config "$CONFIG" \
    --window "$WINDOW" \
    --lr "$LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --clip-norm "$CLIP_NORM" \
    --steps "$STEPS" \
    --ternarize-every "$TERNARIZE_EVERY" \
    --log-every 1000 \
    --seed "$SEED" \
    --output "$WORK/final.dense" \
    2>&1 | tee "$WORK/train.log"

echo "== evaluate scopes =="
python3 - "$CLOB" "$WORK/init.dense" "$WORK/final.dense" "$CORPUS" "$WORK/sections" "$WINDOW" "$EXP/results.tsv" "$EXP/verdict.txt" <<'PY'
import csv
import json
import math
import pathlib
import subprocess
import sys

clob, init_dense, final_dense, corpus, sections_dir, window, results_path, verdict_path = sys.argv[1:]
clob = pathlib.Path(clob)
init_dense = pathlib.Path(init_dense)
final_dense = pathlib.Path(final_dense)
corpus = pathlib.Path(corpus)
sections_dir = pathlib.Path(sections_dir)
results_path = pathlib.Path(results_path)
verdict_path = pathlib.Path(verdict_path)
vocab = 260
alpha = 1.0

section_paths = sorted(sections_dir.glob("section_*.txt"))
scopes = [("whole", corpus, False)]
for path in section_paths:
    structured = not path.name.startswith("section_4_")
    scopes.append((path.stem, path, structured))

def byte_tokens(path):
    return list(path.read_bytes())

def bigram_laplace_nll(tokens):
    if len(tokens) < 2:
        return 0.0
    ctx = [0] * vocab
    trans = {}
    for a, b in zip(tokens, tokens[1:]):
        ctx[a] += 1
        trans[(a, b)] = trans.get((a, b), 0) + 1
    total = 0.0
    n = 0
    for a, b in zip(tokens, tokens[1:]):
        p = (trans.get((a, b), 0) + alpha) / (ctx[a] + alpha * vocab)
        total -= math.log(p)
        n += 1
    return total / n

def eval_dense(dense, scope_path, ternary=False):
    cmd = [
        str(clob),
        "eval-dense",
        "--dense",
        str(dense),
        "--corpus",
        str(scope_path),
        "--window",
        str(window),
    ]
    if ternary:
        cmd.append("--ternary")
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    for line in reversed(out.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"no JSON in eval-dense output for {scope_path}")

rows = []
for scope, path, structured in scopes:
    tokens = byte_tokens(path)
    init_eval = eval_dense(init_dense, path)
    final_eval = eval_dense(final_dense, path)
    ternary_eval = eval_dense(final_dense, path, ternary=True)
    unigram = float(final_eval["unigram"])
    bigram = bigram_laplace_nll(tokens)
    final_nll = float(final_eval["nll"])
    beats_unigram = final_nll < unigram
    beats_bigram = final_nll < bigram
    rows.append({
        "scope": scope,
        "tokens": len(tokens),
        "structured": "yes" if structured else "no",
        "unigram_nll": unigram,
        "bigram_laplace_nll": bigram,
        "init_latent_nll": float(init_eval["nll"]),
        "final_latent_nll": final_nll,
        "final_ternary_nll": float(ternary_eval["nll"]),
        "latent_vs_unigram": final_nll / unigram if unigram > 0 else float("nan"),
        "latent_vs_bigram": final_nll / bigram if bigram > 0 else float("nan"),
        "beats_unigram": "yes" if beats_unigram else "no",
        "beats_bigram": "yes" if beats_bigram else "no",
        "beats_both": "yes" if beats_unigram and beats_bigram else "no",
    })

fieldnames = [
    "scope",
    "tokens",
    "structured",
    "unigram_nll",
    "bigram_laplace_nll",
    "init_latent_nll",
    "final_latent_nll",
    "final_ternary_nll",
    "latent_vs_unigram",
    "latent_vs_bigram",
    "beats_unigram",
    "beats_bigram",
    "beats_both",
]
with results_path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
    w.writeheader()
    for row in rows:
        w.writerow(row)

by_scope = {row["scope"]: row for row in rows}
structured_rows = [r for r in rows if r["structured"] == "yes"]
structured_beats = sum(1 for r in structured_rows if r["beats_both"] == "yes")
whole = by_scope["whole"]

pass_conditions = [
    whole["final_latent_nll"] <= 0.75 * whole["unigram_nll"],
    structured_beats == 4,
    by_scope["section_1_periodic_phrase"]["final_latent_nll"] <= 1.0,
    by_scope["section_2_shift_rule"]["final_latent_nll"] <= 1.5,
    by_scope["section_3_period_two_alternation"]["final_latent_nll"] <= 1.2,
]
kill_conditions = [
    whole["final_latent_nll"] >= whole["unigram_nll"],
    structured_beats < 3,
]

if all(pass_conditions):
    verdict = "PASS"
elif any(kill_conditions):
    verdict = "KILL"
else:
    verdict = "INDETERMINATE"

lines = [
    f"verdict = {verdict}",
    f"structured_sections_beating_both = {structured_beats}/4",
    f"whole_final_latent_nll = {whole['final_latent_nll']:.6f}",
    f"whole_unigram_nll = {whole['unigram_nll']:.6f}",
    f"whole_bigram_laplace_nll = {whole['bigram_laplace_nll']:.6f}",
    f"whole_latent_vs_unigram = {whole['latent_vs_unigram']:.6f}",
    "",
    "pass_conditions = " + ",".join("true" if x else "false" for x in pass_conditions),
    "kill_conditions = " + ",".join("true" if x else "false" for x in kill_conditions),
]
verdict_path.write_text("\n".join(lines) + "\n")

print(results_path.read_text(), end="")
print(verdict_path.read_text(), end="")
PY

echo
echo "artifacts:"
echo "  tracked: $EXP/results.tsv"
echo "  tracked: $EXP/verdict.txt"
echo "  work:    $WORK"
