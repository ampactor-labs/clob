# data/corpus/ — real-corpus artifacts (Phase L)

This directory holds the **pinned, reproducible** training/holdout corpus for
clob's first real run. The text files themselves are *not* committed — they are
regenerated deterministically from the scripts, and their `.sha256` sidecars
(which *are* committed) pin exactly which bytes a given commit expects.

## How to populate

```bash
./scripts/acquire_corpus.sh     # download + train/holdout split + pin checksums
./scripts/tokenize_corpus.sh    # train BPE + encode to .tokens caches
```

Produces:

| File | Committed? | Provenance |
|:--|:--|:--|
| `train.txt` / `holdout.txt` | no (`.gitignore`) | head/tail split of the source text |
| `train.txt.sha256` / `holdout.txt.sha256` | **yes** | pins the expected bytes |
| `tokenizer.bin` | no | `clob train-tokenizer`, 4000 merges (default) |
| `train.tokens` / `holdout.tokens` | no | `clob encode` (`.tokens` format, magic `CLTK`) |

## Default source

`scripts/acquire_corpus.sh` defaults to **Project Gutenberg #2701 — *Moby
Dick* (Herman Melville)** — public domain, plain UTF-8, no attribution
required. The script strips the Gutenberg header/footer boilerplate before
splitting.

Override per-run without editing the script:

```bash
CORPUS_NAME=pride-and-prejudice \
CORPUS_URL=https://www.gutenberg.org/files/1342/1342-0.txt \
HOLDOUT_FRACTION=0.10 \
  ./scripts/acquire_corpus.sh
```

> **Licensing note.** PLAN.md's original candidate ("A Brief History of Time")
> is in copyright and was *not* used. Prefer genuinely public-domain Gutenberg
> texts, or a Wiki-40B slice if you accept CC-BY-SA attribution terms.

## Split discipline

`tests/data_split.rs` (via `src/util/split_check.rs`) enforces that **no
non-trivial holdout sentence appears in train**. When the real corpus is
present it is checked as part of `cargo test`; when absent that test self-skips
so CI stays green on a fresh clone.
