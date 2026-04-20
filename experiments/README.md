# Experiments

Each non-trivial `clob` run goes in its own directory under this root,
named `YYYY-MM-DD_<short-label>/`. Scaffold a new one with:

```bash
./scripts/new_experiment.sh my_label
```

which creates `experiments/YYYY-MM-DD_my_label/` pre-populated with:

- `manifest.toml` — top-level manifest for the whole experiment
  (distinct from the per-artifact sidecar manifests that individual
  subcommands produce).
- `run.sh` — executable stub where the experiment's subcommand
  invocations live.
- `notes.md` — free-form journal; what the run is testing, what you
  expected, what actually happened.

Convention: don't check in large artifacts. The directory is intended
for small scale — manifests, runbooks, bench-suite TSVs, metric jsonls,
notes. Big model / memory / module files stay where the subcommand
wrote them (outside this tree) and the manifest records their sha256
so the run is still reproducible.

A healthy experiment directory can be re-executed from `run.sh` alone,
given the same `--seed`, and produces byte-identical outputs per
Phase G's reproducibility gate.
