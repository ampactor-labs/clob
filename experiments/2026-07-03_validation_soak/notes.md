# validation_soak

## Question

Do the Path B gradient and SSM BPTT checks remain stable under repeated release
test execution, and does the synthetic end-to-end pipeline still run cleanly?

## Run

Launched detached from `run.sh`. The main log is `soak.log`.

## Expected

- Repeated `cargo test --release learn::` passes.
- Repeated `cargo test --release` passes.
- Periodic `SMOKE=1 scripts/first_run.sh` passes.
