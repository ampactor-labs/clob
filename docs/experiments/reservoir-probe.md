# Reservoir Probe Result

Artifact source:
`experiments/2026-06-22_first_run/probe_readout.json`

This probe fit a dense f32 readout over frozen-core hidden states and compared
held-out NLL against a trained bias-only unigram null. The deciding metric is
the contextual gain from the hidden state term `W * h`; bias fitting alone does
not count as context.

Observed result:

- model tied random readout holdout NLL: `8.506331`
- Laplace unigram marginal: `4.916769`
- trained bias-only null: `4.916769`
- trained full readout, final: `4.924845`
- trained full readout, floored against null: `4.916769`
- contextual gain over null: `0.000000` nats, `0.000000` relative
- verdict: `RESERVOIR DEAD`

Interpretation: the current random core did not expose linearly readable
context beyond token marginal statistics. Treat Path A, the frozen-reservoir
interpretation, as failed for this implementation until a replicated probe says
otherwise.

Engineering consequence: pursue Path B. Untie and train the core before
spending more effort on crystallizing corrections over frozen random hidden
states.
