# Historical sampler-preconditioner provenance

`grd_100bin_acceptance099_completed_20261025.zip` remains because
`paper/protein/sampling_transport.json` identifies its draws as the training
source of a frozen sampler-coordinate preconditioner retained for compatibility.
The recorded path inside that JSON describes the producing checkout; the archive
here retains its original bytes and `production/cnt_grd/posterior_draws.csv` member.

This was a superseded variable-gamma, D = theta G² target: four chains of
1,500 retained draws, 140 divergences, maximum Rhat 1.0615 and minimum
bulk/tail ESS 71/83. It failed the sampling screen. Its draws are neither priors
nor posterior uncertainty for the current models. Current production uses
`initial_rates` coordinates and expected-information tuning, independently of
this pilot.

Other inferred-noise, interrupted and unsuccessful pilot archives have been
relocated to the external recovery archive described in the
[archive catalogue](../../README.md#relocated-legacy-outputs). Their original
records are preserved there, without changes to current samples or bands.
