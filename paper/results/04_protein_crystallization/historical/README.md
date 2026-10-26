# Historical inferred-noise target

`cnt_inferred_noise_8abcd0f.zip` preserves the original CNT-only outputs,
figures and producer code from revision 8abcd0f. It inferred two observation
scales and has six coordinates. It is not data for the revised fixed-error
CNT/GRD comparison and must not supply its corners or prediction bands.

The fixed-error pilot archive records a completed CNT attempt with one retained
divergence and the interrupted direct-rate GRD attempt. The unit-shear and
soft-barrier/expected-information archives retain later interrupted GRD tuning
attempts and their producer sources. These have no verified retained posterior
and supply no prediction bands or corner plots. Tuning changes preserve the
physical model, priors and fixed-error likelihood.

`grd_acceptance090_completed_20261025.zip` records the completed GRD attempt:
four chains × 500 retained, 500 warmup, 49 retained divergences, maximum
Rhat 1.01546 and minimum bulk/tail ESS 226/140. Its draws do not supply final
intervals or corner plots. Independent target/mapped-gradient checks passed
at 16 representative retained points; the active workflow uses a stricter,
larger fresh run and a training-range auxiliary growth-rate coordinate.

The 200-bin acceptance-0.99 archive is an interrupted computational attempt,
not a completed posterior. The active 100-bin GRD target is refitted and
sampled independently after calibration-only spatial checks; no 200-bin draws
are relabelled as its results.

`grd_100bin_acceptance099_completed_20261025.zip` preserves the subsequent
completed 100-bin attempt: four chains of 1,500 retained draws, 140 retained
divergences, maximum Rhat 1.0615 and minimum bulk/tail ESS 71/83. It also fails
the sampling screen. Its calibration-only draws inform a frozen sampling-coordinate
preconditioner, not the priors or physical model. Final intervals require fresh
chains with the full inverse-map Jacobian; none of these failed pilot draws
are used in the active figures.
