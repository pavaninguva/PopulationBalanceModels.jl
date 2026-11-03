# Protein parameter uncertainty

The three current scenarios use c* = 2.16 g/L (the Cacioppo–Pusey value), fixed
gamma = 0.64 mJ/m² and the same calibration/test split. `cnt` has three inferred
parameters; `cnt_grd` and `cnt_growth_grd` each have four. Their fourth parameters
are respectively constant D0 (μm²/min) and k_D (μm) in D_G = k_D G.

Original `cnt/`, `cnt_grd/` and `shared/` producer inputs are immutable. The new
scenario has its own producer folder. Analysis files here are derived separately;
the producer hashes are never replaced to make a compatibility check pass.

`paper/_support/protein_case_review.json` records the current reviewed numerical
source inventory and reasons the affected code paths retain the old target.
The current target is evaluated independently at each saved MAP, the first and
last draw of each chain, and coordinate extrema. Recorded densities, physical
parameters and AD/finite-difference gradients must agree. This is pointwise
compatibility evidence, not a global equivalence proof. Some original producer
sources were uncommitted; their complete contents cannot be recovered from the
producer revision alone. Their original identities and numerical records remain
preserved, and this qualification remains in each compatibility receipt.

The current public API requires a versioned MAP result. Original unversioned
MAPs are locally refitted from their saved coordinates for Gauss–Newton only;
`map_analysis.json` retains the original estimate and records the selected
center and real optimizer outcome. Neither the original fit curves nor their
MCMC-derived latent bands are silently recentered or replaced.

`gauss_newton.json` records the full joint precision/covariance in the declared
inference coordinates, parameter order, center, priors and diagnostics. It uses
whitened residual sensitivities plus the full transformed-prior curvature. It
omits residual-weighted second prediction derivatives. It is neither a full
posterior Hessian nor the sampler tuning matrix. Existing numerical rejection
checks remain in force; there is no jitter, pseudoinverse or substitute covariance.

Regenerate new analyses and figures from the repository root:

```sh
julia --project=paper paper/protein/run.jl --gauss-newton --model=all
python paper/protein/plot.py --gauss-newton --tables
python paper/protein/plot.py --tables
```

Calculation modes never enter the production branch, sample, or propagate latent
predictions. Plotting modes check recorded producer/file integrity and their
original compatibility receipt. A redraw does not assert current-source
numerical equivalence. Figures use actual LaTeX and 300-dpi PNG only; analytical
mapped Gaussian marginals/ellipses and retained-draw KDEs share axes and priors.
Seeded Gaussian parameter draws are cheap and untruncated; support/nonfinite
counts are in the figure receipts. These draws do not require PBM solves.

Generated CSV tables compare marginal medians/equal-tailed 95% intervals and
selected correlations in the displayed coordinates. A valid local approximation
need not recover curved compensation, prior-bound effects or multiple basins.
No global identifiability or model-selection claim follows from a narrow interval.
