Milling forward-UQ results (Bhonsale et al., 2018, DOI 10.3390/pr6120255).

data/provenance.json is the original production receipt. It records settings,
source/dependency versions and numerical-file hashes. Do not replace its hashes
after editing code. settings.json also preserves the pre-run configuration.

Each *_inputs.csv pairs sample_id with four standard-normal reference values z
and their physical delta/alpha/gamma/sigma values. The reference law is the
independent product of Normal(0,1); physical = Table-1 mean + SD*z. The means and
SDs are in the source model, case-study table and independent result checker.
Training boxes do not truncate that law. Invalid samples would remain failures.

The columns t1,... in *_outputs.csv.gz and *_coefficients.csv follow times.csv.
All output values are instantaneous outlet volume-weighted diameter medians
in micrometres. The first column of an output file is its sample identity.
PCE coefficients additionally record their total-degree multi-index; bases are
orthonormal probabilists' Hermite products, with constant term first.

MC summaries use sample mean/SD and empirical quantiles. PCE summary means/SDs
are analytical coefficient moments; quantiles use the saved shared 50,000-point
PCE propagation design. surrogate_mean_mcse refers to estimating a surrogate
mean from those cheap draws, not uncertainty in the analytical coefficient mean.
Mean +/-1.96 SD is a labelled Gaussian-reference band, not a confidence interval
for a mean. Quantiles and the Gaussian-reference band are pointwise.

validation_* is independent full-model data, shared by all PCE fits.
*_common_inputs_error.csv instead evaluates surrogates on the MC reference
inputs; it is a paired comparison, not an extra independent validation set.
The MC zero surrogate-error placeholder in costs.csv is not a zero numerical
error claim. Numerical refinement and the unresolved literature discrepancy
are discussed separately in the case study. See reconstruction.toml for all
source-to-code assumptions, including initialization and median interpolation.

Figures use only retained production tables. Replot with:
python docs/benchmarks/plot_milling_uncertainty.py --usetex
Smoke/preview results remain in .cache and cannot replace these figures.
