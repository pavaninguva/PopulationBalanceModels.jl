# Active and historical observations

The active Tang 100 mL isothermal study uses
[`tang2018_isothermal/`](tang2018_isothermal/README.md), with original concentration
and conditional number-CDF readings. The files described below preserve the
older Mitchell-based Bayesian example; the two datasets are not pooled.

# Experimental readings from Figure 4

Source: H. M. Mitchell, D. Jovannus, I. Rosbottom, F. J. Link, N. A. Mitchell,
J. Y. Y. Heng (2023), *Process modelling of protein crystallisation: A case study
of lysozyme*, Chemical Engineering Research and Design 192, 268–279.
<https://doi.org/10.1016/j.cherd.2023.02.016>. The source article is CC BY 4.0.
The concentration experiment is attributed there to Tang et al. (2018).

`figure4_observations.csv` contains graph-read experimental markers, **not raw
laboratory data**. The red markers give 28 concentration readings; the four green
markers give D50. Curves and the model's blue/black markers are excluded. No
synthetic observations or independent copies of the same data are added.

`extract_figure4.py` extracts marker centres from the native 1350×793 image on
PDF page 8 (printed page 275), using colour and component area. Its PDF SHA256 is
`50cb6c6be26a21b83c8fe03a7324d0f4848248352895f69a0e368377d70cfc70`.
Axis calibration: pixel x=111..1244 corresponds to 0..700 min; y=16..700 to
20..4 g/L on the left and 14..0 µm on the right. Pixel centres are preserved.
Times are rounded to the nearest 5 min; image calibration resolves about
0.62 min, 0.0234 g/L and 0.0205 µm per pixel. Marker width and figure compression
introduce additional graph-reading uncertainty. Values are not precision
laboratory measurements, despite the extra digits retained for reproducibility.

The example interprets the paper's quoted 0.1 g/L and 0.5 µm as Gaussian standard
deviations, not variances. The article uses the word variance with these units;
that interpretation is an explicit assumption. Correlations and original error
models are unavailable. D50 is treated as a volume-weighted median of the
sectional distribution; the original instrument weighting needs confirmation.
The t=0 D50 drawn on the axis is not an observed diameter of an empty population
and is excluded.

Table 3 lists initial guesses/ranges for fitting; it does **not** specify Bayesian
priors. Its AG=600–1500 nm/min initial range excludes the final Table 7 AG=0.367
nm/min. The example therefore does not reuse that initial range as hard bounds.
Its chosen prior assumptions are recorded in the script and manual. Rates are
converted explicitly: AJ from particles/(m³ s) to scaled particles/(m³ min), and
AG from nm/min to µm/min.

To repeat extraction, supply the same PDF as `local_papers/Mitchell2023.pdf` and
run the script with Python packages pymupdf, numpy and scipy. These are optional
data-extraction tools; they are not Julia package dependencies. The CSV is
sufficient to run the case study without the PDF or these Python tools.
