# Isothermal 100 mL batch

Original observations from Tang et al., *Journal of Crystal Growth* (2018),
DOI [10.1016/j.jcrysgro.2018.06.023](https://doi.org/10.1016/j.jcrysgro.2018.06.023),
Figure 11(d) and the isothermal inverted-triangle curves of Figure 12(a).
The supplied graph readings are unchanged. These are not Mitchell's replot,
raw laboratory records or synthetic measurements.

`concentration.csv` contains the fixed initial marker, 11 calibration and four
validation measurements. The initial model time is zero; all other graph-read
times remain unrounded. `psd_cdf.csv` contains conditional **number** CDFs at
74, 180, 420 and 660 min, for two alternative conditioning windows. The first
three times calibrate; 660 min validates. The 0.5 μm reconstruction grid is not
the instrument's bin grid and its points are not independent observations.
Use `psd_summary.csv` for diagnostic quantiles only.

`config/` preserves the historical handover specification and output contract.
The active CNT/CNT+GRD models, priors and fixed-error likelihood are defined in
`examples/isothermal_protein_crystallization.jl`; the historical configuration
is not used as its runtime input.
`provenance/` preserves source hashes, pixel calibrations, measurement methods,
source coordinates, selected trace snapshots and source conflicts. No source
PDF or source raster is included. The original handover ZIP hash and integration
revision are recorded in `provenance/integration.json`.

The microscopy measurements describe number-weighted equivalent diameters.
The separate volume-weighted laser-diffraction measurement is not used.
Nominal solubility 2.16 g/L, density 1240 kg/m³, shape factor 0.81 and an identity
equivalent-diameter/length mapping are explicit modelling assumptions. Salt
basis and stirring-speed conflicts remain unresolved. The final concentration
is not assumed to be equilibrium.
