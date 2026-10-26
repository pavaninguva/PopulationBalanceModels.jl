| Parameter | Meaning and units | Prior | Model |
|:--|:--|:--|:--|
| `logAJ` | Natural log nucleation prefactor AJ in particles/(m³ s) (dimensionless) | Uniform(0, 100) | Both |
| `gamma` | Effective interfacial energy (mJ/m²) | LogNormal(log(0.64), 0.7) | Both |
| `AG` | Mean-growth prefactor (nm/min) | log-uniform AG on [0.01, 1500] | Both |
| `g` | Supersaturation growth exponent (dimensionless) | Uniform(1, 3) | Both |
| `grd_strength` | Random-fluctuation growth dispersion strength (min) | Exponential(scale/mean 10 min) | CNT + GRD only |
