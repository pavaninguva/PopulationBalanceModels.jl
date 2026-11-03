| Parameter | Meaning and units | Prior | Model |
|:--|:--|:--|:--|
| ``\gamma`` | Effective interfacial energy (mJ/m²) | 0.64 (fixed) | All |
| ``\ln A_J`` | Natural log of the numerical nucleation prefactor in particles/(L min) (dimensionless) | Uniform(-2.8134, 97.187) | All |
| ``A_G`` | Mean-growth prefactor (μm/min) | log-uniform AG on [1e-05, 1.5] | All |
| ``g`` | Supersaturation growth exponent (dimensionless) | Uniform(1, 5) | All |
| ``D_0`` | Constant internal-coordinate diffusivity D0 (μm²/min) | Exponential(scale/mean 0.01 μm²/min) | Constant GRD |
| ``k_D`` | Growth-dependent GRD coefficient in D_G = k_D G (μm) | Exponential(scale/mean 0.1 μm) | Growth-dependent GRD |
