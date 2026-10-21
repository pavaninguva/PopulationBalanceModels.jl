# Precipitation in a continuous tubular reactor

A supersaturated feed enters a tube that initially contains no solute or
particles. Nuclei form as the feed advances, crystals grow, and precipitation
reduces the dissolved concentration. Axial dispersion spreads the residence-time
distribution. The outlet PSD therefore depends on both reaction kinetics and
mixing along the tube; a well-mixed batch model cannot describe its startup.

This calculation uses Case 7A of [Zhang et al. (2024)](https://doi.org/10.1016/j.compchemeng.2024.108612).
It targets the transient fields and outlet distributions in Figure 9(a,c) and
the error-versus-runtime comparison in Figure 10(a,b). It uses this package's
upwind and MUSCL methods, rather than claiming to reproduce CADET's WENO35
algorithm or its computational times. Case 7B is not included: the printed
lognormal centre needs an unambiguous conversion to SI units before it can be
used as a reproduction input.

## Model and boundaries

Let ``n(x,z,t)`` be number density per reactor-fluid volume per particle length,
``c(z,t)`` dissolved concentration, ``x`` particle length and ``z`` axial position.
The particles obey

```math
\frac{\partial n(x,z,t)}{\partial t}
+v_{\mathrm{ax}}\frac{\partial n(x,z,t)}{\partial z}
+\frac{\partial[G(z,t)n(x,z,t)]}{\partial x}
=D_{\mathrm{ax}}\frac{\partial^2n(x,z,t)}{\partial z^2}.
```

Nucleation supplies the size-boundary flux

```math
G(z,t)n(x_c,z,t)=B_0(z,t),\qquad
B_0=k_p s^{u}+k_b s^{b}M,\qquad
G=k_gs,
```

where

```math
s(z,t)=\max\!\left(\frac{c(z,t)-c_{\mathrm{eq}}}{c_{\mathrm{eq}}},0\right),
\qquad
M(z,t)=\rho k_v\int_{x_c}^{\infty}x^3n(x,z,t)\,dx.
```

Primary nucleation starts the population. Secondary nucleation is proportional
to the crystal mass concentration ``M``. There is no aggregation, breakage or
growth-rate dispersion in this case. The solute balance is

```math
\frac{\partial c(z,t)}{\partial t}
+v_{\mathrm{ax}}\frac{\partial c(z,t)}{\partial z}
=D_{\mathrm{ax}}\frac{\partial^2c(z,t)}{\partial z^2}
-\rho k_v\left[B_0(z,t)x_c^3+
3\int_{x_c}^{\infty}G(z,t)x^2n(x,z,t)\,dx\right].
```

Both balances use Danckwerts inflow and a zero-gradient outlet:

```math
v_{\mathrm{ax}}n(x,0,t)-D_{\mathrm{ax}}\frac{\partial n(x,0,t)}{\partial z}=0,
\qquad \frac{\partial n(x,L,t)}{\partial z}=0,
```

```math
v_{\mathrm{ax}}c(0,t)-D_{\mathrm{ax}}\frac{\partial c(0,t)}{\partial z}
=v_{\mathrm{ax}}c_{\mathrm{feed}},
\qquad \frac{\partial c(L,t)}{\partial z}=0.
```

Initially ``n(x,z,0)=0`` and ``c(z,0)=0``. Concentrations are based on total
process volume. The axial velocity is ``Q/(A\varepsilon)``, not ``Q/A``;
porosity changes the residence time substantially.

| Input | Value and units |
|:--|:--|
| ``c_{\mathrm{feed}}``, ``c_{\mathrm{eq}}`` | 9, 0.4 kg m⁻³ |
| ``\rho``, ``k_v`` | 1200 kg m⁻³, 0.524 |
| ``k_p``, ``u`` | 5 m⁻³ s⁻¹, 10 |
| ``k_b``, ``b`` | 4×10⁸ kg⁻¹ s⁻¹, 2 |
| ``k_g`` | 7×10⁻⁶ m s⁻¹ |
| ``x_c``, ``x_{\max}`` | 1, 900 µm |
| ``Q``, ``A``, ``\varepsilon`` | 1.67×10⁻⁷ m³ s⁻¹, 1.46×10⁻⁴ m², 0.21 |
| ``L``, ``D_{\mathrm{ax}}`` | 0.47 m, 4.2×10⁻⁵ m² s⁻¹ |

These are [supporting-information Table S8](https://ars.els-cdn.com/content/image/1-s2.0-S0098135424000309-mmc1.pdf) inputs. The authors' public
[verification script](https://github.com/cadet/CADET-Verification/blob/090606762fc657510a9e31c547890bf9a17ffb59/src/benchmark_models/settings_crystallization.py)
confirms the empty initial state and porosity convention. Its flow uses exactly
10 mL/min; this script retains the SI table's rounded ``Q``. The coefficients
have the units above because ``s`` is dimensionless.

## Assemble local precipitation, then add axial transport

The kinetic callbacks read the local dissolved concentration. They do not need
to know how the reactor is divided into cells:

```julia
supersaturation(p,ctx) = max(ctx.bulk.c/p.ceq-1,zero(ctx.bulk.c))
growth(x,p,t,ctx) = p.kg*supersaturation(p,ctx)
function nucleation(p,t,ctx)
    s = supersaturation(p,ctx)
    M = 1e12*p.rho*p.kv*moment(ctx.grid,ctx.N,3)
    (p.kp*s^p.u + p.kb*s^p.b*M)/1e12
end
model = PBMModel(
    growth=Growth(rate=growth),
    primary_secondary=Nucleation(rate=nucleation,size=first(grid.edges)),
)
operator = compile(model,grid;methods=TermMethods(
    growth=MUSCL(),by_term=(primary_secondary=BoundaryInjection(:left),)))
system = PBMSystem((crystals=operator,);nonnegative_bulk=(:c,),
    exchanges=(ConservativeExchange(:crystals,:c,
        (x,p,t,ctx)->1e12*p.rho*p.kv*x^3),))
local_problem = PBMProblem(system,(crystals=zeros(length(grid)),),(0.,200.);
    bulk0=(c=0.,),p)
reactor = AxialDispersion(AxialGrid(range(0.,p.L;length=nz+1));
    velocity=p.Q/(p.area*p.porosity),dispersion=p.Dax,
    inlet=(p,t)->(populations=(crystals=zeros(length(grid)),),
                  bulk=(c=p.cfeed,)))
problem = axial_problem(local_problem,reactor;axial_scheme=MUSCL())
```

For numerical scaling the internal particle counts are divided by ``10^{12}``.
The rate divides by that scale, the mass callback multiplies by it, and exported
densities restore physical units. This changes no physical coefficient.

`ConservativeExchange` subtracts the **resolved discrete** crystal-mass increase
from solute. This balances material exactly cell by cell before axial fluxes are
added. At finite resolution it differs from separately quadraturing the continuum
sink above, including the mass assigned to the first bin. That difference must
shrink under size refinement; it is not evidence of a different physical source.
The upper size boundary is closed. Check its occupancy before treating the
finite-domain calculation as an approximation on an infinite size domain.

## Startup, outlet and numerical performance

![Particle-size fields at 80 and 200 seconds](../assets/continuous_precipitation/fields.png)

![Outlet number and volume distributions](../assets/continuous_precipitation/outlets.png)

The fields distinguish the physical axial coordinate from particle size. The
outlet plots use number density and a separately normalized volume-weighted
density. Neither is a histogram of sampled particles.

![Outlet error against elapsed computation time](../assets/continuous_precipitation/performance.png)

Errors integrate the absolute outlet-density difference on the common partition
of each grid and a finer MUSCL reference, then divide by the reference's number
integral. The reference is numerical, not exact. The committed figure uses a
small documentation budget; it demonstrates the comparison workflow, not a
completed reproduction of all Figure 10 convergence curves. Elapsed times exclude
a warmup but remain machine dependent. [verification.txt](../assets/continuous_precipitation/verification.txt) records the reference
mesh and a step-halving check. Reference spatial error remains to be assessed
with a larger run.

```sh
julia --project=. examples/continuous_precipitation.jl
# Expensive size/axial refinement and the SI-scale reference:
julia --project=. examples/continuous_precipitation.jl --full
python docs/benchmarks/plot_new_case_studies.py precipitation --usetex
```

The driver selects an explicit step from size advection, axial advection and
dispersion, and local solute depletion. It does not clip negative states. A
fine logarithmic size mesh can make this expensive; the same assembled residual
is available through `odeproblem` for a suitable implicit solver.

## Complete runnable script

The code below is read from the example source during the documentation build.
Run it from the repository root using the environment described above.

### `examples/continuous_precipitation.jl`

```@eval
using Markdown, PopulationBalanceModels
Markdown.MD([Markdown.Code("julia",read(joinpath(pkgdir(PopulationBalanceModels),"examples","continuous_precipitation.jl"),String))])
```
