# Precipitation in a continuous tubular reactor

A supersaturated feed enters a tube that initially contains no solute or
particles. Nuclei form as the feed advances, crystals grow, and precipitation
reduces the dissolved concentration. Axial dispersion spreads the residence-time
distribution. The outlet PSD therefore depends on both reaction kinetics and
mixing along the tube; a well-mixed batch model cannot describe its startup.

This calculation uses Case 7A of [Zhang et al. (2024)](https://doi.org/10.1016/j.compchemeng.2024.108612).
It targets the transient fields and outlet distributions in Figure 9(a,c) and
the error-versus-runtime comparison in Figure 10(a,b). It compares this package's Upwind, MUSCL and uniform-size WENO5
reconstructions. These are not CADET's WENO23/WENO35 algorithms, boundary
closures or timings. Case 7B is not included: the printed
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
growth-rate dispersion in this case. Below saturation, both rates stop; this
precipitation-only model does not redissolve existing particles. The solute balance is

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

Initially ``n(x,z,0)=0`` and ``c(z,0)=0``. Concentrations refer to local fluid volume, with no extra porosity factor
in the exchange. The axial velocity is ``Q/(A\varepsilon)``, not ``Q/A``;
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
sink above, including the mass assigned to the first bin. On coarse uniform size meshes,
a 1-µm nucleus is represented at a much larger first pivot; a higher-order
growth reconstruction alone cannot eliminate that source/material error.
That difference must
shrink under size refinement; it is not evidence of a different physical source.
The upper size boundary is closed. Check its occupancy before treating the
finite-domain calculation as an approximation on an infinite size domain.

## Startup and the outlet distribution

![Particle-size fields at 80 and 200 seconds](../assets/continuous_precipitation/fields.png)

Each cell colour is number density per particle length and fluid volume,
``n`` in m⁻⁴; both panels use the same colour scale. The feed first raises
supersaturation near the inlet. Nucleation produces many small particles;
growth and solute depletion then reshape the population downstream. At 80 s
the reactor is still starting up. The 200-s outlet is the comparison used in
the paper; it is not an analytical steady solution.

![Outlet number and volume distributions at fixed size resolution and varying axial resolution](../assets/continuous_precipitation/outlets.png)

The left panel retains absolute number density. The right weights each bin by
its centre's ``x^3`` and normalizes by total resolved volume; its integral over
size in µm is one. Large particles carry more volume, so a number peak and a
volume peak need not coincide. These are model densities, not sampled counts. The axis is logarithmic, but
density and normalization still refer to linear size, not log-size. Lines connect
bin-centre values as a visual guide.
The coloured curves keep size resolution fixed while changing axial cells;
axial numerical mixing changes the supersaturation history and hence the PSD.
The dashed line uses the finer numerical reference. The upper size boundary is
closed; final-bin occupancy is exported with each comparison to expose
truncation, rather than treating a small balance error as domain convergence.
For the logarithmic reference at 200 s, the final bin contains about
``2.9\times10^{-6}`` of resolved volume. That occupancy check is not a domain
refinement or a bound on truncation error.

## Compare two independent resolutions

Case 7 has **no analytical coupled PSD solution**. The SI's analytical DPFR
solution belongs to Case 6, a first-order reaction problem. Following Eq. 49,
we use the dimensionless normalized error

```math
E_{L^1}=
\frac{\sum_i\Delta x_i\left|n_i-n_i^{\mathrm{ref}}\right|}
     {\sum_i\Delta x_i n_i^{\mathrm{ref}}}.
```

A value 0.1 means a 10% normalized distribution difference. For different meshes,
the implementation forms the union of their size edges and integrates both
piecewise-constant densities exactly on that common partition. This preserves
bin-number integrals and differs from the paper's spline interpolation. It also
differs from comparing densities only at common bin centres.

The executed documentation budget uses the Cartesian product
``N_x=(12,24)`` and ``N_z=(16,32,48)``: six points per reconstruction pairing.
Here ``N_z`` is the paper's ``N_{\mathrm{col}}``. Axial meshes are uniform.
Logarithmic size meshes compare Upwind/Upwind and
MUSCL/MUSCL (size/axial). Uniform size meshes add Upwind/MUSCL and WENO5/MUSCL,
so the size schemes can be compared with axial MUSCL held fixed. WENO5 is
ineligible on a logarithmic size mesh. The paper instead uses the same scheme
in both coordinates, including its own CADET boundary treatments.

![Measured normalized error versus solve time, with nondominated fronts](../assets/continuous_precipitation/performance.png)

Every marker is a measured configuration. A line connects only the
**nondominated** points within one pairing: no other measured point of that
pairing is both cheaper and at least as accurate, with a strict improvement in
one quantity. These lines are finite-set fronts, not continuous optimal curves.
The two panels use separate references for their size-grid families.
Those references themselves differ by about 20.7% on a common partition,
normalized by the logarithmic reference's number. Thus an error in one panel
and an error in the other are distances to different numerical approximations.
[Reference outlet densities](../assets/continuous_precipitation/reference_outlets.csv)
allow these comparisons to be checked without repeating either reference solve.

![Size refinement at separately fixed axial resolutions](../assets/continuous_precipitation/coordinate_refinement.png)

Each line keeps ``N_z`` fixed and changes ``N_x``. A flattening curve can mean
that axial error is limiting; adding size bins alone cannot remove it.
Conversely, changing ``N_z`` at fixed ``N_x`` exposes the size-resolution limit.
Small differences near the reference's own uncertainty are not a measured
asymptotic order. This small study separates the coordinates but does not
reproduce all 25 points per scheme in the paper's Figures 10–12. For example,
doubling logarithmic size bins from 12 to 24 lowers the MUSCL/MUSCL difference
from 0.602 to 0.488 at 16 axial cells, but from 0.328 to 0.140 at 48 axial cells.
Resolving the axial history makes size refinement more useful. On the finest
uniform comparison, WENO5/MUSCL gives 0.369 versus MUSCL/MUSCL's 0.287. These
finite-resolution differences include source deposition, boundaries and reference
uncertainty; they do not establish the schemes' smooth-solution orders.

## Reference accuracy and computational cost

Each reference uses size/axial MUSCL/MUSCL with 48 size bins and 64 axial cells,
on the same physical domain as its comparisons. Time integration is SSPRK33.
At each step the driver takes 0.3 divided by the sum of size-advection,
axial-advection and axial-diffusion rates, applies a local solute-depletion
bound, and caps the step at 0.25 s. Every native stage retains its state checks;
no clipping is applied.

The table compares each reference with half its step-control fraction, fewer
size bins at fixed axial resolution, and fewer axial cells at fixed size
resolution. These differences are diagnostics, not upper error bounds.

```@eval
using Markdown, PopulationBalanceModels
folder=joinpath(pkgdir(PopulationBalanceModels),"docs","src","assets","continuous_precipitation")
rows=split.(readlines(joinpath(folder,"reference_checks.csv"))[2:end],',')
body=["| Size mesh | Check | Size × axial | Normalized difference |", "|:--|:--|--:|--:|"]
for r in rows
    push!(body,"| $(r[1]) | $(replace(r[2],"_"=>" ")) | $(r[3]) × $(r[4]) | $(round(parse(Float64,r[5]);sigdigits=3)) |")
end
Markdown.parse(join(body,"\n"))
```

The temporal differences are below ``1.5\times10^{-11}``, while the size-grid
changes are 6.4% (logarithmic) and 14.3% (uniform), and the axial changes are
about 10% for both. Spatial reference uncertainty is therefore substantial;
temporal agreement alone would give a misleading impression of accuracy. Finer references are needed to rank differences below that
scale. [reference_checks.csv](../assets/continuous_precipitation/reference_checks.csv)
retains the individual check costs and steps. The cross-family row compares
already computed states; its solve cost is undefined (`NaN`) and steps are zero.

Each timing below is **one measured solve after a separate small compilation
warmup**. Assembly, CSV export and plotting are excluded; stepping, timestep
selection, checks and requested saved states are included. References save
80 and 200 s; comparison runs save 200 s only. These are newly measured Julia
runtimes, without repeated-run uncertainty bands, and are not CADET timings.
CPU load was not controlled; read small timing differences cautiously.

```@eval
using Markdown, PopulationBalanceModels
folder=joinpath(pkgdir(PopulationBalanceModels),"docs","src","assets","continuous_precipitation")
rows=split.(readlines(joinpath(folder,"runtimes.csv"))[2:end],',')
rows=filter(r->r[1]=="reference" || (r[5]=="24" && r[6]=="48"),rows)
body=["| Role / mesh | Size / axial scheme | Size × axial | Steps | Seconds |", "|:--|:--|--:|--:|--:|"]
for r in rows
    push!(body,"| $(r[1]) / $(r[2]) | $(r[3]) / $(r[4]) | $(r[5]) × $(r[6]) | $(r[8]) | $(round(parse(Float64,r[7]);sigdigits=3)) |")
end
machine=only(filter(line->startswith(line,"Julia "),readlines(joinpath(folder,"verification.txt"))))
Markdown.parse(machine*"\n\n"*join(body,"\n"))
```

[verification.txt](../assets/continuous_precipitation/verification.txt) records
CPU, Julia version, Julia/BLAS thread counts and step controls;
[all timings and errors](../assets/continuous_precipitation/performance.csv) are
available. The [failed startup checks](../assets/continuous_precipitation/failures.csv)
are retained too. Axial WENO5 is not included in the completed comparison:
a uniform-size WENO5/axial-MUSCL run does not establish safe empty-reactor startup
for WENO5 in both directions. The separate probe retains the existing negative
bulk check and records its failure rather than relaxing it. The completed size-WENO5 runs
retain small signed bin values (the lowest exported state is about
``-1.24\times10^{-10}`` in scaled bin-number units), within the existing
``10^{-8}`` stage-check tolerance. This is not a positivity guarantee for WENO5;
all values and tolerances are retained in the exports.

```sh
julia --project=. examples/continuous_precipitation.jl
python docs/benchmarks/plot_new_case_studies.py precipitation --usetex
# Much larger, unexecuted SI mesh arrangement; writes a separate output directory:
julia --project=. -e 'include("examples/continuous_precipitation.jl"); ContinuousPrecipitation.run_benchmark(full=true, output="/tmp/precipitation-full")'
```

The larger option uses the paper's reference ``(N_x,N_{\mathrm{col}})=(250,750)``
and all 25 pairs of ``N_x=(12,25,50,100,200)`` with
``N_{\mathrm{col}}=(37,75,150,300,600)`` per scheme and size-grid family.
Additional fixed-axial-MUSCL comparisons remain labelled separately. This option
was not executed for the retained documentation-budget results. Its explicit
steps on fine logarithmic meshes may be costly; `odeproblem` exposes the same
residual for an optional implicit solver, whose accuracy must be checked anew.

## Minimal run and full reproduction

Run this block from the repository root. It constructs and solves one problem;
including a driver does not run its CSV exports.

```julia
using PopulationBalanceModels
include("examples/continuous_precipitation.jl")
problem = ContinuousPrecipitation.build(; nx=24, nz=32, mesh=:log,
    scheme=MUSCL(), axial_scheme=MUSCL())
solution = ContinuousPrecipitation.advance(problem, [80.0, 200.0])
outlet = population(problem, last(solution.u), :crystals)[:, end]
concentration = bulk(problem, last(solution.u), :c)
```

The figure-generation drivers and supporting sources are available below. Keep
the repository's relative paths; the [benchmark guide](../assets/source/plotting/README.txt)
lists plotting commands and optional environments.

- [continuous_precipitation.jl](../assets/source/examples/continuous_precipitation.jl)

- [Benchmark/export helpers](../assets/source/examples/support/precipitation_benchmark.jl).
