# 2. Lipid nanoparticle formation

## Population balance

The second study targets the model curves in Figure 4 of
[Shin et al. (2025)](https://web.mit.edu/braatzgroup/Shin_ChemEngJ_2025.pdf).
Let ``f(L,t)`` be the number density of particles of diameter ``L`` at time ``t``.
Nucleation creates new particles, growth changes their diameters, and
coalescence combines two particles:

```math
\frac{\partial f(L,t)}{\partial t}
+\frac{\partial\!\left[G(L,t)f(L,t)\right]}{\partial L}
=B_n(L,t)+B_c(L,t)-D_c(L,t).
```

The loss ``D_c`` is defined as a positive magnitude, so it is subtracted. The
main paper prints a plus sign in its PBE alongside a positive loss integral;
the implementation uses physical parent removal.

## Initial conditions and time units

The initial particle population is zero:

```math
f(L,0)=0\qquad\text{for every modeled diameter }L.
```

This does not mean zero dissolved lipid. For the default organic feed of
10 mg/mL and flow-rate ratio 3, instantaneous mixing gives

```math
c(0)=\frac{10}{1+3}=2.5\ \mathrm{kg\,m^{-3}},\qquad
w=\frac{3}{1+3}=0.75.
```

The same concentration value in mg/mL equals its value in kg/m³. The Figure 4
setup uses instantaneous mixing (`tau=0`); the delayed-mixing model is a different
operating condition used later in the paper.

Figure 4(a) shows distributions at 1, 40, 100, 170, and 4320 **minutes**. The
solver uses **seconds**, so the last output is at 259200 s. Figure 4(b) covers
both the rapid formation stage and long incubation. A millisecond-only example
cannot represent that comparison.

## Nucleation and growth laws

Define the supersaturation as ``S(t)=c(t)/C_\infty(w,T)``. The reported
classical-nucleation laws are

```math
L_c(t)=\frac{4\gamma V_m}{k_BT\ln S(t)},
```

```math
J(t)=A_n\exp\!\left[-\frac{16\pi\gamma^3V_m^2}
{3(k_BT)^3[\ln S(t)]^2}\right],\qquad S>1.
```

For ``S\le1`` the nucleation rate is zero. ``V_m`` is volume per molecule,
consistent with the Boltzmann constant ``k_B``. The source is a Gaussian about
the critical diameter. For a normalized total-rate convention,

```math
B_n(L,t)=\frac{J(t)}{\sqrt{2\pi}\,\sigma_L}
\exp\!\left[-\frac{(L-L_c(t))^2}{2\sigma_L^2}\right].
```

``\sigma_L`` is the width of the diameter distribution, whereas ``\gamma`` is
interfacial energy. The printed Gaussian omits its normalization factor and
reuses the symbol for these two quantities. The code makes that choice explicit
through `nucleation_normalization`; a finite-domain total-rate source and an
unnormalized density amplitude are not interchangeable.

Growth follows

```math
G(L,t)=k_gL C_\infty(w,T)\left[S(t)-S^*(L,w,T)\right].
```

The size-dependent equilibrium ratio ``S^*`` permits dissolution when it exceeds
the current supersaturation. It is a material closure, not a numerical method.

The reported kinetic values are ``A_n=2.0\times10^{22}`` m⁻³s⁻¹,
``k_g=11.88`` m³kg⁻¹s⁻¹, ``k_{\mathrm{PEG}}=6.5\times10^{-6}``, and
interfacial energy approximately 0.010 N/m.

## Coalescence and conservation

For diameter coordinates, define
``\ell=(L^3-M^3)^{1/3}``. The continuum birth and loss terms are

```math
B_c(L,t)=\frac{L^2}{2}\int_0^L
\frac{K(\ell,M,t)f(\ell,t)f(M,t)}{\ell^2}\,dM,
```

```math
D_c(L,t)=f(L,t)\int_0^\infty K(L,M,t)f(M,t)\,dM.
```

The Brownian kernel with attachment efficiency is

```math
K(L,M,t)=k_{\mathrm{PEG}}\alpha_{\mathrm{base}}(L,M,w,T)
\frac{2k_BT(L+M)^2}{3\eta(w,T)LM}.
```

The supplied attachment function excludes ``k_{\mathrm{PEG}}``. Fixed-pivot event
deposition uses the additive property ``L^3``, producing a diameter
``(L_1^3+L_2^3)^{1/3}``. This conserves particle volume rather than the sum of
diameters. It does not consume free lipid a second time during coalescence.

Free lipid is coupled to the actual resolved growth and nucleation contributions:

```math
\frac{dc(t)}{dt}=-\rho_p\frac{\pi}{6}
\left[\left.\frac{dM_3}{dt}\right|_{\mathrm{growth}}
+\left.\frac{dM_3}{dt}\right|_{\mathrm{nucleation}}\right],
\qquad M_3(t)=\int_0^\infty L^3 f(L,t)\,dL.
```

Thus the fixed-volume invariant is ``c+\rho_p\pi M_3/6``, provided no products
are discarded. This is the recipe's explicit free-lipid closure. If delayed
mixing is selected, particle and dissolved concentrations also require the
same volume-dilution hazard; it is zero for this instantaneous-mixing setup.

## Implement the model in code

The reusable implementation is `examples/papers/cej2025_lnp.jl`. Its mechanisms
are declared independently:

```julia
source = GaussianNucleation(rate=J,
    center=(p,t,ctx)->critical_length(
        ctx.bulk.lipid/ctx.fields.Cinfinity,ctx.fields.T,p.gamma,p.Vm),
    width=(p,t,ctx)->p.nucleation_width,
    truncation=:renormalize)

model = PBMModel(
    nucleation=source,
    growth=Growth(rate=G),
    coalescence=Aggregation(beta;
        coordinate_map=AdditiveCoordinate(L->L^3,cbrt)),
)
```

The bin-integrated Gaussian creates nuclei even when a narrow peak falls between
bin centres. Growth uses the chosen face reconstruction; coalescence uses fixed
pivot. Their method choices are distinct from the temporal solver. The builder
adds `PBMSystem` and `ConservativeExchange` for the free-lipid balance and handles
the inactive source when supersaturation is at most one.

The Figure 4 adapter explicitly constructs the zero state:

```julia
include("examples/shin_lnp_reproduction.jl")
using .ShinLNPReproduction

# material is a NamedTuple containing the material functions/values described below.
grid = Grid1D(exp.(range(log(Lmin),log(Lmax);length=201));coordinate=:diameter)
problem = figure4_problem(;grid,material...)
initial_particles = population(problem,problem.u0,:lnp)
@assert all(iszero,initial_particles)
times = figure4_saveat() # includes formation times and all five Figure 4 snapshots
```

`Lmin` and `Lmax` are positive diameter limits in meters. `material` supplies
`temperature`, `viscosity`, `solubility`, `equilibrium_ratio`,
`attachment_without_peg`, `molecular_volume`, `particle_density`,
`nucleation_width`, and `nucleation_normalization`. The feed, FRR, mixing option,
and 4320-minute horizon have Figure 4 defaults. A nonzero initial particle
array is rejected by this adapter.

The [supplement](https://ars.els-cdn.com/content/image/1-s2.0-S1385894725086267-mmc1.pdf)
supplies the DLVO energy expressions, Hamaker constant ``1.7\times10^{-22}`` J,
and surface potential 10 mV. It was checked with the main paper. Its quantity
called Debye length has an inverse-length formula; its ion-concentration and
energy conventions still need to match the attachment implementation. The
solubility endpoints/conversion, Gaussian width, physical property values,
and equilibrium closure must also be made numerical inputs. A zero initial
population fixes the initial condition but does not determine those functions.
The input file records these functions and constants for the calculation.

## Solve the full Figure 4 horizon

The paper used 200 logarithmic bins, a central growth difference, and MATLAB
ode15s. The package's current growth and collision discretizations differ;
report their convergence before claiming a numerical reproduction. A fixed
microsecond SSPRK step over 4320 minutes is impractical. The exporter therefore
accepts an external adaptive solver, for example in a separate working
environment containing SciMLBase, OrdinaryDiffEqRosenbrock, and ADTypes:

```julia
using SciMLBase, OrdinaryDiffEqRosenbrock
using ADTypes: AutoFiniteDiff

function solve_for_outputs(problem,times)
    result = SciMLBase.solve(odeproblem(problem),Rodas5P(autodiff=AutoFiniteDiff());
        saveat=times,save_everystep=false,reltol=1e-7,abstol=1e-10)
    SciMLBase.successful_retcode(result) || error("Figure 4 integration failed")
    result
end

solution = run_case(problem;saveat=times,solver=solve_for_outputs,
    provenance="Material inputs and conventions: ...; target: Figure 4 model curves",
    output="docs/src/assets/shin_case")
```

Finite-difference Jacobians avoid assuming automatic-differentiation support
for the recipe callbacks. The scalar tolerance is a starting setting, not a demonstrated choice for every
material parameterization. The exporter checks output times, finite values,
and substantial negative concentrations; it does not clip the state. Repeat
with tighter tolerances and refined diameter limits/bins before interpreting
the intensity-weighted tail.

An input Julia file can return
`(inputs=material_and_grid,solver=solve_for_outputs,provenance="...")`.
The driver builds the problem and uses the Figure 4 output schedule:

```sh
julia --project=/path/to/working_environment examples/shin_lnp_reproduction.jl /path/to/shin_inputs.jl
python docs/benchmarks/plot_case_studies.py lnp
```

## Recreate the model panels

The reported intensity weighting and mean are

```math
I(L,t)=L^6f(L,t),\qquad
\overline L_{\mathrm{paper}}(t)=
\frac{\int_0^\infty L^7f(L,t)\,dL}{\int_0^\infty L^6f(L,t)\,dL}.
```

The mean is undefined at the empty initial state and is omitted from that point
on the log-time plot. The exporter writes both area-normalized intensity density
per meter and unit-peak-scaled density. The five PSD panels use peak scaling to
match Figure 4(a)'s displayed vertical scale; the mean-time panel uses minutes
and nanometers. Bin widths must be included before treating logarithmic-bin
numbers as a density.

No experimental data are needed to plot the model-only panels requested here.
Agreement with the publication's black curves still needs a run with the stated
material closures and conventions. A millisecond-only calculation does not cover the Figure 4 incubation horizon.

```@eval
using Markdown, PopulationBalanceModels
folder=joinpath(pkgdir(PopulationBalanceModels),"docs","src","assets","shin_case")
ready=all(isfile(joinpath(folder,f)) for f in ("figure4a_model.png","figure4b_model.png"))
Markdown.parse(ready ? "![Figure 4 layout: calculated intensity PSDs](../assets/shin_case/figure4a_model.png)\n\n![Calculated mean diameter](../assets/shin_case/figure4b_model.png)" : "Run the full-horizon calculation and plotting commands above, then rebuild the manual to display the panels.")
```
