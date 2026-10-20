# 1. Batch cooling crystallization

## Population balance and concentration balance

Example 2 of [Gunawan, Fusman, and Braatz (2004)](https://web.mit.edu/braatzgroup/80_High_resolution_algorithms_for_multidimensional_population_balance_equations.pdf)
considers a seeded batch of potassium-nitrate crystals. Let ``f(L,t)`` be the
number density at crystal length ``L`` and time ``t``. Crystals move to larger
sizes as they grow, and secondary nucleation introduces particles at zero size:

```math
\frac{\partial f(L,t)}{\partial t}
+\frac{\partial\!\left[G(L,c(t),T(t))f(L,t)\right]}{\partial L}
=B_0(c(t),T(t))\,\delta(L).
```

Here ``\delta(L)`` denotes creation at a single size, not a finite-width source.
The growth velocity and total nucleation rate are

```math
G(L,c,T)=k_g\left[\frac{c-c_{\mathrm{sat}}(T)}{c_{\mathrm{sat}}(T)}\right]^g
(1+0.1L),
```

```math
B_0(c,T)=k_b\left[\int_0^\infty L^3 f(L,t)\,dL\right]
\left[\frac{c-c_{\mathrm{sat}}(T)}{c_{\mathrm{sat}}(T)}\right]^b.
```

The solution concentration ``c(t)`` is a dynamical state. Growth consumes solute,
so it must be integrated with the population balance:

```math
\frac{dc(t)}{dt}=-3\rho_c\int_0^\infty
G(L,c(t),T(t))L^2 f(L,t)\,dL.
```

The model uses length in micrometers, time in seconds, and concentration in
grams of solute per gram of water. Its crystal-volume convention is
``\int L^3f\,dL``; a spherical ``\pi/6`` factor is not added.

| Parameter | Value |
|:--|--:|
| ``k_g`` | 116 |
| ``g`` | 1.32 |
| ``k_b`` | ``4.64\times10^{-7}`` |
| ``b`` | 1.78 |
| ``\rho_c`` | ``2.11\times10^{-12}`` g/µm³ |

Temperature is prescribed rather than solved as another ODE:

```math
T(t)=32-4\left(1-e^{-t/18600}\right),
\qquad
c_{\mathrm{sat}}(T)=1.721\times10^{-4}T^2-5.88\times10^{-3}T+0.1286.
```

## Initial and boundary conditions

Equation 34 specifies

```math
f(L,0)=
\begin{cases}
-3.48\times10^{-4}L^2+0.136L-13.3,
  &180.5\le L\le210.5\ \mathrm{\mu m},\\
0,&\text{otherwise}.
\end{cases}
```

The implementation retains this expression verbatim:

```@example gunawan_seed
using PopulationBalanceModels
include(joinpath(pkgdir(PopulationBalanceModels),"examples","papers","gunawan2004_case2.jl"))
using .Gunawan2004Case2

reported_seed_density(195.5)
```

An input check is necessary: the displayed coefficients give a negative density
at 195.5 µm and even at the polynomial's maximum inside its support. Restricting
the support does not remove that negative value. The package therefore rejects
that literal initial population. It does not clip it, reverse its sign, or replace
it with another polynomial. A faithful numerical reproduction needs this
input discrepancy resolved; the runner validates the supplied density before solving.

The runner records ``c(0)`` explicitly in the same concentration units as the
balance. A saturated start would be ``c(0)=c_{\mathrm{sat}}(32)=0.1166704``;
that choice must be stated in the input provenance.

Nucleation is imposed as the inward number flux

```math
G(0,c(t),T(t))f(0,t)=B_0(c(t),T(t)).
```

The finite size domain is 0–1600 µm with a closed upper boundary. Check that its
last bins remain insignificant before interpreting a closed-domain calculation.
The recipe sets the growth and nucleation rates to zero when the solution is
not supersaturated; the printed fractional-power laws do not define dissolution.

## Define the rates in Julia

```julia
function supersaturation(p,t,ctx)
    max(ctx.bulk.c / solubility(ctx.fields.T) - 1, 0.0)
end

function G(L,p,t,ctx)
    p.kg * supersaturation(p,t,ctx)^p.g * (1 + 0.1L)
end

function J(p,t,ctx)
    crystal_volume = moment(ctx.system,:crystals,3)
    p.kb * crystal_volume * supersaturation(p,t,ctx)^p.b
end

model = PBMModel(
    growth=Growth(rate=G),
    nucleation=Nucleation(rate=J,size=0.0),
)
```

`Growth.rate` returns a length velocity. `Nucleation.rate` returns a total
number creation rate and therefore has no length argument. Both read the
concentration and temperature at the current integration stage.

## Discretize and couple to the concentration ODE

```julia
inputs = include("/path/to/gunawan_inputs.jl")
c0 = inputs.c0
grid = Grid1D(0.0:0.5:1600.0;coordinate=:length)
initial_bin_numbers = initial_counts(grid,inputs.initial_density)
parameters = (kg=116.0,g=1.32,kb=4.64e-7,b=1.78,rho=2.11e-12)

methods = TermMethods(
    growth=MUSCL(),
    by_term=(nucleation=BoundaryInjection(:left),),
)
operator = compile(model,grid;methods,bc=SizeBC(NoFlux(),NoFlux()))
system = PBMSystem((crystals=operator,);
    fields=(p,t)->(T=temperature(t),),
    nonnegative_bulk=(:c,),
    exchanges=(ConservativeExchange(:crystals,:c,
        (L,p,t,ctx)->p.rho*L^3;
        mechanisms=(:growth,:nucleation)),))
problem = PBMProblem(system,(crystals=initial_bin_numbers,),
    (0.0,1000.0);bulk0=(c=c0,),p=parameters)
solution = simulate(problem;method=SSPRK33(),dt=0.01,saveat=0.0:10.0:1000.0)
```

The supplied builder performs these steps in
`examples/papers/gunawan2004_case2.jl`. Growth uses conservative face fluxes;
boundary injection adds the nucleation rate without division by a possibly
zero growth velocity. The shared-state solver advances concentration and bin
numbers simultaneously. `ConservativeExchange` uses their actual discrete
material change, including the small finite-pivot contribution of zero-size
creation. This preserves a discrete balance; it does not eliminate continuum
moment error at finite cell width.

## Run and compare locally

Create a Julia input file returning `(initial_density=f0,c0=c0,provenance="...")`
with the resolved initial density and its source. The other benchmark keywords
can also be included. Then run

```sh
julia --project=. examples/cooling_crystallization_benchmark.jl /path/to/gunawan_inputs.jl
python docs/benchmarks/plot_case_studies.py cooling
```

The default moderate grid has 3200 bins (0.5 µm) and a 0.01 s time step, with
output at 1000 s. Add a 32000-bin run to approach the paper's 0.05 µm reference
grid; it is substantially more expensive. The current code compares upwind and
MUSCL, not the paper's HR1 and Lax–Wendroff implementations.

For ``\mu_k(t)=\int_0^\infty L^k f(L,t)\,dL``, an independent reference is

```math
\frac{d\mu_0}{dt}=k_b\mu_3s^b,
\qquad
\frac{d\mu_k}{dt}=k\,k_gs^g(\mu_{k-1}+0.1\mu_k),\quad k=1,2,3,
\qquad
\frac{dc}{dt}=-\rho_c\frac{d\mu_3}{dt}.
```

The runner checks these continuum moment equations using independent RK4 step
halving. Its CSVs contain moment/concentration histories, final size densities,
and the discrete material-balance drift. Plotting produces four panels:
concentration, moment error, final size distribution, and material drift.

```@eval
using Markdown, PopulationBalanceModels
file=joinpath(pkgdir(PopulationBalanceModels),"docs","src","assets","gunawan_example2","coupled_solution.png")
Markdown.parse(isfile(file) ? "![Cooling crystallization calculation](../assets/gunawan_example2/coupled_solution.png)" : "Run the resolved-input calculation and plotting commands above, then rebuild the manual to display the panels.")
```
