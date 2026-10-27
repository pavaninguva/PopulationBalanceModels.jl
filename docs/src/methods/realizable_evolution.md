# Realizable evolution and particle removal

A vector with positive components need not be a set of moments of a nonnegative
population. For example, a population with zero mean size cannot have a positive
third moment. Necessary Hankel tests help detect invalid vectors, but do not
settle every singular truncated moment problem. Even a realizable vector can be
ineligible for a particular closure: two-node fixed QMOM requires a nonsingular
inversion, whereas rank-adaptive QMOM can represent one particle size or an empty
population. Conditioning of that inversion is a separate numerical question.

`MeasureEuler()` is an opt-in native evolution policy. It constructs positive
measures before taking moments, rather than hoping that a sufficiently small
explicit moment predictor is realizable. This matters at startup. From an empty
population, Euler nucleation followed by growth can create moments that fail
joint realizability for every positive timestep.

## Select the policy

```julia
basis = MomentSet(0:5; coordinate=:length)
model = PBMModel(
    growth=Growth((L,p,t,ctx)->0.8),
    nuclei=Nucleation((p,t,ctx)->1.7, 0.0),
)
op = compile(model, basis; closure=RankAdaptiveQMOM(3))
problem = PBMProblem(op, zeros(6), (0.0,1.0))
solution = simulate(problem; dt=0.05, method=MeasureEuler())
report = diagnostics(solution)
```

The selected term rules remain visible through `term_methods(op)`. The policy
uses the physical growth and event laws in those terms. A positive-measure
update is a different **time discretization** from Euler applied to their raw
moment residual; these have the same consistent first-order weak limit in the
supported scope. Custom weak forms need their own admissible evolution rule.

| Representation | `MeasureEuler` batch evolution | Empty startup | Absorption / size escape | Upwind axial transport |
|:--|:--|:--|:--|:--|
| `RankAdaptiveQMOM(N)` | Growth, point nucleation, death, dilution, aggregation, discrete daughter breakage | Yes | Atomic hitting/removal | Yes, without size-boundary events |
| `QMOM(N)` | Same mechanisms while full rank remains admissible | Yes for positive-growth age quadrature that produces full rank | Rejected; removal can reduce rank | Yes while selected closure remains admissible |
| `ExactClosure` | Same supported mechanisms, with explicitly supplied `reconstruction=RankAdaptiveQMOM(N)` and exactly `2N` tracked moments | Yes | Through that declared atomic reconstruction | Yes, with explicit reconstruction |
| Gamma `EQMOM` | Rejected by this policy | No guarantee from this policy | No boundary implementation | No guarantee from this policy |
| `LogNormalClosure` | Rejected by this policy | No guarantee from this policy | No boundary implementation | No guarantee from this policy |
| `DQMOM` | Rejected by this policy | Unsupported | Unsupported, including node collisions | Unsupported |
| 1D sectional | Explicitly selected upwind growth and existing positive source/removal/fixed-pivot collision rules | Yes | `AbsorbingZero` or `OpenBoundary` | Yes, with closed size boundaries |

The existing residual-based integrators and closures remain available. Selecting
this policy does not silently replace EQMOM or DQMOM with QMOM. The
Gamma inversion's higher-precision arithmetic retry does not add MeasureEuler
support or an external-ODE realizability guarantee; see the [audit](../audit.md). A fixed-rank
closure can still fail on a singular state. Select rank adaptation when rank loss
is part of the physical problem. Distributed raw-moment sources, user weak-form
extensions and inter-population interactions are outside this policy's scope.

## Positive measures, then moments

Write the closure quadrature as a nonnegative atomic measure,

```math
n^q(L)=\sum_{i=1}^{r}w_i\delta(L-L_i),\qquad
M_k=\sum_{i=1}^{r}w_iL_i^k,\qquad w_i\geq0,\quad L_i\geq0.
```

Growth maps each atom to `L_i + h*G(L_i,p,t,ctx)`. A node that crosses an explicitly
absorbing or escaping edge is removed at that edge. For nonconstant growth this
uses a frozen velocity, so the characteristic and hitting time are first-order
approximations. Constant-velocity atomic motion and hitting times are exact up
to reconstruction and event tolerances.

Point nucleation adds positive weights at ages given by Gauss–Legendre
quadrature on `[0,h]`. A newborn of age `s` has size `Lc+s*G(Lc,p,t,ctx)`. The rate
and velocity use the current admissible physical state, frozen for this source
substep. For an **initially empty** population, births at **Lc = 0**, constant
`J` and constant positive `G`, `N` age nodes integrate the startup moments
through order `2N-1` exactly, before any boundary escape:

```math
M_k(t)=\frac{JG^kt^{k+1}}{k+1}.
```

The temporary cloud may contain more nodes than the selected quadrature rank.
It is a positive integration measure, not an extra set of evolved moments or a
uniquely reconstructed continuous density. Its moments are reconstructed with
the selected closure before another physical rate is evaluated. Every tracked
moment must match; checking only the moments needed for the next rate is
insufficient. Rank adaptation uses the existing all-moment reconstruction test.

Death and dilution multiply a weight by `1-h*d_i`. Aggregation removes parent
weights and adds positive product weights. For identical parents the event
count is `h*Kii*wi^2/2` and removes two parents. Breakage removes parent events
and adds expected daughter counts. The kernel must be finite, nonnegative and
symmetric on the evaluated nodes. Daughters must conserve the declared additive
property. Positive sampled daughter quadrature is supported. Where `daughter_moment`
supplies an analytical integral, its positive daughter quadrature must agree for
every tracked moment within `reconstruction_rtol`; otherwise the update fails
and asks for higher daughter quadrature order. Without that hook the declared
positive discrete daughter law defines the approximation.

If the growth velocity vanishes or points inward at an approached boundary,
a frozen flight must satisfy `h*abs(Gi) <= cfl*distance_to_edge`. The policy checks
this bound to avoid artificial hits, for example when `G=-L` approaches zero
asymptotically. The growth law must be finite at that edge. Non-Lipschitz laws
with vanishing boundary velocity and finite hitting time are outside this event
treatment. Outgoing constant-velocity flights retain exact hitting/removal.

Each event update requires that the total parent depletion is at most
`cfl*wi`, with default `cfl=0.9`. Upwind sectional updates obey their outgoing
flux/source depletion bound. Bulk depletion is also checked. A failed bound
causes a smaller retry; rejection supplements the positive construction and is
not the mechanism used to make empty startup admissible. Nonfinite or
unreconstructable candidates are rejected with an explicit failure after the
configured rejection limit. No diagonal ridge, moment clipping, seed population
or fallback closure is inserted.

For raw moments, existing-particle growth comes first, then age-integrated births, then removal
and collision terms in their declared order. Shared bulk exchanges accompany
each physical substep. Sectional substeps follow their declared term order. User bulk dynamics uses explicit Euler after these
updates. This is a **first-order Lie composition**, including when source
quadrature happens to be exact. Refinement of the timestep, quadrature rank,
size grid and size domain answers different questions; none substitutes for the
others. Stiff bulk dynamics may require a different solver, outside this native
guarantee.

This construction uses positivity at the measure level, a central idea in
[Nguyen et al. (2016)](https://doi.org/10.1016/j.jcp.2016.08.017). Their continuous
reconstructions address disappearance fluxes; the implementation here instead
declares an atomic characteristic boundary approximation. The positive-mixture
argument for axial transport is also the low-order basis of realizability
preserving kinetic schemes, such as
[Schneider, Alldredge and Kall (2016)](https://doi.org/10.3934/krm.2016.9.193).
The package does not claim to reproduce either paper's complete high-order method.

## Physical zero and computational edges

For a continuous density on `[a,b]`, integration by parts gives

```math
\frac{dM_k}{dt}=
 k\int_a^b L^{k-1}G(L,t)n(L,t)\,dL
 +a^kG(a,t)n(a,t)-b^kG(b,t)n(b,t)
 +\text{source and collision contributions}.
```

At physical zero, the zeroth-moment boundary contribution is `G(0,t)n(0,t)`,
which is negative during dissolution. For `k>0` its contribution vanishes when
the trace is finite. Finitely many raw moments do not determine that trace.

Use `compile(...; boundary=MomentBoundary(left=:absorb))` to select atomic
hitting/removal. This is not evaluation of `n(0,t)`. For a continuous initial
population, quadrature nodes approximate its material and disappearance; the
number curve is a staircase that requires **rank refinement**, even for an
exact constant-growth characteristic. The dissolution example measures this
error separately.

For sectionals, `SizeBC(AbsorbingZero(),NoFlux())` selects absorption at a zero
left edge. The upwind trace is the first-cell density `N1/ΔL1`. It converges to
the continuum trace with size-grid refinement. A finite grid has a decaying
first-bin tail, so it does not become exactly empty at the continuum extinction
time. Positive residual populations are retained, not thresholded away.

A positive computational cutoff is different. For moments use, for example,
`MomentBoundary(lower=0.1,left=:escape,upper=4.0,right=:escape)`. For sectionals
use `OpenBoundary()`. Material at that boundary remains particulate and enters
an escape ledger. `NoFlux()` remains the sectional default; moment boundary
crossings raise an error unless explicitly selected. Collision products use the
existing `overflow=:error` default, or explicit `overflow=:discard` together
with accounting. Size escape is unrelated to reactor inlet/outlet transport.

`MeasureEuler(event_rtol=1e-10)` resolves a hit within that relative size
roundoff tolerance using the current node, its displacement and the finite edge.
This can move a numerical event by the corresponding small size/time error.
It is not a physical nucleation width. `reconstruction_rtol` checks all moments;
`RankAdaptiveQMOM(...;rtol=...)` controls that closure's inversion separately.

## Cumulative removal and dissolved material

A coupled system can append integrated historical states:

```julia
system = PBMSystem((crystals=op,);
    exchanges=(MomentExchange(:crystals,:solute,3; factor=rho_kv),),
    accounting=(CumulativeAccounting(:crystals; properties=(volume=3,)),),
    nonnegative_bulk=(:solute,))
problem = PBMProblem(system, (crystals=initial_moments,), (0.0,tf);
                     bulk0=(solute=c0,))
solution = simulate(problem; dt=dt, method=MeasureEuler())
escaped_volume = cumulative(problem, solution.u[end], :crystals,
                            :upper_escape, :volume)
removed_number = cumulative(problem, solution.u[end], :crystals, :extinction)
```

The mechanism names are `:extinction`, `:lower_escape`, `:upper_escape`,
`:aggregation_underflow`, `:aggregation_overflow`, `:breakage_underflow`,
`:breakage_overflow`, `:nucleation_underflow` and `:nucleation_overflow`.
The last two record out-of-pivot-range point births only when explicitly
discarded by `FixedPivot(overflow=:discard)`; the default remains an error.
Their material is consumed from the coupled bulk once, then recorded as
unresolved particles. Mechanisms are separated for each named population. Ledger
properties must be tracked for a moment population. Sectional collision losses use the first/last **pivot** interval, as in the
existing deposition rule, rather than the exterior grid edges. Their
properties must match the additive deposition coordinate; fixed pivot does not
conserve an unrelated nonlinear weight. For a custom additive map this equality
is checked at representative sizes, and remains a contract of that map over its
physical domain.

For a length-based crystallizer, define `rho_kv=ρ*kv` in material/length³.
`Mk` has units length^k times number/process volume, and solute is material/process
volume. For constant process volume, constant `rho_kv`, compatible mechanisms
and no external material feed, the discrete balance is

```math
c(t)+\rho k_v\left[M_{3,\mathrm{resolved}}(t)
 +V_{\mathrm{lower\ escape}}(t)+V_{\mathrm{upper\ escape}}(t)
 +V_{\mathrm{aggregation\ overflow/underflow}}(t)
 +V_{\mathrm{breakage\ overflow/underflow}}(t)
 +V_{\mathrm{nucleation\ overflow/underflow}}(t)\right]=C_{\mathrm{initial}}.
```

Multiply concentrations by the process volume to obtain total inventory.
Growth exchanges use the **same resolved change plus escaped property** as the
population update. Thus escaped particulate material is not also returned to
solute. A particle hitting zero carries zero `L³`; all its material has already
been released by shrinkage. Positive-cutoff removal carries the cutoff material
instead. For a sectional population, its resolved material is
`sum(Ni*Li^3)`. The computational face ledger uses the adjacent endpoint pivot's
`Li^3`, consistent with this discrete measure, not the geometrical edge cubed.
The zero-absorption ledger carries zero; the remaining first-bin discrete
material is released through the selected dissolution exchange. These are
finite-volume approximations to the physical balance and need grid refinement.

The escape ledger is a historical record, equivalent to an inert unresolved
reservoir for this balance. Escaped particles do **not** continue growing,
dissolving, aggregating or flowing. Continuing their kinetics requires another
modeled population. Inter-population transfer/collision accounting is currently
rejected rather than silently omitted. Automatic domain expansion is not provided.

Use `population`, `bulk` and `cumulative` to access a packed state. Adding the
two explicit nucleation-loss mechanisms enlarges accounting-enabled states;
code that hard-coded ledger offsets or total packed lengths must use these
accessors instead. Problems without accounting keep their existing layouts.

Floating-point quadrature reconstruction has a residual. Moment bulk exchange
uses the physical change of the reconstructed measure, and
`diagnostics(solution).reconstruction_defect` reports the accumulated difference
from the previously stored raw moments, by population (and axial cell). It is
not assigned to solute or a fictitious material reservoir. Account for this
numerical residual in scale-aware balance tolerances. A property from an
untracked moment cannot be used with `MomentExchange` or moment accounting;
closure inference is not evidence of its conservation.

Number is not a conserved property during these mechanisms.
`diagnostics(solution).number_changes` reports integrated physical net number
changes keyed by `(population,mechanism)`. Aggregation's negative change counts
binary events; breakage reports its net daughter gain, which need not equal its
event count. Subtract the removal ledger increments and add the reported numerical
reconstruction residual to relate initial and final resolved number. This avoids
mistaking particle production or extinction for a conservation error.

For sectional residual calculations, `accounting_rates(problem,u,p,t)` returns
the instantaneous selected loss rates. Its cumulative states use the same RK
stages as population and bulk. Atomic removal is an event increment and has no
ordinary instantaneous density-trace rate. Use the moment ledger increments,
not a fabricated `n(0,t)`.

## Axial transport and diagnostics

For the supported axial raw-moment system, every moment and compatible bulk
concentration has the same face flux,

```math
F_{k,i+1/2}=v M_{k,i}
 -D\frac{M_{k,i+1}-M_{k,i}}{z_{i+1}-z_i},\qquad
M_{k,i}^{new}=M_{k,i}+\frac{h}{\Delta z_i}
 (F_{k,i-1/2}-F_{k,i+1/2}),\quad v\geq0.
```

Choose `axial_problem(...;axial_scheme=Upwind())` and `MeasureEuler()`. On a
nonuniform grid the outgoing depletion coefficient in cell `i` is the sum of
`v/Δzi` where that outgoing face is open, and `D/(Δzi*dface)` for each diffusive
neighbor. Require `h*coefficient <= cfl`. Compatible inlet moments are another
nonnegative measure. The update is therefore a nonnegative **common** combination
of entire moment vectors; joint realizability follows without independent
moment limiting. Boundary faces follow the existing periodic, closed or
Danckwerts/zero-gradient-outlet contract. Advection is first order in space;
centered dispersion is conservative, with order depending on mesh regularity.

Mixtures must still be admissible for the selected closure. Fixed-rank QMOM
can fail on singular mixtures; rank-adaptive QMOM handles low rank only when all
tracked moments reconstruct. High-order axial moment reconstructions are
explicitly rejected by this guarantee. Axial extinction and internal-size escape
are also excluded by this policy; do not interpret batch event support as an
axial event implementation.

Ledger histories remain in their reactor cells and have zero axial face flux.
Reactor inlet/outlet fluxes stay separate:
`diagnostics(solution).reactor_flux_integrals.inlet` and `.outlet` integrate the
actual accepted transport fluxes, per unit cross-sectional area. On a constant
area reactor, compare them with `state*axial_grid.widths`; multiplying by area
gives total inventories. They are not losses through an internal-size boundary.

The diagnostics also report `policy`, actual `order`, `accepted_steps`,
`rejected_steps`, `admissibility_failures`, `accepted_state_checks`, active rank at
saved times, and numerical tolerances. A successful run is not a quadrature,
spatial or domain convergence certificate.

The guarantee belongs to this native policy. External adaptive ODE solvers do
not inherit its positive updates or event handling. `odeproblem` rejects moment
boundary events; sectional residual absorption and cumulative flux states remain
available without a solver-independent positivity guarantee. Automatic or
adjoint differentiation through `MeasureEuler`, rank changes and extinction
events is unsupported. Verified residual-based differentiable paths remain
available for compatible event-free models.

Run `examples/realizable_startup.jl`, `examples/realizable_dissolution.jl` and
`examples/realizable_axial_transport.jl` for executable checks, including analytical
extinction, quadrature-rank refinement and the shared-solute balance.
