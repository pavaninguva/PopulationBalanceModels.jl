# Minimal milling uncertainty tutorial.
# Physical definitions and one calculation; numerical helpers are linked on the case page.
# No benchmark exports or retained files are written. See the case-study page.

using PopulationBalanceModels
using LinearAlgebra
import Distributions
# Prepared daughter projection, outlet-CDF integration and checked BDF adapter.
include(joinpath(pkgdir(PopulationBalanceModels), "examples", "support", "milling_model.jl"))
using .MillingModel: MillingHazard, MillingFixedPivot, daughter_projection
include(joinpath(pkgdir(PopulationBalanceModels), "examples", "support", "milling_solver.jl"))

# Physical definitions from examples/support/milling_model.jl.
const PARAMETER_NAMES=(:delta,:alpha,:gamma,:sigma)

const MEAN=(delta=.44,alpha=8.82e-6,gamma=.34,sigma=2.10)

const SD=(delta=.07,alpha=1.01e-6,gamma=.08,sigma=.12)

parameters()=(mass_feed=.0074,rho=740.,feed_median=4.81e-5,feed_log_sd=.25,
    first_pivot=6.54e-17,last_pivot=1e-3,xref=2.23e-9,daughter_ratio=2.68e5,
    screen=1575e-6,change_time=30.,speed_before=4923.,speed_after=1500.,number_scale=1e12)

admissible(p)=all(isfinite,values(p)) && 0<p.delta<1 && p.alpha>0 && p.sigma>0

speed(p,t)=p.segment_speed==0 ? (t<p.change_time ? p.speed_before : p.speed_after) : p.segment_speed

diameter(v)=cbrt(6v/π)

function normal_interval(dist,a,b)
    a>=b && return 0.
    a>Distributions.median(dist) ? Distributions.ccdf(dist,a)-Distributions.ccdf(dist,b) :
        Distributions.cdf(dist,b)-Distributions.cdf(dist,a)
end

function grid(;bins=100,first=parameters().first_pivot,last=parameters().last_pivot)
    ratio=(last/first)^(1/(bins-1))
    Grid1D((2first/(1+ratio)).*ratio.^(0:bins);coordinate=:volume)
end

function retention(grid,p)
    d=diameter.(grid.centers);critical=(1-p.delta)*p.screen
    [x<=critical ? 0. : x>=p.screen ? 1. : (x-critical)/(p.delta*p.screen) for x in d]
end

function net_production(p,N,t)
    p.feed + speed(p,t)*(p.Q*N)
end

function outlet(p,N,t)
    produced=net_production(p,N,t)
    resolved=(1 .-p.retained).*produced
    low0=speed(p,t)*dot(p.low0,N)
    low1=speed(p,t)*dot(p.low1,N)
    (;resolved,underflow_number=low0,underflow_volume=low1)
end

function build(theta=MEAN;bins=100,tf=200.,last_pivot=parameters().last_pivot,quadrature_order=8)
    admissible(theta) || throw(NumericalFailure("milling inputs require 0<delta<1, alpha>0 and sigma>0"))
    constants=parameters();g=grid(;bins,last=last_pivot)
    (1-theta.delta)*constants.screen>=diameter(first(g.centers)) ||
        throw(NumericalFailure("the first milling pivot must lie below the freely discharged screen range; reduce the lower cutoff"))
    daughters=daughter_projection(g,theta.sigma,constants.daughter_ratio;order=quadrature_order)
    rate=MillingHazard(theta.alpha,theta.gamma,constants.xref)
    selection=theta.alpha.*(g.centers./constants.xref).^theta.gamma
    Q=(daughters.eta-Matrix{Float64}(I,bins,bins))*Diagonal(selection)
    distribution=Distributions.LogNormal(log(constants.feed_median),constants.feed_log_sd)
    fractions=[normal_interval(distribution,g.edges[i],g.edges[i+1]) for i in 1:bins]
    resolved_feed_fraction=sum(fractions)
    feed=constants.mass_feed/constants.rho .* (fractions/resolved_feed_fraction)./g.centers/constants.number_scale
    p0=merge(constants,theta,(;pivots=g.centers,edges=g.edges,selection,Q,
        low0=daughters.underflow_number.*selection,low1=daughters.underflow_volume.*selection,
        feed,segment_speed=0.,resolved_feed_fraction))
    p=merge(p0,(retained=retention(g,p0),))
    feedterm=IntegratedBirth((a,b,p,t,c)->p.feed[searchsortedfirst(p.edges,a)])
    # Classification is an algebraic removal of feed + net breakage production,
    # not an arbitrary first-order Death term. Recompute it explicitly so term
    # ordering does not determine the physical equation.
    screen=CustomTerm((du,N,g,p,t,c)->(du .-=outlet(p,N,t).resolved;nothing))
    model=PBMModel(feed=feedterm,breakage=Breakage(rate,daughters;overflow=:discard),classification=screen)
    op=compile(model,g;methods=TermMethods(by_term=(breakage=MillingFixedPivot(),)))
    balance=function(ctx,p,t)
        q=outlet(p,ctx.populations.mill,t)
        (discharged_mass=p.rho*p.number_scale*(dot(p.pivots,q.resolved)+q.underflow_volume),
         fines_mass=p.rho*p.number_scale*q.underflow_volume)
    end
    system=PBMSystem((mill=op,);bulk_rhs=balance,nonnegative_bulk=(:discharged_mass,:fines_mass))
    PBMProblem(system,(mill=zeros(bins),),(0.,tf);bulk0=(discharged_mass=0.,fines_mass=0.),p)
end

physical(z) = NamedTuple{PARAMETER_NAMES}(Tuple(
    getproperty(MEAN,k) + getproperty(SD,k)*getproperty(z,k) for k in PARAMETER_NAMES))

function tutorial_problem(; bins=40)
    times = [1.0,5.0,29.9,30.0,31.0,60.0,200.0]
    inputs = UncertainInputs((delta=Distributions.Normal(), alpha=Distributions.Normal(),
        gamma=Distributions.Normal(), sigma=Distributions.Normal());
        map=physical, valid=admissible, coordinates=:standard_normal,
        metadata=(joint_law=:independent_gaussian,))
    base = build(; bins)
    # sigma changes the daughter projection: rebuild it for each realization.
    setup = (theta,base) -> build(theta; bins)
    # Instantaneous OUTLET volume median, not the resident number median.
    d50 = SystemObservable((state,op,p,t) -> MillingModel.outlet_median(p,state.populations.mill,t))
    # Retain the constrained material-coordinate reduction; generic external
    # ForwardSolver has no reduced-state or CVODE-constraint declaration.
    solver = MillingSolver.solve
    UQProblem(base, inputs; setup, solver, times,
        outputs=(d50=d50,), output_coordinates=(d50=(units=:micrometre,
            weighting=:instantaneous_outlet_volume, interpolation=:linear_diameter_CDF),))
end

function main(; bins=40, mc_samples=16, validation_samples=8, training_samples=12,
              surrogate_draws=10_000)
    uq = tutorial_problem(; bins)
    solver = uq.solver
    nominal = solver(uq.base, uq.times) # Same callable as the uncertainty calculation.
    @show observe(uq.outputs.d50, nominal; index=findfirst(==(31.0),uq.times))
    # Demonstration budgets, not the retained 12,000-realization production run.
    mc = propagate(uq, MonteCarlo(mc_samples; seed=2018255); max_solves=mc_samples)
    validation = propagate(uq, MonteCarlo(validation_samples; seed=2018256); max_solves=validation_samples)
    box = hcat(fill(-2.0,4), fill(2.0,4)) # Training design, not a truncated input law.
    design = pce_design(uq.inputs; n=training_samples, seed=2018257, bounds=box)
    pce = fit_pce(uq, PolynomialChaos(1); design, validation, max_solves=training_samples)
    @show uq_summary(mc; covariance=false).mean.d50
    @show pce.validation.rmse pce.validation.passed
    if pce.validation.passed
        cheap = evaluate_pce(pce, MonteCarlo(surrogate_draws; seed=2018259))
        @show uq_summary(cheap; covariance=false).quantiles.d50
    else
        @warn "Refine the PCE degree/design before using its uncertainty summaries"
    end
    (; nominal, mc, pce)
end

# Optional after import PythonPlot. MC output quantiles are not errors on the mean.
function plot_results(result)
    mc = plot_uq(result.mc; output=:d50,
        xlabel="Time (s)", ylabel="Outlet volume median (micrometres)")
    # PCE analytical moments do not define quantiles; label one output SD explicitly.
    pce = plot_uq(result.pce; output=:d50, band=:sd,
        xlabel="Time (s)", ylabel="Outlet volume median (micrometres)")
    (; mc, pce)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
