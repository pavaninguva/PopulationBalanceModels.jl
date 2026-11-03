# Stage 1: nominal milling; physical volume in m^3, time in s, inventory in kg.
# Use examples/environments/milling. Including this file prepares the problem
# and UQ declarations only; it never launches an ensemble or writes results.
using PopulationBalanceModels, LinearAlgebra
import Distributions
include(joinpath(@__DIR__,"..","support","milling_model.jl"))
include(joinpath(@__DIR__,"..","support","milling_solver.jl"))
using .MillingModel: MillingHazard, MillingFixedPivot, daughter_projection,
    normal_interval, diameter, retention, outlet, admissible

const PARAMETER_NAMES = (:delta,:alpha,:gamma,:sigma)
const MEAN = (delta=0.44,alpha=8.82e-6,gamma=0.34,sigma=2.10)
const SD = (delta=0.07,alpha=1.01e-6,gamma=0.08,sigma=0.12)
const MILL = (mass_feed=0.0074,rho=740.0,feed_median=4.81e-5,feed_log_sd=0.25,
    first_pivot=6.54e-17,last_pivot=1e-3,xref=2.23e-9,daughter_ratio=2.68e5,
    screen=1575e-6,change_time=30.0,speed_before=4923.0,speed_after=1500.0,number_scale=1e12)

# sigma changes the daughter projection, so every parameter realization rebuilds.
function build_mill(theta=MEAN;bins=40)
    admissible(theta) || throw(NumericalFailure("require 0<delta<1, alpha>0 and sigma>0"))
    grid = MillingModel.grid(;bins,first=MILL.first_pivot,last=MILL.last_pivot)
    (1-theta.delta)*MILL.screen>=diameter(first(grid.centers)) ||
        throw(NumericalFailure("lower pivot must lie below the freely discharged screen range"))
    daughters = daughter_projection(grid,theta.sigma,MILL.daughter_ratio;order=8)
    hazard = MillingHazard(theta.alpha,theta.gamma,MILL.xref)
    selection = theta.alpha.*(grid.centers./MILL.xref).^theta.gamma
    Q = (daughters.eta-Matrix{Float64}(I,bins,bins))*Diagonal(selection)
    # A volume-weighted volume-coordinate law; explicitly condition feed on represented bins.
    feed_law = Distributions.LogNormal(log(MILL.feed_median),MILL.feed_log_sd)
    fractions = [normal_interval(feed_law,grid.edges[i],grid.edges[i+1]) for i in 1:bins]
    resolved_feed_fraction = sum(fractions)
    feed = MILL.mass_feed/MILL.rho.*(fractions/resolved_feed_fraction)./grid.centers/MILL.number_scale
    p0 = merge(MILL,theta,(pivots=grid.centers,edges=grid.edges,selection=selection,Q=Q,
        low0=daughters.underflow_number.*selection,low1=daughters.underflow_volume.*selection,
        feed=feed,segment_speed=0.0,resolved_feed_fraction=resolved_feed_fraction))
    p = merge(p0,(retained=retention(grid,p0),))
    # The classifier removes feed + net breakage production algebraically;
    # this is not a first-order death hazard. Unresolved fines leave separately.
    model = PBMModel(
        feed=IntegratedBirth((a,b,p,t,ctx)->p.feed[searchsortedfirst(p.edges,a)]),
        breakage=Breakage(hazard,daughters;overflow=:discard),
        classification=CustomTerm((du,N,g,p,t,ctx)->(du .-=outlet(p,N,t).resolved;nothing)),
    )
    operator = compile(model,grid;methods=TermMethods(by_term=(breakage=MillingFixedPivot(),)))
    function balances(ctx,p,t)
        q = outlet(p,ctx.populations.mill,t)
        (discharged_mass=p.rho*p.number_scale*(dot(p.pivots,q.resolved)+q.underflow_volume),
         fines_mass=p.rho*p.number_scale*q.underflow_volume)
    end
    system = PBMSystem((mill=operator,);bulk_rhs=balances,nonnegative_bulk=(:discharged_mass,:fines_mass))
    PBMProblem(system,(mill=zeros(bins),),(0.0,200.0);p,
        bulk0=(discharged_mass=0.0,fines_mass=0.0))
end
problem = build_mill()
times = [1.0,5.0,29.9,30.0,31.0,60.0,200.0]
# The companion retains the active-coordinate reduction, material scaling,
# CVODE nonnegative constraints and exact restart at 30 s. Observations at
# the change use the NEW speed; the saved population remains continuous.
solver = MillingSolver.solve
d50 = SystemObservable((state,op,p,t)->MillingModel.outlet_median(p,state.populations.mill,t))
if abspath(PROGRAM_FILE) == @__FILE__
    nominal = solver(problem,times)
    @show observe(d50,nominal) # Instantaneous outlet VOLUME median, in micrometres.
    # Optional: import PythonPlot
    # plot_observable(nominal,d50;xlabel="Time (s)",ylabel="Outlet volume median (micrometres)")
end

# Stage 2 inputs: independent Gaussian reference coordinates and the explicit
# affine map. This declared law is not truncated by a bounded PCE design.
physical(z) = NamedTuple{PARAMETER_NAMES}(Tuple(
    getproperty(MEAN,k)+getproperty(SD,k)*getproperty(z,k) for k in PARAMETER_NAMES))
inputs = UncertainInputs((delta=Distributions.Normal(),alpha=Distributions.Normal(),
    gamma=Distributions.Normal(),sigma=Distributions.Normal());map=physical,valid=admissible,
    coordinates=:standard_normal,metadata=(joint_law=:independent_gaussian,))
setup = (theta,base)->build_mill(theta;bins=length(base.p.pivots))
uq = UQProblem(problem,inputs;setup,solver,times,outputs=(d50=d50,),
    output_coordinates=(d50=(units=:micrometre,weighting=:instantaneous_outlet_volume,
        interpolation=:linear_diameter_CDF),))
# Run MC, then PCE explicitly: include("milling_propagation.jl") in this scope.
# That small second-stage script reuses MC for independent PCE validation.
