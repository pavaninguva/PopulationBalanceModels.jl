module IsothermalProteinCrystallization
using PopulationBalanceModels
using DelimitedFiles, Random

export load_data, select_concentration, select_psds, parameters, build, calibration,
       coordinates, physical_coordinates, forward, rates, cfl_solve, DATA
const DATA = joinpath(@__DIR__, "data", "protein_crystallization", "tang2018_isothermal")
const C0 = 19.965841161400512
const BOUNDS = ([log(1e6), log(1e-4), 0., 0.], [log(1e16), log(2.), 10., 6.])

function table(path)
    data, header = readdlm(path, ',', Any; header=true)
    names = Tuple(Symbol.(vec(header)))
    [NamedTuple{names}(Tuple(row)) for row in eachrow(data)]
end

"""Original Tang Figure 11d/12a readings; alternative windows are never pooled."""
load_data() = (concentration=table(joinpath(DATA, "concentration.csv")),
               psd=table(joinpath(DATA, "psd_cdf.csv")),
               summary=table(joinpath(DATA, "psd_summary.csv")))
select_concentration(data, split=:calibration) = filter(r -> r.split == string(split), data.concentration)
function select_psds(data, split=:calibration; window="isothermal_core")
    rows = filter(r -> r.split == string(split) && r.window == window, data.psd)
    [sort(filter(r -> r.time_min == t, rows); by=r -> r.size_um) for t in sort(unique(r.time_min for r in rows))]
end

parameters(; J_ref=1e11, G_ref=.2, j=2., g=2., cstar=2.16, onset=nothing) =
    (; J_ref, G_ref, j, g, cstar, c0=C0, rho=1240., kv=.81, onset)

"""Rates in particles m⁻³ min⁻¹ and μm min⁻¹. Both vanish at/below solubility."""
function rates(p, t, c)
    c <= p.cstar && return (zero(c), zero(c))
    r = (c-p.cstar)/(p.c0-p.cstar)
    activation = p.onset === nothing ? one(c) : inv(1+exp(-(t-p.onset)/10))
    (p.J_ref*r^p.j*activation, p.G_ref*r^p.g)
end
growth(L,p,t,ctx) = last(rates(p,t,ctx.bulk.c))
nucleation(p,t,ctx) = first(rates(p,t,ctx.bulk.c))/1e12

"""Empty batch, bin-integrated counts in 10¹² m⁻³, length in μm and time in min.
The same discrete population increment determines dissolved-material exchange.
"""
function build(; bins=100, Lmax=50., p=parameters(), tf=660.)
    p.cstar < C0 || throw(ArgumentError("solubility must be below the fixed initial concentration"))
    grid = Grid1D(range(0.,Lmax; length=bins+1); coordinate=:length)
    model = PBMModel(growth=Growth(rate=growth), primary=Nucleation(rate=nucleation,size=0.))
    op = compile(model, grid; bc=SizeBC(NoFlux(),OpenBoundary()),methods=TermMethods(growth=MUSCL(),
        by_term=(primary=BoundaryInjection(:left),)))
    system = PBMSystem((crystals=op,); nonnegative_bulk=(:c,),
        accounting=(CumulativeAccounting(:crystals;properties=(material=3,)),), exchanges=(
        ConservativeExchange(:crystals,:c,(L,p,t,ctx)->1e-6*p.rho*p.kv*L^3),))
    PBMProblem(system,(crystals=zeros(bins),),(0.,tf); bulk0=(c=C0,),p)
end

# Smooth box coordinates impose numerical search bounds, not priors or Jacobians.
function physical_coordinates(u; gated=false)
    lo,hi = gated ? (vcat(BOUNDS[1],0.), vcat(BOUNDS[2],150.)) : BOUNDS
    lo .+ (hi.-lo)./(1 .+ exp.(-u))
end
function coordinates(theta; gated=length(theta)==5)
    lo,hi = gated ? (vcat(BOUNDS[1],0.), vcat(BOUNDS[2],150.)) : BOUNDS
    all(lo .< theta .< hi) || throw(ArgumentError("start must be strictly inside numerical bounds"))
    log.((theta.-lo)./(hi.-theta))
end
function decode(u; cstar=2.16,gated=false)
    x = physical_coordinates(u; gated)
    parameters(J_ref=exp(x[1]), G_ref=exp(x[2]), j=x[3], g=x[4], cstar=cstar,
               onset=gated ? x[5] : nothing)
end

"""Calibration only: 11 concentrations and three conditional NUMBER CDF blocks.
Weights are comparison scales, not measured variances. No validation data enter.
The native solver's fixed step bounds growth CFL by 0.35 at each trial. A
power-of-two upper bound on G_ref keeps the schedule locally constant for AD.
"""
function calibration(;data=load_data(),bins=100,Lmax=50.,dt=.5,cfl=.35,
        cstar=2.16,window="isothermal_core",sc=.1,sL=.5,weight=1.,points=256,gated=false)
    cr = select_concentration(data)
    psds = select_psds(data; window)
    length(cr)==11 && length(psds)==3 || throw(ArgumentError("primary calibration requires 11 concentrations and three PSD times"))
    window_edges = (first(first(psds)).normalization_min_um,last(first(psds)).normalization_max_um)
    obs = ResidualObservation([r.model_time_min for r in cr], [r.concentration_g_L for r in cr],
        BulkObservable(:c); scale=sc,name=:concentration)
    distributions = map(psds) do rows
        WassersteinObservation(first(rows).time_min,[r.size_um for r in rows],
            [r.conditional_cdf for r in rows],:crystals; window=window_edges,points,
            scale=sL,weight=weight/3,name=Symbol("PSD_",Int(first(rows).time_min)))
    end
    pb = build(;bins,Lmax,tf=420.,p=parameters(;cstar))
    CalibrationProblem((Experiment(pb,obs,distributions...),);
        parameters=(u,ex)->decode(u;cstar,gated),
        solver=(pb,ts)->cfl_solve(pb,ts;maximum_dt=dt,cfl))
end

"""Call native SSPRK33 with a fixed, trial-specific schedule. For these declared
power laws, G(t) ≤ G_ref while c ≤ c0 and g ≥ 0. Bucket boundaries are numerical
schedule changes, not differentiable events; check time refinement there.
`maximum_dt` resolves the fixed 10-min gate as well as transport.
"""
function cfl_solve(pb,times;maximum_dt=.5,cfl=.35)
    isfinite(maximum_dt) && maximum_dt>0 && 0<cfl<=.35 || throw(ArgumentError("require positive maximum_dt and 0<CFL<=0.35"))
    p=pb.p
    p.G_ref>0 && p.j>=0 && p.g>=0 || throw(ArgumentError("CFL bound requires positive G_ref and nonnegative exponents"))
    bound=ldexp(1.,ceil(Int,log2(p.G_ref)))
    grid=pb.operator.system.populations.crystals.grid
    h=min(maximum_dt,cfl*minimum(grid.widths)/bound)
    simulate(pb;dt=h,saveat=times,method=SSPRK33())
end
forward(p=parameters();bins=100,Lmax=50.,dt=.5,cfl=.35,times=[0.,74.,180.,420.,660.]) =
    cfl_solve(build(;bins,Lmax,p,tf=max(660.,last(times))),times;maximum_dt=dt,cfl)

function minimal_run()
    sol=forward()
    pb=sol.problem; grid=pb.operator.system.populations.crystals.grid
    q=conditional_quantiles(grid,population(pb,last(sol.u),:crystals),[.1,.5,.9];window=(2.5,25.))
    println("Forward smoke calculation, not a calibrated result: c(660)=",bulk(pb,last(sol.u),:c)," g/L; conditional number quantiles=",q," μm")
    sol
end
end

if abspath(PROGRAM_FILE)==@__FILE__
    IsothermalProteinCrystallization.minimal_run()
end
