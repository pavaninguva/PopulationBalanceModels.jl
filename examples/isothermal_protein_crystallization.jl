module IsothermalProteinCrystallization
using PopulationBalanceModels
using Distributions: Uniform, LogNormal, Exponential, params
using DelimitedFiles, LinearAlgebra, ForwardDiff

export load_data, select_concentration, select_psds, parameters, build,
       estimation_problem, forward, rates, cfl_solve, DATA, minimal_run

const DATA = joinpath(@__DIR__, "data", "protein_crystallization", "tang2018_isothermal")
const C0 = 19.965841161400512
const TEMPERATURE = 293.15                 # K
const BOLTZMANN = 1.380649e-23             # J K⁻¹
const WINDOW = (2.5,25.0)
const SIGMA_F = 0.03
const CORRELATION_LENGTH = 2.0
const FIXED_GAMMA = 0.64                  # mJ/m²; fixed Mitchell prior centre
const RHO = 1.24e-12                      # g/μm³ (= 1240 kg/m³)
const LOG_AJ_SHIFT = log(60/1000)          # old m⁻³ s⁻¹ -> L⁻¹ min⁻¹
const AJ_BOUNDS = (LOG_AJ_SHIFT,100+LOG_AJ_SHIFT)
const D0_PRIOR_MEAN = 0.01                # μm²/min; declared regularization
const MOLECULAR_VOLUME = 2.97e-26          # m³ molecule⁻¹; Lin et al. (2017)

function table(path)
    data, header = readdlm(path, ',', Any; header=true)
    names = Tuple(Symbol.(vec(header)))
    [NamedTuple{names}(Tuple(row)) for row in eachrow(data)]
end

"""Original Tang Figure 11d/12a readings. Alternative windows are not pooled."""
load_data() = (concentration=table(joinpath(DATA, "concentration.csv")),
               psd=table(joinpath(DATA, "psd_cdf.csv")),
               summary=table(joinpath(DATA, "psd_summary.csv")))
select_concentration(data, split=:calibration) =
    filter(r -> r.split == string(split), data.concentration)
function select_psds(data, split=:calibration; window="isothermal_core")
    rows = filter(r -> r.split == string(split) && r.window == window, data.psd)
    [sort(filter(r -> r.time_min == t, rows); by=r -> r.size_um)
     for t in sort(unique(r.time_min for r in rows))]
end

parameters(; logAJ=30.5, gamma=FIXED_GAMMA, AG=.0002, g=3.3, cstar=2.16,
           grd_strength=0.) =
    (; logAJ, gamma, AG, g, cstar, grd_strength, c0=C0, rho=RHO, kv=.81)

function model_name(model)
    model = Symbol(model)
    model in (:cnt,:cnt_grd) || throw(ArgumentError("model must be cnt or cnt_grd"))
    model
end

# Mitchell et al., equations (3) and (9), Case 3. gamma is entered in mJ/m².
barrier(gamma) = 16pi * (1e-3gamma)^3 * MOLECULAR_VOLUME^2 /
                 (3 * BOLTZMANN^3 * TEMPERATURE^3)
function growth_rate(p, c)
    c <= p.cstar && return zero(c)
    p.AG * (c/p.cstar - 1)^p.g          # μm/min
end
function nucleation_rate(p, c)
    c <= p.cstar && return zero(c)
    logJ = p.logAJ - barrier(p.gamma)/log(c/p.cstar)^2
    # Avoid underflowed Dual values with nonzero subnormal partials. This is
    # machine underflow, not a physical nucleation cutoff or a seed population.
    logJ < log(floatmin(Float64)) && return zero(c)
    logJ < log(floatmax(Float64)) || throw(NumericalFailure("CNT nucleation overflow"))
    exp(logJ) # particles/(L min)
end
rates(p, t, c) = (nucleation_rate(p,c), growth_rate(p,c))
# Every rate is checked against the concentration cap used by the interval
# controller. A violation rejects the solve; no coefficient or state is clipped.
function stage_concentration(p,ctx)
    c = ctx.bulk.c
    if hasproperty(p,:interval_cmax)
        PopulationBalanceModels._primal(c) <= p.interval_cmax ||
            throw(NumericalFailure("protein interval concentration bound exceeded; reduce coefficient_interval or check upper-size occupancy"))
    end
    c
end
nucleation(p,t,ctx) = nucleation_rate(p,stage_concentration(p,ctx))

"""Unseeded batch, using native growth, boundary nucleation and optional Ito GRD.
N is in particles/L, length in μm, time in min, and rho in g/μm³. Both mechanisms use
current-stage c; the entire discrete population derivative drives the solute.
"""
function build(; bins=200, Lmax=50., p=parameters(), tf=660., model=:cnt)
    model=model_name(model)
    model===:cnt && (p=merge(p,(grd_strength=zero(p.grd_strength),)))
    0 < p.cstar < C0 || throw(ArgumentError("require 0 < solubility < initial concentration"))
    p.grd_strength >= 0 || throw(ArgumentError("GRD strength must be nonnegative"))
    grid = Grid1D(range(0.,Lmax; length=bins+1); coordinate=:length)
    terms=(growth=Growth(PolynomialRate(((p,t,ctx)->growth_rate(p,stage_concentration(p,ctx)),))),
           primary=Nucleation(rate=nucleation,size=0.))
    if model===:cnt_grd
        diffusion=SizeDiffusion(PolynomialRate(((p,t,ctx)->
            p.grd_strength,)))
        terms=merge(terms,(dispersion=diffusion,))
    end
    op = compile(PBMModel(;terms...), grid; bc=SizeBC(),
        methods=TermMethods(growth=MUSCL(),by_term=(primary=BoundaryInjection(:left),)))
    function solute_balance(ctx,p,t)
        dN = ctx.derivatives.crystals
        dM3 = sum(grid.centers[i]^3*dN[i] for i in eachindex(dN))
        (c=-p.rho*p.kv*dM3,)
    end
    system = PBMSystem((crystals=op,); nonnegative_bulk=(:c,),bulk_rhs=solute_balance)
    PBMProblem(system,(crystals=zeros(bins),),(0.,tf);bulk0=(c=C0,),p)
end

"""Native SSPRK33 with the package's combined growth/diffusion/loss bound.
At most coefficient_interval minutes are advanced before recomputing the bound.
Monotone rates are bounded by the interval's starting c plus roundoff allowance;
all rate callbacks reject a violation at ANY RK stage. Power-of-two upward rate
buckets keep the schedule locally constant for ForwardDiff. Bounds are not
physical rate floors and no derivative across schedule changes is claimed.
"""
function cfl_solve(pb,times;maximum_dt=2.,cfl=.35,coefficient_interval=5.,maxsteps=200_000)
    maximum_dt>0 && isfinite(maximum_dt) && 0<cfl<=.35 && coefficient_interval>0 ||
        throw(ArgumentError("require positive step/interval and 0 < CFL ≤ 0.35"))
    p=pb.p
    p.AG>0 && p.g>0 && p.gamma>0 && p.grd_strength>=0 || throw(NumericalFailure("invalid CNT/GRD parameters"))
    ts=collect(times)
    !isempty(ts) && all(isfinite,ts) && issorted(ts) &&
        first(pb.tspan)<=first(ts)<=last(ts)<=last(pb.tspan) ||
        throw(ArgumentError("save times must be sorted within the problem interval"))
    u=copy(pb.u0);states=typeof(u)[];t=first(pb.tspan);steps=0
    for target in ts
        while target>t
            endpoint=min(target,t+coefficient_interval)
            cmax=PopulationBalanceModels._primal(bulk(pb,u,:c))+128eps(Float64)*C0
            # The native recommendation evaluates the assembled growth+diffusion
            # operator and actual mesh geometry. Each component's monotone rate
            # is bounded at cmax, and stage_concentration enforces that bound.
            boundstate=copy(u);boundstate[pb.operator.bulk_indices.c]=cmax
            allowed=suggest_dt(pb.operator,boundstate,p,t;cfl=1.)
            rate=isfinite(allowed) ? PopulationBalanceModels._primal(inv(allowed)) : 0.
            bucket=rate>0 ? ldexp(1.,ceil(Int,log2(rate))) : 0.
            dt=bucket>0 ? min(maximum_dt,cfl/bucket) : maximum_dt
            steps+ceil((endpoint-t)/dt)<=maxsteps ||
                throw(NumericalFailure("trial exceeds $maxsteps explicit steps; inspect rate/dispersion scales or use a separately verified solver"))
            interval=remake(pb;u0=u,tspan=(t,endpoint),p=merge(p,(interval_cmax=cmax,)))
            sol=simulate(interval;dt,saveat=[endpoint],method=SSPRK33())
            u=last(sol.u);steps+=sol.steps;t=endpoint
        end
        push!(states,copy(u))
    end
    PBMSolution(ts,states,pb,steps)
end

forward(p=parameters();bins=200,Lmax=50.,dt=2.,cfl=.35,model=:cnt,
        coefficient_interval=5.,times=[0.,74.,180.,420.,660.]) =
    cfl_solve(build(;bins,Lmax,p,tf=maximum(times),model),times;
        maximum_dt=dt,cfl,coefficient_interval)

"""Three common kinetic parameters with fixed gamma=0.64 mJ/m².
No fitted noise. D0 is a constant diffusivity with Exponential mean 0.01 μm²/min.
AJ and AG priors are the previous priors converted into L, μm and min units;
g uses Uniform(1,5). Prior coordinates and Jacobians remain explicit.
"""
function parameter_specs(; p=parameters(),model=:cnt)
    model=model_name(model)
    common=(ParameterSpec(:logAJ;initial=p.logAJ,transform=LogitTransform(AJ_BOUNDS...),
         prior=Prior(Uniform(AJ_BOUNDS...);coordinates=:physical)),
     ParameterSpec(:AG;initial=p.AG,transform=LogTransform(),
         prior=Prior(Uniform(log(1e-5),log(1.5));coordinates=:inference)),
     ParameterSpec(:g;initial=p.g,transform=LogitTransform(1.,5.),
         prior=Prior(Uniform(1.,5.);coordinates=:physical)))
    model===:cnt ? common : (common...,ParameterSpec(:grd_strength;
        initial=p.grd_strength>0 ? p.grd_strength : D0_PRIOR_MEAN,transform=LogTransform(),
        prior=Prior(Exponential(D0_PRIOR_MEAN);coordinates=:physical)))
end

const PARAMETER_INFO=(
    logAJ=(meaning="Natural log of the numerical nucleation prefactor in particles/(L min)",units="dimensionless",label="ln A_J"),
    AG=(meaning="Mean-growth prefactor",units="μm/min",label="A_G"),
    g=(meaning="Supersaturation growth exponent",units="dimensionless",label="g"),
    grd_strength=(meaning="Constant internal-coordinate diffusivity D0",units="μm²/min",label="D_0"))

"""Machine-readable prior definitions, derived from active ParameterSpecs."""
function prior_metadata(problem)
    map(problem.parameters) do s
        d=s.prior.distribution
        family=d isa Uniform ? "uniform" : d isa LogNormal ? "lognormal" : d isa Exponential ? "exponential" : error("unhandled prior")
        transform=s.transform isa LogTransform ? "log" : s.transform isa LogitTransform ? "logit" : "identity"
        info=getproperty(PARAMETER_INFO,s.name)
        (;name=s.name,info...,family,arguments=collect(params(d)),
          density_coordinates=s.prior.coordinates,transform,
          plotting_coordinate=s.name in (:AG,:grd_strength) ? "log" : "physical",
          lower=s.transform isa LogitTransform ? s.transform.lower : nothing,
          upper=s.transform isa LogitTransform ? s.transform.upper : nothing)
    end
end

# Fixed correlation whitening. Its known determinant is reported separately
# so the normalized original-CDF likelihood is recoverable exactly.
cdf_factor(x)=cholesky(Symmetric([exp(-abs(a-b)/CORRELATION_LENGTH) for a in x,b in x])).L

function original_predictions(problem,z)
    blocks=only(predictions(problem,z))
    measurements=only(problem.experiments).measurements
    (blocks[1],(cdf_factor(m.observable.points)*v for (m,v) in zip(measurements[2:end],blocks[2:end]))...)
end

"""Shared fixed-error Bayesian target: 11 concentrations and three CDF blocks.
Concentration SDs are 5% of OBSERVED values, fixed once. CDF SD is .03 absolute
fraction with correlation length 2 μm and 22 interior ordinates per PSD.
Held-out data, initial concentration and diagnostic quantiles are never fitted.
"""
function estimation_problem(;data=load_data(),bins=200,Lmax=50.,dt=2.,cfl=.35,
        coefficient_interval=5.,cstar=2.16,window="isothermal_core",model=:cnt,
        p=parameters(;cstar,grd_strength=Symbol(model)===:cnt ? 0. : D0_PRIOR_MEAN))
    model=model_name(model)
    window=="isothermal_core" || throw(ArgumentError("this comparison uses isothermal_core only"))
    cr=select_concentration(data)
    psds=select_psds(data;window)
    length(cr)==11 && length(psds)==3 || throw(ArgumentError("expected fixed 11+3 calibration split"))
    observed=[r.concentration_g_L for r in cr]
    concentration=Measurement([r.model_time_min for r in cr],observed,BulkObservable(:c);
        likelihood=GaussianLikelihood(sigma=reshape(.05 .*observed,1,:)),source=:concentration)
    distributions=map(psds) do rows
        selected=rows[2:2:end-1]
        x=[r.size_um for r in selected]
        F=reshape([r.conditional_cdf for r in selected],:,1)
        factor=cdf_factor(x)
        Measurement([first(rows).time_min],factor\F,PopulationCDF(:crystals,x;window=WINDOW);
            response=(values,theta,ex)->factor\values,
            likelihood=GaussianLikelihood(sigma=SIGMA_F),
            source=Symbol("CDF_",Int(first(rows).time_min)))
    end
    pb=build(;bins,Lmax,p,tf=420.,model)
    ex=InferenceExperiment(:batch,pb,concentration,distributions...;
        solver=(pb,ts)->cfl_solve(pb,ts;maximum_dt=dt,cfl,coefficient_interval))
    fixed=model===:cnt ? (cstar=cstar,gamma=FIXED_GAMMA,grd_strength=0.) : (cstar=cstar,gamma=FIXED_GAMMA,)
    EstimationProblem(parameter_specs(;p,model),(ex,);fixed)
end

"""Three calibration-only starts, identical common priors and optimizer budget."""
function fit_model(problem;maxiters=500,initial=nothing)
    model=any(s->s.name===:grd_strength,problem.parameters) ? :cnt_grd : :cnt
    S0=C0/problem.fixed.cstar
    starts=map(((1.5,.001),(2.5,.01),(4.,.05))) do (g,D0)
        values=(logAJ=log(1e8)+barrier(FIXED_GAMMA)/log(S0)^2,
            AG=.15/(S0-1)^g,g=g,grd_strength=model===:cnt ? 0. : D0)
        specs=parameter_specs(;p=parameters(;values...),model)
        initial_coordinates(EstimationProblem(specs,problem.experiments;fixed=problem.fixed))
    end
    starts=collect(starts)
    initial===nothing || (starts[1]=collect(initial))
    fit(problem;starts,maxiters,gradient_tolerance=1e-3)
end

"""Unit-determinant growth-rate shear for sampler tuning, not a new parameter.
Keep logit(logAJ) and logit(g). Sample log G at supersaturation 6 instead of log AG.
Returned draws and the MAP remain in their declared inference coordinates.
"""
function sampling_coordinates(problem)
    logexcess=log(5.)
    gspec=only(filter(s->s.name===:g,problem.parameters))
    physicalg(v)=gspec.transform.lower+(gspec.transform.upper-gspec.transform.lower)/
        (1+exp(-v))
    function to_sampler(z)
        y=copy(z);y[2]+=physicalg(z[3])*logexcess;y
    end
    function from_sampler(y)
        z=copy(y);z[2]-=physicalg(y[3])*logexcess;z
    end
    (;to_sampler,from_sampler,logabsdetjac=y->zero(eltype(y)),label=:CNT_fixed_gamma_rates)
end

"""Expected Gaussian information plus actual prior curvature: tuning only."""
function sampler_tuning(problem,z,mapping=sampling_coordinates(problem))
    y=mapping.to_sampler(z);n=length(z)
    flatten(y)=vcat(vec.(only(predictions(problem,mapping.from_sampler(y))))...)
    J=ForwardDiff.jacobian(flatten,y)
    W=copy(J);offset=0
    for m in only(problem.experiments).measurements
        rows=offset+1:offset+length(m.data)
        sigma=m.likelihood.sigma
        W[rows,:]./=sigma isa Number ? sigma : vec(sigma)
        offset+=length(m.data)
    end
    prior=-ForwardDiff.hessian(v->logprior(problem,mapping.from_sampler(v)),y)
    info=Symmetric(W'*W+prior)
    factor=cholesky(info)
    covariance=Matrix(Symmetric(factor\Matrix{Float64}(I,n,n)))
    (;covariance,information=Matrix(info),condition_number=cond(info),
      interpretation="expected-information sampler tuning only")
end

"""W₂ between the conditional bin density and a tabulated piecewise-linear CDF.
On each common probability interval the quantile difference is affine. Values
at its one-third and two-thirds points integrate its square analytically:
Δp * (d₁² - d₁*d₂ + d₂²). No endpoint trace or quadrature floor is required.
This is a reporting diagnostic, not the Bayesian likelihood.
"""
function tabulated_wasserstein(grid,N,x,F;window=(first(x),last(x)))
    a,b=window
    edges=sort(unique(vcat(a,[e for e in grid.edges if a<e<b],b)))
    model=conditional_cdf(grid,N,edges;window)
    p=sort(unique(vcat(model,F)))
    widths=diff(p)
    u1=p[1:end-1].+widths./3
    u2=p[1:end-1].+2widths./3
    d1=cdf_quantiles(edges,model,u1)-cdf_quantiles(x,F,u1)
    d2=cdf_quantiles(edges,model,u2)-cdf_quantiles(x,F,u2)
    sqrt(sum(widths.*(d1.^2 .- d1.*d2 .+ d2.^2)))
end

function minimal_run()
    sol=forward()
    println("Uncalibrated CNT batch: c(660 min)=",bulk(sol.problem,last(sol.u),:c)," g/L")
    sol
end
end

if abspath(PROGRAM_FILE)==@__FILE__
    IsothermalProteinCrystallization.minimal_run()
end
