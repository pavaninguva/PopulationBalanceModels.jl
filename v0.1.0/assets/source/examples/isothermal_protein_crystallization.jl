module IsothermalProteinCrystallization
using PopulationBalanceModels
using Distributions: Uniform, LogNormal, Exponential, params
using DelimitedFiles, LinearAlgebra, ForwardDiff

export load_data, select_concentration, select_psds, parameters, build,
       estimation_problem, forward, rates, cfl_solve, DATA, minimal_run

# tutorial:start constants
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
const KD_PRIOR_MEAN = 0.1                 # μm; regularization of D_G = k_D G
const MOLECULAR_VOLUME = 2.97e-26          # m³ molecule⁻¹; Lin et al. (2017)

# tutorial:end constants

# tutorial:start data
include("support/protein_data.jl")
# tutorial:end data

# tutorial:start model
function parameters(; logAJ=30.5, gamma=FIXED_GAMMA, AG=.0002, g=3.3, cstar=2.16,
                    grd_strength=0., k_D=nothing)
    p=(; logAJ, gamma, AG, g, cstar, grd_strength, c0=C0, rho=RHO, kv=.81)
    k_D===nothing ? p : merge(p,(;k_D))
end

function model_name(model)
    model = Symbol(model)
    model in (:cnt,:cnt_grd,:cnt_growth_grd) ||
        throw(ArgumentError("model must be cnt, cnt_grd or cnt_growth_grd"))
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

nucleation(p,t,ctx) = nucleation_rate(p,stage_concentration(p,ctx))

"""Unseeded batch, using native growth, boundary nucleation and optional Ito GRD.
N is in particles/L, length in μm, time in min, and rho in g/μm³. Both mechanisms use
current-stage c; the entire discrete population derivative drives the solute.
"""
function build(; bins=200, Lmax=50., p=parameters(), tf=660., model=:cnt)
    model=model_name(model)
    model===:cnt && (p=merge(p,(grd_strength=zero(p.grd_strength),)))
    if model===:cnt_growth_grd
        hasproperty(p,:k_D) || (p=merge(p,(k_D=KD_PRIOR_MEAN,)))
        isfinite(p.k_D) && p.k_D>=0 || throw(ArgumentError("k_D must be finite and nonnegative"))
    end
    0 < p.cstar < C0 || throw(ArgumentError("require 0 < solubility < initial concentration"))
    p.grd_strength >= 0 || throw(ArgumentError("GRD strength must be nonnegative"))
    grid = Grid1D(range(0.,Lmax; length=bins+1); coordinate=:length)
    terms=(growth=Growth(PolynomialRate(((p,t,ctx)->growth_rate(p,stage_concentration(p,ctx)),))),
           primary=Nucleation(rate=nucleation,size=0.))
    if model===:cnt_grd
        diffusion=SizeDiffusion(PolynomialRate(((p,t,ctx)->
            p.grd_strength,)))
        terms=merge(terms,(dispersion=diffusion,))
    elseif model===:cnt_growth_grd
        diffusion=SizeDiffusion(PolynomialRate(((p,t,ctx)->
            p.k_D*growth_rate(p,stage_concentration(p,ctx)),)))
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

# tutorial:end model

# tutorial:start controller
include("support/protein_execution.jl")

forward(p=parameters();bins=200,Lmax=50.,dt=2.,cfl=.35,model=:cnt,
        coefficient_interval=5.,times=[0.,74.,180.,420.,660.]) =
    cfl_solve(build(;bins,Lmax,p,tf=maximum(times),model),times;
        maximum_dt=dt,cfl,coefficient_interval)

# tutorial:end controller

# tutorial:start priors
"""Three common kinetic parameters with fixed gamma=0.64 mJ/m².
No fitted noise. Constant D0 has Exponential mean 0.01 μm²/min; the alternative
D_G = k_D G has an Exponential k_D prior with mean 0.1 μm.
AJ and AG priors are the previous priors converted into L, μm and min units;
g uses Uniform(1,5). Starting values are physical; inference transforms are
explicit. The tagged log-uniform declaration for AG preserves the retained
prior law and export metadata; the other priors use physical distributions.
"""
function parameter_specs(; p=parameters(),model=:cnt)
    model=model_name(model)
    common=(ParameterSpec(:logAJ;initial=p.logAJ,transform=LogitTransform(AJ_BOUNDS...),
         prior=Uniform(AJ_BOUNDS...)),
     ParameterSpec(:AG;initial=p.AG,transform=LogTransform(),
         prior=Prior(Uniform(log(1e-5),log(1.5));coordinates=:inference)),
     ParameterSpec(:g;initial=p.g,transform=LogitTransform(1.,5.),
         prior=Uniform(1.,5.)))
    model===:cnt && return common
    model===:cnt_growth_grd && return (common...,ParameterSpec(:k_D;
        initial=hasproperty(p,:k_D) && p.k_D>0 ? p.k_D : KD_PRIOR_MEAN,
        transform=LogTransform(),prior=Exponential(KD_PRIOR_MEAN)))
    (common...,ParameterSpec(:grd_strength;
        initial=p.grd_strength>0 ? p.grd_strength : D0_PRIOR_MEAN,transform=LogTransform(),
        prior=Exponential(D0_PRIOR_MEAN)))
end

# tutorial:end priors

const PARAMETER_INFO=(
    logAJ=(meaning="Natural log of the numerical nucleation prefactor in particles/(L min)",units="dimensionless",label="ln A_J"),
    AG=(meaning="Mean-growth prefactor",units="μm/min",label="A_G"),
    g=(meaning="Supersaturation growth exponent",units="dimensionless",label="g"),
    grd_strength=(meaning="Constant internal-coordinate diffusivity D0",units="μm²/min",label="D_0"),
    k_D=(meaning="Growth-dependent GRD coefficient in D_G = k_D G",units="μm",label="k_D"))

"""Machine-readable prior definitions, derived from active ParameterSpecs."""
function prior_metadata(problem)
    map(problem.parameters) do s
        d=s.prior.distribution
        family=d isa Uniform ? "uniform" : d isa LogNormal ? "lognormal" : d isa Exponential ? "exponential" : error("unhandled prior")
        transform=s.transform isa LogTransform ? "log" : s.transform isa LogitTransform ? "logit" : "identity"
        info=getproperty(PARAMETER_INFO,s.name)
        (;name=s.name,info...,family,arguments=collect(params(d)),
          density_coordinates=s.prior.coordinates,transform,
          plotting_coordinate=s.name in (:AG,:grd_strength,:k_D) ? "log" : "physical",
          lower=s.transform isa LogitTransform ? s.transform.lower : nothing,
          upper=s.transform isa LogitTransform ? s.transform.upper : nothing)
    end
end

# tutorial:start cdf_covariance
# Fixed correlation whitening. Its known determinant is reported separately
# so the normalized original-CDF likelihood is recoverable exactly.
cdf_factor(x)=cholesky(Symmetric([exp(-abs(a-b)/CORRELATION_LENGTH) for a in x,b in x])).L

# tutorial:end cdf_covariance

function original_predictions(problem,z)
    blocks=only(predictions(problem,z))
    measurements=only(problem.experiments).measurements
    (blocks[1],(cdf_factor(m.observable.points)*v for (m,v) in zip(measurements[2:end],blocks[2:end]))...)
end

# tutorial:start inference
"""Shared fixed-error Bayesian target: 11 concentrations and three CDF blocks.
Concentration SDs are 5% of OBSERVED values, fixed once. CDF SD is .03 absolute
fraction with correlation length 2 μm and 22 interior ordinates per PSD.
Held-out data, initial concentration and diagnostic quantiles are never fitted.
"""
function estimation_problem(;data=load_data(),bins=200,Lmax=50.,dt=2.,cfl=.35,
        coefficient_interval=5.,cstar=2.16,window="isothermal_core",model=:cnt,
        p=parameters(;cstar,grd_strength=Symbol(model)===:cnt_grd ? D0_PRIOR_MEAN : 0.))
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


# tutorial:end inference

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
    prior=-ForwardDiff.hessian(v->sampling_logprior(problem,mapping.from_sampler(v)),y)
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
