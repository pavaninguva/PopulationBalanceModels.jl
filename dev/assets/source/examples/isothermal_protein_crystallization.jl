module IsothermalProteinCrystallization
using PopulationBalanceModels
using Distributions: Uniform, LogNormal, Exponential
using DelimitedFiles, LinearAlgebra, ForwardDiff

export load_data, select_concentration, select_psds, parameters, build,
       estimation_problem, forward, rates, cfl_solve, DATA, minimal_run

const DATA = joinpath(@__DIR__, "data", "protein_crystallization", "tang2018_isothermal")
const C0 = 19.965841161400512
const TEMPERATURE = 293.15                 # K
const BOLTZMANN = 1.380649e-23             # J K⁻¹
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

parameters(; logAJ=41., gamma=.76, AG=1., g=2.4, cstar=2.16,
           sigma_c=1., sigma_F=.1) =
    (; logAJ, gamma, AG, g, cstar, sigma_c, sigma_F, c0=C0, rho=1240., kv=.81)

# Mitchell et al., equations (3) and (9), Case 3. gamma is entered in mJ/m².
barrier(gamma) = 16pi * (1e-3gamma)^3 * MOLECULAR_VOLUME^2 /
                 (3 * BOLTZMANN^3 * TEMPERATURE^3)
function growth_rate(p, c)
    c <= p.cstar && return zero(c)
    1e-3 * p.AG * (c/p.cstar - 1)^p.g   # nm/min -> μm/min
end
function nucleation_rate(p, c)
    c <= p.cstar && return zero(c)
    logJ = log(60) + p.logAJ - barrier(p.gamma)/log(c/p.cstar)^2
    # Avoid underflowed Dual values with nonzero subnormal partials. This is
    # machine underflow, not a physical nucleation cutoff or a seed population.
    logJ < log(floatmin(Float64)) && return zero(c)
    logJ < log(floatmax(Float64)) || throw(NumericalFailure("CNT nucleation overflow"))
    exp(logJ) # #/(m³ s) -> #/(m³ min)
end
rates(p, t, c) = (nucleation_rate(p,c), growth_rate(p,c))
growth(L,p,t,ctx) = growth_rate(p,ctx.bulk.c)
nucleation(p,t,ctx) = nucleation_rate(p,ctx.bulk.c)/1e12

"""Empty isothermal batch. N is in 10¹² particles/m³, length in μm, time in min.
The same stage's discrete population derivative sets the solute derivative.
The closed upper boundary must remain unoccupied; check domain refinement.
"""
function build(; bins=100, Lmax=50., p=parameters(), tf=660.)
    0 < p.cstar < C0 || throw(ArgumentError("require 0 < solubility < initial concentration"))
    grid = Grid1D(range(0.,Lmax; length=bins+1); coordinate=:length)
    model = PBMModel(growth=Growth(PolynomialRate(((p,t,ctx)->growth_rate(p,ctx.bulk.c),))),
                     primary=Nucleation(rate=nucleation,size=0.))
    op = compile(model, grid; bc=SizeBC(NoFlux(),NoFlux()),
        methods=TermMethods(growth=MUSCL(),by_term=(primary=BoundaryInjection(:left),)))
    function solute_balance(ctx,p,t)
        dN = ctx.derivatives.crystals
        dM3 = sum(grid.centers[i]^3*dN[i] for i in eachindex(dN))
        (c=-1e-6*p.rho*p.kv*dM3,)
    end
    system = PBMSystem((crystals=op,); nonnegative_bulk=(:c,),bulk_rhs=solute_balance)
    PBMProblem(system,(crystals=zeros(bins),),(0.,tf);bulk0=(c=C0,),p)
end

"""Native SSPRK33, CFL ≤ 0.35, with a fixed step within each saved interval.
Positive size-independent growth and conservative solute consumption make growth
nonincreasing. Recompute its upper bound at each interval's initial bulk state;
this avoids keeping the initial fast-growth step throughout a depleted batch.
Power-of-two buckets make the schedule locally constant for forward AD. Check
step refinement independently; no derivatives of schedule changes are claimed.
"""
function cfl_solve(pb,times;maximum_dt=2.,cfl=.35)
    maximum_dt>0 && isfinite(maximum_dt) && 0<cfl<=.35 ||
        throw(ArgumentError("require positive maximum_dt and 0 < CFL ≤ 0.35"))
    p=pb.p
    p.AG>0 && p.g>0 && p.gamma>0 || throw(NumericalFailure("CNT requires positive AG, g and gamma"))
    ts=collect(times)
    !isempty(ts) && all(isfinite,ts) && issorted(ts) &&
        first(pb.tspan)<=first(ts)<=last(ts)<=last(pb.tspan) ||
        throw(ArgumentError("save times must be sorted within the problem interval"))
    grid=pb.operator.system.populations.crystals.grid
    u=copy(pb.u0);states=typeof(u)[];t=first(pb.tspan);steps=0
    for target in ts
        if target>t
            G=growth_rate(p,bulk(pb,u,:c))
            bound=G>0 ? ldexp(1.,ceil(Int,log2(G))) : 0.
            dt=bound>0 ? min(maximum_dt,cfl*minimum(grid.widths)/bound) : maximum_dt
            (target-t)/dt+steps<=200_000 ||
                throw(NumericalFailure("trial requires >200000 steps; initial growth exceeds the calculation budget"))
            interval=remake(pb;u0=u,tspan=(t,target))
            sol=simulate(interval;dt,saveat=[target],method=SSPRK33())
            u=last(sol.u);steps+=sol.steps;t=target
        end
        push!(states,copy(u))
    end
    PBMSolution(ts,states,pb,steps)
end

forward(p=parameters();bins=100,Lmax=50.,dt=2.,cfl=.35,
        times=[0.,74.,180.,420.,660.]) =
    cfl_solve(build(;bins,Lmax,p,tf=maximum(times)),times;maximum_dt=dt,cfl)

"""Independent priors informed by Mitchell Table 3's INITIAL ranges, not its
same-data final estimates. The distributions and broadened growth range are our
explicit choices. All likelihood normalizers and coordinate Jacobians are kept.
"""
function parameter_specs(; p=parameters())
    (ParameterSpec(:logAJ;initial=p.logAJ,transform=LogitTransform(0.,100.),
         prior=Prior(Uniform(0.,100.);coordinates=:physical)),
     ParameterSpec(:gamma;initial=p.gamma,transform=LogTransform(),
         prior=Prior(LogNormal(log(.64),.7);coordinates=:physical)),
     ParameterSpec(:AG;initial=p.AG,transform=LogTransform(),
         prior=Prior(Uniform(log(.01),log(1500.));coordinates=:inference)),
     ParameterSpec(:g;initial=p.g,transform=LogitTransform(1.,3.),
         prior=Prior(Uniform(1.,3.);coordinates=:physical)),
     ParameterSpec(:sigma_c;initial=p.sigma_c,transform=LogTransform(),role=:nuisance,
         prior=Prior(Exponential(1.);coordinates=:physical)),
     ParameterSpec(:sigma_F;initial=p.sigma_F,transform=LogTransform(),role=:nuisance,
         prior=Prior(Exponential(.1);coordinates=:physical)))
end

# Fixed linear whitening avoids differentiating a covariance factorization at
# every evaluation. The Gaussian density is in whitened observation coordinates.
# Add -sum(log,diag(factor)) per block to report the original-CDF log density;
# this known constant does not change MAP, Laplace or sampling.
cdf_factor(x)=cholesky(Symmetric([exp(-abs(a-b)/2) for a in x,b in x])).L

function original_predictions(problem,z)
    blocks=only(predictions(problem,z))
    measurements=only(problem.experiments).measurements
    (blocks[1],(cdf_factor(m.observable.points)*v for (m,v) in zip(measurements[2:end],blocks[2:end]))...)
end

"""Bayesian calibration: 11 concentrations and three conditional number-CDFs.
CDF ordinates are evaluated at 1 μm spacing, excluding the fixed endpoints.
Each time has one correlated Gaussian block, Rᵢⱼ=exp(-|Lᵢ-Lⱼ|/2 μm).
Its inferred scale describes observation discrepancy, not particle-count noise.
Validation rows and derived quantiles are excluded from the likelihood.
"""
function estimation_problem(;data=load_data(),bins=100,Lmax=50.,dt=2.,
        cstar=2.16,window="isothermal_core",p=parameters(;cstar))
    cr=select_concentration(data)
    psds=select_psds(data;window)
    length(cr)==11 && length(psds)==3 || throw(ArgumentError("expected fixed 11+3 calibration split"))
    concentration=Measurement([r.model_time_min for r in cr],
        [r.concentration_g_L for r in cr],BulkObservable(:c);
        likelihood=GaussianLikelihood(sigma=(theta,ex)->theta.sigma_c),source=:concentration)
    distributions=map(psds) do rows
        selected=rows[2:2:end-1]
        x=[r.size_um for r in selected]
        F=reshape([r.conditional_cdf for r in selected],:,1)
        factor=cdf_factor(x)
        w=(first(rows).normalization_min_um,last(rows).normalization_max_um)
        Measurement([first(rows).time_min],factor\F,PopulationCDF(:crystals,x;window=w);
            response=(values,theta,ex)->factor\values,
            likelihood=GaussianLikelihood(sigma=(theta,ex)->theta.sigma_F),
            source=Symbol("CDF_",Int(first(rows).time_min)))
    end
    pb=build(;bins,Lmax,p,tf=420.)
    ex=InferenceExperiment(:batch,pb,concentration,distributions...;
        solver=(pb,ts)->cfl_solve(pb,ts;maximum_dt=dt))
    EstimationProblem(parameter_specs(;p),(ex,);fixed=(cstar=cstar,))
end

"""Deterministic multistart MAP of the declared posterior, in inference z.
The starts vary kinetics; none uses the held-out observations.
Load Optim to activate the package fitting extension.
"""
function fit_model(problem; maxiters=500)
    S0=C0/problem.fixed.cstar
    starts=map(((.76,2.4),(.5,1.5),(1.,2.8))) do (gamma,g)
        values=(logAJ=log(1e11/60)+barrier(gamma)/log(S0)^2,
            gamma=gamma,AG=150/(S0-1)^g,g=g,sigma_c=1.,sigma_F=.1)
        initial_coordinates(EstimationProblem(parameter_specs(;p=parameters(;values...)),
            problem.experiments;fixed=problem.fixed))
    end
    fit(problem;starts,maxiters,gradient_tolerance=1e-3)
end

"""Optional NUTS coordinates: initial log nucleation and growth rates instead
of their strongly correlated coefficients. The Jacobian is of z(y), not y(z).
This changes sampling coordinates only; the declared posterior stays in z.
"""
function sampling_coordinates(problem)
    S0=C0/problem.fixed.cstar
    logS2=log(S0)^2
    logexcess=log(S0-1)
    sigmoid(x)=x>=0 ? inv(1+exp(-x)) : exp(x)/(1+exp(x))
    function to_sampler(z)
        p=physical_parameters(problem,z,:batch)
        [log(60)+p.logAJ-barrier(p.gamma)/logS2,
         z[2],log(.001)+z[3]+p.g*logexcess,z[4],z[5],z[6]]
    end
    function inverse_A(y)
        A=y[1]-log(60)+barrier(exp(y[2]))/logS2
        0<A<100 || throw(NumericalFailure("CNT sampler map is outside 0 < logAJ < 100"))
        A
    end
    function from_sampler(y)
        A=inverse_A(y)
        g=1+2sigmoid(y[4])
        [log(A/(100-A)),y[2],y[3]-log(.001)-g*logexcess,y[4],y[5],y[6]]
    end
    logabsdetjac(y)=let A=inverse_A(y)
        log(100)-log(A)-log(100-A)
    end
    (;to_sampler,from_sampler,logabsdetjac,label=:CNT_initial_rates)
end

"""Expected Gaussian information for sampler tuning, in the auxiliary y
coordinates. This is not the observed posterior Hessian or Laplace uncertainty.
The positive prior curvatures follow the stated transformed priors exactly.
No diagonal ridge or posterior-covariance claim is introduced.
"""
function sampler_tuning(problem,z,map=sampling_coordinates(problem))
    y=map.to_sampler(z)
    flatten(y)=vcat(vec.(only(predictions(problem,map.from_sampler(y))))...)
    J=ForwardDiff.jacobian(flatten,y)
    theta=physical_parameters(problem,z,:batch)
    blocks=only(problem.experiments).measurements
    whitened=similar(J)
    offset=0
    for (k,m) in enumerate(blocks)
        count=length(m.data)
        rows=offset+1:offset+count
        if k==1
            whitened[rows,:].=J[rows,:]./theta.sigma_c
        else
            # The CDF predictions have already been whitened by fixed R.
            whitened[rows,:].=J[rows,:]./theta.sigma_F
        end
        offset+=count
    end
    sigmoid=(theta.g-1)/2
    prior_and_scale_information=[0.,1/.7^2,0.,2sigmoid*(1-sigmoid),
        2length(first(blocks).data)+theta.sigma_c,
        2sum(length(m.data) for m in blocks[2:end])+theta.sigma_F/.1]
    info=Symmetric(whitened'*whitened+Diagonal(prior_and_scale_information))
    factor=cholesky(info)
    covariance=Matrix(Symmetric(factor\Matrix{Float64}(I,6,6)))
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
