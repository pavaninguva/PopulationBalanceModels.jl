# Minimal estimating growth tutorial.
# Generated from maintained scientific source sections and a short run block.
# No benchmark exports or retained files are written. See the case-study page.

using PopulationBalanceModels
using Optim, Random, LinearAlgebra
using Distributions: Normal, LogNormal, Uniform, pdf, cdf, ccdf, quantile, truncated

# From examples/estimate_size_dependent_growth.jl (inference).
const TRUTH=(g0=0.434,g1=0.2604)
const TIMES=[0.1,0.25,0.4,0.6,0.8,1.0]
const SIZES=collect(range(0.0,1.4;length=281))
const CENTER=0.2
const WIDTH=sqrt(0.0005/2)
const INITIAL_NUMBER=50sqrt(π*0.0005)
const UNIFORM_BOUNDS=((0.05,1.0),(0.02,1.0))
const LAW=AffineGrowth(:g0,:g1)
initial_density(a)=50exp(-(a-CENTER)^2/0.0005)
# Independent closed-form reference for data generation, not a discretized solve.
shift(t,p)=iszero(p.g1) ? p.g0*t : p.g0*expm1(p.g1*t)/p.g1
function exact_density(a,t,p=TRUTH)
    foot=(a-shift(t,p))*exp(-p.g1*t)
    foot<0 ? 0.0 : initial_density(foot)*exp(-p.g1*t)
end
function initial_integral(lo,hi)
    hi<=lo && return 0.0
    d=Normal(CENTER,WIDTH)
    INITIAL_NUMBER*(lo>CENTER ? ccdf(d,lo)-ccdf(d,hi) : cdf(d,hi)-cdf(d,lo))
end
function initial_moment(k)
    a=-CENTER/WIDTH;phi=pdf(Normal(),a);integrals=[ccdf(Normal(),a),phi]
    for j in 2:k;push!(integrals,a^(j-1)*phi+(j-1)*integrals[j-1]);end
    INITIAL_NUMBER*sum(binomial(k,j)*CENTER^(k-j)*WIDTH^j*integrals[j+1] for j in 0:k)
end
function exact_counts(edges,t,p=TRUTH)
    inverse(a)=max(0.0,(a-shift(t,p))*exp(-p.g1*t))
    [initial_integral(inverse(a),inverse(b)) for (a,b) in zip(edges[1:end-1],edges[2:end])]
end
function exact_moment(k,t,p=TRUTH)
    growth=exp(p.g1*t);translation=shift(t,p)
    sum(binomial(k,j)*growth^j*translation^(k-j)*initial_moment(j) for j in 0:k)
end

"""Evaluate analytical densities and sample independent Gaussian observations conditioned to be nonnegative. The fitted likelihood includes this truncation; no clipping, counts or renormalization are used."""
function synthetic_measurements(;seed=2022,noise=1.0,sizes=SIZES,times=TIMES)
    noise>0 && isfinite(noise) || throw(ArgumentError("noise must be positive"))
    truth=[exact_density(a,t) for a in sizes,t in times]
    rng=MersenneTwister(seed)
    values=[rand(rng,truncated(Normal(truth[i,j],noise),0.0,Inf)) for i in axes(truth,1),j in axes(truth,2)]
    (;values,truth,noise,sizes=collect(sizes),times=collect(times),seed,noise_model=:truncated_gaussian)
end

function physical_problem(;p=TRUTH,forward=:analytic,bins=280,upper=1.4,dt=0.005,scheme=MUSCL())
    boundary=SizeBC(OpenBoundary(),OpenBoundary())
    model=PBMModel(growth=Growth(LAW))
    op=if forward in (:analytic,:inguva)
        grid=characteristic_grid(LAW,(0.0,upper);dt,p)
        compile(model,grid;evolution=Inguva2022(),p,bc=boundary)
    elseif forward===:sectional
        grid=Grid1D(range(0.0,upper;length=bins+1);coordinate=:length)
        compile(model,grid;methods=TermMethods(growth=scheme),bc=boundary)
    else
        throw(ArgumentError("forward must be :analytic, :inguva or :sectional"))
    end
    PBMProblem(op,exact_counts(op.grid.edges,0.0,p),(0.0,1.0);p)
end

function analytical_solver(pb,times)
    states=[reshape(exact_counts(pb.operator.grid.edges,t,pb.p),:,1) for t in times]
    PBMSolution(times,states,pb,0)
end
# Point observations from a bin solver require an explicit reconstruction. This
# linear interpolation of bin-average densities is NOT an exact continuous PSD.
function sample_sectional(N,op,points)
    g=op.grid;f=vec(density(g,N))
    map(points) do x
        i=searchsortedlast(g.centers,x)
        i==0 && return first(f)
        i>=length(f) && return last(f)
        w=(x-g.centers[i])/(g.centers[i+1]-g.centers[i])
        (1-w)*f[i]+w*f[i+1]
    end
end

"""
Two fitted kinetic parameters; the initial density and truncated Gaussian error scale are known.
`prior=:lognormal` retains the previous growth priors. `prior=:uniform` uses
physical bounds from stated growth/stretching timescales. Both use log inference
coordinates as an explicit numerical override. MAP and posterior sampling use the same declared log-coordinate density,
including the physical-prior Jacobian.
The default forward calculation evaluates the analytical continuous density.
The optional bin solvers rebuild the characteristic mesh when parameters change.
"""
function estimation_problem(data=synthetic_measurements();prior=:lognormal,forward=:analytic,bins=280,upper=1.4,dt=0.005,scheme=MUSCL())
    prior in (:lognormal,:uniform) || throw(ArgumentError("unknown prior scenario"))
    base=physical_problem(;forward,bins,upper,dt,scheme)
    observable=forward===:analytic ?
        FunctionObservable((N,op,p,t)->exact_density.(data.sizes,t,Ref(p))) :
        FunctionObservable((N,op,p,t)->sample_sectional(N,op,data.sizes))
    measurement=Measurement(data.times,data.values,observable;
        likelihood=TruncatedGaussianLikelihood(sigma=data.noise),source=:density_measurements)
    setup=(theta,pb)->forward===:inguva ? physical_problem(;p=theta,forward,bins,upper,dt,scheme) : remake(pb;p=theta)
    solver=forward===:analytic ? analytical_solver : forward===:inguva ?
        ((pb,ts)->simulate(pb;dt=pb.operator.dt,saveat=ts)) : nothing
    ex=InferenceExperiment(:growth,base,measurement;setup,solver,dt=forward===:sectional ? dt : nothing)
    distributions=prior===:lognormal ? (LogNormal(log(0.4),0.4),LogNormal(log(0.25),0.6)) :
        (Uniform(UNIFORM_BOUNDS[1]...),Uniform(UNIFORM_BOUNDS[2]...))
    specs=[ParameterSpec(name;initial,transform=LogTransform(),prior=d)
        for (name,initial,d) in zip((:g0,:g1),(0.40,0.23),distributions)]
    EstimationProblem(specs,(ex,);derivatives=forward===:inguva ? :finite_difference : :auto)
end

function tutorial_problem(; prior=:lognormal)
    # Known continuous density + truncated-Gaussian observations, not a histogram.
    data = synthetic_measurements(; seed=2022)
    estimation_problem(data; prior)
end

function main()
    problem = tutorial_problem()
    fitted = fit(problem; nstarts=3, seed=22, maxiters=500)
    fitted.converged || error("MAP failed; inspect the fit diagnostics")
    @show fitted.parameters.shared fitted.convention
    @show posterior_components(problem, fitted.minimizer)
    predictions(problem, fitted.minimizer)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
