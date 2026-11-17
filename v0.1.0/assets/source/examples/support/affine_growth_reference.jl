# Exact affine-growth densities/bin integrals and seeded synthetic point data.
# No optimizer, simulation, output file or posterior sampling is run on include.
const TRUTH=(g0=0.434,g1=0.2604)
const TIMES=[0.1,0.25,0.4,0.6,0.8,1.0]
const SIZES=collect(range(0.0,1.4;length=281))
const CENTER=0.2
const WIDTH=sqrt(0.0005/2)
const INITIAL_NUMBER=50sqrt(π*0.0005)
const UNIFORM_BOUNDS=((0.05,1.0),(0.02,1.0))
const LAW=AffineGrowth(:g0,:g1)
initial_density(a)=50exp(-(a-CENTER)^2/0.0005)
initial_profile()=InitialDistribution(Normal(CENTER,WIDTH);
    total=INITIAL_NUMBER,truncation=:physical)
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

function analytical_solver(pb,times)
    states=[reshape(exact_counts(pb.operator.grid.edges,t,pb.p),:,1) for t in times]
    PBMSolution(times,states,pb,0)
end

# This state/mesh is context for the exact forward solution, not a numerical
# approximation used by its point-density likelihood. Keep the reference layout.
function analytical_context(model;p=TRUTH)
    grid=characteristic_grid(LAW,(0.0,1.4);dt=0.005,p)
    operator=compile(model,grid;evolution=Inguva2022(),p,
        bc=SizeBC(OpenBoundary(),OpenBoundary()))
    PBMProblem(operator,initial_profile(),(0.0,1.0);p)
end
