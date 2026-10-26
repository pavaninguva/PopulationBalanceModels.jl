using Test, PopulationBalanceModels, LinearAlgebra, ForwardDiff
using NonlinearSolve, Optim, SciMLBase, OrdinaryDiffEqTsit5
using Distributions: Normal

@testset "Coupled tensor diffusion through optional solvers and estimation" begin
    grid=Grid2D([0.,.2,.5,1.],[0.,.4,1.,2.])
    operator=compile(PBMModel(
        source=Birth((xy,p,t,ctx)->p.b*ctx.bulk.c),
        dispersion=SizeDiffusion((xy,p,t,ctx)->(.02ctx.bulk.c*(1+xy[1]),.03ctx.bulk.c*(1+xy[2]))),
        withdrawal=Death((xy,p,t,ctx)->p.d),
    ),grid)
    system=PBMSystem((particles=operator,);
        bulk_rhs=(ctx,p,t)->(c=p.F*(p.feed-ctx.bulk.c),),
        # Withdrawal leaves the vessel as particles, without dissolving.
        exchanges=(ConservativeExchange(:particles,:c,(xy,p,t,ctx)->p.alpha;
            mechanisms=(:birth,)),),nonnegative_bulk=(:c,))
    p=(b=.3,d=.4,F=.7,feed=3.,alpha=.2)
    initial = 0.2 .* (grid.x.widths*grid.y.widths')
    transient=PBMProblem(system,(particles=initial,),(0.,.2);bulk0=(c=2.,),p)
    native=simulate(transient;dt=.001,method=RK4())
    external=SciMLBase.solve(odeproblem(transient),Tsit5();reltol=1e-10,abstol=1e-12)
    @test SciMLBase.successful_retcode(external)
    @test last(external.u)≈last(native.u) rtol=1e-8 atol=1e-10

    area=last(grid.x.edges)*last(grid.y.edges)
    concentration(b)=p.F*p.feed/(p.F+p.alpha*b*area)
    number(b)=b*area*concentration(b)/p.d
    dc=-p.F*p.feed*p.alpha*area/(p.F+p.alpha*p.b*area)^2
    dn=area*(concentration(p.b)+p.b*dc)/p.d
    steady=StationaryPBMProblem(system,(particles=initial,);bulk=(c=2.,),p,autonomous=true)
    result=solve_stationary(steady,NewtonRaphson();abstol=1e-12)
    @test result.converged
    @test bulk(result.problem,result.u,:c)≈concentration(p.b) atol=1e-10
    @test sum(population(result.problem,result.u,:particles))≈number(p.b) atol=1e-10
    @test observe(PopulationMoment(:particles,(0,0)),result)≈[number(p.b)] atol=1e-10
    observations=(
        Measurement([number(p.b)],PopulationMoment(:particles,(0,0));likelihood=GaussianLikelihood(sigma=.01)),
        Measurement([concentration(p.b)],BulkObservable(:c);likelihood=GaussianLikelihood(sigma=.01)),
    )
    parameter=ParameterSpec(:b;initial=.25,transform=LogTransform(),
        prior=Prior(Normal(log(p.b),1.);coordinates=:inference))
    experiment=InferenceExperiment(:steady,steady,observations...;solver=StationarySolver(NewtonRaphson()))
    problem=EstimationProblem((parameter,),(experiment,))
    z=[log(p.b)]
    values(q)=vcat((vec(x) for x in predictions(problem,q)[1])...)
    @test values(z)≈[number(p.b),concentration(p.b)] atol=1e-10
    @test vec(ForwardDiff.jacobian(values,z))≈p.b.*[dn,dc] rtol=1e-8
    @test check_gradient(problem,[log(.28)]).valid
    fitted=fit(problem;uncertainty=:gauss_newton)
    @test fitted.converged
    @test fitted.parameters.shared.b≈p.b rtol=1e-6
    @test fitted.uncertainty.covariance[1,1]≈1/(1+p.b^2*(dn^2+dc^2)/.01^2) rtol=1e-7
end
