module ForwardExecutionIntegrationTests
using Test, PopulationBalanceModels, SciMLBase, OrdinaryDiffEqTsit5, ForwardDiff
import PopulationBalanceModels: remake
using Distributions: Uniform, Normal

# Malformed successful solutions exercise the adapter's completion contract,
# independently of the native solution constructor and of ODE return status.
struct IncompleteAlgorithm
    failure::Symbol
end
function SciMLBase.solve(pb::SciMLBase.ODEProblem,alg::IncompleteAlgorithm;saveat,kwargs...)
    ts=collect(saveat);states=[copy(pb.u0) for t in ts]
    if alg.failure===:premature
        ts=ts[1:1];states=states[1:1]
    elseif alg.failure===:missing
        deleteat!(ts,2);deleteat!(states,2)
    elseif alg.failure===:shape
        states=[zeros(length(pb.u0)+1) for t in ts]
    elseif alg.failure===:nonfinite
        states[end].=NaN
    end
    SciMLBase.build_solution(pb,alg,ts,states;retcode=SciMLBase.ReturnCode.Success)
end

@testset "Checked external execution and restarted operating intervals" begin
    g=Grid1D([0.,.2,.6,1.])
    op=compile(PBMModel(removal=Death((x,p,t,c)->p.k)),g)
    pb=PBMProblem(op,g.widths,(0.,1.);p=(k=.4,))
    solver=ForwardSolver(Tsit5();backend=:external,reltol=1e-11,abstol=1e-13)
    ts=[0.,.4,1.]
    sol=solver(pb,ts)
    @test sol.t==ts && sol.problem===pb
    @test [sum(u) for u in sol.u]≈exp.(-.4ts) rtol=2e-11
    @test diagnostics(sol).validation===:saved_states
    @test diagnostics(sol).accepted_steps>0
    @test_throws NumericalFailure simulate(pb;solver=ForwardSolver(Tsit5();backend=:external,maxsteps=1))
    for failure in (:premature,:missing,:nonfinite)
        @test_throws NumericalFailure simulate(pb;solver=ForwardSolver(IncompleteAlgorithm(failure);backend=:external),saveat=ts)
    end
    @test_throws DimensionMismatch simulate(pb;solver=ForwardSolver(IncompleteAlgorithm(:shape);backend=:external),saveat=ts)
    history=OperatingHistory([.4];parameters=((k=.3,),(k=1.1,)))
    changed=remake(pb;history)
    piece=solver(changed,ts)
    @test [sum(u) for u in piece.u]≈[1.,exp(-.12),exp(-.78)] rtol=3e-11
    @test piece.problem===changed
    @test length(diagnostics(piece).segments)==2
    @test_throws ArgumentError odeproblem(changed)
    @test observe(FunctionObservable((u,op,p,t)->p.k),piece;index=2)==[1.1]
    # A solver's successful return alone is insufficient: bulk state validity
    # is a separate check. No clipping or backend stage guarantee is implied.
    system=PBMSystem((A=compile(PBMModel(),g),);bulk_rhs=(c,p,t)->(c=-1.,),nonnegative_bulk=(:c,))
    bad=PBMProblem(system,(A=zeros(3),),(0.,1.);bulk0=(c=.1,))
    @test_throws NumericalFailure solver(bad,[1.])
end

@testset "One solver in direct, inference and data-free UQ calls" begin
    g=Grid1D([0.,.4,1.])
    op=compile(PBMModel(removal=Death((x,p,t,c)->p.k)),g)
    initial=InitialDistribution(Uniform(0.,1.);total=(p,t)->p.N)
    base=PBMProblem(op,initial,(0.,.4);p=(k=.4,N=2.))
    solver=ForwardSolver(Tsit5();backend=:external,reltol=1e-10,abstol=1e-12)
    measurement=Measurement([.4],reshape([2exp(-.16)],1,1),MomentObservable(0);likelihood=GaussianLikelihood(sigma=.1))
    experiment=InferenceExperiment(:experiment,base,measurement;solver)
    problem=EstimationProblem([ParameterSpec(:k;initial=.4,prior=Normal(.4,.1),transform=IdentityTransform())],(experiment,))
    @test isfinite(logposterior(problem,initial_coordinates(problem)))
    legacy_solver=(pb,ts)->begin
        sol=solver(pb,ts)
        (t=sol.t,u=sol.u)
    end
    legacy_ex=InferenceExperiment(:legacy,base,measurement;solver=legacy_solver)
    legacy_problem=EstimationProblem(problem.parameters,(legacy_ex,))
    @test logposterior(legacy_problem,initial_coordinates(legacy_problem))≈logposterior(problem,initial_coordinates(problem))
    inputs=UncertainInputs((k=Uniform(.2,.6),))
    uq=UQProblem(base,inputs;outputs=(number=MomentObservable(0),),times=[.4],solver)
    result=propagate(uq,MonteCarlo(3);samples=reshape([.2,.4,.6],1,3),max_solves=3)
    @test result.values≈reshape(2exp.(-.4 .* [.2,.4,.6]),1,3) rtol=3e-10
    @test diagnostics(result).solves==3
    @test all(work->work.accepted_steps>0,diagnostics(result).integration_work)
    @test_throws ArgumentError propagate(uq,MonteCarlo(3);max_solves=2)
    # Generic external execution must not silently differentiate adaptive step
    # decisions. Continuous sensitivities have a separate tested adapter.
    f=N->sum(last(solver(remake(base;p=(k=.4,N=N,)),[.4]).u))
    @test_throws ArgumentError ForwardDiff.derivative(f,2.)
    @test (f(2.0001)-f(1.9999))/.0002≈exp(-.16) rtol=3e-8
end
end
