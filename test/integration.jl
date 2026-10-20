using Test, PopulationBalanceModels, Optim, SciMLBase, OrdinaryDiffEqTsit5

@testset "Optional package extensions" begin
    grid=Grid1D(range(0,8;length=31))
    op=compile(PBMModel(Death((v,p,t,c)->p.k)),grid)
    initial=exp.(-grid.centers).*grid.widths
    prob=PBMProblem(op,initial,(0.0,2.0);p=(k=0.4,))
    deprob=odeproblem(prob)
    sol=SciMLBase.solve(deprob,Tsit5();reltol=1e-9,abstol=1e-11,saveat=[0.0,1.0,2.0])
    @test SciMLBase.successful_retcode(sol)
    @test vec(sol.u[end])≈initial.*exp(-0.8) rtol=1e-7
    ts=[0.0,0.4,0.9,1.5,2.0]
    obs=Observation(ts,sum(initial).*exp.(-0.4.*ts),MomentObservable(0);sigma=0.02)
    cal=CalibrationProblem([Experiment(prob,obs)];parameters=(z,ex)->(k=exp(z[1]),),dt=0.01)
    result=fit(cal,[log(0.15)];maxiters=100)
    @test result.converged
    @test exp(result.minimizer[1])≈0.4 rtol=1e-5
    @test result.failed_evaluations>=0
    unstable=CalibrationProblem([Experiment(prob,obs)];parameters=(z,ex)->(k=1e4,),dt=0.1)
    @test_throws NumericalFailure fit(unstable,[0.0])
end

@testset "Native moment model SciML solve and fitting" begin
    g=MomentSet(0:3)
    op=compile(PBMModel(Death(PolynomialRate(((p,t,c)->p.k,)))),g)
    initial=[1.0,1.0,2.0,6.0]
    prob=PBMProblem(op,initial,(0.0,1.0);p=(k=0.4,))
    sol=SciMLBase.solve(odeproblem(prob),Tsit5();reltol=1e-10,abstol=1e-12)
    @test SciMLBase.successful_retcode(sol)
    @test vec(sol.u[end])≈initial.*exp(-0.4) rtol=1e-9
    times=[0.2,0.5,1.0]
    data=Observation(times,exp.(-0.4.*times),MomentObservable(0);sigma=0.01)
    cal=CalibrationProblem([Experiment(prob,data)];parameters=(z,e)->(k=exp(z[1]),),dt=0.01)
    result=fit(cal,[log(0.2)])
    @test result.converged
    @test exp(result.minimizer[1])≈0.4 rtol=1e-6
end

@testset "Coupled SciML solve, inference and Gunawan moment reference" begin
    grid=Grid1D(0.5:1:4.5)
    op=compile(PBMModel(Nucleation((p,t,c)->p.k*c.bulk.c,1.0)),grid)
    sys=PBMSystem((A=op,);exchanges=(ConservativeExchange(:A,:c,(x,p,t,c)->x),),nonnegative_bulk=(:c,))
    prob=PBMProblem(sys,(A=zeros(4),),(0.0,1.0);bulk0=(c=2.0,),p=(k=0.3,))
    coupled=SciMLBase.solve(odeproblem(prob),Tsit5();reltol=1e-9,abstol=1e-11)
    @test SciMLBase.successful_retcode(coupled)
    @test bulk(prob,coupled.u[end],:c)≈2exp(-0.3) rtol=1e-7
    times=[0.0,0.3,0.7,1.0]
    obs1=Observation(times,2 .*exp.(-0.3.*times),BulkObservable(:c);sigma=0.02)
    obs2=Observation(times,2 .*(1 .-exp.(-0.3.*times)),PopulationMoment(:A,0);sigma=0.02)
    cal=CalibrationProblem([Experiment(prob,obs1,obs2)];parameters=(z,e)->(k=exp(z[1]),),dt=0.01)
    result=fit(cal,[log(0.1)])
    @test result.converged
    @test exp(result.minimizer[1])≈0.3 rtol=1e-5

    include("../examples/papers/gunawan2004_case2.jl")
    seed=L->180.5<=L<=210.5 ? 0.08*(1-((L-195.5)/15)^2) : 0.0
    errors=Float64[]
    for bins in (400,800)
        g=Grid1D(range(0.0,1600.0;length=bins+1);coordinate=:length)
        pb=Gunawan2004Case2.build(;grid=g,initial_density=seed,
            c0=Gunawan2004Case2.solubility(32.0),tspan=(0.0,50.0))
        N0=population(pb,pb.u0,:crystals)
        init=vcat([moment(g,N0,k) for k in 0:3],bulk(pb,pb.u0,:c))
        refprob=SciMLBase.ODEProblem(Gunawan2004Case2.moment_rhs!,init,pb.tspan,pb.p)
        reference=SciMLBase.solve(refprob,Tsit5();reltol=1e-11,abstol=1e-12)
        native=Gunawan2004Case2.build_moments(initial_moments=init[1:4],c0=init[5],tspan=pb.tspan)
        native_sol=SciMLBase.solve(odeproblem(native),Tsit5();reltol=1e-11,abstol=1e-12)
        @test SciMLBase.successful_retcode(native_sol)
        @test native_sol.u[end]≈reference.u[end] rtol=1e-9
        solution=simulate(pb;dt=0.02,saveat=[50.0])
        numerical=moment(g,population(pb,solution.u[end],:crystals),3)
        push!(errors,abs(numerical-reference.u[end][4])/reference.u[end][4])
        @test SciMLBase.successful_retcode(reference)
    end
    @test errors[2]<errors[1]
    @test errors[2]<1e-3
    println("Gunawan demo third-moment relative errors (400,800 bins): ",errors)
end

include("../examples/shin_lnp_reproduction.jl")
@testset "Shin Figure 4 exporter with a SciML output schedule" begin
    g=Grid1D([1e-9,3e-9,10e-9,30e-9];coordinate=:diameter)
    prob=ShinLNPReproduction.figure4_problem(;grid=g,tspan=(0.0,0.001),
        temperature=t->298.15,viscosity=(w,T)->1e-3,solubility=(w,T)->1.0,
        equilibrium_ratio=(L,w,T)->1.0,attachment_without_peg=(L,M,w,T)->1.0,
        molecular_volume=5e-28,particle_density=1000.0,nucleation_width=1e-9,
        nucleation_normalization=:total_rate,kg=0.0,kPEG=0.0,An=0.0)
    function external(p,times)
        result=SciMLBase.solve(odeproblem(p),Tsit5();saveat=times,save_everystep=false,
            reltol=1e-9,abstol=1e-12)
        SciMLBase.successful_retcode(result) || error("test integration failed")
        result
    end
    mktempdir() do output
        solution=ShinLNPReproduction.run_case(prob;saveat=[0.0,0.0003,0.001],solver=external,
            provenance="Synthetic zero-kinetics adapter check; not a paper reproduction",output)
        @test solution.t==[0.0,0.0003,0.001]
        @test all(u->iszero(sum(population(prob,u,:lnp))),solution.u)
    end
end
