import Distributions
@testset "Configured sensitivity dispatch and physical initialization" begin
    grid=Grid1D([0.,.4,1.])
    op=compile(PBMModel(removal=Death((x,p,t,c)->p.k)),grid)
    base=PBMProblem(op,InitialDistribution(Distributions.Uniform(0.,1.);total=(p,t)->p.N),
        (0.,.5);p=(k=.4,N=2.))
    ts=[0.,.2,.5]
    specs=[ParameterSpec(:k;initial=.4,transform=LogTransform(),prior=Distributions.LogNormal(log(.4),.2)),
        ParameterSpec(:N;initial=2.,transform=LogTransform(),prior=Distributions.LogNormal(log(2.),.2))]
    m=Measurement(ts,2exp.(-.4ts),MomentObservable(0);likelihood=GaussianLikelihood(sigma=.1))
    for mode in (:forward,:adjoint)
        backend=SciMLSensitivitySolver(Tsit5();sensitivity=mode,reltol=1e-10,abstol=1e-12)
        solver=ForwardSolver(backend;backend=:sensitivity)
        ex=InferenceExperiment(:decay,base,m;solver)
        problem=EstimationProblem(specs,(ex,))
        z=initial_coordinates(problem)
        # The likelihood residual vanishes; the transformed lognormal priors
        # also have zero gradient at these z medians.
        counts=PopulationBalanceModels._InferenceCounts()
        @test norm(posterior_gradient(problem,z;_counts=counts))<2e-7
        @test counts.solves==1
        @test counts.integration_accepted>0
        offset=z.+[.1,-.05]
        @test check_gradient(problem,offset;tolerance=2e-5).valid
        @test solver(base,ts).problem===base
        changed=PopulationBalanceModels.remake(base;history=OperatingHistory([.2];parameters=((k=.4,),(k=.8,))))
        @test_throws ArgumentError solver(changed,ts)
    end
end
