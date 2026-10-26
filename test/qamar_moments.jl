include("fixtures/example_modules.jl")
_load_example(:QamarMomentBenchmarks, "qamar_moment_benchmarks.jl")
_load_example(:LaxWendroffAudit, "verification/lax_wendroff.jl")

@testset "Continuous daughters and Qamar moment balances" begin
    d=UniformVolumeDaughters(quadrature_order=8);y=.7
    pairs=collect(d(y,(;),0.,(;)))
    @test sum(last,pairs)≈2
    @test sum(c*x^3 for (x,c) in pairs)≈y^3
    @test [daughter_moment(d,y,k,(;),0.,(;)) for k in 0:8]≈[6y^k/(k+3) for k in 0:8]
    @test_throws ArgumentError UniformVolumeDaughters(quadrature_order=2)
    q=QamarMomentBenchmarks
    pb=q.problem4();du=similar(pb.u0);rhs!(du,pb.u0,pb.operator,(;),0.)
    @test du[1]≈.001-.5
    @test du[4]≈0 atol=1e-16
    sol=simulate(pb;dt=.02,saveat=[0.,10.,50.])
    @test vec(sol.u[end])[[1,4]]≈q.exact4(50.)[[1,4]] rtol=1e-8
    @test maximum(abs.(vec(sol.u[end])./q.exact4(50.).-1))<.006
    # Time convergence is not mistaken for convergence of the quadrature closure.
    fine=simulate(pb;dt=.01)
    @test maximum(abs.(vec(sol.u[end])./vec(fine.u[end]).-1))<1e-7
    # Direct-state methods must initialize weights/weighted abscissas, not
    # reinterpret six raw moments as those six different solver variables.
    direct=q.problem4(method=DQMOM(3))
    @test [moment_value(direct.operator.closure,direct.operator.grid,direct.u0,k) for k in 0:5]≈q.exact4(0.)
    dp=q.problem5(method=DQMOM(3))
    @test [moment_value(dp.operator.closure,dp.operator.grid,dp.u0,k) for k in 0:5]≈[q.initial_moment5(k) for k in 0:5]
    p5=q.problem5(collisions=false);data=simulate(p5;dt=.00025,saveat=[0.,.2,.25,.5])
    for (t,u) in zip(data.t,data.u)
        @test vec(u)≈q.exact5(t) rtol=1e-7
    end
    # Eq.43 prescribes density, whereas Nucleation expects G*n(0,t).
    @test q.nucleation((G=2.,),.215,(;))≈2*(100+1e6)
    # Continuous fragmentation's moments conserve volume exactly, even when
    # higher powers are not resolved by a small daughter quadrature.
    basis=MomentSet(0:5;coordinate=:length)
    for method in (QMOM(3),DQMOM(3))
        model=PBMModel(Breakage((x,p,t,c)->x^6,d;coordinate_map=AdditiveCoordinate(x->x^3,cbrt)))
        x=[.2,.5,.8];w=[.2,.5,.3]
        raw=[sum(w.*x.^k) for k in 0:5]
        state=method isa DQMOM ? quadrature_state(method,x,w) : raw
        op=compile(model,basis;method);derivative=zeros(6,1)
        rhs!(derivative,reshape(state,:,1),op,(;),0.)
        expected=[(6/(k+3)-1)*sum(w.*x.^(k+6)) for k in 0:5]
        @test [moment_rate(method,basis,state,derivative,k) for k in 0:5]≈expected rtol=1e-8 atol=1e-14
    end
end

@testset "Lax–Wendroff reference and boundary audit" begin
    C=.5;v=1.;dx=.1;N=[1.,2.,3.,2.,1.,.5,.2,.1]
    g=Grid1D(range(0.,length(N)*dx;length=length(N)+1))
    op=compile(PBMModel(Growth(AffineGrowth(v,0.))),g;evolution=LaxWendroff(),bc=SizeBC(OpenBoundary(),OpenBoundary()))
    out=vec(simulate(PBMProblem(op,N,(0.,C*dx));dt=C*dx,check_nonnegative=false).u[end])
    ref=LaxWendroffAudit.reference_step(N,C)
    @test out[2:end-1]≈ref[2:end-1] atol=1e-14
    # Zero total inlet flux is distinct from a zero exterior nodal value.
    @test out[1]≈(1-C/2-C^2/2)*N[1]+(C^2-C)/2*N[2]
    @test sum(out)≈sum(N)-C*N[end] atol=1e-14
    f=100exp.(-collect(0.:.005:1.)./.01)
    for _ in 1:160;f=LaxWendroffAudit.reference_step(f,.5);end
    @test minimum(f)<-10
    f=100exp.(-collect(0.:.005:1.)./.01);u0=copy(f)
    for _ in 1:80;f=LaxWendroffAudit.reference_step(f,1.);end
    @test f≈vcat(zeros(80),u0[1:end-80]) atol=1e-13
end

@testset "Resolved growth/source reference" begin
    q=QamarMomentBenchmarks;errors=Float64[]
    for n in (120,240)
        pb=q.sectional5(bins=n,collisions=false)
        sol=q.sectional_solve(pb,[0.,.5])
        @test sum(last(sol.u))≈q.exact5(.5)[1] rtol=1e-12
        push!(errors,abs(only(moment(pb.operator.grid,last(sol.u),2))/q.exact5(.5)[3]-1))
    end
    @test errors[1]/errors[2]>3.5
end

@testset "Case-study family state layouts and reconstructed material" begin
    q=QamarMomentBenchmarks
    for (method,length_expected) in ((QMOM(3),6),(RankAdaptiveQMOM(3),6),(EQMOM(3),7),
                                    (DQMOM(3),6),(LogNormalClosure(quadrature_order=64),3))
        pb=q.problem4(;method,tf=.02)
        @test length(pb.u0)==length_expected
        @test [q.comparison_moment(pb.operator,pb.u0,k) for k in 0:2]≈q.exact4(0.)[1:3] rtol=1e-8
        sol=simulate(pb;dt=.005)
        @test all(isfinite,sol.u[end])
    end
    pb=q.problem4(method=LogNormalClosure(quadrature_order=64),tf=.02)
    @test !(3 in pb.operator.grid.orders)
    @test abs(q.comparison_moment(pb.operator,pb.u0,3)/q.exact4(0.)[4]-1)>.01
    # In a closed raw hierarchy, changing quadrature family cannot change the
    # analytical growth/source moments when its reconstruction remains resolved.
    pb=q.problem5(method=LogNormalClosure(quadrature_order=64),collisions=false,tf=.25)
    sol=simulate(pb;dt=.0002)
    @test vec(sol.u[end])≈q.exact5(.25)[1:3] rtol=2e-6
    run=q.comparison_run((;method)->q.problem4(;method,tf=.02),QMOM(3),[0.,.01,.02],.005)
    @test run.status=="completed" && last(run.t)==.02
end

include("fixtures/qamar_reference.jl")
@testset "Independent characteristic and Gamma feasibility references" begin
    ref=QamarReference
    setprecision(256) do
        # Every required order, including the extra EQMOM order M6; resolve the pulse analytically.
        for time in ("0.0","0.205","0.215","0.225","0.475","0.4788","0.48","0.5")
            moments=ref.characteristic_moments(time)
            @test Float64.(moments)≈QamarMomentBenchmarks.exact5(parse(Float64,time),0:6) rtol=3e-12
        end
        g=MomentSet(0:6)
        moments=ref.characteristic_moments(".4788")
        certificate=ref.gamma_certificate(moments)
        @test certificate.error<1e-12
        @test certificate.condition>1e10
        @test minimum(certificate.weights)>0
        q=quadrature(g,Float64.(moments),EQMOM(3))
        @test q.inversion_precision in (53,256)
        @test [sum(q.weights.*q.nodes.^k)/Float64(moments[k+1]) for k in 0:6]≈ones(7) rtol=1e-7
        @test q.width≈Float64(certificate.width) rtol=1e-8
        # A reproduced RK predictor triggers arithmetic refinement even when
        # nearby rounded reference inputs happen to pass the Float64 path.
        predictor=[17792.436509054474,4706.799347088809,1257.2789062897127,
            346.44390790325554,105.63609414313751,41.987184420369886,25.450594713550938]
        refined=quadrature(g,predictor,EQMOM(3))
        @test refined.inversion_precision in (53,256)
        @test [sum(refined.weights.*refined.nodes.^k)/predictor[k+1] for k in 0:6]≈ones(7) rtol=1e-7
        retry=PopulationBalanceModels._gamma_primary_precise(predictor,EQMOM(3))
        @test retry.inversion_precision==256
        @test [sum(w*prod((x+j*retry.width for j in 0:k-1);init=1.) for
            (x,w) in zip(retry.nodes,retry.weights))/predictor[k+1] for k in 0:6]≈ones(7) rtol=1e-7
        # A full allowed-width search, independent of the package's boundary bisection.
        for bits in (256,512)
            reference=setprecision(()->ref.characteristic_moments(".48"),BigFloat,bits)
            roots=ref.gamma_width_roots(reference;bits)
            @test length(roots)==1
            @test only(roots).shifted_margin< -1e-5
            @test realizability(g,Float64.(reference)).valid
            @test_throws NumericalFailure quadrature(g,Float64.(reference),EQMOM(3))
        end
    end
end

@testset "Independent lognormal high-order breakage rates" begin
    for variance in (.1,.3,.6,1.),order in (32,64,128)
        g=MomentSet(0:2;coordinate=:length);mu=[20.,10.,5exp(variance)]
        model=PBMModel(Breakage((x,p,t,c)->x^6,UniformVolumeDaughters();coordinate_map=AdditiveCoordinate(x->x^3,cbrt)))
        op=compile(model,g;method=LogNormalClosure(quadrature_order=order));du=zeros(3,1)
        rhs!(du,reshape(mu,:,1),op,NamedTuple(),0.)
        expected=QamarReference.lognormal_breakage_rates(20.,.5,variance)
        if order==32 && variance==1.
            # Resolving tracked M2 does not certify the rate's M8 integral.
            @test abs(du[3]/expected[3]-1)>1e-4
        else
            @test vec(du)≈expected rtol=2e-7
        end
    end
end
