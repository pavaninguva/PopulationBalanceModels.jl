@testset "Integrated sources and moment evolution" begin
    @testset "Source number, tails, narrow peaks and bulk exchange" begin
        g=Grid1D([0.0,0.1,0.4,1.0,2.0];coordinate=:length)
        N=zeros(4,1); dN=similar(N); J=(p,t,c)->3.0
        gauss=GaussianNucleation(J;center=0.15,width=1e-6)
        op=compile(PBMModel(gauss),g)
        rhs!(dN,N,op,(;),0.0)
        @test sum(dN)≈3.0 rtol=1e-14
        @test dN[2]≈3.0 rtol=1e-14
        @test count(!iszero,dN)==1
        physical=compile(PBMModel(GaussianNucleation(J;center=0.0,width=0.1,truncation=:physical)),g)
        rhs!(dN,N,physical,(;),0.0)
        @test sum(dN)≈1.5 rtol=2e-14
        @test PopulationBalanceModels._normal_fraction(-12.0,12.0)≈1.0 atol=2e-14
        @test PopulationBalanceModels._normal_fraction(-1.0,1.0)≈0.6826894921370859 atol=2e-14
        @test PopulationBalanceModels._normal_fraction(8.0,9.0)≈6.21983198586583e-16 rtol=1e-11
        integrated=compile(PBMModel(IntegratedBirth((lo,hi,p,t,c)->(hi^3-lo^3)/3)),g)
        rhs!(dN,N,integrated,(;),0.0)
        @test sum(dN)≈8/3
        sys=PBMSystem((a=op,);exchanges=(ConservativeExchange(:a,:c,(x,p,t,c)->x),))
        prob=PBMProblem(sys,(a=zeros(4),),(0.0,0.1);bulk0=(c=2.0,))
        sol=simulate(prob;dt=0.01)
        @test bulk(prob,sol.u[end],:c)+moment(g,population(prob,sol.u[end],:a),1)≈2.0
        @test_throws ArgumentError GaussianNucleation(J;center=0.1,width=0.1,truncation=:bad)
        @test_throws ArgumentError compile(PBMModel(),g;bc=SizeBC(InflowFlux(J),NoFlux()))
    end
    @testset "Exactly closed equations and conservation" begin
        g=MomentSet(0:3)
        μ=reshape([1.0,2.0,5.0,14.0],:,1); dμ=similar(μ)
        G=Growth(PolynomialRate((0.2,0.1)))
        nuc=Nucleation((p,t,c)->0.3,0.4)
        agg=Aggregation(ConstantKernel(0.5))
        br=Breakage(PolynomialRate((0.2,)),SelfSimilarDaughters((0.5,),(2.0,)))
        op=compile(PBMModel(G+nuc+agg+br),g)
        rhs!(dμ,μ,op,(;),0.0)
        expected=[0.3-0.25+0.2,
            0.2+0.1*2+0.3*0.4,
            2*(0.2*2+0.1*5)+0.3*0.4^2+0.5*2^2+0.2*(0.5-1)*5,
            3*(0.2*5+0.1*14)+0.3*0.4^3+3*0.5*2*5+0.2*(0.25-1)*14]
        @test vec(dμ)≈expected
        @test moment(g,vec(μ),3)==14.0
        @test_throws ArgumentError compile(PBMModel(Growth(PolynomialRate((0.0,0.0,1.0)))),g;moment_method=ExactClosure())
        @test_throws ArgumentError compile(PBMModel(Growth((x,p,t,c)->sqrt(x))),g;moment_method=ExactClosure())
        @test_throws ArgumentError compile(PBMModel(Birth((x,p,t,c)->exp(-x))),g)
        @test_throws ArgumentError MomentSet([0,2])
        @test !realizability(g,[1.0,1.0,0.5,0.2]).valid
        @test_throws ArgumentError PBMProblem(op,[1.0,1.0,0.5,0.2],(0.0,1.0))
        # Exact constant-kernel μ0 has a known solution; μ1 stays constant.
        aop=compile(PBMModel(Aggregation(ConstantKernel(0.5))),g)
        prob=PBMProblem(aop,μ,(0.0,1.0)); sol=simulate(prob;dt=0.01)
        @test sol.u[end][1]≈1/1.25 rtol=2e-8
        @test sol.u[end][2]≈2.0 atol=2e-13
        birth=Nucleation((p,t,c)->1.2,0.0)
        sop=compile(PBMModel(birth),g)
        s=simulate(PBMProblem(sop,zeros(4),(0.0,0.5));dt=0.01)
        @test vec(s.u[end])≈[0.6,0,0,0]
        mop=compile(PBMModel(MomentSource((p,t,c)->2.0,(k,p,t,c)->factorial(k))),g)
        rhs!(dμ,μ,mop,(;),0.0)
        @test vec(dμ)≈[2,2,4,12]
    end
    @testset "QMOM inversion and weak-form event integrals" begin
        g=MomentSet(0:3)
        x=[0.5,2.0]; w=[0.7,0.3]; μ=[sum(w.*x.^k) for k in 0:3]
        q=quadrature(g,μ,QuadratureClosure(2))
        @test q.nodes≈x rtol=1e-12
        @test q.weights≈w rtol=1e-12
        @test_throws NumericalFailure quadrature(g,[1,1,1,1],QuadratureClosure(2))
        @test_throws NumericalFailure quadrature(g,[0,1,1,1],QuadratureClosure(2))
        @test quadrature(g,zeros(4)).weights==zeros(2)
        @test_throws ArgumentError compile(PBMModel(),MomentSet(0:2);closure=QuadratureClosure(2))
        model=PBMModel(Growth(PolynomialRate((0.2,0.1)))+
            Death(PolynomialRate((0.03,)))+Aggregation(ConstantKernel(0.5))+
            Breakage(PolynomialRate((0.1,)),SelfSimilarDaughters((0.5,),(2.0,))))
        eop=compile(model,g); qop=compile(model,g;closure=QuadratureClosure(2))
        a=zeros(4,1); b=similar(a); rhs!(a,reshape(μ,:,1),eop,(;),0.0); rhs!(b,reshape(μ,:,1),qop,(;),0.0)
        @test a≈b rtol=1e-12
        nonlinear=compile(PBMModel(Growth((x,p,t,c)->sqrt(x))),g;closure=QuadratureClosure(2))
        rhs!(b,reshape(μ,:,1),nonlinear,(;),0.0)
        @test vec(b)≈[0;[k*sum(w.*x.^(k-0.5)) for k in 1:3]] rtol=1e-12
        diam=MomentSet(0:3;coordinate=:diameter)
        dop=compile(PBMModel(Aggregation(ConstantKernel(0.5);coordinate_map=AdditiveCoordinate(x->x^3,cbrt))),diam;closure=QuadratureClosure(2))
        rhs!(b,reshape(μ,:,1),dop,(;),0.0)
        @test b[4]≈0.0 atol=1e-14
        @test_throws ArgumentError compile(PBMModel(Aggregation(ConstantKernel(1.0))),diam)
    end
    @testset "Coupled moment populations and inference" begin
        g=MomentSet(0:3)
        G=Growth(PolynomialRate(((p,t,c)->p.k*c.bulk.c,)))
        op=compile(PBMModel(G+Nucleation((p,t,c)->0.1,0.2)),g)
        sys=PBMSystem((a=op,b=op);exchanges=(MomentExchange(:a,:c,1),MomentExchange(:b,:c,1)),nonnegative_bulk=(:c,))
        init=[1.0,1.0,2.0,6.0]
        prob=PBMProblem(sys,(a=init,b=2init),(0.0,0.2);bulk0=(c=2.0,),p=(k=0.1,))
        sol=simulate(prob;dt=0.001)
        for u in sol.u
            @test bulk(prob,u,:c)+moment(g,population(prob,u,:a),1)+moment(g,population(prob,u,:b),1)≈5.0 atol=2e-12
        end
        @test moment(PopulationBalanceModels._system_context(prob.operator,prob.u0,zero(prob.u0),prob.p,0.0),:a,2)==2.0
        decay=compile(PBMModel(Death(PolynomialRate(((p,t,c)->p.k,)))),g)
        dp=PBMProblem(decay,init,(0.0,1.0);p=(k=0.4,))
        ts=[0.2,0.5,1.0]
        obs=Observation(ts,exp.(-0.4.*ts),MomentObservable(0);sigma=0.01)
        cal=CalibrationProblem([Experiment(dp,obs)];parameters=(z,e)->(k=exp(z[1]),),dt=0.01)
        @test maximum(abs,residuals(cal,[log(0.4)]))<1e-7
        @test sensitivity_report(cal,[log(0.4)]).rank==1
    end
end
