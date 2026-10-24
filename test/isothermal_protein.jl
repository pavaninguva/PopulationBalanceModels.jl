using Test, PopulationBalanceModels, ForwardDiff, LinearAlgebra
include(joinpath(@__DIR__,"..","examples","isothermal_protein_crystallization.jl"))
using .IsothermalProteinCrystallization

@testset "Conditional number observations" begin
    grid=Grid1D([0.,1.,3.,6.]);N=[1.,4.,0.]
    @test conditional_cdf(grid,N,[.5,1.,1.5,2.];window=(.5,2.)) ≈ [0.,.2,.6,1.]
    @test conditional_quantiles(grid,N,[.1,.2,.6,.9];window=(.5,2.)) ≈ [.75,1.,1.5,1.875]
    @test_throws NumericalFailure conditional_cdf(grid,zeros(3),[1.];window=(.5,2.))
    @test_throws NumericalFailure conditional_cdf(grid,[1.,-1.,2.],[1.];window=(.5,2.))
    D=ForwardDiff.Dual
    @test ForwardDiff.value.(conditional_cdf(grid,[D(1.,0.),D(0.,-1.),D(0.,0.)],[.5,1.5];window=(0.,3.)))==[.5,1.]
    @test_throws NumericalFailure conditional_cdf(grid,[D(0.,1.),D(0.,0.),D(0.,0.)],[1.];window=(0.,3.))
    @test cdf_quantiles([0.,1.,2.,3.],[0.,.5,.5,1.],[.25,.5,.75]) ≈ [.5,1.,2.5]
    for scale in (1.,1e-6),q in (256,512,1024)
        u=((1:q).-0.5)./q
        a=cdf_quantiles(scale.*[0.,2.],[0.,1.],u)
        b=cdf_quantiles(scale.*[3.,5.],[0.,1.],u)
        @test sum(abs2,a-b)/q ≈ 9scale^2
    end
    f(v)=conditional_quantiles(grid,v,[.1,.6];window=(.5,2.))
    h=1e-5
    @test ForwardDiff.jacobian(f,N)[:,1] ≈ (f(N+[h,0,0])-f(N-[h,0,0]))/(2h) rtol=1e-7
end

@testset "CNT protein model, units and conservative stage coupling" begin
    p=parameters();S=p.c0/p.cstar
    # Independent SI expression, including seconds/minutes and nm/μm conversions.
    B=16pi*(p.gamma/1000)^3*(2.97e-26)^2/(3*(1.380649e-23*293.15)^3)
    J=60exp(p.logAJ)*exp(-B/log(S)^2)
    G=p.AG*(S-1)^p.g/1000
    @test collect(rates(p,0.,p.c0)) ≈ [J,G]
    @test rates(p,0.,p.cstar)==(0.,0.)
    @test rates(p,0.,p.cstar/2)==(0.,0.)
    # Subnormal tails must not produce negative Dual-zero source values.
    @test IsothermalProteinCrystallization.nucleation_rate(merge(p,(gamma=100.,)),p.c0)==0
    @test ForwardDiff.derivative(c->IsothermalProteinCrystallization.nucleation_rate(merge(p,(gamma=100.,)),c),p.c0)==0
    pb=build(bins=30,tf=74.);op=pb.operator.system.populations.crystals
    @test all(iszero,population(pb,pb.u0,:crystals))
    for N in (zeros(30),exp.(-op.grid.centers))
        u=copy(pb.u0);population(pb,u,:crystals).=N
        du=similar(u);rhs!(du,u,pb.operator,p,0.)
        @test sum(population(pb,du,:crystals))≈J/1e12 rtol=1e-12
        @test bulk(pb,du,:c)+1e-6*p.rho*p.kv*moment(op.grid,population(pb,du,:crystals),3)≈0 atol=1e-12
        # Cross-check the documented derivative-based coupling against the
        # package's separate ConservativeExchange implementation.
        system=PBMSystem((crystals=op,);exchanges=(ConservativeExchange(:crystals,:c,
            (L,p,t,ctx)->1e-6*p.rho*p.kv*L^3),))
        reference=PBMProblem(system,(crystals=N,),(0.,74.);bulk0=(c=p.c0,),p)
        dr=similar(reference.u0);rhs!(dr,reference.u0,reference.operator,p,0.)
        @test dr≈du rtol=1e-12 atol=1e-12
    end
    sol=forward(p;bins=30,times=[0.,74.])
    @test minimum(last(sol.u))>=0
    @test bulk(pb,last(sol.u),:c)+1e-6*p.rho*p.kv*moment(op.grid,population(pb,last(sol.u),:crystals),3)≈p.c0 atol=1e-10
end

@testset "Bayesian CNT target and held-out exclusion" begin
    data=load_data()
    @test length(data.concentration)==16
    @test length(select_concentration(data))==11
    @test [first(r).time_min for r in select_psds(data)]==[74.,180.,420.]
    @test first(data.concentration).model_time_min==0
    @test first(data.concentration).concentration_g_L==IsothermalProteinCrystallization.C0
    b=estimation_problem(bins=30);z=initial_coordinates(b)
    ex=only(b.experiments)
    @test length(ex.measurements)==4
    @test size(first(ex.measurements).data)==(1,11)
    @test size(ex.measurements[2].data)==(22,1)
    @test [only(m.times) for m in ex.measurements[2:end]]==[74.,180.,420.]
    changed=merge(data,(concentration=[r.split=="validation" ? merge(r,(concentration_g_L=999.,)) : r for r in data.concentration],
        psd=[r.split=="validation" ? merge(r,(conditional_cdf=-999.,)) : r for r in data.psd]))
    @test logposterior(b,z)==logposterior(estimation_problem(data=changed,bins=30),z)
    @test check_gradient(b,z).valid
    # Independently normalize the first correlated Gaussian CDF block.
    predicted=predictions(b,z)[1];theta=physical_parameters(b,z,:batch)
    whitened_residual=vec(predicted[2]-ex.measurements[2].data)
    R=[exp(-abs(i-j)/2) for i in 1:22,j in 1:22]
    Rfactor=cholesky(Symmetric(R)).L
    residual=Rfactor*whitened_residual
    C=theta.sigma_F^2*R
    L=cholesky(Symmetric(C)).L
    expected=-.5*(sum(abs2,L\residual)+2sum(log,diag(L))+22log(2pi))
    actual=PopulationBalanceModels._measurement_loglikelihood(ex.measurements[2].likelihood,
        ex.measurements[2].data,predicted[2],theta,ex)
    @test actual≈expected+sum(log,diag(Rfactor)) rtol=1e-12
    # Nonconstant inverse-map Jacobian: an independent dense determinant.
    map=IsothermalProteinCrystallization.sampling_coordinates(b)
    for delta in (zeros(6),[.01,.02,-.03,.04,.02,-.02])
        at=z+delta;y=map.to_sampler(at)
        @test map.from_sampler(y)≈at rtol=1e-11
        @test map.logabsdetjac(y)≈logabsdet(ForwardDiff.jacobian(map.from_sampler,y))[1] rtol=1e-11
        @test check_sampling_map(b,map,at).valid
    end
    # In initial-rate y coordinates the Uniform(logAJ)/logit Jacobian
    # cancels its inverse-map factor. Check the entire transformed prior.
    y=map.to_sampler(z)
    H=-ForwardDiff.hessian(v->logprior(b,map.from_sampler(v))+map.logabsdetjac(v),y)
    sig=(theta.g-1)/2
    expected=[0.,1/.7^2,0.,2sig*(1-sig),theta.sigma_c,10theta.sigma_F]
    @test H≈Diagonal(expected) atol=1e-10
    tuning=IsothermalProteinCrystallization.sampler_tuning(b,z,map)
    @test isposdef(Symmetric(tuning.covariance))
    @test tuning.information*tuning.covariance≈Matrix{Float64}(I,6,6) atol=1e-8
    @test tuning.covariance[5,5]≈1/(2*11+theta.sigma_c)
    @test tuning.covariance[6,6]≈1/(2*66+10theta.sigma_F)
    y[1]=-1e6
    @test_throws NumericalFailure map.from_sampler(y)
end

@testset "Uniform polynomial growth reuses one stage rate without changing fluxes" begin
    for scheme in (Upwind(),MUSCL(),WENO5()),
        bc in (SizeBC(NoFlux(),OpenBoundary()),SizeBC(Periodic(),Periodic())),
        velocity in (-.2,.3)
        grid=Grid1D(range(0.,2pi;length=21))
        N=(2 .+ sin.(grid.centers)).*grid.widths
        callback=(L,p,t,ctx)->p.v*(1+t)
        polynomial=PolynomialRate(((p,t,ctx)->p.v*(1+t),))
        a=compile(PBMModel(Growth(callback)),grid;growth_scheme=scheme,bc)
        b=compile(PBMModel(Growth(polynomial)),grid;growth_scheme=scheme,bc)
        function residual(v,op)
            state=reshape(eltype(v).(N),:,1);du=similar(state)
            rhs!(du,state,op,(v=only(v),),.4)
            vec(du)
        end
        @test residual([velocity],a)==residual([velocity],b)
        @test ForwardDiff.jacobian(v->residual(v,a),[velocity])≈
              ForwardDiff.jacobian(v->residual(v,b),[velocity]) rtol=1e-13
    end
end

@testset "Coupled initial-state checks inspect physical values during AD" begin
    op=compile(PBMModel(),Grid1D([0.,1.,2.]))
    system=PBMSystem((crystals=op,);nonnegative_bulk=(:c,),
        accounting=(CumulativeAccounting(:crystals;properties=(material=3,)),))
    D=ForwardDiff.Dual
    zero_with_tangent=D(0.,-1.)
    pb=PBMProblem(system,(crystals=[zero_with_tangent,D(0.,0.)],),(0.,1.);
        bulk0=(c=zero_with_tangent,))
    @test ForwardDiff.value(first(pb.u0))==0
    @test ForwardDiff.partials(first(pb.u0))[1]==-1
    restarted=remake(pb;u0=pb.u0,tspan=(.5,1.))
    @test restarted.u0==pb.u0
    @test_throws ArgumentError PBMProblem(system,(crystals=[D(-1e-100,1.),D(0.,0.)],),(0.,1.);bulk0=(c=D(1.,0.),))
    negative=copy(pb.u0);negative[end]=D(-1e-100,1.)
    @test_throws ArgumentError remake(pb;u0=negative)
end

@testset "Tabulated distribution distance integrates affine quantiles exactly" begin
    for scale in (1.,1e-6)
        grid=Grid1D(scale.*[0.,2.,3.,5.,6.])
        x=scale.*[0.,3.,5.,6.];F=[0.,0.,1.,1.]
        @test IsothermalProteinCrystallization.tabulated_wasserstein(grid,[1.,0.,0.,0.],x,F)≈3scale rtol=1e-12
        @test IsothermalProteinCrystallization.tabulated_wasserstein(grid,[1.,0.,0.,0.],grid.edges,[0.,1.,1.,1.,1.])≈0 atol=1e-14
    end
end
