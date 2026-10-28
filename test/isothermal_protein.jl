using Test, PopulationBalanceModels, ForwardDiff, LinearAlgebra
import Distributions
using Distributions: Normal
include(joinpath(@__DIR__,"..","examples","isothermal_protein_crystallization.jl"))
using .IsothermalProteinCrystallization
include(joinpath(@__DIR__,"..","examples","support","isothermal_protein_sampling.jl"))

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

@testset "CNT/GRD units, boundary flux and same-stage material" begin
    p=parameters();S=p.c0/p.cstar
    B=16pi*(p.gamma/1000)^3*(2.97e-26)^2/(3*(1.380649e-23*293.15)^3)
    J=exp(p.logAJ)*exp(-B/log(S)^2);G=p.AG*(S-1)^p.g
    @test collect(rates(p,0.,p.c0)) ≈ [J,G]
    # Independently convert the former units, including the population-state
    # scale. Only units change here, not the fixed-gamma kinetic equations.
    oldAJ=exp(p.logAJ)*1000/60
    oldAG=1000p.AG
    for supersaturation in (1.2,3.,9.)
        @test IsothermalProteinCrystallization.nucleation_rate(p,p.cstar*supersaturation)≈
            60oldAJ*exp(-B/log(supersaturation)^2)/1000 rtol=2e-13
        @test IsothermalProteinCrystallization.growth_rate(p,p.cstar*supersaturation)≈
            oldAG*(supersaturation-1)^p.g/1000 rtol=2e-13
    end
    @test p.rho≈1240*1000/1e18
    @test p.rho*1e9≈1e-6*1240
    @test rates(p,0.,p.cstar)==(0.,0.)
    @test rates(p,0.,p.cstar/2)==(0.,0.)
    @test IsothermalProteinCrystallization.nucleation_rate(merge(p,(gamma=100.,)),p.c0)==0
    @test ForwardDiff.derivative(c->IsothermalProteinCrystallization.nucleation_rate(merge(p,(gamma=100.,)),c),p.c0)==0
    for model in (:cnt,:cnt_grd)
        p=parameters(grd_strength=model===:cnt ? 0. : .01)
        pb=build(bins=30,tf=74.,p=p,model=model);op=pb.operator.system.populations.crystals
        @test all(iszero,population(pb,pb.u0,:crystals))
        for N in (zeros(30),exp.(-op.grid.centers)),c in (p.c0,.8p.c0)
            u=copy(pb.u0);population(pb,u,:crystals).=N;u[end]=c
            du=similar(u);rhs!(du,u,pb.operator,p,0.)
            @test sum(population(pb,du,:crystals))≈rates(p,0.,c)[1] rtol=1e-12
            @test bulk(pb,du,:c)+p.rho*p.kv*moment(op.grid,population(pb,du,:crystals),3)≈0 atol=1e-12
            system=PBMSystem((crystals=op,);exchanges=(ConservativeExchange(:crystals,:c,
                (L,p,t,ctx)->p.rho*p.kv*L^3),))
            reference=PBMProblem(system,(crystals=N,),(0.,74.);bulk0=(c=c,),p=p)
            dr=similar(reference.u0);rhs!(dr,reference.u0,reference.operator,p,0.)
            @test dr≈du rtol=1e-12 atol=1e-12
        end
        sol=forward(p;model=model,bins=30,times=[0.,74.])
        @test minimum(last(sol.u))>=0
        @test bulk(pb,last(sol.u),:c)+p.rho*p.kv*moment(op.grid,population(pb,last(sol.u),:crystals),3)≈p.c0 atol=1e-10
    end
    # The nested model at theta=0 has exactly the same native operator/update.
    p=parameters();a=forward(p;model=:cnt,bins=40,times=[0.,74.,180.])
    b=forward(p;model=:cnt_grd,bins=40,times=[0.,74.,180.])
    @test a.u==b.u
    @test a.steps==b.steps
    # Constant D0 is a different closure from theta*G^2: it still redistributes
    # an existing population at and below saturation, when J=G=0. Its change in
    # the represented material remains included in the same bulk derivative.
    p=parameters(grd_strength=.01)
    pb=build(;bins=30,tf=1.,p,model=:cnt_grd)
    grid=pb.operator.system.populations.crystals.grid
    N=exp.(-((grid.centers.-12.)./3).^2).*grid.widths
    residuals=map((p.cstar,.8p.cstar)) do c
        u=copy(pb.u0);population(pb,u,:crystals).=N;u[end]=c
        du=similar(u);rhs!(du,u,pb.operator,p,0.)
        @test abs(sum(population(pb,du,:crystals)))<1e-14
        @test bulk(pb,du,:c)+p.rho*p.kv*moment(grid,population(pb,du,:crystals),3)≈0 atol=1e-20
        du
    end
    @test residuals[1]==residuals[2]
    @test maximum(abs,residuals[1])>1e-4
    # Explicit interval caps reject violations instead of clipping the rates.
    pb=build(bins=30,tf=1.);u=copy(pb.u0);du=zero(u)
    @test_throws NumericalFailure rhs!(du,u,pb.operator,merge(p,(interval_cmax=1.,)),0.)
end

@testset "Fixed-error three/four-dimensional posterior and held-out exclusion" begin
    C=IsothermalProteinCrystallization;data=load_data()
    @test length(data.concentration)==16
    @test length(select_concentration(data))==11
    @test [first(r).time_min for r in select_psds(data)]==[74.,180.,420.]
    @test first(data.concentration).model_time_min==0
    @test first(data.concentration).concentration_g_L==C.C0
    for model in (:cnt,:cnt_grd)
        b=estimation_problem(bins=30,model=model);z=initial_coordinates(b);n=length(z)
        @test n==(model===:cnt ? 3 : 4)
        @test all(s->s.name ∉ (:sigma_c,:sigma_F),b.parameters)
        ex=only(b.experiments);theta=physical_parameters(b,z,:batch)
        @test length(ex.measurements)==4
        @test size(first(ex.measurements).data)==(1,11)
        @test first(ex.measurements).likelihood.sigma==.05 .* first(ex.measurements).data
        @test size(ex.measurements[2].data)==(22,1)
        @test [only(m.times) for m in ex.measurements[2:end]]==[74.,180.,420.]
        @test all(m.likelihood.sigma==.03 for m in ex.measurements[2:end])
        changed=merge(data,(concentration=[r.split=="validation" ? merge(r,(concentration_g_L=999.,)) : r for r in data.concentration],
            psd=[r.split=="validation" ? merge(r,(conditional_cdf=-999.,)) : r for r in data.psd]))
        @test logposterior(b,z)==logposterior(estimation_problem(data=changed,bins=30,model=model),z)
        @test check_gradient(b,z).valid
        predicted=predictions(b,z)[1]
        whitened=vec(predicted[2]-ex.measurements[2].data)
        R=[exp(-abs(i-j)/2) for i in 1:22,j in 1:22];RF=cholesky(Symmetric(R)).L
        original=RF*whitened;L=cholesky(Symmetric(.03^2*R)).L
        expected=-.5*(sum(abs2,L\original)+2sum(log,diag(L))+22log(2pi))
        actual=PopulationBalanceModels._measurement_loglikelihood(ex.measurements[2].likelihood,
            ex.measurements[2].data,predicted[2],theta,ex)
        @test actual≈expected+sum(log,diag(RF)) rtol=1e-12
        # Independent scalar Gaussian sum verifies every concentration SD and
        # its normalizer, not an averaged residual or an inferred global scale.
        yc=vec(first(ex.measurements).data);pc=vec(predicted[1]);sc=.05yc
        expected=sum(-.5*((a-b)/s)^2-log(s)-.5log(2pi) for (a,b,s) in zip(yc,pc,sc))
        actual=PopulationBalanceModels._measurement_loglikelihood(first(ex.measurements).likelihood,
            first(ex.measurements).data,predicted[1],theta,ex)
        @test actual≈expected rtol=1e-13
        map=C.sampling_coordinates(b)
        for delta in (zeros(n),collect(range(.01,.04;length=n)))
            at=z+delta;y=map.to_sampler(at)
            @test map.from_sampler(y)≈at rtol=1e-11
            @test map.logabsdetjac(y)≈logabsdet(ForwardDiff.jacobian(map.from_sampler,y))[1] rtol=1e-11
            @test check_sampling_map(b,map,at).valid
        end
        y=map.to_sampler(z)
        H=-ForwardDiff.hessian(v->logprior(b,map.from_sampler(v))+map.logabsdetjac(v),y)
        sg=(theta.g-1)/4;sa=(theta.logAJ-C.AJ_BOUNDS[1])/100
        expected=[2sa*(1-sa),0.,2sg*(1-sg)]
        n==4 && push!(expected,theta.grd_strength/C.D0_PRIOR_MEAN)
        @test H≈Diagonal(expected) atol=1e-10
        tuning=C.sampler_tuning(b,z,map)
        @test isposdef(Symmetric(tuning.covariance))
        @test tuning.information*tuning.covariance≈Matrix{Float64}(I,n,n) atol=1e-7
        @test length(C.prior_metadata(b))==n
        if n==4
            @test C.prior_metadata(b)[end].arguments==[C.D0_PRIOR_MEAN]
        else
            @test theta.grd_strength==0
        end
        @test theta.gamma==.64
        @test all(s->s.name!==:gamma,b.parameters)
    end
end

include("fixtures/protein_transport.jl")
@testset "Historical positive-barrier coordinates on an analytical target" begin
    C=LegacyProteinCoordinates;S=IsothermalProteinSampling
    for model in (:cnt,:cnt_grd), Sref in (nothing,6.)
        problem=C.estimation_problem(;bins=30,model)
        z=initial_coordinates(problem);map=S.bounded_rate_coordinates(C,problem,z;
            growth_reference_supersaturation=Sref)
        physical=physical_parameters(problem,z,:batch)
        s=isnothing(Sref) ? C.C0/physical.cstar : Sref
        @test map.to_sampler(z)[3]≈log(.001physical.AG)+physical.g*log(s-1) atol=1e-12
        for A in (.001,30.,95.,99.999),gamma in (1e-6,.64,1.3)
            q=copy(z);q[1]=log(A/(100-A));q[2]=log(gamma)
            y=map.to_sampler(q);J=ForwardDiff.jacobian(map.from_sampler,y)
            @test map.from_sampler(y)≈q atol=2e-13
            @test map.logabsdetjac(y)≈logabsdet(J)[1] atol=2e-12
            @test physical_parameters(problem,map.from_sampler(y),:batch).logAJ≈A rtol=1e-11
            # Independent physical-density change of variables, including the
            # log-uniform AG prior, checks every transform factor exactly once.
            theta=physical_parameters(problem,q,:batch)
            physical_prior=Distributions.logpdf(Distributions.Uniform(0.,100.),theta.logAJ)+
                Distributions.logpdf(Distributions.LogNormal(log(.64),.7),theta.gamma)+
                Distributions.logpdf(Distributions.Uniform(log(.01),log(1500.)),log(theta.AG))-log(theta.AG)+
                Distributions.logpdf(Distributions.Uniform(1.,3.),theta.g)
            length(z)==5 && (physical_prior+=Distributions.logpdf(Distributions.Exponential(10.),theta.grd_strength))
            physical_vector(v)=let p=physical_parameters(problem,map.from_sampler(v),:batch)
                [getproperty(p,s.name) for s in problem.parameters]
            end
            actual=physical_prior+logabsdet(ForwardDiff.jacobian(physical_vector,y))[1]
            @test logprior(problem,q)+map.logabsdetjac(y)≈actual atol=2e-11
        end
        @test check_sampling_map(problem,map,z).valid
        @test_throws ArgumentError S.bounded_rate_coordinates(C,problem,z;growth_reference_supersaturation=1.)
        tuning=S.expected_information_tuning(problem,z,map)
        @test isposdef(Symmetric(tuning.covariance))
        @test tuning.information*tuning.covariance≈Matrix{Float64}(I,length(z),length(z)) atol=1e-7
    end
end

@testset "Local curvature is only sampler tuning" begin
    isdefined(@__MODULE__,:bayesian_linear_fixture) || include("fixtures/bayesian_linear.jl")
    f=bayesian_linear_fixture();p=f.problem
    A=[2. .7;-.3 1.5]
    map=(to_sampler=z->A*z,from_sampler=y->A\y,
         logabsdetjac=y->-logabsdet(A)[1],label=:linear_tuning_test)
    tuning=IsothermalProteinSampling.curvature_tuning(p,f.mode,map)
    @test tuning.covariance≈A*f.covariance*A' rtol=1e-12
    @test tuning.information≈inv(f.covariance) rtol=1e-12
end

@testset "Sampler transport preserves physical priors and inverse derivatives" begin
    C=LegacyProteinCoordinates;S=IsothermalProteinSampling
    problem=C.estimation_problem(model=:cnt_grd,bins=30)
    reference=initial_coordinates(problem);configuration=protein_transport_fixture()
    mapping=S.transported_rate_coordinates(C,problem,reference,configuration;growth_reference_supersaturation=6.)
    for A in (.001,30.,83.4,99.999),gamma in (.1,.6,1.1),g in (1.001,2.7,2.999)
        z=copy(reference);z[1]=log(A/(100-A));z[2]=log(gamma);z[4]=log((g-1)/(3-g))
        u=mapping.to_sampler(z)
        @test mapping.from_sampler(u)≈z atol=3e-10
        @test mapping.logabsdetjac(u)≈logabsdet(ForwardDiff.jacobian(mapping.from_sampler,u))[1] atol=3e-10
        theta=physical_parameters(problem,z,:batch)
        density=Distributions.logpdf(Distributions.Uniform(0.,100.),theta.logAJ)+
            Distributions.logpdf(Distributions.LogNormal(log(.64),.7),theta.gamma)+
            Distributions.logpdf(Distributions.Uniform(log(.01),log(1500.)),log(theta.AG))-log(theta.AG)+
            Distributions.logpdf(Distributions.Uniform(1.,3.),theta.g)+
            Distributions.logpdf(Distributions.Exponential(10.),theta.grd_strength)
        physical(v)=let p=physical_parameters(problem,mapping.from_sampler(v),:batch)
            [getproperty(p,s.name) for s in problem.parameters]
        end
        @test logprior(problem,z)+mapping.logabsdetjac(u)≈
            density+logabsdet(ForwardDiff.jacobian(physical,u))[1] atol=3e-10
    end
    @test check_sampling_map(problem,mapping,reference).valid
    @test_throws NumericalFailure mapping.from_sampler([-2.4723409036924306,361.36754204893145,
        51.971302738649065,-110.08883962313041,4.55248276569897])
    # Independent inverse-function derivatives, including nested Duals.
    mixture=Distributions.MixtureModel([Normal(-1.,.2),Normal(1.5,1.)],[.2,.8])
    for p in (.0001,.05,.2,.5,.95,.9999)
        x=Distributions.quantile(mixture,p)
        f=Distributions.pdf(mixture,x)
        fp=sum(w*Distributions.pdf(d,x)*(-(x-Distributions.mean(d))/Distributions.std(d)^2)
            for (w,d) in zip((.2,.8),(Normal(-1.,.2),Normal(1.5,1.))))
        @test ForwardDiff.derivative(q->S._mixture_quantile(mixture,q),p)≈inv(f) rtol=2e-12
        @test ForwardDiff.derivative(q->ForwardDiff.derivative(v->S._mixture_quantile(mixture,v),q),p)≈-fp/f^3 rtol=2e-10
    end
    @test_throws ArgumentError S.transported_rate_coordinates(C,problem,reference,
        merge(configuration,(mixture_weights=[0.,1.],)))
    @test_throws ArgumentError S.transported_rate_coordinates(C,problem,reference,
        merge(configuration,(blend_width=0.,)))
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
