using Test, PopulationBalanceModels, ForwardDiff
include(joinpath(@__DIR__,"..","examples","isothermal_protein_crystallization.jl"))
using .IsothermalProteinCrystallization

@testset "Conditional number measurements and deterministic calibration" begin
    grid=Grid1D([0.,1.,3.,6.])
    N=[1.,4.,0.]
    # Window [0.5,2]: uniform densities 1 and 2; total 2.5.
    @test conditional_cdf(grid,N,[.5,1.,1.5,2.];window=(.5,2.)) ≈ [0.,.2,.6,1.]
    @test conditional_quantiles(grid,N,[.1,.2,.6,.9];window=(.5,2.)) ≈ [.75,1.,1.5,1.875]
    @test_throws NumericalFailure conditional_cdf(grid,zeros(3),[1.];window=(.5,2.))
    @test_throws NumericalFailure conditional_cdf(grid,[1.,-1.,2.],[1.];window=(.5,2.))
    @test cdf_quantiles([0.,1.,2.,3.],[0.,.5,.5,1.],[.25,.5,.75]) ≈ [.5,1.,2.5]
    # Two translated uniform laws: W₂² equals shift², in length² units.
    for scale in (1.,1e-6), q in (256,512,1024)
        u=((1:q).-0.5)./q
        a=cdf_quantiles(scale.*[0.,2.],[0.,1.],u)
        b=cdf_quantiles(scale.*[3.,5.],[0.,1.],u)
        @test sum(abs2,a-b)/q ≈ 9scale^2
    end
    f(v)=conditional_quantiles(grid,v,[.1,.6];window=(.5,2.))
    jac=ForwardDiff.jacobian(f,N)
    h=1e-5
    @test jac[:,1] ≈ (f(N+[h,0,0])-f(N-[h,0,0]))/(2h) rtol=1e-7

    data=load_data()
    @test length(data.concentration)==16
    @test length(select_concentration(data))==11
    @test [first(r).time_min for r in select_psds(data)]==[74.,180.,420.]
    @test first(data.concentration).model_time_min==0
    @test first(data.concentration).concentration_g_L==IsothermalProteinCrystallization.C0
    p=parameters(); pb=build(bins=30,tf=74.)
    @test all(iszero,population(pb,pb.u0,:crystals))
    du=similar(pb.u0);rhs!(du,pb.u0,pb.operator,p,0.)
    @test sum(population(pb,du,:crystals))≈p.J_ref/1e12
    g=pb.operator.system.populations.crystals.grid
    @test bulk(pb,du,:c)+1e-6*p.rho*p.kv*moment(g,population(pb,du,:crystals),3)≈0 atol=1e-14
    @test rates(merge(p,(j=0.,g=0.)),0.,p.cstar)==(0.,0.)
    sol=simulate(pb;dt=.05,saveat=[0.,74.])
    @test minimum(last(sol.u))>=0
    @test bulk(pb,last(sol.u),:c)+1e-6*p.rho*p.kv*moment(g,population(pb,last(sol.u),:crystals),3)≈p.c0 atol=1e-10

    cal=calibration(bins=30)
    u=coordinates([log(p.J_ref),log(p.G_ref),p.j,p.g])
    @test length(first(cal.experiments).observations)==4
    @test size(first(first(cal.experiments).observations).data)==(1,11)
    changed=merge(data,(concentration=[r.split=="validation" ? merge(r,(concentration_g_L=999.,)) : r for r in data.concentration],
        psd=[r.split=="validation" ? merge(r,(conditional_cdf=-999.,)) : r for r in data.psd]))
    @test objective(cal,u)==objective(calibration(data=changed,bins=30),u)
    @test objective(cal,u)≈sum(r.value for r in loss_contributions(cal,u))
    automatic=ForwardDiff.gradient(v->objective(cal,v),u)
    independent=[(objective(cal,u+h.*(1:4 .== k))-objective(cal,u-h.*(1:4 .== k)))/(2h) for k in 1:4]
    @test automatic≈independent rtol=2e-5 atol=1e-5
    @test_throws ArgumentError fit_diagnostics(cal,u)
    # Repeated integration nodes must not acquire additional statistical weight.
    fixed=build(bins=10)
    function discrepancy(n)
        ob=ResidualObservation([0.],reshape(repeat([0.,1.],n),:,1),
            FunctionObservable((N,op,p,t)->repeat([1.,3.],n));scale=2.,weight=2.)
        CalibrationProblem((Experiment(fixed,ob),);dt=.01)
    end
    @test objective(discrepancy(1),[1.])≈1.25
    @test objective(discrepancy(17),[1.])≈1.25
    @test_throws ArgumentError confidence_ellipsoid(cal,u)
    # The reference-rate convention is exactly reversible in the legacy units.
    sref=(p.c0-p.cstar)/p.cstar
    AJ=p.J_ref/(60*sref^p.j);AG=1000*p.G_ref/sref^p.g
    @test 60AJ*sref^p.j≈p.J_ref
    @test 1e-3AG*sref^p.g≈p.G_ref
    ex=first(cal.experiments)
    @test_throws ArgumentError CalibrationProblem((Experiment(ex.problem,first(ex.observations),
        Observation([74.],[10.],BulkObservable(:c))),);dt=.1)
end

@testset "Reference-rate empty startup and exactly closed moment check" begin
    p=parameters(J_ref=1e8,G_ref=.1,j=0.,g=0.)
    J=p.J_ref/1e12; T=10.
    exact=[J*p.G_ref^k*T^(k+1)/(k+1) for k in 0:3]
    model=PBMModel(growth=Growth(PolynomialRate(((p,t,ctx)->p.G_ref,))),
        primary=Nucleation(rate=(p,t,ctx)->p.J_ref/1e12,size=0.))
    op=compile(model,MomentSet(0:3);method=ExactClosure())
    system=PBMSystem((crystals=op,);nonnegative_bulk=(:c,),exchanges=(
        MomentExchange(:crystals,:c,3;factor=1e-6*p.rho*p.kv),))
    pb=PBMProblem(system,(crystals=zeros(4),),(0.,T);bulk0=(c=p.c0,),p)
    sol=simulate(pb;dt=.1,method=MeasureEuler(reconstruction=RankAdaptiveQMOM(2)))
    @test population(pb,last(sol.u),:crystals)≈exact rtol=1e-10
    @test bulk(pb,last(sol.u),:c)≈p.c0-1e-6*p.rho*p.kv*exact[4] atol=1e-11
    errors=[]
    for bins in (50,100)
        sec=forward(p;bins,Lmax=2.,dt=.01,times=[0.,T])
        grid=sec.problem.operator.system.populations.crystals.grid
        n=population(sec.problem,last(sec.u),:crystals)
        @test sum(n)≈exact[1] rtol=1e-9
        push!(errors,abs(moment(grid,n,3)-exact[4]))
    end
    @test errors[2]<errors[1]
end
