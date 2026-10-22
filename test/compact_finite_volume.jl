using LinearAlgebra, SpecialFunctions
const CompactPBM=PopulationBalanceModels
const compact_periodic=SizeBC(Periodic(),Periodic())
const compact_physical=SizeBC(OpenBoundary(),OpenBoundary())

@testset "Conservative HOC compilation and structural contracts" begin
    g=Grid1D(range(0,1;length=17)); policy=HOCFiniteVolume()
    build(law;bc=compact_physical,p=(;))=compile(PBMModel(growth=Growth(law)),g;evolution=policy,bc,p)
    for law in (AffineGrowth(.4,.1),ExponentialGrowth(.4,.1),
        TimeScaledGrowth(AffineGrowth(.4,.1),PrescribedRate((p,t,c)->1+t)))
        op=build(law)
        @test term_methods(op).growth===policy
        @test diagnostics(simulate(PBMProblem(op,g.widths,(0.,.1));dt=.01)).policy===:HOCFiniteVolume
    end
    # This callback agrees with a constant at any finite diagnostic set chosen
    # here, but varies elsewhere. Classification never relies on such samples.
    fake(x,p,t,c)=.4+prod((x-q)^2 for q in (0.,.25,.5,.75,1.))
    @test_throws ArgumentError build(fake)
    declared=HOCGrowthLaw(fake;lower_bound=.4)
    @test build(declared) isa CompactPBM.CompiledHOC
    @test_throws ArgumentError build(HOCGrowthLaw(fake;lower_bound=.4,dependencies=(:coordinate,:bulk)))
    @test_throws ArgumentError build(HOCGrowthLaw(fake;lower_bound=.4,dependencies=(:coordinate,:y)))
    @test_throws ArgumentError build(declared;bc=compact_periodic)
    @test_throws ArgumentError build(AffineGrowth(.4,.1);bc=compact_periodic)
    @test_throws ArgumentError build(AffineGrowth(-.4,0))
    @test_throws ArgumentError build(AffineGrowth(.4,0);bc=SizeBC())
    @test_throws ArgumentError build(AffineGrowth(.4,0);bc=SizeBC(AbsorbingZero(),OpenBoundary()))
    @test_throws ArgumentError HOCFiniteVolume(time_order=4)
    @test_throws ArgumentError compile(PBMModel(Growth(AffineGrowth(1,0))),Grid1D([0.,.1,.3,.5,.7,.9,1.]);evolution=policy,bc=compact_physical)
    @test_throws ArgumentError compile(PBMModel(Growth(AffineGrowth(1,0))),MomentSet(0:3);evolution=policy)
    @test_throws ArgumentError compile(PBMModel(Growth(AffineGrowth(1,0))+Death((x,p,t,c)->1)),g;evolution=policy,bc=compact_physical)
    @test_throws ArgumentError compile(PBMModel(Growth(AffineGrowth(1,0))+Nucleation((p,t,c)->1,.2)),g;evolution=policy,bc=compact_physical)
    @test_throws ArgumentError compile(PBMModel(Growth(AffineGrowth(1,0))+Nucleation(1.,0.)),g;
        evolution=policy,bc=SizeBC(InflowFlux(1.),OpenBoundary()))
    @test_throws ArgumentError compile(PBMModel(Growth(AffineGrowth(1,0))),g;
        evolution=policy,bc=SizeBC(InflowFlux(PrescribedRate((p,t,c)->1;dependencies=(:bulk,))),OpenBoundary()))
    op=build(AffineGrowth(:v,0);p=(v=.4,))
    prob=PBMProblem(op,g.widths,(0.,.2);p=(v=.4,))
    @test suggest_dt(op,prob.u0,prob.p,0.)≈.5/(16*.4)
    @test_throws ArgumentError remake(prob;p=(v=-.2,))
    periodic_op=compile(PBMModel(Growth(AffineGrowth(.4,:s))),g;evolution=policy,bc=compact_periodic,p=(s=0.,))
    periodic_prob=PBMProblem(periodic_op,g.widths,(0.,.2);p=(s=0.,))
    @test_throws ArgumentError remake(periodic_prob;p=(s=.1,))
    bad_source=compile(PBMModel(Growth(AffineGrowth(.4,0))+Nucleation(-1.,0.)),g;evolution=policy,bc=compact_physical)
    @test_throws NumericalFailure PBMProblem(bad_source,zeros(16),(0.,.1))
    law=HOCGrowthLaw((x,p,t,c)->p.rate[1];lower_bound=(p,t,c)->p.rate[1],
        dependencies=(:parameters,),periodic=true)
    mutable_parameters=(rate=[.4],)
    changed=PBMProblem(build(law;bc=compact_periodic),g.widths,(0.,.1);p=mutable_parameters)
    mutable_parameters.rate[1]=-.4
    @test_throws NumericalFailure simulate(changed;dt=.01,saveat=[0.])
    mutable_parameters.rate[1]=.4
    changed.u0[1]=-1.
    @test_throws NumericalFailure simulate(changed;dt=.01,saveat=[0.])
    s1=simulate(prob;dt=.02,check_nonnegative=false);s2=simulate(remake(prob;p=(v=.8,));dt=.02,check_nonnegative=false)
    @test s1.u[end]!=s2.u[end]
    law=TimeScaledGrowth(AffineGrowth(1,0),PrescribedRate((p,t,c)->1-10t))
    prob=PBMProblem(build(law),zeros(length(g)),(0.,.2)); original=copy(prob.u0)
    @test_throws NumericalFailure simulate(prob;dt=.05)
    @test prob.u0==original
    @test_throws NumericalFailure PBMProblem(build(HOCGrowthLaw((x,p,t,c)->.1;lower_bound=.2)),g.widths,(0.,1.))
    op=build(AffineGrowth(.4,0))
    @test_throws NumericalFailure simulate(PBMProblem(op,zeros(16),(0.,1.));dt=1.)
    @test_throws ArgumentError PBMSystem((particles=op,))
    @test_throws ArgumentError rhs!(zeros(16,1),ones(16,1),op,(;),0.)
    # Choosing the method at simulate uses the same strict validator.
    raw=compile(PBMModel(Growth(fake)),g;bc=compact_physical)
    @test_throws ArgumentError simulate(PBMProblem(raw,g.widths,(0.,.1));method=policy,dt=.01)
    raw=compile(PBMModel(Growth(AffineGrowth(.4,0))),g;bc=compact_physical)
    @test simulate(PBMProblem(raw,g.widths,(0.,.1));method=policy,dt=.01,check_nonnegative=false).u[end]≈
          simulate(PBMProblem(op,g.widths,(0.,.1));dt=.01,check_nonnegative=false).u[end]
    g2=Grid2D(g,g)
    @test_throws ArgumentError compile(PBMModel(Growth2D((x,y,p,t,c)->(.4+y,.2))),g2;evolution=policy)
    @test_throws ArgumentError compile(PBMModel(Growth2D(SeparableGrowth2D(AffineGrowth(.4,0),fake))),g2;
        evolution=policy,bc=SizeBC2D(x=compact_physical,y=compact_physical))
    @test_throws ArgumentError compile(PBMModel(Growth2D((1.,1.))+Nucleation2D(1.,(0.,0.))),g2;
        evolution=HOCADI(),bc=SizeBC2D(x=compact_periodic,y=compact_periodic))
    old=compile(PBMModel(Growth2D((.4,-.2))),g2;evolution=HOCADI(),bc=SizeBC2D(x=compact_periodic,y=compact_periodic))
    @test diagnostics(simulate(PBMProblem(old,g.widths*g.widths',(0.,.1));dt=.01)).policy===:HOCADI
    # Constant tuples declare no field dependency. Unused prescribed context
    # must not make compilation stricter than selecting the policy at simulate.
    unused=compile(old.model,g2;evolution=HOCADI(),bc=old.bc,fields=(c,t)->error("unused field was evaluated"))
    @test simulate(PBMProblem(unused,g.widths*g.widths',(0.,.1));dt=.01).u[end]≈
          simulate(PBMProblem(old,g.widths*g.widths',(0.,.1));dt=.01).u[end]
end

@testset "HOC face systems versus independently assembled dense equations" begin
    for periodic in (false,true), n in (6,13), δ in (.001,.4,4.)
        h=1/n;v=periodic ? fill(.7,n+1) : [.4+.2*j/n for j in 0:n]
        P=Matrix{Float64}(I,n,n);Q=zeros(n,n);D=Matrix{Float64}(I,n,n)
        for j in 1:n-1
            j>1 && (P[j,j-1]=1/4)
            P[j,j+1]=1/4; Q[j,j]=Q[j,j+1]=3/4
            D[j+1,j]=-1
        end
        if periodic
            P[1,n]=P[n,n-1]=P[n,1]=1/4
            Q[n,n]=Q[n,1]=3/4;D[1,n]=-1
        else
            # Derive outflow weights from cell-average monomial integrals,
            # independently of _HOC_OUTFLOW and its directional assembly.
            V=[((-l)^(k+1)-(-l-1)^(k+1))/(k+1) for k in 0:4,l in 0:4]
            Q[n,n:-1:n-4]=V\[1.,0,0,0,0]
        end
        A=P*Diagonal(1 ./v[2:end])+δ/(2h)*Q*D
        b=[sin(j)+.4cos(2j) for j in 1:n];x=similar(b)
        factor=CompactPBM._hoc_directional_factor(v,h,δ,periodic,Float64)
        CompactPBM._hoc_solve!(x,factor,b)
        @test x≈A\b rtol=2e-12 atol=2e-12
        @test norm(A*x-b)<2e-12*norm(b)
    end
    @test_throws NumericalFailure CompactPBM._hoc_factor(zeros(2),zeros(3),zeros(2),(1,),zeros(1,3))
end

@testset "HOC independently integrated characteristic references" begin
    # Primitive of a Gaussian, not sampling a density at bin centers.
    F(x)=.12sqrt(pi)/2*erf((x-.65)/.12)
    for family in (:constant,:affine,:exponential)
        errors=Float64[]
        for n in (40,80,160)
            g=Grid1D(range(0,2;length=n+1));tf=.4
            law=family===:constant ? AffineGrowth(.4,0) : family===:affine ? AffineGrowth(.4,.15) : ExponentialGrowth(.4,.15)
            foot(x)=family===:constant ? x-.4tf : family===:affine ? (x+.4/.15)*exp(-.15tf)-.4/.15 : -log(exp(-.15x)+.4*.15tf)/.15
            ref=diff(F.(max.(0,foot.(g.edges))))
            op=compile(PBMModel(Growth(law)),g;bc=compact_physical,evolution=HOCFiniteVolume())
            sol=simulate(PBMProblem(op,diff(F.(g.edges)),(0.,tf));dt=.01/n,saveat=[tf],check_nonnegative=false)
            push!(errors,sum(abs,vec(sol.u[end])-ref))
            @test abs(diagnostics(sol).number_accounting[end].defect)<2e-13*sum(sol.problem.u0)
        end
        @test 3.8<log2(errors[2]/errors[3])<4.3
    end
    # Smooth incoming density crosses both physical boundaries. Time error is
    # held small while spatial resolution changes.
    errors=Float64[]
    for n in (20,40,80)
        g=Grid1D(range(0,1;length=n+1));v=.4;tf=.4
        integral(x,t)=2x-cos(2pi*(x-v*t))/(2pi)
        influx=PrescribedRate((p,t,c)->v*(2+sin(-2pi*v*t)))
        op=compile(PBMModel(Growth(AffineGrowth(v,0))),g;
            bc=SizeBC(InflowFlux(influx),OpenBoundary()),evolution=HOCFiniteVolume())
        sol=simulate(PBMProblem(op,diff(integral.(g.edges,0.)),(0.,tf));dt=.01/n,saveat=[tf])
        ref=diff(integral.(g.edges,tf));push!(errors,sum(abs,vec(sol.u[end])-ref))
        ledger=diagnostics(sol).number_accounting[end]
        @test abs(ledger.defect)<5e-14
        # Analytic time integral of incoming number flux.
        exact_in=2v*tf+(cos(2pi*v*tf)-1)/(2pi)
        @test only(ledger.inflow)≈exact_in atol=1e-8
    end
    @test 3.8<log2(errors[2]/errors[3])<4.2
end

@testset "HOC temporal order and prescribed time-dependent growth" begin
    # A fixed-grid matrix exponential isolates time error without using the
    # implementation's Cayley factor or a grid-refinement surrogate.
    n=24;g=Grid1D(range(0,1;length=n+1));h=1/n;v=.4
    M=Matrix{Float64}(I,n,n);D=zeros(n,n)
    for i in 1:n
        M[i,mod1(i-1,n)]=M[i,mod1(i+1,n)]=.25
        D[i,mod1(i+1,n)]=3/(4h);D[i,mod1(i-1,n)]=-3/(4h)
    end
    N=2g.widths+diff(sin.(2pi*g.edges))/(2pi);tf=.5
    for modulation in (false,true)
        law=modulation ? TimeScaledGrowth(AffineGrowth(v,0),PrescribedRate((p,t,c)->1+t^2)) : AffineGrowth(v,0)
        op=compile(PBMModel(Growth(law)),g;bc=compact_periodic,evolution=HOCFiniteVolume())
        travel=v*(modulation ? tf+tf^3/3 : tf)
        reference=exp(-travel*(M\D))*N;errors=Float64[]
        for dt in (tf/8,tf/16,tf/32)
            sol=simulate(PBMProblem(op,N,(0.,tf));dt,saveat=[tf])
            push!(errors,norm(vec(sol.u[end])-reference))
        end
        @test 1.95<log2(errors[2]/errors[3])<2.05
    end
end

@testset "HOC rectangular temporal order" begin
    gx=Grid1D(range(0,1;length=25));gy=Grid1D(range(0,2;length=19));grid=Grid2D(gx,gy)
    function derivative(axis)
        n=length(axis);h=first(axis.widths);M=Matrix{Float64}(I,n,n);D=zeros(n,n)
        for i in 1:n
            M[i,mod1(i-1,n)]=M[i,mod1(i+1,n)]=.25
            D[i,mod1(i-1,n)]=-3/(4h);D[i,mod1(i+1,n)]=3/(4h)
        end
        M\D
    end
    initial=(gx.widths+diff(sin.(2pi*gx.edges))/(10pi))*(gy.widths+diff(sin.(pi*gy.edges))/(5pi))'
    op=compile(PBMModel(Growth2D((.4,.7))),grid;bc=SizeBC2D(x=compact_periodic,y=compact_periodic),evolution=HOCFiniteVolume())
    reference=exp(-.4*.25*derivative(gx))*initial*exp(-.7*.25*derivative(gy))'
    errors=Float64[]
    for dt in (.25/4,.25/8,.25/16)
        sol=simulate(PBMProblem(op,initial,(0.,.25));dt,saveat=[.25])
        push!(errors,norm(sol.u[end]-reference))
    end
    @test 1.95<log2(errors[2]/errors[3])<2.05
end

@testset "HOC physical and variable-coefficient time refinement" begin
    n=16;g=Grid1D(range(0,2;length=n+1));h=first(g.widths)
    P=Matrix{Float64}(I,n,n);Q=zeros(n,n);D=Matrix{Float64}(I,n,n)
    for j in 1:n-1
        j>1 && (P[j,j-1]=.25)
        P[j,j+1]=.25;Q[j,j]=Q[j,j+1]=.75;D[j+1,j]=-1
    end
    V=[((-l)^(k+1)-(-l-1)^(k+1))/(k+1) for k in 0:4,l in 0:4]
    Q[n,n:-1:n-4]=V\[1.,0,0,0,0]
    for variable in (false,true)
        law=variable ? ExponentialGrowth(.4,.15) : AffineGrowth(.4,0)
        v=[law(x,(;),0.,nothing) for x in g.edges]
        # Independent method-of-lines matrix, using density (not midpoint
        # flux) as the reconstruction unknown, then an exact exponential.
        A=-D*Diagonal(v[2:end])*(P\Q)/h
        initial=2g.widths+diff(sin.(pi*g.edges))/pi;tf=.4
        if variable
            bc=compact_physical;reference=exp(tf*A)*initial
        else
            # Exact augmented linear system for the sinusoidal incoming flux.
            unit=zeros(n);unit[1]=1
            b=unit+D*Diagonal(v[2:end])*(P\(.25/v[1]*unit))
            augmented=zeros(n+3,n+3);augmented[1:n,1:n]=A
            augmented[1:n,n+1]=.8b;augmented[1:n,n+2]=-.4b
            ω=.8pi;augmented[n+2,n+3]=ω;augmented[n+3,n+2]=-ω
            reference=(exp(tf*augmented)*vcat(initial,[1.,0.,1.]))[1:n]
            bc=SizeBC(InflowFlux(PrescribedRate((p,t,c)->.4*(2-sin(ω*t)))),OpenBoundary())
        end
        op=compile(PBMModel(Growth(law)),g;bc,evolution=HOCFiniteVolume())
        errors=Float64[]
        for dt in (.4/8,.4/16,.4/32)
            sol=simulate(PBMProblem(op,initial,(0.,tf));dt,saveat=[tf],check_nonnegative=false)
            push!(errors,norm(vec(sol.u[end])-reference))
        end
        @test 1.9<log2(errors[2]/errors[3])<2.1
    end
end

@testset "HOC explicit general-law contracts, prescribed fields and scalar types" begin
    errors=Float64[]
    F(x)=.12sqrt(pi)/2*erf((x-.65)/.12)
    for n in (40,80,160)
        g=Grid1D(range(0,2;length=n+1));tf=.4
        # A(t,x)=.4+.1x+.2t is not an autonomous G*f transformation.
        law=HOCGrowthLaw((x,p,t,c)->.4+.1x+c.fields.offset;
            lower_bound=(p,t,c)->.4+c.fields.offset,dependencies=(:coordinate,:fields))
        op=compile(PBMModel(Growth(law)),g;bc=compact_physical,evolution=HOCFiniteVolume(),fields=(cell,t)->(;offset=.2t))
        drift=4expm1(.1tf)+20(expm1(.1tf)-.1tf)
        feet=(g.edges.-drift).*exp(-.1tf)
        ref=diff(F.(max.(0,feet)))
        sol=simulate(PBMProblem(op,diff(F.(g.edges)),(0.,tf));dt=.01/n,saveat=[tf],check_nonnegative=false)
        push!(errors,sum(abs,vec(sol.u[end])-ref))
    end
    @test log2(errors[2]/errors[3])>3.8
    # Smooth, non-affine periodic law with a known stationary density 1/G.
    stationary_errors=Float64[]
    for n in (20,40,80)
        g=Grid1D(range(0,1;length=n+1));a=.4;b=.1;d=sqrt(a^2-b^2)
        primitive(x)=(atan((a*tan(pi*x)+b)/d)+(x>.5 ? pi : 0))/(pi*d)
        initial=diff(primitive.(g.edges))
        law=HOCGrowthLaw((x,p,t,c)->a+b*sin(2pi*x);lower_bound=a-b,dependencies=(:coordinate,),periodic=true)
        op=compile(PBMModel(Growth(law)),g;bc=compact_periodic,evolution=HOCFiniteVolume())
        sol=simulate(PBMProblem(op,initial,(0.,.1));dt=.001,saveat=[.1])
        push!(stationary_errors,sum(abs,vec(sol.u[end])-initial))
    end
    @test 3.6<log2(stationary_errors[2]/stationary_errors[3])<4.4
    for T in (Float32,BigFloat)
        g=Grid1D(range(T(0),T(1);length=9))
        op=compile(PBMModel(Growth(AffineGrowth(T(.4),T(0)))),g;bc=compact_periodic,evolution=HOCFiniteVolume())
        sol=simulate(PBMProblem(op,g.widths,(T(0),T(.1)));dt=T(.01))
        @test all(isfinite,sol.u[end])
        @test sum(sol.u[end])≈one(T) atol=T(20)*eps(T)
    end
end

@testset "Asymmetric HOC tensor transport, corner births and dispersive lobes" begin
    # Analytic rectangular periodic translation, with distinct velocities,
    # lengths and spacings. Bin integrals are exact trigonometric primitives.
    periodic_errors=Float64[]
    for n in (16,32,64)
        g=Grid2D(range(0,1;length=n+1),range(0,2;length=div(3n,4)+1))
        reference(t)=diff((g.x.edges+.2sin.(2pi*(g.x.edges.-.4t))/(2pi)))*
            diff((g.y.edges+.3sin.(pi*(g.y.edges.-.7t))/pi))'
        op=compile(PBMModel(Growth2D((.4,.7))),g;
            bc=SizeBC2D(x=compact_periodic,y=compact_periodic),evolution=HOCFiniteVolume())
        sol=simulate(PBMProblem(op,reference(0.),(0.,.25));dt=2e-4,saveat=[.25])
        push!(periodic_errors,sum(abs,sol.u[end]-reference(.25)))
        @test sum(sol.u[end])≈sum(reference(0.)) atol=4e-14
    end
    @test 3.8<log2(periodic_errors[2]/periodic_errors[3])<4.2
    errors=Float64[]
    F(x)=.12sqrt(pi)/2*erf((x-.65)/.12)
    H(y)=.18sqrt(pi)/2*erf((y-.8)/.18)
    for n in (24,48,96)
        g=Grid2D(range(0,2;length=n+1),range(0,3;length=div(3n,4)+1));tf=.25
        law=SeparableGrowth2D(AffineGrowth(.4,.1),ExponentialGrowth(.3,.12))
        op=compile(PBMModel(Growth2D(law)),g;bc=SizeBC2D(x=compact_physical,y=compact_physical),evolution=HOCFiniteVolume())
        N=diff(F.(g.x.edges))*diff(H.(g.y.edges))'
        fx(x)=(x+4)*exp(-.1tf)-4
        fy(y)=-log(exp(-.12y)+.3*.12tf)/.12
        ref=diff(F.(max.(0,fx.(g.x.edges))))*diff(H.(max.(0,fy.(g.y.edges))))'
        sol=simulate(PBMProblem(op,N,(0.,tf));dt=.01/n,saveat=[tf],check_nonnegative=false)
        push!(errors,sum(abs,sol.u[end]-ref))
        @test abs(diagnostics(sol).number_accounting[end].defect)<1e-14
        @test CompactPBM._observe(MomentObservable((0,0)),sol.u[end],op,(;),tf)==[sum(sol.u[end])]
    end
    @test log2(errors[2]/errors[3])>3.6
    g=Grid2D(range(0,2;length=25),range(0,3;length=19))
    source=PrescribedRate((p,t,c)->2+t)
    op=compile(PBMModel(Growth2D((.4,.7))+Nucleation2D(source,(0.,0.))),g;
        bc=SizeBC2D(x=compact_physical,y=compact_physical),evolution=HOCFiniteVolume())
    prob=PBMProblem(op,zeros(size(g)),(0.,.1))
    sol=simulate(prob;dt=.01,saveat=[.1],check_nonnegative=false)
    ledger=diagnostics(sol).number_accounting[end]
    @test ledger.births≈.205 atol=5e-16
    @test abs(ledger.defect)<1e-15
    @test ledger.inventory+sum(ledger.outflow)≈.205 atol=1e-15
    @test_throws NumericalFailure simulate(prob;dt=.01) # no hidden clipping
    # Uniform physical inflow in two unequal directions, preserving a constant
    # smooth density exactly; each face flux integrates transverse widths.
    bc=SizeBC2D(x=SizeBC(InflowFlux(.8),OpenBoundary()),y=SizeBC(InflowFlux(1.4),OpenBoundary()))
    op=compile(PBMModel(Growth2D((.4,.7))),g;bc,evolution=HOCFiniteVolume())
    N=2g.x.widths*g.y.widths'
    s=simulate(PBMProblem(op,N,(0.,.1));dt=.01)
    @test maximum(abs,s.u[end]-N)<=32eps(Float64)*maximum(N)
    a=diagnostics(s).number_accounting[end]
    @test collect(a.inflow)≈[.24,.28]
    @test abs(a.defect)<2e-14
    # Sharp periodic translation: retain signed undershoots and exact number.
    g1=Grid1D(range(0,1;length=49));N=[max(0.,min(b,.4)-max(a,.2)) for (a,b) in zip(g1.edges[1:end-1],g1.edges[2:end])]
    op=compile(PBMModel(Growth(AffineGrowth(.4,0))),g1;bc=compact_periodic,evolution=HOCFiniteVolume())
    prob=PBMProblem(op,N,(0.,.1))
    @test_throws NumericalFailure simulate(prob;dt=.01)
    s=simulate(prob;dt=.01,check_nonnegative=false)
    @test minimum(s.u[end])<0
    @test sum(s.u[end])≈sum(N) atol=1e-15
    # Physical zero-size births: the exact low moments are J*G^k*t^(k+1)/(k+1).
    # M1 uses the package's midpoint bin-property convention, so its spatial
    # quadrature error is assessed separately from transport error.
    moment_errors=Float64[]
    for n in (40,80)
        grid=Grid1D(range(0,2;length=n+1))
        op=compile(PBMModel(Growth(AffineGrowth(.4,0))+Nucleation(2.,0.)),grid;
            bc=compact_physical,evolution=HOCFiniteVolume())
        sol=simulate(PBMProblem(op,zeros(n),(0.,.5));dt=.0005,saveat=[.5],check_nonnegative=false)
        ledger=diagnostics(sol).number_accounting[end]
        @test ledger.births≈1. atol=6e-14
        @test sum(sol.u[end])+sum(ledger.outflow)≈1. atol=6e-14
        push!(moment_errors,abs(only(moment(grid,sol.u[end],1))-.1))
    end
    @test moment_errors[2]<moment_errors[1]/2
end
