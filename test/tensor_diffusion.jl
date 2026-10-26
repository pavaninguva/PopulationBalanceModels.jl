using Test, PopulationBalanceModels, LinearAlgebra, ForwardDiff
using SpecialFunctions: erf

_tensor_periodic=SizeBC(Periodic(),Periodic())
function _tensor_diffusion_matrix(axis,D,periodic)
    A=zeros(length(axis),length(axis))
    faces=[(i,i+1,axis.centers[i+1]-axis.centers[i]) for i in 1:length(axis)-1]
    periodic && push!(faces,(length(axis),1,(axis.widths[end]+axis.widths[1])/2))
    for (i,j,h) in faces
        A[i,i]-=D/(h*axis.widths[i]);A[j,i]+=D/(h*axis.widths[i])
        A[j,j]-=D/(h*axis.widths[j]);A[i,j]+=D/(h*axis.widths[j])
    end
    A
end
function _tensor_smooth_bins(g,t;G=(0.,0.),D=(.02,.03))
    kx=2pi/last(g.x.edges);ky=2pi/last(g.y.edges)
    sx=diff(-cos.(kx.*(g.x.edges.-G[1]*t))./kx)
    cy=diff(sin.(ky.*(g.y.edges.-G[2]*t))./ky)
    g.x.widths*g.y.widths' + .2exp(-(D[1]*kx^2+D[2]*ky^2)*t).*sx*cy'
end
function _tensor_gaussian_bins(g,t;G=(.3,.15),D=(.05,.02))
    fractions=map((g.x,g.y),(4.,3.),(.16,.16),G,D) do axis,mu,var,velocity,diffusivity
        diff(erf.((axis.edges.-mu.-velocity*t)./sqrt(2(var+2diffusivity*t))))/2
    end
    fractions[1]*fractions[2]'
end

@testset "Tensor Ito diffusion and coupled bulk balances" begin
    @testset "Independent generator and combined timestep on rectangular meshes" begin
        for edges in (([0.,.1,.4,1.],[0.,.2,.7,1.5,2.]),([0.,.3,1.],[0.,.7,2.])),
            periodic in ((false,false),(true,true),(true,false))
            g=Grid2D(edges...);D=(.03,.07);G=(.2,-.1);loss=.4
            bc=SizeBC2D(x=periodic[1] ? _tensor_periodic : SizeBC(),
                        y=periodic[2] ? _tensor_periodic : SizeBC())
            model=PBMModel(dispersion=SizeDiffusion(D))
            op=compile(model,g;bc)
            N=reshape(collect(1.:length(g)),size(g));d=zero(N);rhs!(d,N,op,(;),0.)
            Ax=_tensor_diffusion_matrix(g.x,D[1],periodic[1])
            Ay=_tensor_diffusion_matrix(g.y,D[2],periodic[2])
            A=kron(Matrix{Float64}(I,length(g.y),length(g.y)),Ax)+
              kron(Ay,Matrix{Float64}(I,length(g.x),length(g.x)))
            @test vec(d)≈A*vec(N) rtol=1e-13 atol=1e-13
            @test abs(sum(d))<1e-12
            @test suggest_dt(op,N,(;),0.;cfl=.7)≈.7/maximum(-diag(A))
            @test term_methods(op).dispersion isa FiniteVolumeDiffusion
            mixed=compile(PBMModel(Growth2D(G)+SizeDiffusion(D)+Death((xy,p,t,c)->loss)),g;bc)
            rates=reshape(-diag(A),size(g)).+abs(G[1])./g.x.widths.+
                  (abs(G[2])./g.y.widths)'.+loss
            @test suggest_dt(mixed,N,(;),0.;cfl=.7)≈.7/maximum(rates)
        end
        for n in (12,24)
            g=Grid2D(range(0,1;length=n+1),range(0,2;length=n+1))
            op=compile(PBMModel(SizeDiffusion((.02,.03))),g;
                bc=SizeBC2D(x=_tensor_periodic,y=_tensor_periodic))
            dt=suggest_dt(op,ones(size(g)),(;),0.;cfl=1)
            @test dt≈1/(2*.02*n^2+2*.03*(n/2)^2)
            initial=zeros(size(g));initial[n÷2,n÷2]=1
            pb=PBMProblem(op,initial,(0.,dt))
            @test minimum(last(simulate(pb;dt=.8dt,method=ForwardEuler()).u))>=0
            @test_throws NumericalFailure simulate(PBMProblem(op,initial,(0.,1.1dt));dt=1.1dt,method=ForwardEuler())
        end
    end

    @testset "Coefficient conventions, support and zero diffusion" begin
        g=Grid2D([0.,.1,.4,1.],[0.,.2,.7,2.]);N=ones(size(g));d=zero(N)
        law=(xy,p,t,ctx)->((1+xy[1]^2)*(1+.1xy[2]),(2+xy[2])*(1+.2xy[1]))
        op=compile(PBMModel(SizeDiffusion(law)),g)
        f=[1/((1+x^2)*(2+y)) for x in g.x.centers,y in g.y.centers]
        stationary=f.*(g.x.widths*g.y.widths')
        rhs!(d,stationary,op,(;),0.)
        @test maximum(abs,d)<1e-13 # Dx*n constant in x; Dy*n constant in y
        rhs!(d,g.x.widths*g.y.widths',op,(;),0.)
        @test maximum(abs,d)>.01 # uniform density is not the Ito equilibrium
        scalar=compile(PBMModel(SizeDiffusion(.03)),g)
        pair=compile(PBMModel(SizeDiffusion((.03,.03))),g)
        ds=zero(N);dp=zero(N);rhs!(ds,N,scalar,(;),0.);rhs!(dp,N,pair,(;),0.)
        @test ds==dp
        for D in ((-.1,.2),(0.,Inf),(NaN,0.))
            @test_throws DomainError compile(PBMModel(SizeDiffusion(D)),g)
            bad=compile(PBMModel(SizeDiffusion((xy,p,t,c)->D)),g)
            @test_throws DomainError rhs!(d,N,bad,(;),0.)
            @test_throws DomainError suggest_dt(bad,N,(;),0.)
        end
        for D in ((.1,),(.1,.2,.3),[.1 .2;.3 .4])
            @test_throws ArgumentError compile(PBMModel(SizeDiffusion(D)),g)
        end
        for b in (SizeBC(OpenBoundary(),NoFlux()),SizeBC(InflowFlux((p,t,c)->1),NoFlux()),
                  SizeBC(AbsorbingZero(),NoFlux()))
            @test_throws ArgumentError compile(PBMModel(SizeDiffusion(.1)),g;bc=SizeBC2D(x=b))
        end
        @test_throws ArgumentError compile(PBMModel(SizeDiffusion(.1)),g;evolution=HOCFiniteVolume())
        @test_throws ArgumentError compile(PBMModel(SizeDiffusion(.1)),g;
            methods=TermMethods(size_diffusion=Upwind()))
        base=compile(PBMModel(Growth2D((.2,.1))),g)
        zeroD=compile(PBMModel(Growth2D((.2,.1))+SizeDiffusion((0.,0.))),g)
        @test simulate(PBMProblem(base,N,(0.,.1));dt=.002).u==
              simulate(PBMProblem(zeroD,N,(0.,.1));dt=.002).u
    end

    @testset "Independent continuum and temporal refinement" begin
        for stretch in (0.,.3)
            errors=Float64[]
            for n in (12,24,48)
                edges(L,m)=[L*(s+stretch*sin(2pi*s)/(2pi)) for s in range(0,1;length=m+1)]
                g=Grid2D(edges(1.,n),edges(2.,n÷2));D=(.02,.03)
                op=compile(PBMModel(SizeDiffusion(D)),g;
                    bc=SizeBC2D(x=_tensor_periodic,y=_tensor_periodic))
                pb=PBMProblem(op,_tensor_smooth_bins(g,0.),(0.,.2))
                dt=suggest_dt(op,pb.u0,pb.p,0.;cfl=.4)
                final=last(simulate(pb;dt).u)
                push!(errors,sum(abs,final-_tensor_smooth_bins(g,.2)))
                @test sum(final)≈sum(pb.u0) atol=2e-14
            end
            @test log2(errors[2]/errors[3])>1.8
            @info "Tensor diffusion spatial refinement" stretch errors order=log2(errors[2]/errors[3])
        end
        g=Grid2D(range(0,1;length=9),range(0,2;length=7));D=(.1,.2)
        op=compile(PBMModel(SizeDiffusion(D)),g;bc=SizeBC2D(x=_tensor_periodic,y=_tensor_periodic))
        initial=_tensor_smooth_bins(g,0.;D);pb=PBMProblem(op,initial,(0.,.08))
        Ax=_tensor_diffusion_matrix(g.x,D[1],true);Ay=_tensor_diffusion_matrix(g.y,D[2],true)
        A=kron(Matrix{Float64}(I,6,6),Ax)+kron(Ay,Matrix{Float64}(I,8,8))
        reference=reshape(exp(.08A)*vec(initial),size(g))
        errors=[sum(abs,last(simulate(pb;dt).u)-reference) for dt in (.02,.01,.005)]
        @test log2(errors[2]/errors[3])>2.8
        @info "Tensor SSPRK33 temporal refinement against matrix exponential" errors order=log2(errors[2]/errors[3])

        errors=Float64[]
        for (nx,ny) in ((48,32),(96,64),(192,128))
            g=Grid2D(range(0,12;length=nx+1),range(0,10;length=ny+1))
            G=(.3,.15);D=(.05,.02);t=.2
            model=PBMModel(Growth2D(G)+SizeDiffusion(D))
            op=compile(model,g)
            initial=_tensor_gaussian_bins(g,0.);pb=PBMProblem(op,initial,(0.,t))
            dt=suggest_dt(op,initial,(;),0.;cfl=.35)
            if nx==48
                # A high-order reconstruction has no positivity guarantee on
                # this coarse, nearly empty-tailed profile; retain its failure.
                high=PBMProblem(compile(model,g;growth_scheme=WENO5()),initial,(0.,t))
                @test_throws NumericalFailure simulate(high;dt)
            end
            final=last(simulate(pb;dt).u)
            push!(errors,sum(abs,final-_tensor_gaussian_bins(g,t)))
            @test sum(final)≈sum(initial) atol=2e-14
            for (order,velocity,diffusivity) in (((1,0),G[1],D[1]),((0,1),G[2],D[2]))
                second=(2order[1],2order[2])
                mu0=moment(g,initial,order);mu=moment(g,final,order)
                var0=moment(g,initial,second)-mu0^2
                var=moment(g,final,second)-mu^2
                @test abs(mu-mu0-velocity*t)<.003
                h=order[1]==1 ? first(g.x.widths) : first(g.y.widths)
                # Upwind adds known numerical variance |G|*h*t. It vanishes
                # under refinement and is distinct from physical 2D*t.
                @test var-var0≈2diffusivity*t+abs(velocity)*h*t atol=1e-10
            end
        end
        @test errors[3]<errors[2]<errors[1]
        @test log2(errors[2]/errors[3])>.8 # first-order Upwind limits the combination
        @info "Tensor Gaussian with Upwind and physical diffusion" errors order=log2(errors[2]/errors[3])
    end

    @testset "Current-stage bulk, shaped accessors and discrete material" begin
        g=Grid2D(range(0,4;length=17),range(0,3;length=13))
        growth=Growth2D((x,y,p,t,c)->(.08c.bulk.c,.04c.bulk.c))
        diffusion=SizeDiffusion((xy,p,t,c)->(.02c.bulk.c*(1+t),.03c.bulk.c*(1+t)))
        birth=Nucleation2D((p,t,c)->.1c.bulk.c,(1.,1.))
        op=compile(PBMModel(growth=growth,dispersion=diffusion,births=birth),g)
        weight=(xy,p,t,c)->.01xy[1]*xy[2]^2
        system=PBMSystem((crystals=op,);exchanges=(ConservativeExchange(:crystals,:c,weight),),
            fields=(p,t)->(temperature=300.,),nonnegative_bulk=(:c,))
        initial=[exp(-((x-2)^2+(y-1.5)^2)/.1) for x in g.x.centers,y in g.y.centers]
        initial .*= g.x.widths*g.y.widths';initial ./= sum(initial)
        pb=PBMProblem(system,(crystals=initial,),(0.,.08);bulk0=(c=2.,))
        property=[weight((x,y),pb.p,0.,nothing) for x in g.x.centers,y in g.y.centers]
        invariant(u)=bulk(pb,u,:c)+sum(property.*population(pb,u,:crystals))
        @test size(population(pb,pb.u0,:crystals))==size(g)
        @test size(stateview(pb,pb.u0).populations.crystals)==size(g)
        @test_throws DimensionMismatch PBMProblem(system,(crystals=initial',),(0.,.1);bulk0=(c=2.,))
        d=zero(pb.u0);rhs!(d,pb.u0,pb.operator,pb.p,.4)
        @test bulk(pb,d,:c)+sum(property.*population(pb,d,:crystals))≈0 atol=2e-16
        @test sum(population(pb,d,:crystals))≈.2 atol=3e-15
        ctx=PopulationBalanceModels._system_context(pb.operator,pb.u0,d,pb.p,.4)
        @test ctx.fields.temperature==300.
        @test moment(ctx,:crystals,(1,2))≈moment(g,initial,(1,2))
        @test observe(PopulationMoment(:crystals,(0,0)),pb,pb.u0;time=0.)≈[1.]
        @test_throws ArgumentError observe(PopulationHistogram(:crystals,[0.,1.,4.]),pb,pb.u0;time=0.)
        @test_throws ArgumentError observe(PopulationMoment(:crystals,0),pb,pb.u0;time=0.)
        doubled=copy(pb.u0);doubled[end]=4.;d2=zero(doubled)
        rhs!(d2,doubled,pb.operator,pb.p,.4)
        @test d2≈2d rtol=2e-14
        @test suggest_dt(pb.operator,pb.u0,pb.p,0.)≈suggest_dt(op,initial,pb.p,0.;context=ctx)
        for method in (ForwardEuler(),SSPRK33(),RK4())
            sol=simulate(pb;dt=.001,method)
            @test maximum(abs(invariant(u)-invariant(pb.u0)) for u in sol.u)<2e-14
            @test minimum(last(sol.u))>=0
        end
        manual=PBMSystem((crystals=op,);bulk_rhs=(ctx,p,t)->
            (c=-sum(property.*ctx.derivatives.crystals),),nonnegative_bulk=(:c,))
        manualpb=PBMProblem(manual,(crystals=initial,),pb.tspan;bulk0=(c=2.,))
        dm=zero(manualpb.u0);rhs!(dm,manualpb.u0,manualpb.operator,pb.p,.4)
        @test dm≈d atol=2e-15
        for mechanisms in ((:growth,),(:size_diffusion,))
            selected=PBMSystem((crystals=op,);exchanges=(ConservativeExchange(
                :crystals,:c,weight;mechanisms),))
            pulse=zeros(size(g));pulse[7,6]=1
            selectedpb=PBMProblem(selected,(crystals=pulse,),(0.,.01);bulk0=(c=2.,))
            ds=zero(selectedpb.u0);rhs!(ds,selectedpb.u0,selectedpb.operator,selectedpb.p,.4)
            if mechanisms==(:size_diffusion,)
                expected=-.01*2*(.03*2*1.4)*g.x.centers[7]
                @test bulk(selectedpb,ds,:c)≈expected atol=2e-15
            else
                onlygrowth=compile(PBMModel(growth),g)
                r=zero(pulse);rhs!(r,pulse,onlygrowth,pb.p,.4;context=ctx)
                @test bulk(selectedpb,ds,:c)≈-sum(property.*r) atol=2e-15
            end
        end
        @test_throws ArgumentError PBMSystem((crystals=compile(PBMModel(Growth2D((1.,0.))),g;
            bc=SizeBC2D(x=SizeBC(NoFlux(),OpenBoundary()))),);
            exchanges=(ConservativeExchange(:crystals,:c,weight),))
        @test_throws ArgumentError PBMSystem((crystals=op,);accounting=(CumulativeAccounting(:crystals),))
        @test_throws ArgumentError simulate(pb;dt=.001,method=MeasureEuler())

        function endpoint(k)
            dop=compile(PBMModel(SizeDiffusion((xy,p,t,c)->
                (p.k*.01c.bulk.c*(1+xy[1]),p.k*.02c.bulk.c*(1+t)))),g)
            sys=PBMSystem((crystals=dop,);exchanges=(ConservativeExchange(:crystals,:c,weight),))
            problem=PBMProblem(sys,(crystals=initial.+zero(k),),(0.,.01);
                bulk0=(c=2+zero(k),),p=(k=k,))
            bulk(problem,last(simulate(problem;dt=.0005).u),:c)
        end
        @test ForwardDiff.derivative(endpoint,.8)≈(endpoint(.80001)-endpoint(.79999))/.00002 rtol=2e-5 atol=1e-9
        # Zero-valued dual entries with negative derivatives are not negative populations.
        u=ForwardDiff.Dual.(zeros(size(g)),-ones(size(g)))
        @test all(iszero,ForwardDiff.value.(PBMProblem(op,u,(0.,.1)).u0))
    end

    @testset "Mixed representations and axial tensor inventories" begin
        g=Grid2D([0.,.2,.5,1.],[0.,.3,1.])
        tensor=compile(PBMModel(Growth2D((.1,.05))),g)
        oneD=compile(PBMModel(),g.x)
        moments=compile(PBMModel(),MomentSet(0:1))
        sys=PBMSystem((tensor=tensor,line=oneD,moments=moments))
        values=g.x.widths*g.y.widths'
        pb=PBMProblem(sys,(tensor=values,line=g.x.widths,moments=[1.,.5]),(0.,.01))
        ctx=stateview(pb,pb.u0)
        @test size(ctx.populations.tensor)==size(g)
        @test ctx.populations.line isa AbstractVector
        @test ctx.populations.moments isa AbstractVector
        d=zero(pb.u0);rhs!(d,pb.u0,pb.operator,pb.p,0.)
        @test sum(population(pb,d,:tensor))≈0 atol=1e-14
        @test all(iszero,population(pb,d,:line))

        property=[.01x*y^2 for x in g.x.centers,y in g.y.centers]
        system=PBMSystem((particles=tensor,);exchanges=(ConservativeExchange(
            :particles,:c,(xy,p,t,c)->.01xy[1]*xy[2]^2),))
        localpb=PBMProblem(system,(particles=values,),(0.,.01);bulk0=(c=2.,))
        z=AxialGrid([0.,.1,.4,1.])
        axial=axial_problem(localpb,AxialDispersion(z;velocity=.2,dispersion=.01,boundary=:periodic))
        axial.u0[:,2].*=1.4
        du=zero(axial.u0);rhs!(du,axial.u0,axial.operator,axial.p,0.)
        weighted(u)=sum(z.widths[c]*(axial_stateview(axial,u,c).bulk.c+
            sum(property.*axial_stateview(axial,u,c).populations.particles)) for c in 1:length(z))
        @test abs(weighted(du))<1e-14
        @test weighted(last(simulate(axial;dt=.0005).u))≈weighted(axial.u0) atol=1e-13
        @test size(axial_stateview(axial,axial.u0,1).populations.particles)==size(g)
        dispersed=compile(PBMModel(SizeDiffusion((.01,.02))),g)
        dpb=PBMProblem(PBMSystem((particles=dispersed,)),(particles=values,),(0.,.01))
        @test_throws ArgumentError axial_problem(dpb,AxialDispersion(z;velocity=.2,dispersion=.01))
        @test_throws ArgumentError compile(PBMModel(SizeDiffusion(.1)),g;
            domain=AxialDispersion(z;velocity=.2,dispersion=.01))
    end
end
