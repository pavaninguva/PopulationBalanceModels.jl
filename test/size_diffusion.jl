using Test, ForwardDiff
using SpecialFunctions: erf

const _grd_periodic = SizeBC(Periodic(),Periodic())
_grd_rhs(op,N,p=(;),t=0.0) = (d=zero(N); rhs!(d,N,op,p,t); d)
function _grd_gaussian_bins(g,mean,variance)
    cdf(x) = (1+erf((x-mean)/sqrt(2variance)))/2
    diff(cdf.(g.edges))
end

@testset "Native Ito size diffusion" begin
    @testset "Declared uniform diffusion reuses the current coefficient" begin
        g=Grid1D([0.,.1,.3,.6,1.]);N=reshape([.1,.3,.2,.4],:,1)
        for bc in (SizeBC(),_grd_periodic),t in (0.,.4)
            generic=compile(PBMModel(SizeDiffusion((x,p,t,c)->p.D*(1+t))),g;bc)
            polynomial=compile(PBMModel(SizeDiffusion(PolynomialRate(((p,t,c)->p.D*(1+t),)))),g;bc)
            @test _grd_rhs(generic,N,(D=.2,),t)==_grd_rhs(polynomial,N,(D=.2,),t)
            function derivative(D,op)
                u=N .+ zero(D);du=zero(u);rhs!(du,u,op,(D=D,),t);vec(du)
            end
            @test ForwardDiff.derivative(D->sum(abs2,derivative(D,generic)),.2) ≈
                ForwardDiff.derivative(D->sum(abs2,derivative(D,polynomial)),.2) rtol=1e-13
        end
        calls=Ref(0)
        law=PolynomialRate(((p,t,c)->(calls[]+=1;p.D),))
        op=compile(PBMModel(SizeDiffusion(law)),g)
        _grd_rhs(op,N,(D=.1,))
        @test calls[]==1
    end
    @testset "Declarations, geometry and support" begin
        g = Grid1D([0.,.1,.35,.9,1.5];coordinate=:length)
        N = reshape([.2,.3,.1,.4],:,1)
        model = PBMModel(dispersion=SizeDiffusion(;diffusivity=.03))
        op = compile(model,g;methods=TermMethods(size_diffusion=FiniteVolumeDiffusion()))
        @test mechanism(model.terms[1]) === :size_diffusion
        @test term_methods(op).dispersion isa FiniteVolumeDiffusion
        @test term_methods(compile(model,g;methods=TermMethods(
            by_term=(dispersion=FiniteVolumeDiffusion(),)))).dispersion isa FiniteVolumeDiffusion
        @test_throws ArgumentError compile(model,g;methods=TermMethods(size_diffusion=Upwind()))
        for bc in (SizeBC(),_grd_periodic)
            op = compile(model,g;bc)
            d = _grd_rhs(op,N)
            # Independent conductance matrix, including both connecting faces
            # when the grid has only two cells.
            A = zeros(length(g),length(g))
            pairs = [(i,i+1,g.centers[i+1]-g.centers[i]) for i in 1:length(g)-1]
            bc.left isa Periodic && push!(pairs,(length(g),1,(g.widths[end]+g.widths[1])/2))
            for (i,j,h) in pairs
                A[i,i] -= .03/(h*g.widths[i]); A[j,i] += .03/(h*g.widths[i])
                A[j,j] -= .03/(h*g.widths[j]); A[i,j] += .03/(h*g.widths[j])
            end
            @test vec(d) ≈ A*vec(N) atol=1e-15
            @test abs(sum(d)) < 1e-14
            @test suggest_dt(op,N,(;),0.;cfl=.8) ≈ .8/maximum(-A[i,i] for i in 1:length(g))
            mixed = compile(PBMModel(growth=Growth((x,p,t,c)->.2),
                dispersion=SizeDiffusion(.03)),g;bc)
            @test suggest_dt(mixed,N,(;),0.;cfl=.8) ≈
                .8/maximum(-A[i,i]+.2/g.widths[i] for i in 1:length(g))
            @test diagnostics(op,N,(;),0.).unresolved_birth_number_rate == [0.]
        end
        for bc in (SizeBC(OpenBoundary(),NoFlux()),SizeBC(NoFlux(),OpenBoundary()),
                   SizeBC(AbsorbingZero(),NoFlux()),SizeBC(InflowFlux((p,t,c)->1),NoFlux()))
            @test_throws ArgumentError compile(model,g;bc)
        end
        for D in (-.1,Inf,NaN)
            @test_throws DomainError compile(PBMModel(SizeDiffusion(D)),g)
            bad = compile(PBMModel(SizeDiffusion((x,p,t,c)->D)),g)
            @test_throws DomainError _grd_rhs(bad,N)
            @test_throws DomainError suggest_dt(bad,N,(;),0.)
        end
        @test_throws DomainError _grd_rhs(compile(PBMModel(SizeDiffusion((x,p,t,c)->1im)),g),N)
        for closure in (ExactClosure(),QMOM(2),RankAdaptiveQMOM(2),EQMOM(2),DQMOM(2),LogNormalClosure())
            @test_throws ArgumentError compile(model,MomentSet(0:3);closure)
        end
        @test term_methods(compile(model,Grid2D(g,g))).dispersion isa FiniteVolumeDiffusion
        for method in (HOCFiniteVolume(),Inguva2022(),LaxWendroff())
            @test_throws ArgumentError compile(model,g;evolution=method)
            @test_throws ArgumentError simulate(PBMProblem(compile(model,g),N,(0.,.1));dt=.01,method)
        end
        @test_throws ArgumentError simulate(PBMProblem(compile(model,g),N,(0.,.1));dt=.01,method=MeasureEuler())
        domain = AxialDispersion(AxialGrid([0.,.5,1.]);velocity=.1,dispersion=.01)
        @test_throws ArgumentError compile(model,g;domain)
        localprob = PBMProblem(PBMSystem((A=compile(model,g),)),(A=vec(N),),(0.,.1))
        @test_throws ArgumentError axial_problem(localprob,domain)
        for edges in ([0.,.3,1.],Float32[0,.2,.5,1])
            small = Grid1D(edges)
            for bc in (SizeBC(),_grd_periodic)
                smallop = compile(PBMModel(SizeDiffusion(.1f0)),small;bc)
                dn = _grd_rhs(smallop,reshape(copy(small.widths),:,1))
                @test all(iszero,dn)
                @test eltype(dn) == eltype(small.widths)
            end
        end
    end

    @testset "Variable diffusivity is Ito, not Fickian" begin
        for edges in (range(0,2;length=33),[0.,.01,.06,.2,.5,1.,2.])
            g = Grid1D(edges)
            D(x,p,t,c) = p.scale*(1+x^2)
            op = compile(PBMModel(SizeDiffusion(D)),g)
            stationary = reshape(g.widths ./ (1 .+ g.centers.^2),:,1)
            @test maximum(abs.(_grd_rhs(op,stationary,(scale=.4,)))) < 2e-13
            uniform = reshape(copy(g.widths),:,1)
            @test maximum(abs.(_grd_rhs(op,uniform,(scale=.4,)))) > .01
        end
        # In the interior, n=1 and D=1+L² give ∂²(D n)/∂L²=2.
        g = Grid1D(range(0,1;length=21))
        op = compile(PBMModel(SizeDiffusion((x,p,t,c)->1+x^2)),g)
        d = _grd_rhs(op,reshape(copy(g.widths),:,1))
        @test maximum(abs.(vec(d)[2:end-1] ./ g.widths[2:end-1] .- 2)) < 5e-13
    end

    @testset "Exactly one imposed total boundary birth flux" begin
        g = Grid1D([0.,.1,.3,.6,1.])
        for side in (:left,:right), velocity in (nothing,.2,-.2)
            terms = velocity === nothing ? (;dispersion=SizeDiffusion(.03)) :
                (;growth=Growth((x,p,t,c)->velocity),dispersion=SizeDiffusion(.03))
            x = side === :left ? first(g.edges) : last(g.edges)
            model = PBMModel(;terms...,primary=Nucleation((p,t,c)->p.J*(1+t),x))
            op = compile(model,g;methods=TermMethods(nucleation=BoundaryInjection(side)))
            N = reshape(g.widths,:,1)
            @test sum(_grd_rhs(op,N,(J=2.,),.4)) ≈ 2.8 atol=1e-14
            pb = PBMProblem(op,zero(N),(0.,.2);p=(J=2.,))
            sol = simulate(pb;dt=.001)
            @test sum(sol.u[end]) ≈ 2*(.2+.2^2/2) atol=2e-14
            @test minimum(sol.u[end]) >= 0
            @test_throws ArgumentError compile(model,g;bc=_grd_periodic,
                methods=TermMethods(nucleation=BoundaryInjection(side)))
        end
        @test_throws ArgumentError compile(PBMModel(Nucleation((p,t,c)->1.,0.)),g;
            methods=TermMethods(nucleation=BoundaryInjection()))
    end

    @testset "Combined outgoing-rate recommendation and native checks" begin
        for n in (20,40)
            g = Grid1D(range(0,1;length=n+1)); h = 1/n
            model = PBMModel(growth=Growth((x,p,t,c)->p.G),
                dispersion=SizeDiffusion((x,p,t,c)->p.D),death=Death((x,p,t,c)->p.k))
            op = compile(model,g;bc=_grd_periodic)
            N = zeros(n,1); N[n÷2] = 1
            p = (G=.2,D=.03,k=.7)
            dt = suggest_dt(op,N,p,0.;cfl=1)
            @test dt ≈ 1/(p.G/h+2p.D/h^2+p.k)
            @test dt < min(h/p.G,h^2/(2p.D),1/p.k)
            for method in (ForwardEuler(),SSPRK22(),SSPRK33(),SSPRK54())
                sol = simulate(PBMProblem(op,N,(0.,2dt);p);dt=.8dt,method)
                @test minimum(sol.u[end]) >= -1e-14
            end
            # RK4 has no SSP guarantee: its fourth stage is negative at .8dt
            # for this impulse, although a smaller step passes all stages.
            @test_throws NumericalFailure simulate(PBMProblem(op,N,(0.,2dt);p);dt=.8dt,method=RK4())
            @test minimum(simulate(PBMProblem(op,N,(0.,2dt);p);dt=.5dt,method=RK4()).u[end]) >= -1e-14
            @test_throws NumericalFailure simulate(PBMProblem(op,N,(0.,1.1dt);p);dt=1.1dt,method=ForwardEuler())
            pure = compile(PBMModel(SizeDiffusion(p.D)),g;bc=_grd_periodic)
            @test suggest_dt(pure,N,(;),0.;cfl=1) ≈ h^2/(2p.D)
            @test_throws ArgumentError suggest_dt(op,N,p,0.;cfl=0)
        end
    end

    @testset "Analytical drifting Gaussian and material-rate refinement" begin
        errors = Float64[]; variance_errors = Float64[]; rate_errors = Float64[]
        for n in (120,240,480,960)
            g = Grid1D(range(0,12;length=n+1);coordinate=:length)
            G,D,t = .2,.08,.5
            model = PBMModel(growth=Growth((x,p,t,c)->G),dispersion=SizeDiffusion(D))
            op = compile(model,g;methods=TermMethods(growth=MUSCL()))
            N0 = _grd_gaussian_bins(g,4.,.25)
            dt = suggest_dt(op,reshape(N0,:,1),(;),0.;cfl=.45)
            sol = simulate(PBMProblem(op,N0,(0.,t));dt)
            N = vec(sol.u[end]); exact = _grd_gaussian_bins(g,4+G*t,.25+2D*t)
            push!(errors,sum(abs.(N-exact)))
            mean = moment(g,N,1)/sum(N)
            variance = moment(g,N,2)/sum(N)-mean^2
            push!(variance_errors,abs(variance-(.25+2D*t)))
            @test abs(mean-(4+G*t)) < 2e-4
            @test sum(N) ≈ sum(N0) atol=2e-13
            @test minimum(N) >= -1e-14
            dN = vec(_grd_rhs(op,reshape(N0,:,1)))
            push!(rate_errors,abs(moment(g,dN,3)-(3G*moment(g,N0,2)+6D*moment(g,N0,1))))
            diffusion = compile(PBMModel(SizeDiffusion(D)),g)
            dD = vec(_grd_rhs(diffusion,reshape(N0,:,1)))
            @test moment(g,dD,3) ≈ 6D*moment(g,N0,1) atol=2e-11
        end
        @test all(errors[1:end-1] ./ errors[2:end] .> 3.)
        @test errors[end] < .001
        @test all(variance_errors[1:end-1] ./ variance_errors[2:end] .> 3.)
        # The limited growth stencil's coarsest pair is not yet asymptotic.
        @test all(rate_errors[2:end-1] ./ rate_errors[3:end] .> 3.)
        println("GRD Gaussian refinement: L1=",errors," variance=",variance_errors," material rate=",rate_errors)
    end

    @testset "Smooth graded meshes and time-dependent coefficients" begin
        for periodic in (false,true)
            errors = Float64[]
            for n in (24,48,96)
                s = collect(range(0,2pi;length=n+1))
                g = Grid1D(s .+ .3sin.(s))
                # D(t)=.2(1+t): the exact heat equation integrates D in time.
                op = compile(PBMModel(SizeDiffusion((x,p,t,c)->p.D*(1+t))),g;
                    bc=periodic ? _grd_periodic : SizeBC())
                N0 = g.widths .+ .2diff(sin.(g.edges))
                t = .4
                exact = g.widths .+ .2exp(-.2*(t+t^2/2))*diff(sin.(g.edges))
                dt = suggest_dt(op,reshape(N0,:,1),(D=.2,),t;cfl=.8)
                sol = simulate(PBMProblem(op,N0,(0.,t);p=(D=.2,));dt)
                push!(errors,sum(abs.(vec(sol.u[end])-exact)))
                @test sum(sol.u[end]) ≈ sum(N0) atol=1e-13
            end
            @test all(errors[1:end-1] ./ errors[2:end] .> 3.5)
            @info "GRD graded heat equation" periodic L1=errors
        end
        # Independent temporal refinement of one fixed semidiscrete eigenmode:
        # use its exact discrete eigenvalue, not the continuum spatial operator.
        g = Grid1D(range(0,2pi;length=9)); h = first(g.widths)
        lambda = 4sin(h/2)^2/h^2
        initial = g.widths .+ .2diff(sin.(g.edges))
        op = compile(PBMModel(SizeDiffusion((x,p,t,c)->.5*(1+t))),g;bc=_grd_periodic)
        exact = g.widths .+ .2exp(-lambda*.75)*diff(sin.(g.edges))
        errors = [maximum(abs.(vec(simulate(PBMProblem(op,initial,(0.,1.));dt).u[end])-exact))
                  for dt in (.2,.1,.05)]
        @test all(errors[1:end-1] ./ errors[2:end] .> 6.)
        println("GRD SSPRK33 temporal refinement: ",errors)
    end

    @testset "Current-stage bulk coefficients and discrete exchange" begin
        g = Grid1D(range(0,2pi;length=41);coordinate=:length)
        N0 = g.widths .+ .2diff(sin.(g.edges))
        D(x,p,t,ctx) = p.D*ctx.bulk.b
        op = compile(PBMModel(SizeDiffusion(D)),g;bc=_grd_periodic)
        sys = PBMSystem((A=op,);bulk_rhs=(ctx,p,t)->(b=-ctx.bulk.b,))
        pb = PBMProblem(sys,(A=N0,),(0.,.4);bulk0=(b=1.,),p=(D=.2,))
        sol = simulate(pb;dt=.001)
        # Independent discrete Fourier eigenvalue; only time integration remains.
        h = first(g.widths); lambda = 4sin(h/2)^2/h^2
        expected = g.widths .+ .2exp(-lambda*.2*(1-exp(-.4)))*diff(sin.(g.edges))
        @test maximum(abs.(population(pb,sol.u[end],:A)-expected)) < 2e-13
        # SSPRK33's scalar decay error is t*dt³*exp(-t)/24 + O(dt⁴),
        # approximately 1.12e-11 here, rather than roundoff.
        @test bulk(pb,sol.u[end],:b) ≈ exp(-.4) atol=2e-11
        @test suggest_dt(pb.operator,pb.u0,pb.p,0.) ≈ h^2/(4*.2)
        @test suggest_dt(pb.operator,sol.u[end],pb.p,.4) ≈ h^2/(4*.2exp(-.4))
        changed = remake(pb;p=(D=-.1,))
        @test_throws DomainError simulate(changed;dt=.001)
        # A callback becoming invalid later must also be checked at that stage.
        invalid = compile(PBMModel(SizeDiffusion((x,p,t,c)->.01-t)),g)
        @test_throws DomainError simulate(PBMProblem(invalid,N0,(0.,.02));dt=.005)

        for bc in (SizeBC(),_grd_periodic), selected in (nothing,(:growth,),(:size_diffusion,))
            g = Grid1D(range(0,5;length=51);coordinate=:length)
            model = PBMModel(growth=Growth((x,p,t,c)->.1),
                dispersion=SizeDiffusion((x,p,t,c)->.02*c.bulk.c/100))
            pop = compile(model,g;bc)
            N0 = _grd_gaussian_bins(g,2.,.1)
            weight(x,p,t,c) = .3x^3
            exchange = selected === nothing ? ConservativeExchange(:A,:c,weight) :
                ConservativeExchange(:A,:c,weight;mechanisms=selected)
            selected !== nothing && @test exchange.mechanisms == selected
            # Save the current population derivative as a bulk diagnostic, proving
            # that SizeDiffusion is already included in ctx.derivatives.
            sys = PBMSystem((A=pop,);exchanges=(exchange,),
                bulk_rhs=(ctx,p,t)->(marker=sum(g.centers.^3 .*ctx.derivatives.A),),
                accounting=(CumulativeAccounting(:A;properties=(material=3,)),),
                nonnegative_bulk=(:c,))
            pb = PBMProblem(sys,(A=N0,),(0.,.1);bulk0=(c=100.,marker=0.))
            du = zero(pb.u0); rhs!(du,pb.u0,pb.operator,pb.p,0.)
            rate = moment(g,population(pb,du,:A),3)
            @test bulk(pb,du,:marker) ≈ rate
            if selected === nothing
                @test bulk(pb,du,:c) ≈ -.3rate
                sol = simulate(pb;dt=min(.001,suggest_dt(pb.operator,pb.u0,pb.p,0.)))
                totals = [bulk(pb,u,:c)+.3moment(g,population(pb,u,:A),3) for u in sol.u]
                @test maximum(abs.(totals .- first(totals))) < 2e-12
                @test all(iszero,sol.u[end][pb.operator.physical_length+1:end])
            else
                onlyterm = selected == (:growth,) ? PBMModel(model.terms[1]) : PBMModel(model.terms[2])
                partial = compile(onlyterm,g;bc)
                d = zero(reshape(N0,:,1))
                ctx = PopulationBalanceModels._system_context(pb.operator,pb.u0,du,pb.p,0.)
                rhs!(d,reshape(N0,:,1),partial,pb.p,0.;context=ctx)
                @test bulk(pb,du,:c) ≈ -.3only(moment(g,d,3))
            end
        end
        # Pure diffusion transfers material without changing number.
        g = Grid1D(range(0,6;length=61);coordinate=:length)
        op = compile(PBMModel(SizeDiffusion(.03)),g)
        N = reshape(_grd_gaussian_bins(g,2.,.1),:,1)
        d = _grd_rhs(op,N)
        @test abs(sum(d)) < 1e-14
        @test only(moment(g,d,3)) > .3
        sys = PBMSystem((A=op,);exchanges=(ConservativeExchange(:A,:c,(L,p,t,c)->L^3),))
        pb = PBMProblem(sys,(A=vec(N),),(0.,.1);bulk0=(c=100.,))
        sol = simulate(pb;dt=.001)
        @test bulk(pb,sol.u[end],:c) < 100
        @test bulk(pb,sol.u[end],:c)+moment(g,population(pb,sol.u[end],:A),3) ≈
            100+moment(g,vec(N),3) atol=3e-12

        # Coefficient fields and multiple independent populations use the same
        # context machinery; the most restrictive population controls the bound.
        A = compile(PBMModel(SizeDiffusion((L,p,t,c)->p.D*c.fields.factor)),g)
        B = compile(PBMModel(SizeDiffusion(.06)),g)
        sys = PBMSystem((A=A,B=B);fields=(p,t)->(factor=2.,))
        pb = PBMProblem(sys,(A=vec(N),B=vec(N)),(0.,.1);p=(D=.01,))
        du = zero(pb.u0); rhs!(du,pb.u0,pb.operator,pb.p,0.)
        @test population(pb,du,:B) ≈ 3 .* population(pb,du,:A)
        @test suggest_dt(pb.operator,pb.u0,pb.p,0.) ≈ first(g.widths)^2/(4*.06)
    end

    @testset "AD and zero-diffusivity recovery" begin
        g = Grid1D(range(0,2pi;length=65)); h = first(g.widths)
        initial = g.widths .+ .2diff(sin.(g.edges))
        op = compile(PBMModel(SizeDiffusion((x,p,t,c)->p.D)),g;bc=_grd_periodic)
        function observable(D)
            pb = PBMProblem(op,initial .+ zero(D),(0.,.3);p=(D=D,))
            sol = simulate(pb;dt=.001,method=RK4())
            sum(vec(sol.u[end]).*cos.(g.centers))
        end
        ad = ForwardDiff.derivative(observable,.2)
        fd = (observable(.2+1e-5)-observable(.2-1e-5))/(2e-5)
        lambda = 4sin(h/2)^2/h^2
        exact = -.3lambda*.2length(g)*sin(h/2)*exp(-.3*.2lambda)
        @test ad ≈ fd rtol=1e-8
        @test ad ≈ exact rtol=2e-10
        @test isfinite(ForwardDiff.derivative(z->observable(-z),0.))
        for bc in (SizeBC(),_grd_periodic), method in (ForwardEuler(),SSPRK33(),RK4())
            model = PBMModel(growth=Growth((x,p,t,c)->.2),death=Death((x,p,t,c)->.01))
            clean = compile(model,g;bc)
            zeroD = compile(PBMModel(;growth=model.terms[1],death=model.terms[2],dispersion=SizeDiffusion(0.)),g;bc)
            @test _grd_rhs(clean,reshape(initial,:,1)) == _grd_rhs(zeroD,reshape(initial,:,1))
            a = simulate(PBMProblem(clean,initial,(0.,.1));dt=.001,method)
            b = simulate(PBMProblem(zeroD,initial,(0.,.1));dt=.001,method)
            @test a.u[end] == b.u[end]
            @test suggest_dt(clean,reshape(initial,:,1),(;),0.) == suggest_dt(zeroD,reshape(initial,:,1),(;),0.)
        end
    end
end
