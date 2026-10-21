@testset "Growth reconstructions and explicit time methods" begin
    for (method,order) in ((ForwardEuler(),1),(SSPRK22(),2),(SSPRK33(),3),(RK4(),4),(SSPRK54(),4))
        g=Grid1D(0:0.5:1)
        prob=PBMProblem(compile(PBMModel(Death((a,p,t,ctx)->1.0)),g),ones(2),(0.0,1.0))
        errors=[abs(simulate(prob;method,dt).u[end][1]-exp(-1)) for dt in (0.1,0.05)]
        @test errors[1]/errors[2]>0.9*2^order
        @test_throws NumericalFailure simulate(prob;method,dt=0.01,maxsteps=1)
        @test_throws ArgumentError simulate(prob;method,dt=0.0)
        source=PBMProblem(compile(PBMModel(Birth((a,p,t,ctx)->t^2)),g),zeros(2),(0.0,1.0))
        integral=simulate(source;method,dt=0.05,saveat=[0.0,0.37,1.0]).u[end]
        @test sum(integral)≈1/3 atol=(order>=3 ? 1e-13 : 0.03)
        # Every stage must see the simultaneous bulk state, including RK4.
        op=compile(PBMModel(Nucleation((p,t,ctx)->0.3ctx.bulk.c,0.25)),g)
        system=PBMSystem((particles=op,);nonnegative_bulk=(:c,),
            exchanges=(ConservativeExchange(:particles,:c,(a,p,t,ctx)->1.0;mechanisms=(:nucleation,)),))
        coupled=PBMProblem(system,(particles=zeros(2),),(0.0,1.0);bulk0=(c=2.0,))
        result=simulate(coupled;method,dt=0.01)
        @test bulk(coupled,result.u[end],:c)+sum(population(coupled,result.u[end],:particles))≈2 atol=1e-13
        @test bulk(coupled,result.u[end],:c)≈2exp(-0.3) atol=1e-3
    end
    for scheme in (Centered(),LinearUpwind()),v in (-0.7,0.7)
        errors=Float64[]
        for n in (40,80,160)
            g=Grid1D(range(0,2pi;length=n+1));N=reshape(2g.widths.+cos.(g.edges[1:end-1]).-cos.(g.edges[2:end]),:,1)
            op=compile(PBMModel(Growth(AffineGrowth(v,0))),g;methods=TermMethods(growth=scheme),bc=SizeBC(Periodic(),Periodic()))
            dN=similar(N);rhs!(dN,N,op,(;),0.0)
            exact=v.*(sin.(g.edges[1:end-1]).-sin.(g.edges[2:end]))
            push!(errors,maximum(abs.(vec(dN).-exact)./g.widths))
            @test abs(sum(dN))<1e-12
        end
        @test errors[1]/errors[2]>3.8
        @test errors[2]/errors[3]>3.8
        @test_throws ArgumentError compile(PBMModel(Growth(AffineGrowth(v,0))),Grid1D([0,1,2,4]);growth_scheme=scheme)
    end
end

@testset "Independent transport and time-order checks" begin
    # A time-dependent rate tests the nonautonomous SSPRK(5,4) stage times.
    g=Grid1D(0:0.5:1)
    problem=PBMProblem(compile(PBMModel(Death((a,p,t,ctx)->t)),g),ones(2),(0.,1.))
    errors=[abs(simulate(problem;method=SSPRK54(),dt).u[end][1]-exp(-0.5)) for dt in (0.1,0.05)]
    @test errors[1]/errors[2]>14
    # Compare to the independently expanded Lax--Wendroff difference formula.
    for v in (-1.,1.),C in (0.25,0.5,1.)
        grid=Grid1D(range(0,1;length=65))
        initial=[2+sin(2pi*i/64)+0.1cos(22pi*i/64) for i in 1:64]
        op=compile(PBMModel(Growth(AffineGrowth(v,0))),grid;evolution=LaxWendroff(),bc=SizeBC(Periodic(),Periodic()))
        dt=C/64;signedC=v*C
        truth=initial.-signedC/2 .* (circshift(initial,-1).-circshift(initial,1)).+
            signedC^2/2 .* (circshift(initial,-1).-2initial.+circshift(initial,1))
        sol=simulate(PBMProblem(op,initial,(0.,dt));dt,check_nonnegative=false)
        @test vec(sol.u[end])≈truth atol=1e-14
        long=simulate(PBMProblem(op,initial,(0.,1.));dt,check_nonnegative=false)
        @test sum(abs2,long.u[end])<=sum(abs2,initial)*(1+1e-12)
    end
    for stretched in (false,true),v in (-0.7,0.7)
        errors=Float64[]
        for n in (320,640)
            s=collect(range(0,2pi;length=n+1));edges=stretched ? s.+0.2sin.(s) : s
            grid=Grid1D(edges);initial=reshape(2grid.widths.+cos.(edges[1:end-1]).-cos.(edges[2:end]),:,1)
            op=compile(PBMModel(Growth(AffineGrowth(v,0))),grid;
                methods=TermMethods(growth=WENO3()),bc=SizeBC(Periodic(),Periodic()))
            d=similar(initial);rhs!(d,initial,op,(;),0.)
            truth=v.*(sin.(edges[1:end-1]).-sin.(edges[2:end]))
            push!(errors,sum(abs.(vec(d).-truth)))
            @test abs(sum(d))<1e-12
        end
        @test errors[1]/errors[2]>6
    end
    # New limited linear method: signed transport and conserved number.
    for v in (-1.,1.)
        grid=Grid1D(0:0.01:1);initial=grid.widths.*(1 .+0.2sin.(2pi.*grid.centers))
        op=compile(PBMModel(Growth(AffineGrowth(v,0))),grid;
            methods=TermMethods(growth=VanLeer()),bc=SizeBC(Periodic(),Periodic()))
        out=simulate(PBMProblem(op,initial,(0.,0.2));method=SSPRK54(),dt=.002)
        @test minimum(out.u[end])>0
        @test sum(out.u[end])≈sum(initial) rtol=1e-12
        stretched=Grid1D([0.,.1,.3,.6,1.,1.5,2.1])
        for (scheme,initial,truth) in (
            (VanLeer(),(1 .+stretched.centers).*stretched.widths,-v*stretched.widths[3]),
            # Large epsilon isolates the ideal geometric weights: a quadratic
            # reconstructed from nonuniform cell integrals must be exact.
            (WENO3(epsilon=1e12),stretched.widths.+diff(stretched.edges.^3)./3,
             -v*(stretched.edges[4]^2-stretched.edges[3]^2)))
            op=compile(PBMModel(Growth(AffineGrowth(v,0))),stretched;methods=TermMethods(growth=scheme))
            d=similar(reshape(initial,:,1));rhs!(d,reshape(initial,:,1),op,(;),0.)
            @test d[3]≈truth atol=1e-12
        end
    end
    @test_throws ArgumentError WENO3(epsilon=0)
    @test_throws ArgumentError compile(PBMModel(Growth(AffineGrowth(1,0))),Grid1D([0,1,2]);growth_scheme=WENO3())
end

@testset "Constant-growth space–time methods" begin
    for method in (LaxWendroff(),BeamWarming()),v in (-1.0,1.0)
        g=Grid1D(range(0,1;length=101));N=1 .+ sin.(2pi.*g.centers)
        op=compile(PBMModel(Growth(AffineGrowth(v,0))),g;evolution=method,bc=SizeBC(Periodic(),Periodic()))
        prob=PBMProblem(op,N,(0.0,0.1))
        sol=simulate(prob;dt=0.01)
        @test vec(sol.u[end])≈circshift(N,v>0 ? 10 : -10) atol=2e-13
        @test sum(sol.u[end])≈sum(N) atol=1e-12
        @test_throws ArgumentError simulate(prob;dt=0.011)
        @test_throws ArgumentError rhs!(similar(prob.u0),prob.u0,op,(;),0.0)
        @test_throws ArgumentError simulate(prob;dt=0.01,method=RK4())
        errors=Float64[]
        for n in (40,80)
            grid=Grid1D(range(0,1;length=n+1));initial=grid.widths.+(cos.(2pi.*grid.edges[1:end-1]).-cos.(2pi.*grid.edges[2:end]))./(4pi)
            operator=compile(PBMModel(Growth(AffineGrowth(v,0))),grid;evolution=method,bc=SizeBC(Periodic(),Periodic()))
            out=simulate(PBMProblem(operator,initial,(0.0,0.5));dt=0.5/n)
            truth=grid.widths.+(cos.(2pi.*(grid.edges[1:end-1].-v/2)).-cos.(2pi.*(grid.edges[2:end].-v/2)))./(4pi)
            push!(errors,sum(abs.(vec(out.u[end]).-truth)))
        end
        @test errors[1]/errors[2]>3.8
    end
    g=Grid1D(0:0.1:1)
    for method in (LaxWendroff(),BeamWarming())
        @test_throws ArgumentError compile(PBMModel(Growth(AffineGrowth(1,1))),g;evolution=method)
        @test_throws ArgumentError compile(PBMModel(Growth((a,p,t,ctx)->1.0)),g;evolution=method)
        @test_throws ArgumentError compile(PBMModel(Growth(AffineGrowth(1,0))+Death((a,p,t,ctx)->1.0)),g;evolution=method)
        @test_throws ArgumentError compile(PBMModel(Growth(AffineGrowth(1,0))),g;evolution=method,methods=TermMethods(growth=Upwind()))
        @test_throws ArgumentError compile(PBMModel(Growth2D((1.0,1.0))),Grid2D(g,g);evolution=method)
    end
end
