include("fixtures/example_modules.jl")
_load_example(:PBM2DBenchmark, "two_coordinate_transport.jl")
@testset "Two-coordinate transport" begin
    @testset "Tensor sources and directional fluxes" begin
        g=Grid2D(0:0.1:1,0:0.2:2)
        N=zeros(size(g)); dN=similar(N)
        op=compile(PBMModel(Nucleation2D((p,t,c)->2.0,(0.34,0.71))),g)
        rhs!(dN,N,op,(;),0.0)
        @test sum(dN)≈2.0
        @test moment(g,dN,(1,0))≈2*0.34
        @test moment(g,dN,(0,1))≈2*0.71
        dop=compile(PBMModel(DistributedNucleation((p,t,c)->3.0,(xy,p,t,c)->exp(-sum(abs2,xy)))),g)
        rhs!(dN,N,dop,(;),0.0)
        @test sum(dN)≈3.0
        periodic=SizeBC(Periodic(),Periodic())
        for velocity in ((0.4,-0.2),(-0.4,0.2))
            top=compile(PBMModel(Growth2D(velocity)),g;growth_scheme=WENO5(),bc=SizeBC2D(x=periodic,y=periodic))
            N.=((1 .+ 0.2cos.(2π.*g.x.centers))*(1 .+ 0.3sin.(π.*g.y.centers))').*(g.x.widths*g.y.widths')
            rhs!(dN,N,top,(;),0.0)
            @test abs(sum(dN))<1e-14
            @test suggest_dt(top,N,(;),0.0)≈0.5/(0.4/0.1+0.2/0.2)
        end
        @test_throws ArgumentError compile(PBMModel(Aggregation(ConstantKernel(1.0))),g)
        @test_throws ArgumentError compile(PBMModel(Growth2D((1,1))),g;growth_scheme=MUSCL())
    end
    @testset "Analytical convergence and exact shift" begin
        ew=Float64[]; eh=Float64[]
        for n in (16,32,64)
            prob=PBM2DBenchmark.problem(n); ref=PBM2DBenchmark.reference(prob.operator.grid,0.25)
            a=simulate(prob;dt=0.4/n,saveat=[0.25],check_nonnegative=false)
            b=simulate(prob;dt=2/n,method=HOCADI(),saveat=[0.25],check_nonnegative=false)
            c=simulate(prob;dt=2/n,method=ExactShift(),saveat=[0.25])
            area=prob.operator.grid.x.widths*prob.operator.grid.y.widths'
            push!(ew,sum(abs.(a.u[end].-ref)))
            push!(eh,sum(abs.(b.u[end].-ref)))
            @test maximum(abs,(c.u[end].-ref)./area)<8e-15
            @test abs(sum(b.u[end])-sum(prob.u0))<3e-13
        end
        @test log2(ew[2]/ew[3])>3.0
        @test log2(eh[2]/eh[3])>3.7
        # Preserve exact discontinuities without a fractional remap.
        prob=PBM2DBenchmark.problem(32;profile=:rectangle)
        sol=simulate(prob;dt=2/32,method=ExactShift(),saveat=[0.25])
        @test sol.u[end]≈PBM2DBenchmark.reference(prob.operator.grid,0.25;profile=:rectangle) atol=1e-16
        @test minimum(sol.u[end])==0.0
        @test_throws NumericalFailure simulate(prob;dt=2/32,method=HOCADI(),saveat=[0.25])
        @test_throws ArgumentError simulate(prob;dt=0.01,method=ExactShift())
        @test_throws ArgumentError HOCADI(time_order=3)
        g=prob.operator.grid
        source=compile(PBMModel(Growth2D((0.5,0.25))+Birth((xy,p,t,c)->1.0)),g;bc=prob.operator.bc)
        @test_throws ArgumentError simulate(PBMProblem(source,prob.u0,prob.tspan);method=HOCADI(),dt=0.01)
        nonperiodic=compile(PBMModel(Growth2D((0.5,0.25))),g;bc=SizeBC2D(x=SizeBC(OpenBoundary(),OpenBoundary()),y=SizeBC(OpenBoundary(),OpenBoundary())))
        np=PBMProblem(nonperiodic,prob.u0,prob.tspan)
        @test_throws ArgumentError simulate(np;dt=2/32,method=HOCADI())
        @test simulate(np;dt=2/32,method=ExactShift(),saveat=[0.25]).u[end]≈sol.u[end]
        @test PopulationBalanceModels._observe(MomentObservable((0,0)),sol.u[end],prob.operator,prob.p,0.25)==[sum(sol.u[end])]
        @test_throws ArgumentError PopulationBalanceModels._observe(MomentObservable(0),sol.u[end],prob.operator,prob.p,0.25)
    end
    @testset "HOCADI Fourier factors on unequal axes" begin
        # Closed-form amplification, independent of the cyclic solve and its
        # matrix assembly. Initial/reference values are exact bin integrals.
        nx,ny=12,9
        lengths=(2.0,3.0)
        g=Grid2D(range(0,lengths[1];length=nx+1),range(0,lengths[2];length=ny+1))
        area=g.x.widths*g.y.widths'
        periodic=SizeBC(Periodic(),Periodic())
        for velocity in ((0.7,-0.2),(0.0,0.4)),order in (2,4)
            policy=HOCADI(time_order=order)
            op=compile(PBMModel(Growth2D(velocity)),g;
                evolution=policy,bc=SizeBC2D(x=periodic,y=periodic))
            for (kx,ky) in ((0,0),(1,2),(5,4),(0,3)),cfl in (0.5,2.0,8.0)
                dt=cfl/max(abs(velocity[1])/first(g.x.widths),abs(velocity[2])/first(g.y.widths))
                phase=[2π*(kx*x/lengths[1]+ky*y/lengths[2]) for x in g.x.centers,y in g.y.centers]
                amplitude=0.25sinc(kx/nx)*sinc(ky/ny)
                initial=(2 .+ amplitude.*cos.(phase)).*area
                gamma=1/(2-2^(1/3))
                factors=order==2 ? (1.0,) : (gamma,1-2gamma,gamma)
                gain=1.0+0im
                for (k,n,h,v) in zip((kx,ky),(nx,ny),
                                      (first(g.x.widths),first(g.y.widths)),velocity)
                    theta=2π*k/n
                    frequency=3sin(theta)/(2h*(1+0.5cos(theta)))
                    for factor in factors
                        z=0.5factor*dt*v*frequency
                        gain*=(1-im*z)/(1+im*z)
                    end
                end
                reference=(2 .+ amplitude.*real.(gain.*cis.(phase))).*area
                solution=simulate(PBMProblem(op,initial,(0.0,dt));dt)
                final=last(solution.u)
                @test final≈reference rtol=3e-13 atol=3e-15
                @test sum(final)≈sum(initial) rtol=3e-14
                @test sqrt(sum(abs2,final.-2 .*area))≈sqrt(sum(abs2,initial.-2 .*area)) rtol=3e-13 atol=3e-14
            end
        end
    end
    @testset "Independent temporal order of the compact method" begin
        n=64; g=Grid2D(range(0,1;length=n+1),range(0,1;length=n+1))
        periodic=SizeBC(Periodic(),Periodic()); velocity=(0.6,-0.2)
        op=compile(PBMModel(Growth2D(velocity)),g;bc=SizeBC2D(x=periodic,y=periodic))
        f(t,ωx,ωy)=[2+0.3cos(2π*x-ωx*t)+0.2sin(2π*y-ωy*t) for x in g.x.centers,y in g.y.centers]
        θ=2π/n; symbol=(3sin(θ)*n/2)/(1+0.5cos(θ))
        area=g.x.widths*g.y.widths'; reference=f(0.5,velocity[1]*symbol,velocity[2]*symbol).*area
        prob=PBMProblem(op,f(0,0,0).*area,(0.0,0.5))
        errors=Dict(order=>Float64[] for order in (2,4))
        for order in (2,4),dt in (0.5/8,0.5/16,0.5/32)
            sol=simulate(prob;method=HOCADI(time_order=order),dt,saveat=[0.5])
            push!(errors[order],sum(abs.(sol.u[end].-reference)))
        end
        @test 1.9<log2(errors[2][2]/errors[2][3])<2.1
        @test 3.8<log2(errors[4][2]/errors[4][3])<4.2
    end
end
