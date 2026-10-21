@testset "Axial conservative transport" begin
    intrinsic=Grid1D([0,1,3]);z=AxialGrid([0,0.1,0.4,1.0])
    for boundary in (:danckwerts,:periodic,:closed)
        v=boundary===:closed ? 0.0 : 0.2
        domain=AxialDispersion(z;velocity=v,dispersion=0.03,boundary,
            inlet=boundary===:danckwerts ? ((p,t)->[2.0,3.0]) : nothing)
        op=compile(PBMModel(),intrinsic;domain)
        N=[1.0 2.0 3.0;4.0 2.0 1.0];du=similar(N);rhs!(du,N,op,(;),0.0)
        F=axial_fluxes(op,N,(;),0.0)
        @test vec(sum(du.*z.widths';dims=2))≈F[:,1]-F[:,end] atol=1e-13
        @test suggest_dt(op,N,(;),0.0)>0
        if boundary!==:danckwerts
            @test sum(du.*z.widths')≈0 atol=1e-13
        end
        sol=simulate(PBMProblem(op,N,(0.0,0.1));dt=0.5*suggest_dt(op,N,(;),0.0),method=SSPRK22())
        @test minimum(sol.u[end])>=0
    end
    # An inlet at the same concentration balances the advective outlet exactly.
    domain=AxialDispersion(z;velocity=0.7,dispersion=0.3,inlet=(p,t)->[2.0,3.0])
    op=compile(PBMModel(),intrinsic;domain);N=repeat([2.0,3.0],1,3);du=similar(N)
    rhs!(du,N,op,(;),0.0);@test maximum(abs.(du))<1e-13
    @test_throws DomainError AxialDispersion(z;velocity=-1,dispersion=0)
    @test_throws ArgumentError AxialDispersion(z;velocity=1,dispersion=0,boundary=:closed)
    @test_throws DomainError AxialDispersion(z;velocity=0,dispersion=-1)
    @test_throws ArgumentError AxialDispersion(z;velocity=1,dispersion=0,boundary=:periodic,inlet=(p,t)->[1,1])
    @test_throws ArgumentError AxialGrid([0,1,1])
    @test_throws DimensionMismatch PBMProblem(op,zeros(2,2),(0,1))
    invalid=compile(PBMModel(),intrinsic;domain=AxialDispersion(z;velocity=1,dispersion=0,inlet=(p,t)->[-1,1]))
    @test_throws NumericalFailure axial_fluxes(invalid,N,(;),0.0)
    invalidshape=compile(PBMModel(),intrinsic;domain=AxialDispersion(z;velocity=1,dispersion=0,inlet=(p,t)->[1]))
    @test_throws DimensionMismatch axial_fluxes(invalidshape,N,(;),0.0)

    # Periodic dispersion has second-order convergence to the damped Fourier mode.
    for velocity in (0.0,0.3)
        errors=Float64[]
        for n in (20,40,80)
            grid=AxialGrid(range(0,1;length=n+1));D=0.03;tf=0.1;k=2pi
            domain=AxialDispersion(grid;velocity,dispersion=D,boundary=:periodic)
            operator=compile(PBMModel(),intrinsic;domain)
            initial=1 .+0.2.*(cos.(k.*grid.edges[1:end-1]).-cos.(k.*grid.edges[2:end]))./(k.*grid.widths)
            state=repeat(initial',2,1)
            solution=simulate(PBMProblem(operator,state,(0.0,tf));dt=0.3*minimum(grid.widths)^2/D,method=RK4())
            truth=1 .+0.2exp(-D*k^2*tf).*(cos.(k.*(grid.edges[1:end-1].-velocity*tf)).-cos.(k.*(grid.edges[2:end].-velocity*tf)))./(k.*grid.widths)
            push!(errors,sum(abs.(solution.u[end][1,:].-truth).*grid.widths))
            @test sum(solution.u[end].*grid.widths')≈2 atol=1e-12
        end
        @test errors[2]/errors[3]>(velocity==0 ? 3.8 : 1.8)
    end
end

@testset "Axial cells with intrinsic mechanisms" begin
    g=Grid1D(0:0.2:1);z=AxialGrid(0:0.25:1)
    domain=AxialDispersion(z;velocity=0.1,dispersion=0.02,boundary=:periodic)
    source=PBMModel(Birth((a,p,t,ctx)->ctx.fields.z+ctx.fields.extra))
    op=compile(source,g;domain,fields=(c,t)->(extra=2.0,))
    N=zeros(length(g),length(z));dN=similar(N);rhs!(dN,N,op,(;),0.0)
    @test dN≈g.widths*(z.centers.+2)'
    @test_throws ArgumentError compile(PBMModel(Growth(AffineGrowth(1,0))),g;domain,evolution=Inguva2022())
    # Intrinsic growth remains conservative independently of the physical mesh.
    for grid in (g,Grid2D(g,g))
        model=grid isa Grid1D ? PBMModel(Growth(AffineGrowth(0.2,0))) : PBMModel(Growth2D((0.2,-0.1)))
        bc=grid isa Grid1D ? SizeBC(Periodic(),Periodic()) : SizeBC2D(x=SizeBC(Periodic(),Periodic()),y=SizeBC(Periodic(),Periodic()))
        operator=compile(model,grid;domain,bc)
        state=reshape(collect(1.0:length(grid)*length(z)),length(grid),length(z));du=similar(state)
        rhs!(du,state,operator,(;),0.0)
        @test sum(du.*z.widths')≈0 atol=1e-12
        @test size(axial_population(operator,state,2))==(grid isa Grid1D ? (length(grid),1) : size(grid))
        step=suggest_dt(operator,state,(;),0.0)
        solution=simulate(PBMProblem(operator,state,(0.0,0.01));dt=step)
        @test minimum(solution.u[end])>=0
    end
end
