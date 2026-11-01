using Test, PopulationBalanceModels, LinearAlgebra, ForwardDiff
using Distributions: Normal, pdf, cdf, truncated

@testset "Standalone axial reconstruction matches the coupled route" begin
    grids=(Grid1D([0.,0.15,0.6,1.]),Grid2D([0.,0.3,1.],[0.,0.2,0.7,2.]))
    s=collect(range(0.,1.;length=13))
    for g in grids, z in (AxialGrid(s),AxialGrid(s.+0.05sin.(2pi .* s))),
            boundary in (:danckwerts,:periodic,:closed), scheme in (Upwind(),MUSCL(),WENO3(),WENO5())
        oneD=g isa Grid1D
        model=PBMModel(growth=oneD ? Growth((x,p,t,c)->p.G) : Growth2D((0.02,0.01)),
            removal=Death((x,p,t,c)->p.k*(1+c.fields.z)))
        methods=TermMethods(growth=oneD ? WENO3() : Upwind())
        bc=oneD ? SizeBC() : SizeBC2D()
        volume=oneD ? g.widths : g.x.widths*g.y.widths'
        density=DensityInlet((x,p,t)->p.feed*(1+t))
        exact=(p,t)->p.feed*(1+t) .* volume
        v=boundary===:closed ? 0. : 0.3
        inlet=boundary===:danckwerts ? density : nothing
        d=AxialDispersion(z;velocity=v,dispersion=0.004,boundary,inlet)
        op=compile(model,g;domain=d,axial_scheme=scheme,methods,bc)
        localop=compile(model,g;methods,bc)
        initial=reshape(collect(1.:length(g)*length(z)),length(g),length(z)) ./ 100 .+0.1
        p=(G=0.02,k=0.03,feed=2.)
        localstate=oneD ? initial[:,1] : reshape(initial[:,1],size(g))
        localpb=PBMProblem(PBMSystem((A=localop,)),(A=localstate,),(0.,0.002);p)
        # Preserve both packed and named/tensor coupled inlet contracts.
        feed=boundary!==:danckwerts ? nothing : scheme isa MUSCL ?
            ((p,t)->(populations=(A=exact(p,t),),bulk=(;))) : ((p,t)->vec(exact(p,t)))
        coupled=axial_problem(localpb,AxialDispersion(z;velocity=v,dispersion=0.004,boundary,inlet=feed);
            initial,axial_scheme=scheme)
        F=axial_fluxes(op,initial,p,0.1)
        @test F≈axial_fluxes(coupled.operator,initial,p,0.1) rtol=2e-14 atol=2e-15
        du=similar(initial);other=similar(initial)
        rhs!(du,initial,op,p,0.1);rhs!(other,initial,coupled.operator,p,0.1)
        @test du≈other rtol=2e-14 atol=2e-14
        # Independent number balance: closed intrinsic growth conserves number;
        # only prescribed removal and external reactor fluxes change inventory.
        removal=sum(p.k*(1+z.centers[c])*sum(initial[:,c])*z.widths[c] for c in 1:length(z))
        boundary_rate=boundary===:danckwerts ? v*(sum(exact(p,0.1))-sum(initial[:,end])) : 0.
        @test sum(du .* z.widths')≈boundary_rate-removal atol=2e-13
        @test typeof(op.scheme)===typeof(scheme)
        @test term_methods(op).growth isa (oneD ? WENO3 : Upwind)
        if boundary===:danckwerts && scheme isa MUSCL
            standalone=PBMProblem(op,initial,(0.,0.002);p)
            @test simulate(standalone;dt=0.0001).u[end]≈simulate(coupled;dt=0.0001).u[end] rtol=2e-13
        end
    end
end

@testset "Axial reconstruction selection and eligibility" begin
    g=Grid1D([0.,0.4,1.]);s=collect(range(0.,1.;length=13))
    for z in (AxialGrid(s),AxialGrid(s.+0.05sin.(2pi .* s))), scheme in (MUSCL(),VanLeer(),WENO3(),WENO5(),Centered(),LinearUpwind())
        d=AxialDispersion(z;velocity=0.3,dispersion=0.)
        if z.widths!=fill(first(z.widths),length(z)) && scheme isa Union{Centered,LinearUpwind} && maximum(z.widths)-minimum(z.widths)>1e-10
            @test_throws ArgumentError compile(PBMModel(),g;domain=d,axial_scheme=scheme)
            continue
        end
        op=compile(PBMModel(),g;domain=d,axial_scheme=scheme)
        # Axial states are cell averages of concentration, not cell inventories.
        # Every eligible reconstruction reproduces an interior affine field.
        N=g.widths*(1 .+ z.centers)'
        @test axial_fluxes(op,N,(;),0.)[:,6]≈0.3 .* g.widths .* (1+z.edges[6]) atol=2e-14
    end
    z=AxialGrid(0.:0.1:1.);d=AxialDispersion(z;velocity=0.2,dispersion=0.01)
    @test compile(PBMModel(),g;domain=d).scheme isa Upwind
    @test_throws ArgumentError compile(PBMModel(),g;domain=d,axial_scheme=RK4())
    @test_throws ArgumentError compile(PBMModel(),g;axial_scheme=MUSCL())
    @test_throws ArgumentError compile(PBMModel(),Grid2D(g,g);axial_scheme=MUSCL())
    for scheme in (WENO5(),WENO3(),VanLeer(),Centered(),LinearUpwind())
        short=AxialDispersion(AxialGrid([0.,0.5,1.]);velocity=0.2,dispersion=0.)
        @test_throws ArgumentError compile(PBMModel(),g;domain=short,axial_scheme=scheme)
    end
    # The recommendation is unchanged; it is not a new high-order positivity theorem.
    N=ones(length(g),length(z))
    @test suggest_dt(compile(PBMModel(),g;domain=d),N,(;),0.)==
        suggest_dt(compile(PBMModel(),g;domain=d,axial_scheme=WENO5()),N,(;),0.)
end

@testset "Density inlet quadrature, units and domain tails" begin
    g=Grid1D([0.,0.1,0.4,0.9]);z=AxialGrid(0.:0.2:1.);v=0.37
    p=(N=7.,mu=0.2,sigma=0.11)
    true_bins=p.N .* diff(cdf.(Normal(p.mu,p.sigma),g.edges))
    errors=Float64[]
    for q in (1,2,4,8,16,32)
        inlet=DensityInlet((x,p,t)->p.N*pdf(Normal(p.mu,p.sigma),x);quadrature=GaussLegendre(q))
        op=compile(PBMModel(),g;domain=AxialDispersion(z;velocity=v,dispersion=0.01,inlet),axial_scheme=MUSCL())
        F=axial_fluxes(op,zeros(length(g),length(z)),p,0.)
        push!(errors,norm(F[:,1]/v-true_bins,Inf))
    end
    @test errors[1]>errors[2]>errors[3]
    @test errors[1]/errors[3]>100
    @test errors[end]<2e-14
    fraction=sum(true_bins)/p.N
    @test 0.96<fraction<0.97 # the Gaussian's omitted tail is not redistributed
    inlet=DensityInlet((x,p,t)->p.N*pdf(truncated(Normal(p.mu,p.sigma),first(g.edges),last(g.edges)),x);
        quadrature=GaussLegendre(32))
    op=compile(PBMModel(),g;domain=AxialDispersion(z;velocity=v,dispersion=0.01,inlet))
    @test sum(axial_fluxes(op,zeros(length(g),length(z)),p,0.)[:,1])≈v*p.N atol=2e-14

    tensor=Grid2D([0.,0.2,1.],[0.,0.3,0.8,2.]);p2=(A=2.,);t=0.4
    inlet=DensityInlet((xy,p,t)->p.A*(xy[1]+t)*(1+xy[2]);quadrature=GaussLegendre(2))
    op=compile(PBMModel(),tensor;domain=AxialDispersion(z;velocity=v,dispersion=0.,inlet),axial_scheme=MUSCL())
    ax=diff(tensor.x.edges.^2) ./ 2 .+ t .* tensor.x.widths
    ay=tensor.y.widths .+ diff(tensor.y.edges.^2) ./ 2
    exact=p2.A .* (ax*ay')
    N=zeros(length(tensor),length(z));F=axial_fluxes(op,N,p2,t)
    @test F[:,1]≈v .* vec(exact) atol=5e-15
    du=similar(N);rhs!(du,N,op,p2,t)
    @test sum(du .* z.widths')≈v*sum(exact) atol=2e-14
end

@testset "Live inlet parameters, time and forward derivatives" begin
    g=Grid1D([0.,0.1,0.4,0.9]);z=AxialGrid(0.:1/12:1.);v=0.3
    inlet=DensityInlet((x,p,t)->p.N*(1+t)*pdf(Normal(p.mu,p.sigma),x);quadrature=GaussLegendre(32))
    op=compile(PBMModel(),g;domain=AxialDispersion(z;velocity=v,dispersion=0.,inlet),
        axial_scheme=Upwind(),p=(N=99.,mu=0.4,sigma=0.2))
    N=zeros(length(g),length(z));q=[7.,0.2,0.11];t=0.4
    flux(q)=axial_fluxes(op,N,(N=q[1],mu=q[2],sigma=q[3]),t)[:,1]
    normal=Normal(q[2],q[3]);fraction=diff(cdf.(normal,g.edges))
    edge_pdf=pdf.(normal,g.edges)
    dmu=edge_pdf[1:end-1].-edge_pdf[2:end]
    dsigma=((g.edges[1:end-1].-q[2]) .* edge_pdf[1:end-1].-
        (g.edges[2:end].-q[2]) .* edge_pdf[2:end]) ./ q[3]
    exact=v*(1+t) .* hcat(fraction,q[1] .* dmu,q[1] .* dsigma)
    derivative=ForwardDiff.jacobian(flux,q)
    @test derivative≈exact rtol=2e-11 atol=2e-12
    h=1e-6
    finite=hcat([(flux(q+h .* I(3)[:,i])-flux(q-h .* I(3)[:,i]))/(2h) for i in 1:3]...)
    @test derivative≈finite rtol=1e-7 atol=1e-8
    @test ForwardDiff.jacobian(s->axial_fluxes(op,N,(N=q[1],mu=q[2],sigma=q[3]),s[1])[:,1],[t])[:,1]≈v*q[1] .* fraction atol=2e-14
    for time in (0.,0.8),amplitude in (2.,5.)
        @test axial_fluxes(op,N,(N=amplitude,mu=q[2],sigma=q[3]),time)[:,1]≈v*amplitude*(1+time) .* fraction atol=2e-14
    end
    function inventory(amplitude)
        pb=PBMProblem(op,zeros(typeof(amplitude),length(g),length(z)),(0.,0.05);
            p=(N=amplitude,mu=q[2],sigma=q[3]))
        sol=simulate(pb;method=RK4(),dt=0.001)
        sum(sol.u[end] .* z.widths')
    end
    # Before material reaches the distant outlet, the number inventory is
    # the analytical integral of the linearly time-dependent incoming flux.
    expected=v*sum(fraction)*(0.05+0.05^2/2)
    @test inventory(q[1])≈q[1]*expected atol=2e-13
    @test ForwardDiff.derivative(inventory,q[1])≈expected atol=2e-13
end

@testset "Inlet validation and discretized-feed compatibility" begin
    g=Grid1D([0.,0.2,1.]);z=AxialGrid(0.:0.2:1.)
    build(feed,grid=g)=compile(PBMModel(),grid;
        domain=AxialDispersion(z;velocity=0.3,dispersion=0.01,inlet=feed),axial_scheme=MUSCL())
    N=zeros(length(g),length(z))
    for feed in (nothing,(p,t)->nothing,(p,t)->zeros(length(g)),DensityInlet((x,p,t)->0.))
        @test axial_fluxes(build(feed),N,(;),0.)==zeros(length(g),length(z)+1)
    end
    for bad in (-1.,NaN,Inf,[1.,2.])
        op=build(DensityInlet((x,p,t)->bad))
        @test_throws NumericalFailure axial_fluxes(op,N,(;),0.)
    end
    @test_throws ArgumentError DensityInlet((x,p,t)->1.;quadrature=MidpointRule())
    for bad in ((p,t)->1.,(p,t)->[1.],(p,t)->(x=1.,),(p,t)->ones(1,2))
        @test_throws DimensionMismatch axial_fluxes(build(bad),N,(;),0.)
    end
    for feed in ((p,t)->[2.,3.],(p,t)->reshape([2.,3.],2,1),(p,t)->Any[2.,3.])
        @test axial_fluxes(build(feed),N,(;),0.)[:,1]≈0.3 .* [2.,3.]
    end
    tensor=Grid2D([0.,0.3,1.],[0.,0.2,0.6]);values=[1. 3.;2. 4.]
    for feed in ((p,t)->values,(p,t)->vec(values))
        @test axial_fluxes(build(feed,tensor),zeros(4,length(z)),(;),0.)[:,1]≈0.3 .* vec(values)
    end
    inlet=DensityInlet((x,p,t)->p.A*(1-t))
    op=build(inlet);copy_initial=copy(N)
    @test_throws NumericalFailure axial_fluxes(op,N,(A=2.,),2.)
    @test N==copy_initial
    @test axial_fluxes(op,N,(A=2.,),0.)[:,1]≈0.6 .* g.widths
    @test_throws ArgumentError AxialDispersion(z;velocity=0.3,dispersion=0.,inlet,boundary=:periodic)
    for grid in (g,tensor)
        localop=compile(PBMModel(),grid)
        initial=grid isa Grid1D ? ones(length(grid)) : ones(size(grid))
        localpb=PBMProblem(PBMSystem((A=localop,)),(A=initial,),(0.,1.))
        @test_throws ArgumentError axial_problem(localpb,AxialDispersion(z;velocity=0.3,dispersion=0.,inlet))
        named=AxialDispersion(z;velocity=0.3,dispersion=0.,
            inlet=(p,t)->(populations=(A=inlet,),bulk=(;)))
        pb=axial_problem(localpb,named)
        @test_throws ArgumentError axial_fluxes(pb.operator,pb.u0,(A=2.,),0.)
    end
end
