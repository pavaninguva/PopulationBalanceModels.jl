using ForwardDiff

@testset "Nonuniform finite-volume WENO5" begin
    P = PopulationBalanceModels
    periodic = SizeBC(Periodic(),Periodic())
    open = SizeBC(OpenBoundary(),OpenBoundary())
    @testset "Independent polynomial reconstruction and uniform limit" begin
        grid = Grid1D([0.,.07,.18,.34,.58,.9,1.3,1.8,2.4,3.1])
        cache = prepare_term(Growth((x,p,t,c)->1.),WENO5(),grid;bc=open)
        for direction in (-1,1), face in 4:length(grid)-2
            stencil = (direction==1 ? cache.positive : cache.negative)[face]
            # Nonlinear weights must reconstruct every quadratic exactly.
            numbers = diff(grid.edges .+ grid.edges.^2 ./ 2 .+ grid.edges.^3 ./ 3)
            actual = P._face_density(numbers./grid.widths,face,cache,direction,false,grid)
            x = grid.edges[face]
            @test actual ≈ 1+x+x^2 rtol=3e-14
            # The optimal combination must reproduce quartic bin integrals.
            numbers = diff(grid.edges.^5)./5
            averages = numbers./grid.widths
            linear = sum(stencil.weights[k]*sum(stencil.polynomials[k][1][col]*
                averages[stencil.indices[k+col-1]] for col in 1:3) for k in 1:3)
            @test linear ≈ x^4 rtol=3e-13
        end
        uniform = Grid1D(range(0,1;length=17))
        values = [2+sin(1.4i) for i in 1:16]
        for direction in (-1,1), face in (1,7,17)
            stencil = P._weno_stencil(uniform,face,direction,true,Float64)
            @test P._weno_value(values,stencil,1e-6) ≈
                P._face_density(values,face,WENO5(),direction,true) rtol=5e-14
        end
        # Geometry depends on ratios, not on the chosen length unit.
        scaled = Grid1D(grid.edges.*1e-9)
        other = prepare_term(Growth((x,p,t,c)->1.),WENO5(),scaled;bc=open)
        for direction in (-1,1), face in 4:length(grid)-2
            a = P._face_density(values[1:length(grid)],face,cache,direction,false,grid)
            b = P._face_density(values[1:length(grid)],face,other,direction,false,scaled)
            @test a ≈ b rtol=1e-13
        end
        @test prepare_term(Growth((x,p,t,c)->1.),WENO5(),uniform) isa WENO5
        @test_throws ArgumentError compile(PBMModel(Growth((x,p,t,c)->1.)),Grid1D(0:5);growth_scheme=WENO5())
        unresolved = Grid1D([0.,1e-12,2e-12,3e-12,4e-12,5e-12,1.])
        @test_throws ArgumentError compile(PBMModel(Growth((x,p,t,c)->1.)),unresolved;growth_scheme=WENO5())
        # A change of length unit scales velocity and density, and epsilon
        # must scale as density squared. Bin numbers and their rates agree.
        numbers = reshape(grid.widths.*(2 .+sin.(grid.centers)),:,1)
        derivatives = Matrix{Float64}[]
        for factor in (1.,1e-9)
            axis = Grid1D(grid.edges.*factor)
            op = compile(PBMModel(Growth((x,p,t,c)->.7factor)),axis;
                growth_scheme=WENO5(epsilon=1e-6/factor^2),bc=open)
            derivative = zero(numbers)
            rhs!(derivative,numbers,op,(;),0.)
            push!(derivatives,derivative)
        end
        @test derivatives[1] ≈ derivatives[2] rtol=2e-12
    end

    @testset "Periodic conservative residual and spatial order" begin
        for velocity in (-.7,.7)
            errors = Float64[]
            for bins in (32,64,128)
                s = range(0,2pi;length=bins+1)
                grid = Grid1D(s .+ .3sin.(s))
                numbers = reshape(2grid.widths.+cos.(grid.edges[1:end-1]).-cos.(grid.edges[2:end]),:,1)
                op = compile(PBMModel(Growth((x,p,t,c)->velocity)),grid;
                    methods=TermMethods(growth=WENO5(epsilon=1e-6)),bc=periodic)
                derivative = zero(numbers)
                rhs!(derivative,numbers,op,(;),0.)
                exact = velocity.*(sin.(grid.edges[1:end-1]).-sin.(grid.edges[2:end]))
                push!(errors,sum(abs.(vec(derivative).-exact)))
                @test abs(sum(derivative)) < 1e-13
            end
            @test log2(errors[2]/errors[3]) > 4.5
        end
    end

    @testset "Logarithmic mesh, affine growth and physical fluxes" begin
        errors = Float64[]
        for bins in (40,80,160)
            grid = Grid1D(exp.(range(0,2;length=bins+1));coordinate=:length)
            # n(x)=2+sin(log(x)); exact integral, with G(x)=x.
            primitive(x) = 2x+x/2*(sin(log(x))-cos(log(x)))
            numbers = reshape(diff(primitive.(grid.edges)),:,1)
            op = compile(PBMModel(Growth((x,p,t,c)->x)),grid;growth_scheme=WENO5(),bc=open)
            derivative = zero(numbers)
            rhs!(derivative,numbers,op,(;),0.)
            exact = -diff(grid.edges.*(2 .+sin.(log.(grid.edges))))
            interior = 4:bins-3 # Neither bounding face uses boundary fallback.
            push!(errors,sum(abs.(vec(derivative)[interior].-exact[interior])))
            # Open boundary outflow uses the actual last-bin average.
            @test sum(derivative) ≈ -last(grid.edges)*numbers[end]/last(grid.widths) rtol=1e-13
        end
        @test log2(errors[2]/errors[3]) > 4.5
        grid = Grid1D(exp.(range(0,2;length=25)))
        for velocity in (-1.,1.)
            bc = velocity>0 ? SizeBC(InflowFlux((p,t,c)->3.),OpenBoundary()) :
                SizeBC(OpenBoundary(),InflowFlux((p,t,c)->3.))
            op = compile(PBMModel(Growth((x,p,t,c)->velocity)),grid;growth_scheme=WENO5(),bc)
            numbers = reshape(2grid.widths,:,1)
            derivative = zero(numbers)
            rhs!(derivative,numbers,op,(;),0.)
            @test sum(derivative) ≈ 1. atol=2e-13
            @test term_methods(op).growth isa WENO5
            result = simulate(PBMProblem(op,numbers,(0.,.01));method=RK4(),dt=.001)
            @test sum(result.u[end]) ≈ sum(numbers)+.01 atol=1e-12
        end
    end

    @testset "Tensor axes and axial inventories" begin
        x = Grid1D(exp.(range(0,1;length=25)))
        y = Grid1D(range(0,2;length=19))
        grid = Grid2D(x,y)
        for velocity in ((.7,-.2),(-.7,.2))
            op = compile(PBMModel(Growth2D(velocity)),grid;growth_scheme=WENO5(),
                bc=SizeBC2D(x=periodic,y=periodic))
            numbers = (x.widths*y.widths').*(2 .+ .1sin.(x.centers).*cos.(y.centers'))
            derivative = zero(numbers)
            rhs!(derivative,numbers,op,(;),0.)
            @test abs(sum(derivative)) < 1e-13
            @test term_methods(op).growth isa WENO5
        end
        errors = Float64[]
        for bins in (24,48,96)
            sx = range(0,2pi;length=bins+1)
            sy = range(0,3pi;length=3bins÷4+1)
            g = Grid2D(Grid1D(sx .+.2sin.(sx)),Grid1D(sy .+.2sin.(2sy./3)))
            xe, ye = g.x.edges, g.y.edges
            a = 2g.x.widths.+cos.(xe[1:end-1]).-cos.(xe[2:end])
            b = g.y.widths .+ 0.3 .* (sin.(2ye[2:end]./3).-sin.(2ye[1:end-1]./3))
            da = 0.7 .* (sin.(xe[1:end-1]).-sin.(xe[2:end]))
            db = -0.2*0.2 .* (cos.(2ye[1:end-1]./3).-cos.(2ye[2:end]./3))
            op = compile(PBMModel(Growth2D((.7,-.2))),g;growth_scheme=WENO5(),
                bc=SizeBC2D(x=periodic,y=periodic))
            state = a*b'
            derivative = zero(state)
            rhs!(derivative,state,op,(;),0.)
            push!(errors,sum(abs.(derivative.-(da*b'+a*db'))))
            @test abs(sum(derivative)) < 2e-13
        end
        @test log2(errors[2]/errors[3]) > 4.3
        # The axial adapter transports cell averages, with volume-weighted
        # inventory. Diffusion remains the existing centered approximation.
        localop = compile(PBMModel(),Grid1D([0.,1.,2.]))
        localpb = PBMProblem(PBMSystem((particles=localop,)),(particles=[1.,2.],),(0.,.01))
        z = AxialGrid(exp.(range(0,1;length=25)).-1)
        for boundary in (:periodic,:danckwerts)
            domain = AxialDispersion(z;velocity=.3,dispersion=.02,boundary,
                inlet=boundary===:danckwerts ? (p,t)->[1.,2.] : nothing)
            pb = axial_problem(localpb,domain;axial_scheme=WENO5())
            state = pb.u0.*(1 .+.1sin.(2pi*z.centers/last(z.edges)))'
            derivative = zero(state)
            rhs!(derivative,state,pb.operator,(;),0.)
            flux = axial_fluxes(pb.operator,state,(;),0.)
            @test vec(sum(derivative.*z.widths';dims=2)) ≈ flux[:,1]-flux[:,end] atol=2e-13
            @test pb.operator.scheme isa WENO5
        end
    end

    @testset "Shared solute and cumulative escape" begin
        grid = Grid1D(exp.(range(0,1;length=25));coordinate=:length)
        for velocity in (-.2,.2)
            op = compile(PBMModel(Growth((x,p,t,c)->velocity*c.bulk.c)),grid;
                growth_scheme=WENO5(),bc=open)
            system = PBMSystem((particles=op,);nonnegative_bulk=(:c,),
                exchanges=(ConservativeExchange(:particles,:c,(x,p,t,c)->x^3),),
                accounting=(CumulativeAccounting(:particles),))
            pb = PBMProblem(system,(particles=grid.widths,),(.0,.01);bulk0=(c=10.,))
            total(u) = bulk(pb,u,:c)+sum(grid.centers.^3 .* population(pb,u,:particles))+
                cumulative(pb,u,:particles,:upper_escape,:material)+
                cumulative(pb,u,:particles,:lower_escape,:material)
            result = simulate(pb;method=RK4(),dt=.0001)
            @test total(result.u[end]) ≈ total(pb.u0) rtol=5e-14
            side = velocity>0 ? :upper_escape : :lower_escape
            @test cumulative(pb,result.u[end],:particles,side)>0
        end
    end

    @testset "Resolved characteristic evolution on a log mesh" begin
        include(joinpath(@__DIR__,"..","examples","nonuniform_weno.jl"))
        errors = Float64[]
        for bins in (80,160,320)
            pb = NonuniformWENOExample.problem(bins)
            dt = .1minimum(pb.operator.grid.widths./pb.operator.grid.edges[2:end])
            sol = simulate(pb;method=RK4(),dt,check_nonnegative=false)
            exact = NonuniformWENOExample.reference(pb.operator.grid,last(pb.tspan))
            push!(errors,sum(abs.(vec(sol.u[end]).-exact)))
            @test abs(sum(sol.u[end])-sum(pb.u0)) < 2e-12
        end
        @test log2(errors[2]/errors[3]) > 4.3
    end

    @testset "AD through cached reconstruction" begin
        grid = Grid1D(exp.(range(0,1;length=25)))
        op = compile(PBMModel(Growth((x,p,t,c)->p.k)),grid;growth_scheme=WENO5())
        numbers = reshape(grid.widths.*(1 .+sin.(grid.centers)),:,1)
        function prediction(k)
            state = k.*numbers
            derivative = zero(state)
            rhs!(derivative,state,op,(k=k,),0.)
            sum(abs2,derivative)
        end
        actual = ForwardDiff.derivative(prediction,.7)
        reference = (prediction(.700001)-prediction(.699999))/2e-6
        @test actual ≈ reference rtol=2e-8
    end

    @testset "Undershoots remain visible" begin
        grid = Grid1D(exp.(range(0,1;length=33)))
        initial = zeros(32)
        initial[12:18] .= grid.widths[12:18]
        op = compile(PBMModel(Growth((x,p,t,c)->1.)),grid;
            growth_scheme=WENO5(),bc=periodic)
        pb = PBMProblem(op,initial,(0.,.1))
        result = simulate(pb;method=RK4(),dt=.002,check_nonnegative=false)
        @test minimum(result.u[end]) < -1e-8
        @test sum(result.u[end]) ≈ sum(initial) atol=2e-14
        @test_throws NumericalFailure simulate(pb;method=RK4(),dt=.002,negative_tolerance=0.)
    end
end
