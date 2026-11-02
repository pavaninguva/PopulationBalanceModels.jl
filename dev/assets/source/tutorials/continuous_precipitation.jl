# Minimal continuous precipitation tutorial.
# Generated from maintained scientific source sections and a short run block.
# No benchmark exports or retained files are written. See the case-study page.

using PopulationBalanceModels

# From examples/continuous_precipitation.jl (model).
"""Case 7A, Zhang et al. (2024), SI Table S8; metres, seconds, kg/m³."""
parameters() = (cfeed=9.,ceq=.4,rho=1200.,kv=.524,kp=5.,u=10.,kb=4e8,b=2.,
    kg=7e-6,xc=1e-6,xmax=900e-6,Q=1.67e-7,area=1.46e-4,porosity=.21,L=.47,Dax=4.2e-5)
supersaturation(p,ctx) = max(ctx.bulk.c/p.ceq-1,zero(ctx.bulk.c))
growth(x,p,t,ctx) = p.kg*supersaturation(p,ctx)

function nucleation(p,t,ctx)
    s = supersaturation(p,ctx)
    solid = 1e12*p.rho*p.kv*moment(ctx.grid,ctx.N,3)
    (p.kp*s^p.u + p.kb*s^p.b*solid)/1e12
end

"""Closed size domain and discrete solute exchange; both phases move axially."""
function build(;nx=32,nz=48,scheme=Upwind(),axial_scheme=scheme,mesh=:log,p=parameters(),tf=200.)
    mesh in (:log,:uniform) || throw(ArgumentError("mesh must be :log or :uniform"))
    edges = mesh === :log ? exp.(range(log(p.xc),log(p.xmax);length=nx+1)) :
        collect(range(p.xc,p.xmax;length=nx+1))
    grid = Grid1D(edges;coordinate=:length)
    model = PBMModel(
        growth=Growth(rate=growth),
        primary_secondary=Nucleation(rate=nucleation,size=first(edges)),
    )
    methods = TermMethods(growth=scheme,
        by_term=(primary_secondary=BoundaryInjection(:left),))
    operator = compile(model,grid;methods)
    exchange = ConservativeExchange(:crystals,:c,(x,p,t,ctx)->1e12*p.rho*p.kv*x^3)
    system = PBMSystem((crystals=operator,);nonnegative_bulk=(:c,),exchanges=(exchange,))
    local_problem = PBMProblem(system,(crystals=zeros(nx),),(0.,tf);bulk0=(c=0.,),p)
    z = AxialGrid(range(0.,p.L;length=nz+1))
    reactor = AxialDispersion(z;velocity=p.Q/(p.area*p.porosity),dispersion=p.Dax,
        inlet=(p,t)->(populations=(crystals=zeros(nx),),bulk=(c=p.cfeed,)))
    axial_problem(local_problem,reactor;axial_scheme)
end

# From examples/continuous_precipitation.jl (controller).
# Native explicit RK with a state-dependent explicit step bound. This controller is
# a benchmark driver, not an adaptive-error solver or a differentiable schedule.
function advance(problem,times;fraction=.3,method=RK4(),progress=false)
    times = collect(times)
    isfinite(fraction) && 0 < fraction <= 1 || throw(ArgumentError("fraction must be in (0,1]"))
    method isa PopulationBalanceModels.AbstractExplicitRK || throw(ArgumentError("advance requires a native explicit Runge–Kutta method"))
    all(isfinite,times) && issorted(times) && all(t->first(problem.tspan)<=t<=last(problem.tspan),times) ||
        throw(ArgumentError("output times must be finite, sorted and inside the problem time span"))
    u = copy(problem.u0)
    t = first(problem.tspan)
    operator = problem.operator
    grid = first(operator.local_operators).system.populations.crystals.grid
    p = problem.p
    dz,dx = minimum(operator.domain.grid.widths),minimum(grid.widths)
    transport = operator.domain.velocity/dz + 2p.Dax/dz^2
    cache = ntuple(_->similar(u),6)
    du = similar(u)
    states = typeof(u)[]
    steps = 0
    for target in times
        while t < target
            rhs!(du,u,operator,p,t)
            maxG = p.kg*max(maximum(bulk(problem,u,:c))/p.ceq-1,0)
            h = fraction/(transport+maxG/dx+eps())
            for j in axes(u,2)
                c,dc = bulk(problem,u,:c)[j],bulk(problem,du,:c)[j]
                dc < 0 && (h = min(h,fraction*max(c-p.ceq,1e-5)/(-dc)))
            end
            h = min(h,target-t,.25)
            h > 1e-12 || throw(NumericalFailure("precipitation step collapsed; inspect the state"))
            PopulationBalanceModels._rk_step!(u,cache,problem,t,h,method,true,1e-8)
            PopulationBalanceModels._check_problem_state(problem,u,1e-8)
            t = h == target-t ? target : t+h
            steps += 1
        end
        push!(states,copy(u))
        if progress
            println("  t = ",target," s; ",steps," steps")
            flush(stdout)
        end
    end
    PBMSolution(times,states,problem,steps)
end

function tutorial_problem(; nx=50, nz=50)
    # Log size bins on [1,900] µm, uniform physical reactor cells.
    # Size and axial reconstruction are deliberately selected separately.
    build(; nx, nz, mesh=:log, scheme=WENO5(), axial_scheme=MUSCL(), tf=5.0)
end

function main()
    problem = tutorial_problem()
    # The controller above bounds the next explicit RK4 step; it is not an
    # adaptive-error solver or an unconditional positivity guarantee for WENO.
    solution = advance(problem, [0.0, 5.0]; method=RK4())
    grid = first(problem.operator.local_operators).system.populations.crystals.grid
    physical_density = 1e12 .* density(grid, population(problem, last(solution.u), :crystals))
    @show extrema(bulk(problem, last(solution.u), :c)) extrema(physical_density)
    solution
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
