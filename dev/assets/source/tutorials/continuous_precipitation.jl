# Minimal continuous precipitation tutorial.
# Physical definitions and one calculation; numerical helpers are linked on the case page.
# No benchmark exports or retained files are written. See the case-study page.

using PopulationBalanceModels
# Existing RK4 controller: rate/depletion bounds and stage checks.
include(joinpath(pkgdir(PopulationBalanceModels), "examples", "continuous_precipitation.jl"))
using .ContinuousPrecipitation: advance

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

function tutorial_problem(; nx=50, nz=50)
    # Log size bins on [1,900] µm, uniform physical reactor cells.
    # Size and axial reconstruction are deliberately selected separately.
    build(; nx, nz, mesh=:log, scheme=WENO5(), axial_scheme=MUSCL(), tf=5.0)
end

function main()
    nz = 50
    problem = tutorial_problem(; nz)
    # The imported controller bounds each explicit RK4 step; it is not an
    # adaptive-error solver or an unconditional positivity guarantee for WENO.
    solution = advance(problem, [0.0, 5.0]; method=RK4())
    @show extrema(bulk(problem, last(solution.u), :c))
    # Restore the population scaling when reporting particles/m³.
    outlet = CellObservable(PopulationMoment(:crystals, 0), nz)
    @show 1e12 .* observe(outlet, solution)
    solution
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
