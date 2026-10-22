module ContinuousPrecipitation
using PopulationBalanceModels
export parameters,build,run_benchmark

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

# Native SSPRK33 with a state-dependent explicit step bound. This controller is
# a benchmark driver, not an adaptive-error solver or a differentiable schedule.
function advance(problem,times;fraction=.3)
    times = collect(times)
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
            PopulationBalanceModels._rk_step!(u,cache,problem,t,h,SSPRK33(),true,1e-8)
            PopulationBalanceModels._check_problem_state(problem,u,1e-8)
            t = h == target-t ? target : t+h
            steps += 1
        end
        push!(states,copy(u))
    end
    PBMSolution(times,states,problem,steps)
end

"""Normalized outlet L1 difference, integrating both densities on their union grid."""
function outlet_error(problem,u,reference,v)
    grid = first(problem.operator.local_operators).system.populations.crystals.grid
    refgrid = first(reference.operator.local_operators).system.populations.crystals.grid
    edges = sort!(unique(vcat(grid.edges,refgrid.edges)))
    a = population(problem,u,:crystals)[:,end] ./ grid.widths
    b = population(reference,v,:crystals)[:,end] ./ refgrid.widths
    total,normalization = 0.,0.
    for i in 1:length(edges)-1
        x = (edges[i]+edges[i+1])/2
        da = a[clamp(searchsortedlast(grid.edges,x),1,length(grid))]
        db = b[clamp(searchsortedlast(refgrid.edges,x),1,length(refgrid))]
        total += abs(da-db)*(edges[i+1]-edges[i])
        normalization += db*(edges[i+1]-edges[i])
    end
    normalization > 0 || throw(NumericalFailure("reference outlet number must be positive"))
    total/normalization
end

include("support/precipitation_benchmark.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmark(full="--full" in ARGS)
end
end
