# Minimal crystallization tutorial.
# Physical definitions and one calculation; numerical helpers are linked on the case page.
# No benchmark exports or retained files are written. See the case-study page.

using PopulationBalanceModels

# Physical definitions from examples/papers/gunawan2004_case2.jl.
parameters(;kb=4.64e-7,b=1.78,kg=116.0,g=1.32,rho=2.11e-12,size_factor=0.1)=
    (;kb,b,kg,g,rho,size_factor)

temperature(t)=32.0-4.0*(-expm1(-t/18600.0))

solubility(T)=1.721e-4*T^2-5.88e-3*T+0.1286

benchmark_initial_density(L)=180.5<=L<=210.5 ? 0.08*(1-((L-195.5)/15)^2) : 0.0

supersaturation(ctx)=max(ctx.bulk.c/solubility(ctx.fields.T)-1,0.0)

growth_scale(p,t,ctx)=p.kg*supersaturation(ctx)^p.g

growth_rate(L,p,t,ctx)=growth_scale(p,t,ctx)*(1+p.size_factor*L)

nucleation_rate(p,t,ctx)=p.kb*moment(ctx,3)*supersaturation(ctx)^p.b

# Physical definitions from examples/papers/gunawan2004_case2.jl.
function assemble_problem(grid,initial;c0,scheme,tspan,p)
    first(grid.edges)==0 || throw(ArgumentError("Gunawan boundary nucleation requires a grid starting at L=0"))
    model=PBMModel(growth=Growth(rate=growth_rate),nucleation=Nucleation(rate=nucleation_rate,size=0.0))
    methods=TermMethods(growth=scheme,by_term=(nucleation=BoundaryInjection(:left),))
    op=compile(model,grid;methods,bc=SizeBC(NoFlux(),NoFlux()))
    system=PBMSystem((crystals=op,);
        fields=(p,t)->(T=temperature(t),),nonnegative_bulk=(:c,),
        exchanges=(ConservativeExchange(:crystals,:c,(L,p,t,ctx)->p.rho*L^3;
            mechanisms=(:growth,:nucleation)),))
    PBMProblem(system,(crystals=initial,),tspan;bulk0=(c=float(c0),),p)
end

function density_problem(;grid,initial_density,c0,support=nothing,scheme=MUSCL(),
                         tspan=(0.0,1000.0),p=parameters())
    initial=InitialDensity((L,p,t)->initial_density(L);support,quadrature=GaussLegendre(3))
    assemble_problem(grid,initial;c0,scheme,tspan,p)
end

function tutorial_problem()
    p = parameters() # µm, seconds, g solute/g water; rho is g/µm³.
    edges = sort!(unique(vcat(collect(0.0:2.0:1600.0), 180.5, 210.5)))
    grid = Grid1D(edges; coordinate=:length)
    # The compact support breakpoints are grid edges; the density is not normalized.
    density_problem(;grid,initial_density=benchmark_initial_density,support=(180.5,210.5),
        c0=solubility(temperature(0.0)),tspan=(0.0,10.0),p)
end

function main(; times=[0.0,5.0,10.0])
    problem = tutorial_problem()
    solver = ForwardSolver(SSPRK33(); dt=0.05)
    solution = simulate(problem; solver, saveat=times)
    @show bulk(problem, last(solution.u), :c)
    @show observe(PopulationMoment(:crystals, 3), solution)
    solution
end

# Optional after import PythonPlot; reuse the returned solution.
function plot_results(solution)
    distribution = plot_distribution(solution; population=:crystals,
        xlabel="Length (micrometres)", ylabel="Number density")
    concentration = plot_observable(solution, BulkObservable(:c);
        xlabel="Time (s)", ylabel="Solute mass / water mass")
    (; distribution, concentration)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
