# Minimal crystallization tutorial.
# Physical definitions and one calculation; numerical helpers are linked on the case page.
# No benchmark exports or retained files are written. See the case-study page.

using PopulationBalanceModels
include(joinpath(pkgdir(PopulationBalanceModels), "examples", "papers", "gunawan2004_case2.jl"))
using .Gunawan2004Case2: initial_counts

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

function tutorial_problem()
    p = parameters() # µm, seconds, g solute/g water; rho is g/µm³.
    edges = sort!(unique(vcat(collect(0.0:2.0:1600.0), 180.5, 210.5)))
    grid = Grid1D(edges; coordinate=:length)
    model = PBMModel(
        growth=Growth(growth_rate),
        nucleation=Nucleation(nucleation_rate, 0.0),
    )
    methods = TermMethods(growth=MUSCL(),
        by_term=(nucleation=BoundaryInjection(:left),))
    operator = compile(model, grid; methods, bc=SizeBC())
    exchange = ConservativeExchange(:crystals, :c, (L,p,t,ctx) -> p.rho*L^3)
    system = PBMSystem((crystals=operator,);
        fields=(p,t) -> (T=temperature(t),), nonnegative_bulk=(:c,),
        exchanges=(exchange,))
    initial = initial_counts(grid, benchmark_initial_density)
    PBMProblem(system, (crystals=initial,), (0.0, 10.0);
        bulk0=(c=solubility(temperature(0.0)),), p)
end

function main()
    problem = tutorial_problem()
    solution = simulate(problem; method=SSPRK33(), dt=0.05, saveat=[0.0, 5.0, 10.0])
    @show bulk(problem, last(solution.u), :c)
    @show observe(PopulationMoment(:crystals, 3), solution)
    solution
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
