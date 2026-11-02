# Minimal crystallization tutorial.
# Generated from maintained scientific source sections and a short run block.
# No benchmark exports or retained files are written. See the case-study page.

using PopulationBalanceModels

# From examples/papers/gunawan2004_case2.jl (physics).
"""Kinetic constants in µm, seconds, and g/g-water; size_factor has units µm⁻¹."""
parameters(;kb=4.64e-7,b=1.78,kg=116.0,g=1.32,rho=2.11e-12,size_factor=0.1)=
    (;kb,b,kg,g,rho,size_factor)

temperature(t)=32.0-4.0*(-expm1(-t/18600.0))
solubility(T)=1.721e-4*T^2-5.88e-3*T+0.1286
reported_initial_density(L)=180.5<=L<=210.5 ? -3.48e-4*L^2+0.136*L-13.3 : 0.0
# Retained for existing user scripts; the function denotes a printed initial condition.
reported_seed_density(L)=reported_initial_density(L)
"""Chosen nonnegative initial density for our numerical benchmark, not corrected paper coefficients."""
benchmark_initial_density(L)=180.5<=L<=210.5 ? 0.08*(1-((L-195.5)/15)^2) : 0.0
supersaturation(ctx)=max(ctx.bulk.c/solubility(ctx.fields.T)-1,0.0)
growth_scale(p,t,ctx)=p.kg*supersaturation(ctx)^p.g
growth_rate(L,p,t,ctx)=growth_scale(p,t,ctx)*(1+p.size_factor*L)
nucleation_rate(p,t,ctx)=p.kb*moment(ctx,3)*supersaturation(ctx)^p.b

"""Three-point cell quadrature; align any discontinuities with grid edges."""
function initial_counts(grid,f)
    a=sqrt(3/5)
    [width/2*(5/9*f(x-width*a/2)+8/9*f(x)+5/9*f(x+width*a/2))
        for (x,width) in zip(grid.centers,grid.widths)]
end

# From examples/papers/gunawan2004_case2.jl (assembly).
"""
Gunawan et al. 2004, Example 2, Eqs. 28-34 and Table 3.
initial_density and c0 are explicit inputs. Eq. 34 is retained verbatim in
reported_initial_density; its displayed coefficients evaluate to negative values
on the stated support. The separate benchmark_initial_density is an explicit
alternative initial condition, not a repair of the printed coefficients.

Units: length in micrometers, time in seconds, concentration in the paper's
g/g-water basis, crystal density in g/micrometer^3. The state basis and shape
factor follow the paper's V=integral(L^3*n) convention.
"""
function build(;grid=Grid1D(range(0.0,1600.0;length=1601);coordinate=:length),
                initial_density,c0,scheme=MUSCL(),tspan=(0.0,1000.0),p=parameters())
    first(grid.edges)==0 || throw(ArgumentError("Gunawan boundary nucleation requires a grid starting at L=0"))
    model=PBMModel(growth=Growth(rate=growth_rate),nucleation=Nucleation(rate=nucleation_rate,size=0.0))
    methods=TermMethods(growth=scheme,by_term=(nucleation=BoundaryInjection(:left),))
    op=compile(model,grid;methods,bc=SizeBC(NoFlux(),NoFlux()))
    system=PBMSystem((crystals=op,);
        fields=(p,t)->(T=temperature(t),),nonnegative_bulk=(:c,),
        exchanges=(ConservativeExchange(:crystals,:c,(L,p,t,ctx)->p.rho*L^3;
            mechanisms=(:growth,:nucleation)),))
    N0=initial_counts(grid,initial_density)
    PBMProblem(system,(crystals=N0,),tspan;bulk0=(c=float(c0),),p)
end

function tutorial_problem()
    p = parameters() # µm, seconds, g solute/g water; rho is g/µm³.
    edges = sort!(unique(vcat(collect(0.0:2.0:1600.0), 180.5, 210.5)))
    grid = Grid1D(edges; coordinate=:length)
    c0 = solubility(temperature(0.0))
    # Explicit nonnegative benchmark IC, not the negative printed polynomial.
    build(; grid, initial_density=benchmark_initial_density, c0, p,
        scheme=MUSCL(), tspan=(0.0, 10.0))
end

function main()
    problem = tutorial_problem()
    solution = simulate(problem; method=SSPRK33(), dt=0.05, saveat=[0.0, 5.0, 10.0])
    grid = problem.operator.system.populations.crystals.grid
    total(u) = bulk(problem, u, :c) +
        problem.p.rho * moment(grid, population(problem, u, :crystals), 3)
    @show bulk(problem, last(solution.u), :c)
    @show maximum(abs(total(u)-total(problem.u0)) for u in solution.u)
    solution
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
