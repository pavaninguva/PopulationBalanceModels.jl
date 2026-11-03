# Cooling batch: micrometres, seconds, g solute/g water; rho in g/micrometre^3.
using PopulationBalanceModels

p = (kb=4.64e-7,b=1.78,kg=116.0,g=1.32,rho=2.11e-12,size_factor=0.1)
temperature(t) = 32.0-4.0*(-expm1(-t/18600.0))
solubility(T) = 1.721e-4*T^2-5.88e-3*T+0.1286
supersaturation(ctx) = max(ctx.bulk.c/solubility(ctx.fields.T)-1,0.0)
growth(L,p,t,ctx) = p.kg*supersaturation(ctx)^p.g*(1+p.size_factor*L)
nucleation(p,t,ctx) = p.kb*moment(ctx,3)*supersaturation(ctx)^p.b
seed_density(L,p,t) = 180.5<=L<=210.5 ? 0.08*(1-((L-195.5)/15)^2) : 0.0

model = PBMModel(growth=Growth(growth),nucleation=Nucleation(nucleation,0.0))
# Align the compact seed support with edges. This nonnegative benchmark seed
# is an explicit alternative to the paper's printed negative polynomial.
edges = sort!(unique(vcat(collect(0.0:2.0:1600.0),180.5,210.5)))
grid = Grid1D(edges;coordinate=:length)
methods = TermMethods(growth=MUSCL(),nucleation=BoundaryInjection(:left))
operator = compile(model,grid;methods,bc=SizeBC(NoFlux(),NoFlux()))
exchange = ConservativeExchange(:crystals,:c,(L,p,t,ctx)->p.rho*L^3;
    mechanisms=(:growth,:nucleation))
system = PBMSystem((crystals=operator,);exchanges=(exchange,),
    fields=(p,t)->(T=temperature(t),),nonnegative_bulk=(:c,))
initial = InitialDensity(seed_density;support=(180.5,210.5),quadrature=GaussLegendre(3))
problem = PBMProblem(system,(crystals=initial,),(0.0,10.0);
    bulk0=(c=solubility(temperature(0.0)),),p)

if abspath(PROGRAM_FILE) == @__FILE__
    solution = simulate(problem;method=SSPRK33(),dt=0.05,saveat=[0.0,5.0,10.0])
    @show observe(BulkObservable(:c),solution)
    @show observe(PopulationMoment(:crystals,3),solution)
    # Optional: import PythonPlot
    # plot_distribution(solution;population=:crystals,xlabel="Length (micrometres)")
    # plot_observable(solution,BulkObservable(:c);ylabel="Solute mass / water mass")
end
