using PopulationBalanceModels

# Physical laws are named separately from all algorithm choices.
grid=Grid1D(0:0.25:10;coordinate=:volume)
model=PBMModel(
    growth=Growth((v,p,t,ctx)->p.G),
    primary=Nucleation((p,t,ctx)->p.J,0.0),
    secondary=Nucleation((p,t,ctx)->p.J/2,0.8),
    aggregation=Aggregation(ConstantKernel((p,t,ctx)->p.K)),
    breakage=Breakage((v,p,t,ctx)->v>=0.5 ? 0.01 : 0.0,
        SelfSimilarDaughters((0.5,),(2.0,))),
    loss=Death((v,p,t,ctx)->0.0001v^2))

numerics=TermMethods(
    growth=MUSCL(),
    aggregation=FixedPivot(overflow=:discard),
    breakage=FixedPivot(),
    death=CellAverageLoss(quadrature=GaussLegendre(4)),
    by_term=(
        primary=BoundaryInjection(:left),
        secondary=GaussianInjection(width=0.05,truncation=:renormalize)))

op=compile(model,grid;methods=numerics)
problem=PBMProblem(op,zeros(length(grid)),(0.0,0.2);p=(G=0.2,J=1.0,K=0.01))
solution=simulate(problem;dt=0.002)
println(term_methods(op))
println((number=sum(solution.u[end]),
    overflow=diagnostics(op,solution.u[end],problem.p,0.2).unresolved_birth_number_rate))
# The example explicitly discards aggregation products outside the grid.
# Enlarge the grid/use an overflow reservoir for a closed physical balance.
