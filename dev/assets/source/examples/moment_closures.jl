using PopulationBalanceModels

model=PBMModel(
    growth=Growth(PolynomialRate((0.2,0.01))),
    aggregation=Aggregation(ConstantKernel(0.05)))
initial=[1.0,1.0,2.0,6.0] # exponential density; shared raw moments

for (family,basis,u0) in ((ExactClosure(),MomentSet(0:3),initial),
        (QMOM(2),MomentSet(0:3),initial),
        (LogNormalClosure(),MomentSet(0:2),initial[1:3]))
    # Exact growth integration can coexist with a family's default collision rule.
    op=compile(model,basis;method=family,
        methods=TermMethods(growth=ExactMomentRule()))
    solution=simulate(PBMProblem(op,u0,(0.0,0.2));dt=0.002)
    println((family=typeof(family),methods=term_methods(op),mu2=solution.u[end][3]))
end
# The first three equations close for these laws, so this checks shared moment
# bookkeeping. General nonlinear laws require an independent closure study.
