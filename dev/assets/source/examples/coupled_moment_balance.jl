using PopulationBalanceModels

moments=MomentSet(0:3;coordinate=:volume)
growth=Growth(PolynomialRate(((p,t,ctx)->p.g*ctx.bulk.c,)))
birth=Nucleation((p,t,ctx)->p.J,0.1)
aggregation=Aggregation(ConstantKernel((p,t,ctx)->p.K))
fragmentation=Breakage(PolynomialRate((0.03,)),SelfSimilarDaughters((0.5,),(2.0,)))
model=PBMModel(growth+birth+aggregation+fragmentation)
op=compile(model,moments;closure=ExactClosure())
system=PBMSystem((particles=op,);
    exchanges=(MomentExchange(:particles,:c,1),),nonnegative_bulk=(:c,))
μ0=[1.0,1.0,2.0,6.0] # unit exponential density on the positive volume axis
problem=PBMProblem(system,(particles=μ0,),(0.0,1.0);
    bulk0=(c=2.0,),p=(g=0.1,J=0.2,K=0.1))
solution=simulate(problem;dt=0.002,saveat=0:0.1:1)
final=stateview(problem,solution.u[end])
println((moments=final.populations.particles,solute=final.bulk.c,
    conserved=final.bulk.c+moment(moments,final.populations.particles,1)))

# Non-polynomial laws require an explicit closure. QMOM approximates their
# weak-form integrals using a positive quadrature reconstruction at each stage.
qop=compile(PBMModel(Growth((x,p,t,ctx)->0.1*sqrt(x))),moments;
    closure=QuadratureClosure(2))
qproblem=PBMProblem(qop,μ0,(0.0,0.1))
qsolution=simulate(qproblem;dt=0.001)
println((quadrature=quadrature(moments,qsolution.u[end]),))

# MomentObservable / PopulationMoment, Experiment and CalibrationProblem use
# the same inference interfaces as distribution models. Histograms require an
# explicit distribution reconstruction/measurement model.
