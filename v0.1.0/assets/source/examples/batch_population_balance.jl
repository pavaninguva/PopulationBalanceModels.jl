using PopulationBalanceModels

grid=Grid1D(range(0,10;length=101);coordinate=:volume)
model=PBMModel(
    Growth((v,p,t,ctx)->p.growth) +
    Nucleation((p,t,ctx)->p.nucleation,0.2) +
    Death((v,p,t,ctx)->p.death)
)
op=compile(model,grid;growth_scheme=Upwind(),bc=SizeBC(NoFlux(),OpenBoundary()))
initial=exp.(-((grid.centers.-2)./0.4).^2).*grid.widths
problem=PBMProblem(op,initial,(0.0,2.0);p=(growth=0.3,nucleation=0.05,death=0.02))
solution=simulate(problem;dt=0.01,saveat=range(0,2;length=21))
println("Final number concentration: ",only(moment(grid,solution.u[end],0)))
println("Final first moment: ",only(moment(grid,solution.u[end],1)))
println("Explicit step suggestion at t=0: ",suggest_dt(op,problem.u0,problem.p,0.0))
