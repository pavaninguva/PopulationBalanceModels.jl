using PopulationBalanceModels

grid=Grid1D(0.5:1.0:40.5;coordinate=:volume)
aggregation=Aggregation((v,w,p,t,ctx)->p.K;overflow=:discard)
breakage=Breakage((v,p,t,ctx)->v>=2 ? p.b : 0,
    (v,p,t,ctx)->((v/2,2.0),);overflow=:discard)
op=compile(PBMModel(aggregation+breakage),grid)
initial=zeros(length(grid)); initial[4]=1
problem=PBMProblem(op,initial,(0.0,1.0);p=(K=0.03,b=0.02))
solution=simulate(problem;dt=0.01,saveat=[0.0,1.0])
println("Initial/final resolved first moments: ",
    (sum(grid.centers.*initial),only(moment(grid,solution.u[end],1))))
println("Final unresolved birth rates: ",diagnostics(op,solution.u[end],problem.p,1.0))
# These are instantaneous loss rates, not cumulative overflow reservoirs.
# To close a time-integrated balance, integrate the rates with consistent
# quadrature, or add explicit reservoirs in a future ledger implementation.
