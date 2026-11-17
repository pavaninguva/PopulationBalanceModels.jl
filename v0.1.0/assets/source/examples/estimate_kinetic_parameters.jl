using PopulationBalanceModels, Optim

# Install Optim in the active working environment to enable PopulationBalanceModels's fit extension.
# Data are analytical, independent of the implementation used for fitting.
grid=Grid1D(range(0,8;length=41))
model=PBMModel(Death((v,p,t,ctx)->p.k))
op=compile(model,grid)
initial=exp.(-grid.centers).*grid.widths
times=[0.0,0.2,0.6,1.2,2.0]
experiments=Experiment[]
for scale in (1.0,2.0)
    problem=PBMProblem(op,scale*initial,(0.0,2.0);p=(k=0.4,))
    data=scale*sum(initial).*exp.(-0.4.*times)
    observation=Observation(times,data,MomentObservable(0);sigma=0.02)
    push!(experiments,Experiment(problem,observation))
end
calibration=CalibrationProblem(experiments;
    parameters=(z,experiment)->(k=exp(z[1]),),dt=0.01)
result=fit(calibration,[log(0.15)];maxiters=100)
report=sensitivity_report(calibration,result.minimizer)
println("Estimated k: ",exp(result.minimizer[1]))
println("Optimizer converged: ",result.converged)
println("Local residual-Jacobian rank: ",report.rank)
println("Local covariance is in log(k) coordinates: ",report.covariance)
