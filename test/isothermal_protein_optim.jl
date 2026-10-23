using Test, PopulationBalanceModels, Optim
@testset "Nonstatistical discrepancy fit and legacy likelihood separation" begin
    op=compile(PBMModel(),Grid1D([0.,.5,1.]))
    pb=PBMProblem(op,[0.,0.],(0.,1.);p=(a=0.,))
    obs=ResidualObservation([0.,1.],[3.,3.],FunctionObservable((N,op,p,t)->p.a);scale=.5)
    cal=CalibrationProblem((Experiment(pb,obs),);parameters=(z,e)->(a=z[1],),dt=.1)
    fitted=fit(cal,[1.])
    @test fitted.converged
    @test only(fitted.minimizer)≈3. atol=1e-7
    @test fitted.minimum<1e-12
    @test sensitivity_report(cal,[3.]).covariance===nothing
    @test_throws ArgumentError profile_likelihood(cal,fitted,1,[2.,3.,4.])
    gaussian=CalibrationProblem((Experiment(pb,Observation([0.],[3.],
        FunctionObservable((N,op,p,t)->p.a);sigma=.5)),);
        parameters=(z,e)->(a=z[1],),dt=.1)
    @test objective(gaussian,[3.])≈log(.5sqrt(2pi))
end
