module KineticParameterUncertainty
using PopulationBalanceModels, Optim, Random

export run_example

"""A seeded synthetic realization with analytical data, two fitted rates, and known Gaussian errors."""
function run_example(;output=joinpath(@__DIR__,"..","docs","src","assets","parameter_estimation"))
    grid=Grid1D([0.0,0.5,1.0])
    model=PBMModel(primary=Nucleation(rate=(p,t,ctx)->p.J,size=0.25),
        loss=Death(rate=(v,p,t,ctx)->p.k))
    operator=compile(model,grid)
    initial=[0.25,0.0]; truth=(J=0.8,k=0.6); sigma=0.03
    problem=PBMProblem(operator,initial,(0.0,4.0);p=truth)
    times=collect(range(0.2,4.0;length=20))
    analytical(t)=sum(initial)*exp(-truth.k*t)+truth.J/truth.k*(-expm1(-truth.k*t))
    data=analytical.(times)+sigma*randn(MersenneTwister(20261020),length(times))
    observation=Observation(times,data,MomentObservable(0);sigma)
    experiments=(Experiment(problem,observation),)
    parameter_map=(z,ex)->(J=exp(z[1]),k=exp(z[2]))
    calibration=CalibrationProblem(experiments;parameters=parameter_map,dt=0.01)
    refined=CalibrationProblem(experiments;parameters=parameter_map,dt=0.005)
    starts=([log(0.4),log(0.3)],[log(1.2),log(1.0)],[log(0.8),log(0.5)])
    multistart=fit_multistart(calibration,starts;maxiters=300)
    fitted=multistart.best
    region=confidence_ellipsoid(calibration,fitted;refined)
    report=region.diagnostics
    z=fitted.minimizer; theta=parameter_map(z,nothing)
    sd=sqrt(report.covariance[1,1])
    profile=profile_likelihood(calibration,fitted,1,range(z[1]-3sd,z[1]+3sd;length=31);maxiters=200)
    all(profile.converged) || error("Some conditional profile fits did not converge")
    mkpath(output)
    predicted=vec(predictions(calibration,z)[1][1])
    open(joinpath(output,"observations.csv"),"w") do io
        println(io,"time,observed,predicted,whitened_residual")
        for row in zip(times,data,predicted,report.residuals);println(io,join(row,','));end
    end
    curve_times=collect(0.0:0.02:4.0)
    curve=simulate(remake(problem;p=theta);dt=0.005,saveat=curve_times)
    open(joinpath(output,"curve.csv"),"w") do io
        println(io,"time,fitted,analytical_truth")
        for (t,u) in zip(curve.t,curve.u);println(io,join((t,sum(u),analytical(t)),','));end
    end
    points=ellipse_points(region)
    open(joinpath(output,"region.csv"),"w") do io
        println(io,"logJ,logk,J,k")
        for v in eachcol(points);println(io,join((v...,exp.(v)...),','));end
    end
    open(joinpath(output,"profile.csv"),"w") do io
        println(io,"logJ,twice_delta_nll,converged")
        for row in zip(profile.values,profile.twice_delta_nll,Int.(profile.converged));println(io,join(row,','));end
    end
    open(joinpath(output,"estimates.csv"),"w") do io
        println(io,"J,k,true_J,true_k")
        println(io,join((theta.J,theta.k,truth.J,truth.k),','))
    end
    open(joinpath(output,"diagnostics.txt"),"w") do io
        println(io,"SYNTHETIC DATA: exact creation/removal solution with independent Gaussian sigma=$sigma; RNG seed=20261020.")
        println(io,"Local 95% joint region is in log(J),log(k), then mapped through exp; not a physical-space ellipse.")
        for key in (:rank,:condition_number,:degrees_of_freedom,:rss,:goodness_of_fit_pvalue,:scaled_gradient,:difference_error,:numerical)
            println(io,key," = ",getproperty(report,key))
        end
        println(io,"Converged starting points: ",count(r->r!==nothing && r.converged,multistart.results),"/",length(starts))
        println(io,"Forward SSPRK dt=0.01, refined=0.005; Julia ",VERSION)
    end
    println("Estimated rates: ",theta)
    println("Diagnostics: ",(;report.rank,report.rss,report.goodness_of_fit_pvalue,report.numerical))
    (;calibration,fitted,region,profile)
end
if abspath(PROGRAM_FILE)==@__FILE__;run_example();end
end
