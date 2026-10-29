module PCERefinementExample

using PopulationBalanceModels
import Distributions

"""Independent beta-removal reference: N(t)=2 exp(-k t), k~Beta(2,2)."""
function run_example()
    grid = Grid1D([0.,0.3,1.])
    model = PBMModel(removal=Death((x,p,t,ctx) -> p.k))
    base = PBMProblem(compile(model,grid),2 .* grid.widths,(0.,1.);p=(k=0.5,))
    inputs = UncertainInputs((k=Distributions.Beta(2,2),))
    problem = UQProblem(base,inputs;
        outputs=(number=MomentObservable(0),),times=[0.,0.5,1.],
        dt=0.02,method=RK4())

    # One full-model validation set is reused for selecting the two surrogates.
    # The later test set remains independent of this selection.
    validation = propagate(problem,MonteCarlo(128;seed=201);max_solves=128)
    sizes = (12,18,24,32)
    regression = refine_pce(problem,PolynomialChaos.((1,2,3,4));
        designs=[pce_design(inputs;n,seed=200+n) for n in sizes],
        validation,max_solves=sum(sizes),validation_atol=1e-8,validation_rtol=0.01,
        mean_atol=1e-8,mean_rtol=1e-4,variance_atol=1e-8,variance_rtol=0.01)

    projection = refine_pce(problem,
        [PolynomialChaos(p;construction=:projection) for p in 1:4];
        designs=[pce_quadrature(inputs;order=q) for q in 3:6],
        validation,max_solves=18,validation_atol=1e-8,validation_rtol=0.01,
        mean_atol=1e-8,mean_rtol=1e-4,variance_atol=1e-8,variance_rtol=0.01)
    regression.converged && projection.converged ||
        error("PCE selection criteria were not met; inspect the retained candidates/history")

    test = propagate(problem,MonteCarlo(64;seed=202);max_solves=64)
    checks = map((regression,projection)) do study
        validate_pce(study.surrogate,test;atol=1e-8,rtol=0.01)
    end
    all(check -> check.passed,checks) || error("independent final validation failed")

    # Cheap surrogate samples provide quantiles; they do not run the PBM again.
    samples = evaluate_pce(projection.surrogate,MonteCarlo(10_000;seed=203))
    quantiles = uq_summary(samples).quantiles.number
    E(s) = 6*((s-2)+(s+2)*exp(-s))/s^3
    exact_mean, exact_variance = 2E(1.), 4*(E(2.)-E(1.)^2)
    println("Analytical final mean/variance: ",(exact_mean,exact_variance))
    for (name,study) in zip(("SVD regression","Gaussian projection"),(regression,projection))
        summary = uq_summary(study.surrogate)
        println(name,": degree=",study.surrogate.method.degree,
            ", mean=",summary.mean.number[1,end],
            ", variance=",summary.variance.number[1,end],
            ", new training solves=",diagnostics(study).solves)
    end
    println("Pointwise final 2.5/50/97.5% surrogate quantiles: ",quantiles[1,end,:])
    println("Validation solves: 128; independent test solves: 64; surrogate solves: ",diagnostics(samples).solves)
    (;problem,regression,projection,validation,test,checks,quantiles)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_example()
end

end
