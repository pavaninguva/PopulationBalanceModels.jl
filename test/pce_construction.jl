using Test, PopulationBalanceModels, LinearAlgebra, Random
import Distributions

function pce_test_problem(inputs; output=(N,o,p,t) -> p.x^3, solver=nothing)
    grid = Grid1D([0.,0.4,1.])
    base = PBMProblem(compile(PBMModel(),grid),grid.widths,(0.,1.))
    UQProblem(base,inputs; outputs=(q=FunctionObservable(output),),times=[0.,0.2],
        solver,dt=solver === nothing ? 0.1 : nothing)
end

@testset "Jacobi probability normalization and independent beta moments" begin
    rawmoment(α,β,k) = prod((α+j)/(α+β+j) for j in 0:k-1;init=1.0)
    for (α,β) in ((2.,5.),(5.,2.),(.5,.5),(.2,.8),(.1,.1),(1.,1.),(.01,4.),(4.,.01))
        input = UncertainInputs((x=Distributions.Beta(α,β),))
        rule = pce_quadrature(input;order=8)
        @test rule.basis_families == (:jacobi,)
        @test all(>(0),rule.weights)
        @test sum(rule.weights) ≈ 1 atol=3e-13
        # These moments come directly from the beta integral, independently of
        # either the Jacobi recurrence or its numerical quadrature construction.
        for k in 0:15
            @test sum(rule.weights.*rule.samples[1,:].^k) ≈ rawmoment(α,β,k) atol=3e-13 rtol=5e-11
        end
        μ,m2,m3,m4 = [rawmoment(BigFloat(α),BigFloat(β),k) for k in 1:4]
        σ = sqrt(m2-μ^2)
        a = (m3-μ*m2)/(m2-μ^2)
        b = a*μ-m2
        norm2 = m4-2a*m3+(a^2+2b)*m2-2a*b*μ+b^2
        points = reshape([.07,.33,.81],1,:)
        V = PopulationBalanceModels._pce_basis(input,reshape([0,1,2],3,1),points)
        @test V[:,1] == ones(3)
        @test V[:,2] ≈ (vec(points).-μ)./σ rtol=2e-12
        @test V[:,3] ≈ (vec(points).^2 .- a.*vec(points) .+ b)./sqrt(norm2) rtol=2e-9
    end
    @test_throws ArgumentError pce_basis(Distributions.Beta(),-1)
    @test_throws ArgumentError pce_basis(Distributions.Gamma(),2)
    @test_throws ArgumentError PCEBasis(:bad,[0.,1.],[-1.])
    @test_throws ArgumentError PCEBasis(:bad,[0.],[];scale=0)
    @test_throws ArgumentError PCEBasis(:bad,[0.,1.],[])
end

@testset "Quadrature resolution is distinct from polynomial degree" begin
    input = UncertainInputs((x=Distributions.Normal(),))
    problem = pce_test_problem(input;output=(N,o,p,t) -> exp(.4p.x))
    method = PolynomialChaos(4;construction=:projection)
    fits = [fit_pce(problem,method;design=pce_quadrature(input;order=q),max_solves=q) for q in (5,10)]
    # E[exp(a X)*He_k(X)]/sqrt(k!) = exp(a²/2)*a^k/sqrt(k!).
    exact = [exp(.08)*.4^k/sqrt(factorial(k)) for k in 0:4]
    @test norm(fits[2].coefficients[:,1]-exact) < norm(fits[1].coefficients[:,1]-exact)/100
    @test fits[2].coefficients[:,1] ≈ exact atol=2e-11
    validation = propagate(problem,MonteCarlo(80;seed=83);max_solves=80)
    study = refine_pce(problem,[method,method];
        designs=[pce_quadrature(input;order=q) for q in (5,10)],validation,max_solves=15,
        mean_atol=1e-12,variance_atol=1e-12,validation_rtol=.01)
    @test study.converged
    @test diagnostics(study).solves == 15
    @test validate_pce(study.surrogate,validation;atol=1e-10,rtol=.01).passed
end

@testset "Gaussian probability quadrature and projected mixed outputs" begin
    # Known normal even moments check tail contributions independently, including
    # orders where first components of numerical eigenvectors can vanish.
    for order in (16,64,128)
        input = UncertainInputs((x=Distributions.Normal(2.,3.),))
        rule = pce_quadrature(input;order)
        z = (vec(rule.samples).-2)./3
        @test sum(rule.weights) ≈ 1 atol=3e-13
        @test all(>(0),rule.weights)
        @test sum(rule.weights.*z.^24) ≈ prod(Float64.(1:2:23)) rtol=2e-12
    end
    input = UncertainInputs((b=Distributions.Beta(2,5),x=Distributions.Normal(2,.5),
                            u=Distributions.Uniform(-3,1));coordinates=:reference)
    problem = pce_test_problem(input;output=(N,o,p,t) -> [p.b,p.b^2,p.x+p.u^2,p.b*p.x])
    rule = pce_quadrature(input;order=(3,3,3))
    projected = fit_pce(problem,PolynomialChaos(2;construction=:projection);design=rule,max_solves=27)
    regressed = fit_pce(problem,PolynomialChaos(2);design=rule.samples,max_solves=27)
    @test projected.coefficients ≈ regressed.coefficients atol=5e-14
    summary = uq_summary(projected)
    @test summary.mean.q[:,1] ≈ [2/7,3/28,13/3,4/7] atol=1e-13
    @test summary.variance.q[:,1] ≈ [5/196,1/42-(3/28)^2,1/4+304/45,(3/28)*4.25-(4/7)^2] atol=1e-13
    @test diagnostics(projected).construction === :projection
    @test diagnostics(projected).quadrature_gram_error < 1e-12
    @test diagnostics(projected).basis_families == (:jacobi,:hermite,:legendre)
    @test diagnostics(projected).accuracy_status === :not_run
    reused = fit_pce(problem,PolynomialChaos(2;construction=:projection);design=projected.training,max_solves=0)
    @test reused.coefficients == projected.coefficients
    @test diagnostics(reused).new_full_solves == 0
    validation = propagate(problem,MonteCarlo(24;seed=36);max_solves=24)
    checked = validate_pce(projected,validation;atol=(q=fill(1e-12,4,2),),rtol=1e-12)
    @test checked.passed
    @test maximum(checked.rmse) < 1e-13
    @test evaluate_pce(projected,MonteCarlo(7)).metadata.solves == 0
    @test_throws ArgumentError fit_pce(problem,PolynomialChaos(2);design=rule,max_solves=27)
    @test_throws ArgumentError fit_pce(problem,PolynomialChaos(2;construction=:projection);design=rule.samples,max_solves=27)
    @test_throws ArgumentError fit_pce(problem,PolynomialChaos(2;construction=:projection);design=merge(rule,(weights=2rule.weights,)),max_solves=27)
    @test_throws ArgumentError fit_pce(problem,PolynomialChaos(2;construction=:projection);design=merge(rule,(weights=-rule.weights,)),max_solves=27)
    @test_throws ArgumentError fit_pce(problem,PolynomialChaos(2;construction=:projection);design=merge(rule,(weights=rule.weights[1:end-1],)),max_solves=27)
    @test_throws ArgumentError fit_pce(problem,PolynomialChaos(2;construction=:projection);design=pce_quadrature(input;order=(2,2,3)),max_solves=12)
    @test_throws ArgumentError fit_pce(problem,PolynomialChaos(1);design=pce_design(input;n=14),validation=rule,max_solves=41)
    @test_throws ArgumentError validate_pce(projected,projected.training)
    @test_throws ArgumentError pce_quadrature(input;order=(2,3))
    @test_throws ArgumentError pce_quadrature(input;order=0)
    @test_throws ArgumentError pce_quadrature(input;order=100,max_points=999_999)
    @test_throws ArgumentError PolynomialChaos(2;construction=:intrusive)
    @test_throws ArgumentError PolynomialChaos(2;quadrature_tolerance=0)
end

# A user-owned distribution exercises the public recurrence contract; this is
# the exponential weight's shifted-sign Laguerre basis, not another built-in.
struct PCEUnitExponential <: Distributions.ContinuousUnivariateDistribution end
Base.rand(rng::AbstractRNG,::PCEUnitExponential) = randexp(rng)
Distributions.insupport(::PCEUnitExponential,x::Real) = x >= 0
Distributions.quantile(::PCEUnitExponential,u::Real) = -log1p(-u)
function PopulationBalanceModels.pce_basis(::PCEUnitExponential,p::Integer)
    p >= 0 || throw(ArgumentError("nonnegative degree required"))
    PCEBasis(:test_laguerre,[2.0k+1 for k in 0:p],Float64.(1:p))
end

@testset "Public PCE recurrence extension" begin
    input = UncertainInputs((x=PCEUnitExponential(),))
    problem = pce_test_problem(input;output=(N,o,p,t) -> p.x^2)
    rule = pce_quadrature(input;order=3)
    @test sum(rule.weights.*vec(rule.samples).^4) ≈ 24 atol=2e-13
    fit = fit_pce(problem,PolynomialChaos(2;construction=:projection);design=rule,max_solves=3)
    @test vec(fit.coefficients[:,1]) ≈ [2.,4.,2.] atol=2e-14
    @test uq_summary(fit).mean.q == fill(2.,1,2)
    @test uq_summary(fit).variance.q ≈ fill(20.,1,2) atol=2e-13
    @test all(isfinite,pce_design(input;n=12).samples)
end

@testset "Validation detects aliasing and refinement requires accuracy and stability" begin
    input = UncertainInputs((x=Distributions.Normal(),))
    problem = pce_test_problem(input)
    designs = [reshape(x,1,:) for x in ([-1.,0.,1.],[-2.,-1.,0.,1.,2.],[-2.5,-1.5,-.5,.5,1.5,2.5])]
    methods = PolynomialChaos.((1,3,3))
    validation = pce_design(input;n=200,method=:mc,seed=810)
    study = refine_pce(problem,methods;designs,validation,max_solves=214,
        validation_atol=1e-10,validation_rtol=1e-10,mean_atol=1e-10,variance_atol=1e-10)
    @test study.converged
    @test length(study.candidates) == 3
    @test diagnostics(study).solves == 214
    @test diagnostics(study).stop_reason === :criteria_met
    @test diagnostics(first(study.candidates)).condition < 2
    @test diagnostics(first(study.candidates)).training_rmse == [0.,0.]
    @test !first(study.history).validation_passed
    @test diagnostics(first(study.candidates)).accuracy_status === :failed
    @test study.candidates[2].validation.passed && !study.history[2].selected
    @test uq_summary(study.surrogate).variance.q ≈ fill(15.,1,2) atol=1e-12
    @test study.history[3].stability.passed
    @test diagnostics(study).validation_used_for_selection
    # A separate final test is not the validation set used to select the fit.
    final = propagate(problem,MonteCarlo(40;seed=811);max_solves=40)
    @test validate_pce(study.surrogate,final;atol=1e-10,rtol=1e-10).passed
    synthetic = evaluate_pce(study.surrogate,MonteCarlo(40;seed=812))
    @test_throws ArgumentError validate_pce(study.surrogate,synthetic)
    @test_throws ArgumentError fit_pce(problem,PolynomialChaos(3);design=synthetic,max_solves=0)
    @test_throws ArgumentError fit_pce(problem,PolynomialChaos(1);design=designs[1],validation=synthetic,max_solves=3)
    @test_throws ArgumentError refine_pce(problem,methods;designs,validation=synthetic,max_solves=14)
    exhausted = refine_pce(problem,methods[1:2];designs=designs[1:2],validation=study.validation,
        max_solves=8,validation_atol=1e-10,mean_atol=1e-10)
    @test !exhausted.converged && exhausted.surrogate === nothing
    @test diagnostics(exhausted).stop_reason === :plan_exhausted
    @test length(exhausted.candidates) == 2
    calls = Ref(0)
    counted = pce_test_problem(input;solver=(p,ts) -> begin calls[] += 1; simulate(p;dt=.1,saveat=ts) end)
    @test_throws ArgumentError refine_pce(counted,methods;designs,validation,max_solves=213)
    @test calls[] == 0
    @test_throws ArgumentError refine_pce(counted,methods;designs,validation=designs[3],max_solves=30)
    @test calls[] == 0
    @test_throws ArgumentError refine_pce(problem,methods[2:3];designs=[designs[2],designs[2]],validation,max_solves=210)
    @test_throws ArgumentError refine_pce(problem,methods;designs,validation=nothing,max_solves=214)
    @test_throws ArgumentError validate_pce(study.surrogate,final;atol=-1)
    @test_throws ArgumentError validate_pce(study.surrogate,final;rtol=NaN)
    @test_throws ArgumentError validate_pce(study.surrogate,final;atol=(unknown=1.,))
    @test_throws DimensionMismatch validate_pce(study.surrogate,final;atol=(q=zeros(2,1),))
end

@testset "Refinement reuses evaluations and resolves beta-removal uncertainty" begin
    g = Grid1D([0.,.3,1.])
    op = compile(PBMModel(Death((x,p,t,c) -> p.k)),g)
    base = PBMProblem(op,2g.widths,(0.,1.);p=(k=.5,))
    inputs = UncertainInputs((k=Distributions.Beta(2,2),))
    problem = UQProblem(base,inputs;outputs=(number=MomentObservable(0),),times=[1.],dt=.02,method=RK4())
    validation = propagate(problem,MonteCarlo(60;seed=39);max_solves=60)
    designs = [pce_quadrature(inputs;order=q) for q in (3,4,5,7)]
    methods = [PolynomialChaos(p;construction=:projection) for p in (1,2,3,4)]
    study = refine_pce(problem,methods;designs,validation,max_solves=19,
        validation_atol=1e-6,validation_rtol=.01,mean_atol=1e-8,mean_rtol=1e-4,
        variance_atol=1e-8,variance_rtol=.01)
    @test study.converged
    # Petzke's first example: N(t)=2 exp(-k t), k~Beta(2,2).
    # Direct beta integration gives E[exp(-s k)] below; no PCE is used in it.
    E(s) = 6*((s-2)+(s+2)*exp(-s))/s^3
    summary = uq_summary(study.surrogate)
    @test only(summary.mean.number) ≈ 2E(1.) rtol=2e-8
    @test only(summary.variance.number) ≈ 4*(E(2.)-E(1.)^2) rtol=1e-6
    @test diagnostics(study).solves <= 19
    # The same full-model samples support degree refinement at zero new cost.
    design = propagate(problem,MonteCarlo(16;seed=40);max_solves=16)
    reused = refine_pce(problem,PolynomialChaos.((3,4));designs=[design,design],
        validation,max_solves=0,validation_atol=1e-6,validation_rtol=.01,
        mean_atol=1e-6,mean_rtol=.001,variance_atol=1e-6,variance_rtol=.01)
    @test reused.converged
    @test diagnostics(reused).solves == 0
end

@testset "Refinement failure keeps partial work and sample-level evidence" begin
    input = UncertainInputs((x=Distributions.Normal(),))
    problem = pce_test_problem(input;output=(N,o,p,t) -> p.x > 2 ?
        throw(NumericalFailure("deliberate failed forward observation")) : p.x^3)
    methods = PolynomialChaos.((1,3,3))
    designs = [reshape(x,1,:) for x in ([-1.,0.,1.],[-2.,-1.,0.,1.,2.],[-2.5,-1.5,-.5,.5,1.5,2.5])]
    validation = reshape([-.8,-.3,.2,.7],1,:)
    study = refine_pce(problem,methods;designs,validation,max_solves=18,
        validation_atol=1e-10,mean_atol=1e-10,variance_atol=1e-10)
    @test !study.converged && study.surrogate === nothing
    @test length(study.candidates) == 2
    @test diagnostics(study).stop_reason === :forward_failure
    @test diagnostics(study).failure isa PCEFailure
    @test diagnostics(study).solves == 18
    @test count(diagnostics(study).failure.evaluations.success) == 5
    @test only(diagnostics(study).failure.evaluations.failures).stage === :observation
    @test last(study.history).candidate == 3
    failed_validation = refine_pce(problem,methods;designs,
        validation=reshape([.1,2.1],1,:),max_solves=16)
    @test !failed_validation.converged && isempty(failed_validation.candidates)
    @test diagnostics(failed_validation).stop_reason === :forward_failure
    @test diagnostics(failed_validation).solves == 2
    bug = pce_test_problem(input;output=(N,o,p,t) -> p.x > 2 ? error("programming error") : p.x^3)
    @test_throws ErrorException refine_pce(bug,methods;designs,validation,max_solves=18)

    constant = pce_test_problem(input;output=(N,o,p,t) -> [2.,20.])
    test = propagate(constant,MonteCarlo(12;seed=84);max_solves=12)
    fit = fit_pce(constant,PolynomialChaos(1);design=designs[1],validation=test,max_solves=3,
        validation_atol=(q=1e-12,),validation_rtol=0)
    @test fit.validation.passed
    @test all(isnan,fit.validation.normalized_rmse)
    @test fit.validation.rmse_tolerance == fill(1e-12,4)
end

@testset "Projected outputs retain moment, tensor and coupled semantics" begin
    input = UncertainInputs((scale=Distributions.Beta(2,5),);map=p -> (scale=1+p.scale,))
    rule = pce_quadrature(input;order=2)
    for method in (QMOM(2),DQMOM(2))
        basis = MomentSet(0:3)
        op = compile(PBMModel(),basis;method)
        initial = method isa DQMOM ? quadrature_state(method,[1.,3.],[2.,1.]) : [3.,5.,11.,29.]
        base = PBMProblem(op,initial,(0.,.1))
        problem = UQProblem(base,input;setup=(p,b) -> remake(b;u0=p.scale*b.u0),
            outputs=(moment=MomentObservable(1),),times=[.1],dt=.1)
        fit = fit_pce(problem,PolynomialChaos(1;construction=:projection);design=rule,max_solves=2)
        @test only(uq_summary(fit).mean.moment) ≈ 45/7 atol=2e-13
        @test only(uq_summary(fit).variance.moment) ≈ 125/196 atol=2e-13
    end
    tensor = Grid2D(Grid1D([0.,.3,1.]),Grid1D([0.,.5,2.]))
    base = PBMProblem(compile(PBMModel(),tensor),tensor.x.widths*tensor.y.widths',(0.,.1))
    problem = UQProblem(base,input;setup=(p,b) -> remake(b;u0=p.scale*b.u0),
        outputs=(joint=MomentObservable((1,1)),),times=[.1],dt=.1)
    fit = fit_pce(problem,PolynomialChaos(1;construction=:projection);design=rule,max_solves=2)
    @test only(uq_summary(fit).mean.joint) ≈ 9/7 atol=2e-14
    @test only(uq_summary(fit).variance.joint) ≈ 5/196 atol=2e-14
    g = Grid1D([0.,.4,1.])
    sys = PBMSystem((cells=compile(PBMModel(),g),);bulk_rhs=(c,p,t) -> (solute=0.,))
    batch = PBMProblem(sys,(cells=g.widths,),(0.,.1);bulk0=(solute=3.,))
    problem = UQProblem(batch,input;setup=(p,b) ->
        PBMProblem(sys,(cells=p.scale*g.widths,),b.tspan;bulk0=(solute=3p.scale,)),
        outputs=(number=PopulationMoment(:cells,0),solute=BulkObservable(:solute)),times=[0.,.1],dt=.1)
    fit = fit_pce(problem,PolynomialChaos(1;construction=:projection);design=rule,max_solves=2)
    @test uq_summary(fit).mean.number ≈ fill(9/7,1,2) atol=2e-14
    @test uq_summary(fit).mean.solute ≈ fill(27/7,1,2) atol=2e-14
    @test uq_summary(fit).covariance[1,3] ≈ 15/196 atol=2e-14
end
