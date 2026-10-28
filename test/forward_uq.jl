using Test, PopulationBalanceModels, LinearAlgebra, Random
import Distributions
import Statistics

@testset "Forward UQ: orthonormal PCE and output layout" begin
    grid=Grid1D([0.,.4,1.])
    base=PBMProblem(compile(PBMModel(),grid),grid.widths,(0.,1.))
    inputs=UncertainInputs((x=Distributions.Normal(),u=Distributions.Uniform(-1,1));coordinates=:reference)
    outputs=(polynomial=FunctionObservable((N,op,p,t)->[1+2p.x+3p.u+4(p.x^2-1)+t,p.x*p.u]),
             square=FunctionObservable((N,op,p,t)->p.x^2))
    problem=UQProblem(base,inputs;outputs,times=[0.,.4],dt=.1,
        output_coordinates=(polynomial=(units=:dimensionless,),square=(units=:dimensionless,)))
    design=pce_design(inputs;n=32,seed=45)
    fit=fit_pce(problem,PolynomialChaos(2);design,max_solves=32)
    summary=uq_summary(fit)
    @test diagnostics(fit).terms==6
    @test summary.mean.polynomial≈[1. 1.4;0. 0.] atol=3e-14
    @test summary.mean.square≈ones(1,2) atol=3e-14
    @test summary.variance.polynomial≈[39. 39.;1/3 1/3] atol=2e-12
    @test summary.variance.square≈fill(2.,1,2) atol=1e-13
    @test summary.covariance[1,5]≈8 atol=2e-13
    row(a)=only(findall(i->fit.multi_indices[i,:]==a,axes(fit.multi_indices,1)))
    @test fit.coefficients[row([1,0]),1]≈2 atol=2e-14
    @test fit.coefficients[row([0,1]),1]≈sqrt(3) atol=2e-14
    @test fit.coefficients[row([2,0]),1]≈4sqrt(2) atol=3e-14
    @test fit.coefficients[row([1,1]),2]≈1/sqrt(3) atol=2e-14

    # Independent Gaussian/Legendre quadratures integrate degree-four products.
    x=[-sqrt(3.),0.,sqrt(3.)];wx=[1/6,2/3,1/6]
    u=[-sqrt(3/5),0.,sqrt(3/5)];wu=[5/18,4/9,5/18]
    points=hcat(([a,b] for a in x for b in u)...)
    weights=[a*b for a in wx for b in wu]
    V=PopulationBalanceModels._pce_basis(inputs,fit.multi_indices,points)
    @test V'*Diagonal(weights)*V≈Matrix{Float64}(I,6,6) atol=2e-15
    validation=pce_design(inputs;n=50,seed=19)
    checked=validate_pce(fit,propagate(problem,MonteCarlo(50);samples=validation,max_solves=50))
    @test maximum(checked.rmse)<2e-13
    @test evaluate_pce(fit,points).values[1,:]≈[1+2a+3b+4(a^2-1) for a in x for b in u] atol=1e-13
    @test diagnostics(evaluate_pce(fit,MonteCarlo(7))).solves==0
    @test_throws ArgumentError validate_pce(fit,fit.training)
    @test_throws ArgumentError fit_pce(problem,PolynomialChaos(2);design=zeros(2,12),max_solves=12)
    @test_throws ArgumentError fit_pce(problem,PolynomialChaos(2);design=points[:,1:5],max_solves=5)
    @test_throws ArgumentError fit_pce(problem,PolynomialChaos(2;condition_limit=10);design=1e-8*points,max_solves=9)
    @test_throws ArgumentError fit_pce(problem,PolynomialChaos(1);design,validation=design,max_solves=64)
    @test_throws ArgumentError fit_pce(problem,PolynomialChaos(1);design,validation,max_solves=81)
    wrong=merge(design,(coordinates=:physical,))
    @test_throws ArgumentError fit_pce(problem,PolynomialChaos(1);design=wrong,max_solves=32)
    reused=fit_pce(problem,PolynomialChaos(2);design=fit.training,max_solves=0)
    @test reused.coefficients≈fit.coefficients
    @test diagnostics(reused).new_full_solves==0

    four=UncertainInputs((a=Distributions.Normal(),b=Distributions.Normal(),c=Distributions.Normal(),d=Distributions.Normal()))
    fourproblem=UQProblem(base,four;outputs=(y=FunctionObservable((N,o,p,t)->p.a+p.b+p.c+p.d),),times=[0.],dt=.1)
    for (degree,n,terms) in ((1,12,5),(2,32,15))
        pce=fit_pce(fourproblem,PolynomialChaos(degree);design=pce_design(four;n,seed=n),max_solves=n)
        @test diagnostics(pce).terms==terms
        @test only(uq_summary(pce).variance.y)≈4 atol=1e-13
    end
    bounded=pce_design(inputs;n=12,bounds=[-2. 2.;-.8 .8],seed=2)
    @test maximum(abs,bounded.samples[1,:])<2
    @test bounded.design_measure===:uniform_training_box && !bounded.changes_input_law
    @test pce_design(inputs;n=12,seed=3).samples==pce_design(inputs;n=12,seed=3).samples
end

@testset "Forward UQ transformations, support and retained PCE failures" begin
    g=Grid1D([0.,.5,1.]);base=PBMProblem(compile(PBMModel(),g),g.widths,(0.,1.))
    input=UncertainInputs((a=Distributions.Normal(3.,2.),b=Distributions.Uniform(2.,6.));
        coordinates=:reference,map=z->(q=(z.a-3)/2+z.b-4,))
    problem=UQProblem(base,input;outputs=(q=FunctionObservable((N,o,p,t)->p.q),),times=[0.],dt=.1)
    pce=fit_pce(problem,PolynomialChaos(1);design=pce_design(input;n=14,seed=61),max_solves=14)
    @test only(uq_summary(pce).mean.q)≈0 atol=2e-14
    @test only(uq_summary(pce).variance.q)≈7/3 atol=3e-14
    @test evaluate_pce(pce,[3. 3.;2. 6.]).values≈[-2. 2.] atol=1e-13
    @test_throws ArgumentError evaluate_pce(pce,reshape([3.,7.],2,1))
    @test_throws ArgumentError fit_pce(problem,PolynomialChaos(1);design=[3. 2. 4.;7. 4. 3.],max_solves=3)
    normal=UncertainInputs((a=Distributions.Normal(),))
    invalid=UQProblem(base,normal;outputs=(q=FunctionObservable((N,o,p,t)->p.a>0 ? throw(NumericalFailure("failed observation")) : p.a),),times=[0.],dt=.1)
    failed=try
        fit_pce(invalid,PolynomialChaos(1);design=[-1. 0. 1.],max_solves=3)
    catch e
        e
    end
    @test failed isa PCEFailure
    @test failed.evaluations.success==[true,true,false]
    @test failed.evaluations.failures[1].stage===:observation
    @test failed.evaluations.metadata.solves==3
    nan=UQProblem(base,normal;outputs=(q=FunctionObservable((N,o,p,t)->NaN),),times=[0.],dt=.1)
    result=propagate(nan,MonteCarlo(2);max_solves=2)
    @test !any(result.success) && length(result.failures)==2
    @test_throws ArgumentError uq_summary(result;conditional_on_success=true)
end

@testset "Forward MC: joint inputs, imported draws, failures and budgets" begin
    grid=Grid1D([0.,.4,1.]);base=PBMProblem(compile(PBMModel(),grid),grid.widths,(0.,1.))
    source=UncertainInputs(Distributions.MvNormal([1.,2.],[1. .8;.8 2.]);names=(:a,:b))
    problem=UQProblem(base,source;outputs=(y=FunctionObservable((N,o,p,t)->[p.a,p.b]),),times=[0.],dt=.1)
    result=propagate(problem,MonteCarlo(3000;seed=97);max_solves=3000)
    s=uq_summary(result)
    @test vec(s.mean.y)≈[1.,2.] atol=.07
    @test s.covariance≈[1. .8;.8 2.] atol=.1
    @test result.inputs==propagate(problem,MonteCarlo(3000;seed=97);max_solves=3000).inputs
    @test diagnostics(result).solves==3000
    @test s.mean_mcse!==nothing
    for p in (0.,.01,.5,.99,1.)
        @test PopulationBalanceModels._uq_quantile(sort(result.values[1,:]),p)≈Statistics.quantile(result.values[1,:],p)
    end
    supplied=UncertainInputs([1. 2. 3.;3. 6. 9.];names=(:a,:b),ids=[:first,:second,:third])
    fixed=UQProblem(base,supplied;outputs=problem.outputs,times=[0.],dt=.1)
    f=propagate(fixed,MonteCarlo(3);max_solves=3)
    @test f.ids==[:first,:second,:third]
    @test f.values==[1. 2. 3.;3. 6. 9.]
    @test uq_summary(f).mean_mcse===nothing
    @test_throws ArgumentError propagate(fixed,MonteCarlo(4);max_solves=4)
    @test_throws ArgumentError pce_design(supplied;n=10)
    calls=Ref(0)
    solver=(pb,ts)->begin calls[]+=1;simulate(pb;dt=.1,saveat=ts) end
    budget=UQProblem(base,source;outputs=problem.outputs,times=[0.],solver)
    @test_throws ArgumentError propagate(budget,MonteCarlo(3);max_solves=2)
    @test calls[]==0

    draws=(z=reshape([0.,1.,2.],1,:),physical=[(a=1.,b=2.),(a=2.,b=4.),(a=3.,b=6.)],
        in_support=[true,true,true],coordinates=:inference,method=:mcmc,indices=[8,2,11])
    posterior=UncertainInputs(draws)
    pp=UQProblem(base,posterior;outputs=problem.outputs,times=[0.],dt=.1)
    pr=propagate(pp,MonteCarlo(3);max_solves=3)
    @test pr.ids==draws.indices
    @test pr.metadata.input_metadata.inference_draws==draws.z
    @test !pr.metadata.iid && uq_summary(pr).mean_mcse===nothing

    invalid=UncertainInputs((rng)->(a=randn(rng),b=0.);iid=true,valid=p->p.a>0)
    ip=UQProblem(base,invalid;outputs=problem.outputs,times=[0.],dt=.1)
    ir=propagate(ip,MonteCarlo(40;seed=8);max_solves=40)
    @test 0<count(ir.success)<40
    @test diagnostics(ir).solves==count(ir.success)
    @test all(f->f.stage===:input,ir.failures)
    @test all(isnan,ir.values[:,.!ir.success])
    @test_throws ArgumentError uq_summary(ir)
    @test uq_summary(ir;conditional_on_success=true).conditional_on_success

    partial=UQProblem(base,supplied;outputs=problem.outputs,times=[0.],
        solver=(pb,ts)->pb.p.a==2 ? throw(NumericalFailure("deliberate numerical failure")) : simulate(pb;dt=.1,saveat=ts))
    failed=propagate(partial,MonteCarlo(3);max_solves=3)
    @test failed.success==[true,false,true] && diagnostics(failed).solves==3
    @test only(failed.failures).id===:second
    bug=UQProblem(base,supplied;outputs=problem.outputs,times=[0.],solver=(p,t)->error("programming error"))
    @test_throws ErrorException propagate(bug,MonteCarlo(3);max_solves=3)
    changed=UQProblem(base,supplied;outputs=(q=FunctionObservable((N,o,p,t)->ones(Int(p.a))),),times=[0.],dt=.1)
    @test_throws DimensionMismatch propagate(changed,MonteCarlo(3);max_solves=3)
    badtimes=UQProblem(base,supplied;outputs=problem.outputs,times=[0.],solver=(p,t)->simulate(p;dt=.1,saveat=[.5]))
    @test_throws ArgumentError propagate(badtimes,MonteCarlo(3);max_solves=3)
    iid=UncertainInputs(Distributions.Uniform();names=(:a,),iid=true)
    ep=UQProblem(base,iid;outputs=(q=FunctionObservable((N,o,p,t)->p.a),),times=[0.],dt=.1)
    event=uq_summary(propagate(ep,MonteCarlo(30);max_solves=30);events=(never=y->only(y.q)<0,))
    @test event.events.never.probability==0
    @test event.events.never.binomial_95_interval[2]>0
end

@testset "Analytical bounded-hazard PBM uncertainty" begin
    g=Grid1D([0.,.3,1.]);op=compile(PBMModel(Death((x,p,t,c)->p.k)),g)
    base=PBMProblem(op,2g.widths,(0.,.4);p=(k=.4,))
    input=UncertainInputs((k=Distributions.Uniform(.2,.6),))
    problem=UQProblem(base,input;outputs=(number=MomentObservable(0),),times=[.4],dt=.02,method=RK4())
    result=propagate(problem,MonteCarlo(4000;seed=6);max_solves=4000)
    summary=uq_summary(result)
    E(t)=(exp(-.2t)-exp(-.6t))/(.4t)
    expected=2E(.4);variance=4(E(.8)-E(.4)^2)
    @test abs(only(summary.mean.number)-expected)<5sqrt(variance/4000)
    @test only(summary.variance.number)≈variance rtol=.06
    validation=propagate(problem,MonteCarlo(120;seed=23);max_solves=120)
    fits=[fit_pce(problem,PolynomialChaos(degree);design=pce_design(input;n=20,seed=3),validation,max_solves=20) for degree in (1,3)]
    @test maximum(fits[2].validation.rmse)<maximum(fits[1].validation.rmse)/100
    @test only(uq_summary(fits[2]).mean.number)≈expected atol=2e-7
    @test only(uq_summary(fits[2]).variance.number)≈variance rtol=2e-5
end

@testset "Forward UQ respects representation and coupling observations" begin
    inputs=UncertainInputs([(scale=1.,),(scale=2.,)])
    for method in (QMOM(2),DQMOM(2))
        basis=MomentSet(0:3);op=compile(PBMModel(),basis;method)
        initial=method isa DQMOM ? quadrature_state(method,[1.,3.],[2.,1.]) : [3.,5.,11.,29.]
        base=PBMProblem(op,initial,(0.,.1))
        problem=UQProblem(base,inputs;setup=(p,b)->remake(b;u0=p.scale*b.u0),
            outputs=(M0=MomentObservable(0),M1=MomentObservable(1)),times=[0.],dt=.1)
        result=propagate(problem,MonteCarlo(2);max_solves=2)
        @test result.values≈[3. 6.;5. 10.]
    end
    tensor=Grid2D(Grid1D([0.,.3,1.]),Grid1D([0.,.5,2.]))
    base=PBMProblem(compile(PBMModel(),tensor),tensor.x.widths*tensor.y.widths',(0.,.1))
    problem=UQProblem(base,inputs;setup=(p,b)->remake(b;u0=p.scale*b.u0),
        outputs=(joint=MomentObservable((1,1)),),times=[0.,.1],dt=.1)
    @test propagate(problem,MonteCarlo(2);max_solves=2).values≈[1. 2.;1. 2.]
    decay=compile(PBMModel(Death((xy,p,t,c)->p.scale)),tensor)
    dynamic=PBMProblem(decay,base.u0,(0.,.1);p=(scale=1.,))
    dynamicuq=UQProblem(dynamic,inputs;outputs=(joint=MomentObservable((1,1)),),
        times=[0.,.1],dt=.01,method=RK4())
    @test propagate(dynamicuq,MonteCarlo(2);max_solves=2).values≈[1. 1.;exp(-.1) exp(-.2)] atol=3e-10
    g=Grid1D([0.,.4,1.]);op=compile(PBMModel(),g)
    sys=PBMSystem((cells=op,);bulk_rhs=(c,p,t)->(solute=0.,))
    batch=PBMProblem(sys,(cells=g.widths,),(0.,.1);bulk0=(solute=2.,))
    problem=UQProblem(batch,inputs;setup=(p,b)->PBMProblem(sys,(cells=p.scale*g.widths,),b.tspan;bulk0=(solute=2p.scale,)),
        outputs=(number=PopulationMoment(:cells,0),solute=BulkObservable(:solute)),times=[0.],dt=.1)
    @test propagate(problem,MonteCarlo(2);max_solves=2).values≈[1. 2.;2. 4.]
    decayop=compile(PBMModel(Death((x,p,t,c)->p.scale)),g)
    dynamicsys=PBMSystem((cells=decayop,);bulk_rhs=(ctx,p,t)->(solute=-sum(ctx.derivatives.cells),))
    dynamicbase=PBMProblem(dynamicsys,(cells=g.widths,),(0.,.1);bulk0=(solute=2.,),p=(scale=1.,))
    dynamicuq=UQProblem(dynamicbase,inputs;outputs=(number=PopulationMoment(:cells,0),solute=BulkObservable(:solute)),
        times=[.1],dt=.01,method=RK4())
    dynamicresult=propagate(dynamicuq,MonteCarlo(2);max_solves=2)
    @test dynamicresult.values[1,:]≈exp.(-0.1 .* [1.,2.]) atol=3e-10
    @test vec(sum(dynamicresult.values;dims=1))≈[3.,3.] atol=1e-14
    rebuilt=UQProblem(PBMProblem(op,g.widths,(0.,.1)),inputs;
        setup=(p,b)->begin
            mesh=Grid1D(range(0,1;length=Int(2p.scale)+1))
            PBMProblem(compile(PBMModel(),mesh),mesh.widths,b.tspan)
        end,outputs=(number=MomentObservable(0),),times=[0.],dt=.1)
    @test propagate(rebuilt,MonteCarlo(2);max_solves=2).values≈ones(1,2)
end
