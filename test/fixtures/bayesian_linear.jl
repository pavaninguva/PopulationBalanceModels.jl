function bayesian_linear_fixture(;gradient=nothing)
    g=Grid1D([0.0,0.5,1.0])
    pb=PBMProblem(compile(PBMModel(),g),[1.0,0.0],(0.0,3.0))
    times=[0.0,1.0,2.0,3.0];A=hcat(ones(4),times)
    y=[0.9,1.7,2.2,3.1];C=[0.04*0.2^abs(i-j) for i in 1:4,j in 1:4]
    specs=[ParameterSpec(:a;initial=0.8,prior=Prior(Normal(0,2);coordinates=:physical)),
        ParameterSpec(:b;initial=0.5,prior=Prior(Normal(0,1);coordinates=:inference),role=:nuisance)]
    m=Measurement(times,y,FunctionObservable((u,op,p,t)->p.a+p.b*t);likelihood=GaussianLikelihood(covariance=C))
    solver=(p,ts)->PBMSolution(ts,[copy(p.u0) for _ in ts],p,0)
    ex=InferenceExperiment(:run,pb,m;solver)
    problem=EstimationProblem(specs,(ex,);gradient)
    precision=A'*(C\A)+Diagonal([1/4,1.0]);covariance=precision\Matrix{Float64}(I,2,2)
    mode=precision\(A'*(C\y))
    (;problem,mode,covariance,precision,A,C,y,pb,solver)
end
