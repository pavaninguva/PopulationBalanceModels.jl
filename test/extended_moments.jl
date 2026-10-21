using LinearAlgebra, SpecialFunctions
using Distributions: Gamma, Normal, cdf, pdf

@testset "Adaptive and extended moment reconstructions" begin
    raw(x,w,K)=[sum(w.*x.^k) for k in 0:K]
    for scale in (1.0,1e-6), rank in (1,2,3)
        x=scale.*collect(1.0:rank);w=collect(1.0:rank)
        g=MomentSet(0:7);mu=raw(x,w,7)
        q=quadrature(g,mu,RankAdaptiveQMOM(4))
        @test q.rank==rank
        @test q.nodes≈x rtol=1e-6
        @test q.weights≈w rtol=1e-6
        @test q.moment_error<1e-8
    end
    @test quadrature(MomentSet(0:5),zeros(6),RankAdaptiveQMOM(3)).rank==0
    @test_throws NumericalFailure quadrature(MomentSet(0:5),[1.,1,1,1,1,2],RankAdaptiveQMOM(3))
    for nodes in ([2.0],[1.0,3.0]), scale in (1.0,1e-6)
        n=length(nodes);w=ones(n)./n;sigma=0.2scale;x=nodes.*scale
        d=[Gamma(xi/sigma,sigma) for xi in x]
        mu=[sum(w[i]*prod((x[i]+j*sigma for j in 0:k-1);init=1.) for i in 1:n) for k in 0:2n]
        g=MomentSet(0:2n);method=EQMOM(n);q=quadrature(g,mu,method)
        @test q.width≈sigma rtol=1e-6
        @test q.primary_nodes≈x rtol=1e-6
        @test all(q.weights.>0)
        @test [sum(q.weights.*q.nodes.^k) for k in g.orders]≈mu rtol=1e-7
        edges=scale.*[0.,1.,2.,5.,Inf]
        expected=[sum(w[i]*(cdf(d[i],edges[j+1])-cdf(d[i],edges[j])) for i in 1:n) for j in 1:4]
        @test reconstructed_bins(g,mu,method,edges)≈expected rtol=1e-7
        @test reconstructed_density(g,mu,method,2scale)≈sum(w[i]*pdf(d[i],2scale) for i in 1:n) rtol=1e-6
        @test sum(reconstructed_bins(g,mu,method,edges;weight_order=2))≈mu[3] rtol=1e-6
    end
    # A non-polynomial growth integral has a known Gamma-function value.
    g=MomentSet(0:2);mu=[1.,2.,4.4];op=compile(PBMModel(Growth((x,p,t,c)->sqrt(x))),g;method=EQMOM(1;secondary_nodes=32))
    du=zeros(3,1);rhs!(du,reshape(mu,:,1),op,(;),0.)
    @test du[2]≈sqrt(.2)*exp(loggamma(10.5)-loggamma(10)) rtol=1e-7
    @test du[1]==0
    @test_throws ArgumentError EQMOM(2;kernel=:lognormal)
    @test_throws NumericalFailure quadrature(MomentSet(0:2),[1.,0.,1.],EQMOM(1))
    @test_throws ArgumentError compile(PBMModel(),MomentSet(0:3);method=EQMOM(2))
    @test sum(reconstructed_bins(MomentSet(0:2),[1.,2.,4.],EQMOM(1),[0.,2.,3.]))≈1
    @test_throws ArgumentError reconstructed_density(MomentSet(0:2),[1.,2.,4.],EQMOM(1),2.)
end

@testset "Direct quadrature dynamics and exchange" begin
    m=DQMOM(2);g=MomentSet(0:3;coordinate=:volume)
    x=[1.,3.];w=[.4,.6];u=quadrature_state(m,x,w)
    op=compile(PBMModel(Growth((x,p,t,c)->.2x)),g;method=m)
    prob=PBMProblem(op,u,(0.,.5));sol=simulate(prob;dt=.005)
    q=quadrature(g,last(sol.u),m)
    @test q.weights≈w atol=1e-10
    @test q.nodes≈x.*exp(.1) rtol=1e-9
    # Collisions conserve additive material and count one lost particle/event.
    agg=compile(PBMModel(Aggregation(ConstantKernel(.3))),g;method=m)
    du=zeros(4,1);rhs!(du,reshape(u,:,1),agg,(;),0.)
    @test moment_rate(m,g,u,du,0)≈-.15
    @test moment_rate(m,g,u,du,1)≈0 atol=1e-12
    sys=PBMSystem((particles=op,);exchanges=(MomentExchange(:particles,:c,1),))
    coupled=PBMProblem(sys,(particles=u,),(0.,.5);bulk0=(c=5.,))
    ss=simulate(coupled;dt=.005)
    @test bulk(coupled,last(ss.u),:c)+moment_value(m,g,population(coupled,last(ss.u),:particles),1)≈7.2 atol=1e-10
    @test_throws NumericalFailure quadrature_state(m,[1.,1.],w)
    @test_throws NumericalFailure quadrature_state(m,x,[0.,1.])
    @test_throws NumericalFailure quadrature_state(m,[-1.,3.],w)
    # Micrometre coordinates must not trigger a dimensionful matrix condition test.
    @test quadrature(g,quadrature_state(m,1e-6.*x,w),m).nodes≈1e-6.*x
end

@testset "Moment method statistical observations" begin
    # DQMOM differentiates its fixed-rank LU source solve away from collisions.
    method=DQMOM(2);basis=MomentSet(0:3)
    op=compile(PBMModel(Growth((x,p,t,ctx)->p.k*x)),basis;method)
    pb=PBMProblem(op,quadrature_state(method,[1.,3.],[.4,.6]),(0.,.2);p=(k=.2,))
    measurement=Measurement([.2],[2.2exp(.04)],MomentObservable(1);likelihood=GaussianLikelihood(sigma=.1))
    ex=InferenceExperiment(:run,pb,measurement;dt=.005)
    problem=EstimationProblem((ParameterSpec(:k;initial=.2,prior=Prior(Normal(.2,.1);coordinates=:physical)),),(ex,))
    @test check_gradient(problem,initial_coordinates(problem)).valid
    # EQMOM histograms are opt-in kernel integrals, never implicit QMOM atoms.
    basis=MomentSet(0:2);eq=EQMOM(1)
    @test_throws NumericalFailure reconstructed_bins(basis,zeros(3),eq,[0.,1.];normalize=true)
    @test sum(reconstructed_bins(basis,[1.,2.,4.4],eq,[0.,1.,2.];normalize=true))≈1
end

@testset "Quadrature collision evolution" begin
    basis=MomentSet(0:3);x=[1.,3.];w=[.4,.6]
    model=PBMModel(aggregation=Aggregation(ConstantKernel(.3)))
    direct=DQMOM(2);dq=compile(model,basis;method=direct)
    qm=compile(model,basis;method=QMOM(2))
    dsol=simulate(PBMProblem(dq,quadrature_state(direct,x,w),(0.,.5));dt=.002)
    qsol=simulate(PBMProblem(qm,[sum(w.*x.^k) for k in 0:3],(0.,.5));dt=.002)
    @test [moment_value(direct,basis,last(dsol.u),k) for k in 0:3]≈vec(last(qsol.u)) rtol=1e-7
    @test moment_value(direct,basis,last(dsol.u),0)≈1/(1+.3*.5/2) rtol=1e-9
    @test moment_value(direct,basis,last(dsol.u),1)≈2.2 atol=1e-11
    eqbasis=MomentSet(0:2)
    eq=compile(model,eqbasis;method=EQMOM(1))
    u=reshape([1.,2.,4.4],:,1);du=similar(u);rhs!(du,u,eq,(;),0.)
    @test du[1]≈-.15 atol=1e-12
    @test du[2]≈0 atol=1e-12
    @test du[3]≈.3*2^2 atol=1e-11
end
