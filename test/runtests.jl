using Test
using PopulationBalanceModels
include("quality.jl")
struct UnsupportedDomain <: AbstractPopulationDomain end
include("audit_regressions.jl")
include("compact_finite_volume.jl")

@testset "PopulationBalanceModels.jl scientific foundation" begin
    @testset "Representation and model declarations" begin
        g=Grid1D([0,1,3,6])
        @test density(g,[1,4,9]) == [1,2,3]
        @test moment(g,[1,4,9],0) == 14
        @test moment(g,[1,4,9],1) == 0.5+8+40.5
        @test_throws ArgumentError Grid1D([0,1,1])
        @test_throws ArgumentError Grid1D([-1,0,1])
        model=PBMModel(Growth((v,p,t,c)->1)+Birth((v,p,t,c)->0)+Death((v,p,t,c)->0))
        @test length(model.terms)==3
        @test_throws ArgumentError compile(model,g;growth_scheme=WENO5())
        @test_throws ArgumentError compile(PBMModel(Aggregation((v,w,p,t,c)->1)),Grid1D(0:10;coordinate=:diameter))
        @test_throws ArgumentError SizeBC(Periodic(),NoFlux())
        @test_throws ArgumentError compile(PBMModel(),g;domain=UnsupportedDomain())
        @test_throws ArgumentError PBMProblem(compile(PBMModel(),g),[-1,0,1],(0,1))
    end

    @testset "Growth flux conservation and WENO convergence" begin
        for velocity in (-0.7,0.7)
            errors=Float64[]
            for bins in (20,40,80)
                g=Grid1D(range(0,2π;length=bins+1))
                N=reshape(2 .* g.widths .+ cos.(g.edges[1:end-1]).-cos.(g.edges[2:end]),:,1)
                op=compile(PBMModel(Growth((v,p,t,c)->velocity)),g;
                    growth_scheme=WENO5(epsilon=1e-10),bc=SizeBC(Periodic(),Periodic()))
                dN=similar(N); rhs!(dN,N,op,NamedTuple(),0.0)
                exact=velocity.*(sin.(g.edges[1:end-1]).-sin.(g.edges[2:end]))
                push!(errors,maximum(abs.(vec(dN).-exact)./g.widths))
                @test abs(sum(dN))<1e-12
            end
            @test errors[1]/errors[2]>12
            @test errors[2]/errors[3]>12
        end
        g=Grid1D(range(0,1;length=21)); N=reshape(copy(g.widths),:,1)
        model=PBMModel(Growth((v,p,t,c)->2.0))
        dN=similar(N)
        closed=compile(model,g)
        rhs!(dN,N,closed,NamedTuple(),0.0)
        @test sum(dN)≈0 atol=1e-14
        opened=compile(model,g;bc=SizeBC(OpenBoundary(),OpenBoundary()))
        rhs!(dN,N,opened,NamedTuple(),0.0)
        @test sum(dN)≈-2
        @test suggest_dt(opened,N,NamedTuple(),0.0)≈0.0125
        @test_throws ArgumentError compile(PBMModel(),Grid1D([0,1,2,3,4,5,7]);growth_scheme=WENO5())
    end

    @testset "Distributed birth, death and point nucleation" begin
        g=Grid1D(0.5:1.0:10.5)
        N=zeros(10,1); dN=similar(N)
        op=compile(PBMModel(Nucleation((p,t,c)->2.0,2.5)),g)
        rhs!(dN,N,op,NamedTuple(),0.0)
        @test sum(dN)≈2
        @test only(moment(g,dN,1))≈5
        @test_throws ArgumentError compile(PBMModel(Nucleation((p,t,c)->2,0.75)),g)
        op=compile(PBMModel(Birth((v,p,t,c)->2.0)+Death((v,p,t,c)->0.3)),g)
        N.=1
        rhs!(dN,N,op,NamedTuple(),0.0)
        @test all(isapprox.(dN,1.7))
        sol=simulate(PBMProblem(op,N,(0,1));dt=0.01)
        exact=2/0.3+(1-2/0.3)*exp(-0.3)
        @test maximum(abs.(sol.u[end].-exact))<1e-8
    end

    @testset "Aggregation event counting, conservation and overflow" begin
        g=Grid1D(0.5:1.0:10.5)
        N=zeros(10,1); N[1]=2
        dN=similar(N)
        op=compile(PBMModel(Aggregation((v,w,p,t,c)->0.5)),g)
        rhs!(dN,N,op,NamedTuple(),0.0)
        @test dN[1]≈-2
        @test dN[2]≈1
        @test sum(dN)≈-1
        @test only(moment(g,dN,1))≈0 atol=1e-14
        N.=0; N[2]=3; N[3]=2
        rhs!(dN,N,op,NamedTuple(),0.0)
        @test sum(dN)≈-0.5*0.5*sum(N)^2
        @test only(moment(g,dN,1))≈0 atol=1e-13
        # Nonuniform pivot interpolation must preserve first moment too.
        gn=Grid1D([0.1,1.0,2.0,3.0,6.0,12.0])
        nn=reshape([0.0,3.0,2.0,0.0,0.0],:,1); dn=similar(nn)
        on=compile(PBMModel(Aggregation((v,w,p,t,c)->0.5)),gn)
        rhs!(dn,nn,on,NamedTuple(),0.0)
        @test only(moment(gn,dn,1))≈0 atol=1e-12
        N.=0; N[end]=2
        @test_throws NumericalFailure rhs!(dN,N,op,NamedTuple(),0.0)
        discard=compile(PBMModel(Aggregation((v,w,p,t,c)->0.5;overflow=:discard)),g)
        rhs!(dN,N,discard,NamedTuple(),0.0)
        diag=diagnostics(discard,N,NamedTuple(),0.0)
        @test only(diag.unresolved_birth_number_rate)≈1
        @test only(moment(g,dN,1))+only(diag.unresolved_birth_size_rate)≈0 atol=1e-14
        @test sum(dN)≈-2
        bad=compile(PBMModel(Aggregation((v,w,p,t,c)->v)),g)
        N.=1
        @test_throws ArgumentError rhs!(dN,N,bad,NamedTuple(),0.0)
    end

    @testset "Breakage daughter normalization and conservation" begin
        g=Grid1D(0.5:1.0:10.5)
        N=zeros(10,1); N[4]=2; dN=similar(N)
        op=compile(PBMModel(Breakage((v,p,t,c)->v>=4 ? 0.3 : 0,
            (v,p,t,c)->((v/2,2.0),))),g)
        rhs!(dN,N,op,NamedTuple(),0.0)
        @test sum(dN)≈0.6
        @test only(moment(g,dN,1))≈0 atol=1e-14
        sol=simulate(PBMProblem(op,N,(0.0,2.0));dt=0.005)
        @test sol.u[end][4]≈2*exp(-0.6) atol=1e-9
        @test sol.u[end][2]≈4*(1-exp(-0.6)) atol=1e-9
        bad=compile(PBMModel(Breakage((v,p,t,c)->1,(v,p,t,c)->((v/2,1.0),))),g)
        @test_throws ArgumentError rhs!(dN,N,bad,NamedTuple(),0.0)
    end

    @testset "Observation likelihood and local sensitivities" begin
        g=Grid1D(range(0,8;length=41))
        op=compile(PBMModel(Death((v,p,t,c)->p.k)),g)
        initial=exp.(-g.centers).*g.widths
        ts=[0.0,0.25,0.75,1.0]; rate=0.35
        experiments=Experiment[]
        for scale in (1.0,2.0)
            prob=PBMProblem(op,scale*initial,(0.0,1.0);p=(k=rate,))
            data=scale*sum(initial).*exp.(-rate.*ts)
            obs=Observation(ts,data,MomentObservable(0);sigma=0.01)
            push!(experiments,Experiment(prob,obs))
        end
        cal=CalibrationProblem(experiments;parameters=(z,ex)->(k=exp(z[1]),),dt=0.0025)
        z=[log(rate)]
        @test maximum(abs.(residuals(cal,z)))<1e-8
        @test objective(cal,z)<objective(cal,[log(0.5)])
        report=sensitivity_report(cal,z)
        @test report.rank==1
        @test report.covariance!==nothing
        exact=vcat((-ts.*rate.*scale.*sum(initial).*exp.(-rate.*ts)./0.01 for scale in (1.0,2.0))...)
        @test report.jacobian[:,1]≈exact rtol=1e-7 atol=1e-7
        nonident=CalibrationProblem(experiments;parameters=(z,ex)->(k=exp(z[1])+exp(z[2]),),dt=0.005)
        @test sensitivity_report(nonident,log.([rate/2,rate/2])).rank==1
        @test sensitivity_report(nonident,log.([rate/2,rate/2])).covariance===nothing
        histogram=HistogramObservable([0.0,1.3,4.0,8.0])
        @test sum(PopulationBalanceModels._observe(histogram,reshape(initial,:,1),op,NamedTuple(),0.0))≈sum(initial)
        @test_throws ArgumentError Observation(ts,ones(4),MomentObservable(0);sigma=0)
        @test_throws DimensionMismatch Observation(ts,ones(2),MomentObservable(0))
        badcal=CalibrationProblem(experiments;parameters=(z,ex)->(k=-1.0,),dt=0.01)
        @test_throws DomainError objective(badcal,z;on_failure=:infinity)
        unstable=CalibrationProblem(experiments;parameters=(z,ex)->(k=1e4,),dt=0.1)
        @test objective(unstable,z;on_failure=:infinity)==Inf
    end

    @testset "Solver failures are explicit" begin
        g=Grid1D(0.0:0.1:1.0)
        op=compile(PBMModel(Death((v,p,t,c)->10.0)),g)
        prob=PBMProblem(op,ones(10),(0.0,1.0))
        @test_throws NumericalFailure simulate(prob;dt=1.0)
        @test_throws ArgumentError simulate(prob;dt=0.01,saveat=[0.5,0.1])
        @test_throws NumericalFailure simulate(prob;dt=0.01,maxsteps=1)
    end
end

include("systems.jl")
include("papers.jl")
include("sources_moments.jl")
include("tensor.jl")
include("method_interfaces.jl")

include("numerical_contracts.jl")

include("uncertainty.jl")
include("bayesian.jl")
include("differentiation.jl")
include("characteristic.jl")
include("transport_integrators.jl")
include("axial.jl")

include("li_supplement.jl")
include("extended_moments.jl")
include("axial_system.jl")
include("qamar_moments.jl")

include("measure_evolution.jl")
