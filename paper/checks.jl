using Test, PopulationBalanceModels, LinearAlgebra
include("case_studies/01_growth_time_methods/run.jl")
include("case_studies/02_aggregation_fragmentation/run.jl")
include("case_studies/03_tubular_precipitation/run.jl")
const P = PopulationBalanceModels

@testset "Paper numerical contracts" begin
    @testset "Exact growth initialization and roundoff orders" begin
        C = PaperGrowth.C
        g = Grid1D(range(0,1;length=101))
        @test sum(C.exact_counts(g,0.)) ≈ -expm1(-100.) atol=2e-15
        @test sum(C.smooth_counts(g,0.)) ≈ 1. atol=2e-15
        @test sum(C.exact_counts(g,4.)) ≈ -expm1(-60.) atol=2e-15
        @test length(PaperGrowth.choices()) == 6
        @test PaperGrowth.PaperWorkflow.overlap_error([0.,.2,1.],[2.,2.],[0.,.7,1.],[3.,3.]) ≈ 1/3
        widths = [.25,.75]
        @test sqrt(sum(widths.*[1.,3.].^2)/sum(widths)) ≈ sqrt(7.)
    end
    @testset "Independent Riccati number and material solution" begin
        Q = PaperMoments.Q
        steady = sqrt(.002)
        r = (1-steady)/(1+steady)
        for t in (0.,1.,10.,50.)
            number = steady*(1+r*exp(-steady*t))/(1-r*exp(-steady*t))
            @test Q.exact4(t)[1] ≈ number rtol=1e-14
            derivative = P.ForwardDiff.derivative(s->Q.exact4(s)[1],t)
            @test derivative ≈ -.5number^2+.001 atol=3e-15
            @test Q.exact4(t)[4] ≈ .001 rtol=2e-15
        end
        method = DQMOM(3)
        pb = Q.problem4(;method,tf=.01)
        @test [moment_value(method,pb.operator.grid,pb.u0,k) for k in 0:3] ≈ Q.exact4(0.,0:3) rtol=1e-12
        @test vec(pb.u0) != Q.exact4(0.,0:5)
        @test daughter_moment(UniformVolumeDaughters(),.37,3,(;),0.,(;)) ≈ .37^3
    end
    @testset "New log-grid injection and material exchange" begin
        C = PaperPrecipitation.C
        for bins in (100,200,400)
            pb = C.build(nx=bins,nz=8,mesh=:log,scheme=WENO5(),axial_scheme=MUSCL(),tf=5.)
            @test iszero(sum(pb.u0))
            pop = first(pb.operator.local_operators).system.populations.crystals
            @test only(filter(x->x.term isa Growth,pop.prepared)).cache isa P._NonuniformWENO5
            @test maximum(diff(pop.grid.widths)) > 0
            p = pb.p
            sys = PBMSystem((crystals=pop,);exchanges=(ConservativeExchange(:crystals,:c,(x,p,t,c)->1e12*p.rho*p.kv*x^3),),nonnegative_bulk=(:c,))
            localpb = PBMProblem(sys,(crystals=zeros(bins),),(0.,1.);bulk0=(c=2p.ceq,),p)
            du = zero(localpb.u0);rhs!(du,localpb.u0,localpb.operator,p,0.)
            @test population(localpb,du,:crystals)[1] ≈ p.kp/1e12 rtol=1e-13
            @test sum(population(localpb,du,:crystals)) ≈ p.kp/1e12 rtol=1e-13
            transfer = 1e12*p.rho*p.kv*sum(pop.grid.centers.^3 .* population(localpb,du,:crystals))
            @test bulk(localpb,du,:c) ≈ -transfer rtol=1e-13
            @test transfer ≈ p.kp*p.rho*p.kv*first(pop.grid.centers)^3 rtol=1e-13
        end
    end
    @testset "Conservative comparison on unmatched size and axial bins" begin
        C = PaperPrecipitation.C
        a = C.build(nx=8,nz=6,scheme=WENO5(),axial_scheme=MUSCL(),mesh=:log)
        b = C.build(nx=12,nz=8,scheme=WENO5(),axial_scheme=MUSCL(),mesh=:log)
        u,v = copy(a.u0),copy(b.u0)
        population(a,u,:crystals) .= 2PaperPrecipitation.grid(a).widths
        population(b,v,:crystals) .= 3PaperPrecipitation.grid(b).widths
        @test PaperPrecipitation.field_error(a,u,b,v) ≈ 1/3 rtol=1e-13
    end
    @testset "Checked implicit adapter versus independently refined native RK4" begin
        C,S = PaperPrecipitation.C,PaperPrecipitation.S
        pb = C.build(nx=12,nz=12,mesh=:log,scheme=WENO5(),axial_scheme=MUSCL(),tf=10.)
        times = [1.,5.,10.]
        a = S.advance(pb,times;reltol=1e-8,population_abstol=1e-16,bulk_abstol=1e-12,
            dtmax=.1,late_max_order=2,restart_time=5.,progress=false)
        b = C.advance(pb,times;fraction=.15)
        c = C.advance(pb,times;fraction=.075)
        @test diagnostics(a).checked_steps == a.steps
        @test diagnostics(a).maximum_bdf_order == 3
        @test diagnostics(a).late_maximum_bdf_order == 2
        @test diagnostics(a).restarts == 1
        @test diagnostics(a).phase_end_times == [5.,10.]
        @test diagnostics(a).phase_maximum_orders == [3,2]
        @test all((1 .<= diagnostics(a).phase_maximum_observed_orders) .&
            (diagnostics(a).phase_maximum_observed_orders .<= [3,2]))
        @test sum(diagnostics(a).phase_steps) == a.steps
        @test_throws ErrorException S.advance(pb,times;max_order=6,progress=false)
        @test_throws ErrorException S.advance(pb,times;late_max_order=6,progress=false)
        @test_throws ErrorException S.advance(pb,times;late_max_order=2,restart_time=2.,progress=false)
        @test a.t == times
        for i in eachindex(times)
            nref = sum(abs,population(pb,c.u[i],:crystals))
            ndiff = sum(abs,population(pb,a.u[i],:crystals).-population(pb,c.u[i],:crystals))/nref
            native = sum(abs,population(pb,b.u[i],:crystals).-population(pb,c.u[i],:crystals))/nref
            @test ndiff < 2e-3
            @test native < 2e-3
            @test maximum(abs.(bulk(pb,a.u[i],:c).-bulk(pb,c.u[i],:c))) < 1e-4
            println("CVODE/native verification t=",times[i],": relative population difference=",ndiff,
                "; native halving=",native,"; steps ",(a.steps,b.steps,c.steps))
        end
    end
    @testset "Preconditioner against independent dense face-flux operators" begin
        C,S = PaperPrecipitation.C,PaperPrecipitation.S
        pb = C.build(nx=8,nz=6,mesh=:log,scheme=WENO5(),axial_scheme=MUSCL(),tf=5.)
        g,z = PaperPrecipitation.grid(pb),pb.operator.domain.grid
        nx,nz = length(g),length(z)
        state = zero(pb.u0)
        state[end,:] .= 2pb.p.ceq
        velocity,dispersion = pb.operator.domain.velocity,pb.operator.domain.dispersion
        function axial_rhs(row)
            F = zeros(nz+1)
            for j in 2:nz
                F[j] = velocity*row[j-1]-dispersion*(row[j]-row[j-1])/(z.centers[j]-z.centers[j-1])
            end
            F[end] = velocity*row[end]
            (F[1:end-1].-F[2:end])./z.widths
        end
        function size_rhs(row)
            F = zeros(nx+1)
            F[2:nx] .= pb.p.kg.*row[1:nx-1]./g.widths[1:nx-1]
            F[1:end-1].-F[2:end]
        end
        Az = hcat([axial_rhs(Matrix{Float64}(I,nz,nz)[:,j]) for j in 1:nz]...)
        Ax = zeros(nx+1,nx+1)
        Ax[1:nx,1:nx] .= hcat([size_rhs(Matrix{Float64}(I,nx,nx)[:,j]) for j in 1:nx]...)
        Ax[end,end] = -pb.p.rho*pb.p.kv*first(g.centers)^3*pb.p.kp*pb.p.u/pb.p.ceq
        right = reshape([1+sin(i) for i in 1:length(pb.u0)],size(pb.u0))
        prec = S.preconditioner(pb)
        for gamma in (.1,.2,.2)
            answer = zero(right)
            prec(vec(answer),vec(right),pb.p,0.,vec(state),vec(zero(state)),gamma,1e-8,1)
            expected = (I-gamma*Ax)\right
            for i in axes(expected,1)
                expected[i,:] .= (I-gamma*Az)\expected[i,:]
            end
            @test answer ≈ expected rtol=3e-13
        end
    end
end
