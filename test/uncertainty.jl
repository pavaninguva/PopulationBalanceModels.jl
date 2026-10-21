@testset "Gaussian likelihood and local joint uncertainty" begin
    grid=Grid1D([0.0,0.5,1.0])
    problem=PBMProblem(compile(PBMModel(),grid),[1.0,0.0],(0.0,5.0))
    times=collect(0.0:1.0:5.0);A=hcat(ones(6),times)
    y=A*[1.2,0.3]+[0.02,-0.03,0.01,0.04,-0.02,0.0]
    C=[0.01*0.3^abs(i-j) for i in 1:6,j in 1:6]
    obs=Observation(times,y,FunctionObservable((N,op,p,t)->p.a+p.b*t);covariance=C)
    cal=CalibrationProblem((Experiment(problem,obs),);
        parameters=(z,ex)->(a=z[1],b=z[2]),dt=0.1)
    expected_covariance=inv(A'*(C\A))
    z=expected_covariance*(A'*(C\y)); raw=A*z-y
    r=residuals(cal,z)
    @test sum(abs2,r)≈dot(raw,C\raw) rtol=1e-11
    @test objective(cal,z)≈(dot(raw,C\raw)+logdet(C)+6log(2π))/2 rtol=1e-11
    @test vec(predictions(cal,z)[1][1])≈A*z
    sensitivity=sensitivity_report(cal,z)
    @test sensitivity.covariance≈expected_covariance rtol=1e-9
    report=fit_diagnostics(cal,z)
    @test report.rank==2 && report.degrees_of_freedom==4 && report.stationary
    @test report.difference_error<1e-8
    @test 0<report.goodness_of_fit_pvalue<1
    region=confidence_ellipsoid(cal,z)
    @test region.cutoff≈5.991464547107979 rtol=1e-12
    @test z in region
    @test confidence_ellipsoid(cal,z;level=Float32(0.95)).level≈0.95 rtol=1e-7
    points=ellipse_points(region;n=37)
    distances=[dot(v-z,expected_covariance\(v-z)) for v in eachcol(points)]
    @test all(isapprox.(distances,region.cutoff;rtol=1e-10))
    @test !(z+10(points[:,1]-z) in region)
    @test_throws ArgumentError confidence_ellipsoid(cal,z.+0.1)
    estimated=confidence_ellipsoid(cal,z;variance=:estimated)
    @test estimated.covariance≈expected_covariance*sum(abs2,r)/4 rtol=1e-9
    @test estimated.cutoff≈13.88854381999831 rtol=1e-10
    @test estimated.diagnostics.goodness_of_fit_pvalue===nothing
    @test_throws ArgumentError Observation(times,y,MomentObservable(0);sigma=1.0,covariance=C)
    @test_throws ArgumentError Observation(times,y,MomentObservable(0);covariance=ones(6,6))
    @test_throws ArgumentError Observation(times,y,MomentObservable(0);covariance=C.+triu(ones(6,6),1))
    @test_throws DimensionMismatch Observation(times,y,MomentObservable(0);covariance=ones(3,3))
    @test_throws ArgumentError Observation(times,complex.(y),obs.observable;sigma=0.1)
    @test_throws ArgumentError Observation(times,y,obs.observable;covariance=complex.(C))
    redundant=CalibrationProblem(cal.experiments;parameters=(z,e)->(a=z[1]+z[2],b=0.0),dt=0.1)
    @test fit_diagnostics(redundant,[0.5,0.5]).covariance===nothing
    @test_throws ArgumentError confidence_ellipsoid(redundant,[0.5,0.5])
    nonlinear=Observation(times,zeros(6),FunctionObservable((N,op,p,t)->p.a^3))
    nonlinear_cal=CalibrationProblem((Experiment(problem,nonlinear),);
        parameters=(z,e)->(a=z[1],),dt=0.1)
    @test_throws ArgumentError confidence_ellipsoid(nonlinear_cal,[0.0];relative_step=0.01)
    external=CalibrationProblem(cal.experiments;parameters=cal.parameters,
        solver=(p,ts)->simulate(p;dt=0.2,saveat=ts))
    @test residuals(external,z)≈r
    @test fit_diagnostics(cal,z;refined=external).numerical.prediction_change<1e-12
    bad_solver=CalibrationProblem(cal.experiments;parameters=cal.parameters,
        solver=(p,ts)->simulate(p;dt=0.1,saveat=[0.0,1.0]))
    @test_throws ArgumentError residuals(bad_solver,z)
end

include("../examples/temperature_programmed_crystallization.jl")
@testset "Native crystallization moment recipe and DLVO material helpers" begin
    recipe=TemperatureProgrammedCrystallization
    p=recipe.parameters(density=1.293)
    T=t->303.15
    initial=[2.0*100.0^k for k in 0:4]
    problem=recipe.build_moments(;p,c0=0.24,temperature_profile=T,tspan=(0.0,1.0),initial_moments=initial)
    du=similar(problem.u0);rhs!(du,problem.u0,problem.operator,p,0.0)
    ctx=(;fields=(T=T(0.0),),bulk=(c=0.24,),operator=problem.operator.system.populations.crystals,N=initial)
    G=recipe.growth_velocity(p,0.0,ctx)
    expected=[recipe.primary_rate(p,0.0,ctx)+recipe.secondary_rate(p,0.0,ctx)]
    append!(expected,[k*G*initial[k] for k in 1:4])
    @test population(problem,du,:crystals)≈expected
    @test bulk(problem,du,:c)+p.material_factor*expected[4]≈0 atol=1e-20
    @test p.material_factor≈1.293*0.797e-15
    @test recipe.solubility(303.15)≈0.21619430869662537
    @test recipe.growth_velocity(p,0.0,merge(ctx,(bulk=(c=0.1,),)))<0
    temperature=recipe.linear_temperature([0.,1.,3.],[50.,30.,20.])
    @test temperature(2.0)≈298.15
    @test_throws ArgumentError temperature(4.0)
    @test CEJ2025LNP.debye_wavenumber(0.0,298.15,78.4)==0
    @test inv(CEJ2025LNP.debye_wavenumber(1.0,298.15,78.4))*1e9≈9.612 rtol=1e-3
    for gap in (1e-10,1e-8,1e-3)
        u=CEJ2025LNP.vdw_energy(10e-9,20e-9,gap,1.7e-22)
        @test u<0
        @test u==CEJ2025LNP.vdw_energy(20e-9,10e-9,gap,1.7e-22)
        reference=setprecision(256) do
            a=BigFloat(10e-9);b=BigFloat(20e-9);h=BigFloat(gap)
            z=4a*b/(h*(h+2(a+b)))
            -BigFloat(1.7e-22)/6*(z/2+z/(2*(1+z))-log1p(z))
        end
        @test u≈Float64(reference) rtol=1e-12
    end
    @test_throws ArgumentError CEJ2025LNP.vdw_energy(10e-9,20e-9,0.0,1.7e-22)
end

@testset "Forward error must be small relative to measurement uncertainty" begin
    g=Grid1D([0.0,0.5,1.0])
    op=compile(PBMModel(Death(rate=(x,p,t,ctx)->p.k)),g)
    pb=PBMProblem(op,[1.0,0.0],(0.0,2.0);p=(k=0.4,))
    obs=Observation([1.0,2.0],exp.(-0.4.*[1.0,2.0]),MomentObservable(0);sigma=0.01)
    ex=(Experiment(pb,obs),);mapping=(z,e)->(k=z[1],)
    fine=CalibrationProblem(ex;parameters=mapping,dt=0.01)
    inadequate=CalibrationProblem(ex;parameters=mapping,dt=0.5)
    @test fit_diagnostics(fine,[0.4];refined=inadequate).numerical.prediction_change>0.01
    @test_throws ArgumentError confidence_ellipsoid(fine,[0.4];refined=inadequate)
end
