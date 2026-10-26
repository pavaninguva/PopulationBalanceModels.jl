module CompactGrowthBenchmark
using PopulationBalanceModels, LinearAlgebra
include("../compact_growth.jl")

function metrics(sol,reference)
    grid=sol.problem.operator.grid; N=sol.u[end]
    volume=grid isa Grid1D ? reshape(grid.widths,:,1) : grid.x.widths*grid.y.widths'
    error=N-reference
    L1=sum(abs,error)
    rmse=sqrt(sum(error.^2 ./volume)/sum(volume))
    M0=abs(sum(error))
    mx=grid isa Grid1D ? abs(only(moment(grid,error,1))) : abs(moment(grid,error,(1,0)))
    my=grid isa Grid1D ? NaN : abs(moment(grid,error,(0,1)))
    balance=hasproperty(diagnostics(sol),:number_accounting) ?
        diagnostics(sol).number_accounting[end].defect : sum(N)-sum(sol.problem.u0)
    (;L1,rmse,M0,mx,my,balance,minimum=minimum(N./volume),negative=-sum(min.(N,0.)))
end

function measure(io,scenario,label,problem,reference,dt;method=nothing,profile=nothing)
    kwargs=method===nothing ? (;dt,saveat=[last(problem.tspan)],check_nonnegative=false) :
        (;dt,saveat=[last(problem.tspan)],check_nonnegative=false,method)
    # Separate full warm-up; one measured solve. Excludes assembly/export/plots.
    simulate(problem;kwargs...)
    elapsed=@timed simulate(problem;kwargs...)
    sol=elapsed.value; m=metrics(sol,reference)
    grid=problem.operator.grid
    nx=grid isa Grid1D ? length(grid) : size(grid)[1]
    ny=grid isa Grid1D ? 1 : size(grid)[2]
    row=(scenario,label,nx,ny,dt,sol.steps,elapsed.time,elapsed.bytes,
        m.L1,m.rmse,m.M0,m.mx,m.my,m.balance,m.minimum,m.negative,"completed")
    println(io,join(row,','));flush(io)
    println(row)
    if profile!==nothing
        open(profile,"w") do output
            println(output,"x,y,number,reference_number,density,reference_density")
            for j in 1:ny,i in 1:nx
                x=grid isa Grid1D ? grid.centers[i] : grid.x.centers[i]
                y=grid isa Grid1D ? 0. : grid.y.centers[j]
                area=grid isa Grid1D ? grid.widths[i] : grid.x.widths[i]*grid.y.widths[j]
                println(output,join((x,y,sol.u[end][i,j],reference[i,j],sol.u[end][i,j]/area,reference[i,j]/area),','))
            end
        end
    end
    m.L1
end

function source_time_refinement(io)
    # Independent density reconstruction / method-of-lines matrices. Polynomial
    # cell integrals derive the physical outflow row; no package factor is used.
    function matrix_and_inlet(axis,velocity)
        n=length(axis);h=first(axis.widths)
        P=Matrix{Float64}(I,n,n);Q=zeros(n,n);D=Matrix{Float64}(I,n,n)
        for i in 1:n-1
            i>1 && (P[i,i-1]=.25)
            P[i,i+1]=.25;Q[i,i]=Q[i,i+1]=.75;D[i+1,i]=-1
        end
        V=[((-l)^(k+1)-(-l-1)^(k+1))/(k+1) for k in 0:4,l in 0:4]
        Q[n,n:-1:n-4]=V\[1.,0,0,0,0]
        unit=zeros(n);unit[1]=1
        -velocity*D*(P\Q)/h, unit+.25D*(P\unit)
    end
    J=PrescribedRate((p,t,c)->2+t);tf=.2
    g=Grid1D(range(0,2;length=17));A,b=matrix_and_inlet(g,.4)
    for scenario in (:boundary,:cell,:corner)
        if scenario===:corner
            grid=Grid2D(range(0,2;length=7),range(0,3;length=9))
            Ax,_=matrix_and_inlet(grid.x,.4);Ay,_=matrix_and_inlet(grid.y,.7)
            matrix=kron(Matrix{Float64}(I,8,8),Ax)+kron(Ay,Matrix{Float64}(I,6,6))
            source=zeros(48);source[1]=1
            model=PBMModel(Growth2D((.4,.7))+Nucleation2D(J,(0.,0.)))
            bc=SizeBC2D(x=CompactGrowth.open_ends,y=CompactGrowth.open_ends)
            initial=zeros(6,8)
        else
            grid=g;matrix=A
            source=scenario===:boundary ? b : [i==5 ? 1. : 0. for i in 1:16]
            location=scenario===:boundary ? 0. : .54
            model=PBMModel(Growth(AffineGrowth(.4,0))+Nucleation(J,location))
            bc=CompactGrowth.open_ends;initial=zeros(16,1)
        end
        n=length(source);augmented=zeros(n+2,n+2)
        augmented[1:n,1:n]=matrix
        augmented[1:n,n+1]=2source;augmented[1:n,n+2]=source
        augmented[n+2,n+1]=1 # independent exact integration of J(t)=2+t
        reference=reshape((exp(tf*augmented)*vcat(zeros(n),[1.,0.]))[1:n],size(initial))
        op=compile(model,grid;evolution=HOCFiniteVolume(),bc)
        problem=PBMProblem(op,initial,(0.,tf));errors=Float64[]
        for dt in (tf/8,tf/16,tf/32,tf/64)
            push!(errors,measure(io,"temporal_$(scenario)_births","HOCFiniteVolume",problem,reference,dt))
        end
        1.9<log2(errors[end-1]/errors[end])<2.1 || error("source time refinement did not establish order two: $scenario")
    end
end

function record_rejections(output)
    g=Grid1D(range(0,1;length=17));bc=CompactGrowth.open_ends
    policy=HOCFiniteVolume()
    model=PBMModel(Growth(AffineGrowth(.4,0)))
    cases=(
        ("unclassified_callback",ArgumentError,()->compile(PBMModel(Growth((x,p,t,c)->.4)),g;evolution=policy,bc)),
        ("nonuniform_mesh",ArgumentError,()->compile(model,Grid1D([0.,.1,.2,.4,.6,.8,1.]);evolution=policy,bc)),
        ("closed_outflow",ArgumentError,()->compile(model,g;evolution=policy)),
        ("fourth_order_sources",ArgumentError,()->HOCFiniteVolume(time_order=4)),
        ("boundary_front_nonnegativity",NumericalFailure,()->simulate(CompactGrowth.nucleating_population();dt=.001)),
        ("runtime_vanishing_growth",NumericalFailure,()->begin
            law=TimeScaledGrowth(AffineGrowth(1.,0),PrescribedRate((p,t,c)->1-10t))
            op=compile(PBMModel(Growth(law)),g;evolution=policy,bc)
            simulate(PBMProblem(op,zeros(16),(0.,.2));dt=.02)
        end))
    open(joinpath(output,"rejections.csv"),"w") do io
        println(io,"configuration,error_type,reason")
        for (label,expected,run_case) in cases
            try
                run_case()
            catch err
                err isa expected || rethrow()
                reason="\""*replace(sprint(showerror,err),"\""=>"\"\"")*"\""
                println(io,join((label,typeof(err),reason),','))
                continue
            end
            error("expected rejection did not occur: $label")
        end
    end
end

function unit_rescaling_check(output)
    function solution(size_unit,time_unit)
        q=collect(range(0,1;length=41));g=Grid1D(size_unit.*q)
        initial=diff(2q-.3cos.(2pi*q)/(2pi))
        model=PBMModel(Growth(AffineGrowth(.4size_unit/time_unit,.1/time_unit))+
            Nucleation(.2/time_unit,.313size_unit))
        op=compile(model,g;evolution=HOCFiniteVolume(),bc=SizeBC(InflowFlux(.8/time_unit),OpenBoundary()))
        simulate(PBMProblem(op,initial,(0.,.2time_unit));dt=.001time_unit)
    end
    base=solution(1.,1.);reference=base.u[end]
    ledger=diagnostics(base).number_accounting[end]
    open(joinpath(output,"unit_rescaling.csv"),"w") do io
        println(io,"size_unit,time_unit,max_bin_difference,max_number_accounting_difference")
        for (size_unit,time_unit) in ((1e-9,1.),(1.,60.),(1e3,1e-3))
            sol=solution(size_unit,time_unit);a=diagnostics(sol).number_accounting[end]
            delta=maximum(abs,sol.u[end]-reference)
            accounting=max(abs(a.births-ledger.births),maximum(abs.(a.inflow.-ledger.inflow)),maximum(abs.(a.outflow.-ledger.outflow)))
            max(delta,accounting)<128eps(Float64)*sum(base.problem.u0) || error("HOC coordinate/time unit covariance failed")
            println(io,join((size_unit,time_unit,delta,accounting),','))
        end
    end
end

function run(;output=joinpath(@__DIR__,"..","..","docs","src","assets","compact_fv"))
    mkpath(output)
    open(joinpath(output,"results.csv"),"w") do io
        println(io,"scenario,method,nx,ny,dt,steps,seconds,bytes,L1,RMSE,M0_error,M1x_error,M1y_error,number_balance_error,minimum_density,negative_number,status")
        # Spatial tests use a fixed small dt, independent of mesh refinement.
        for family in (:affine,:exponential)
            for n in (40,80,160)
                case=CompactGrowth.variable_growth(n;family)
                # The plotting script derives measured slopes from these rows.
                measure(io,String(family),"HOCFiniteVolume",case.problem,case.reference(.4),2e-5)
            end
        end
        for n in (20,40,80,160)
            case=CompactGrowth.smooth_1d(n);g=case.problem.operator.grid
            for scheme in (:HOC,:WENO3,:WENO5,:Inguva)
                label=scheme===:HOC ? "HOCFiniteVolume" : scheme===:Inguva ? "Inguva2022" : String(scheme)*"_RK4"
                op=if scheme===:HOC
                    case.problem.operator
                elseif scheme===:Inguva
                    compile(case.problem.operator.model,g;evolution=Inguva2022(),bc=CompactGrowth.periodic)
                else
                    compile(case.problem.operator.model,g;growth_scheme=scheme===:WENO3 ? WENO3() : WENO5(),bc=CompactGrowth.periodic)
                end
                problem=PBMProblem(op,case.problem.u0,case.problem.tspan)
                dt=scheme===:HOC ? .2/(.4n^2) : scheme===:Inguva ? 1/(.4n) : .25/(.4n)
                measure(io,"work_precision",label,problem,case.reference(.5),dt;method=scheme in (:WENO3,:WENO5) ? RK4() : nothing)
            end
        end
        for n in (24,48,96)
            case=CompactGrowth.smooth_2d(n,div(3n,4))
            measure(io,"rectangular_translation","HOCFiniteVolume",case.problem,case.reference(.25),2e-5)
            measure(io,"rectangular_work","HOCFiniteVolume",case.problem,case.reference(.25),.2/(.7n^2))
            g=case.problem.operator.grid
            op=compile(case.problem.operator.model,g;growth_scheme=WENO5(),bc=case.problem.operator.bc)
            measure(io,"rectangular_work","WENO5_RK4",PBMProblem(op,case.problem.u0,case.problem.tspan),case.reference(.25),.1min(first(g.x.widths)/.4,first(g.y.widths)/.7);method=RK4())
            legacy=compile(case.problem.operator.model,g;evolution=HOCADI(time_order=4),bc=case.problem.operator.bc)
            measure(io,"rectangular_work","HOCADI4",PBMProblem(legacy,case.problem.u0,case.problem.tspan),case.reference(.25),.4min(first(g.x.widths)/.4,first(g.y.widths)/.7))
            variable=CompactGrowth.variable_2d(n,div(3n,4))
            measure(io,"rectangular_variable","HOCFiniteVolume",variable.problem,variable.reference(.25),2e-5;
                profile=n==48 ? joinpath(output,"variable_2d.csv") : nothing)
        end
        variable=CompactGrowth.variable_2d(192,144)
        measure(io,"rectangular_variable","HOCFiniteVolume",variable.problem,variable.reference(.25),2e-5)
        for n in (20,40,80)
            case=CompactGrowth.incoming_population(n)
            measure(io,"physical_inflow","HOCFiniteVolume",case.problem,case.reference(.4),2e-5)
        end
        for n in (40,80,160)
            grid=Grid1D(range(0,2;length=n+1));tf=.4
            primitive(x)=.12sqrt(pi)/2*CompactGrowth.erf((x-1.85)/.12)
            initial=diff(primitive.(grid.edges))
            reference=reshape(diff(primitive.(max.(0.,grid.edges.-.4tf))),:,1)
            op=compile(PBMModel(Growth(AffineGrowth(.4,0))),grid;
                evolution=HOCFiniteVolume(),bc=CompactGrowth.open_ends)
            measure(io,"upper_escape","HOCFiniteVolume",PBMProblem(op,initial,(0.,tf)),reference,2e-5)
        end
        # f(x,t)=exp(-s*t), G=b+s*x has no spatial truncation error: the
        # compact faces reproduce a constant density and an affine flux.
        # This isolates temporal error against a continuum (not matrix) reference.
        grid=Grid1D(range(0,2;length=25));b=.4;s=.15;tf=.4
        flux=PrescribedRate((p,t,c)->b*exp(-s*t))
        op=compile(PBMModel(Growth(AffineGrowth(b,s))),grid;
            evolution=HOCFiniteVolume(),bc=SizeBC(InflowFlux(flux),OpenBoundary()))
        problem=PBMProblem(op,grid.widths,(0.,tf));errors=Float64[]
        for dt in (tf/8,tf/16,tf/32,tf/64)
            push!(errors,measure(io,"temporal_affine_inflow","HOCFiniteVolume",problem,reshape(exp(-s*tf).*grid.widths,:,1),dt))
        end
        1.9<log2(errors[end-1]/errors[end])<2.1 || error("affine inflow temporal order failed")
        # Fixed-grid, independent semidiscrete matrix exponential: temporal error.
        n=32;h=1/n;M=Matrix{Float64}(I,n,n);D=zeros(n,n)
        for i in 1:n
            M[i,mod1(i-1,n)]=M[i,mod1(i+1,n)]=.25
            D[i,mod1(i-1,n)]=-3/(4h);D[i,mod1(i+1,n)]=3/(4h)
        end
        for time_dependent in (false,true)
            case=CompactGrowth.smooth_1d(n;time_dependent)
            tf=.5; travel=.4*(time_dependent ? tf+tf^3/3 : tf)
            ref=reshape(exp(-travel*(M\D))*vec(case.problem.u0),:,1)
            for dt in (.5/8,.5/16,.5/32,.5/64)
                measure(io,time_dependent ? "temporal_prescribed" : "temporal_constant","HOCFiniteVolume",case.problem,ref,dt)
            end
        end
        case=CompactGrowth.smooth_2d(24,18);g=case.problem.operator.grid
        function derivative_matrix(axis)
            n=length(axis);h=first(axis.widths);M=Matrix{Float64}(I,n,n);D=zeros(n,n)
            for i in 1:n
                M[i,mod1(i-1,n)]=M[i,mod1(i+1,n)]=.25
                D[i,mod1(i-1,n)]=-3/(4h);D[i,mod1(i+1,n)]=3/(4h)
            end
            M\D
        end
        ref=exp(-.4*.25*derivative_matrix(g.x))*case.problem.u0*exp(-.7*.25*derivative_matrix(g.y))'
        for dt in (.25/4,.25/8,.25/16,.25/32)
            measure(io,"temporal_rectangular","HOCFiniteVolume",case.problem,ref,dt)
        end
        for n in (80,160)
            g=Grid1D(range(0,1;length=n+1));tf=.5
            initial=[max(0.,min(b,.4)-max(a,.2)) for (a,b) in zip(g.edges[1:end-1],g.edges[2:end])]
            ref=reshape([max(0.,min(b,.6)-max(a,.4)) for (a,b) in zip(g.edges[1:end-1],g.edges[2:end])],:,1)
            for (label,evolution,reconstruction) in (("HOCFiniteVolume",HOCFiniteVolume(),Upwind()),("WENO5_RK4",nothing,WENO5()),("Inguva2022",Inguva2022(),Upwind()))
                op=compile(PBMModel(Growth(AffineGrowth(.4,0))),g;bc=CompactGrowth.periodic,evolution,growth_scheme=reconstruction)
                measure(io,"sharp",label,PBMProblem(op,initial,(0.,tf)),ref,
                    label=="Inguva2022" ? 1/(.4n) : .4/(.4n);method=evolution===nothing ? RK4() : nothing)
            end
            problem=CompactGrowth.nucleating_population(n)
            measure(io,"boundary_births","HOCFiniteVolume",problem,CompactGrowth.nucleation_reference(problem,.5),.0002;
                profile=n==160 ? joinpath(output,"boundary_births.csv") : nothing)
        end
        source_time_refinement(io)
    end
    record_rejections(output)
    unit_rescaling_check(output)
    open(joinpath(output,"environment.txt"),"w") do io
        println(io,"Starting revision: 1e644a184f39c36ca81a9759d9b42a456e69e456; HOC working-tree implementation")
        println(io,"Julia ",VERSION,"; ",Sys.KERNEL," ",Sys.ARCH)
        println(io,"CPU: ",Sys.cpu_info()[1].model,"; Julia threads: ",Threads.nthreads(),"; BLAS threads: ",BLAS.get_num_threads())
        println(io,"One measured solve after separate full warm-up; excludes compilation, model assembly, exports and plotting.")
        println(io,"Errors compare bin integrals, no clipping; RMSE is volume-weighted density RMSE.")
        println(io,"Temporal rows use independent fixed-grid matrix exponentials, except temporal_affine_inflow (an exact spatially constant continuum solution).")
    end
end

if abspath(PROGRAM_FILE)==@__FILE__
    run()
end
end
