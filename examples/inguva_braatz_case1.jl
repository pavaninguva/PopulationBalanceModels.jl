module InguvaBraatzCase1
using PopulationBalanceModels

export problem, reference, run_benchmark
const FINAL_TIME=1.0
const SIGMA=0.05
const GAUSSIAN=GaussianNucleation((p,t,ctx)->1.0;center=0.0,width=1.0,truncation=:physical)

# Exact integrals of the paper's Gaussian, truncated to its initial domain.
function axis_integrals(axis,t)
    center=0.4+t
    [begin
        lo=max(axis.edges[i],t); hi=min(axis.edges[i+1],2+t)
        hi<=lo ? 0.0 : SIGMA*sqrt(2π)*GAUSSIAN.fraction(
            (lo-center)/SIGMA,(hi-center)/SIGMA,(;),t,(;))
    end for i in 1:length(axis)]
end
function reference(grid,t)
    50 .* (axis_integrals(grid.x,t)*axis_integrals(grid.y,t)')
end
function problem(n=100;scheme=WENO5(epsilon=1e-12),boundary=:paper)
    grid=Grid2D(range(0,2;length=n+1),range(0,2;length=n+1))
    bc=boundary===:paper ? SizeBC(OpenBoundary(),NoFlux()) :
       boundary===:periodic ? SizeBC(Periodic(),Periodic()) :
       boundary===:open ? SizeBC(OpenBoundary(),OpenBoundary()) :
       throw(ArgumentError("unknown boundary"))
    model=PBMModel(growth=Growth2D((1.0,1.0)))
    operator=compile(model,grid;methods=TermMethods(growth=scheme),bc=SizeBC2D(x=bc,y=bc))
    PBMProblem(operator,reference(grid,0.0),(0.0,FINAL_TIME))
end
function write_field(path,grid,values)
    open(path,"w") do io
        println(io,"a1,a2,density")
        f=density(grid,values)
        for j in axes(f,2),i in axes(f,1)
            println(io,grid.x.centers[i],',',grid.y.centers[j],',',f[i,j])
        end
    end
end
function run_benchmark(;grids=(50,100,200),output=joinpath(@__DIR__,"..","docs","src","assets","inguva_case1"))
    mkpath(output); rows=NamedTuple[]
    for n in grids
        iseven(n) || throw(ArgumentError("even bin count required for final integer shifts"))
        h=2/n
        for (name,scheme,method,dt,boundary) in (
            ("upwind",Upwind(),SSPRK33(),0.2h,:paper),
            ("weno5",WENO5(epsilon=1e-12),SSPRK33(),0.2h,:paper),
            ("compact_adi4",Upwind(),HOCADI(time_order=4),h,:periodic),
            ("exact_shift",Upwind(),ExactShift(),h,:open))
            prob=problem(n;scheme,boundary)
            solution=simulate(prob;method,dt,saveat=[FINAL_TIME],check_nonnegative=false)
            truth=reference(prob.operator.grid,FINAL_TIME)
            f=density(prob.operator.grid,solution.u[end]); exact=density(prob.operator.grid,truth)
            error=f.-exact
            row=(;n,method=name,boundary=String(boundary),dt,steps=solution.steps,
                RMSE=sqrt(sum(abs2,error)/length(error)),MAE=maximum(abs,error),
                L1=sum(abs,solution.u[end].-truth),minimum_density=minimum(f),
                negative_number=-sum(min.(solution.u[end],0.0)),
                relative_number_error=abs(sum(solution.u[end])-sum(prob.u0))/sum(prob.u0))
            push!(rows,row); println(row)
            if n==100 || n==maximum(grids)
                write_field(joinpath(output,"$(name)_$(n).csv"),prob.operator.grid,solution.u[end])
                name=="upwind" && write_field(joinpath(output,"reference_$(n).csv"),prob.operator.grid,truth)
            end
        end
    end
    open(joinpath(output,"errors.csv"),"w") do io
        println(io,join(keys(first(rows)),','))
        for row in rows; println(io,join(values(row),',')); end
    end
    open(joinpath(output,"environment.txt"),"w") do io
        println(io,"PopulationBalanceModels ",Base.pkgversion(PopulationBalanceModels),"; Julia ",VERSION)
        println(io,Sys.KERNEL,' ',Sys.ARCH,"; threads=",Threads.nthreads())
        println(io,"Inguva & Braatz (2023), Section 3.1, Eqs. 55–56, Fig. 2; t=1, domain=[0,2]^2.")
        println(io,"All states and errors use bin integrals/averages; paper's Exact/Upwind used nodal values.")
        println(io,"FV: zero incoming, no outgoing flux. Compact: periodic variant. Shift: paired open.")
        println(io,"Initial Gaussian has negligible boundary tails; boundary variants are declared, not identical.")
    end
    rows
end
if abspath(PROGRAM_FILE)==@__FILE__; run_benchmark(); end
end
