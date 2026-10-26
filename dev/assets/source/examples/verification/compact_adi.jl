module CompactADIAudit
using PopulationBalanceModels, LinearAlgebra
include("../two_coordinate_transport.jl")

function amplification(theta,cfl;order=4)
    m=1+0.5cos(theta); z=0.75cfl*sin(theta)
    cayley(c)=(m-im*c*z)/(m+im*c*z)
    gamma=1/(2-2^(1/3))
    order==2 ? cayley(1.0) : cayley(gamma)^2*cayley(1-2gamma)
end
function run_audit(;output=joinpath(@__DIR__,"..","..",".cache","legacy_docs_assets","compact_audit"))
    mkpath(output)
    # Every resolved Fourier mode, either speed sign, small to large CFL.
    gain=maximum(abs(amplification(2π*k/64,c;order)) for k in 0:63,c in (-10.0,-1.0,-0.1,0.1,1.0,10.0),order in (2,4))
    open(joinpath(output,"audit.csv"),"w") do io
        println(io,"profile,order,CFL,periods,relative_l2_change,relative_number_error,minimum_density,negative_number")
        for profile in (:smooth,:rectangle),order in (2,4),cfl in (0.25,1.0,4.0)
            prob=PBM2DBenchmark.problem(32;profile)
            # Both axes have a common travel period of 2 time units.
            prob=remake(prob;tspan=(0.0,20.0))
            sol=simulate(prob;method=HOCADI(time_order=order),dt=cfl*2/32,
                saveat=[20.0],check_nonnegative=false)
            f=density(prob.operator.grid,sol.u[end]); initial=density(prob.operator.grid,prob.u0)
            row=(profile,order,cfl,10,norm(f)/norm(initial)-1,
                abs(sum(sol.u[end])-sum(prob.u0))/sum(prob.u0),minimum(f),-sum(min.(sol.u[end],0.0)))
            println(io,join(row,',')); println(row)
        end
    end
    open(joinpath(output,"environment.txt"),"w") do io
        println(io,"PopulationBalanceModels ",Base.pkgversion(PopulationBalanceModels),"; Julia ",VERSION)
        println(io,Sys.KERNEL,' ',Sys.ARCH,"; n=32; t=20; ten periods; both time orders.")
        println(io,"Courant numbers 0.25,1,4; negativity checks disabled, no clipping.")
        println(io,"Maximum sampled Fourier amplification magnitude: ",gain)
    end
    println("Maximum Fourier amplification magnitude: ",gain)
    gain
end
if abspath(PROGRAM_FILE)==@__FILE__;run_audit();end
end
