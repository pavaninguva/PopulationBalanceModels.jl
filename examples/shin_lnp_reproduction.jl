module ShinLNPReproduction
using PopulationBalanceModels
include("papers/cej2025_lnp.jl")
using .CEJ2025LNP

"""
    run_case(problem; dt, saveat, provenance, output)

Export time histories and number/intensity PSDs for a supplied Shin-model
problem. `provenance` records the input source, figure target, and unresolved
conventions. Exported intensity is a normalized density per meter, not a raw
count per logarithmic bin. This routine does not certify paper reproduction.
"""
function run_case(problem;dt,saveat,provenance,output)
    isempty(strip(provenance)) && throw(ArgumentError("record input provenance and figure target"))
    solution=simulate(problem;dt,saveat)
    grid=problem.operator.system.populations.lnp.grid
    mkpath(output)
    open(joinpath(output,"history.csv"),"w") do io
        println(io,"time,number,free_lipid,intensity_mean_m,material_element_lipid")
        for (t,u) in zip(solution.t,solution.u)
            N=population(problem,u,:lnp); c=bulk(problem,u,:lipid)
            mean=sum(N)>0 ? paper_intensity_mean(grid,N) : NaN
            w=aqueous_fraction(t,problem.p.FRR,problem.p.tau)
            material=(c+problem.p.rho*π/6*moment(grid,N,3))/(1-w)
            println(io,join((t,sum(N),c,mean,material),','))
        end
    end
    open(joinpath(output,"psd.csv"),"w") do io
        println(io,"time,diameter_m,number_density_per_m,intensity_density_per_m")
        for (t,u) in zip(solution.t,solution.u)
            N=population(problem,u,:lnp);weights=grid.centers.^6 .*N;total=sum(weights)
            for i in 1:length(grid)
                intensity=total>0 ? weights[i]/(total*grid.widths[i]) : NaN
                println(io,join((t,grid.centers[i],N[i]/grid.widths[i],intensity),','))
            end
        end
    end
    write(joinpath(output,"inputs.txt"),provenance*"\nJulia $(VERSION); dt=$(dt); saveat=$(collect(saveat))\n")
    solution
end

# An input script must return (inputs=..., dt=..., saveat=..., provenance=...).
# Its `inputs` named tuple contains explicit keywords for CEJ2025LNP.build.
# Run: julia --project=WORKENV examples/shin_lnp_reproduction.jl /path/to/inputs.jl
if abspath(PROGRAM_FILE)==@__FILE__
    isempty(ARGS) && error("Supply an input .jl file returning (inputs,dt,saveat,provenance). See the LNP manual; no paper inputs are silently substituted.")
    config=Base.include(@__MODULE__,abspath(first(ARGS)))
    problem=CEJ2025LNP.build(;config.inputs...)
    output=length(ARGS)>1 ? abspath(ARGS[2]) : joinpath(@__DIR__,"..","docs","src","assets","shin_case")
    run_case(problem;config.dt,config.saveat,config.provenance,output)
end
end
