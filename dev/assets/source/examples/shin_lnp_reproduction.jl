module ShinLNPReproduction
using PopulationBalanceModels
include("papers/cej2025_lnp.jl")
using .CEJ2025LNP

export FIGURE4_MINUTES, figure4_saveat, figure4_problem, run_case
const FIGURE4_MINUTES=(1.0,40.0,100.0,170.0,4320.0)
function figure4_saveat()
    snapshots=60 .*collect(FIGURE4_MINUTES)
    logtimes=exp.(range(log(1e-5),log(last(snapshots));length=300))
    logtimes[end]=last(snapshots) # avoid exp(log(tf)) rounding past the horizon
    sort!(unique(vcat(0.0,logtimes,snapshots)))
end

"""Figure 4 setup: no particles initially, nonzero dissolved lipid, seconds internally."""
function figure4_problem(;grid,kwargs...)
    inputs=(;kwargs...)
    if haskey(inputs,:initial_distribution)
        all(iszero,inputs.initial_distribution) || throw(ArgumentError("Figure 4 starts with zero particles"))
    end
    defaults=(initial_lipid=10.0,FRR=3.0,tau=0.0,tspan=(0.0,4320*60.0))
    options=merge(defaults,inputs,(initial_distribution=zeros(length(grid)),))
    CEJ2025LNP.build(;grid,options...)
end

"""
    run_case(problem; dt, saveat, provenance, output)

Export time histories and number/intensity PSDs for a supplied Shin-model
problem. `provenance` records the input source, figure target, and unresolved
conventions. Exported intensity is a normalized density per meter, not a raw
count per logarithmic bin. This routine does not certify paper reproduction.
"""
function run_case(problem;dt=nothing,saveat,provenance,output,solver=nothing,psd_times=nothing)
    isempty(strip(provenance)) && throw(ArgumentError("record input provenance and figure target"))
    if solver===nothing
        dt===nothing && throw(ArgumentError("supply dt for SSPRK or solver=(problem,times)->solution"))
        solution=simulate(problem;dt,saveat)
    else
        # Adapter may use a stiff SciML solver in a separate working environment.
        solution=solver(problem,collect(saveat))
        length(solution.t)==length(saveat) && all(isapprox.(solution.t,collect(saveat);rtol=1e-12,atol=1e-12)) ||
            throw(ArgumentError("external solver must return the requested output times"))
        all(u->all(isfinite,u),solution.u) || throw(NumericalFailure("external solver returned nonfinite states"))
        for u in solution.u
            N=population(problem,u,:lnp)
            minimum(N)>=-1e-12*max(1.0,maximum(N)) || throw(NumericalFailure("external solver returned negative particle concentrations"))
            bulk(problem,u,:lipid)>=-1e-12 || throw(NumericalFailure("external solver returned negative free lipid"))
        end
    end
    grid=problem.operator.system.populations.lnp.grid
    mkpath(output)
    open(joinpath(output,"history.csv"),"w") do io
        println(io,"time,number,free_lipid,intensity_mean_m,material_element_lipid,supersaturation,nucleation_rate,critical_diameter_m,number_mean_growth_m_s,nucleation_uptake_kg_m3_s,growth_uptake_kg_m3_s,overflow_kg_m3_s,nucleated_lipid,grown_lipid,escaped_lipid,lower_volume_fraction,upper_volume_fraction,negative_number_fraction,negative_volume_fraction,negative_intensity_fraction")
        for (t,u) in zip(solution.t,solution.u)
            N=population(problem,u,:lnp); c=bulk(problem,u,:lipid)
            mean=sum(N)>0 ? paper_intensity_mean(grid,N) : NaN
            w=aqueous_fraction(t,problem.p.FRR,problem.p.tau)
            material=(c+problem.p.rho*π/6*moment(grid,N,3))/(1-w)
            fields=problem.operator.system.fields(problem.p,t)
            state=stateview(problem,u)
            op=problem.operator.system.populations.lnp
            ctx=(;grid,operator=op,N,cell=1,fields,bulk=state.bulk)
            S=c/fields.Cinfinity
            J=nucleation_rate(S,fields.T,problem.p.gamma,problem.p.Vm,problem.p.An)
            Lc=critical_length(S,fields.T,problem.p.gamma,problem.p.Vm)
            term=only(filter(item->item.term isa Growth,op.prepared)).term
            G=sum(N)>0 ? sum(N[i]*term.rate(grid.centers[i],problem.p,t,ctx) for i in eachindex(N))/sum(N) : NaN
            rates=mechanism_rates(op,N,problem.p,t,fields,state.bulk)
            accumulator(name)=hasproperty(state.bulk,name) ? getproperty(state.bulk,name) : NaN
            volumes=grid.centers.^3 .* N; totalvolume=sum(volumes)
            edge_bins=min(2,length(grid))
            lower=totalvolume>0 ? sum(volumes[1:edge_bins])/totalvolume : 0.0
            upper=totalvolume>0 ? sum(volumes[end-edge_bins+1:end])/totalvolume : 0.0
            negative_fraction(weights)=sum(max.(weights,0))>0 ? -sum(min.(weights,0))/sum(max.(weights,0)) : 0.0
            println(io,join((t,sum(N),c,mean,material,S,J,Lc,G,rates.nucleation,rates.growth,rates.overflow,
                accumulator(:nucleated_lipid),accumulator(:grown_lipid),accumulator(:escaped_lipid),lower,upper,
                negative_fraction(N),negative_fraction(volumes),negative_fraction(grid.centers.^6 .*N)),','))
        end
    end
    open(joinpath(output,"psd.csv"),"w") do io
        println(io,"time,diameter_m,number_density_per_m,intensity_density_per_m,intensity_peak_scaled")
        for (t,u) in zip(solution.t,solution.u)
            psd_times===nothing || any(s->isapprox(t,s;rtol=1e-12,atol=1e-12),psd_times) || continue
            N=population(problem,u,:lnp);weights=grid.centers.^6 .*N;total=sum(weights)
            peak=maximum(weights./grid.widths)
            for i in 1:length(grid)
                intensity=total>0 ? weights[i]/(total*grid.widths[i]) : NaN
                println(io,join((t,grid.centers[i],N[i]/grid.widths[i],intensity,peak>0 ? weights[i]/grid.widths[i]/peak : NaN),','))
            end
        end
    end
    write(joinpath(output,"inputs.txt"),provenance*"\nJulia $(VERSION); dt=$(dt); external_solver=$(solver!==nothing); saveat=$(collect(saveat))\n")
    solution
end

# An input script must return (inputs=..., dt=..., saveat=..., provenance=...).
# Its `inputs` named tuple contains explicit keywords for CEJ2025LNP.build.
# Run: julia --project=WORKENV examples/shin_lnp_reproduction.jl /path/to/inputs.jl
if abspath(PROGRAM_FILE)==@__FILE__
    isempty(ARGS) && error("Supply an input .jl file returning (inputs,dt,saveat,provenance). See the LNP manual; no paper inputs are silently substituted.")
    config=Base.include(@__MODULE__,abspath(first(ARGS)))
    problem=figure4_problem(;config.inputs...)
    output=length(ARGS)>1 ? abspath(ARGS[2]) : joinpath(@__DIR__,"..","docs","src","assets","shin_case")
    run_case(problem;dt=get(config,:dt,nothing),saveat=get(config,:saveat,figure4_saveat()),
        provenance=config.provenance,solver=get(config,:solver,nothing),output)
end
end
