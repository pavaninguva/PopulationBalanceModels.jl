"""
Formation and incubation with explicitly reconstructed material inputs.
Run in an environment containing the local package, OrdinaryDiffEqRosenbrock
and ADTypes. No parameters are fitted to Figure 4 diameter curves.

    julia --project=WORKENV examples/shin_lnp_long_time.jl [bins=80] [reltol=1e-7] [output] [FRRs=1,3,5]
"""
module ShinLNPLongTime
using PopulationBalanceModels, LinearAlgebra
using OrdinaryDiffEqRosenbrock, ADTypes
import SciMLBase
include("shin_lnp_reproduction.jl")
using .ShinLNPReproduction
const material=ShinLNPReproduction.CEJ2025LNP

"""Mass-solubility interpolation inferred from two approximate S values reported in the text, not independent solubility measurements."""
function reconstructed_solubility(w,T)
    # FRR=3, organic=7 kg/m³, S≈240; FRR=5, organic=10 kg/m³, S≈600.
    c3=(7/4)/240; c5=(10/6)/600
    exp(log(c3)+(w-3/4)/(5/6-3/4)*log(c5/c3))
end

function make_problem(;bins=120,FRR=3.0,scheme=Upwind(),diameter_limits=(1e-9,5e-6),equilibrium=:kelvin)
    grid=Grid1D(exp.(range(log(first(diameter_limits)),log(last(diameter_limits));length=bins+1));coordinate=:diameter)
    equilibrium in (:kelvin,:flat) || throw(ArgumentError("equilibrium must be :kelvin or :flat"))
    w=FRR/(1+FRR); T=298.15
    # These mixture-property correlations and Vm/rho are assumptions. The SI
    # supplies Hamaker and surface potential, but not a complete property table.
    permittivity=24.3*(1-w)+78.4*w
    mu=1.2e-3*(1-w)+0.89e-3*w
    table=Matrix{Float64}(undef,bins,bins)
    quadrature=material.dlvo_quadrature(128)
    for j in 1:bins,i in 1:j
        value=material.dlvo_attachment(grid.centers[i]/2,grid.centers[j]/2;
            temperature=T,ionic_strength=100w,relative_permittivity=permittivity,quadrature)
        table[i,j]=table[j,i]=value
    end
    println("DLVO table ready; range=",extrema(table));flush(stdout)
    indices=Dict(x=>i for (i,x) in enumerate(grid.centers))
    attachment=(L,M,w,T)->table[indices[L],indices[M]]
    figure4_problem(;grid,FRR,temperature=t->T,viscosity=(w,T)->mu,
        solubility=reconstructed_solubility,
        equilibrium_ratio=(L,w,T)->equilibrium===:kelvin ? material.kelvin_ratio(L,T,0.010,5e-28) : 1.0,
        attachment_without_peg=attachment,molecular_volume=5e-28,particle_density=1000.0,
        nucleation_width=1e-9,nucleation_normalization=:total_rate,
        overflow=:discard,track_consumption=true,scheme)
end

"""Scale particle counts and bulk concentrations before stiff integration; return physical states without clipping."""
function stiff_solve(problem,times;reltol=1e-7,max_evaluations=2_000_000)
    max_evaluations isa Integer && max_evaluations>0 || throw(ArgumentError("max_evaluations must be positive"))
    concentration=bulk(problem,problem.u0,:lipid)
    number_scale=concentration/(problem.p.rho*π/6*(100e-9)^3)
    scales=fill(concentration,length(problem.u0))
    scales[problem.operator.ranges.lnp].=number_scale
    evaluations=Ref(0)
    function scaled_rhs!(dv,v,p,t)
        rhs!(dv,v.*scales,problem.operator,p,t)
        dv./=scales
        evaluations[]+=1
        evaluations[]<=max_evaluations || throw(NumericalFailure("LNP RHS evaluation budget exceeded; inspect method, tolerances and state scales"))
        if evaluations[]%100000==0
            println("RHS evaluations=",evaluations[],"; t=",t," s");flush(stdout)
        end
        nothing
    end
    scaled=SciMLBase.ODEProblem(scaled_rhs!,problem.u0./scales,problem.tspan,problem.p)
    println("Starting scaled stiff integration");flush(stdout)
    result=SciMLBase.solve(scaled,Rodas5P(autodiff=AutoFiniteDiff());saveat=times,
        save_everystep=false,reltol,abstol=reltol*1e-3,maxiters=10^7)
    SciMLBase.successful_retcode(result) || error("LNP integration failed: $(result.retcode)")
    println("solver statistics: ",result.destats)
    (;t=result.t,u=[v.*scales for v in result.u],stats=result.destats)
end

const provenance="""
Shin et al., Chem Eng J (2025), DOI 10.1016/j.cej.2025.167786, and 2-page SI.
Targets: Figure 4 model panels (FRR=3), Figure 8 (FRR=1,3,5); no experimental data.
Zero particles; organic lipid 10 kg/m³; instantaneous mixing; 4320 min = 259200 s.
Paper/SI inputs: An=2e22, kg=11.88, kPEG=6.5e-6, gamma=.010 N/m,
Hamaker=1.7e-22 J, potential=.010 V, aqueous buffer .1 mol/L.
RECONSTRUCTION ASSUMPTIONS: T=298.15 K; Vm=5e-28 m³/molecule; rho=1000 kg/m³;
nucleation sigma=1 nm; total-rate Gaussian with renormalized domain truncation;
Kelvin equilibrium ratio; linearly mixed permittivity (ethanol 24.3/water 78.4)
and viscosity (1.2/.89 mPa s); buffer ionic strength=100*w mol/m³.
Solubility is a log-linear mass correlation inferred from reported approximate
initial supersaturations S≈240 (FRR3, organic7) and S≈600 (FRR5, organic10).
It is not an independent material-property measurement or the paper's fitted
mole-fraction correlation. DLVO is the specified constant-potential Derjaguin
reconstruction, inverse no-hydrodynamic Fuchs integral, not a verified literal
transcription of the SI electrostatic prefactor / main-paper attachment formula.
No calibration to diameter curves. Upwind growth, fixed-pivot volume addition,
closed size boundaries; discarded collision volume integrated as escaped_lipid.
These results test a reconstructed model; they do not establish reproduction.
"""

function run(;bins=80,reltol=1e-7,FRRs=(1.,3.,5.),output=joinpath(@__DIR__,"..","docs","src","assets","shin_case"))
    isempty(FRRs) && throw(ArgumentError("supply at least one flow-rate ratio"))
    times=sort!(unique(vcat(figure4_saveat(),collect(range(0,0.15;length=151)))))
    for FRR in FRRs
        println("Building FRR=$FRR, bins=$bins");flush(stdout)
        problem=make_problem(;bins,FRR)
        label=isinteger(FRR) ? string(Int(FRR)) : replace(string(FRR),'.'=>'p')
        folder=joinpath(output,"frr$label")
        elapsed=@elapsed solution=run_case(problem;saveat=times,
            solver=(p,t)->stiff_solve(p,t;reltol),provenance=provenance*"\nFRR=$FRR; bins=$bins; reltol=$reltol\n",
            psd_times=vcat(0.,.15,60 .*collect(FIGURE4_MINUTES)),output=folder)
        println("Finished FRR=$FRR in $elapsed s; final mean nm=",1e9*material.paper_intensity_mean(
            problem.operator.system.populations.lnp.grid,population(problem,solution.u[end],:lnp)))
        flush(stdout)
    end
end
if abspath(PROGRAM_FILE)==@__FILE__
    run(;bins=isempty(ARGS) ? 80 : parse(Int,ARGS[1]),
        reltol=length(ARGS)<2 ? 1e-7 : parse(Float64,ARGS[2]),
        FRRs=length(ARGS)<4 ? (1.,3.,5.) : parse.(Float64,split(ARGS[4],',')),
        output=length(ARGS)<3 ? joinpath(@__DIR__,"..","docs","src","assets","shin_case") : abspath(ARGS[3]))
end
end
