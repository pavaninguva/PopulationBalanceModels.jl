# Advanced reconstructed Shin LNP model: SI units, one fixed FRR=3 mixture.
# The short startup below is not a reproduction of the long-time figures.
using PopulationBalanceModels, LinearAlgebra
include(joinpath(@__DIR__,"..","support","lnp_material.jl"))
include(joinpath(@__DIR__,"..","support","lnp_attachment.jl"))

p = (An=2e22,kg=11.88,kPEG=6.5e-6,gamma=0.010,FRR=3.0,tau=0.0,
     Vm=5e-28,rho=1000.0,nucleation_width=1e-9)
T = 298.15
w = p.FRR/(1+p.FRR)
mu = 1.2e-3*(1-w)+0.89e-3*w
permittivity = 24.3*(1-w)+78.4*w
# Reconstructed mass solubility, inferred from approximate reported S values;
# this is not an independently measured property or a fit to diameter curves.
c3,c5 = (7/4)/240,(10/6)/600
solubility(w,T) = exp(log(c3)+(w-3/4)/(5/6-3/4)*log(c5/c3))
fields(p,t) = (w=w,T=T,mu=mu,Cinfinity=solubility(w,T),dilution=0.0)
supersaturation(ctx) = ctx.bulk.lipid/ctx.fields.Cinfinity
J(p,t,ctx) = nucleation_rate(supersaturation(ctx),ctx.fields.T,p.gamma,p.Vm,p.An)
critical_size(p,t,ctx) = supersaturation(ctx)>1 ?
    critical_length(supersaturation(ctx),ctx.fields.T,p.gamma,p.Vm) : first(grid.centers)
growth(L,p,t,ctx) = p.kg*L*ctx.fields.Cinfinity*(supersaturation(ctx)-
    kelvin_ratio(L,ctx.fields.T,p.gamma,p.Vm))
coalescence(L,M,p,t,ctx) = p.kPEG*attachment(L,M,ctx.fields.w,ctx.fields.T)*
    brownian_kernel(L,M,ctx.fields.T,ctx.fields.mu)
source = GaussianNucleation(J;center=critical_size,width=(p,t,ctx)->p.nucleation_width,
    truncation=:renormalize) # J is the TOTAL represented number birth rate.
model = PBMModel(nucleation=source,growth=Growth(growth),
    coalescence=Aggregation(coalescence;coordinate_map=AdditiveCoordinate(L->L^3,cbrt),overflow=:discard),
    dilution=Dilution((L,p,t,ctx)->ctx.fields.dilution))
grid = Grid1D(exp.(range(log(1e-9),log(5e-6);length=81));coordinate=:diameter)
attachment = prepare_attachment(grid;water_fraction=w,temperature=T,relative_permittivity=permittivity)
operator = compile(model,grid;methods=TermMethods(growth=Upwind(),aggregation=FixedPivot()),
    bc=SizeBC(NoFlux(),NoFlux()))
exchange = ConservativeExchange(:lnp,:lipid,(L,p,t,ctx)->p.rho*pi/6*L^3;
    mechanisms=(:growth,:nucleation))
# Keep the production material diagnostics, including discarded collision mass.
function balances(ctx,p,t)
    rates = mechanism_rates(operator,ctx.populations.lnp,p,t,ctx.fields,ctx.bulk)
    d = ctx.fields.dilution
    (lipid=-d*ctx.bulk.lipid,
     nucleated_lipid=rates.nucleation-d*ctx.bulk.nucleated_lipid,
     grown_lipid=rates.growth-d*ctx.bulk.grown_lipid,
     escaped_lipid=rates.overflow-d*ctx.bulk.escaped_lipid)
end
system = PBMSystem((lnp=operator,);fields,exchanges=(exchange,),
    nonnegative_bulk=(:lipid,),bulk_rhs=balances)
problem = PBMProblem(system,(lnp=zeros(length(grid)),),(0.0,1e-5);p,
    bulk0=(lipid=10.0*(1-w),nucleated_lipid=0.0,grown_lipid=0.0,escaped_lipid=0.0))

if abspath(PROGRAM_FILE) == @__FILE__
    solution = simulate(problem;method=SSPRK33(),dt=1e-8,saveat=[0.0,1e-5])
    @show observe(BulkObservable(:lipid),solution)
    # Optional: import PythonPlot
    # plot_observable(solution,BulkObservable(:lipid);xlabel="Time (s)",ylabel="Dissolved lipid (kg/m^3)")
end
# Full-horizon stiff execution remains a separate, qualified workflow. Its
# scaling/completion checks are in lnp_stiff_execution.jl; do not extend this
# tiny native step to 259200 s and call it a verified long-time reproduction.
