# tutorial:start model
module CEJ2025LNP
using PopulationBalanceModels
using LinearAlgebra

export build,loglinear_mole_solubility,critical_length,nucleation_rate,
       brownian_kernel,kelvin_ratio,paper_intensity_mean,aqueous_fraction,
       debye_wavenumber,vdw_energy,dlvo_attachment,dlvo_quadrature,mechanism_rates

include("../support/lnp_material.jl")

"""
Main-paper LNP mechanisms, coupled depletion, prescribed mixing and DLS output.
All lengths are meters; time is seconds; concentrations are kg/m^3.

Required closures/inputs prevent hidden assumptions:
  solubility(w,T): C_infinity in kg/m^3 (not mole fraction).
  equilibrium_ratio(L,w,T): S_star for growth/dissolution.
  attachment_without_peg(L,M,w,T): dimensionless kernel modifier excluding kPEG.
  viscosity(w,T): Pa s; temperature(t): kelvin.
  molecular_volume: per molecule in m^3; particle_density: kg/m^3.
  nucleation_width: meters, separate from gamma in N/m.
  nucleation_normalization: :total_rate or :density_amplitude.

The paper's printed death sign, alpha formula, Gaussian normalization and SI
energy convention need reconciliation with author code. This recipe applies
physical parent loss, uses a user-supplied attachment law and makes the source
normalization explicit. It does not assert reproduction of published figures.
"""
function build(;grid,initial_lipid,FRR,tau,temperature,viscosity,solubility,
                equilibrium_ratio,attachment_without_peg,molecular_volume,
                particle_density,nucleation_width,nucleation_normalization,
                tspan,scheme=MUSCL(),initial_distribution=zeros(length(grid)),
                An=2.0e22,kg=11.88,kPEG=6.5e-6,gamma=0.010,
                overflow=:error,track_consumption=false)
    grid.coordinate in (:diameter,:length) || throw(ArgumentError("CEJ recipe requires a length/diameter grid in meters"))
    all(x->isfinite(x) && x>0,(initial_lipid,molecular_volume,particle_density,nucleation_width,gamma)) ||
        throw(ArgumentError("concentration and physical input scales must be positive"))
    first(tspan)==0 || throw(ArgumentError("the mixing recipe starts at t=0; supply a custom system for later initialization"))
    p=(An=An,kg=kg,kPEG=kPEG,gamma=gamma,FRR=FRR,tau=tau,
       Vm=molecular_volume,rho=particle_density,nucleation_width=nucleation_width)
    function fields(p,t)
        w=aqueous_fraction(t,p.FRR,p.tau)
        T=temperature(t); mu=viscosity(w,T); C=solubility(w,T)
        all(x->isfinite(x) && x>0,(T,mu,C)) || throw(ArgumentError("invalid temperature, viscosity or solubility"))
        dw=p.tau==0 ? 0.0 : p.FRR/(p.FRR+1)*exp(-t/p.tau)/p.tau
        # Volume growth of a material element: V/Vorganic=1/(1-w).
        (;w,T,mu,Cinfinity=C,dilution=dw/(1-w))
    end
    supersaturation(ctx)=ctx.bulk.lipid/ctx.fields.Cinfinity
    function J(p,t,ctx)
        nucleation_rate(supersaturation(ctx),ctx.fields.T,p.gamma,p.Vm,p.An)
    end
    center=(p,t,ctx)->critical_length(supersaturation(ctx),ctx.fields.T,p.gamma,p.Vm)
    # At S<=1, J is zero and the source residual returns before evaluating its
    # center. Use a finite unused center so the integral itself remains defined.
    finite_center=(p,t,ctx)->supersaturation(ctx)>1 ? center(p,t,ctx) : first(grid.centers)
    gaussian=GaussianNucleation(J;center=finite_center,
        width=(p,t,ctx)->p.nucleation_width,truncation=:renormalize)
    nucleation_normalization in (:total_rate,:density_amplitude) ||
        throw(ArgumentError("invalid nucleation normalization"))
    # Preserve the legacy unnormalized exp(-z²/2) amplitude convention explicitly.
    physical=GaussianNucleation(J;center=finite_center,
        width=(p,t,ctx)->p.nucleation_width,truncation=:physical)
    source=nucleation_normalization===:total_rate ? gaussian :
        IntegratedNucleation(J,(lo,hi,p,t,ctx)->sqrt(2π)*p.nucleation_width*
            physical.fraction(lo,hi,p,t,ctx);normalization=:density_amplitude)
    function growth_rate(L,p,t,ctx)
        p.kg*L*ctx.fields.Cinfinity*
            (supersaturation(ctx)-equilibrium_ratio(L,ctx.fields.w,ctx.fields.T))
    end
    function coalescence_kernel(L,M,p,t,ctx)
        p.kPEG*attachment_without_peg(L,M,ctx.fields.w,ctx.fields.T)*
            brownian_kernel(L,M,ctx.fields.T,ctx.fields.mu)
    end
    geometry=AdditiveCoordinate(L->L^3,cbrt)
    model=PBMModel(
        nucleation=source,
        growth=Growth(rate=growth_rate),
        coalescence=Aggregation(coalescence_kernel;coordinate_map=geometry,overflow=overflow),
        dilution=Dilution(rate=(L,p,t,ctx)->ctx.fields.dilution))
    methods=TermMethods(growth=scheme,aggregation=FixedPivot())
    op=compile(model,grid;methods,bc=SizeBC(NoFlux(),NoFlux()))
    exchange=ConservativeExchange(:lnp,:lipid,(L,p,t,ctx)->p.rho*π/6*L^3;
        mechanisms=(:growth,:nucleation))
    function balances(ctx,p,t)
        lipid=-ctx.fields.dilution*ctx.bulk.lipid
        track_consumption || return (;lipid)
        rates=mechanism_rates(op,ctx.populations.lnp,p,t,ctx.fields,ctx.bulk)
        (;lipid,nucleated_lipid=rates.nucleation-ctx.fields.dilution*ctx.bulk.nucleated_lipid,
          grown_lipid=rates.growth-ctx.fields.dilution*ctx.bulk.grown_lipid,
          escaped_lipid=rates.overflow-ctx.fields.dilution*ctx.bulk.escaped_lipid)
    end
    sys=PBMSystem((lnp=op,);fields,exchanges=(exchange,),nonnegative_bulk=(:lipid,),bulk_rhs=balances)
    w0=aqueous_fraction(0.0,FRR,tau)
    initial=(lipid=initial_lipid*(1-w0),)
    bulk0=track_consumption ? merge(initial,(nucleated_lipid=0.0,grown_lipid=0.0,escaped_lipid=0.0)) : initial
    PBMProblem(sys,(lnp=initial_distribution,),tspan;bulk0,p)
end


end
# tutorial:end model
