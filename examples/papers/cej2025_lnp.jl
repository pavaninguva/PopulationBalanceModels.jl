module CEJ2025LNP
using PopulationBalanceModels

export build,loglinear_mole_solubility,critical_length,nucleation_rate,
       brownian_kernel,kelvin_ratio,paper_intensity_mean,aqueous_fraction

const kB=1.380649e-23
loglinear_mole_solubility(xwater,xethanol,w)=exp(w*log(xwater)+(1-w)*log(xethanol))
critical_length(S,T,gamma,Vm)=S>1 ? 4gamma*Vm/(kB*T*log(S)) : Inf
nucleation_rate(S,T,gamma,Vm,An)=S>1 ? An*exp(-16π*gamma^3*Vm^2/(3*(kB*T)^3*log(S)^2)) : 0.0
brownian_kernel(L,M,T,mu)=2kB*T*(L+M)^2/(3mu*L*M)
kelvin_ratio(L,T,gamma,Vm)=exp(4gamma*Vm/(kB*T*L))
function aqueous_fraction(t,FRR,tau)
    FRR>=0 && tau>=0 || throw(ArgumentError("FRR and mixing time must be nonnegative"))
    tau==0 ? FRR/(FRR+1) : FRR/(FRR+1)*(-expm1(-t/tau))
end

"""Eq. 3 in the final paper: M7/M6, kept under an explicit paper-specific name."""
function paper_intensity_mean(grid,N)
    denominator=moment(grid,N,6)
    denominator>0 || throw(ArgumentError("an empty distribution has no mean diameter"))
    moment(grid,N,7)/denominator
end

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
                overflow=:error)
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
    function shape(L,p,t,ctx)
        Lc=critical_length(supersaturation(ctx),ctx.fields.T,p.gamma,p.Vm)
        exp(-0.5*((L-Lc)/p.nucleation_width)^2)
    end
    G=(L,p,t,ctx)->p.kg*L*ctx.fields.Cinfinity*
        (supersaturation(ctx)-equilibrium_ratio(L,ctx.fields.w,ctx.fields.T))
    beta=(L,M,p,t,ctx)->p.kPEG*attachment_without_peg(L,M,ctx.fields.w,ctx.fields.T)*
        brownian_kernel(L,M,ctx.fields.T,ctx.fields.mu)
    geometry=AdditiveCoordinate(L->L^3,cbrt)
    model=PBMModel(
        DistributedNucleation(J,shape;normalization=nucleation_normalization)+
        Growth(G)+Aggregation(beta;coordinate_map=geometry,overflow=overflow)+
        Dilution((L,p,t,ctx)->ctx.fields.dilution))
    op=compile(model,grid;growth_scheme=scheme,bc=SizeBC(NoFlux(),NoFlux()))
    exchange=ConservativeExchange(:lnp,:lipid,(L,p,t,ctx)->p.rho*π/6*L^3;
        mechanisms=(:growth,:nucleation))
    sys=PBMSystem((lnp=op,);fields,exchanges=(exchange,),nonnegative_bulk=(:lipid,),
        bulk_rhs=(ctx,p,t)->(lipid=-ctx.fields.dilution*ctx.bulk.lipid,))
    w0=aqueous_fraction(0.0,FRR,tau)
    PBMProblem(sys,(lnp=initial_distribution,),tspan;bulk0=(lipid=initial_lipid*(1-w0),),p)
end
end
