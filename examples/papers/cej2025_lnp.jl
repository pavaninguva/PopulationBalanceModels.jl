module CEJ2025LNP
using PopulationBalanceModels
using LinearAlgebra

export build,loglinear_mole_solubility,critical_length,nucleation_rate,
       brownian_kernel,kelvin_ratio,paper_intensity_mean,aqueous_fraction,
       debye_wavenumber,vdw_energy,dlvo_attachment,dlvo_quadrature,mechanism_rates

const kB=1.380649e-23
const NA=6.02214076e23
const elementary_charge=1.602176634e-19
const vacuum_permittivity=8.8541878128e-12

"""Inverse Debye length (m⁻¹) for ionic strength I in mol/m³, kelvin, and dimensionless relative permittivity."""
function debye_wavenumber(I,T,relative_permittivity)
    isfinite(I) && I>=0 && all(x->isfinite(x) && x>0,(T,relative_permittivity)) ||
        throw(ArgumentError("invalid ionic strength, temperature or permittivity"))
    sqrt(2elementary_charge^2*NA*I/(vacuum_permittivity*relative_permittivity*kB*T))
end

"""
    dlvo_attachment(r1,r2; temperature, ionic_strength, relative_permittivity,
                    potential=0.010, hamaker=1.7e-22, order=128)

Dimensionless inverse Fuchs stability integral, without hydrodynamic correction
or PEG. Uses a constant-potential Derjaguin approximation in SI units. This is
an explicitly specified reconstruction, not a transcription of the ambiguous
electrostatic prefactor printed in the Shin supplement. `ionic_strength` is
mol/m³, radii are meters, potential is volts. No contact cutoff is introduced.
"""
function dlvo_quadrature(order)
    order isa Integer && order>=8 || throw(ArgumentError("quadrature order must be at least eight"))
    pair=eigen(SymTridiagonal(zeros(order),[j/sqrt(4j^2-1) for j in 1:order-1]))
    (;nodes=(pair.values.+1)./2,weights=pair.vectors[1,:].^2)
end
function dlvo_attachment(r1,r2;temperature,ionic_strength,relative_permittivity,
                         potential=0.010,hamaker=1.7e-22,order=128,quadrature=dlvo_quadrature(order))
    all(x->isfinite(x) && x>0,(r1,r2,temperature,relative_permittivity)) ||
        throw(ArgumentError("positive radii, temperature and permittivity required"))
    isfinite(potential) && isfinite(hamaker) && hamaker>=0 || throw(ArgumentError("invalid potential or Hamaker constant"))
    order isa Integer && order>=8 || throw(ArgumentError("quadrature order must be at least eight"))
    kappa=debye_wavenumber(ionic_strength,temperature,relative_permittivity)
    # q=(r1+r2)/(r1+r2+h) maps the infinite stability integral to [0,1].
    q=quadrature.nodes;weights=quadrature.weights
    prefactor=64π*vacuum_permittivity*relative_permittivity*(r1*r2/(r1+r2))*
        (kB*temperature/elementary_charge)^2*tanh(elementary_charge*potential/(4kB*temperature))^2
    logterms=map(eachindex(q)) do i
        gap=(r1+r2)*(1-q[i])/q[i]
        U=vdw_energy(r1,r2,gap,hamaker)+prefactor*exp(-kappa*gap)
        log(weights[i])+U/(kB*temperature)
    end
    peak=maximum(logterms)
    exp(-(peak+log(sum(exp.(logterms.-peak)))))
end

"""Unequal-sphere van der Waals energy in joules. Radii and positive surface gap are meters; Hamaker constant is joules. No electrostatic or steric energy is included."""
function vdw_energy(r1,r2,gap,hamaker)
    all(x->isfinite(x) && x>0,(r1,r2,gap)) && isfinite(hamaker) && hamaker>=0 ||
        throw(ArgumentError("radii and gap must be positive; Hamaker constant nonnegative"))
    z=4r1*r2/(gap*(gap+2(r1+r2)))
    bracket=if z<0.1
        # Avoid cancellation of the first two asymptotic orders at large gap.
        polynomial=zero(z)
        for j in 16:-1:0
            polynomial=-z*polynomial+(j+1)/(2(j+3))
        end
        z^3*polynomial
    else
        z/2+z/(2(1+z))-log1p(z)
    end
    -hamaker/6*bracket
end
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

"""Actual discrete mass-transfer rates in kg/m³/s, including discarded collision products."""
function mechanism_rates(op,N,p,t,fields,bulk)
    ctx=(;grid=op.grid,operator=op,N,cell=1,fields,bulk)
    scratch=similar(N); nucleation=zero(eltype(N)); growth=nucleation; overflow=nucleation
    for item in op.prepared
        kind=mechanism(item.term)
        kind in (:growth,:nucleation,:aggregation) || continue
        fill!(scratch,0)
        _,lostvolume=add_term!(scratch,N,item.term,item.method,item.cache,op,p,t,ctx)
        uptake=p.rho*π/6*moment(op.grid,scratch,3)
        kind===:growth && (growth+=uptake)
        kind===:nucleation && (nucleation+=uptake)
        kind===:aggregation && (overflow+=p.rho*π/6*lostvolume)
    end
    (;nucleation,growth,overflow)
end
end
