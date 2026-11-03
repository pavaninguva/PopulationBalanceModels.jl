# Material/measurement formulas shared by the explicit tutorial and the model
# recipe. Requires PopulationBalanceModels and LinearAlgebra in the caller.
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
