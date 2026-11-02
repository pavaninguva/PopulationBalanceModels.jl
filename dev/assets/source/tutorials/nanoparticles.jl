# Minimal nanoparticles tutorial.
# Physical definitions and one calculation; numerical helpers are linked on the case page.
# No benchmark exports or retained files are written. See the case-study page.

using PopulationBalanceModels
# DLVO quadrature and mechanism accounting are unchanged numerical helpers.
include(joinpath(pkgdir(PopulationBalanceModels), "examples", "papers", "cej2025_lnp.jl"))
using .CEJ2025LNP: kB, mechanism_rates
const material = CEJ2025LNP

# Physical definitions from examples/papers/cej2025_lnp.jl.
critical_length(S,T,gamma,Vm)=S>1 ? 4gamma*Vm/(kB*T*log(S)) : Inf

nucleation_rate(S,T,gamma,Vm,An)=S>1 ? An*exp(-16π*gamma^3*Vm^2/(3*(kB*T)^3*log(S)^2)) : 0.0

brownian_kernel(L,M,T,mu)=2kB*T*(L+M)^2/(3mu*L*M)

function aqueous_fraction(t,FRR,tau)
    FRR>=0 && tau>=0 || throw(ArgumentError("FRR and mixing time must be nonnegative"))
    tau==0 ? FRR/(FRR+1) : FRR/(FRR+1)*(-expm1(-t/tau))
end

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

# Physical definitions from examples/shin_lnp_reproduction.jl.
function figure4_problem(;grid,kwargs...)
    inputs=(;kwargs...)
    if haskey(inputs,:initial_distribution)
        all(iszero,inputs.initial_distribution) || throw(ArgumentError("Figure 4 starts with zero particles"))
    end
    defaults=(initial_lipid=10.0,FRR=3.0,tau=0.0,tspan=(0.0,4320*60.0))
    options=merge(defaults,inputs,(initial_distribution=zeros(length(grid)),))
    build(;grid,options...)
end

# Physical definitions from examples/shin_lnp_long_time.jl.
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

function tutorial_problem(; bins=80)
    # Explicit reconstruction assumptions are in make_problem above.
    # This startup check does not reproduce the historical full-horizon figures.
    remake(make_problem(; bins, FRR=3.0); tspan=(0.0, 1e-5))
end

function main(; bins=80, times=[0.0,1e-5])
    problem = tutorial_problem(; bins)
    solver = ForwardSolver(SSPRK33(); dt=1e-8)
    solution = simulate(problem; solver, saveat=times)
    @show bulk(problem, last(solution.u), :lipid)
    solution
end

# Optional after import PythonPlot; the short startup only, not a full-horizon claim.
plot_results(solution) = plot_observable(solution, BulkObservable(:lipid);
    xlabel="Time (s)", ylabel="Dissolved lipid concentration")

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
