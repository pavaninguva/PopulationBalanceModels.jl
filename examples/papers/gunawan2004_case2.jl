module Gunawan2004Case2
using PopulationBalanceModels

export temperature,solubility,reported_seed_density,initial_counts,build,build_moments,moment_rhs!

temperature(t)=32.0-4.0*(-expm1(-t/18600.0))
solubility(T)=1.721e-4*T^2-5.88e-3*T+0.1286
reported_seed_density(L)=180.5<=L<=210.5 ? -3.48e-4*L^2+0.136*L-13.3 : 0.0

"""Three-point cell quadrature; align any discontinuities with grid edges."""
function initial_counts(grid,f)
    a=sqrt(3/5)
    [width/2*(5/9*f(x-width*a/2)+8/9*f(x)+5/9*f(x+width*a/2))
        for (x,width) in zip(grid.centers,grid.widths)]
end

"""Native exactly closed moment model. Supply μ0:μ3 and c0 explicitly."""
function build_moments(;initial_moments,c0,tspan=(0.0,1000.0))
    p=(kb=4.64e-7,b=1.78,kg=1.16e2,g=1.32,rho=2.11e-12)
    h=(p,t,ctx)->p.kg*max(ctx.bulk.c/solubility(ctx.fields.T)-1,0.0)^p.g
    J=(p,t,ctx)->p.kb*moment(ctx,3)*max(ctx.bulk.c/solubility(ctx.fields.T)-1,0.0)^p.b
    model=PBMModel(
        growth=Growth(rate=PolynomialRate((h,(p,t,ctx)->0.1h(p,t,ctx)))),
        nucleation=Nucleation(rate=J,size=0.0))
    op=compile(model,MomentSet(0:3;coordinate=:length))
    system=PBMSystem((crystals=op,);fields=(p,t)->(T=temperature(t),),
        nonnegative_bulk=(:c,),exchanges=(MomentExchange(:crystals,:c,3;
            factor=(p,t,ctx)->p.rho,mechanisms=(:growth,)),))
    PBMProblem(system,(crystals=initial_moments,),tspan;bulk0=(c=float(c0),),p)
end

"""
Gunawan et al. 2004, case 2, Eqs. 28-34 and Table 3.
initial_density and c0 are REQUIRED: Eq. 34's printed rounded polynomial is
negative throughout its support and the initial dissolved concentration is not
specified in the case-study text. Do not silently replace or clip that seed.

Units: length in micrometers, time in seconds, concentration in the paper's
g/g-water basis, crystal density in g/micrometer^3. The state basis and shape
factor follow the paper's V=integral(L^3*n) convention.
"""
function build(;grid=Grid1D(range(0.0,1600.0;length=1601);coordinate=:length),
                initial_density,c0,scheme=MUSCL(),tspan=(0.0,1000.0))
    p=(kb=4.64e-7,b=1.78,kg=1.16e2,g=1.32,rho=2.11e-12)
    supersaturation(ctx)=max(ctx.bulk.c/solubility(ctx.fields.T)-1,0.0)
    G=(L,p,t,ctx)->p.kg*supersaturation(ctx)^p.g*(1+0.1L)
    J=(p,t,ctx)->p.kb*moment(ctx.system,:crystals,3)*supersaturation(ctx)^p.b
    first(grid.edges)==0 || throw(ArgumentError("Gunawan boundary nucleation requires a grid starting at L=0"))
    model=PBMModel(growth=Growth(rate=G),nucleation=Nucleation(rate=J,size=0.0))
    methods=TermMethods(growth=scheme,by_term=(nucleation=BoundaryInjection(:left),))
    op=compile(model,grid;methods,bc=SizeBC(NoFlux(),NoFlux()))
    system=PBMSystem((crystals=op,);
        fields=(p,t)->(T=temperature(t),),nonnegative_bulk=(:c,),
        exchanges=(ConservativeExchange(:crystals,:c,(L,p,t,ctx)->p.rho*L^3;
            mechanisms=(:growth,:nucleation)),))
    N0=initial_counts(grid,initial_density)
    PBMProblem(system,(crystals=N0,),tspan;bulk0=(c=float(c0),),p)
end

"""
Independent continuum moment equations for this linear-in-length growth law.
State is [mu0,mu1,mu2,mu3,c]. Boundary nucleation at L=0 adds no higher moment.
Use an independent ODE solver to validate distribution discretization.
"""
function moment_rhs!(du,u,p,t)
    s=max(u[5]/solubility(temperature(t))-1,0.0)
    h=p.kg*s^p.g
    du[1]=p.kb*u[4]*s^p.b
    for k in 1:3
        du[k+1]=k*h*(u[k]+0.1u[k+1])
    end
    du[5]=-p.rho*du[4]
    nothing
end
end
