module TemperatureProgrammedCrystallization
using PopulationBalanceModels

export parameters,solubility,linear_temperature,build_moments,build_sectional,run_comparison
const kB=1.380649e-23
const NA=6.02214076e23
const R=8.31446261815324

# Kelvin correlation from the supporting information; c is g solute/g solvent.
solubility(T)=-8.707+0.09669T-3.610e-4*T^2+4.590e-7*T^3

"""Published kinetic constants (Tables 2 and 6). Density is an explicit material input in g/cm³."""
function parameters(;density,shape_factor=0.797,molar_mass=151.16,
        kb1=10.6,sigma=3.83e-3,kb2=1.39e6,alpha=2.39,beta=0.41,
        kg=2.26e8,Eag=3.62e4,gamma_g=1.14,kd=-4.08e4,Ead=9.80e3,gamma_d=0.93)
    all(x->isfinite(x) && x>0,(density,shape_factor,molar_mass)) || throw(ArgumentError("invalid material scales"))
    Vm=molar_mass/(density*NA)*1e-6 # cm³ per molecule -> m³ per molecule
    material_factor=density*shape_factor*1e-15 # µm³/kg solvent -> g/g solvent
    (;density,shape_factor,molar_mass,Vm,material_factor,kb1,sigma,kb2,alpha,beta,
      kg,Eag,gamma_g,kd,Ead,gamma_d)
end

"""Piecewise-linear temperature history; input values in °C, returned values in K."""
function linear_temperature(times,celsius)
    times=float.(collect(times));values=float.(collect(celsius))
    length(times)==length(values)>=2 && all(isfinite,times) && all(diff(times).>0) &&
        all(isfinite,values) && all(values.>-273.15) || throw(ArgumentError("invalid temperature history"))
    function temperature(t)
        first(times)<=t<=last(times) || throw(ArgumentError("temperature requested outside supplied history"))
        i=min(searchsortedlast(times,t),length(times)-1)
        values[i]+(values[i+1]-values[i])*(t-times[i])/(times[i+1]-times[i])+273.15
    end
end

function growth_velocity(p,t,ctx)
    T=ctx.fields.T;cs=solubility(T);c=ctx.bulk.c
    cs>0 || throw(ArgumentError("solubility correlation used outside its positive range"))
    c>=cs ? p.kg*exp(-p.Eag/(R*T))*(c-cs)^p.gamma_g :
        p.kd*exp(-p.Ead/(R*T))*(cs-c)^p.gamma_d
end
function primary_rate(p,t,ctx)
    S=ctx.bulk.c/solubility(ctx.fields.T)
    S<=1 && return 0.0
    p.kb1*exp(-16π*p.Vm^2*p.sigma^3/(3*(kB*ctx.fields.T)^3*log(S)^2))
end
function secondary_rate(p,t,ctx)
    S=ctx.bulk.c/solubility(ctx.fields.T)
    S<=1 && return 0.0
    mass=p.material_factor*moment(ctx,3)
    mass>=0 || throw(NumericalFailure("negative crystal material in nucleation law"))
    p.kb2*(S-1)^p.alpha*mass^p.beta
end

"""
Native moments 0:4 with a shared concentration. Zero-size nucleation and
size-independent signed growth give an exact closure ONLY while the number
flux at zero size vanishes. This builder supplies no extinction closure.
"""
function build_moments(;p,c0,temperature_profile,tspan,initial_moments=zeros(5))
    model=PBMModel(
        growth=Growth(rate=PolynomialRate((growth_velocity,))),
        primary=Nucleation(rate=primary_rate,size=0.0),
        secondary=Nucleation(rate=secondary_rate,size=0.0))
    operator=compile(model,MomentSet(0:4;coordinate=:length);method=ExactClosure())
    system=PBMSystem((crystals=operator,);fields=(p,t)->(T=temperature_profile(t),),
        nonnegative_bulk=(:c,),exchanges=(MomentExchange(:crystals,:c,3;
            factor=(p,t,ctx)->p.material_factor,mechanisms=(:growth,)),))
    PBMProblem(system,(crystals=initial_moments,),tspan;bulk0=(c=c0,),p)
end

"""Closed-grid sectional comparison for the nonextinguishing cooling calculation."""
function build_sectional(;p,c0,temperature_profile,tspan,bins=800,scheme=MUSCL())
    grid=Grid1D(range(0.0,2000.0;length=bins+1);coordinate=:length)
    model=PBMModel(
        growth=Growth(rate=(L,p,t,ctx)->growth_velocity(p,t,ctx)),
        primary=Nucleation(rate=primary_rate,size=0.0),
        secondary=Nucleation(rate=secondary_rate,size=0.0))
    operator=compile(model,grid;methods=TermMethods(growth=scheme,
        by_term=(primary=BoundaryInjection(:left),secondary=BoundaryInjection(:left))))
    system=PBMSystem((crystals=operator,);fields=(p,t)->(T=temperature_profile(t),),
        nonnegative_bulk=(:c,),exchanges=(ConservativeExchange(:crystals,:c,
            (L,p,t,ctx)->p.material_factor*L^3;mechanisms=(:growth,:nucleation)),))
    PBMProblem(system,(crystals=zeros(length(grid)),),tspan;bulk0=(c=c0,),p)
end

function quantities(problem,u)
    op=problem.operator.system.populations.crystals
    N=population(problem,u,:crystals)
    mu=[moment(op.grid,N,k) for k in 0:4]
    c=bulk(problem,u,:c)
    (;mu,c,L30=mu[1]>0 ? cbrt(mu[4]/mu[1]) : NaN,
      L43=mu[4]>0 ? mu[5]/mu[4] : NaN)
end

"""
Reconstructed noncycling input motivated by Figure 13 Case I, not an exact
reproduction of all four panels. The protocol and c0 are graph-read choices;
density=1.293 g/cm³ is an explicit material assumption, not a fitted parameter.
"""
function run_comparison(;output=joinpath(@__DIR__,"..","docs","src","assets","moment_crystallization"),
        bins=(400,800),dt=0.02)
    p=parameters(density=1.293)
    temperature=linear_temperature([0.,20.,140.,460.,550.],[50.,30.,30.,20.,20.])
    c0=0.300; times=collect(0.0:1.0:550.0)
    prob=build_moments(;p,c0,temperature_profile=temperature,tspan=(0.0,550.0))
    # Intermediate RK moment vectors need not themselves represent a PSD.
    # Check saved states and convergence explicitly; never project or clip them.
    reference=simulate(prob;dt=dt/2,saveat=times,check_nonnegative=false)
    coarse=simulate(prob;dt,saveat=times,check_nonnegative=false)
    refinement=maximum(abs.(last(reference.u).-last(coarse.u))./max.(abs.(last(reference.u)),eps()))
    mkpath(output)
    open(joinpath(output,"history.csv"),"w") do io
        println(io,"method,bins,time,temperature_K,concentration,supersaturation,mu0,mu1,mu2,mu3,mu4,L30,L43,material_drift")
        cases=Tuple{String,Int,Any,Any}[("moments",0,prob,reference)]
        for n in bins
            fv=build_sectional(;p,c0,temperature_profile=temperature,tspan=prob.tspan,bins=n)
            push!(cases,("muscl",n,fv,simulate(fv;dt,saveat=times)))
        end
        for (method,n,problem,solution) in cases
            if method=="muscl"
                N=population(problem,last(solution.u),:crystals)
                sum(N[end-2:end])/sum(N)<1e-8 || throw(ArgumentError("upper size boundary is populated; enlarge the domain"))
            end
            for (t,u) in zip(solution.t,solution.u)
                q=quantities(problem,u)
                all(x->isfinite(x) && x>=0,q.mu) || throw(NumericalFailure("invalid saved moments"))
                method=="moments" && !realizability(problem.operator.system.populations.crystals.grid,q.mu).valid &&
                    throw(NumericalFailure("saved moments fail necessary Hankel checks"))
                isfinite(q.c) && q.c>=0 || throw(NumericalFailure("invalid saved concentration"))
                # The comparison has no dissolution of a populated distribution.
                q.mu[1]>0 && q.c/solubility(temperature(t))<1-1e-7 &&
                    throw(ArgumentError("this comparison requires a vanishing extinction flux"))
                println(io,join((method,n,t,temperature(t),q.c,q.c/solubility(temperature(t)),
                    q.mu...,q.L30,q.L43,q.c+p.material_factor*q.mu[4]-c0),','))
            end
        end
    end
    write(joinpath(output,"inputs.txt"),
        "Kim et al. (2023), doi:10.1021/acs.iecr.2c03465. Kinetics: Tables 2 and 6; solubility: SI.\n"*
        "ILLUSTRATIVE reconstruction of noncycling Figure 13 Case I, not full Figure 13 reproduction.\n"*
        "Temperature knots (min, Celsius): (0,50),(20,30),(140,30),(460,20),(550,20); c0=0.300 g/g.\n"*
        "Density assumed 1.293 g/cm^3; shape factor 0.797; molecular weight 151.16 g/mol.\n"*
        "No extinction closure or PSD reconstruction is inferred from five raw moments.\n"*
        "Julia $(VERSION); dt=$dt; native step-halving relative difference=$refinement\n")
    println("Native moment step-halving difference: ",refinement)
    reference
end
if abspath(PROGRAM_FILE)==@__FILE__;run_comparison();end
end
