module QamarMomentBenchmarks
using PopulationBalanceModels
using SpecialFunctions: gamma, erf
export problem4,problem5,exact4,exact5,run_benchmarks
const volume_map=AdditiveCoordinate(x->x^3,cbrt)
const daughter_law=UniformVolumeDaughters()
const gl=PopulationBalanceModels._gauss_legendre(GaussLegendre(24))

function initial_state(method,basis,raw)
    if method isa DQMOM
        q=quadrature(basis,raw,QMOM(method.nodes))
        return quadrature_state(method,q.nodes,q.weights)
    end
    raw
end

"""Problem 4: constant aggregation, volume-proportional breakage and exponential volume distribution."""
function problem4(;nodes=3,method=QMOM(nodes),tf=50.)
    basis=MomentSet(0:2nodes-1;coordinate=:length)
    model=PBMModel(
        aggregation=Aggregation(ConstantKernel(1.);coordinate_map=volume_map),
        breakage=Breakage((x,p,t,ctx)->x^3,daughter_law;coordinate_map=volume_map))
    mu=exact4(0.,basis.orders)
    PBMProblem(compile(model,basis;method),initial_state(method,basis,mu),(0.,tf))
end

# Number obeys dμ0/dt = a0 μ3 - β0 μ0²/2; μ3=V0 is constant.
# This also disambiguates the C0 symbol in the printed Eq.42: here β0=C0=1.
function exact4(t,orders=0:5)
    N0=1.;V0=.001;beta=1.;a0=1.
    steady=sqrt(2a0*V0/beta)/N0
    phi=steady*(1+steady*tanh(steady*beta*N0*t/2))/(steady+tanh(steady*beta*N0*t/2))
    [N0*phi*(V0/(N0*phi))^(k/3)*gamma(1+k/3) for k in orders]
end

# Eq.43 is a boundary DENSITY. Convert it to number flux using G=1 m/s.
boundary_density(t)=100+1e6*exp(-1e4*(t-.215)^2)
nucleation(p,t,ctx)=p.G*boundary_density(t)
initial5(x)=0<=x<=2 ? (.4<=x<=.6 ? 100. : .01) : 0.
initial_moment5(k)=.01*2.0^(k+1)/(k+1)+99.99*(.6^(k+1)-.4^(k+1))/(k+1)

"""Problem 5 with or without aggregation/breakage. The paper's background is
restricted to its stated initial window [0,2]; it is zero outside. Moments are
full-population integrals, so a sectional reference must extend beyond x=2.
"""
function problem5(;nodes=3,collisions=true,tf=.5,method=QMOM(nodes))
    terms=(growth=Growth(PolynomialRate((1.,))),primary=Nucleation(nucleation,0.))
    if collisions
        terms=merge(terms,(
            aggregation=Aggregation(ConstantKernel(1.5e-5);coordinate_map=volume_map),
            breakage=Breakage((x,p,t,ctx)->x^6,daughter_law;coordinate_map=volume_map)))
    end
    basis=MomentSet(0:2nodes-1;coordinate=:length)
    op=compile(PBMModel(;terms...),basis;method,
        methods=TermMethods(by_term=(growth=ExactMomentRule(),primary=ExactMomentRule())))
    raw=[initial_moment5(k) for k in basis.orders]
    PBMProblem(op,initial_state(method,basis,raw),(0.,tf);p=(G=1.,))
end

# Analytical full moments of advected initial particles and boundary births.
# Subdivide the time integral across the narrow pulse before applying Gauss rules.
function exact5(t,orders=0:5)
    out=Float64[]
    for k in orders
        value=sum(binomial(k,j)*t^(k-j)*initial_moment5(j) for j in 0:k)
        points=sort!(unique(vcat(0.,t,filter(s->0<s<t,.215 .+ .01 .* collect(-12:12)))))
        for (a,b) in zip(points[1:end-1],points[2:end])
            value+=PopulationBalanceModels._integrate_bin(s->(t-s)^k*boundary_density(s),a,b,gl)
        end
        push!(out,value)
    end
    out
end

"""Independent finite-volume reference in LENGTH, conserving volume for collisions.
The full residual has MUSCL growth and fixed-pivot collisions; the standalone
reference driver uses the stated Strang split. Unresolved event rates are exported.
"""
function sectional5(;bins=200,Lmax=3.,collisions=true,tf=.5)
    # Uniform length bins resolve the nucleation front; refine their width.
    edges=collect(range(0.,Lmax;length=bins+1))
    grid=Grid1D(edges;coordinate=:length)
    model=PBMModel(
        growth=Growth(PolynomialRate((1.,))),primary=Nucleation(nucleation,0.))
    if collisions
        model=PBMModel(growth=only(filter(t->t isa Growth,model.terms)),primary=Nucleation(nucleation,0.),
            aggregation=Aggregation(ConstantKernel(1.5e-5);coordinate_map=volume_map,overflow=:discard),
            breakage=Breakage((x,p,t,ctx)->x^6,daughter_law;coordinate_map=volume_map,overflow=:discard))
    end
    op=compile(model,grid;methods=TermMethods(growth=MUSCL(),by_term=(primary=BoundaryInjection(:left),)),
        bc=SizeBC(NoFlux(),OpenBoundary()))
    N=[.01*max(min(b,2.)-a,0.)+99.99*max(min(b,.6)-max(a,.4),0.) for (a,b) in zip(edges[1:end-1],edges[2:end])]
    PBMProblem(op,N,(0.,tf);p=(G=1.,))
end

"""Run paper Problems 4 and 5. No claims of exact PSD reconstruction from moments.
Reference5 is optional because a resolved collision calculation is more costly.
"""
function run_benchmarks(;output=joinpath(@__DIR__,"..","docs","src","assets","qamar_moments"),
        dt4=.01,dt5=.0002,reference_bins=(600,1200),reference_solver=nothing)
    mkpath(output)
    open(joinpath(output,"aggregation_breakage.csv"),"w") do io
        println(io,"nodes,time,order,moment,exact,relative_error")
        for nodes in (2,3,4)
            pb=problem4(;nodes);sol=simulate(pb;dt=dt4,saveat=0.:.5:50.)
            for (t,u) in zip(sol.t,sol.u),k in 0:min(5,2nodes-1)
                truth=exact4(t,0:2nodes-1)[k+1]
                println(io,join((nodes,t,k,u[k+1],truth,abs(u[k+1]/truth-1)),','))
            end
        end
    end
    times=collect(0.:.005:.5)
    open(joinpath(output,"pulse.csv"),"w") do io
        println(io,"processes,nodes,time,order,moment,reference,relative_difference")
        for collisions in (false,true),nodes in (3,4)
            pb=problem5(;nodes,collisions)
            solution=simulate(pb;dt=dt5,saveat=times)
            for (t,u) in zip(solution.t,solution.u),k in 0:4
                truth=collisions ? NaN : exact5(t,0:4)[k+1]
                println(io,join((collisions ? "all" : "growth_nucleation",nodes,t,k,u[k+1],truth,isfinite(truth) ? abs(u[k+1]/truth-1) : NaN),','))
            end
        end
    end
    open(joinpath(output,"refinement.csv"),"w") do io
        println(io,"problem,nodes,order,max_relative_time_difference,final_relative_time_difference")
        for (label,builder,dt) in ((4,problem4,dt4),(5,problem5,dt5))
            for nodes in (3,4)
                pb=builder(;nodes);ts=label==4 ? collect(0.:.5:50.) : times
                coarse=simulate(pb;dt,saveat=ts);fine=simulate(pb;dt=dt/2,saveat=ts)
                for k in 0:min(5,2nodes-1)
                    difference=maximum(abs(a[k+1]/b[k+1]-1) for (a,b) in zip(coarse.u,fine.u))
                    println(io,join((label,nodes,k,difference,abs(last(coarse.u)[k+1]/last(fine.u)[k+1]-1)),','))
                end
            end
        end
    end
    # The default standalone reference driver is defined below; an optional
    # checked implicit solver can be supplied for expensive resolved runs.
    checks=NamedTuple[]
    open(joinpath(output,"sectional_reference.csv"),"w") do io
        println(io,"bins,time,order,moment")
        for n in reference_bins
            pb=sectional5(bins=n)
            solution=reference_solver===nothing ? sectional_solve(pb,times) : reference_solver(pb,times)
            for (t,u) in zip(solution.t,solution.u),k in 0:4
                println(io,join((n,t,k,only(moment(pb.operator.grid,u,k))),','))
            end
            collision_model=PBMModel(Tuple(term for term in pb.operator.model.terms if term isa Union{Aggregation,Breakage}))
            cop=compile(collision_model,pb.operator.grid)
            final=last(solution.u);du=similar(final);rhs!(du,final,cop,pb.p,.5)
            rates=diagnostics(cop,final,pb.p,.5)
            lost=only(rates.unresolved_birth_size_rate)
            push!(checks,(bins=n,upper_length=last(pb.operator.grid.edges),
                unresolved_number_rate=only(rates.unresolved_birth_number_rate),unresolved_volume_rate=lost,
                collision_volume_balance=only(moment(cop.grid,du,3))+lost))
        end
    end
    open(joinpath(output,"reference_checks.csv"),"w") do io
        !isempty(checks) && println(io,join(keys(first(checks)),','))
        for row in checks;println(io,join(values(row),','));end
    end
    write(joinpath(output,"inputs.txt"),"Qamar et al., DOI 10.1002/aic.12264, Problems 4 and 5; online 2010, journal 2011.\n"*
        "Problem 4: β0=a0=N0=1, V0=.001; Eq.42 C0 interpreted as the stated β0, independently checked by the number balance.\n"*
        "Problem 5: x0=0, G=1, density boundary 100+10^6exp[-10^4(t-.215)^2], β=1.5e-5, a(x)=x^6.\n"*
        "Initial background .01 restricted to [0,2], square peak 100 on [.4,.6]. Full moments include particles advected beyond x=2.\n"*
        "Daughters: b(x|y)=6x²/y³, analytically integrated in moment weak forms.\n"*
        "QMOM 2/3/4 nodes; Julia $(VERSION); dt4=$dt4; dt5=$dt5.\n"*
        "Sectional comparison uses Strang splitting: exact C=1 growth/source cell integrals and SSPRK33 fixed-pivot collisions, not author FVS data.\n"*
        "Reference bins=$(reference_bins), Lmax=3; time/domain and lower-pivot losses must be assessed.\n")
end

function sectional_solve(pb,times)
    # Strang splitting supplies an independent, resolved distribution reference.
    # Growth + the known boundary source is advanced exactly in cell integrals
    # at C=1, so the pulse is not smeared by boundary upwinding.
    grid=pb.operator.grid;h=first(grid.widths);p=pb.p
    growthop=compile(PBMModel(Growth(AffineGrowth(1.,0.))),grid;
        evolution=Inguva2022(),bc=SizeBC(OpenBoundary(),OpenBoundary()))
    terms=Tuple(term for term in pb.operator.model.terms if term isa Union{Aggregation,Breakage})
    collisions=compile(PBMModel(terms),grid)
    N=copy(pb.u0);states=typeof(N)[];steps=0
    # Conservative SSP step estimate for this fixed benchmark; every RK stage
    # is still checked for nonnegative particle numbers.
    total_bound=sum(N)+50+1e6*sqrt(pi/1e4)
    reaction_dt=.8/(last(grid.edges)^6+1.5e-5*total_bound)
    ts=collect(times)
    for target in ts
        targetsteps=round(Int,target/h)
        isapprox(targetsteps*h,target;atol=1e-12,rtol=1e-10) ||
            throw(ArgumentError("reference output times must lie on the C=1 time grid"))
        while steps<targetsteps
            t=steps*h
            firsthalf=PBMProblem(collisions,N,(t,t+h/2);p)
            N=last(simulate(firsthalf;dt=reaction_dt,saveat=[t+h/2]).u)
            growthpb=PBMProblem(growthop,N,(t,t+h);p)
            N=last(simulate(growthpb;dt=h,saveat=[t+h]).u)
            N[1]+=100h+1e6*sqrt(pi)/200*(erf(100*(t+h-.215))-erf(100*(t-.215)))
            secondhalf=PBMProblem(collisions,N,(t+h/2,t+h);p)
            N=last(simulate(secondhalf;dt=reaction_dt,saveat=[t+h]).u)
            steps+=1
        end
        push!(states,copy(N))
    end
    PBMSolution(ts,states,pb,steps)
end

if abspath(PROGRAM_FILE)==@__FILE__;run_benchmarks();end
end
