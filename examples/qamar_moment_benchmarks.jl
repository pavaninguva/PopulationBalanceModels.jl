module QamarMomentBenchmarks
using PopulationBalanceModels
using LinearAlgebra
using SpecialFunctions: gamma, erf
export problem4,problem5,exact4,exact5,run_benchmarks,run_method_comparison,comparison_moment,method_basis
const volume_map=AdditiveCoordinate(x->x^3,cbrt)
const daughter_law=UniformVolumeDaughters()
const gl=PopulationBalanceModels._gauss_legendre(GaussLegendre(24))

function method_basis(method,nodes=3)
    lastorder=method isa EQMOM ? 2method.nodes : method isa LogNormalClosure ? 2 :
        method isa RankAdaptiveQMOM ? 2method.max_nodes-1 :
        method isa Union{QuadratureClosure,DQMOM} ? 2method.nodes-1 : 2nodes-1
    MomentSet(0:lastorder;coordinate=:length)
end

function initial_state(method,basis,raw)
    if method isa DQMOM
        q=quadrature(basis,raw,QMOM(method.nodes))
        return quadrature_state(method,q.nodes,q.weights)
    end
    raw
end

"""Problem 4: constant aggregation, volume-proportional breakage and exponential volume distribution."""
function problem4(;nodes=3,method=QMOM(nodes),tf=50.)
    basis=method_basis(method,nodes)
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
    basis=method_basis(method,nodes)
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
        dt4=.01,dt5=.0002,reference_bins=(600,1200),reference_solver=nothing,
        compare_methods=true,comparison_dt4=.02)
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
    compare_methods && run_method_comparison(;output,dt4=comparison_dt4,dt5)
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


"""A tracked raw moment, or an explicitly labelled assumed-lognormal moment.
Only the lognormal family reconstructs untracked moments here; quadrature nodes
are never presented as a uniquely reconstructed density.
"""
function comparison_moment(op,u,k)
    k in op.grid.orders && return moment_value(op.closure,op.grid,u,k)
    op.closure isa LogNormalClosure || throw(ArgumentError("untracked moment $k"))
    a,b,c=vec(u)
    iszero(a) && return 0.0
    a*exp(k*log(b/a)+k*(k-1)/2*log(c*a/b^2))
end

function comparison_run(builder,method,times,dt)
    pb=builder(;method)
    states=typeof(pb.u0)[copy(pb.u0)];saved=Float64[first(times)]
    status="completed";failure="";max_rank=0;min_rank=typemax(Int);max_condition=NaN
    started=time_ns()
    for target in times[2:end]
        try
            localpb=remake(pb;u0=last(states),tspan=(last(saved),target))
            sol=simulate(localpb;dt,saveat=[target],method=SSPRK33())
            # Accepted moments must also be evaluable by the selected closure.
            q=quadrature(pb.operator.grid,last(sol.u),method)
            # Lognormal integration nodes are not a moment-reconstruction rank.
            if !(method isa LogNormalClosure)
                rank=hasproperty(q,:rank) ? q.rank : count(>(0),q.weights)
                max_rank=max(max_rank,rank);min_rank=min(min_rank,rank)
            end
            if method isa DQMOM
                scaled=q.nodes./(sum(q.weights.*q.nodes)/sum(q.weights));n=method.nodes
                A=[i<=n ? (1-k)*scaled[i]^k : k==0 ? 0. : k*scaled[i-n]^(k-1) for k in 0:2n-1,i in 1:2n]
                singular=svdvals(A)
                condition=maximum(singular)/minimum(singular)
                max_condition=isnan(max_condition) ? condition : max(max_condition,condition)
            end
            push!(states,copy(last(sol.u)));push!(saved,target)
        catch e
            e isa Union{NumericalFailure,ArgumentError} || rethrow()
            status="failed";failure=sprint(showerror,e);break
        end
    end
    (;pb,t=saved,u=states,status,failure,elapsed=(time_ns()-started)/1e9,
      min_rank=method isa LogNormalClosure ? NaN : min_rank==typemax(Int) ? 0 : min_rank,
      max_rank=method isa LogNormalClosure ? NaN : max_rank,max_condition)
end

"""
Compare the actual moment families on the same physical laws. Uses SSPRK33 and
checks time refinement separately. Each method has its required state layout;
lognormal higher moments are labelled reconstructions, not conserved states.
Failed runs retain only completed output intervals and their error message.
The committed sectional comparison is reused for plotting Problem 5 collisions.
"""
function run_method_comparison(;output=joinpath(@__DIR__,"..","docs","src","assets","qamar_moments"),
        dt4=.02,dt5=.0002,check_reconstruction=true,methods=(QMOM=QMOM(3),RankAdaptive=RankAdaptiveQMOM(3),
            GammaEQMOM=EQMOM(3),DQMOM=DQMOM(3),LogNormal=LogNormalClosure(quadrature_order=64)))
    mkpath(output)
    histories=NamedTuple[];summaries=NamedTuple[]
    for (scenario,builder,dt,times) in (
            ("aggregation_breakage",problem4,dt4,collect(0.:.5:50.)),
            ("growth_nucleation",(;method)->problem5(;method,collisions=false),dt5,collect(0.:.005:.5)),
            ("growth_nucleation_collisions",problem5,dt5,collect(0.:.005:.5)))
        for (name,method) in pairs(methods)
            # Compilation warmup only; elapsed time below is one subsequent run.
            try
                comparison_run(builder,method,times[1:2],dt)
                run=comparison_run(builder,method,times,dt)
                fine=comparison_run(builder,method,times,dt/2)
                common=min(length(run.t),length(fine.t));time_error=NaN
                if common>1
                    time_error=maximum(abs(comparison_moment(run.pb.operator,run.u[i],k)/
                        comparison_moment(fine.pb.operator,fine.u[i],k)-1) for i in 2:common,k in 0:2)
                end
                for (t,u) in zip(run.t,run.u),k in 0:min(5,last(run.pb.operator.grid.orders))
                    truth=scenario=="aggregation_breakage" ? exact4(t,0:5)[k+1] :
                          scenario=="growth_nucleation" ? exact5(t,0:5)[k+1] : NaN
                    value=comparison_moment(run.pb.operator,u,k)
                    push!(histories,(scenario=scenario,method=String(name),time=t,order=k,
                        moment=value,reference=truth,relative_error=isfinite(truth) ? abs(value/truth-1) : NaN,tracked=true))
                end
                if method isa LogNormalClosure
                    for (t,u) in zip(run.t,run.u)
                        truth=scenario=="aggregation_breakage" ? exact4(t)[4] :
                              scenario=="growth_nucleation" ? exact5(t)[4] : NaN
                        value=comparison_moment(run.pb.operator,u,3)
                        push!(histories,(scenario=scenario,method=String(name),time=t,order=3,
                            moment=value,reference=truth,relative_error=isfinite(truth) ? abs(value/truth-1) : NaN,tracked=false))
                    end
                end
                third0=comparison_moment(run.pb.operator,first(run.u),3)
                thirdf=comparison_moment(run.pb.operator,last(run.u),3)
                push!(summaries,(scenario=scenario,method=String(name),states=length(run.pb.u0),
                    tracked_M3=3 in run.pb.operator.grid.orders,status=run.status,last_time=last(run.t),
                    seconds=run.elapsed,time_refinement=time_error,fine_status=fine.status,fine_last_time=last(fine.t),max_source_condition=run.max_condition,
                    third_moment_relative_change=thirdf/third0-1,min_rank=run.min_rank,max_rank=run.max_rank,
                    failure=replace(run.failure,','=>';','\n'=>' ')))
                println(scenario," ",name,": ",run.status," through t=",last(run.t),"; seconds=",run.elapsed,
                    "; time difference=",time_error,"; ",run.failure);flush(stdout)
            catch e
                e isa Union{NumericalFailure,ArgumentError} || rethrow()
                push!(summaries,(scenario=scenario,method=String(name),states=length(method_basis(method)),
                    tracked_M3=3 in method_basis(method).orders,status="initialization_failed",last_time=0.,
                    seconds=0.,time_refinement=NaN,fine_status="not_run",fine_last_time=0.,max_source_condition=NaN,third_moment_relative_change=NaN,
                    min_rank=0,max_rank=0,failure=replace(sprint(showerror,e),','=>';','\n'=>' ')))
            end
        end
    end
    for (filename,rows) in (("method_comparison.csv",histories),("method_summary.csv",summaries))
        open(joinpath(output,filename),"w") do io
            isempty(rows) && return
            println(io,join(keys(first(rows)),','))
            for row in rows;println(io,join(values(row),','));end
        end
    end
    if check_reconstruction
        checks=NamedTuple[]
        for (scenario,builder) in (("growth_nucleation",(;method)->problem5(;method,collisions=false)),
                                  ("growth_nucleation_collisions",problem5))
            for (name,method,order) in (("LogNormal",LogNormalClosure(quadrature_order=16),16),
                    ("LogNormal",LogNormalClosure(quadrature_order=32),32),
                    ("LogNormal",LogNormalClosure(quadrature_order=64),64),
                    ("GammaEQMOM",EQMOM(3),12),("GammaEQMOM",EQMOM(3;secondary_nodes=48),48))
                r=comparison_run(builder,method,collect(0.:.005:.5),dt5/2)
                push!(checks,(scenario=scenario,method=name,quadrature_order=order,dt=dt5/2,status=r.status,
                    last_time=last(r.t),failure=replace(r.failure,','=>';','\n'=>' ')))
            end
        end
        open(joinpath(output,"reconstruction_checks.csv"),"w") do io
            println(io,join(keys(first(checks)),','))
            for row in checks;println(io,join(values(row),','));end
        end
    end
    (;histories,summaries)
end

if abspath(PROGRAM_FILE)==@__FILE__
    "--methods-only" in ARGS ? run_method_comparison() : run_benchmarks()
end
end
