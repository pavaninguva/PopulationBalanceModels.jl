"""
    MeasureEuler(; reconstruction=nothing, reconstruction_rtol=1e-7,
                 cfl=0.9, event_rtol=1e-10, max_rejections=40)

Opt-in first-order positive-measure evolution. Growth pushes quadrature atoms
along frozen-velocity characteristics; point births use positive age quadrature;
removal and collision events use nonnegative forward-Euler measure updates under
a depletion bound. Every physical substep is reconstructed and checked. Bulk
exchanges use the same moment differences and escaped material. No moments are
clipped or projected. ExactClosure requires an explicitly supplied
RankAdaptiveQMOM reconstruction; QMOM and RankAdaptiveQMOM use their own rules.
EQMOM, LogNormalClosure and DQMOM are not supported by this policy. Fixed QMOM
cannot cross rank loss; use RankAdaptiveQMOM for extinction. Axial upwind/centered
transport is a common positive mixture under a mesh-dependent CFL bound.
"""
struct MeasureEuler{R}
    reconstruction::R
    reconstruction_rtol::Float64
    cfl::Float64
    event_rtol::Float64
    max_rejections::Int
    function MeasureEuler(;reconstruction=nothing,reconstruction_rtol=1e-7,cfl=0.9,event_rtol=1e-10,max_rejections=40)
        0<reconstruction_rtol<1 && 0<cfl<1 && 0<=event_rtol<1e-3 && max_rejections>=1 || throw(ArgumentError("invalid MeasureEuler tolerances/CFL"))
        reconstruction===nothing || reconstruction isa RankAdaptiveQMOM || throw(ArgumentError("explicit reconstruction must be RankAdaptiveQMOM"))
        new{typeof(reconstruction)}(reconstruction,reconstruction_rtol,cfl,event_rtol,max_rejections)
    end
end
mutable struct _MeasureStatistics
    reconstruction::Dict{Symbol,Vector{Float64}}
    number::Dict{Tuple{Symbol,Symbol},Float64}
end
_MeasureStatistics()=_MeasureStatistics(Dict{Symbol,Vector{Float64}}(),Dict{Tuple{Symbol,Symbol},Float64}())
struct _MeasureRestriction <: Exception
    message::String
end
_measure_restriction(message)=throw(_MeasureRestriction(message))
function _measure_closure(op,m)
    if op.closure isa ExactClosure
        m.reconstruction===nothing && throw(ArgumentError("MeasureEuler with ExactClosure requires explicit reconstruction=RankAdaptiveQMOM(n)"))
        length(op.grid)==2m.reconstruction.max_nodes || throw(ArgumentError("reconstruction must use every tracked moment"))
        return m.reconstruction
    end
    op.closure isa Union{QuadratureClosure,RankAdaptiveQMOM} || throw(ArgumentError("MeasureEuler supports raw QMOM/RankAdaptiveQMOM and explicitly reconstructed ExactClosure; EQMOM, lognormal and DQMOM are unavailable"))
    op.closure
end
function _measure_quadrature(op,u,m)
    closure=_measure_closure(op,m)
    q=quadrature(op.grid,u,closure)
    all(isfinite,q.nodes) && all(>=(op.boundary.lower),q.nodes) && all(<=(op.boundary.upper),q.nodes) &&
        all(isfinite,q.weights) && all(>=(0),q.weights) || throw(NumericalFailure("quadrature outside declared nonnegative support"))
    if iszero(u[1])
        all(iszero,u) || throw(NumericalFailure("empty measure has nonzero moments"))
    else
        scale=u[2]>0 ? u[2]/u[1] : 1.0
        error=maximum(abs(sum(q.weights.*q.nodes.^k)-u[k+1])/(u[1]*scale^k+abs(u[k+1])) for k in op.grid.orders)
        error<=m.reconstruction_rtol || throw(NumericalFailure("positive reconstruction does not resolve every tracked moment"))
    end
    q
end
function _validate_measure_operator(op::CompiledSystem,m)
    isempty(op.system.interactions) || throw(ArgumentError("MeasureEuler does not support inter-population interaction callbacks"))
    for e in op.system.exchanges
        pop=getproperty(op.system.populations,e.population)
        if pop isa CompiledPBM && !haskey(op.ledger_indices,e.population) &&
           any(item->item.term isa Union{Aggregation,Breakage} && _effective_term(item).overflow===:discard &&
               mechanism(item.term) in e.mechanisms,pop.prepared)
            throw(ArgumentError("MeasureEuler with collision discard and bulk exchange requires CumulativeAccounting"))
        end
    end
    for pop in values(op.system.populations)
        if pop isa CompiledMomentPBM
            _measure_closure(pop,m)
            pop.closure isa QuadratureClosure && (pop.boundary.left===:absorb || pop.boundary.left===:escape || pop.boundary.right===:escape) &&
                throw(ArgumentError("fixed-rank QMOM cannot handle boundary rank loss; select RankAdaptiveQMOM"))
            for item in pop.prepared
                item.term isa Union{Growth,Nucleation,Death,Dilution,Aggregation,Breakage} || throw(ArgumentError("MeasureEuler has no positive-measure rule for $(typeof(item.term))"))
                item.method isa Union{ExactMomentRule,QuadratureMomentRule,_AutomaticMomentRule} || throw(ArgumentError("custom weak forms require their own admissible evolution rule"))
            end
        elseif pop isa CompiledPBM
            any(item->item.term isa SizeDiffusion,pop.prepared) && throw(ArgumentError("SizeDiffusion requires a residual RK integrator; MeasureEuler diffusion splitting is not implemented"))
            all(item->!(item.term isa Growth) || item.method isa Upwind,pop.prepared) || throw(ArgumentError("MeasureEuler's sectional guarantee requires explicitly selected Upwind growth"))
            any(item->item.term isa CustomTerm,pop.prepared) && throw(ArgumentError("custom sectional sources have no positivity contract"))
            pop.bc.left isa Periodic && throw(ArgumentError("sectional periodic size transport is not supported by MeasureEuler"))
        else
            throw(ArgumentError("unsupported population in MeasureEuler"))
        end
    end
end
function _measure_check(op::CompiledSystem,u,m)
    all(isfinite,u) || throw(NumericalFailure("nonfinite coupled measure state"))
    ranks=Dict{Symbol,Int}()
    for name in keys(op.ranges)
        pop=getproperty(op.system.populations,name);v=population(op,u,name)
        if pop isa CompiledMomentPBM
            q=_measure_quadrature(pop,v,m)
            ranks[name]=hasproperty(q,:rank) ? q.rank : count(>(0),q.weights)
        else
            all(>=(0),v) || throw(NumericalFailure("negative sectional population"))
        end
    end
    for name in op.system.nonnegative_bulk
        bulk(op,u,name)>=0 || _measure_restriction("bulk depletion requires a smaller step")
    end
    all(>=(0),view(u,op.physical_length+1:op.length)) || throw(NumericalFailure("negative cumulative accounting state"))
    ranks
end
function _measure_context(op,u,p,t,name)
    ctx=_system_context(op,u,zero(u),p,t)
    pop=getproperty(op.system.populations,name)
    _context(pop,reshape(population(op,u,name),:,1),1,t,ctx)
end
_measure_moments(grid,x,w)=[sum((wi*xi^k for (xi,wi) in zip(x,w));init=0.0) for k in grid.orders]
function _measure_commit!(u,op,name,new,changes,losses,p,t,m,defects)
    pop=getproperty(op.system.populations,name)
    # Validate the new positive-measure moments before another closure/rate call.
    pop isa CompiledMomentPBM && _measure_quadrature(pop,new,m)
    ctx=_measure_context(op,u,p,t,name)
    for e in op.system.exchanges
        e.population==name || continue
        if e isa MomentExchange
            factor=_nonnegative(_source_value(e.factor,p,t,ctx),"moment exchange factor")
            delta=sum((change[e.order+1] for (kind,change) in pairs(changes) if kind in e.mechanisms);init=0.0)
            u[getproperty(op.bulk_indices,e.bulk)]-=factor*delta
        else
            haskey(op.ledger_indices,name) && _validate_exchange_property(e,op,pop,p,t,ctx)
            delta=sum((sum(e.weight(x,p,t,ctx)*change[i] for (i,x) in enumerate(pop.grid.centers)) for (kind,change) in pairs(changes) if kind in e.mechanisms);init=0.0)
            delta+=sum((e.weight(x,p,t,ctx)*a for (mechanism,x,a,kind) in losses if kind in e.mechanisms);init=0.0)
            u[getproperty(op.bulk_indices,e.bulk)]-=delta
        end
    end
    if pop isa CompiledMomentPBM
        # Never disguise reconstruction roundoff/rank tolerance as solute transfer.
        delta=sum(values(changes))
        escaped=[sum((a*x^k for (_,x,a,_) in losses);init=0.0) for k in pop.grid.orders]
        defect=new.-population(op,u,name).-delta.+escaped
        defects.reconstruction[name]=get(defects.reconstruction,name,zeros(length(new))).+defect
    end
    for (kind,change) in pairs(changes)
        amount=pop isa CompiledMomentPBM ? change[1] : sum(change)+sum((a for (_,_,a,k) in losses if k===kind);init=0.0)
        key=(name,kind)
        defects.number[key]=get(defects.number,key,0.0)+amount
    end
    population(op,u,name).=new
    for (mechanism,x,amount,_) in losses
        _record_loss!(u,op,name,mechanism,x,amount)
    end
    _measure_check(op,u,m)
end
function _characteristic(x,G,h,b,event_rtol)
    isfinite(G) || throw(NumericalFailure("nonfinite characteristic velocity"))
    y=x+h*G
    isfinite(y) || throw(NumericalFailure("nonfinite characteristic position"))
    tolerance=event_rtol*max(abs(x),abs(h*G),abs(b.lower))
    if G<0 && y<=b.lower+tolerance
        b.left===:error && throw(ArgumentError("characteristic reached the lower support boundary; explicitly select absorption or escape"))
        return b.lower,b.left===:absorb ? :extinction : :lower_escape
    elseif G>0 && y>=b.upper-event_rtol*max(abs(x),abs(h*G),isfinite(b.upper) ? abs(b.upper) : 0.0)
        b.right===:error && throw(ArgumentError("characteristic reached the upper support boundary; explicitly select escape"))
        return b.upper,:upper_escape
    end
    y>=b.lower && y<=b.upper || throw(NumericalFailure("characteristic outside declared support"))
    y,nothing
end
function _growth_flight(term,x,G,h,b,m,p,t,ctx)
    isfinite(G) || throw(NumericalFailure("nonfinite characteristic velocity"))
    # A frozen Euler flight must not invent a boundary hit when the vector
    # field points inward or vanishes there (e.g. dL/dt=-L). This bound does
    # not prevent genuine constant-velocity outward hitting events.
    if G<0
        edge_velocity=term.rate(b.lower,p,t,ctx)
        isfinite(edge_velocity) || throw(ArgumentError("MeasureEuler requires a finite lower-edge growth law"))
        edge_velocity>=0 && h*(-G)>m.cfl*(x-b.lower) &&
            _measure_restriction("growth bound at a non-outgoing lower edge")
    elseif G>0 && isfinite(b.upper)
        edge_velocity=term.rate(b.upper,p,t,ctx)
        isfinite(edge_velocity) || throw(ArgumentError("MeasureEuler requires a finite upper-edge growth law"))
        edge_velocity<=0 && h*G>m.cfl*(b.upper-x) &&
            _measure_restriction("growth bound at a non-outgoing upper edge")
    end
    _characteristic(x,G,h,b,m.event_rtol)
end
function _moment_measure_step!(u,op,name,p,t,h,m,defects)
    pop=getproperty(op.system.populations,name);g=pop.grid
    growth=filter(item->item.term isa Growth,pop.prepared)
    length(growth)<=1 || throw(ArgumentError("combine growth velocities before MeasureEuler"))
    # Lie ordering: existing-particle growth, age-integrated births, then each
    # removal/collision term in declared order. Coupling is first order.
    for item in (growth...,filter(item->item.term isa Nucleation,pop.prepared)...,
                 filter(item->!(item.term isa Union{Growth,Nucleation}),pop.prepared)...)
        _measure_check(op,u,m)
        old=copy(population(op,u,name));q=_measure_quadrature(pop,old,m)
        x=collect(q.nodes);w=collect(q.weights)
        reconstructed_old=_measure_moments(g,x,w)
        ctx=_measure_context(op,u,p,t,name);term=item.term;kind=mechanism(term)
        losses=Tuple{Symbol,Float64,Float64,Symbol}[]
        if term isa Growth
            nextx=Float64[];nextw=Float64[]
            for (xi,wi) in zip(x,w)
                iszero(wi) && continue
                y,loss=_growth_flight(term,xi,term.rate(xi,p,t,ctx),h,pop.boundary,m,p,t,ctx)
                if loss===nothing
                    push!(nextx,y);push!(nextw,wi)
                else
                    push!(losses,(loss,y,wi,:growth))
                end
            end
            x=nextx;w=nextw
        elseif term isa Nucleation
            J=_nonnegative(term.rate(p,t,ctx),"nucleation rate")
            b=pop.boundary
            b.lower<=term.size<=b.upper || throw(ArgumentError("nucleation point lies outside moment support"))
            # Positive quadrature in particle age; exact through the tracked
            # polynomial degree for constant G,J. This avoids an invalid Euler
            # predictor from an initially empty population.
            n=length(g)÷2;aq=_gauss_legendre(GaussLegendre(max(n,1)))
            G=iszero(J) || isempty(growth) ? 0.0 : only(growth).term.rate(term.size,p,t,ctx)
            for j in eachindex(aq.nodes)
                amount=h*J*aq.weights[j]/2
                iszero(amount) && continue
                y,loss=isempty(growth) ? (term.size,nothing) :
                    _growth_flight(only(growth).term,term.size,G,h*(1+aq.nodes[j])/2,b,m,p,t,ctx)
                if loss===nothing
                    push!(x,y);push!(w,amount)
                else
                    push!(losses,(loss,y,amount,:growth))
                end
            end
        elseif term isa Union{Death,Dilution}
            for i in eachindex(w)
                iszero(w[i]) && continue
                hazard=_nonnegative(term.rate(x[i],p,t,ctx),"removal hazard")
                h*hazard<=m.cfl || _measure_restriction("removal depletion bound")
                w[i]*=1-h*hazard
            end
        elseif term isa Union{Aggregation,Breakage}
            depletion=zeros(length(w));bx=Float64[];bw=Float64[]
            function product(y,amount)
                isfinite(y) && y>=0 || throw(NumericalFailure("invalid product size"))
                iszero(amount) && return
                if y<pop.boundary.lower || y>pop.boundary.upper
                    term.overflow===:error && throw(NumericalFailure("collision product outside moment support; use overflow=:discard explicitly"))
                    side=y<pop.boundary.lower ? :underflow : :overflow
                    push!(losses,(Symbol(kind,"_",side),y,amount,kind))
                else
                    push!(bx,y);push!(bw,amount)
                end
            end
            if term isa Aggregation
                for i in eachindex(w),j in 1:i
                    (iszero(w[i]) || iszero(w[j])) && continue
                    K=_nonnegative(term.kernel(x[i],x[j],p,t,ctx),"aggregation kernel")
                    isapprox(K,term.kernel(x[j],x[i],p,t,ctx);rtol=1e-10,atol=0) || throw(ArgumentError("aggregation kernel must be symmetric"))
                    amount=h*K*w[i]*w[j]*(i==j ? 0.5 : 1)
                    depletion[i]+=amount;depletion[j]+=amount
                    y=term.coordinate_map===nothing ? x[i]+x[j] : term.coordinate_map.inverse(_measure(term,x[i])+_measure(term,x[j]))
                    product(y,amount)
                end
            else
                for i in eachindex(w)
                    iszero(w[i]) && continue
                    amount=h*_nonnegative(term.rate(x[i],p,t,ctx),"breakage hazard")*w[i]
                    daughters=collect(term.daughters(x[i],p,t,ctx))
                    all(d->isfinite(d[1]) && 0<=d[1]<=x[i] && isfinite(d[2]) && d[2]>=0,daughters) || throw(ArgumentError("invalid daughter measure"))
                    isapprox(sum(count*_measure(term,y) for (y,count) in daughters),_measure(term,x[i]);rtol=1e-8,atol=0) || throw(ArgumentError("daughters do not conserve additive property"))
                    for k in g.orders
                        applicable(daughter_moment,term.daughters,x[i],k,p,t,ctx) || continue
                        exact=daughter_moment(term.daughters,x[i],k,p,t,ctx)
                        computed=sum(count*y^k for (y,count) in daughters)
                        isapprox(computed,exact;rtol=m.reconstruction_rtol,atol=0) ||
                            throw(ArgumentError("positive daughter quadrature does not resolve tracked moment $k; increase its quadrature order"))
                    end
                    depletion[i]+=amount
                    for (y,count) in daughters;product(y,amount*count);end
                end
            end
            all(depletion.<=m.cfl.*w) || _measure_restriction("collision depletion bound")
            w=w.-depletion;append!(x,bx);append!(w,bw)
        end
        new=_measure_moments(g,x,w)
        escaped=[sum((amount*y^k for (_,y,amount,_) in losses);init=0.0) for k in g.orders]
        if term isa Nucleation
            source=[h*J*term.size^k for k in g.orders]
            changes=(nucleation=source,growth=new.-reconstructed_old.+escaped.-source)
        else
            changes=NamedTuple{(kind,)}((new.-reconstructed_old.+escaped,))
        end
        _measure_commit!(u,op,name,new,changes,losses,p,t,m,defects)
    end
end
function _sectional_measure_step!(u,op,name,p,t,h,m,defects)
    pop=getproperty(op.system.populations,name);N=copy(population(op,u,name))
    ctx=_measure_context(op,u,p,t,name)
    h<=suggest_dt(pop,reshape(N,:,1),p,t;cfl=m.cfl,context=ctx.system) || _measure_restriction("sectional upwind/source CFL bound")
    # Evaluate one selected term at a time, with the same deposition and losses.
    for item in pop.prepared
        _measure_check(op,u,m)
        old=copy(population(op,u,name));ctx=_measure_context(op,u,p,t,name)
        d=zero(old);add_term!(d,old,item.term,item.method,item.cache,pop,p,t,ctx)
        new=old.+h.*d
        all(>=(0),new) || _measure_restriction("sectional depletion bound")
        # Select by name, so two terms of one mechanism are never counted twice.
        oneop=CompiledPBM(pop.model,pop.grid,pop.scheme,pop.bc,pop.domain,pop.fields,(item,))
        losses=[(mech,x,h*rate,mechanism(item.term)) for (mech,x,rate) in _sectional_losses(oneop,old,p,t,ctx)]
        changes=NamedTuple{(mechanism(item.term),)}((new.-old,))
        _measure_commit!(u,op,name,new,changes,losses,p,t,m,defects)
    end
end
function _measure_local_step!(u,op,p,t,h,m,defects)
    _measure_check(op,u,m)
    for name in keys(op.ranges)
        pop=getproperty(op.system.populations,name)
        if pop isa CompiledMomentPBM
            _moment_measure_step!(u,op,name,p,t,h,m,defects)
        else
            _sectional_measure_step!(u,op,name,p,t,h,m,defects)
        end
    end
    # User bulk dynamics is explicit Euler after population/exchange substeps.
    ctx=_system_context(op,u,zero(u),p,t);du=zero(u)
    _add_bulk!(du,op,op.system.bulk_rhs(ctx,p,t));u.+=h.*du
    _measure_check(op,u,m)
end
function _axial_measure_bound(op)
    d=op.domain;g=d.grid;n=length(g);rates=zeros(n)
    for c in 1:n
        if c<n || d.boundary!==:closed;rates[c]+=d.velocity/g.widths[c];end
        c>1 && (rates[c]+=d.dispersion/(g.centers[c]-g.centers[c-1])/g.widths[c])
        c<n && (rates[c]+=d.dispersion/(g.centers[c+1]-g.centers[c])/g.widths[c])
        if d.boundary===:periodic && (c==1 || c==n)
            rates[c]+=d.dispersion/((first(g.widths)+last(g.widths))/2)/g.widths[c]
        end
    end
    maximum(rates)==0 ? Inf : 1/maximum(rates)
end
function _validate_measure_operator(op::CompiledAxialSystem,m)
    op.scheme isa Upwind || throw(ArgumentError("joint axial realizability guarantee requires Upwind; independent high-order moment reconstruction is excluded"))
    for localop in op.local_operators
        _validate_measure_operator(localop,m)
        for pop in values(localop.system.populations)
            if pop isa CompiledMomentPBM
                b=pop.boundary
                (b.left===:error && b.right===:error) || throw(ArgumentError("axial size-boundary events/extinction are not implemented by MeasureEuler"))
            else
                (pop.bc.left isa NoFlux && pop.bc.right isa NoFlux) || throw(ArgumentError("axial size escape/extinction is not implemented by MeasureEuler"))
            end
        end
    end
end
function _measure_check(op::CompiledAxialSystem,u,m)
    ranks=Dict{Symbol,Int}()
    for c in axes(u,2)
        for (name,r) in _measure_check(op.local_operators[c],view(u,:,c),m)
            ranks[Symbol(name,"_cell_",c)]=r
        end
    end
    ranks
end
function _measure_local_step!(u,op::CompiledAxialSystem,p,t,h,m,defects)
    h<=m.cfl*_axial_measure_bound(op) || _measure_restriction("axial convex-mixture CFL bound")
    if op.domain.boundary===:danckwerts
        feed=_packed_inlet(op,p,t);s=first(op.local_operators)
        for name in keys(s.ranges)
            pop=getproperty(s.system.populations,name)
            pop isa CompiledMomentPBM && _measure_quadrature(pop,view(feed,getproperty(s.ranges,name)),m)
        end
    end
    for c in axes(u,2)
        cell_defects=_MeasureStatistics()
        _measure_local_step!(view(u,:,c),op.local_operators[c],p,t,h,m,cell_defects)
        for (name,value) in cell_defects.reconstruction
            key=Symbol(name,"_cell_",c)
            defects.reconstruction[key]=get(defects.reconstruction,key,zero(value)).+value
        end
        for ((name,kind),value) in cell_defects.number
            key=(Symbol(name,"_cell_",c),kind)
            defects.number[key]=get(defects.number,key,0.0)+value
        end
    end
    F=axial_fluxes(op,u,p,t)
    for c in axes(u,2),i in axes(u,1)
        u[i,c]+=h*(F[i,c]-F[i,c+1])/op.domain.grid.widths[c]
    end
    _measure_check(op,u,m)
    (inlet=h.*F[:,1],outlet=h.*F[:,end])
end
_measure_dual(x)=false
_measure_dual(x::ForwardDiff.Dual)=true
_measure_dual(x::Union{NamedTuple,Tuple,AbstractArray})=any(_measure_dual,x)
function _simulate(prob::PBMProblem,m::MeasureEuler;dt,saveat,check_nonnegative,negative_tolerance,maxsteps)
    (_measure_dual(prob.u0) || _measure_dual(prob.p)) && throw(ArgumentError("MeasureEuler sensitivities, including event derivatives, are not implemented; use a verified differentiable residual path"))
    check_nonnegative || throw(ArgumentError("MeasureEuler requires all physical stage checks"))
    if prob.operator isa Union{CompiledPBM,CompiledMomentPBM}
        s=PBMSystem((particles=prob.operator,))
        pb=PBMProblem(s,(particles=vec(prob.u0),),prob.tspan;p=prob.p)
        sol=_simulate(pb,m;dt,saveat,check_nonnegative,negative_tolerance,maxsteps)
        return PBMSolution(sol.t,[reshape(copy(u),size(prob.u0)) for u in sol.u],prob,sol.steps,sol.integration_diagnostics)
    end
    prob.operator isa Union{CompiledSystem,CompiledAxialSystem} || throw(ArgumentError("MeasureEuler requires a batch or coupled axial system"))
    op=prob.operator;_validate_measure_operator(op,m)
    ts=_evolution_args(prob,dt,saveat,negative_tolerance,maxsteps)
    u=copy(prob.u0);t=first(prob.tspan);steps=0;rejected=0;failures=0;checks=0
    ranks=_measure_check(op,u,m);states=typeof(u)[];rank_history=typeof(ranks)[]
    defects=_MeasureStatistics()
    reactor_integrals=op isa CompiledAxialSystem ? (inlet=zeros(size(u,1)),outlet=zeros(size(u,1))) : nothing
    for target in ts
        while t<target
            h=min(dt,target-t);attempts=0
            while true
                candidate=copy(u);candidate_defects=deepcopy(defects)
                try
                    flux=_measure_local_step!(candidate,op,prob.p,t,h,m,candidate_defects)
                    ranks=_measure_check(op,candidate,m);checks+=1
                    if reactor_integrals!==nothing
                        reactor_integrals.inlet .+= flux.inlet
                        reactor_integrals.outlet .+= flux.outlet
                    end
                    u=candidate;defects=candidate_defects;break
                catch err
                    err isa Union{_MeasureRestriction,NumericalFailure} || rethrow()
                    rejected+=1;attempts+=1
                    err isa NumericalFailure && (failures+=1)
                    attempts<=m.max_rejections || throw(NumericalFailure("MeasureEuler exhausted step rejections: $(sprint(showerror,err))"))
                    h/=2
                    t+h>t || throw(NumericalFailure("MeasureEuler cannot advance: $(sprint(showerror,err))"))
                end
            end
            steps+=1;steps<=maxsteps || throw(NumericalFailure("maximum MeasureEuler steps exceeded"))
            t=h==target-t ? target : t+h
        end
        push!(states,copy(u));push!(rank_history,copy(ranks))
    end
    info=(policy=:MeasureEuler,order=1,accepted_steps=steps,rejected_steps=rejected,
        admissibility_failures=failures,accepted_state_checks=checks,
        active_rank=rank_history,reconstruction_defect=defects.reconstruction,number_changes=defects.number,reactor_flux_integrals=reactor_integrals,
        reconstruction_rtol=m.reconstruction_rtol,event_rtol=m.event_rtol,cfl=m.cfl)
    PBMSolution(ts,states,prob,steps,info)
end
