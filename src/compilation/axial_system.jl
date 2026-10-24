struct CompiledAxialSystem{D,O,S,R}
    domain::D
    local_operators::O
    scheme::S
    reconstruction::R
end

"""
    axial_problem(local_problem, domain::AxialDispersion; initial=nothing,
                  axial_scheme=Upwind())

Transport every population and scalar bulk balance of a local PBMSystem along
an axial mesh. Local mechanisms and exchanges are evaluated simultaneously in
each cell. The initial matrix has packed local states in rows and axial cells
in columns; by default repeat local_problem.u0. domain.inlet(p,t) returns either
that packed vector or `(populations=(name=bin_numbers,...),bulk=(name=value,...))`.
Danckwerts inlet fluxes and zero-gradient outlets apply to all components.
Supports 1D sectional populations and raw moments, with residual-based methods.
Direct quadrature states cannot be transported as if they were conserved raw
moments and are rejected. Velocity and dispersion are prescribed, not solved.
"""
function axial_problem(local_problem::PBMProblem{<:CompiledSystem},domain::AxialDispersion;
                       initial=nothing,axial_scheme=Upwind())
    axial_scheme isa AbstractReconstruction || throw(ArgumentError("axial_scheme must be a residual reconstruction"))
    _validate_reconstruction(axial_scheme,domain.grid)
    periodic=domain.boundary===:periodic
    bc=periodic ? SizeBC(Periodic(),Periodic()) : SizeBC()
    reconstruction=_prepare_reconstruction(axial_scheme,domain.grid,bc)
    localop=local_problem.operator
    any(pop->any(term->term isa SizeDiffusion,pop.model.terms),values(localop.system.populations)) &&
        throw(ArgumentError("SizeDiffusion is currently supported in batch populations only, not axial adapters"))
    all(values(localop.system.populations)) do op
        op isa CompiledPBM || (op isa CompiledMomentPBM && op.closure isa AbstractMomentClosure)
    end || throw(ArgumentError("axial systems support fixed sectional or raw-moment populations; direct quadrature states require different transport equations"))
    locals=map(domain.grid.centers) do z
        s=localop.system
        sys=PBMSystem(s.populations;bulk_rhs=s.bulk_rhs,interactions=s.interactions,
            exchanges=s.exchanges,fields=(p,t)->merge(s.fields(p,t),(z=z,)),
            nonnegative_bulk=s.nonnegative_bulk,accounting=s.accounting)
        CompiledSystem(sys,localop.ranges,localop.bulk_indices,localop.length,localop.physical_length,localop.ledger_indices)
    end
    op=CompiledAxialSystem(domain,locals,axial_scheme,reconstruction)
    u=initial===nothing ? repeat(local_problem.u0,1,length(domain.grid)) : initial
    PBMProblem(op,u,local_problem.tspan;p=local_problem.p)
end
function PBMProblem(op::CompiledAxialSystem,u::AbstractMatrix,tspan;p=NamedTuple())
    localop=first(op.local_operators)
    size(u)==(localop.length,length(op.domain.grid)) || throw(DimensionMismatch("expected packed local states × axial cells"))
    for c in axes(u,2)
        PBMProblem(op.local_operators[c],view(u,:,c),tspan;p)
    end
    PBMProblem(op,float.(copy(u)),(float(tspan[1]),float(tspan[2])),p)
end
population(op::CompiledAxialSystem,u,name::Symbol)=view(u,getproperty(first(op.local_operators).ranges,name),:)
bulk(op::CompiledAxialSystem,u,name::Symbol)=view(u,getproperty(first(op.local_operators).bulk_indices,name),:)
"""Named local population and bulk views at one axial cell of an axial PBMSystem."""
axial_stateview(prob::PBMProblem,u,cell::Integer)=axial_stateview(prob.operator,u,cell)
axial_stateview(op::CompiledAxialSystem,u,cell::Integer)=stateview(op.local_operators[cell],view(u,:,cell))
function _packed_inlet(op::CompiledAxialSystem,p,t)
    d=op.domain;s=first(op.local_operators)
    feed=d.inlet===nothing ? zeros(s.physical_length) : d.inlet(p,t)
    if feed isa NamedTuple
        Set(keys(feed))==Set((:populations,:bulk)) || throw(ArgumentError("inlet needs populations and bulk NamedTuples"))
        Set(keys(feed.populations))==Set(keys(s.ranges)) && Set(keys(feed.bulk))==Set(keys(s.bulk_indices)) ||
            throw(ArgumentError("inlet population and bulk names must match the local system"))
        all(name->length(getproperty(feed.populations,name))==length(getproperty(s.ranges,name)),keys(s.ranges)) ||
            throw(DimensionMismatch("inlet population length mismatch"))
        feed=vcat((vec(getproperty(feed.populations,name)) for name in keys(s.ranges))...,
                  [getproperty(feed.bulk,name) for name in keys(s.bulk_indices)])
    end
    feed isa AbstractVector && length(feed)==s.physical_length || throw(DimensionMismatch("inlet must provide a complete packed local state"))
    all(x->x isa Real && isfinite(x),feed) || throw(NumericalFailure("inlet values must be finite real scalars"))
    # Enforce realizability for moment feeds, as for initial cell states.
    for name in keys(s.ranges)
        all(>=(0),view(feed,getproperty(s.ranges,name))) || throw(NumericalFailure("inlet population must be nonnegative"))
        _validate_population_state(getproperty(s.system.populations,name),view(feed,getproperty(s.ranges,name)))
    end
    for name in s.system.nonnegative_bulk
        feed[getproperty(s.bulk_indices,name)]>=0 || throw(NumericalFailure("negative constrained inlet bulk value"))
    end
    feed
end
function axial_fluxes(op::CompiledAxialSystem,u,p,t)
    d=op.domain;g=d.grid;n=length(g);v=d.velocity;D=d.dispersion
    size(u)==(first(op.local_operators).length,n) || throw(DimensionMismatch("axial state size mismatch"))
    F=zero(similar(u,size(u,1),n+1));periodic=d.boundary===:periodic
    for i in 1:first(op.local_operators).physical_length
        row=view(u,i,:)
        for c in 2:n
            F[i,c]=v*_face_density(row,c,op.reconstruction,v,periodic,g)-D*(row[c]-row[c-1])/(g.centers[c]-g.centers[c-1])
        end
        if periodic
            F[i,1]=v*_face_density(row,1,op.reconstruction,v,true,g)-D*(row[1]-row[end])/((first(g.widths)+last(g.widths))/2)
            F[i,end]=F[i,1]
        elseif d.boundary===:closed
            F[i,1]=0;F[i,end]=0
        else
            F[i,end]=v*row[end]
        end
    end
    if d.boundary===:danckwerts
        feed=_packed_inlet(op,p,t)
        F[1:first(op.local_operators).physical_length,1].=v.*feed
    end
    all(isfinite,F) || throw(NumericalFailure("nonfinite axial system flux"))
    F
end
function rhs!(du::AbstractMatrix,u::AbstractMatrix,op::CompiledAxialSystem,p,t)
    size(du)==size(u) || throw(DimensionMismatch("axial derivative size mismatch"))
    for c in axes(u,2)
        rhs!(view(du,:,c),view(u,:,c),op.local_operators[c],p,t)
    end
    F=axial_fluxes(op,u,p,t)
    for c in axes(u,2),i in axes(u,1)
        du[i,c]+=(F[i,c]-F[i,c+1])/op.domain.grid.widths[c]
    end
    all(isfinite,du) || throw(NumericalFailure("nonfinite axial system derivative"))
    nothing
end
function _check_problem_state(prob::PBMProblem{<:CompiledAxialSystem},u,tolerance)
    for c in axes(u,2)
        localprob=PBMProblem(prob.operator.local_operators[c],view(u,:,c),prob.tspan,prob.p)
        _check_problem_state(localprob,view(u,:,c),tolerance)
    end
    nothing
end
function _check_admissible_state(op::CompiledAxialSystem,u)
    for c in axes(u,2)
        _check_admissible_state(op.local_operators[c],view(u,:,c))
    end
    nothing
end
