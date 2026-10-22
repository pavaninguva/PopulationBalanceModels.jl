# Conservative compact face interpolation. With U_i=N_i/h and face densities f_j:
#   f_{j-1}/4 + f_j + f_{j+1}/4 = 3(U_j+U_{j+1})/4.
# This identity is exact through degree three for CELL AVERAGES. At outflow,
# a degree-four polynomial fitted to the last five cell integrals supplies f_n.
# See methods/compact_verification.md for the derivation and time update.
const _HOC_OUTFLOW=(137//60,-163//60,137//60,-21//20,1//5)

# Fixed-rank corrections to a tridiagonal LU. Both the periodic corner entries
# and the one-sided outflow row have bounded rank. Setup/storage/solve are O(n).
struct _HOCFactor{F,Z,V,C,W}
    factor::F
    Z::Z
    V::V
    correction::C
    scratch::W
end
function _hoc_factor(lower,diagonal,upper,rows,V)
    T=eltype(diagonal); n=length(diagonal); r=length(rows)
    all(a->all(isfinite,a),(lower,diagonal,upper,V)) ||
        throw(NumericalFailure("nonfinite HOC linear-system coefficients; growth/size/time scaling is unresolved in this scalar type"))
    try
        factor=lu(Tridiagonal(lower,diagonal,upper))
        U=zeros(T,n,r)
        for j in 1:r
            U[rows[j],j]=one(T)
        end
        Z=factor\U
        correction=lu(Matrix{T}(I,r,r)+V*Z)
        _HOCFactor(factor,Z,V,correction,zeros(T,r))
    catch err
        err isa Union{SingularException,ZeroPivotException} || rethrow()
        throw(NumericalFailure("singular HOC directional system; reduce dt and check the growth/boundary assumptions"))
    end
end
function _hoc_solve!(dest,f::_HOCFactor,b)
    copyto!(dest,b)
    ldiv!(f.factor,dest)
    mul!(f.scratch,f.V,dest)
    ldiv!(f.correction,f.scratch)
    correction=f.scratch
    for j in eachindex(correction), i in eachindex(dest)
        dest[i]-=f.Z[i,j]*correction[j]
    end
    all(isfinite,dest) || throw(NumericalFailure("nonfinite HOC face solve; reduce dt and check growth coefficients"))
    dest
end

function _hoc_directional_factor(v,h,δ,periodic,T)
    n=length(v)-1; a=T(3)*δ/(T(8)*h)
    lower=[T(1)/(T(4)*v[j])-a for j in 2:n]
    diagonal=[one(T)/v[j] for j in 2:n+1]
    upper=[T(1)/(T(4)*v[j])+a for j in 3:n+1]
    if periodic
        V=zeros(T,2,n)
        V[1,end]=T(1)/(T(4)*v[end])-a
        V[2,1]=T(1)/(T(4)*v[2])+a
        return _hoc_factor(lower,diagonal,upper,(1,n),V)
    end
    # The last row is f_n=(1/h)sum(c_l*N_{n-l}). Substitute
    # N_mid=N_old-(δ/2)D*F to obtain its banded flux equation.
    lower[end]=zero(T)
    V=zeros(T,1,n)
    for l in 0:4
        i=n-l; c=T(_HOC_OUTFLOW[l+1])*δ/(2h)
        V[1,i]+=c
        V[1,i-1]-=c
    end
    _hoc_factor(lower,diagonal,upper,(n,),V)
end

function _hoc_axis!(out,N,op,dimension,v,δ,p,t,ctx,cache)
    axis=_hoc_axes(op.grid)[dimension]
    boundary=_hoc_boundaries(op.bc)[dimension]
    n=length(axis); h=first(axis.widths); periodic=boundary.left isa Periodic
    T=eltype(N)
    if periodic
        isapprox(v[1],v[end];rtol=100eps(T),atol=0) ||
            throw(NumericalFailure("HOC periodic growth has mismatched endpoint values; check the declared periodic law"))
    end
    previous=cache[]
    if previous===nothing || previous.δ!=δ || previous.v!=v
        factor=_hoc_directional_factor(v,h,δ,periodic,T)
        cache[]=(;δ,v=copy(v),factor,b=zeros(T,n),flux=zeros(T,n))
    end
    # One dynamic dispatch at the cache boundary; scalar line loops specialize
    # on the concrete factor/workspace. No Any-valued arithmetic per cell.
    _hoc_sweep_lines!(out,N,op,dimension,v,δ,p,t,ctx,cache[])
end

function _hoc_sweep_lines!(out,N,op,dimension,v,δ,p,t,ctx,work)
    axis=_hoc_axes(op.grid)[dimension]
    boundary=_hoc_boundaries(op.bc)[dimension]
    n=length(axis); h=first(axis.widths); periodic=boundary.left isa Periodic
    T=eltype(N); b=work.b; flux=work.flux
    lines=dimension==1 ? size(N,2) : size(N,1)
    incoming=boundary.left isa InflowFlux ?
        _hoc_rate(boundary.left.rate,p,t,ctx) : zero(T)
    incoming isa Real && isfinite(incoming) && incoming>=0 || throw(NumericalFailure("HOC incoming number flux must be finite and nonnegative at t=$t"))
    boundary_births=zero(T)
    if op.grid isa Grid1D && !periodic
        for term in op.model.terms
            term isa Nucleation && term.size==first(axis.edges) || continue
            rate=_hoc_rate(term.rate,p,t,ctx)
            rate isa Real && isfinite(rate) && rate>=0 || throw(NumericalFailure("HOC boundary nucleation must be finite and nonnegative at t=$t"))
            boundary_births+=rate
        end
    end
    input=zero(T); output=zero(T)
    for j in 1:lines
        src=dimension==1 ? view(N,:,j) : view(N,j,:)
        dest=dimension==1 ? view(out,:,j) : view(out,j,:)
        transverse=op.grid isa Grid1D ? one(T) :
            first(_hoc_axes(op.grid)[3-dimension].widths)
        F0=incoming*transverse+boundary_births
        for i in 1:n-1
            b[i]=T(3)/(4h)*(src[i]+src[i+1])
        end
        if periodic
            b[n]=T(3)/(4h)*(src[n]+src[1])
        else
            lastrow=zero(T)
            for l in 0:4
                lastrow+=T(_HOC_OUTFLOW[l+1])*src[n-l]
            end
            b[n]=lastrow/h
            b[1]-=(one(T)/(4v[1])-3δ/(8h))*F0
        end
        _hoc_solve!(flux,work.factor,b)
        prev=periodic ? flux[end] : F0
        for i in 1:n
            dest[i]=src[i]+δ*(prev-flux[i])
            prev=flux[i]
        end
        if !periodic
            input+=δ*incoming*transverse
            output+=δ*flux[end]
        end
    end
    input,output,δ*boundary_births
end

function _hoc_births!(N,op,p,t,δ)
    ctx=_hoc_context(op,t); total=zero(eltype(N)); axes=_hoc_axes(op.grid)
    for term in op.model.terms
        term isa Union{Nucleation,Nucleation2D} || continue
        # In 1D a point at the incoming endpoint is a number flux, not a
        # cell-centered particle. Its midpoint flux is handled in _hoc_axis!.
        op.grid isa Grid1D && !(op.bc.left isa Periodic) && term.size==first(op.grid.edges) && continue
        rate=_hoc_rate(term.rate,p,t,ctx)
        rate isa Real && isfinite(rate) && rate>=0 || throw(NumericalFailure("HOC nucleation must be finite and nonnegative at t=$t"))
        position=term isa Nucleation ? (term.size,) : term.size
        indices=map((x,a,b)->b.left isa Periodic && x==last(a.edges) ? 1 :
            min(searchsortedlast(a.edges,x),length(a)),position,axes,_hoc_boundaries(op.bc))
        born=δ*rate
        if op.grid isa Grid1D
            N[only(indices),1]+=born
        else
            N[indices...]+=born  # one cell source, including a corner, ONCE
        end
        total+=born
    end
    total
end

function _simulate(prob::PBMProblem,method::HOCFiniteVolume;dt,saveat,check_nonnegative,negative_tolerance,maxsteps)
    op=prob.operator
    if !(op isa CompiledHOC && op.evolution isa HOCFiniteVolume)
        op isa Union{CompiledPBM,CompiledPBM2D,CompiledHOC} || throw(ArgumentError("compile a standalone sectional population with evolution=HOCFiniteVolume(); coupled/axial/moment populations are unsupported"))
        # The same structural validation is used when a policy is chosen later.
        op=_compile_evolution(op.model,op.grid,method,op.bc,prob.p;fields=op.fields,methods=TermMethods())
        prob=PBMProblem(op,prob.u0,prob.tspan;p=prob.p)
    end
    _validate_hoc(op.model,op.grid,method,op.bc,prob.p)
    ts=_evolution_args(prob,dt,saveat,negative_tolerance,maxsteps)
    T=promote_type(eltype(prob.u0),map(a->eltype(a.edges),_hoc_axes(op.grid))...,_numeric_type(prob.p))
    T<:AbstractFloat || throw(ArgumentError("HOC supports real floating scalar types; automatic differentiation through its fully discrete solves is not verified"))
    initial_context=_hoc_context(op,first(prob.tspan))
    for (law,axis) in zip(op.rates,_hoc_axes(op.grid))
        _hoc_velocities(law,axis,prob.p,first(prob.tspan),initial_context)
    end
    _hoc_check_prescribed_values(op,prob.p,first(prob.tspan),initial_context)
    N=T.(prob.u0); next=similar(N); states=typeof(N)[]
    d=length(_hoc_axes(op.grid)); caches=ntuple(_->Ref{Any}(nothing),d)
    input=zeros(T,d); output=zeros(T,d); births=zero(T); initial=sum(N)
    accounting=NamedTuple[]; steps=0; t=first(prob.tspan)
    tolerance=check_nonnegative ? negative_tolerance : Inf
    check(state)=_checkstate(state,tolerance,"HOC bin population",
        "resolve the smooth profile, reduce dt, or use a positivity-preserving method; check_nonnegative=false retains signed dispersive lobes for diagnostics")
    check(N)
    for target in ts
        while t<target
            δ=min(dt,target-t)
            t+δ>t || throw(NumericalFailure("HOC time step cannot advance"))
            steps+=1
            steps<=maxsteps || throw(NumericalFailure("maximum HOC step count exceeded"))
            tm=t+δ/2; ctx=_hoc_context(op,tm)
            velocities=map((law,axis)->_hoc_velocities(law,axis,prob.p,tm,ctx),op.rates,_hoc_axes(op.grid))
            for (law,axis,boundary,v) in zip(op.rates,_hoc_axes(op.grid),_hoc_boundaries(op.bc),velocities)
                unrestricted=boundary.left isa Periodic && _hoc_spatially_constant(law,prob.p)
                unrestricted || maximum(v)*δ/first(axis.widths)<=1+16eps(T) ||
                    throw(NumericalFailure("HOCFiniteVolume's verified physical-boundary/variable-growth envelope requires max(G)*dt/Δx ≤ 1 on each axis; reduce dt (this is not a positivity guarantee)"))
            end
            # Symmetric source/transport/source composition. Midpoint quadrature
            # integrates each prescribed half source at its own quarter time.
            births+=_hoc_births!(N,op,prob.p,t+δ/4,δ/2)
            check(N)
            sweeps=d==1 ? ((1,δ),) : ((1,δ/2),(2,δ),(1,δ/2))
            for (direction,step) in sweeps
                a,b,c=_hoc_axis!(next,N,op,direction,velocities[direction],step,prob.p,tm,ctx,caches[direction])
                check(next)
                N,next=next,N
                input[direction]+=a; output[direction]+=b
                births+=c
            end
            births+=_hoc_births!(N,op,prob.p,t+3δ/4,δ/2)
            check(N)
            t=δ==target-t ? target : t+δ
            # Check parameter/field-dependent restrictions at accepted endpoints,
            # as well as at every coefficient evaluation. No coefficient cache
            # lives on the operator or survives a remake/new simulation.
            for (law,axis) in zip(op.rates,_hoc_axes(op.grid))
                _hoc_velocities(law,axis,prob.p,t,_hoc_context(op,t))
            end
            _hoc_check_prescribed_values(op,prob.p,t,_hoc_context(op,t))
        end
        push!(states,copy(N))
        push!(accounting,(;inflow=Tuple(input),outflow=Tuple(output),births,
            inventory=sum(N),defect=sum(N)-initial-sum(input)-births+sum(output)))
    end
    PBMSolution(ts,states,prob,steps,(;policy=:HOCFiniteVolume,interior_spatial_order=4,time_order=2,
        accepted_steps=steps,rejected_steps=0,number_accounting=accounting))
end
