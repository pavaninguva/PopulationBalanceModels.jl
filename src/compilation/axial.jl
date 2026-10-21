"""Finite-volume physical axial mesh, distinct from the particle-size representation."""
struct AxialGrid{T}
    edges::Vector{T}
    centers::Vector{T}
    widths::Vector{T}
    function AxialGrid(edges)
        e=float.(collect(edges))
        length(e)>=3 && all(isfinite,e) && all(diff(e).>0) ||
            throw(ArgumentError("axial edges must be finite, increasing and define at least two cells"))
        new{eltype(e)}(e,(e[1:end-1].+e[2:end])./2,diff(e))
    end
end
Base.length(g::AxialGrid)=length(g.centers)

"""
    AxialDispersion(grid::AxialGrid; velocity, dispersion, inlet=nothing,
                    boundary=:danckwerts)

Prescribed constant nonnegative axial velocity and dispersion coefficient.
`inlet(p,t)` returns intrinsic-bin concentrations (not densities or fluxes).
The Danckwerts total inlet flux is velocity*inlet; the outlet has zero dispersive
flux. `inlet=nothing` means zero feed. `:periodic` is also supported; `:closed`
requires zero velocity. State shape is (intrinsic bins, axial cells), including
flattened Grid2D bins. No flow-field or bulk-balance solver is introduced.
"""
struct AxialDispersion{G,V,D,I} <: AbstractPopulationDomain
    grid::G
    velocity::V
    dispersion::D
    inlet::I
    boundary::Symbol
    function AxialDispersion(grid::AxialGrid;velocity,dispersion,inlet=nothing,boundary=:danckwerts)
        _nonnegative(velocity,"axial velocity");_nonnegative(dispersion,"axial dispersion")
        boundary in (:danckwerts,:periodic,:closed) || throw(ArgumentError("unknown axial boundary"))
        boundary===:closed && !iszero(velocity) && throw(ArgumentError("closed axial boundaries require zero velocity"))
        boundary!==:danckwerts && inlet!==nothing && throw(ArgumentError("an inlet is only meaningful for Danckwerts boundaries"))
        new{typeof(grid),typeof(velocity),typeof(dispersion),typeof(inlet)}(grid,velocity,dispersion,inlet,boundary)
    end
end

struct CompiledAxial{G,D,O,P}
    grid::G
    domain::D
    local_operators::O
    prepared::P
end
_compile_axial(model,grid,domain;kwargs...)=throw(ArgumentError("unsupported population domain $(typeof(domain))"))
function _compile_axial(model,grid,domain::AxialDispersion;growth_scheme,bc,fields,methods,evolution,p)
    evolution===nothing || throw(ArgumentError("axial transport requires residual-based local methods, not a direct evolution scheme"))
    # Separate local operators preserve ordinary intrinsic callbacks and caches.
    # The prescribed axial position is available as ctx.fields.z.
    locals=map(eachindex(domain.grid.centers)) do c
        localfields=(unused,t)->merge(fields(c,t),(z=domain.grid.centers[c],))
        compile(model,grid;growth_scheme,bc,fields=localfields,methods,p)
    end
    CompiledAxial(grid,domain,locals,first(locals).prepared)
end
_axial_shape(op::CompiledPBM)=(length(op.grid),1)
_axial_shape(op::CompiledPBM2D)=size(op.grid)

"""View the intrinsic state at one axial cell; reshape Grid2D bins to their tensor shape."""
function axial_population(op::CompiledAxial,N,cell::Integer)
    size(N)==(length(op.grid),length(op.domain.grid)) || throw(DimensionMismatch("axial state must be intrinsic bins × axial cells"))
    reshape(view(N,:,cell),_axial_shape(op.local_operators[cell]))
end
function PBMProblem(op::CompiledAxial,u0::AbstractMatrix,tspan;p=NamedTuple())
    size(u0)==(length(op.grid),length(op.domain.grid)) || throw(DimensionMismatch("axial state must be intrinsic bins × axial cells"))
    all(isfinite,u0) && all(>=(0),u0) || throw(ArgumentError("initial numbers must be finite and nonnegative"))
    length(tspan)==2 && all(isfinite,tspan) && tspan[2]>tspan[1] || throw(ArgumentError("invalid time span"))
    PBMProblem(op,float.(copy(u0)),(float(tspan[1]),float(tspan[2])),p)
end

"""Axial total face fluxes, intrinsic bins × (axial cells+1), positive downstream."""
function axial_fluxes(op::CompiledAxial,N,p,t)
    d=op.domain;g=d.grid;n=length(g);v=d.velocity;D=d.dispersion
    size(N)==(length(op.grid),n) || throw(DimensionMismatch("axial state size mismatch"))
    F=similar(N,size(N,1),n+1)
    for c in 2:n,i in axes(N,1)
        F[i,c]=v*N[i,c-1]-D*(N[i,c]-N[i,c-1])/(g.centers[c]-g.centers[c-1])
    end
    if d.boundary===:periodic
        distance=(first(g.widths)+last(g.widths))/2
        for i in axes(N,1)
            F[i,1]=v*N[i,end]-D*(N[i,1]-N[i,end])/distance
            F[i,end]=F[i,1]
        end
    elseif d.boundary===:closed
        F[:,1].=0;F[:,end].=0
    else
        feed=d.inlet===nothing ? nothing : d.inlet(p,t)
        if feed!==nothing
            size(feed)==_axial_shape(first(op.local_operators)) || size(feed)==(length(op.grid),) ||
                throw(DimensionMismatch("inlet must return intrinsic bin concentrations"))
            all(isfinite,feed) && all(>=(0),feed) || throw(NumericalFailure("inlet concentrations must be finite and nonnegative"))
        end
        for i in axes(N,1)
            F[i,1]=feed===nothing ? zero(v) : v*feed[i]
            F[i,end]=v*N[i,end]
        end
    end
    all(isfinite,F) || throw(NumericalFailure("nonfinite axial flux"))
    F
end
function rhs!(dN::AbstractMatrix,N::AbstractMatrix,op::CompiledAxial,p,t)
    size(dN)==size(N) || throw(DimensionMismatch("state and derivative sizes differ"))
    for c in axes(N,2)
        rhs!(axial_population(op,dN,c),axial_population(op,N,c),op.local_operators[c],p,t)
    end
    F=axial_fluxes(op,N,p,t)
    for c in axes(N,2),i in axes(N,1)
        dN[i,c]+=(F[i,c]-F[i,c+1])/op.domain.grid.widths[c]
    end
    all(isfinite,dN) || throw(NumericalFailure("nonfinite axial derivative"))
    nothing
end
function suggest_dt(op::CompiledAxial,N,p,t;cfl=0.5)
    0<cfl<=1 || throw(ArgumentError("cfl must be in (0,1]"))
    d=op.domain;g=d.grid;n=length(g);maxrate=0.0
    for c in 1:n
        localdt=suggest_dt(op.local_operators[c],axial_population(op,N,c),p,t;cfl)
        rate=cfl/localdt+d.velocity/g.widths[c]
        c>1 && (rate+=d.dispersion/(g.widths[c]*(g.centers[c]-g.centers[c-1])))
        c<n && (rate+=d.dispersion/(g.widths[c]*(g.centers[c+1]-g.centers[c])))
        if d.boundary===:periodic && (c==1 || c==n)
            rate+=d.dispersion/(g.widths[c]*(first(g.widths)+last(g.widths))/2)
        end
        maxrate=max(maxrate,rate)
    end
    iszero(maxrate) ? Inf : cfl/maxrate
end
