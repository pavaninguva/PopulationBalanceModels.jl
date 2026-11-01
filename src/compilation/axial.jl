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
    DensityInlet(density; quadrature=GaussLegendre(4))

Continuous number concentration density for a standalone Grid1D/Grid2D axial
population. `density(v,p,t)` returns number/(fluid volume × intrinsic coordinate);
on Grid2D use `density((x,y),p,t)` per product coordinate measure. Compilation
prepares bin quadrature geometry; every residual evaluates current p and t.
Grid2D uses tensor-product Gauss–Legendre quadrature, with the given order per
axis. Values must be finite, nonnegative real scalars.

The density is integrated only over the represented intrinsic domain, without
renormalization. Supply an explicitly normalized truncated distribution if that
is the intended feed. The reactor multiplies the resulting bin concentrations
by axial velocity once to form the Danckwerts influx. A bare `inlet(p,t)` still
returns already integrated bin concentrations. DensityInlet is not supported
for coupled-system or moment feeds; those retain their explicit packed/named
inlet contracts.
"""
struct DensityInlet{F,Q}
    density::F
    quadrature::Q
    function DensityInlet(density;quadrature=GaussLegendre(4))
        quadrature isa GaussLegendre || throw(ArgumentError("DensityInlet quadrature must be GaussLegendre(order)"))
        new{typeof(density),typeof(quadrature)}(density,quadrature)
    end
end

"""
    AxialDispersion(grid::AxialGrid; velocity, dispersion, inlet=nothing,
                    boundary=:danckwerts)

Prescribed constant nonnegative axial velocity and dispersion coefficient.
`inlet(p,t)` returns intrinsic-bin concentrations (not densities or fluxes).
For a standalone directly discretized population, `inlet=DensityInlet(f)` instead
integrates the continuous density `f(v,p,t)` (coordinate tuple on Grid2D).
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

struct _PreparedDensityInlet{F,A}
    density::F
    axes::A
end
function _density_inlet_axis(g,q)
    # Geometry is immutable during a solve. Only density values are evaluated
    # again, so parameter/time derivatives are never stored in a Float64 cache.
    nodes=[g.centers[i]+g.widths[i]/2*x for x in q.nodes,i in eachindex(g.centers)]
    weights=[g.widths[i]/2*w for w in q.weights,i in eachindex(g.centers)]
    (;nodes,weights)
end
_prepare_axial_inlet(inlet,grid)=inlet
function _prepare_axial_inlet(inlet::DensityInlet,grid::Grid1D)
    q=_gauss_legendre(inlet.quadrature)
    _PreparedDensityInlet(inlet.density,(_density_inlet_axis(grid,q),))
end
function _prepare_axial_inlet(inlet::DensityInlet,grid::Grid2D)
    q=_gauss_legendre(inlet.quadrature)
    _PreparedDensityInlet(inlet.density,(_density_inlet_axis(grid.x,q),_density_inlet_axis(grid.y,q)))
end
_prepare_axial_inlet(::DensityInlet,grid)=throw(ArgumentError("DensityInlet requires a standalone Grid1D or Grid2D axial population"))
function _inlet_density(f,x,p,t)
    value=f(x,p,t)
    value isa Real && isfinite(value) && value>=0 ||
        throw(NumericalFailure("DensityInlet must return a finite nonnegative real density at $x"))
    value
end
_evaluate_axial_inlet(::Nothing,p,t)=nothing
_evaluate_axial_inlet(inlet,p,t)=inlet(p,t)
function _evaluate_axial_inlet(inlet::_PreparedDensityInlet{F,<:Tuple{A}},p,t) where {F,A}
    a=only(inlet.axes)
    [sum(a.weights[k,i]*_inlet_density(inlet.density,a.nodes[k,i],p,t) for k in axes(a.nodes,1)) for i in axes(a.nodes,2)]
end
function _evaluate_axial_inlet(inlet::_PreparedDensityInlet{F,<:Tuple{A,B}},p,t) where {F,A,B}
    a,b=inlet.axes
    [sum(a.weights[k,i]*b.weights[l,j]*_inlet_density(inlet.density,(a.nodes[k,i],b.nodes[l,j]),p,t)
        for k in axes(a.nodes,1),l in axes(b.nodes,1)) for i in axes(a.nodes,2),j in axes(b.nodes,2)]
end

function _prepare_axial_reconstruction(scheme,domain)
    scheme isa AbstractReconstruction || throw(ArgumentError("axial_scheme must be a residual reconstruction"))
    bc=domain.boundary===:periodic ? SizeBC(Periodic(),Periodic()) : SizeBC()
    _prepare_reconstruction(scheme,domain.grid,bc)
end

struct CompiledAxial{G,D,O,P,S,R,I}
    grid::G
    domain::D
    local_operators::O
    prepared::P
    scheme::S
    reconstruction::R
    prepared_inlet::I
end
_compile_axial(model,grid,domain;kwargs...)=throw(ArgumentError("unsupported population domain $(typeof(domain))"))
function _compile_axial(model,grid,domain::AxialDispersion;growth_scheme,bc,fields,methods,evolution,p,axial_scheme=Upwind())
    evolution===nothing || throw(ArgumentError("axial transport requires residual-based local methods, not a direct evolution scheme"))
    reconstruction=_prepare_axial_reconstruction(axial_scheme,domain)
    inlet=_prepare_axial_inlet(domain.inlet,grid)
    # Separate local operators preserve ordinary intrinsic callbacks and caches.
    # The prescribed axial position is available as ctx.fields.z.
    locals=map(eachindex(domain.grid.centers)) do c
        localfields=(unused,t)->merge(fields(c,t),(z=domain.grid.centers[c],))
        compile(model,grid;growth_scheme,bc,fields=localfields,methods,p)
    end
    CompiledAxial(grid,domain,locals,first(locals).prepared,axial_scheme,reconstruction,inlet)
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

function _axial_row_fluxes!(F,N,d,reconstruction,physical_length)
    g=d.grid;n=length(g);v=d.velocity;D=d.dispersion;periodic=d.boundary===:periodic
    for i in 1:physical_length
        row=view(N,i,:)
        for c in 2:n
            F[i,c]=v*_face_density(row,c,reconstruction,v,periodic,g)-D*(row[c]-row[c-1])/(g.centers[c]-g.centers[c-1])
        end
        if periodic
            F[i,1]=v*_face_density(row,1,reconstruction,v,true,g)-D*(row[1]-row[end])/((first(g.widths)+last(g.widths))/2)
            F[i,end]=F[i,1]
        elseif d.boundary===:closed
            F[i,1]=0;F[i,end]=0
        else
            # The caller supplies the total Danckwerts influx separately.
            F[i,end]=v*row[end]
        end
    end
    F
end

function _axial_flux_type(N,d,feed)
    T=promote_type(eltype(N),typeof(d.velocity),typeof(d.dispersion))
    feed===nothing && return T
    # Abstractly typed legacy feed arrays are still valid. Use their actual
    # scalar types, including Duals, instead of allocating an Any-valued flux.
    isconcretetype(eltype(feed)) ? promote_type(T,eltype(feed)) :
        foldl((S,x)->promote_type(S,typeof(x)),feed;init=T)
end

"""Axial total face fluxes, intrinsic bins × (axial cells+1), positive downstream.
Flux units are bin number concentration × axial length/time. Axial reconstruction
acts on axial cell averages; dispersion retains the conservative two-point flux."""
function axial_fluxes(op::CompiledAxial,N,p,t)
    d=op.domain;n=length(d.grid);v=d.velocity
    size(N)==(length(op.grid),n) || throw(DimensionMismatch("axial state size mismatch"))
    feed=d.boundary===:danckwerts ? _evaluate_axial_inlet(op.prepared_inlet,p,t) : nothing
    if feed!==nothing
        feed isa AbstractArray && (size(feed)==_axial_shape(first(op.local_operators)) || size(feed)==(length(op.grid),)) ||
            throw(DimensionMismatch("inlet must return intrinsic bin concentrations as a vector or the intrinsic grid array"))
        all(x->x isa Real && isfinite(x) && x>=0,feed) || throw(NumericalFailure("inlet concentrations must be finite and nonnegative"))
    end
    T=_axial_flux_type(N,d,feed)
    F=similar(N,T,size(N,1),n+1)
    _axial_row_fluxes!(F,N,d,op.reconstruction,length(op.grid))
    if d.boundary===:danckwerts
        for i in axes(N,1)
            F[i,1]=feed===nothing ? zero(v) : v*feed[i]
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
