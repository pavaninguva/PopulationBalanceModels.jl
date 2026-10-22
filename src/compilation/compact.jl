"""
    HOCFiniteVolume(; time_order=2)

Conservative compact transport of sectional bin integrals, in one or two
coordinates. Fourth-order interior spatial reconstruction, midpoint time
integration of order two, and symmetric directional/source composition.
Select `compile(model, grid; evolution=HOCFiniteVolume(), bc, p)`.

Requires uniform axes (at least six cells), positive declared growth laws,
and periodic or physical inflow/outflow boundaries. Supports prescribed point
nucleation. Unknown callbacks require `HOCGrowthLaw` / `PrescribedRate`
contracts. No collisions, removal, axial transport, or evolving bulk coupling.
This independently derived finite-volume adaptation is not the printed
Zhang/Li nodal update. It has no positivity guarantee; the usual explicit
nonnegativity check remains enabled. See Compact-method verification.
"""
struct HOCFiniteVolume
    function HOCFiniteVolume(;time_order=2)
        time_order==2 || throw(ArgumentError("HOCFiniteVolume has verified time order 2; use HOCADI(time_order=4) only for homogeneous periodic constant 2D growth"))
        new()
    end
end

struct CompiledHOC{M,G,B,F,R,E,P}
    model::M
    grid::G
    bc::B
    fields::F
    rates::R
    evolution::E
    prepared::P
end
_default_evolution(op::CompiledHOC)=op.evolution
_hoc_axes(g::Grid1D)=(g,)
_hoc_axes(g::Grid2D)=(g.x,g.y)
_hoc_boundaries(b::SizeBC)=(b,)
_hoc_boundaries(b::SizeBC2D)=(b.x,b.y)
_hoc_spatially_constant(law::AffineGrowth,p)=iszero(_affine_coefficient(law.slope,p))
_hoc_spatially_constant(law::ExponentialGrowth,p)=iszero(_affine_coefficient(law.exponent,p))
_hoc_spatially_constant(law::TimeScaledGrowth,p)=_hoc_spatially_constant(law.law,p)
_hoc_spatially_constant(law::HOCGrowthLaw,p)=!(:coordinate in law.dependencies)

function _hoc_dependencies(dependencies,allowed,description)
    bad=setdiff(dependencies,allowed)
    isempty(bad) || throw(ArgumentError("HOC $description declares unsupported dependencies $(Tuple(bad)); use own-coordinate separable prescribed laws, or a residual method such as WENO5"))
end
function _hoc_prescribed(rate,description)
    if rate isa PrescribedRate
        _hoc_dependencies(rate.dependencies,(:time,:parameters,:fields),description)
    elseif !(rate isa Real && isfinite(rate))
        throw(ArgumentError("HOC $description needs a finite constant or PrescribedRate(callback); an unclassified callback cannot certify independence from populations and bulk states"))
    end
end

function _hoc_law_structure(law,axis,bc,p,description)
    if law isa AffineGrowth
        b,s=_positive_affine(law,first(axis.edges),last(axis.edges),p;description="HOC $description")
        all(x->isfinite(b+s*x),(first(axis.edges),last(axis.edges))) ||
            throw(ArgumentError("HOC $description overflows the finite growth range on this domain; rescale coordinates or parameters"))
        bc.left isa Periodic && !iszero(_affine_coefficient(law.slope,p)) &&
            throw(ArgumentError("HOC $description: a nonconstant affine law has no smooth periodic extension; use physical boundaries"))
    elseif law isa ExponentialGrowth
        a=_affine_coefficient(law.amplitude,p); b=_affine_coefficient(law.exponent,p)
        a>0 && all(x->isfinite(a*exp(b*x)) && a*exp(b*x)>0,(first(axis.edges),last(axis.edges))) ||
            throw(ArgumentError("HOC $description requires finite strictly positive exponential growth on the domain"))
        bc.left isa Periodic && !iszero(b) &&
            throw(ArgumentError("HOC $description: nonconstant exponential growth requires physical boundaries"))
    elseif law isa TimeScaledGrowth
        law.law isa Union{AffineGrowth,ExponentialGrowth} ||
            throw(ArgumentError("HOC TimeScaledGrowth needs an autonomous AffineGrowth or ExponentialGrowth spatial law; use HOCGrowthLaw for a declared general law"))
        _hoc_law_structure(law.law,axis,bc,p,description)
        _hoc_prescribed(law.factor,description*" time factor")
    elseif law isa HOCGrowthLaw
        _hoc_dependencies(law.dependencies,(:coordinate,:time,:parameters,:fields),description)
        bc.left isa Periodic && !law.periodic &&
            throw(ArgumentError("HOC $description requires a declared smooth periodic law (periodic=true), or physical boundaries"))
    else
        throw(ArgumentError("HOC $description has an unclassified growth callback; use AffineGrowth, ExponentialGrowth, TimeScaledGrowth, or HOCGrowthLaw with an explicit positive smooth-law contract"))
    end
end

function _validate_hoc(model,grid,method::HOCFiniteVolume,bc,p)
    grid isa Union{Grid1D,Grid2D} || throw(ArgumentError("HOCFiniteVolume requires sectional Grid1D or Grid2D bin integrals"))
    (grid isa Grid1D ? bc isa SizeBC : bc isa SizeBC2D) || throw(ArgumentError("HOC boundary dimension does not match grid"))
    axes=_hoc_axes(grid); boundaries=_hoc_boundaries(bc)
    for (axis,boundary) in zip(axes,boundaries)
        _uniform(axis) && length(axis)>=6 || throw(ArgumentError("HOC requires uniform rectangular axes with at least six bins per axis; use WENO3/Upwind on nonuniform grids"))
        if !(boundary.left isa Periodic)
            boundary.left isa Union{OpenBoundary,NoFlux,InflowFlux} && boundary.right isa OpenBoundary ||
                throw(ArgumentError("positive-growth HOC physical boundaries require left OpenBoundary/NoFlux/InflowFlux and right OpenBoundary; absorption and closed outflow are unsupported"))
            boundary.left isa InflowFlux && _hoc_prescribed(boundary.left.rate,"incoming boundary flux")
        end
    end
    expected=grid isa Grid1D ? Growth : Growth2D
    growth=findall(t->t isa expected,model.terms)
    length(growth)==1 || throw(ArgumentError("HOC requires exactly one $(expected) term; combine the velocities in that term"))
    i=only(growth); rate=model.terms[i].rate; name=model.names[i]
    rates=if grid isa Grid1D
        (rate,)
    elseif rate isa Tuple && length(rate)==2 && all(v->v isa Real && isfinite(v),rate)
        (AffineGrowth(rate[1],0),AffineGrowth(rate[2],0))
    elseif rate isa SeparableGrowth2D
        (rate.x,rate.y)
    else
        throw(ArgumentError("HOC term $name requires G₁(x) and G₂(y), optionally prescribed in time: use a constant tuple or SeparableGrowth2D; general/cross-coordinate callbacks require a residual method"))
    end
    for (d,(law,axis,boundary)) in enumerate(zip(rates,axes,boundaries))
        _hoc_law_structure(law,axis,boundary,p,"term $name, direction $d")
    end
    for (name,term) in zip(model.names,model.terms)
        term isa expected && continue
        supported=grid isa Grid1D ? term isa Nucleation : term isa Nucleation2D
        supported || throw(ArgumentError("HOC term $name ($(typeof(term))) is unsupported: only growth and prescribed point nucleation can be combined; use residual methods for other mechanisms/coupling"))
        _hoc_prescribed(term.rate,"nucleation term $name")
        position=grid isa Grid1D ? (term.size,) : term.size
        all(first(a.edges)<=x<=last(a.edges) for (x,a) in zip(position,axes)) ||
            throw(ArgumentError("HOC nucleation term $name lies outside the size domain; no silent source discard"))
        any(x==last(a.edges) && !(b.left isa Periodic) for (x,a,b) in zip(position,axes,boundaries)) &&
            throw(ArgumentError("HOC nucleation term $name is on a positive-growth outflow face; move it inside the domain or use an explicit unresolved-source model"))
        # A boundary point and an incoming face flux are different models. Do
        # not accept an ambiguous double specification of the same birth.
        any(x==first(a.edges) && b.left isa InflowFlux for (x,a,b) in zip(position,axes,boundaries)) &&
            throw(ArgumentError("HOC term $name touches an InflowFlux boundary; specify point births or the incoming flux, not both at that location"))
    end
    rates
end

function _validate_hoc(model,grid,method::HOCADI,bc,p)
    grid isa Grid2D && bc isa SizeBC2D || throw(ArgumentError("HOCADI requires a 2D sectional grid; use HOCFiniteVolume for 1D"))
    all(a->_uniform(a) && length(a)>=5,_hoc_axes(grid)) || throw(ArgumentError("HOCADI requires uniform axes with at least five bins"))
    all(b->b.left isa Periodic,_hoc_boundaries(bc)) || throw(ArgumentError("HOCADI requires periodic boundaries; use HOCFiniteVolume for physical inflow/outflow"))
    length(model.terms)==1 && only(model.terms) isa Growth2D && only(model.terms).rate isa Tuple ||
        throw(ArgumentError("HOCADI requires only Growth2D((gx,gy)); use HOCFiniteVolume for declared variable/time-dependent growth and nucleation"))
    _validate_velocity(only(model.terms).rate)
    only(model.terms).rate
end

function _compile_evolution(model,grid,method::Union{HOCFiniteVolume,HOCADI},bc,p;fields,methods)
    methods isa TermMethods && isempty(methods.defaults) && isempty(methods.by_term) ||
        throw(ArgumentError("HOC is a complete evolution policy; omit per-term residual methods"))
    rates=_validate_hoc(model,grid,method,bc,p)
    prepared=Tuple(PreparedTerm(name,term,method,nothing) for (name,term) in zip(model.names,model.terms))
    CompiledHOC(model,grid,bc,fields,rates,method,prepared)
end

_hoc_context(op,t)=(;grid=op.grid,operator=op,N=nothing,cell=1,fields=op.fields(1,t),
    bulk=(;),populations=(;),system=nothing)
_hoc_rate(rate,p,t,ctx)=rate isa Real ? rate : rate(p,t,ctx)

function _hoc_velocities(law,axis,p,t,ctx)
    bound=if law isa HOCGrowthLaw
        _hoc_rate(law.lower_bound,p,t,ctx)
    elseif law isa TimeScaledGrowth
        factor=law.factor(p,t,ctx)
        factor isa Real && isfinite(factor) && factor>0 || throw(NumericalFailure("HOC time factor is not finite and strictly positive at t=$t; signed/vanishing growth requires a different method"))
        zero(factor)
    else
        0
    end
    bound isa Real && isfinite(bound) && (law isa HOCGrowthLaw ? bound>0 : bound>=0) ||
        throw(NumericalFailure("HOCGrowthLaw needs a finite positive domain lower bound at t=$t"))
    v=[law(x,p,t,ctx) for x in axis.edges]
    all(x->x isa Real && isfinite(x) && x>0 && x>=bound,v) ||
        throw(NumericalFailure("HOC growth violates finite positive growth or its declared domain lower bound at t=$t; check parameters, fields and law contract"))
    v
end

function _hoc_check_prescribed_values(op,p,t,ctx)
    for (name,term) in zip(op.model.names,op.model.terms)
        term isa Union{Nucleation,Nucleation2D} || continue
        value=_hoc_rate(term.rate,p,t,ctx)
        value isa Real && isfinite(value) && value>=0 ||
            throw(NumericalFailure("HOC nucleation term $name must return a finite nonnegative real rate at t=$t"))
    end
    for boundary in _hoc_boundaries(op.bc)
        boundary.left isa InflowFlux || continue
        value=_hoc_rate(boundary.left.rate,p,t,ctx)
        value isa Real && isfinite(value) && value>=0 ||
            throw(NumericalFailure("HOC incoming flux must be a finite nonnegative real rate at t=$t"))
    end
end

function PBMProblem(op::CompiledHOC,u0,tspan;p=NamedTuple())
    _validate_hoc(op.model,op.grid,op.evolution,op.bc,p)
    N=op.grid isa Grid1D ? reshape(float.(copy(u0)),:,1) : float.(copy(u0))
    expected=op.grid isa Grid1D ? (length(op.grid),1) : size(op.grid)
    size(N)==expected || throw(DimensionMismatch("HOC needs one bin-integrated number per size cell"))
    all(isfinite,N) && all(>=(0),N) || throw(ArgumentError("initial bin populations must be finite and nonnegative"))
    length(tspan)==2 && all(isfinite,tspan) && tspan[2]>tspan[1] || throw(ArgumentError("invalid time span"))
    if op.evolution isa HOCFiniteVolume
        ctx=_hoc_context(op,first(tspan))
        for (law,axis) in zip(op.rates,_hoc_axes(op.grid))
            _hoc_velocities(law,axis,p,first(tspan),ctx)
        end
        _hoc_check_prescribed_values(op,p,first(tspan),ctx)
    end
    PBMProblem(op,N,(float(tspan[1]),float(tspan[2])),p)
end
rhs!(dN,N,op::CompiledHOC,p,t)=throw(ArgumentError("HOC is a fully discrete method: use simulate; external ODE solvers and event/AD sensitivities are not supported"))

"""Current-coefficient Courant step estimate for compact transport; time-varying laws are checked again during evolution."""
function suggest_dt(op::CompiledHOC,N,p,t;cfl=.5)
    0<cfl<=1 || throw(ArgumentError("cfl must be in (0,1]"))
    if op.evolution isa HOCADI
        return minimum(cfl*first(axis.widths)/abs(v) for (axis,v) in zip(_hoc_axes(op.grid),op.rates))
    end
    ctx=_hoc_context(op,t)
    minimum(cfl*first(axis.widths)/maximum(_hoc_velocities(law,axis,p,t,ctx))
        for (law,axis) in zip(op.rates,_hoc_axes(op.grid)))
end
