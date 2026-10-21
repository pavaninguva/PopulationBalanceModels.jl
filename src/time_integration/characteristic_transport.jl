"""
    Inguva2022()

Double-transformation method for homogeneous 1D transport with a positive
autonomous AffineGrowth law (Inguva, Schickel & Braatz, 2022, §2.2.1, fourth
approach). Select with `compile(...; evolution=Inguva2022(),p)`. The physical
mesh must be uniform in travel time. Bin-integrated numbers shift one cell per
step; this is the conservative cell-integral version of the paper's nodal shift.
It is not a new reconstruction for an arbitrary residual or an SSPRK step.
"""
struct Inguva2022 end

"""
    InguvaBraatz2023()

Commuting dimensional shifts after the double transformation for homogeneous
2D transport with SeparableGrowth2D (Inguva & Braatz, 2023, §2.2.1). Both axes
must have the same travel-time spacing. Sources, collisions, bulk feedback,
cross-coordinate growth and time-dependent laws are rejected, not split silently.
"""
struct InguvaBraatz2023 end

_compile_evolution(model,grid,method,bc,p;kwargs...)=_compile_characteristic(model,grid,method,bc,p;kwargs...)

function _positive_affine(g::AffineGrowth,lower,upper,p)
    b=_affine_coefficient(g.offset,p);s=_affine_coefficient(g.slope,p)
    min(b+s*lower,b+s*upper)>0 || throw(ArgumentError("the transformation requires strictly positive growth throughout the size domain"))
    b,s
end
function _travel_time(g::AffineGrowth,a,origin,p)
    b=_affine_coefficient(g.offset,p);s=_affine_coefficient(g.slope,p)
    v=b+s*origin
    iszero(s) ? (a-origin)/v : log1p(s*(a-origin)/v)/s
end
function _travel_inverse(g::AffineGrowth,tau,origin,p)
    b=_affine_coefficient(g.offset,p);s=_affine_coefficient(g.slope,p)
    origin+(iszero(s) ? b*tau : (b+s*origin)*expm1(s*tau)/s)
end

"""
    characteristic_grid(law, bounds; dt, p=(;), coordinate=:length, endpoint=:extend)

Construct physical edges with constant travel-time increment dt. `bounds` is
`(lower,upper)` for AffineGrowth, or a pair of bounds for SeparableGrowth2D.
`endpoint=:extend` includes the requested domain and adds a whole last cell;
`:truncate` omits a fractional last cell; `:exact` requires an aligned endpoint.
Inspect the returned physical edges: no short final cell is inserted. Both axes
of a 2D grid use the SAME dt, which gives CFL=1 in each split subproblem.
"""
function characteristic_grid(g::AffineGrowth,bounds;dt,p=NamedTuple(),coordinate=:length,endpoint=:extend)
    length(bounds)==2 && all(isfinite,bounds) && 0<=bounds[1]<bounds[2] || throw(ArgumentError("invalid characteristic size bounds"))
    isfinite(dt) && dt>0 || throw(ArgumentError("travel-time step must be finite and positive"))
    endpoint in (:extend,:truncate,:exact) || throw(ArgumentError("endpoint must be :extend, :truncate or :exact"))
    lo,hi=bounds;_positive_affine(g,lo,hi,p)
    cells=_travel_time(g,hi,lo,p)/dt
    nearest=round(cells)
    aligned=isapprox(cells,nearest;atol=1e-9,rtol=0)
    n=aligned ? Int(nearest) : endpoint===:extend ? ceil(Int,cells) :
        endpoint===:truncate ? floor(Int,cells) : throw(ArgumentError("upper endpoint does not lie on a whole travel-time cell"))
    n>=2 || throw(ArgumentError("choose a smaller dt to construct at least two characteristic cells"))
    edges=[_travel_inverse(g,i*dt,lo,p) for i in 0:n]
    _positive_affine(g,first(edges),last(edges),p)
    Grid1D(edges;coordinate)
end
function characteristic_grid(g::SeparableGrowth2D,bounds;dt,p=NamedTuple(),coordinates=(:length,:width),endpoint=:extend)
    length(bounds)==length(coordinates)==2 || throw(ArgumentError("two size bounds and coordinate names required"))
    Grid2D(characteristic_grid(g.x,bounds[1];dt,p,coordinate=coordinates[1],endpoint),
           characteristic_grid(g.y,bounds[2];dt,p,coordinate=coordinates[2],endpoint))
end

struct CompiledCharacteristic{M,G,B,R,E,P}
    model::M
    grid::G
    bc::B
    rates::R
    evolution::E
    dt::Float64
    prepared::P
end
_default_evolution(op::CompiledCharacteristic)=op.evolution

function _characteristic_spacing(g,axis,p)
    _positive_affine(g,first(axis.edges),last(axis.edges),p)
    tau=[_travel_time(g,a,first(axis.edges),p) for a in axis.edges]
    steps=diff(tau);h=first(steps)
    h>0 && all(s->isapprox(s,h;rtol=1e-9,atol=1e-12),steps) ||
        throw(ArgumentError("mesh is not uniform in travel time for these parameters; rebuild it with characteristic_grid"))
    h
end
function _characteristic_boundary(law,bc,p)
    if bc.left isa Periodic
        iszero(_affine_coefficient(law.slope,p)) || throw(ArgumentError("periodic characteristic transport requires constant growth"))
    else
        bc.left isa OpenBoundary && bc.right isa OpenBoundary ||
            throw(ArgumentError("characteristic transport requires paired zero-inflow OpenBoundary ends, or periodic constant growth"))
    end
end
function _compile_characteristic(model,grid,method,bc,p;fields,methods)
    fields===_emptyfields || throw(ArgumentError("characteristic methods do not use prescribed field callbacks"))
    methods isa TermMethods && isempty(methods.defaults) && isempty(methods.by_term) ||
        throw(ArgumentError("select evolution alone for characteristic transport; per-term residual methods cannot be combined with it"))
    length(model.terms)==1 || throw(ArgumentError("characteristic methods require homogeneous growth only; sources, losses and collisions are not supported"))
    term=only(model.terms)
    rates,axes,boundaries=if method isa Inguva2022 && grid isa Grid1D && term isa Growth && term.rate isa AffineGrowth
        ((term.rate,),(grid,),(bc,))
    elseif method isa InguvaBraatz2023 && grid isa Grid2D && term isa Growth2D && term.rate isa SeparableGrowth2D
        ((term.rate.x,term.rate.y),(grid.x,grid.y),(bc.x,bc.y))
    else
        throw(ArgumentError("Inguva2022 requires 1D Growth(AffineGrowth(...)); InguvaBraatz2023 requires 2D Growth2D(SeparableGrowth2D(...)). Arbitrary callbacks do not certify the required functional form"))
    end
    spacings=map((law,axis,boundary)->begin
        _characteristic_boundary(law,boundary,p)
        _characteristic_spacing(law,axis,p)
    end,rates,axes,boundaries)
    all(h->isapprox(h,first(spacings);rtol=1e-9,atol=1e-12),spacings) ||
        throw(ArgumentError("both axes must have a common travel-time spacing; build them with a common dt"))
    prepared=(PreparedTerm(only(model.names),term,method,nothing),)
    CompiledCharacteristic(model,grid,bc,rates,method,Float64(first(spacings)),prepared)
end

function PBMProblem(op::CompiledCharacteristic,u0,tspan;p=NamedTuple())
    N=op.grid isa Grid1D ? reshape(float.(copy(u0)),:,1) : float.(copy(u0))
    expected=op.grid isa Grid1D ? (length(op.grid),1) : size(op.grid)
    size(N)==expected || throw(DimensionMismatch("one bin-integrated number per characteristic cell required"))
    all(isfinite,N) && all(>=(0),N) || throw(ArgumentError("initial numbers must be finite and nonnegative"))
    length(tspan)==2 && all(isfinite,tspan) && tspan[2]>tspan[1] || throw(ArgumentError("invalid time span"))
    PBMProblem(op,N,(float(tspan[1]),float(tspan[2])),p)
end
rhs!(dN,N,op::CompiledCharacteristic,p,t)=throw(ArgumentError("characteristic transport is a direct evolution method; use simulate, not a semi-discrete ODE residual"))

function _simulate(prob::PBMProblem,method::Union{Inguva2022,InguvaBraatz2023};dt,saveat,check_nonnegative,negative_tolerance,maxsteps)
    op=prob.operator
    op isa CompiledCharacteristic && typeof(method)===typeof(op.evolution) ||
        throw(ArgumentError("compile the physical model with the requested characteristic evolution method first"))
    ts=_evolution_args(prob,dt,saveat,negative_tolerance,maxsteps)
    grid_axes=op.grid isa Grid1D ? (op.grid,) : (op.grid.x,op.grid.y)
    boundaries=op.grid isa Grid1D ? (op.bc,) : (op.bc.x,op.bc.y)
    for (law,axis,boundary) in zip(op.rates,grid_axes,boundaries)
        _characteristic_boundary(law,boundary,prob.p)
        h=_characteristic_spacing(law,axis,prob.p)
        isapprox(dt,h;rtol=1e-9,atol=1e-12) || throw(ArgumentError("dt must equal the travel-time mesh spacing (CFL=1); neither partial nor skipped cells are used"))
    end
    targets=[_integer_cells((t-first(prob.tspan))/dt) for t in ts]
    last(targets)<=maxsteps || throw(NumericalFailure("maximum step count exceeded"))
    N=copy(prob.u0);tmp=similar(N);states=typeof(N)[];steps=0
    for target in targets
        while steps<target
            if op.grid isa Grid1D
                periodic=op.bc.left isa Periodic
                for i in axes(N,1)
                    tmp[i,1]=i>1 ? N[i-1,1] : periodic ? N[end,1] : zero(eltype(N))
                end
            else
                _shift2d!(tmp,N,1,1,op.bc.x.left isa Periodic,op.bc.y.left isa Periodic)
            end
            N,tmp=tmp,N;steps+=1
        end
        _checkstate(N,check_nonnegative ? negative_tolerance : Inf)
        push!(states,copy(N))
    end
    PBMSolution(ts,states,prob,steps)
end

"""
    characteristic_density(operator, initial_density, position, elapsed; p=(;))

Evaluate the continuous characteristic solution for an eligible compiled model.
`initial_density(a)` (or `(a₁,a₂)`) supplies an actual continuous initial density;
the velocity-ratio Jacobian converts it to the physical density at elapsed time.
Zero inflow is imposed outside the initial physical domain; periodic constant
growth wraps the foot. This evaluation needs no interpolation or CFL-aligned
output time. The separate bin-shift solver DOES require aligned mesh/time steps.
When fitting rate parameters, this continuous evaluator can reuse the structural
compilation; a bin-shift run must rebuild its mesh for changed parameters.
"""
function characteristic_density(op::CompiledCharacteristic,f0,position,elapsed;p=NamedTuple())
    isfinite(elapsed) && elapsed>=0 || throw(ArgumentError("elapsed time must be finite and nonnegative"))
    one_dim=op.grid isa Grid1D
    positions=one_dim ? (position,) : position
    axes=one_dim ? (op.grid,) : (op.grid.x,op.grid.y)
    boundaries=one_dim ? (op.bc,) : (op.bc.x,op.bc.y)
    length(positions)==length(axes) || throw(DimensionMismatch("one position per intrinsic coordinate required"))
    feet=Float64[];jacobian=1.0
    for (law,axis,boundary,a) in zip(op.rates,axes,boundaries,positions)
        isfinite(a) && first(axis.edges)<=a<=last(axis.edges) || throw(ArgumentError("observation position lies outside the compiled domain"))
        _positive_affine(law,first(axis.edges),last(axis.edges),p)
        _characteristic_boundary(law,boundary,p)
        tau=_travel_time(law,a,first(axis.edges),p)-elapsed
        if boundary.left isa Periodic
            tau=mod(tau,_travel_time(law,last(axis.edges),first(axis.edges),p))
        elseif tau<0
            return 0.0
        end
        foot=_travel_inverse(law,tau,first(axis.edges),p)
        first(axis.edges)<=foot<=last(axis.edges) || return 0.0
        push!(feet,foot)
        jacobian*=law(foot,p,0.0,nothing)/law(a,p,0.0,nothing)
    end
    value=f0(feet...)*jacobian
    isfinite(value) && value>=0 || throw(NumericalFailure("invalid characteristic density"))
    value
end
