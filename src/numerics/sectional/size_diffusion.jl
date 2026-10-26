# Fixed geometry only: coefficients must be evaluated anew at every RK stage,
# including when exchange re-evaluates a selected mechanism's residual.
struct _SizeDiffusionGeometry{V,T}
    inverse_distances::V
    wrap_inverse_distance::T
    loss_factors::V
end

function _validate_size_diffusion(model, grid, bc, domain, evolution)
    any(term -> term isa SizeDiffusion, model.terms) || return
    grid isa Union{Grid1D,Grid2D} && domain isa BatchDomain || throw(ArgumentError(
        "SizeDiffusion requires a batch Grid1D or Grid2D sectional population; moment and axial adapters are not supported"))
    evolution === nothing || throw(ArgumentError(
        "SizeDiffusion requires a residual time integrator (e.g. SSPRK33), not a direct transport update"))
    boundaries=if grid isa Grid1D
        bc isa SizeBC || throw(ArgumentError("Grid1D SizeDiffusion requires SizeBC"))
        (bc,)
    else
        bc isa SizeBC2D || throw(ArgumentError("Grid2D SizeDiffusion requires SizeBC2D"))
        (bc.x,bc.y)
    end
    all(b->(b.left isa NoFlux && b.right isa NoFlux) ||
           (b.left isa Periodic && b.right isa Periodic),boundaries) || throw(ArgumentError(
        "SizeDiffusion requires NoFlux at both ends or paired Periodic boundaries on each axis; open, absorbing and prescribed face-flux diffusion closures are unavailable"))
end

function _size_diffusivity(term::SizeDiffusion, x, p, t, ctx)
    law = term.diffusivity
    value = law isa Real ? law : law(x,p,t,ctx)
    value isa Real && isfinite(value) && _primal(value) >= 0 || throw(DomainError(
        value, "SizeDiffusion diffusivity at size $x, time $t must be a finite nonnegative real value"))
    value
end

default_method(::SizeDiffusion, ::Grid1D) = FiniteVolumeDiffusion()
function prepare_term(term::SizeDiffusion, ::FiniteVolumeDiffusion, g::Grid1D;
                      bc=SizeBC(), kwargs...)
    _validate_size_diffusion(PBMModel(term),g,bc,BatchDomain(),nothing)
    term.diffusivity isa Real && _size_diffusivity(term,first(g.centers),nothing,0,nothing)
    _diffusion_geometry(g,bc)
end

function _diffusion_geometry(g,bc)
    distances = inv.(diff(g.centers))
    wrap = bc.left isa Periodic ? inv((first(g.widths)+last(g.widths))/2) : zero(eltype(distances))
    factors = zeros(eltype(distances),length(g))
    for i in eachindex(distances)
        factors[i] += distances[i]
        factors[i+1] += distances[i]
    end
    factors[1] += wrap
    factors[end] += wrap
    factors ./= g.widths
    _SizeDiffusionGeometry(distances,wrap,factors)
end

function add_term!(dN, N, term::SizeDiffusion, ::FiniteVolumeDiffusion,
                   geometry, op::CompiledPBM, p, t, ctx)
    g = op.grid
    # A degree-zero polynomial declares spatial uniformity, as for Growth.
    # Reuse its current-stage coefficient, never one from a previous stage.
    uniform = term.diffusivity isa PolynomialRate && length(term.diffusivity.coefficients)==1
    coefficient = uniform ? _size_diffusivity(term,g.centers[1],p,t,ctx) : nothing
    first_value = (uniform ? coefficient : _size_diffusivity(term,g.centers[1],p,t,ctx))*(N[1]/g.widths[1])
    previous = first_value
    for i in 1:length(N)-1
        next = (uniform ? coefficient : _size_diffusivity(term,g.centers[i+1],p,t,ctx))*(N[i+1]/g.widths[i+1])
        q = -(next-previous)*geometry.inverse_distances[i]
        dN[i] -= q
        dN[i+1] += q
        previous = next
    end
    if op.bc.left isa Periodic
        q = -(first_value-previous)*geometry.wrap_inverse_distance
        dN[end] -= q
        dN[1] += q
    end
    (zero(eltype(dN)),zero(eltype(dN)))
end
