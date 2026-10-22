struct CompiledPBM{M,G,S,B,D,F,P}
    model::M
    grid::G
    scheme::S
    bc::B
    domain::D
    fields::F
    prepared::P
end

"""
    compile(model, grid; methods=TermMethods(), growth_scheme=Upwind(), bc=SizeBC(),
            domain=BatchDomain(), fields=(cell,t)->(;), evolution=nothing, p=(;))

Prepare a conservative sectional operator. Callback context contains `grid`,
`N` (the local bin concentrations), `cell`, and `fields` (prescribed conditions).
Methods select numerical treatment by mechanism class or individual term name.
Named choices in `methods.by_term` take precedence over class choices. Collision
deposition is evaluated during each residual call.

Alternatively, `evolution=Inguva2022()` compiles eligible homogeneous affine
growth as a direct characteristic update. Supply p for named coefficients;
the compiler checks the physical mesh, growth form and boundaries. Residual
term methods cannot be combined with that complete transport update.
"""
function compile(model::PBMModel, grid::Grid1D; growth_scheme=Upwind(), bc=SizeBC(),
                 domain=BatchDomain(),fields=_emptyfields,methods=TermMethods(),evolution=nothing,p=NamedTuple())
    growth_scheme isa AbstractReconstruction || throw(ArgumentError("unsupported growth scheme"))
    bc isa SizeBC || throw(ArgumentError("bc must be SizeBC"))
    bc.right isa AbsorbingZero && throw(ArgumentError("AbsorbingZero is a left boundary"))
    bc.left isa AbsorbingZero && first(grid.edges)!=0 && throw(ArgumentError("AbsorbingZero requires a zero left edge"))
    domain isa BatchDomain || return _compile_axial(model,grid,domain;growth_scheme,bc,fields,methods,evolution,p)
    evolution!==nothing && bc.left isa AbsorbingZero && throw(ArgumentError("AbsorbingZero requires residual sectional transport, not a direct evolution method"))
    evolution===nothing || return _compile_evolution(model,grid,evolution,bc,p;fields,methods)
    methods isa TermMethods || throw(ArgumentError("methods must be TermMethods"))
    !any(term->term isa Growth,model.terms) && _validate_reconstruction(growth_scheme,grid)
    plan=TermMethods(;by_term=methods.by_term,merge((growth=growth_scheme,),methods.defaults)...)
    count(x -> x isa Growth, model.terms) <= 1 ||
        throw(ArgumentError("combine size velocities into a single Growth term"))
    (bc.left isa InflowFlux || bc.right isa InflowFlux) && !any(term->term isa Growth,model.terms) &&
        throw(ArgumentError("an InflowFlux boundary requires a Growth term"))
    for term in model.terms
        if term isa Union{Aggregation,Breakage}
            grid.coordinate in (:volume,:mass) || term.coordinate_map!==nothing ||
                throw(ArgumentError("supply an AdditiveCoordinate map for a nonadditive size axis"))
            axis=_measureaxis(term,grid)
            all(isfinite,axis) && all(axis .> 0) && all(diff(axis).>0) || throw(ArgumentError("additive map must be positive and increasing"))
            if term.coordinate_map!==nothing
                all(isapprox.(term.coordinate_map.inverse.(axis),grid.centers;rtol=1e-9,atol=0)) ||
                    throw(ArgumentError("additive coordinate inverse fails the round-trip check"))
            end
            bc.left isa Periodic && throw(ArgumentError("periodic size boundaries are incompatible with collisions"))
        end
    end
    prepared=_prepare_terms(model,plan,grid;bc)
    any(item->item.method isa BoundaryInjection,prepared) && !any(term->term isa Growth,model.terms) &&
        throw(ArgumentError("BoundaryInjection requires a Growth term"))
    growth=[item.method for item in prepared if item.term isa Growth]
    scheme=isempty(growth) ? growth_scheme : only(growth)
    CompiledPBM(model,grid,scheme,bc,domain,fields,prepared)
end

"""Overwrite dN with dN/dt. State and derivative have shape (nbins,nphysicalcells)."""
function rhs!(dN::AbstractMatrix,N::AbstractMatrix,op::CompiledPBM,p,t;context=nothing)
    size(N)==size(dN) || throw(DimensionMismatch("state and derivative sizes differ"))
    size(N,1)==length(op.grid) || throw(DimensionMismatch("internal grid size mismatch"))
    size(N,2)==1 || throw(DimensionMismatch("BatchDomain requires one column"))
    fill!(dN,zero(eltype(dN)))
    for c in axes(N,2)
        ctx=_context(op,N,c,t,context)
        for item in op.prepared
            _add_prepared!(view(dN,:,c),view(N,:,c),item,op,p,t,ctx)
        end
    end
    all(isfinite,dN) || throw(NumericalFailure("nonfinite PopulationBalanceModels derivative"))
    nothing
end

"""Instantaneous resolved moments and rates of collision births outside the grid."""
function diagnostics(op::CompiledPBM,N,p,t)
    scratch=similar(N); fill!(scratch,zero(eltype(N)))
    lost0=zeros(eltype(N),size(N,2)); lost1=similar(lost0); fill!(lost1,zero(eltype(N)))
    for c in axes(N,2), item in op.prepared
        a,b=_add_prepared!(view(scratch,:,c),view(N,:,c),item,op,p,t,_context(op,N,c,t))
        lost0[c]+=a; lost1[c]+=b
    end
    (; minimum=minimum(N), M0=moment(op.grid,N,0), M1=moment(op.grid,N,1),
       unresolved_birth_number_rate=lost0, unresolved_birth_size_rate=lost1)
end

"""
Conservative heuristic step bound for explicit Upwind/source updates. WENO5 and
CustomTerm have no positivity guarantee from this bound. Check every RK stage.
"""
function suggest_dt(op::CompiledPBM,N,p,t;cfl=0.5,context=nothing)
    0 < cfl <= 1 || throw(ArgumentError("cfl must be in (0,1]"))
    r=zeros(promote_type(eltype(N),eltype(op.grid.widths)),size(N))
    x=op.grid.centers
    for c in axes(N,2)
        ctx=_context(op,N,c,t,context)
        for term in op.model.terms
            if term isa Growth
                for i in axes(N,1)
                    l=term.rate(op.grid.edges[i],p,t,ctx); h=term.rate(op.grid.edges[i+1],p,t,ctx)
                    r[i,c]+=(max(-l,zero(l))+max(h,zero(h)))/op.grid.widths[i]
                end
            elseif term isa Union{Death,Breakage,Dilution}
                for i in axes(N,1)
                    r[i,c]+=_nonnegative(term.rate(x[i],p,t,ctx),"hazard")
                end
            elseif term isa Aggregation
                for i in axes(N,1), j in axes(N,1)
                    r[i,c]+=_nonnegative(term.kernel(x[i],x[j],p,t,ctx),"kernel")*max(N[j,c],zero(eltype(N)))
                end
            end
        end
    end
    vmax=maximum(r)
    iszero(vmax) ? Inf : cfl/vmax
end
