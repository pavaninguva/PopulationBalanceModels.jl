struct CompiledPBM2D{M,G,S,B,F,P}
    model::M
    grid::G
    scheme::S
    bc::B
    fields::F
    prepared::P
end
_uniform(g::Grid1D)=maximum(abs.(g.widths.-first(g.widths)))<=1e-10*first(g.widths)
function compile(model::PBMModel,g::Grid2D;growth_scheme=Upwind(),bc=SizeBC2D(),fields=_emptyfields,methods=TermMethods(),evolution=nothing,p=NamedTuple(),domain=BatchDomain())
    domain isa BatchDomain || return _compile_axial(model,g,domain;growth_scheme,bc,fields,methods,evolution,p)
    growth_scheme isa Union{Upwind,WENO5} || throw(ArgumentError("2D transport currently supports Upwind or WENO5"))
    bc isa SizeBC2D || throw(ArgumentError("use SizeBC2D for two coordinates"))
    any(b->b.left isa AbsorbingZero || b.right isa AbsorbingZero,(bc.x,bc.y)) &&
        throw(ArgumentError("AbsorbingZero material accounting is implemented for 1D populations only"))
    evolution===nothing || return _compile_evolution(model,g,evolution,bc,p;fields,methods)
    methods isa TermMethods || throw(ArgumentError("methods must be TermMethods"))
    plan=TermMethods(;by_term=methods.by_term,merge((growth=growth_scheme,),methods.defaults)...)
    count(t->t isa Growth2D,model.terms)<=1 || throw(ArgumentError("combine velocities in one Growth2D"))
    any(b->b.left isa InflowFlux || b.right isa InflowFlux,(bc.x,bc.y)) && !any(t->t isa Growth2D,model.terms) &&
        throw(ArgumentError("a 2D InflowFlux boundary requires Growth2D"))
    for term in model.terms
        if term isa Growth2D && term.rate isa Tuple
            _validate_velocity(term.rate)
        end
    end
    prepared=_prepare_terms(model,plan,g;bc)
    growth=[item.method for item in prepared if item.term isa Growth2D]
    scheme=isempty(growth) ? growth_scheme : only(growth)
    CompiledPBM2D(model,g,scheme,bc,fields,prepared)
end

function PBMProblem(op::CompiledPBM2D,u0::AbstractMatrix,tspan;p=NamedTuple())
    size(u0)==size(op.grid) || throw(DimensionMismatch("one state value per internal-coordinate bin is required"))
    N=float.(copy(u0)); all(isfinite,N) && all(N.>=0) || throw(ArgumentError("initial number must be finite and nonnegative"))
    length(tspan)==2 && all(isfinite,tspan) && tspan[2]>tspan[1] || throw(ArgumentError("invalid time span"))
    PBMProblem(op,N,(float(tspan[1]),float(tspan[2])),p)
end
_tensor_context(op,N,t)=(;grid=op.grid,operator=op,N,cell=1,fields=op.fields(1,t),
    bulk=(;),populations=(;),system=nothing)

function rhs!(dN::AbstractMatrix,N::AbstractMatrix,op::CompiledPBM2D,p,t)
    size(dN)==size(N)==size(op.grid) || throw(DimensionMismatch("tensor state size mismatch"))
    fill!(dN,zero(eltype(dN))); ctx=_tensor_context(op,N,t)
    for item in op.prepared
        _add_prepared!(dN,N,item,op,p,t,ctx)
    end
    all(isfinite,dN) || throw(NumericalFailure("nonfinite 2D derivative"))
    nothing
end
function suggest_dt(op::CompiledPBM2D,N,p,t;cfl=0.5)
    0<cfl<=1 || throw(ArgumentError("cfl must be in (0,1]"))
    g=op.grid; ctx=_tensor_context(op,N,t); hazards=zeros(size(g))
    for term in op.model.terms
        if term isa Growth2D
            for j in axes(N,2), i in axes(N,1)
                vl=_velocity(term,g.x.edges[i],g.y.centers[j],p,t,ctx)[1]
                vr=_velocity(term,g.x.edges[i+1],g.y.centers[j],p,t,ctx)[1]
                vb=_velocity(term,g.x.centers[i],g.y.edges[j],p,t,ctx)[2]
                vt=_velocity(term,g.x.centers[i],g.y.edges[j+1],p,t,ctx)[2]
                hazards[i,j]+=(max(-vl,0)+max(vr,0))/g.x.widths[i]+(max(-vb,0)+max(vt,0))/g.y.widths[j]
            end
        elseif term isa Union{Death,Dilution}
            for j in axes(N,2),i in axes(N,1)
                hazards[i,j]+=_nonnegative(term.rate((g.x.centers[i],g.y.centers[j]),p,t,ctx),"removal rate")
            end
        end
    end
    r=maximum(hazards); iszero(r) ? Inf : cfl/r
end
