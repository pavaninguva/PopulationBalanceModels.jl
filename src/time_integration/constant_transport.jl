"""
    LaxWendroff()

Conservative second-order space–time update for homogeneous constant 1D growth
on a uniform grid. Select `compile(...; evolution=LaxWendroff())` with an
explicit `AffineGrowth(g,0)`. Not a reconstruction for an arbitrary RK residual.
Requires |g|dt/Δa ≤ 1; oscillations and negative densities are possible.
"""
struct LaxWendroff end

"""
    BeamWarming()

Second-order upwind space–time update for the same constant-growth class as
LaxWendroff. This implementation restricts |g|dt/Δa ≤ 1. The first upstream
interior face falls back to Upwind at nonperiodic boundaries. No limiter.
"""
struct BeamWarming end

struct CompiledConstantTransport{M,G,B,E,R,P}
    model::M
    grid::G
    bc::B
    evolution::E
    rate::R
    prepared::P
end
_default_evolution(op::CompiledConstantTransport)=op.evolution
function _compile_evolution(model,grid,method::Union{LaxWendroff,BeamWarming},bc,p;fields,methods)
    grid isa Grid1D && bc isa SizeBC || throw(ArgumentError("this space–time scheme requires a 1D grid and SizeBC"))
    fields===_emptyfields || throw(ArgumentError("constant transport does not use prescribed fields"))
    methods isa TermMethods && isempty(methods.defaults) && isempty(methods.by_term) ||
        throw(ArgumentError("select evolution alone, without per-term residual methods"))
    length(model.terms)==1 && only(model.terms) isa Growth && only(model.terms).rate isa AffineGrowth ||
        throw(ArgumentError("declare homogeneous constant growth with Growth(AffineGrowth(g,0))"))
    rate=only(model.terms).rate
    iszero(_affine_coefficient(rate.slope,p)) || throw(ArgumentError("size-dependent growth is not supported by this space–time scheme"))
    _uniform(grid) && length(grid)>=3 || throw(ArgumentError("a uniform grid with at least three cells is required"))
    all(b->b isa Union{Periodic,OpenBoundary,NoFlux},(bc.left,bc.right)) ||
        throw(ArgumentError("supported size boundaries are Periodic, OpenBoundary and NoFlux"))
    CompiledConstantTransport(model,grid,bc,method,rate,(PreparedTerm(only(model.names),only(model.terms),method,nothing),))
end
function PBMProblem(op::CompiledConstantTransport,u0,tspan;p=NamedTuple())
    N=reshape(float.(copy(u0)),:,1)
    size(N,1)==length(op.grid) || throw(DimensionMismatch("one bin number per cell required"))
    all(isfinite,N) && all(>=(0),N) || throw(ArgumentError("initial numbers must be finite and nonnegative"))
    length(tspan)==2 && all(isfinite,tspan) && tspan[2]>tspan[1] || throw(ArgumentError("invalid time span"))
    PBMProblem(op,N,(float(tspan[1]),float(tspan[2])),p)
end
rhs!(dN,N,op::CompiledConstantTransport,p,t)=throw(ArgumentError("use simulate with the compiled space–time scheme; no semi-discrete residual exists"))
function _simulate(prob::PBMProblem,method::Union{LaxWendroff,BeamWarming};dt,saveat,check_nonnegative,negative_tolerance,maxsteps)
    op=prob.operator
    op isa CompiledConstantTransport && typeof(method)===typeof(op.evolution) ||
        throw(ArgumentError("compile with the requested space–time scheme first"))
    ts=_evolution_args(prob,dt,saveat,negative_tolerance,maxsteps)
    iszero(_affine_coefficient(op.rate.slope,prob.p)) || throw(ArgumentError("growth must remain constant after changing parameters"))
    v=_affine_coefficient(op.rate.offset,prob.p); dx=first(op.grid.widths)
    abs(v)*dt/dx<=1+1e-12 || throw(ArgumentError("this space–time scheme requires |growth|dt/Δa ≤ 1"))
    N=copy(prob.u0); next=similar(N); F=zeros(eltype(N),length(N)+1)
    periodic=op.bc.left isa Periodic; m=length(N);t=first(prob.tspan);steps=0;states=typeof(N)[]
    at(i)=N[periodic ? mod1(i,m) : i]/dx
    for target in ts
        while t<target
            h=min(dt,target-t);t+h>t || throw(NumericalFailure("time step is too small to advance"))
            steps+=1;steps<=maxsteps || throw(NumericalFailure("maximum step count exceeded"))
            C=abs(v)*h/dx
            for j in (periodic ? (1:m) : (2:m))
                if method isa LaxWendroff
                    F[j]=v/2*(at(j-1)+at(j))-v^2*h/(2dx)*(at(j)-at(j-1))
                else
                    i=v>=0 ? j-1 : j;up=v>=0 ? i-1 : i+1
                    F[j]=v*(at(i)+(!periodic && !(1<=up<=m) ? zero(v) : (1-C)/2*(at(i)-at(up))))
                end
            end
            if periodic
                F[end]=F[1]
            else
                F[1]=op.bc.left isa NoFlux || v>=0 ? zero(v) : v*at(1)
                F[end]=op.bc.right isa NoFlux || v<=0 ? zero(v) : v*at(m)
            end
            for i in 1:m; next[i]=N[i]+h*(F[i]-F[i+1]); end
            N .= next
            _checkstate(N,check_nonnegative ? negative_tolerance : Inf)
            t=h==target-t ? target : t+h
        end
        push!(states,copy(N))
    end
    PBMSolution(ts,states,prob,steps,(policy=Symbol(nameof(typeof(method))),accepted_steps=steps,rejected_steps=0))
end
