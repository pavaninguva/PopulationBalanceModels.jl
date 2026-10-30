function _needed_moments(term,k)
    deps=moment_dependencies(term,k)
    deps===nothing && throw(ArgumentError("no analytical rule is implemented for this rate-law declaration ($(typeof(term))), moment equation $k"))
    collect(deps)
end


function _moment_source!(dμ,μ,term::Nucleation,op,p,t,ctx)
    J=_nonnegative(term.rate(p,t,ctx),"nucleation rate")
    for k in op.grid.orders
        dμ[k+1]+=J*term.size^k
    end
end
function _moment_source!(dμ,μ,term::MomentSource,op,p,t,ctx)
    J=_nonnegative(term.rate(p,t,ctx),"moment source rate")
    for k in op.grid.orders
        dμ[k+1]+=J*_nonnegative(term.rawmoment(k,p,t,ctx),"source raw moment")
    end
end
function _exact_moment_term!(dμ,μ,term,op,p,t,ctx)
    term isa Union{Nucleation,MomentSource} && return _moment_source!(dμ,μ,term,op,p,t,ctx)
    for k in op.grid.orders
        dμ[k+1]+=analytical_moment(term,k,j->μ[j+1],p,t,ctx)
    end
end

function _quadrature_term!(dμ,μ,term,op,p,t,ctx,q)
    term isa Union{Nucleation,MomentSource} && return _moment_source!(dμ,μ,term,op,p,t,ctx)
    x=q.nodes; w=q.weights
    if term isa Aggregation
        for i in eachindex(x), j in 1:i
            (iszero(w[i]) || iszero(w[j])) && continue
            K=_nonnegative(term.kernel(x[i],x[j],p,t,ctx),"aggregation kernel")
            isapprox(K,term.kernel(x[j],x[i],p,t,ctx);rtol=1e-10) || throw(ArgumentError("kernel must be symmetric"))
            z=term.coordinate_map===nothing ? x[i]+x[j] : term.coordinate_map.inverse(term.coordinate_map.measure(x[i])+term.coordinate_map.measure(x[j]))
            isfinite(z) && z>=0 || throw(ArgumentError("invalid aggregate coordinate"))
            event=K*w[i]*w[j]*(i==j ? 0.5 : 1)
            for k in op.grid.orders
                dμ[k+1]+=event*(z^k-x[i]^k-x[j]^k)
            end
        end
    else
        for i in eachindex(x)
            iszero(w[i]) && continue
            if term isa Growth
                G=term.rate(x[i],p,t,ctx)
                isfinite(G) || throw(NumericalFailure("nonfinite growth rate"))
                for k in op.grid.orders
                    k>0 && (dμ[k+1]+=k*G*w[i]*x[i]^(k-1))
                end
            elseif term isa Union{Death,Dilution}
                event=_nonnegative(term.rate(x[i],p,t,ctx),"removal rate")*w[i]
                for k in op.grid.orders
                    dμ[k+1]-=event*x[i]^k
                end
            elseif term isa Breakage
                event=_nonnegative(term.rate(x[i],p,t,ctx),"breakage rate")*w[i]
                daughters=collect(term.daughters(x[i],p,t,ctx))
                measure=term.coordinate_map===nothing ? identity : term.coordinate_map.measure
                all(d->isfinite(d[1]) && 0<=d[1]<=x[i] && isfinite(d[2]) && d[2]>=0,daughters) || throw(ArgumentError("invalid daughters"))
                isapprox(sum(count*measure(y) for (y,count) in daughters),measure(x[i]);rtol=1e-8) || throw(ArgumentError("daughters must conserve the declared measure"))
                for k in op.grid.orders
                    integral=applicable(daughter_moment,term.daughters,x[i],k,p,t,ctx) ?
                        daughter_moment(term.daughters,x[i],k,p,t,ctx) : sum(count*y^k for (y,count) in daughters)
                    isfinite(integral) && integral>=0 || throw(NumericalFailure("invalid daughter moment"))
                    dμ[k+1]+=event*(integral-x[i]^k)
                end
            end
        end
    end
end

prepare_moment_state(::ExactClosure,g,u,p,t,ctx)=nothing
prepare_moment_state(::AbstractMomentMethod,g,u,p,t,ctx)=throw(ArgumentError("moment family must implement prepare_moment_state"))
prepare_moment_state(m::Union{QuadratureClosure,LogNormalClosure},g,u,p,t,ctx)=quadrature(g,u,m)
function _prepare_checked_moment_state(m,g,u,p,t,ctx)
    validate_moment_state(m,g,u)
    prepare_moment_state(m,g,u,p,t,ctx)
end
function _prepare_checked_moment_state(m::ExactClosure,g,u,p,t,ctx)
    # Runtime invalid states are numerical failures, distinct from constructor
    # argument errors. Inference/external solvers can then reject the evaluation.
    check_moment_state(m,g,u,1e-10)
    nothing
end
# These preparations already validate every input/reconstructed moment.
_prepare_checked_moment_state(m::Union{QuadratureClosure,LogNormalClosure},g,u,p,t,ctx)=
    prepare_moment_state(m,g,u,p,t,ctx)
function moment_term!(du,u,term,::ExactMomentRule,cache,method,q,op,p,t,ctx)
    _exact_moment_term!(du,u,term,op,p,t,ctx)
end
function moment_term!(du,u,term,::QuadratureMomentRule,cache,method,q,op,p,t,ctx)
    _quadrature_term!(du,u,term,op,p,t,ctx,q)
end
moment_term!(du,u,term,::ExactMomentRule,cache,::AbstractDirectMomentMethod,q,op,p,t,ctx)=throw(ArgumentError("direct moment family must implement its state evolution"))
moment_term!(du,u,term,::QuadratureMomentRule,cache,::AbstractDirectMomentMethod,q,op,p,t,ctx)=throw(ArgumentError("direct moment family must implement its state evolution"))
