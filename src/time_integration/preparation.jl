"""
    prepare_run(operator, p)
    prepare_run(term, method, structural_cache, representation, p)

Prepare numerical information for one parameter binding. `simulate` calls this
automatically, once per operating interval. Compilation owns fixed geometry;
this hook owns parameter-dependent coefficients and solve-local mutable storage.
State/time-dependent laws must still be evaluated in `add_term!` at the current
stage. Never capture an incidental parameter value during `prepare_term`.

The term-level default copies its cache. Extensions may specialize it to rebuild
declared parameter-dependent projections. Return a cache accepted by the same
`add_term!` method. Custom operator types may specialize the two-argument hook;
its default is `deepcopy`. Neither hook may mutate the compiled input.
"""
prepare_run(op,p)=deepcopy(op)
prepare_run(term,method,cache,representation,p)=deepcopy(cache)
_effective_run_term(item,op,p)=_effective_term(item)
_run_terms(op,p)=map(op.prepared) do item
    PreparedTerm(item.name,item.term,item.method,
        prepare_run(item.term,item.method,item.cache,op.grid,p))
end
prepare_run(op::CompiledPBM,p)=CompiledPBM(op.model,op.grid,op.scheme,op.bc,
    op.domain,op.fields,_run_terms(op,p))
prepare_run(op::CompiledPBM2D,p)=CompiledPBM2D(op.model,op.grid,op.scheme,op.bc,
    op.fields,_run_terms(op,p))
prepare_run(op::CompiledMomentPBM,p)=CompiledMomentPBM(op.model,op.grid,op.closure,
    op.fields,_run_terms(op,p),op.boundary)
function prepare_run(op::CompiledSystem,p)
    s=op.system
    populations=map(pop->prepare_run(pop,p),s.populations)
    system=PBMSystem(populations;bulk_rhs=s.bulk_rhs,interactions=s.interactions,
        exchanges=s.exchanges,fields=s.fields,nonnegative_bulk=s.nonnegative_bulk,
        accounting=s.accounting)
    CompiledSystem(system,op.ranges,op.bulk_indices,op.length,op.physical_length,op.ledger_indices)
end
function prepare_run(op::CompiledAxial,p)
    locals=map(pop->prepare_run(pop,p),op.local_operators)
    CompiledAxial(op.grid,op.domain,locals,first(locals).prepared,op.scheme,
        op.reconstruction,op.prepared_inlet)
end
prepare_run(op::CompiledAxialSystem,p)=CompiledAxialSystem(op.domain,
    map(pop->prepare_run(pop,p),op.local_operators),op.scheme,op.reconstruction)
