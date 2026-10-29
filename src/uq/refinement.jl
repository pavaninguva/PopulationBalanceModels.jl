function _pce_tolerance(value, layout, label)
    total = sum(length, layout.slices)
    if value isa Real
        result = fill(Float64(value), total)
    elseif value isa NamedTuple
        keys(value) == layout.names || throw(ArgumentError("$label must name every output block in layout order"))
        nt = layout.times === nothing ? 1 : length(layout.times)
        result = Float64[]
        for (block,n) in zip(values(value),layout.lengths)
            if block isa Real
                append!(result,fill(Float64(block),n*nt))
            else
                block isa AbstractMatrix && size(block) == (n,nt) ||
                    throw(DimensionMismatch("$label blocks must be scalars or output-by-time matrices"))
                append!(result,vec(Float64.(block)))
            end
        end
    else
        throw(ArgumentError("$label must be a nonnegative scalar or named output tolerances"))
    end
    all(x -> isfinite(x) && x >= 0, result) || throw(ArgumentError("$label must be finite and nonnegative"))
    result
end

function _pce_validation_design(validation)
    validation isa UQResult && _pce_require_forward(validation)
    recorded = validation isa UQResult ? get(validation.metadata,:design,nothing) : validation
    recorded isa NamedTuple && haskey(recorded,:weights) && throw(ArgumentError(
        "PCE validation uses unweighted full-model samples; quadrature weights are for construction, not validation"))
    nothing
end

"""
Result of [`refine_pce`](@ref). `candidates` and `history` retain every completed
attempt. `surrogate` is present only when `converged=true`: two successive,
distinct candidates passed validation and their analytical means/variances met
the requested stability criteria. Otherwise it is `nothing`; diagnostic fits
remain in `candidates`. This is an empirical selection rule, not a proof of
spectral convergence or an error bound over the entire input support.
"""
struct PCERefinement{S,C,H,V,M}
    surrogate::S
    candidates::C
    history::H
    validation::V
    converged::Bool
    report::M
end
diagnostics(study::PCERefinement) = study.report

function _pce_stability(previous, current; mean_atol,mean_rtol,variance_atol,variance_rtol)
    a, b = previous.coefficients, current.coefficients
    μa, μb = vec(a[1,:]), vec(b[1,:])
    va, vb = vec(sum(abs2,a[2:end,:];dims=1)), vec(sum(abs2,b[2:end,:];dims=1))
    layout = current.training.layout
    mean_change, variance_change = abs.(μb-μa), abs.(vb-va)
    mean_limit = _pce_tolerance(mean_atol,layout,"mean_atol") .+
        _pce_tolerance(mean_rtol,layout,"mean_rtol").*max.(abs.(μa),abs.(μb))
    variance_limit = _pce_tolerance(variance_atol,layout,"variance_atol") .+
        _pce_tolerance(variance_rtol,layout,"variance_rtol").*max.(va,vb)
    passed = all(mean_change .<= mean_limit) && all(variance_change .<= variance_limit)
    (;passed,mean_change,mean_limit,variance_change,variance_limit)
end

"""
    refine_pce(problem, methods; designs, validation, max_solves,
               validation_atol=0, validation_rtol=0.05,
               mean_atol=0, mean_rtol=0.01,
               variance_atol=0, variance_rtol=0.05)

Evaluate an explicitly ordered plan of at least two `PolynomialChaos` methods
and matching training designs. Increase degree, regression sample size or
quadrature order in `designs`; the plan, seeds and all evaluated points remain
available. A single disjoint full-model validation set is reused across the plan.

Stop only after **two consecutive candidates pass validation** and their
analytical PCE means/variances agree within the respective `atol + rtol*scale`
limits. Here scale is the larger absolute mean or variance of the pair; absolute
limits have output or squared-output units. Scalars or named output-by-time
tolerances work as in [`validate_pce`](@ref). Set absolute limits for near-zero
means/variances. Exhausting the plan returns `converged=false, surrogate=nothing`.
All candidates and detailed validation/stability results are retained. A forward
failure stops with `stop_reason=:forward_failure`; `diagnostics(study).failure`
is a `PCEFailure` retaining its sample-level evaluations.

`max_solves` must cover the entire plan and validation before any solve starts.
Existing `UQResult` designs/validation cost zero new solves. Other designs each
incur their full stated count, including repeated points across designs; reuse
evaluations explicitly to avoid those repeats. One solve still supplies every
output and time. Programming errors propagate. No failed sample or candidate is
silently skipped, and a failed refinement never returns a selected surrogate.

Validation used for this selection is not an untouched final test. Use fresh
inputs with `validate_pce(study.surrogate, test)` for a final assessment. Tail/
event accuracy, input-law validity, and forward spatial/time/closure refinement
remain separate checks. The full tensor projection backend is not adaptive PCE.
"""
function refine_pce(problem::UQProblem, methods; designs, validation, max_solves,
        validation_atol=0, validation_rtol=0.05, mean_atol=0, mean_rtol=0.01,
        variance_atol=0, variance_rtol=0.05)
    methods, designs = collect(methods), collect(designs)
    length(methods) >= 2 && length(methods) == length(designs) &&
        all(m -> m isa PolynomialChaos, methods) ||
        throw(ArgumentError("PCE refinement needs at least two methods with one design each"))
    validation === nothing && throw(ArgumentError("PCE refinement requires explicit full-model validation inputs"))
    _pce_validation_design(validation)
    Z = _pce_points(problem.inputs,validation)
    validation_points = Set(Tuple(column) for column in eachcol(Z))
    X = [_pce_points(problem.inputs,design) for design in designs]
    # Preflight every candidate before spending the forward-solve budget.
    for i in eachindex(methods)
        _pce_preparation(problem,methods[i],designs[i])
        any(c -> Tuple(c) in validation_points, eachcol(X[i])) &&
            throw(ArgumentError("PCE refinement validation overlaps candidate $i training inputs"))
        if i > 1 && methods[i].degree == methods[i-1].degree &&
                methods[i].construction == methods[i-1].construction &&
                Set(Tuple(c) for c in eachcol(X[i])) == Set(Tuple(c) for c in eachcol(X[i-1]))
            throw(ArgumentError("consecutive PCE candidates must change degree, construction or training points"))
        end
        if designs[i] isa UQResult
            designs[i].problem === problem || throw(ArgumentError("PCE refinement evaluations must use the same UQProblem"))
            _pce_success(designs[i])
        end
    end
    validation isa UQResult && validation.problem !== problem &&
        throw(ArgumentError("PCE refinement validation must use the same UQProblem"))
    needed = (validation isa UQResult ? 0 : size(Z,2)) +
        sum(designs[i] isa UQResult ? 0 : size(X[i],2) for i in eachindex(designs))
    max_solves isa Integer && max_solves >= needed ||
        throw(ArgumentError("PCE refinement plan needs a budget of $needed new full solves"))
    tick = time_ns()
    reference = validation isa UQResult ? validation :
        propagate(problem,MonteCarlo(size(Z,2));samples=validation,max_solves=size(Z,2))
    solves = validation isa UQResult ? 0 : reference.metadata.solves
    candidates, history = PCESurrogate[], NamedTuple[]
    selected = nothing
    function finish(reason; failure=nothing)
        converged = selected !== nothing
        report = (;converged,stop_reason=reason,failure,
            solves,planned_solves=needed,budget=max_solves,candidates=length(candidates),
            plan_methods=methods,plan_designs=designs,
            validation_used_for_selection=true,elapsed_seconds=(time_ns()-tick)/1e9,
            validation_atol,validation_rtol,mean_atol,mean_rtol,variance_atol,variance_rtol,
            coordinates=problem.inputs.coordinates)
        PCERefinement(selected,candidates,history,reference,converged,report)
    end
    all(reference.success) || return finish(:forward_failure;
        failure=PCEFailure("PCE refinement validation has failed forward evaluations",reference))
    # Validate tolerances before training solves, including the first stability
    # comparison's quantities. Layout is only known after obtaining validation.
    for (value,label) in ((validation_atol,"validation_atol"),(validation_rtol,"validation_rtol"),
            (mean_atol,"mean_atol"),(mean_rtol,"mean_rtol"),
            (variance_atol,"variance_atol"),(variance_rtol,"variance_rtol"))
        _pce_tolerance(value,reference.layout,label)
    end
    for i in eachindex(methods)
        budget = designs[i] isa UQResult ? 0 : size(X[i],2)
        fit = try
            fit_pce(problem,methods[i];design=designs[i],validation=reference,
                max_solves=budget,validation_atol,validation_rtol)
        catch error
            error isa PCEFailure || rethrow()
            solves += designs[i] isa UQResult ? 0 : error.evaluations.metadata.solves
            push!(history,(;candidate=i,degree=methods[i].degree,construction=methods[i].construction,
                points=size(X[i],2),failure=error,total_new_solves=solves))
            return finish(:forward_failure;failure=error)
        end
        solves += designs[i] isa UQResult ? 0 : fit.training.metadata.solves
        stability = isempty(candidates) ? nothing : _pce_stability(last(candidates),fit;
            mean_atol,mean_rtol,variance_atol,variance_rtol)
        accepted = stability !== nothing && stability.passed && fit.validation.passed &&
            last(candidates).validation.passed
        push!(candidates,fit)
        push!(history,(;candidate=i,degree=methods[i].degree,construction=methods[i].construction,
            points=size(X[i],2),condition=diagnostics(fit).condition,
            validation_passed=fit.validation.passed,validation=fit.validation,
            stability,selected=accepted,new_training_solves=budget,total_new_solves=solves))
        if accepted
            selected = fit
            break
        end
    end
    finish(selected === nothing ? :plan_exhausted : :criteria_met)
end
