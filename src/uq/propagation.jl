"""
    UQProblem(base, inputs; outputs, times=nothing, setup, solver=nothing,
              dt=nothing, method=SSPRK33(), output_coordinates=(;),
              negative_tolerance=1e-12)

One forward scenario with named observable outputs, without estimation data.
`setup(physical_inputs, base)` returns a PBMProblem or StationaryPBMProblem and
may rebuild initial states, fields, meshes and compiled operators. The default
merges physical NamedTuple inputs into base.p. Setup must not mutate base.

Transient outputs share explicit increasing `times`. Supply `dt` for native
integration or `solver(problem,times)` returning a checked PBMSolution. Stationary
outputs have no times and use `solver(problem)` (StationarySolver by default).
External solvers must check their own successful completion; native guarantees
do not transfer. Saved states also receive the stated nonnegativity check.

`outputs` is a NamedTuple of existing observables, including FunctionObservable
and SystemObservable callbacks. `output_coordinates` records each output's units,
weighting and coordinates; callback declarations are not a proof of their units.
Representation-aware observations retain their ordinary restrictions.
"""
struct UQProblem{B,I,O,T,F,S,C}
    base::B
    inputs::I
    outputs::O
    times::T
    setup::F
    solver::S
    output_coordinates::C
    negative_tolerance::Float64
end
function UQProblem(base::Union{PBMProblem,StationaryPBMProblem},inputs::UncertainInputs;
        outputs::NamedTuple,times=nothing,setup=_default_setup,solver=nothing,
        dt=nothing,method=SSPRK33(),output_coordinates=(;),negative_tolerance=1e-12)
    !isempty(outputs) && all(o->o isa AbstractObservable,outputs) ||
        throw(ArgumentError("provide named observables; wrap callbacks in FunctionObservable or SystemObservable"))
    output_coordinates isa NamedTuple || throw(ArgumentError("output coordinates must be named metadata"))
    isempty(output_coordinates) || keys(output_coordinates)==keys(outputs) ||
        throw(ArgumentError("output coordinate metadata must follow every named output"))
    isfinite(negative_tolerance) && negative_tolerance>=0 || throw(ArgumentError("invalid saved-state negativity tolerance"))
    if base isa StationaryPBMProblem
        times===nothing && dt===nothing || throw(ArgumentError("stationary UQ has no times or timestep"))
        solver===nothing && (solver=StationarySolver())
    else
        times===nothing && throw(ArgumentError("transient UQ requires a common output-time grid"))
        times=float.(collect(times))
        !isempty(times) && all(isfinite,times) && all(diff(times).>0) &&
            base.tspan[1]<=first(times)<=last(times)<=base.tspan[2] ||
            throw(ArgumentError("UQ output times must increase inside the problem time span"))
        if solver===nothing
            dt isa Real && isfinite(dt) && dt>0 || throw(ArgumentError("supply dt or a checked solver(problem,times)"))
            solver=(pb,ts)->simulate(pb;dt,saveat=ts,method,negative_tolerance)
        elseif dt!==nothing
            throw(ArgumentError("configure timestep controls in the custom solver, or supply dt for native integration"))
        end
    end
    UQProblem(base,inputs,outputs,times,setup,solver,output_coordinates,Float64(negative_tolerance))
end

"""Raw forward ensemble. `values[:,i]` belongs to `ids[i]`; failed samples remain
as NaNs with an explicit stage/message. Layout records named blocks, output/time
order and coordinates. `metadata.solves` counts actual solver invocations."""
struct UQResult{P,I,X,D,L,M}
    problem::P
    inputs::I
    physical::X
    ids::D
    values::Matrix{Float64}
    success::BitVector
    failures::Vector{NamedTuple}
    layout::L
    metadata::M
end

function _uq_layout(problem,lengths)
    stop=0
    nt=problem.times===nothing ? 1 : length(problem.times)
    slices=map(lengths) do n
        first=stop+1;stop+=n*nt
        first:stop
    end
    (;names=keys(problem.outputs),lengths=Tuple(lengths),slices=Tuple(slices),
      times=problem.times===nothing ? nothing : copy(problem.times),
      coordinates=problem.output_coordinates)
end
function _uq_blocks(layout,values)
    nt=layout.times===nothing ? 1 : length(layout.times)
    blocks=map(layout.slices,layout.lengths) do slice,n
        values isa AbstractVector ? reshape(copy(values[slice]),n,nt) :
            reshape(copy(values[slice,:]),n,nt,size(values,2))
    end
    NamedTuple{layout.names}(blocks)
end

function _uq_observations(problem,pb,solution)
    if pb isa PBMProblem
        solution isa PBMSolution && solution.problem===pb ||
            throw(ArgumentError("a transient UQ solver must return PBMSolution for the supplied problem"))
        collect(solution.t)==problem.times || throw(ArgumentError("UQ solver must return exactly the common requested output times"))
        length(solution.u)==length(problem.times) || throw(DimensionMismatch("one UQ state per output time is required"))
        for state in solution.u
            size(state)==size(pb.u0) || throw(DimensionMismatch("UQ solver changed the problem state layout"))
            _check_problem_state(pb,state,problem.negative_tolerance)
            _check_admissible_state(pb.operator,state)
        end
        blocks=map(problem.outputs) do observable
            columns=[observe(observable,solution;index=i) for i in eachindex(problem.times)]
            all(c->length(c)==length(first(columns)),columns) || throw(DimensionMismatch("UQ observable changed dimension across times"))
            hcat(columns...)
        end
    else
        solution isa StationaryPBMSolution && solution.problem===pb ||
            throw(ArgumentError("a stationary UQ solver must return StationaryPBMSolution for the supplied problem"))
        solution.converged || throw(NumericalFailure("stationary UQ solve did not converge"))
        residual=similar(solution.u);stationary_residual!(residual,solution.u,pb)
        physical=_stationary_physical(pb.operator,_stationary_rates(pb,solution.u,pb.p))
        tolerance=diagnostics(solution).acceptance_tolerance
        max(maximum(abs,residual),maximum(abs,physical./pb.residual_scale))<=tolerance ||
            throw(NumericalFailure("stationary UQ solution failed its balance acceptance check"))
        blocks=map(o->reshape(observe(o,solution),:,1),problem.outputs)
    end
    lengths=Tuple(size(b,1) for b in blocks)
    all(>(0),lengths) || throw(DimensionMismatch("UQ outputs must be nonempty"))
    layout=_uq_layout(problem,lengths)
    flat=vcat((vec(b) for b in blocks)...)
    all(x->x isa Real && isfinite(x),flat) || throw(NumericalFailure("nonfinite UQ observation"))
    Float64.(flat),layout
end

"""
    propagate(problem, MonteCarlo(n; seed); max_solves, samples=nothing)

Run deterministic serial forward propagation, one solve supplies all outputs and
times. The worst-case solve budget is checked before execution. Optional `samples`
is a d-by-n matrix or a recorded `pce_design`; these are design evaluations,
not iid MC draws. Only NumericalFailure is recorded; programming errors propagate.
Invalid inputs are recorded without replacement. Summary statistics reject failed
ensembles unless success conditioning is explicitly requested.
"""
function propagate(problem::UQProblem,method::MonteCarlo;max_solves,samples=nothing)
    n=method.n
    max_solves isa Integer && max_solves>=n || throw(ArgumentError("UQ requires a budget of at least $n full solves"))
    input=problem.inputs
    source=samples===nothing ? input : UncertainInputs(
        samples isa NamedTuple ? samples.samples : samples;
        names=input.names,map=input.map,valid=input.valid,coordinates=input.coordinates,
        iid=false,metadata=input.metadata)
    raw,ids=_uq_samples(source,n,MersenneTwister(method.seed))
    physical=Any[nothing for _ in 1:n]
    trajectories=Vector{Union{Nothing,Vector{Float64}}}(nothing,n)
    success=falses(n);failures=NamedTuple[];layout=nothing
    solves=0;setup_seconds=0.;solve_seconds=0.;observation_seconds=0.;started=time_ns()
    for i in 1:n
        stage=:input
        pending=0
        try
            _uq_finite(raw[i]) || throw(NumericalFailure("nonfinite uncertain input"))
            physical[i]=source.map(deepcopy(raw[i]))
            flags=get(source.metadata,:support_flags,nothing)
            (flags===nothing || flags[i]) && _uq_finite(physical[i]) && source.valid(physical[i]) ||
                throw(NumericalFailure("uncertain input violates the declared physical support"))
            stage=:setup;tick=time_ns();pending=tick
            pb=problem.setup(deepcopy(physical[i]),problem.base)
            setup_seconds+=(time_ns()-tick)/1e9
            pending=0
            (pb isa PBMProblem)==(problem.base isa PBMProblem) && pb isa Union{PBMProblem,StationaryPBMProblem} ||
                throw(ArgumentError("UQ setup must preserve transient/stationary problem kind"))
            pb isa PBMProblem && !(pb.tspan[1]<=first(problem.times)<=last(problem.times)<=pb.tspan[2]) &&
                throw(ArgumentError("rebuilt UQ problem does not contain the common output times"))
            stage=:solve;solves+=1;tick=time_ns();pending=tick
            solution=pb isa PBMProblem ? problem.solver(pb,problem.times) : problem.solver(pb)
            solve_seconds+=(time_ns()-tick)/1e9
            pending=0
            stage=:observation;tick=time_ns();pending=tick
            values,current=_uq_observations(problem,pb,solution)
            observation_seconds+=(time_ns()-tick)/1e9
            pending=0
            layout===nothing ? (layout=current) : layout==current ||
                throw(DimensionMismatch("UQ output dimensions, coordinates or times changed between realizations"))
            trajectories[i]=values;success[i]=true
        catch error
            error isa NumericalFailure || rethrow()
            if pending!=0
                elapsed=(time_ns()-pending)/1e9
                stage===:setup && (setup_seconds+=elapsed)
                stage===:solve && (solve_seconds+=elapsed)
                stage===:observation && (observation_seconds+=elapsed)
            end
            push!(failures,(;sample=i,id=ids[i],stage,message=sprint(showerror,error)))
        end
    end
    m=layout===nothing ? 0 : sum(length,layout.slices)
    values=fill(NaN,m,n)
    for i in 1:n
        success[i] && (values[:,i]=trajectories[i])
    end
    meta=(;method=samples===nothing ? :monte_carlo : :design,seed=method.seed,
        iid=source.iid,coordinates=source.coordinates,input_metadata=source.metadata,
        design=samples isa NamedTuple ? samples : nothing,solves,requested=n,
        setup_seconds,solve_seconds,observation_seconds,total_seconds=(time_ns()-started)/1e9)
    UQResult(problem,raw,physical,ids,values,success,failures,layout,meta)
end

"""Counts, timing, coordinate and sampling assumptions; numerical failures remain
indexed in the result. This report is separate from input/output uncertainty."""
diagnostics(result::UQResult)=(;result.metadata...,successful=count(result.success),
    failed=count(!,result.success),failures=result.failures)

function _uq_quantile(sorted,p)
    0<=p<=1 || throw(ArgumentError("quantile probabilities must lie in [0,1]"))
    x=1+(length(sorted)-1)*p;i=floor(Int,x);j=min(i+1,length(sorted))
    (j-x)*sorted[i]+(x-i)*sorted[j] + (i==j ? sorted[i] : 0.)
end

"""
    uq_summary(ensemble; probabilities=(.025,.5,.975), events=(;),
               covariance=true, conditional_on_success=false)

Named means, unbiased variances, pointwise empirical quantiles and optional full
output/time covariance in the recorded flattening order. Event callbacks receive
one NamedTuple of output-by-time blocks and return a Bool per realization.
Only declared iid samples receive mean MCSEs and binomial event intervals.
No iid error estimate is assigned to imported MCMC samples. Output SD is not
the sampling error of a mean, nor is a pointwise band simultaneous coverage.
Sampling errors for variance and quantiles are not estimated by this routine.
"""
function uq_summary(result::UQResult;probabilities=(.025,.5,.975),events=(;),
        covariance=true,conditional_on_success=false)
    all(result.success) || conditional_on_success || throw(ArgumentError(
        "UQ has failed samples; inspect failures or explicitly request conditional_on_success=true"))
    selected=findall(result.success);n=length(selected)
    n>0 && result.layout!==nothing || throw(ArgumentError("no successful UQ outputs to summarize"))
    probs=float.(collect(probabilities));all(p->isfinite(p)&&0<=p<=1,probs) || throw(ArgumentError("invalid quantile probabilities"))
    Y=result.values[:,selected];mu=vec(sum(Y;dims=2))/n;centered=Y.-mu
    variance=n>1 ? vec(sum(abs2,centered;dims=2))/(n-1) : fill(NaN,length(mu))
    C=covariance && n>1 ? centered*centered'/(n-1) : nothing
    quantiles=Matrix{Float64}(undef,length(mu),length(probs))
    for i in eachindex(mu)
        sorted=sort(Y[i,:])
        for (j,p) in enumerate(probs)
            quantiles[i,j]=_uq_quantile(sorted,p)
        end
    end
    iid=result.metadata.iid && all(result.success)
    event_results=map(events) do event
        flags=[event(_uq_blocks(result.layout,Y[:,j])) for j in 1:n]
        all(x->x isa Bool,flags) || throw(ArgumentError("UQ event callbacks must return Bool"))
        k=count(flags);p=k/n
        interval=iid ? (k==0 ? 0. : Distributions.quantile(Distributions.Beta(k,n-k+1),.025),
            k==n ? 1. : Distributions.quantile(Distributions.Beta(k+1,n-k),.975)) : nothing
        (;probability=p,count=k,n,mcse=iid ? sqrt(p*(1-p)/n) : nothing,
          binomial_95_interval=interval)
    end
    (;mean=_uq_blocks(result.layout,mu),variance=_uq_blocks(result.layout,variance),
      covariance=C,probabilities=probs,quantiles=_uq_blocks(result.layout,quantiles),events=event_results,
      mean_mcse=iid ? _uq_blocks(result.layout,sqrt.(variance/n)) : nothing,
      n,failed=count(!,result.success),conditional_on_success=!all(result.success),
      layout=result.layout,iid_error_estimates=iid)
end
