# Value extraction only. No plotting dependency, solver or probabilistic target.
function _plot_indices(sol::PBMSolution,times)
    length(sol.t)==length(sol.u) || throw(DimensionMismatch("saved states and times differ"))
    times===nothing && return collect(eachindex(sol.t))
    requested=times isa Number ? [times] : collect(times)
    !isempty(requested) || throw(ArgumentError("select at least one saved time"))
    map(requested) do t
        i=findfirst(==(t),sol.t)
        i===nothing && throw(ArgumentError("time $t is not saved; rendering does not solve or interpolate"))
        i
    end
end

function _plot_population(op,u,name,cell)
    if op isa Union{CompiledAxial,CompiledAxialSystem}
        cell===nothing && throw(ArgumentError("select an axial cell explicitly"))
        1<=cell<=length(op.local_operators) || throw(ArgumentError("invalid axial cell"))
        if op isa CompiledAxialSystem
            name===nothing && throw(ArgumentError("name the population to plot"))
            localop=op.local_operators[cell]
            haskey(localop.system.populations,name) || throw(ArgumentError("unknown population $name"))
            state=axial_stateview(op,u,cell)
            return getproperty(localop.system.populations,name),getproperty(state.populations,name)
        end
        return _plot_population(op.local_operators[cell],axial_population(op,u,cell),name,nothing)
    end
    if op isa CompiledSystem
        name===nothing && throw(ArgumentError("name the population to plot"))
        haskey(op.system.populations,name) || throw(ArgumentError("unknown population $name"))
        return getproperty(op.system.populations,name),population(op,u,name)
    end
    name===nothing || throw(ArgumentError("population names apply to PBMSystem"))
    cell===nothing || cell==1 || throw(ArgumentError("batch population has one cell"))
    op,u
end

function _plot_density_snapshot(op,u,p,t;population=nothing,cell=nothing,marginal=nothing,reconstruction=nothing)
    op,u=_plot_population(op,u,population,cell)
    if reconstruction!==nothing
        result=reconstruction(u,op,p,t)
        e=collect(result.edges);y=vec(collect(result.density))
        length(e)==length(y)+1 || throw(DimensionMismatch("reconstruction needs density per edge interval"))
    else
        op isa CompiledMomentPBM && throw(ArgumentError("moment distribution plots require explicit reconstruction; use plot_observable with MomentObservable otherwise"))
        g=op.grid
        if g isa Grid1D
            marginal===nothing || throw(ArgumentError("marginal selection applies to Grid2D"))
            e=copy(g.edges);y=vec(density(g,u))
        elseif g isa Grid2D
            marginal in (:x,:y) || throw(ArgumentError("select marginal=:x/:y explicitly, or use plot_field"))
            # A sum of bin integrals already integrates the transverse axis.
            axis=marginal===:x ? g.x : g.y
            bins=vec(sum(u;dims=marginal===:x ? 2 : 1))
            e=copy(axis.edges);y=density(axis,bins)
        else
            throw(ArgumentError("unsupported population representation for density display"))
        end
    end
    all(isfinite,e) && all(diff(e).>0) && all(isfinite,y) || throw(ArgumentError("invalid displayed distribution geometry/values"))
    e,Float64.(y)
end

function _distribution_data(sol::PBMSolution;times=nothing,population=nothing,cell=nothing,
                            marginal=nothing,reconstruction=nothing)
    indices=_plot_indices(sol,times);edges=nothing;values=Vector{Float64}[]
    for i in indices
        operator,p=_observation_inputs(sol.problem,sol.t[i])
        e,y=_plot_density_snapshot(operator,sol.u[i],p,sol.t[i];
            population,cell,marginal,reconstruction)
        edges===nothing ? (edges=e) : edges==e || throw(ArgumentError("different saved reconstruction edges; compose separate axes explicitly"))
        push!(values,y)
    end
    (;values=permutedims(hcat(values...)),edges,labels=["t = $(sol.t[i])" for i in indices])
end
function _distribution_data(sol::StationaryPBMSolution;times=nothing,kwargs...)
    times===nothing || throw(ArgumentError("a stationary distribution has no output times"))
    sol.converged || throw(ArgumentError("cannot plot an unconverged stationary state as a solution"))
    e,y=_plot_density_snapshot(sol.problem.operator,sol.u,sol.problem.p,sol.problem.time;kwargs...)
    (;values=reshape(y,1,:),edges=e,labels=["Stationary state"])
end

function _measurement_errors(m;theta=nothing,experiment=nothing)
    if m isa Observation
        return m.factor===nothing ? (;sigma=copy(m.sigma)) : (;covariance=Matrix(m.factor.L*m.factor.L'))
    end
    l=m.likelihood
    if l isa GaussianLikelihood
        l.contrasts===nothing || return (;)
        value=l.sigma===nothing ? l.covariance : l.sigma
        if !(value isa Number || value isa AbstractArray)
            theta===nothing && return (;)
            value=value(theta,experiment)
        end
        return l.sigma===nothing ? (;covariance=copy(value)) : (;sigma=value isa Number ? value : copy(value))
    end
    # Truncated Gaussian's sigma is a latent-location scale, not automatically
    # the observed conditional SD. Count likelihoods also need an explicit choice.
    (;)
end

function _observable_data(sol::PBMSolution,observable::AbstractObservable;times=nothing,measurement=nothing)
    indices=_plot_indices(sol,measurement===nothing ? times : measurement.times)
    values=hcat([observe(observable,sol;index=i) for i in indices]...)
    data=measurement===nothing ? nothing : copy(measurement.data)
    errors=measurement===nothing ? (;) : _measurement_errors(measurement)
    (;times=Float64.(sol.t[indices]),values,data,errors...)
end
function _observable_data(sol::StationaryPBMSolution,o::AbstractObservable;times=nothing,measurement=nothing)
    times===nothing || throw(ArgumentError("stationary observations have no output times"))
    value=observe(o,sol)
    if measurement===nothing
        return (;times=collect(1:length(value)),values=reshape(value,1,:),xlabel="Output index")
    end
    measurement.times===nothing || throw(ArgumentError("stationary data have no timestamps"))
    length(value)==size(measurement.data,1) || throw(DimensionMismatch("stationary observable dimension differs from data"))
    (;times=collect(1:size(measurement.data,2)),values=repeat(value,1,size(measurement.data,2)),
        data=copy(measurement.data),_measurement_errors(measurement)...)
end
function _observable_data(m::Union{Measurement,Observation})
    times=m.times===nothing ? collect(1:size(m.data,2)) : copy(m.times)
    (;times,values=nothing,data=copy(m.data),_measurement_errors(m)...)
end

function _measurement_abscissa(m)
    o=m.observable
    o isa CellObservable && (o=o.observable)
    if o isa PopulationCDF
        return copy(o.points)
    elseif o isa PopulationQuantiles
        return copy(o.probabilities)
    elseif o isa HistogramObservable
        return (o.edges[1:end-1]+o.edges[2:end])/2
    elseif o isa PopulationHistogram
        e=o.histogram.edges;return (e[1:end-1]+e[2:end])/2
    elseif o isa DistributionObservable
        return (o.edges[1:end-1]+o.edges[2:end])/2
    end
    collect(1:size(m.data,1))
end

function _fit_data(problem::EstimationProblem,predicted;experiments=nothing,blocks=nothing,
                   parameter_coordinates=nothing,bands=nothing,band_label=nothing,abscissa=nothing)
    length(predicted)==length(problem.experiments) || throw(DimensionMismatch("prepared predictions must follow experiments"))
    result=NamedTuple[]
    for (i,ex) in enumerate(problem.experiments)
        experiments===nothing || ex.name in (experiments isa Symbol ? (experiments,) : experiments) || continue
        length(predicted[i])==length(ex.measurements) || throw(DimensionMismatch("prepared blocks must follow each experiment"))
        for (j,m) in enumerate(ex.measurements)
            blocks===nothing || j in (blocks isa Integer ? (blocks,) : blocks) || continue
            P=predicted[i][j]
            size(P)==size(m.data) && all(isfinite,P) || throw(DimensionMismatch("prepared measurement prediction shape/finiteness mismatch"))
            # Reuse the statistical interface's actual expected counts. Bin
            # masses/fractions are not counts until exposure/totals are applied.
            m.likelihood isa Union{PoissonLikelihood,MultinomialLikelihood} &&
                (P=_count_means(m.likelihood,m.data,P))
            theta=parameter_coordinates===nothing ? nothing : physical_parameters(problem,parameter_coordinates,ex.name)
            errors=_measurement_errors(m;theta,experiment=ex)
            single=abscissa!==nothing || (size(P,2)==1 && size(P,1)>1)
            x=abscissa!==nothing ? collect(abscissa) : single ? _measurement_abscissa(m) : m.times===nothing ? collect(1:size(P,2)) : copy(m.times)
            abscissa===nothing || length(x)==size(P,1) || throw(DimensionMismatch("explicit abscissa must match measured outputs"))
            values=single ? permutedims(P) : copy(P);data=single ? permutedims(m.data) : copy(m.data)
            if single && haskey(errors,:sigma) && !(errors.sigma isa Number)
                errors=(;sigma=permutedims(reshape(errors.sigma,size(P))))
            elseif single && haskey(errors,:covariance)
                permutation=vec(permutedims(reshape(1:length(P),size(P))))
                errors=(;covariance=errors.covariance[permutation,permutation])
            end
            block=(;times=x,values,data,identity="$(ex.name)/$(m.source===nothing ? j : m.source)",errors...)
            if single && size(P,2)>1
                block=merge(block,(;labels=m.times===nothing ? ["Replicate $i" for i in axes(P,2)] : ["t = $t" for t in m.times]))
            end
            if bands!==nothing
                band_label===nothing && throw(ArgumentError("prepared bands need an explicit band_label"))
                lo,hi=bands[i][j]
                size(lo)==size(P)==size(hi) || throw(DimensionMismatch("prepared band shape mismatch"))
                block=merge(block,(;band=single ? (permutedims(lo),permutedims(hi)) : (copy(lo),copy(hi)),band_label))
            end
            push!(result,block)
        end
    end
    isempty(result) && throw(ArgumentError("no selected measurement blocks"))
    result
end

function _field_data(sol::PBMSolution;index=length(sol.u),population=nothing,bulk=nothing,
                     orientation,cell=nothing)
    checkbounds(sol.u,index);op=sol.problem.operator;u=sol.u[index]
    if op isa Union{CompiledAxial,CompiledAxialSystem} && cell===nothing
        orientation===:axial_x || throw(ArgumentError("axial fields require orientation=:axial_x"))
        z=copy(op.domain.grid.edges)
        if bulk!==nothing
            throw(ArgumentError("bulk is a scalar profile, not an internal-coordinate density field; use plot_observable with CellObservable(BulkObservable(name),cell)"))
        end
        localop,state=_plot_population(op,u,population,1)
        localop isa CompiledMomentPBM && throw(ArgumentError("moment fields require explicit observables; states need not be raw moments"))
        localop.grid isa Grid1D || throw(ArgumentError("select a tensor axial cell explicitly; a three-coordinate field needs an explicit projection"))
        values=hcat([begin
            pop,state=_plot_population(op,u,population,c)
            vec(density(pop.grid,state))
        end for c in eachindex(op.local_operators)]...)
        return (;values,xedges=z,yedges=copy(localop.grid.edges),orientation="rows_y")
    end
    bulk===nothing || throw(ArgumentError("bulk field requires axial transport"))
    localop,state=_plot_population(op,u,population,cell)
    localop isa CompiledMomentPBM && throw(ArgumentError("moment fields need an explicit reconstruction"))
    g=localop.grid
    g isa Grid2D && orientation===:internal_x || throw(ArgumentError("Grid2D field requires orientation=:internal_x"))
    (;values=copy(density(g,state)),xedges=copy(g.x.edges),yedges=copy(g.y.edges),orientation="rows_x")
end

function _uq_data(result::Union{UQResult,PCESurrogate};output,band=:quantiles,probabilities=(.025,.975))
    summary=result isa UQResult ? uq_summary(result;probabilities,covariance=false) : uq_summary(result;covariance=false)
    _uq_plot_data(summary;output,band,probabilities,
        report=result isa UQResult ? (;failed=count(!,result.success),iid=result.metadata.iid) : result.validation)
end
function _uq_data(summary::NamedTuple;output,band=:quantiles,probabilities=(.025,.975))
    _uq_plot_data(summary;output,band,probabilities,report=(;failed=get(summary,:failed,0)))
end
function _uq_plot_data(summary;output,band,probabilities,report)
    get(summary,:conditional_on_success,false) && throw(ArgumentError("explicitly label success-conditioned summaries instead of silently dropping failed realizations"))
    haskey(summary.mean,output) || throw(ArgumentError("unknown UQ output $output"))
    times=summary.layout.times===nothing ? collect(1:size(getproperty(summary.mean,output),2)) : copy(summary.layout.times)
    base=(;times,mean=copy(getproperty(summary.mean,output)),diagnostics=report)
    band===:none && return base
    if band===:sd
        return merge(base,(;sd=sqrt.(getproperty(summary.variance,output)),band_label="Mean +/- one output SD (not a quantile interval)"))
    end
    band===:quantiles || throw(ArgumentError("band must be :none, :sd or :quantiles"))
    get(summary,:quantiles,nothing)===nothing && throw(ArgumentError("PCE analytical moments do not define quantiles; explicitly evaluate_pce first or choose band=:none/:sd"))
    selected=[findfirst(==(p),summary.probabilities) for p in probabilities]
    length(selected)==2 && all(!isnothing,selected) || throw(ArgumentError("select two available quantile endpoints"))
    q=getproperty(summary.quantiles,output)[:,:,Int.(selected)]
    merge(base,(;quantiles=copy(q),probabilities=collect(probabilities)))
end
