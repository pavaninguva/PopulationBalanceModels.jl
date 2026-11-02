module PopulationBalanceModelsPythonPlotExt
import PopulationBalanceModels as PBM
using PythonPlot, PythonCall

const renderer=PythonCall.pynew()
function __init__()
    # The renderer ships inside the installed Julia package, not paper/examples.
    path=joinpath(pkgdir(PBM),"python")
    sys=pyimport("sys")
    pyconvert(Bool,sys.path.__contains__(path)) || sys.path.insert(0,path)
    PythonCall.pycopy!(renderer,pyimport("pbm_plotting"))
end

_py(x::NamedTuple)=pydict(string(k)=>_py(v) for (k,v) in pairs(x))
_py(x::Symbol)=string(x)
_py(x::Tuple)=pytuple(_py(v) for v in x)
_py(x::AbstractVector{<:NamedTuple})=pylist(_py(v) for v in x)
_py(x::AbstractVector{<:Tuple})=pylist(_py(v) for v in x)
_py(x::AbstractDict)=pydict(string(k)=>_py(v) for (k,v) in pairs(x))
_py(x)=x
function _render(name,data;kwargs...)
    merged=merge(data,(;kwargs...))
    output=getproperty(renderer,name)(;(_key=>_py(value) for (_key,value) in pairs(merged))...)
    (;figure=PythonPlot.Figure(output[0]),axes=output[1])
end

function PBM.plot_distribution(sol::Union{PBM.PBMSolution,PBM.StationaryPBMSolution};times=nothing,population=nothing,cell=nothing,
                              marginal=nothing,reconstruction=nothing,kwargs...)
    data=PBM._distribution_data(sol;times,population,cell,marginal,reconstruction)
    _render(:plot_distribution,data;kwargs...)
end
function PBM.plot_observable(sol::Union{PBM.PBMSolution,PBM.StationaryPBMSolution},o::PBM.AbstractObservable;times=nothing,measurement=nothing,kwargs...)
    _render(:plot_observable,PBM._observable_data(sol,o;times,measurement);kwargs...)
end
function PBM.plot_observable(m::Union{PBM.Measurement,PBM.Observation};kwargs...)
    _render(:plot_observable,PBM._observable_data(m);kwargs...)
end
function PBM.plot_fit(problem::PBM.EstimationProblem,prepared;experiments=nothing,blocks=nothing,
                      parameter_coordinates=nothing,bands=nothing,band_label=nothing,abscissa=nothing,kwargs...)
    data=PBM._fit_data(problem,prepared;experiments,blocks,parameter_coordinates,bands,band_label,abscissa)
    _render(:plot_fit,(;blocks=data);kwargs...)
end
function PBM.plot_parameters(problem::PBM.EstimationProblem,result;coordinates=:inference,indices=nothing,priors=true,kwargs...)
    _render(:plot_parameters,PBM._parameter_data(problem,result;coordinates,indices,priors);kwargs...)
end
function PBM.plot_uq(result::Union{PBM.UQResult,PBM.PCESurrogate,NamedTuple};output,band=:quantiles,probabilities=(.025,.975),kwargs...)
    _render(:plot_uq,PBM._uq_data(result;output,band,probabilities);kwargs...)
end
function PBM.plot_field(sol::PBM.PBMSolution;index=length(sol.u),population=nothing,bulk=nothing,
                       orientation,cell=nothing,kwargs...)
    _render(:plot_field,PBM._field_data(sol;index,population,bulk,orientation,cell);kwargs...)
end
function PBM.save_figure(fig::Union{PythonPlot.Figure,PythonCall.Py,NamedTuple},path;kwargs...)
    figure=fig isa NamedTuple ? fig.figure : fig
    renderer.save_figure(figure,string(path);kwargs...)
end
end
