"""Stage the two milling paper figures' data, or explicitly recalculate the study.

Default/--retained: validate and extract existing production exports.
--full --output=/fresh/path: run the shared forward and 12,000-solve UQ drivers.
Including this module never starts simulations or writes files.
"""
module PaperMilling
using SHA, TOML
import JSON3

const ROOT = normpath(joinpath(@__DIR__,"../.."))
const CONFIG = joinpath(@__DIR__,"config.toml")
const SUMMARIES = ("nominal.csv","mc_summary.csv","pce1_summary.csv","pce2_summary.csv")

sha(path) = bytes2hex(sha256(read(path)))

function producer(folder, mode)
    path = joinpath(folder,"provenance.json")
    record = JSON3.read(read(path,String))
    record.complete && record.mode == mode || error("Expected completed $mode data: $path")
    for (name,digest) in pairs(record.numerical_files)
        sha(joinpath(folder,string(name))) == digest || error("Changed numerical data: $name")
    end
    record
end

function put(path, content)
    bytes = content isa String ? Vector{UInt8}(codeunits(content)) : content
    if isfile(path)
        read(path) == bytes || error("Refusing to replace retained data: $path; use a fresh --output directory")
    else
        mkpath(dirname(path))
        write(path,bytes)
    end
end

function stage(forward, uq, output)
    cfg = TOML.parsefile(CONFIG)
    f, u = producer(forward,"forward_comparison"), producer(uq,"production")
    u.settings.mc_samples == 12000 || error("Paper UQ needs the completed 12,000-realization ensemble")
    cfg["forward_bins"] in f.configuration.calculation.bins || error("Missing selected forward grid")
    reference = joinpath(ROOT,cfg["reference_source"])
    sha(reference) == f.configuration.reference_sha256 || error("Changed reference readings")
    mkpath(output)
    # Select one grid and the same pivot-CDF observation as the UQ example.
    # Keep source decimal strings unchanged; no refitting or interpolation.
    lines = readlines(joinpath(forward,"trajectories.csv"))
    column = findfirst(==("bins"),split(first(lines),','))
    selected = [line for line in lines[2:end] if parse(Int,split(line,',')[column]) == cfg["forward_bins"]]
    isempty(selected) && error("No selected forward trajectory")
    put(joinpath(output,"forward.csv"),join(vcat(first(lines),selected),'\n')*"\n")
    put(joinpath(output,"reference.csv"),read(reference))
    put(joinpath(output,"forward_producer.json"),read(joinpath(forward,"provenance.json")))
    put(joinpath(output,"uq_producer.json"),read(joinpath(uq,"provenance.json")))
    for name in SUMMARIES
        put(joinpath(output,name),read(joinpath(uq,name)))
    end
    files = Dict(name => sha(joinpath(output,name)) for name in
        ("forward.csv","reference.csv","forward_producer.json","uq_producer.json",SUMMARIES...))
    index = joinpath(output,"index.json")
    if isfile(index)
        saved = JSON3.read(read(index,String))
        saved.files == JSON3.read(JSON3.write(files)) && saved.configuration == JSON3.read(JSON3.write(cfg)) ||
            error("Existing paper selection differs; use a fresh --output directory")
        println("Reusing verified milling paper data: ",output)
        return output
    end
    sources = vcat(read(joinpath(forward,"provenance.json")),read(joinpath(uq,"provenance.json")))
    workflow = read(@__FILE__,String)*read(CONFIG,String)
    record = (;complete=true,case="milling",mode="production",configuration=cfg,files,
        source_digest=bytes2hex(sha256(sources)),workflow_digest=bytes2hex(sha256(workflow)),
        staging_revision=strip(read(`git -C $ROOT rev-parse HEAD`,String)),
        staging_working_tree=read(`git -C $ROOT status --short`,String),
        forward_source=relpath(forward,ROOT),uq_source=relpath(uq,ROOT),
        forward_derivation=(;source_file="trajectories.csv",source_sha256=f.numerical_files["trajectories.csv"],
            filter_bins=cfg["forward_bins"],statistic=cfg["forward_statistic"],
            transformation="Row selection only; stored ordinates unchanged"),
        reference_sha256=sha(reference),
        producer_note="Original producers are preserved byte-for-byte. Staging/redrawing does not assert current-source numerical equivalence.")
    open(io -> JSON3.pretty(io,record),index,"w")
    println("Staged production milling curves and original producer records: ",output)
    output
end

function run(;full=false,output=joinpath(@__DIR__,"data"))
    cfg = TOML.parsefile(CONFIG)
    if full
        (!ispath(output) || isempty(readdir(output))) || error("--full requires a fresh --output directory")
        # These are the package's existing physical builders and checked BDF/UQ
        # workflows; no paper-specific solver or kinetic model is introduced.
        include(joinpath(ROOT,"examples/verification/milling_forward.jl"))
        forward, uq = joinpath(output,"full_forward"),joinpath(output,"full_uq")
        Base.invokelatest(MillingForward.run;output=forward)
        Base.invokelatest(MillingForward.MU.run;full=true,output=uq)
    else
        forward, uq = joinpath(ROOT,cfg["forward_source"]),joinpath(ROOT,cfg["uq_source"])
    end
    stage(forward,uq,output)
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    all(arg -> arg in ("--full","--retained") || startswith(arg,"--output="),ARGS) ||
        error("Usage: run.jl [--retained | --full --output=/fresh/path]")
    !("--full" in ARGS && "--retained" in ARGS) || error("Choose --full or --retained")
    directories = [split(arg,'=';limit=2)[2] for arg in ARGS if startswith(arg,"--output=")]
    length(directories) <= 1 || error("Specify --output once")
    "--full" in ARGS && isempty(directories) && error("Use --full --output=/fresh/path to preserve retained results")
    output = isempty(directories) ? joinpath(@__DIR__,"data") : abspath(only(directories))
    PaperMilling.run(;full="--full" in ARGS,output)
end
