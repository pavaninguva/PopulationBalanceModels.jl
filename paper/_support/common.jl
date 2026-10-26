module PaperWorkflow
using PopulationBalanceModels, SHA, TOML, Dates, Pkg, LinearAlgebra

const ROOT = dirname(dirname(@__DIR__))
const PAPER = dirname(@__DIR__)
normpath(dirname(dirname(pathof(PopulationBalanceModels)))) == normpath(ROOT) ||
    error("The paper must load PopulationBalanceModels from this checkout; run paper/setup.jl")

filehash(path) = bytes2hex(sha256(read(path)))
function digest_files(files)
    io = IOBuffer()
    for path in sort(files)
        print(io,relpath(path,ROOT),'\0',filehash(path),'\n')
    end
    bytes2hex(sha256(take!(io)))
end
function source_files()
    files = String[]
    for dir in ("src","ext","examples"), (root,_,names) in walkdir(joinpath(ROOT,dir)), name in names
        endswith(name,".jl") && push!(files,joinpath(root,name))
    end
    push!(files,joinpath(ROOT,"Project.toml"))
    files
end
function configuration(folder,mode)
    mode in ("smoke","production") || error("mode must be smoke or production")
    TOML.parsefile(joinpath(folder,"config.toml"))[mode]
end
function context(folder,mode;support=String[])
    config = configuration(folder,mode)
    source = digest_files(source_files())
    workflow = digest_files(vcat([@__FILE__,joinpath(folder,"run.jl"),joinpath(folder,"config.toml")],support))
    project = Base.active_project()
    manifest = joinpath(dirname(project),"Manifest.toml")
    environment_digest = digest_files(isfile(manifest) ? [project,manifest] : [project])
    name = basename(folder)
    output = mode=="smoke" ? joinpath(ROOT,".cache","paper","checks",name) : joinpath(PAPER,name,"data")
    mkpath(output)
    (;name,mode,config,source,workflow,environment_digest,output,folder)
end
function csv(path,rows)
    isempty(rows) && error("Refuse an empty numerical table: $path")
    quotevalue(x) = occursin(r"[,\n\"]",string(x)) ? "\""*replace(string(x),'"'=>"\"\"")*"\"" : string(x)
    open(path,"w") do io
        println(io,join(keys(first(rows)),','))
        for row in rows
            keys(row)==keys(first(rows)) || error("inconsistent CSV columns")
            println(io,join(quotevalue.(values(row)),','))
        end
    end
end
function write_toml(path,data)
    open(path,"w") do io
        TOML.print(io,data;sorted=true)
    end
end
function environment()
    versions = Dict(string(v.name)=>string(v.version) for v in values(Pkg.dependencies()) if v.version!==nothing)
    Dict("julia"=>string(VERSION),"packages"=>versions,
        "cpu"=>first(Sys.cpu_info()).model,"cpu_target"=>Sys.CPU_NAME,
        "julia_threads"=>Threads.nthreads(),"blas_threads"=>BLAS.get_num_threads())
end
function cached(compute,ctx,id,settings)
    canonical = IOBuffer()
    TOML.print(canonical,settings;sorted=true)
    key = bytes2hex(sha256(ctx.source*ctx.workflow*ctx.environment_digest*ctx.mode*String(take!(canonical))))
    folder = joinpath(ctx.output,id*"_"*key[1:12])
    record = joinpath(folder,"provenance.toml")
    if isfile(record)
        p = TOML.parsefile(record)
        if get(p,"complete",false) && p["key"]==key &&
            all(isfile(joinpath(folder,f["path"])) && filehash(joinpath(folder,f["path"]))==f["sha256"] for f in p["files"])
            println("Reuse ",ctx.name," / ",id);flush(stdout)
            return folder
        end
    end
    mkpath(folder)
    revision = strip(read(`git -C $ROOT rev-parse HEAD`,String))
    dirty = read(`git -C $ROOT status --porcelain=v1 --untracked-files=all`,String)
    println("Run ",ctx.name," / ",id," ",settings);flush(stdout)
    files,summary = try
        compute(folder)
    catch failure
        write_toml(joinpath(folder,"failure.toml"),Dict("key"=>key,"case"=>ctx.name,
            "mode"=>ctx.mode,"settings"=>settings,"revision"=>revision,"working_tree"=>dirty,
            "source_digest"=>ctx.source,"workflow_digest"=>ctx.workflow,
            "status"=>"failed","error"=>sprint(showerror,failure)))
        rethrow()
    end
    p = Dict("schema"=>1,"case"=>ctx.name,"mode"=>ctx.mode,"complete"=>true,
        "key"=>key,"source_digest"=>ctx.source,"workflow_digest"=>ctx.workflow,
        "environment_digest"=>ctx.environment_digest,
        "revision"=>revision,"dirty"=>!isempty(dirty),"working_tree"=>dirty,
        "created_utc"=>string(now(UTC)),"settings"=>settings,"summary"=>summary,
        "environment"=>environment(),"timing_protocol"=>"one measured solve after separate compilation warmup; setup, explicit GC, export and plotting excluded",
        "files"=>[Dict("path"=>f,"sha256"=>filehash(joinpath(folder,f))) for f in files])
    write_toml(record,p)
    folder
end
function index(ctx,runs)
    write_toml(joinpath(ctx.output,"index.toml"),Dict("schema"=>1,"case"=>ctx.name,
        "mode"=>ctx.mode,"source_digest"=>ctx.source,"workflow_digest"=>ctx.workflow,
        "configuration"=>ctx.config,"runs"=>Dict(k=>relpath(v,ctx.output) for (k,v) in runs),
        "derived_files"=>[Dict("path"=>f,"sha256"=>filehash(joinpath(ctx.output,f)))
            for f in readdir(ctx.output) if endswith(f,".csv")]))
end
function method_label(x)
    string(nameof(typeof(x)))
end

"""Exact PWC L1 integral on the union of two bin partitions."""
function overlap_error(edges,a,refedges,b)
    first(edges)≈first(refedges) && last(edges)≈last(refedges) || error("comparison domains differ")
    common = sort!(unique(vcat(edges,refedges)))
    value = 0.0
    for i in 1:length(common)-1
        x = (common[i]+common[i+1])/2
        ai = min(searchsortedlast(edges,x),length(a))
        bi = min(searchsortedlast(refedges,x),length(b))
        value += (common[i+1]-common[i])*abs(a[ai]-b[bi])
    end
    value/sum(b.*diff(refedges))
end

export ROOT,PAPER,context,csv,write_toml,cached,index,configuration,filehash,overlap_error
end
