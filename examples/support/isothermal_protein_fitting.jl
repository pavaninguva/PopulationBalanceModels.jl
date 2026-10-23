# Included by the paper driver, after loading Optim and JSON3. No solve on import.
module IsothermalProteinFitting
using PopulationBalanceModels, Optim, JSON3, Random, SHA, Dates
using ..IsothermalProteinCrystallization
const Case = IsothermalProteinCrystallization

function json(path,x)
    mkpath(dirname(path))
    open(path,"w") do io
        JSON3.pretty(io,x); println(io)
    end
end
function csv(path,rows)
    isempty(rows) && error("cannot export an empty table: $path")
    escape(x)=occursin(r"[,\n\"]",string(x)) ? "\""*replace(string(x),'"'=>"\"\"")*"\"" : string(x)
    open(path,"w") do io
        println(io,join(keys(first(rows)),','))
        for r in rows
            println(io,join(escape.(values(r)),','))
        end
    end
end

function fit_case(output;bins=100,Lmax=50.,dt=.5,cfl=.35,cstar=2.16,
        window="isothermal_core",sc=.1,weight=1.,points=256,gated=false,
        nstarts=8,maxiters=400,seed=2027,initial=nothing,tag="baseline")
    settings=(;bins,Lmax,dt,cfl,cstar,window,sc,weight,points,gated,nstarts,maxiters,seed,
        initial_theta=initial===nothing ? Float64[] : collect(initial))
    # Hash numerical/statistical definitions and authoritative data, not reporting code.
    root=dirname(dirname(@__DIR__))
    files=[joinpath(root,"examples","isothermal_protein_crystallization.jl"),@__FILE__,
        joinpath(Case.DATA,"concentration.csv"),joinpath(Case.DATA,"psd_cdf.csv"),
        joinpath(root,"Project.toml"),Base.active_project(),
        joinpath(dirname(Base.active_project()),"Manifest.toml")]
    for sub in ("src","ext"), (dir,_,names) in walkdir(joinpath(root,sub)), name in names
        endswith(name,".jl") && push!(files,joinpath(dir,name))
    end
    hashes=Dict(relpath(f,root)=>bytes2hex(sha256(read(f))) for f in files if isfile(f))
    code=join(k*":"*hashes[k]*"\n" for k in sort(collect(keys(hashes))))
    hash=bytes2hex(sha256(code*JSON3.write(settings)*string(VERSION)))
    folder=joinpath(output,"fits",tag*"_"*hash[1:12]);mkpath(folder)
    cal=Case.calibration(;bins,Lmax,dt,cfl,cstar,window,sc,weight,points,gated)
    rng=MersenneTwister(seed)
    theta0=initial===nothing ? [log(1e11),log(.2),2.,2.] : collect(initial)[1:4]
    gated && push!(theta0,initial!==nothing && length(initial)==5 ? initial[5] : 60.)
    starts=[Case.coordinates(theta0;gated)]
    for k in 2:nstarts
        # Reproducible, dispersed physical starts within the declared search box.
        theta=[log(1e11)+2.5randn(rng), log(.15)+.7randn(rng), .5+4rand(rng), .5+3rand(rng)]
        gated && push!(theta,20+100rand(rng))
        push!(starts,Case.coordinates(theta;gated))
    end
    rows=[]
    for (k,u0) in enumerate(starts)
        path=joinpath(folder,"start_$(k).json")
        if isfile(path)
            record=JSON3.read(read(path,String));push!(rows,record)
            println("Reuse ",tag," start ",k," loss=",record.loss);flush(stdout)
            continue
        end
        println("Fit ",tag," bins=",bins," start ",k,"/",nstarts);flush(stdout)
        t0=time(); result=nothing; failure=""
        try
            result=fit(cal,u0;maxiters,
                algorithm=Optim.BFGS(linesearch=Optim.LineSearches.BackTracking()),
                options=(g_abstol=1e-4,f_reltol=1e-10,x_abstol=1e-8,))
        catch e
            (e isa NumericalFailure || e isa Optim.LineSearches.LineSearchException) || rethrow()
            failure=sprint(showerror,e)
        end
        seconds=time()-t0
        u=result===nothing ? u0 : result.minimizer
        loss=result===nothing ? Inf : result.minimum
        contributions=isfinite(loss) ? loss_contributions(cal,u) : []
        record=(;start=k,initial_theta=Case.physical_coordinates(u0;gated),u=collect(u),
            theta=Case.physical_coordinates(u;gated),loss=isfinite(loss) ? loss : nothing,
            concentration_loss=isempty(contributions) ? nothing : first(contributions).value,
            distribution_loss=isempty(contributions) ? nothing : sum(r.value for r in contributions[2:end]),
            converged=result!==nothing && result.converged,seconds,
            iterations=result===nothing ? 0 : Optim.iterations(result.raw),
            invalid_trials=result===nothing ? 1 : result.failed_evaluations,
            failure_messages=result===nothing ? [failure] : result.failure_messages)
        json(path,record);push!(rows,record)
        println("  loss=",record.loss," c=",record.concentration_loss," PSD=",record.distribution_loss,
                " theta=",record.theta," converged=",record.converged," seconds=",seconds);flush(stdout)
    end
    valid=filter(r->r.loss!==nothing && isfinite(r.loss),rows)
    isempty(valid) && error("No valid fit for $tag; inspect $folder")
    best=valid[argmin([r.loss for r in valid])]
    json(joinpath(folder,"fit.json"),(;settings,hash,best,starts=rows,source_hashes=hashes,
        interpretation="deterministic comparison loss; no probability model or confidence interval"))
    (;cal,best,rows,settings,folder,theta=collect(best.theta),u=collect(best.u))
end
end
