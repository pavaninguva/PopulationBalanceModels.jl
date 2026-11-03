include("posterior_chain_aggregation.jl")

function _parallel_protein_chains(problem,settings,output)
    # Only this fixed physical recipe is launched; workers receive numeric
    # settings, not a closure with shared mutable solver caches.
    directory=mktempdir()
    Serialization.serialize(joinpath(directory,"settings.bin"),settings)
    worker=joinpath(@__DIR__,"protein_posterior_worker.jl")
    println("Isolated chain logs: ",directory);flush(stdout)
    started=time_ns()
    @sync for chain in 1:settings.chains
        @async begin
            command=`$(Base.julia_cmd()) --threads=1 --project=$(dirname(Base.active_project())) $worker $directory $chain`
            logfile=joinpath(directory,"chain_$chain.txt")
            open(logfile,"w") do io
                try
                    run(pipeline(command;stdout=io,stderr=io))
                catch error
                    throw(ErrorException("isolated chain $chain failed; inspect $logfile: $(sprint(showerror,error))"))
                end
            end
            println("Isolated posterior chain ",chain," completed");flush(stdout)
        end
    end
    elapsed_seconds=(time_ns()-started)/1e9
    runs=[Serialization.deserialize(joinpath(directory,"chain_$chain.bin")) for chain in 1:settings.chains]
    logdir=joinpath(output,"sampling_logs");mkpath(logdir)
    for chain in 1:settings.chains
        cp(joinpath(directory,"chain_$chain.txt"),joinpath(logdir,"chain_$chain.txt");force=true)
    end
    _combine_protein_chains(problem,runs;elapsed_seconds)
end
