# Run with each audited --project; append the rows to environments.csv.
using Pkg
label=isempty(ARGS) ? "core" : first(ARGS)
println("environment,julia,platform,package,version,uuid,tree_hash,direct")
for (uuid,info) in sort!(collect(Pkg.dependencies());by=x->x.second.name)
    println(join((label,VERSION,Sys.MACHINE,info.name,something(info.version,"stdlib"),
                  uuid,something(info.tree_hash,""),info.is_direct_dep),','))
end
