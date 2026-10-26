# A small warmed-path allocation/timing probe; not part of the timing benchmarks.
# julia --project=. test/tools/profile_paths.jl LABEL > performance.csv
using PopulationBalanceModels
include(joinpath(pkgdir(PopulationBalanceModels),"examples","qamar_moment_benchmarks.jl"))

function measure_path(label, name, pb, repetitions)
    op=pb.operator;du=zero(pb.u0)
    run()=rhs!(du,pb.u0,op,pb.p,0.)
    for _ in 1:20;run();end
    GC.gc()
    bytes=@allocated for _ in 1:repetitions;run();end
    elapsed=@elapsed for _ in 1:repetitions;run();end
    println(join((label,VERSION,name,repetitions,bytes/repetitions,elapsed/repetitions,
                  sum(abs2,du)),','))
end
function profile_paths(label=isempty(ARGS) ? "working_tree" : first(ARGS))
    println("revision,julia,path,evaluations,bytes_per_evaluation,seconds_per_evaluation,squared_residual_norm")
    grid=Grid1D(range(0.,10.;length=201))
    op=compile(PBMModel(growth=Growth((x,p,t,c)->p.G),death=Death((x,p,t,c)->p.d)),grid)
    sectional=PBMProblem(op,exp.(-grid.centers),(0.,1.);p=(G=.1,d=.2))
    measure_path(label,"sectional_200",sectional,1000)
    measure_path(label,"QMOM_3_collision",QamarMomentBenchmarks.problem4(),500)
end
profile_paths()
