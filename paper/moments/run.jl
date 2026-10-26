module PaperMoments
using PopulationBalanceModels, TOML
include("../_support/common.jl")
using .PaperWorkflow
include(joinpath(ROOT,"examples","qamar_moment_benchmarks.jl"))
const Q = QamarMomentBenchmarks

function run(mode="production")
    ctx = context(@__DIR__,mode)
    cfg = ctx.config
    choices = [("qmom3",QMOM(3)),("adaptive3",RankAdaptiveQMOM(3)),
        ("eqmom3",EQMOM(3;secondary_nodes=cfg["eqmom_secondary_nodes"])),
        ("dqmom3",DQMOM(3)),("lognormal",LogNormalClosure(quadrature_order=cfg["lognormal_quadrature"])),
        ("qmom4",QMOM(4))]
    times = collect(0.:cfg["save_interval"]:cfg["final_time"])
    runs = Dict{String,String}()
    for (id,method) in choices
        warm = Q.problem4(;method,tf=cfg["dt"])
        simulate(warm;dt=cfg["dt"],method=SSPRK33())
        for refinement in (1,2)
            id=="qmom4" && refinement==2 && continue
            dt = cfg["dt"]/refinement
            options = method isa EQMOM ? Dict("primary_nodes"=>3,"secondary_nodes"=>method.secondary_nodes,"rtol"=>method.rtol) :
                method isa LogNormalClosure ? Dict("quadrature_order"=>cfg["lognormal_quadrature"]) :
                Dict("nodes"=>id=="qmom4" ? 4 : 3)
            settings = Dict("method"=>id,"representation"=>string(nameof(typeof(method))),"options"=>options,"dt"=>dt,
                "api_time_method"=>"SSPRK33","tf"=>cfg["final_time"],"save_interval"=>cfg["save_interval"],
                "beta0"=>1.,"a0"=>1.,"V0"=>.001,"initial_number"=>1.)
            key = "$(id)_dt$(refinement)"
            folder = cached(ctx,key,settings) do output
                pb = Q.problem4(;method,tf=cfg["final_time"])
                GC.gc()
                measured = @timed simulate(pb;dt,method=SSPRK33(),saveat=times)
                solution = measured.value
                histories = NamedTuple[]
                ranks = Int[]
                residuals = Float64[]
                for (t,u) in zip(solution.t,solution.u)
                    moments = [Q.comparison_moment(pb.operator,u,k) for k in 0:3]
                    q = quadrature(pb.operator.grid,u,method)
                    rank = method isa RankAdaptiveQMOM ? q.rank : method isa LogNormalClosure ? 0 : 3
                    id=="qmom4" && (rank=4)
                    push!(ranks,rank)
                    du = zero(u);rhs!(du,u,pb.operator,pb.p,t)
                    # Independent event count, using this family's material
                    # moment. Assumed-shape material is not forced to V0.
                    expected_rate = -.5moments[1]^2+moments[4]
                    actual_rate = moment_rate(method,pb.operator.grid,u,du,0)
                    push!(residuals,abs(actual_rate-expected_rate))
                    reference = Q.exact4(t,0:3)
                    for k in 0:3
                        value,exact = moments[k+1],reference[k+1]
                        push!(histories,(;method=id,time_s=t,order=k,moment=value,exact,
                            signed_difference=value-exact,signed_relative=(value-exact)/exact,
                            absolute_relative_percent=100abs(value/exact-1),
                            tracked=k in pb.operator.grid.orders,active_rank=rank))
                    end
                end
                csv(joinpath(output,"moments.csv"),histories)
                initial3 = Q.comparison_moment(pb.operator,first(solution.u),3)
                final3 = Q.comparison_moment(pb.operator,last(solution.u),3)
                summary = (;method=id,states=length(pb.u0),dt,seconds=measured.time,steps=solution.steps,
                    completed_time=last(solution.t),status="completed",minimum_rank=minimum(ranks),maximum_rank=maximum(ranks),
                    tracked_M3=3 in pb.operator.grid.orders,initial_material_ratio=initial3/.001,
                    final_material_ratio=final3/.001,material_relative_change=final3/initial3-1,
                    max_number_rate_residual=maximum(residuals),
                    max_mu1_error_percent=maximum(r.absolute_relative_percent for r in histories if r.order==1),
                    max_mu2_error_percent=maximum(r.absolute_relative_percent for r in histories if r.order==2),
                    quadrature_points=length(quadrature(pb.operator.grid,first(solution.u),method).nodes))
                csv(joinpath(output,"summary.csv"),[summary])
                (["moments.csv","summary.csv"],Dict("status"=>"completed","seconds"=>measured.time,"steps"=>solution.steps))
            end
            runs[key] = folder
        end
    end
    index(ctx,runs)
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    PaperMoments.run("--smoke" in ARGS ? "smoke" : "production")
end
