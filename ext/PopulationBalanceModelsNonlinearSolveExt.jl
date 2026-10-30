module PopulationBalanceModelsNonlinearSolveExt

using PopulationBalanceModels
import NonlinearSolve
const PBM=PopulationBalanceModels

# Gaussian moment inversion uses eigensystems that do not support Dual entries.
# This is a nonlinear Jacobian choice, not a change of closure/physical equations.
_quadrature_residual(op)=false
# Even a fully analytical evolution plan must validate the explicitly selected
# family. Its inversion can still require finite-difference state Jacobians.
_quadrature_residual(op::PBM.CompiledMomentPBM)=!(op.closure isa PBM.ExactClosure)
_quadrature_residual(op::PBM.CompiledSystem)=any(_quadrature_residual,values(op.system.populations))
_quadrature_residual(op::Union{PBM.CompiledAxial,PBM.CompiledAxialSystem})=any(_quadrature_residual,op.local_operators)

function PBM.nonlinearproblem(prob::PBM.StationaryPBMProblem)
    f! = function(r,x,p)
        PBM.stationary_residual!(reshape(r,size(prob.guess)),
            reshape(x,size(prob.guess)),prob,
            p isa NonlinearSolve.SciMLBase.NullParameters ? prob.p : p)
    end
    # As in odeproblem: old SciML remake paths recurse for an empty NamedTuple.
    p=prob.p isa NamedTuple && isempty(prob.p) ? NonlinearSolve.SciMLBase.NullParameters() : prob.p
    NonlinearSolve.NonlinearProblem(f!,vec(copy(prob.guess)),p)
end

function PBM.solve_stationary(prob::PBM.StationaryPBMProblem,algorithm=nothing;
        abstol=1e-9,reltol=0.0,maxiters=1000,kwargs...)
    isfinite(abstol) && abstol>0 && isfinite(reltol) && reltol>=0 ||
        throw(ArgumentError("stationary tolerances require abstol>0 and reltol>=0"))
    maxiters isa Integer && maxiters>0 || throw(ArgumentError("maxiters must be a positive integer"))
    r0=similar(prob.guess)
    PBM.stationary_residual!(r0,prob.guess,prob)
    initial_norm=maximum(abs,r0)
    alg=if algorithm!==nothing
        algorithm
    elseif _quadrature_residual(prob.operator)
        NonlinearSolve.RobustMultiNewton(autodiff=NonlinearSolve.ADTypes.AutoFiniteDiff())
    else
        NonlinearSolve.RobustMultiNewton()
    end
    started=time_ns()
    raw=NonlinearSolve.solve(PBM.nonlinearproblem(prob),alg;abstol,reltol,maxiters,kwargs...)
    elapsed=(time_ns()-started)/1e9
    u=reshape(copy(raw.u),size(prob.guess))
    r=similar(u)
    PBM.stationary_residual!(r,u,prob)
    balances=copy(PBM._stationary_physical(prob.operator,PBM._stationary_rates(prob,u,prob.p)))
    norm_r=maximum(abs,r)
    norm_balance=maximum(abs,balances./prob.residual_scale)
    threshold=abstol+reltol*initial_norm
    passed=NonlinearSolve.SciMLBase.successful_retcode(raw) && max(norm_r,norm_balance)<=threshold
    d=(policy=:stationary,algorithm=Symbol(nameof(typeof(alg))),
        scaled_residual_norm=norm_r,scaled_balance_norm=norm_balance,
        balance_norm=maximum(abs,balances),initial_residual_norm=initial_norm,
        acceptance_tolerance=threshold,elapsed_seconds=elapsed,stats=raw.stats,
        autonomous=prob.autonomous,evaluation_time=prob.time,
        physical_unknowns=length(u),constraints=length(prob.constraints))
    PBM.StationaryPBMSolution(u,prob,r,balances,passed,raw.retcode,raw,d)
end

end
