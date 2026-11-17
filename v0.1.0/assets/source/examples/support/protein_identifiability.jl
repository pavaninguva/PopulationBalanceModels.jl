# Included in ProteinCrystallization. Reuses the saved MAP; no optimization or MCMC.
"""Export local likelihood sensitivities separately from prior curvature.

The mixed-unit and noise-scaled SVDs use the declared inference coordinates.
Observed Hessians are local derivatives, not verified Laplace covariances.
"""
function run_identifiability(;output=joinpath(@__DIR__,"..","..",".cache","legacy_docs_assets","protein_crystallization"),bins=80,priors=:broad)
    problem = estimation_problem(;bins,solver=scheduled_solver,priors)
    z = saved_map(output,problem)
    report = identifiability_report(problem,z)
    data = observations()
    sigma = vcat(fill(data.concentration.sigma,length(data.concentration.t)),
        fill(data.D50.sigma,length(data.D50.t)))
    scaled = svdvals(report.prediction_jacobian ./ sigma)
    _table(joinpath(output,"identifiability.csv"),
        "index,prediction_singular_value,noise_scaled_singular_value,weak_likelihood_direction_z",
        [(i,report.prediction_singular_values[i],scaled[i],report.weak_direction[i]) for i in eachindex(z)])
    _table(joinpath(output,"local_curvature.csv"),"row,column,likelihood_hessian,prior_hessian",
        [(i,j,report.likelihood_hessian[i,j],report.prior_hessian[i,j]) for i in eachindex(z),j in eachindex(z)])
    report
end
