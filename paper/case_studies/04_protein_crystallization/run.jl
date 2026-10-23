using PopulationBalanceModels, Optim, JSON3, SHA, Dates, LinearAlgebra, Pkg
include(joinpath(@__DIR__,"..","..","..","examples","isothermal_protein_crystallization.jl"))
include(joinpath(@__DIR__,"..","..","..","examples","support","isothermal_protein_fitting.jl"))
using .IsothermalProteinCrystallization
using .IsothermalProteinFitting: fit_case, csv, json

const ROOT=realpath(joinpath(@__DIR__,"..","..",".."))
const MODE="--smoke" in ARGS ? "smoke" : "production"
const OUTPUT=joinpath(ROOT,"paper","results","04_protein_crystallization",MODE)
const CONFIG=JSON3.read(read(joinpath(IsothermalProteinCrystallization.DATA,"config","case.json"),String))
mkpath(OUTPUT)
normpath(dirname(dirname(pathof(PopulationBalanceModels))))==ROOT || error("load the checkout, not a registry release")
const START_STATE=(revision=strip(read(`git -C $ROOT rev-parse HEAD`,String)),
    dirty=read(`git -C $ROOT status --porcelain=v1 --untracked-files=all`,String))
loaded_observations=load_data()
initial=only(select_concentration(loaded_observations,:initial_condition))
initial.model_time_min==0 && initial.concentration_g_L==CONFIG.initial_condition.c0_g_L==IsothermalProteinCrystallization.C0 ||
    error("initial observation and fixed model/configuration do not agree")
for (role,expected) in ((:calibration,CONFIG.split.calibration_concentration_ids),
                        (:validation,CONFIG.split.validation_concentration_ids))
    [r.observation_id for r in select_concentration(loaded_observations,role)]==String.(expected) ||
        error("concentration selector disagrees with the declared $role IDs")
end
[first(r).time_min for r in select_psds(loaded_observations)]==collect(CONFIG.split.calibration_PSD_times_min) ||
    error("PSD calibration times disagree with the fixed split")

if MODE=="smoke"
    sol=IsothermalProteinCrystallization.minimal_run()
    json(joinpath(OUTPUT,"smoke.json"),(;mode=MODE,steps=sol.steps,
        note="uncalibrated forward solve, not paper results",START_STATE))
else
    baseline=fit_case(OUTPUT;tag="baseline")
    if "--baseline-only" in ARGS
        exit()
    end
    fine_cal=calibration(bins=200)
    coarse_outputs=predictions(baseline.cal,baseline.u)[1]
    fine_outputs=predictions(fine_cal,baseline.u)[1]
    baseline_change=max(maximum(abs.(coarse_outputs[1]-fine_outputs[1]))/.1,
        maximum(maximum(abs.(coarse_outputs[k]-fine_outputs[k]))/.5 for k in 2:4))
    json(joinpath(OUTPUT,"baseline_numerical_check.json"),(;scaled_change=baseline_change,
        threshold=.25,calibration_only=true))
    baseline100=baseline
    if baseline_change>.25
        baseline=fit_case(OUTPUT;bins=200,initial=baseline.theta,tag="baseline_refined")
    end
    # The retained concentration and PSD group losses identify a baseline
    # calibration mismatch; examine only the predeclared single extension.
    gated=fit_case(OUTPUT;bins=baseline.settings.bins,gated=true,initial=baseline.theta,tag="onset_candidate")
    selected_gate=gated.best.loss<=.8baseline.best.loss &&
        gated.best.concentration_loss<=1.1baseline.best.concentration_loss &&
        gated.best.distribution_loss<=1.1baseline.best.distribution_loss
    chosen=selected_gate ? gated : baseline
    json(joinpath(OUTPUT,"calibration_selection.json"),(;
        baseline_loss=baseline.best.loss,candidate_loss=gated.best.loss,selected_gate,
        criterion="20% lower total calibration loss; neither group >1.1 baseline; no validation used",
        baseline_groups=(baseline.best.concentration_loss,baseline.best.distribution_loss),
        candidate_groups=(gated.best.concentration_loss,gated.best.distribution_loss)))
    include("report.jl")
end
