# Open in VS Code and run "Julia: Execute Active File".
import Pkg
Pkg.activate(@__DIR__)

using CSV, DataFrames, DeepCompartmentModels, DynamicPPL, LinearAlgebra, Plots, Random
import Distributions, NaturalOptimisers, Optimisers

# CHANGE: random-effect names and initial typical PK values.
const ETA = LocalVariables(:η, (:Ka, :CL, :V))
const INITIAL = (; Ka=1.5, CL=2.5, V=30.0) # Order must match the ODE.
const FIT = (;
    cycles=10,
    local_epochs=20,      # Natural updates of each individual q(η).
    global_epochs=20,     # Adam updates of typical PK parameters.
    m_epochs=1,           # Set to 0 to hold σ and Ω fixed.
    samples=8,            # Monte Carlo draws per Natural/M-step update.
    natural_rate=5e-3,
    global_rate=1e-3,
)

function load_theophylline(file)
    # CHANGE: column names and dosing construction for another dataset.
    groups = groupby(DataFrame(CSV.File(file)), :ID)
    individuals = [
        begin
            observed = group.MDV .== 0
            dose = group.DOSE[1] * group.WEIGHT[1] # mg/kg -> mg
            callback = generate_dosing_callback(reshape([group.TIME[1], dose], 1, 2), Float64)
            Individual(group.ID[1], Float64[], Float64.(group.TIME[observed]),
                Float64.(group.DV[observed]), callback, Float64)
        end for group in groups
    ]
    return Population(individuals)
end

DynamicPPL.@model function individual_model(dcm, individual, theta, noise,
    observation=get_y(individual))
    η ~ Distributions.MvNormal(zeros(ETA.dimension), noise.Ω)
    eta = ETA(η)
    # CHANGE: covariate and IIV relationships belong here.
    pars = (; Ka=theta.Ka * exp(eta.Ka),
        CL=theta.CL * exp(eta.CL), V=theta.V * exp(eta.V))
    prediction = DeepCompartmentModels.predict(dcm, individual, pars)
    # CHANGE: replace this likelihood for another residual-error model.
    observation ~ Distributions.MvNormal(prediction, noise.σ^2 * I)
    return (; pars, prediction)
end

data_file = get(ENV, "DCM_DATA_FILE",
    joinpath(@__DIR__, "..", "..", ".scratch", "theophylline_nmready.csv"))
population = load_theophylline(data_file)

init_global(_, dims...) = reshape(log.(expm1.(collect(INITIAL))), dims...)
parameter_model = AddGlobalParameters(3, 1:3, Float64;
    init_theta=init_global, activation=Lux.softplus)
# CHANGE: select the ODE, parameter layer, error model, and observed compartment.
dcm = DCM(one_comp_abs!, parameter_model, AdditiveError(1.0);
    target=2, parameter_names=keys(INITIAL))

model_builder(individual, theta, noise) = individual_model(dcm, individual, theta, noise)
objective = VariationalEM(model_builder; local_variables=ETA)
ps, st = setup(objective, Random.Xoshiro(1), dcm, population, Float64;
    init_omega=0.09)

natural = NaturalOptimisers.NaturalDescent(FIT.natural_rate, (0.0, 0.0);
    tau=1.0, meanfield=false,
    manifold=NaturalOptimisers.RiemannianManifold()) # Full-covariance q(η).

fit = fit_vem(Random.Xoshiro(11), objective, dcm, population, ps, st;
    local_optimizer=natural,
    global_optimizer=Optimisers.Adam(FIT.global_rate),
    cycles=parse(Int, get(ENV, "DCM_THEO_CYCLES", string(FIT.cycles))),
    local_epochs=FIT.local_epochs, global_epochs=FIT.global_epochs,
    m_epochs=FIT.m_epochs, samples=FIT.samples)

ps, st = fit.ps, fit.st
typical, _ = predict_typ_parameters(dcm, population, ps, st)
noise = vem_noise(dcm, ps)
predictions = map(eachindex(population)) do i
    theta = named_parameters(dcm, typical[:, i])
    model = model_builder(population[i], theta, noise)
    DynamicPPL.returned(model, ETA.pack(ps.phi.μ[i])).prediction
end

observed = reduce(vcat, [get_y(individual) for individual in population])
predicted = reduce(vcat, predictions)
limit = 1.05 * max(maximum(observed), maximum(predicted))
figure = scatter(observed, predicted; label=nothing,
    xlabel="Observed concentration", ylabel="Individual prediction")
plot!(figure, [0, limit], [0, limit]; color=:black, label=nothing)
display(figure)

println("Typical parameters = ", named_parameters(dcm, typical[:, 1]))
println("ETA SDs = ", ETA(sqrt.(diag(ps.omega))))
println("Residual SD = ", noise.σ)
println("Natural optimiser state is available as fit.local_opt_state")
