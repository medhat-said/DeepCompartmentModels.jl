# Open this file in VS Code and run "Julia: Execute Active File".
import Pkg
project_file = joinpath(@__DIR__, "Project.toml")
Base.active_project() == project_file || Pkg.activate(@__DIR__)

using CSV, DataFrames, DeepCompartmentModels, DynamicPPL, LinearAlgebra
using Plots, Random, Statistics
import Distributions, NaturalOptimisers, Optimisers

const ETA = LocalVariables(:η, (:Ka, :CL, :V))
const INITIAL = (; Ka=1.5, CL=2.5, V=30.0)
const FIT_OPTIONS = (;
    cycles=20,
    adam=(epochs=50, rate=5e-3),
    natural=(epochs=20, samples=8, rate=5e-3),
    global_step=(epochs=40, rate=1e-3),
    residual_epochs=8, # Reduce if the MC negative ELBO turns upward.
    evaluation_samples=32,
)

DynamicPPL.@model function theophylline_model(dcm, individual, theta, noise,
    observation=get_y(individual))
    η ~ Distributions.MvNormal(zeros(ETA.dimension), noise.Ω)
    eta = ETA(η)
    pars = (; Ka=theta.Ka * exp(eta.Ka),
        CL=theta.CL * exp(eta.CL), V=theta.V * exp(eta.V))
    prediction = DeepCompartmentModels.predict(dcm, individual, pars)
    observation ~ Distributions.MvNormal(prediction, noise.σ^2 * I)
    return (; pars, prediction)
end

function load_theophylline(file)
    groups = groupby(DataFrame(CSV.File(file)), :ID)
    return Population([
        begin
            observed = group.MDV .== 0
            dose = group.DOSE[1] * group.WEIGHT[1] # mg/kg -> mg
            callback = generate_dosing_callback(reshape([group.TIME[1], dose], 1, 2), Float64)
            Individual(group.ID[1], Float64[], Float64.(group.TIME[observed]),
                Float64.(group.DV[observed]), callback, Float64)
        end for group in groups
    ])
end

function setup_fit(population)
    init_global(_, dims...) = reshape(log.(expm1.(collect(INITIAL))), dims...)
    layer = AddGlobalParameters(length(INITIAL), 1:length(INITIAL), Float64;
        init_theta=init_global, activation=Lux.softplus)
    dcm = DCM(one_comp_abs!, layer, AdditiveError(1.0);
        target=2, parameter_names=keys(INITIAL))
    builder(individual, theta, noise) = theophylline_model(dcm, individual, theta, noise)
    objective = VariationalEM(builder; local_variables=ETA)
    ps, st = setup(objective, Random.Xoshiro(1), dcm, population, Float64;
        init_omega=0.09)
    L = LowerTriangular(0.2 * Matrix{Float64}(I, ETA.dimension, ETA.dimension))
    phi = (; μ=[zeros(ETA.dimension) for _ in population],
        L=[copy(L) for _ in population])
    return (; dcm, builder, objective, ps=merge(ps, (; phi)), st)
end

function fit_method(method, setup_values, population, options)
    local_settings = getproperty(options, method)
    local_optimizer = method === :adam ? Optimisers.Adam(local_settings.rate) :
                      NaturalOptimisers.NaturalDescent(local_settings.rate, (0.0, 0.0); tau=1.0,
        meanfield=false, manifold=NaturalOptimisers.RiemannianManifold())
    cycles = parse(Int, get(ENV, "DCM_THEO_CYCLES", string(options.cycles)))
    evaluation_samples = parse(Int, get(ENV, "DCM_THEO_EVAL_SAMPLES",
        string(options.evaluation_samples)))
    settings = (; cycles, local_epochs=local_settings.epochs,
        global_epochs=options.global_step.epochs, m_epochs=options.residual_epochs,
        samples=options.natural.samples)
    global_optimizer = Optimisers.Adam(options.global_step.rate)
    rng = Random.Xoshiro(11)
    fit = fit_vem(rng, setup_values.objective, setup_values.dcm,
        population, deepcopy(setup_values.ps), deepcopy(setup_values.st);
        local_optimizer, global_optimizer,
        merge(settings, (; cycles=0))..., verbose=false)
    fit = merge(fit, (; dcm=setup_values.dcm, objective=setup_values.objective))
    mc_history = [(; cycle=0, negative_elbo=mc_negative_elbo(fit, population;
        samples=evaluation_samples, seed=41))]
    training_history = NamedTuple[]
    for cycle in 1:settings.cycles
        fit = fit_vem(rng, setup_values.objective, setup_values.dcm, population,
            fit.ps, fit.st; local_optimizer, global_optimizer,
            local_opt_state=fit.local_opt_state, global_opt_state=fit.global_opt_state,
            merge(settings, (; cycles=1))..., verbose=false)
        fit = merge(fit, (; dcm=setup_values.dcm, objective=setup_values.objective))
        push!(training_history, (; cycle,
            negative_elbo=only(fit.history).negative_elbo))
        value = mc_negative_elbo(fit, population;
            samples=evaluation_samples, seed=41)
        push!(mc_history, (; cycle, negative_elbo=value))
        println("$(method) cycle $cycle: MC negative ELBO = $value")
    end
    return merge(fit, (; method, dcm=setup_values.dcm,
        builder=setup_values.builder, objective=setup_values.objective,
        settings, history=training_history, mc_history,
        completed_cycles=settings.cycles))
end

function mc_negative_elbo(fit, population; samples=32, seed=41)
    rng = Random.Xoshiro(seed)
    state = deepcopy(fit.st)
    return mean(1:samples) do _
        update_epsilon!(rng, state)
        fit.objective(fit.dcm, population, fit.ps, state)
    end
end

function estimate_row(label, fit, population)
    typical, _ = predict_typ_parameters(fit.dcm, population, fit.ps, fit.st)
    theta = named_parameters(fit.dcm, typical[:, 1])
    omega_sd = sqrt.(diag(fit.ps.omega))
    correlation = fit.ps.omega ./ (omega_sd * omega_sd')
    return (; Method=label, theta..., IIV_Ka=omega_sd[1], IIV_CL=omega_sd[2],
        IIV_V=omega_sd[3], Rho_Ka_CL=correlation[1, 2],
        Rho_Ka_V=correlation[1, 3], Rho_CL_V=correlation[2, 3],
        Residual_SD=vem_noise(fit.dcm, fit.ps).σ,
        MC_negative_ELBO=last(fit.mc_history).negative_elbo)
end

function posterior_curves(fit, population, i, grid; draws=100, seed=1000 + i)
    rng = Random.Xoshiro(seed)
    typical, _ = predict_typ_parameters(fit.dcm, population, fit.ps, fit.st)
    theta = named_parameters(fit.dcm, typical[:, i])
    model = fit.builder(population[i], theta, vem_noise(fit.dcm, fit.ps))
    covariance = Matrix(fit.ps.phi.L[i] * fit.ps.phi.L[i]')
    latent_draws = rand(rng, Distributions.MvNormal(fit.ps.phi.μ[i], covariance), draws)
    return [DeepCompartmentModels.predict(fit.dcm, population[i],
        DynamicPPL.returned(model, ETA.pack(latent_draws[:, draw])).pars;
        saveat=grid) for draw in 1:draws]
end

function curve_interval(curves)
    values = reduce(hcat, curves)
    summaries = [[quantile(collect(row), probability)
                  for probability in (0.025, 0.5, 0.975)] for row in eachrow(values)]
    return reduce(hcat, summaries)
end

function plot_comparison(adam_fit, natural_fit, population; draws=100)
    columns = min(3, length(population));
    rows = cld(length(population), columns)
    figure = plot(layout=(rows, columns), size=(360 * columns, 260 * rows),
        link=:both, xlabel="Time (h)", ylabel="Concentration (mg/L)")
    for i in eachindex(population)
        individual = population[i]
        grid = collect(range(0.0, maximum(get_t(individual)); length=250))
        for (fit, label, colour) in ((adam_fit, "Adam q", :steelblue),
            (natural_fit, "Natural q", :darkorange))
            interval = curve_interval(posterior_curves(fit, population, i, grid; draws))
            plot!(figure, grid, interval[2, :]; subplot=i, label=i == 1 ? label : "",
                ribbon=(interval[2, :] - interval[1, :], interval[3, :] - interval[2, :]),
                color=colour, fillalpha=0.12, linewidth=2)
        end
        scatter!(figure, get_t(individual), get_y(individual); subplot=i,
            label=i == 1 ? "Observed" : "", color=:black, markersize=3, markerstrokewidth=0,
            title="Individual $(individual.id)")
    end
    return figure
end

function plot_elbo(adam_fit, natural_fit)
    figure = plot(xlabel="VEM cycle", ylabel="MC negative ELBO (lower is better)",
        markershape=:circle, linewidth=2)
    for (fit, label, colour) in ((adam_fit, "Adam q", :steelblue),
            (natural_fit, "Natural q", :darkorange))
        plot!(figure, getproperty.(fit.mc_history, :cycle),
            getproperty.(fit.mc_history, :negative_elbo);
            label, color=colour)
    end
    return figure
end

data_file = get(ENV, "DCM_DATA_FILE",
    joinpath(@__DIR__, "..", "..", ".scratch", "theophylline_nmready.csv"))
isfile(data_file) || error("Dataset not found: $data_file")
population = load_theophylline(data_file)
setup_values = setup_fit(population)

println("Fitting standard Gaussian VI (Adam)...")
adam_fit = fit_method(:adam, setup_values, population, FIT_OPTIONS)
println("Fitting NaturalOptimisers Gaussian VI...")
natural_fit = fit_method(:natural, setup_values, population, FIT_OPTIONS)

comparison = DataFrame([
    estimate_row("Adam q", adam_fit, population),
    estimate_row("Natural q", natural_fit, population),
])
println(comparison)

output = joinpath(@__DIR__, "output")
mkpath(output)
CSV.write(joinpath(output, "theophylline_vem_estimates.csv"), comparison)
elbo_history = vcat(
    DataFrame(Method="Adam q", Cycle=getproperty.(adam_fit.mc_history, :cycle),
        MC_negative_ELBO=getproperty.(adam_fit.mc_history, :negative_elbo)),
    DataFrame(Method="Natural q", Cycle=getproperty.(natural_fit.mc_history, :cycle),
        MC_negative_ELBO=getproperty.(natural_fit.mc_history, :negative_elbo)))
CSV.write(joinpath(output, "theophylline_vem_elbo.csv"), elbo_history)
if get(ENV, "DCM_THEO_PLOT", "1") == "1"
    figure = plot_comparison(adam_fit, natural_fit, population;
        draws=parse(Int, get(ENV, "DCM_THEO_PLOT_DRAWS", "100")))
    savefig(figure, joinpath(output, "theophylline_vem_profiles.png"))
    elbo_figure = plot_elbo(adam_fit, natural_fit)
    savefig(elbo_figure, joinpath(output, "theophylline_vem_elbo.png"))
    display(figure)
    display(elbo_figure)
end
