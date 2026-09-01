# Run from repository root: julia --project=examples/poc examples/poc/turing_model_vem.jl
# Requires this DCM branch; three cycles demonstrate wiring, not convergence.
using DeepCompartmentModels, DynamicPPL, LinearAlgebra, Random
import Optimisers

const LOCAL_VARIABLES = LocalVariables(:η, (:CL, :V))

DynamicPPL.@model function individual_model(dcm, individual, theta, noise,
        observation=get_y(individual))
    η ~ MvNormal(zeros(LOCAL_VARIABLES.dimension), noise.Ω)
    eta = LOCAL_VARIABLES(η)
    pars = (; Ka=theta.Ka, CL=theta.CL * exp(eta.CL), V=theta.V * exp(eta.V))
    prediction = DeepCompartmentModels.predict(dcm, individual, pars)
    observation ~ MvNormal(prediction, noise.σ^2 * I)
    return (; pars, eta, prediction)
end

function setup_example()
    initial = (; Ka=1.0, CL=1.2, V=10.0)
    init_global(_, dims...) = reshape(log.(expm1.(collect(initial))), dims...)
    layer = AddGlobalParameters(3, [1, 2, 3], Float64;
        init_theta=init_global, activation=Lux.softplus)
    dcm = DCM(one_comp_abs!, layer, AdditiveError(0.3);
        target=2, parameter_names=keys(initial))

    rng = Random.Xoshiro(7)
    times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0]
    population = Population(map(1:2) do i
        cb = generate_dosing_callback(reshape([0.0, 100.0], 1, 2), Float64)
        template = Individual("p$i", Float64[], times, zeros(length(times)), cb, Float64)
        eta = i == 1 ? (; CL=-0.2, V=-0.1) : (; CL=0.2, V=0.15)
        pars = (; Ka=initial.Ka, CL=initial.CL * exp(eta.CL), V=initial.V * exp(eta.V))
        y = DeepCompartmentModels.predict(dcm, template, pars) +
            0.3 * randn(rng, length(times))
        Individual("p$i", Float64[], times, y, cb, Float64)
    end)

    model_builder(individual, theta, noise) = individual_model(dcm, individual, theta, noise)
    objective = VariationalEM(model_builder; local_variables=LOCAL_VARIABLES)
    ps, st = setup(objective, Random.Xoshiro(1), dcm, population, Float64; init_omega=0.09)
    n_eta = LOCAL_VARIABLES.dimension
    phi = (; μ=[zeros(n_eta) for _ in population],
        L=[LowerTriangular(0.2 * Matrix(I, n_eta, n_eta)) for _ in population])
    return (; dcm, population, objective, model_builder, ps=merge(ps, (; phi)), st)
end

function fit_example()
    (; dcm, population, objective, model_builder, ps, st) = setup_example()
    rng = Random.Xoshiro(11)
    local_opt_state = Optimisers.setup(Optimisers.Adam(0.01), ps.phi)
    global_opt_state = Optimisers.setup(Optimisers.Adam(1e-3), ps.theta)

    for _ in 1:3
        for _ in 1:20
            update_epsilon!(rng, st)
            grad = DeepCompartmentModels.gradient(objective, dcm, population, ps, st)
            local_opt_state, phi = Optimisers.update(local_opt_state, ps.phi, grad.phi)
            ps = merge(ps, (; phi))
        end

        update_epsilon!(rng, st)
        grad = DeepCompartmentModels.gradient(objective, dcm, population, ps, st)
        global_opt_state, theta = Optimisers.update(global_opt_state, ps.theta, grad.theta)
        ps = merge(ps, (; theta))

        ps = m_step(objective, rng, dcm, population, ps, st;
            epochs=1, num_samples=8, verbose=false)
    end
    typical, _ = predict_typ_parameters(dcm, population, ps, st)
    predictions = [DynamicPPL.returned(model_builder(population[i],
        named_parameters(dcm, typical[:, i]), vem_noise(dcm, ps)),
        LOCAL_VARIABLES.pack(ps.phi.μ[i])).prediction for i in eachindex(population)]
    @assert all(p -> all(isfinite, p), predictions)
    println("Typical parameters: ", named_parameters(dcm, typical[:, 1]))
    println("Individual ETA means: ", ps.phi.μ)
    return (; ps, st, local_opt_state, global_opt_state, dcm, population, objective,
        individual_ids=[i.id for i in population], predictions)
end

fit = fit_example()
