# Run from repository root: julia --project=examples/poc examples/poc/turing_natural_vem.jl
# NaturalOptimisers 0.2.0, pinned in that environment; requires this DCM branch.
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

import ForwardDiff, NaturalOptimisers

function initialise_local_state(rule, μ, L)
    parameters = (; η=zeros(length(μ)))
    state = Optimisers.setup(rule, parameters)
    covariance = Symmetric(L * L')
    state.η.state = merge(state.η.state, (; q=(copy(μ), Symmetric(inv(covariance)))))
    return (; parameters, state)
end

function export_phi(states)
    moments = map(states) do state
        μ, precision = state.η.state.q
        covariance = Symmetric(inv(precision))
        @assert all(isfinite, μ) && isposdef(covariance)
        (; μ=copy(μ), L=LowerTriangular(Matrix(cholesky(covariance).L)))
    end
    return (; μ=[m.μ for m in moments], L=[m.L for m in moments])
end

function fit_example()
    (; dcm, population, objective, model_builder, ps, st) = setup_example()
    rng = Random.Xoshiro(11)
    rule = NaturalOptimisers.NaturalDescent(0.02, (0.0, 0.0);
        tau=1.0, meanfield=false, manifold=NaturalOptimisers.RiemannianManifold())
    local_setup = [initialise_local_state(rule, ps.phi.μ[i], ps.phi.L[i])
        for i in eachindex(population)]
    local_parameters = [item.parameters for item in local_setup]
    local_opt_states = [item.state for item in local_setup]
    global_opt_state = Optimisers.setup(Optimisers.Adam(1e-3), ps.theta)

    for _ in 1:3
        typical, _ = predict_typ_parameters(dcm, population, ps, st)
        noise = vem_noise(dcm, ps)
        models = [model_builder(population[i], named_parameters(dcm, typical[:, i]), noise)
            for i in eachindex(population)]
        for _ in 1:20, i in eachindex(population)
            draws = NaturalOptimisers.sample(
                rng, local_parameters[i], local_opt_states[i]; num_samples=8)
            gradients = [(; η=ForwardDiff.gradient(
                z -> -DynamicPPL.logjoint(models[i], LOCAL_VARIABLES.pack(z)), draw.η))
                for draw in draws]
            local_opt_states[i], local_parameters[i] = Optimisers.update(
                local_opt_states[i], local_parameters[i], gradients)
        end

        ps = merge(ps, (; phi=export_phi(local_opt_states)))
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
    return (; ps, st, local_opt_state=local_opt_states, local_parameters, global_opt_state,
        dcm, population, objective,
        individual_ids=[i.id for i in population], predictions)
end

fit = fit_example()
