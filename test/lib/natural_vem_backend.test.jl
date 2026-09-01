module NaturalVEMBackendTests

using DeepCompartmentModels, DynamicPPL, ForwardDiff, LinearAlgebra
using NaturalOptimisers, Random, Statistics, Test

@test Base.get_extension(DeepCompartmentModels, :DCMNaturalOptimisersExt) !== nothing

const LOCALS = LocalVariables(:η, (:CL, :V))

DynamicPPL.@model function gaussian_model(individual, noise,
        observation=get_y(individual))
    η ~ MvNormal(zeros(2), noise.Ω)
    observation ~ MvNormal(η, 0.25 * I)
end

function fixture(n=2)
    dcm = DCM(one_comp_abs!, 2, Lux.Chain(), AdditiveError(0.1);
        target=2, parameter_names=(:Ka, :CL, :V))
    cb = generate_dosing_callback(reshape([0.0, 1.0], 1, 2), Float64)
    population = Population([Individual("$i", [1.0, 0.2, 2.0],
        [0.5, 1.0], [0.6, -0.3], cb, Float64) for i in 1:n])
    builder = (individual, theta, noise) -> gaussian_model(individual, noise)
    objective = VariationalEM(builder; local_variables=LOCALS)
    ps, st = setup(objective, Random.Xoshiro(1), dcm, population, Float64;
        init_omega=1.0, scale=1.0)
    return (; dcm, population, objective, ps, st)
end

@testset "selectable local VI backend" begin
    (; dcm, population, objective, ps, st) = fixture()
    rule = NaturalDescent(0.03, (0.0, 0.0); tau=1.0, meanfield=false,
        manifold=RiemannianManifold())
    initialised = fit_vem(Random.Xoshiro(3), objective, dcm, population, ps, st;
        local_optimizer=rule, cycles=0, global_epochs=0, verbose=false)
    extension = Base.get_extension(DeepCompartmentModels, :DCMNaturalOptimisersExt)
    initial_moments = extension._extract_moments(initialised.local_opt_state.states[1], :η)
    @test initial_moments.μ == ps.phi.μ[1]
    @test initial_moments.L == ps.phi.L[1]

    draws = NaturalOptimisers.sample(Random.Xoshiro(3),
        initialised.local_opt_state.parameters[1], initialised.local_opt_state.states[1];
        num_samples=2_000)
    draw_matrix = reduce(hcat, getindex.(draws, :η))
    @test vec(mean(draw_matrix; dims=2)) ≈ ps.phi.μ[1] atol=0.08
    @test cov(draw_matrix; dims=2) ≈ Matrix(I, 2, 2) atol=0.08

    noise = vem_noise(dcm, ps)
    model = gaussian_model(population[1], noise)
    eta = [0.1, -0.2]
    density_gradient = ForwardDiff.gradient(eta) do value
        -DeepCompartmentModels._individual_logjoint(model, LOCALS.pack(value))
    end
    @test density_gradient ≈ eta + 4 .* (eta - get_y(population[1]))

    fit = fit_vem(Random.Xoshiro(4), objective, dcm, population, ps, st;
        local_optimizer=rule, local_opt_state=initialised.local_opt_state,
        cycles=1, local_epochs=75,
        global_epochs=0, m_epochs=0, samples=32, verbose=false)

    posterior_covariance = inv(Matrix(I, 2, 2) + 4 * Matrix(I, 2, 2))
    posterior_mean = posterior_covariance * (4 .* [0.6, -0.3])
    @test fit.ps.phi.μ[1] ≈ posterior_mean atol=0.12
    @test Matrix(fit.ps.phi.L[1] * fit.ps.phi.L[1]') ≈ posterior_covariance atol=0.12
    @test all(L -> isposdef(Symmetric(L * L')), fit.ps.phi.L)
    @test fit.individual_ids == ["1", "2"]
    @test length(fit.local_opt_state.states) == 2
    @test fit.local_opt_state === initialised.local_opt_state
    @test fit.local_opt_state.states[1] !== fit.local_opt_state.states[2]
    @test length(fit.history) == 1

    resumed = fit_vem(Random.Xoshiro(5), objective, dcm, population, fit.ps, fit.st;
        local_optimizer=rule, local_opt_state=fit.local_opt_state,
        global_opt_state=fit.global_opt_state, cycles=1, local_epochs=1,
        global_epochs=0, m_epochs=0, samples=4, verbose=false)
    @test resumed.local_opt_state === fit.local_opt_state
    @test all(isfinite, resumed.ps.phi.μ[1])

    adam = fit_vem(Random.Xoshiro(4), objective, dcm, population, ps, st;
        local_optimizer=Optimisers.Adam(0.01), cycles=1, local_epochs=1,
        global_epochs=0, m_epochs=0, samples=2, verbose=false)
    @test all(isfinite, adam.ps.phi.μ[1])

    wrong = NaturalDescent(0.03; tau=1.0, meanfield=true,
        manifold=RiemannianManifold())
    @test_throws ArgumentError fit_vem(Random.Xoshiro(4), objective, dcm,
        population, ps, st; local_optimizer=wrong, cycles=0,
        global_epochs=0, verbose=false)
end

end
