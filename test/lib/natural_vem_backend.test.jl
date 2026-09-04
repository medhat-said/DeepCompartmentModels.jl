module NaturalVEMBackendTests

using DeepCompartmentModels, DynamicPPL, ForwardDiff, LinearAlgebra
using NaturalOptimisers, Random, Statistics, Test

@test Base.get_extension(DeepCompartmentModels, :DCMNaturalOptimisersExt) !== nothing

const LOCALS = LocalVariables(:η, (:CL, :V))

# Conjugate Gaussian target with an analytic posterior for any latent dimension.
DynamicPPL.@model function gaussian_model(individual, noise,
        observation=get_y(individual))
    η ~ MvNormal(zeros(eltype(noise.Ω), size(noise.Ω, 1)), noise.Ω)
    observation ~ MvNormal(η, 0.25 * I)
end

# `locals` declares the latent dimension; the observations are sized to match so that the
# likelihood is the conjugate one above.
function fixture(n=2; locals=LOCALS, T=Float64)
    d = locals.dimension
    dcm = DCM(one_comp_abs!, 2, Lux.Chain(), AdditiveError(0.1);
        target=2, parameter_names=(:Ka, :CL, :V))
    cb = generate_dosing_callback(reshape([0.0, 1.0], 1, 2), T)
    times = T.(range(0.5, 1.0; length=d + 1)[1:d])
    observations = T.(range(0.6, -0.3; length=d + 1)[1:d])
    population = Population([Individual("$i", T[1.0, 0.2, 2.0],
        times, observations, cb, T) for i in 1:n])
    builder = (individual, theta, noise) -> gaussian_model(individual, noise)
    objective = VariationalEM(builder; local_variables=locals)
    ps, st = setup(objective, Random.Xoshiro(1), dcm, population, T;
        init_omega=1.0, scale=1.0)
    return (; dcm, population, objective, ps, st, observations)
end

riemannian(T=Float64) = NaturalDescent(T(0.03), (T(0.0), T(0.0)); tau=T(1.0),
    meanfield=false, manifold=RiemannianManifold())

# Exact posterior of the conjugate model above.
function analytic_posterior(observations)
    d = length(observations)
    covariance = inv(Matrix(I, d, d) + 4 * Matrix(I, d, d))
    return covariance, covariance * (4 .* observations)
end

@testset "selectable local VI backend" begin
    (; dcm, population, objective, ps, st, observations) = fixture()
    rule = riemannian()
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

    posterior_covariance, posterior_mean = analytic_posterior(observations)
    @test fit.ps.phi.μ[1] ≈ posterior_mean atol=0.12
    @test Matrix(fit.ps.phi.L[1] * fit.ps.phi.L[1]') ≈ posterior_covariance atol=0.12
    @test all(L -> isposdef(Symmetric(L * L')), fit.ps.phi.L)
    @test fit.individual_ids == ["1", "2"]
    @test length(fit.local_opt_state.states) == 2
    @test fit.local_opt_state === initialised.local_opt_state
    @test fit.local_opt_state.states[1] !== fit.local_opt_state.states[2]
    @test length(fit.history) == 1

    reordered = Population(reverse(population.data))
    @test_throws ArgumentError fit_vem(Random.Xoshiro(5), objective, dcm,
        reordered, fit.ps, fit.st; local_optimizer=rule,
        local_opt_state=fit.local_opt_state, cycles=1, local_epochs=1,
        global_epochs=0, m_epochs=0, samples=4, verbose=false)
    changed_rule = NaturalDescent(0.01, (0.0, 0.0); tau=1.0,
        meanfield=false, manifold=RiemannianManifold())
    @test_throws ArgumentError fit_vem(Random.Xoshiro(5), objective, dcm,
        population, fit.ps, fit.st; local_optimizer=changed_rule,
        local_opt_state=fit.local_opt_state, cycles=1, local_epochs=1,
        global_epochs=0, m_epochs=0, samples=4, verbose=false)

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
    # A wrong rule is reported even when the optimiser state is supplied.
    @test_throws ArgumentError fit_vem(Random.Xoshiro(4), objective, dcm,
        population, ps, st; local_optimizer=wrong,
        local_opt_state=initialised.local_opt_state, cycles=1, local_epochs=1,
        global_epochs=0, m_epochs=0, verbose=false)
    tempered = NaturalDescent(0.03; tau=0.5, meanfield=false,
        manifold=RiemannianManifold())
    @test_throws ArgumentError fit_vem(Random.Xoshiro(4), objective, dcm,
        population, ps, st; local_optimizer=tempered,
        local_opt_state=initialised.local_opt_state, cycles=0,
        global_epochs=0, verbose=false)

    for manifold in (LieGroupManifold(), EuclidianManifold())
        other_rule = NaturalDescent(0.01, (0.0, 0.0); tau=1.0,
            meanfield=false, manifold)
        other = fit_vem(Random.Xoshiro(4), objective, dcm, population, ps, st;
            local_optimizer=other_rule, cycles=1, local_epochs=1,
            global_epochs=0, m_epochs=0, samples=4, verbose=false)
        @test all(L -> isposdef(Symmetric(L * L')), other.ps.phi.L)
    end
end

# Smoke-test that no local-VI code is specialised to two random effects.
@testset "arbitrary latent dimension: $(name)" for (name, locals) in [
        ("1 named", LocalVariables(:η, (:CL,))),
        ("3 named", LocalVariables(:η, (:CL, :V, :Ka))),
        ("5 anonymous", LocalVariables(:η, 5)),
        ("12 anonymous", LocalVariables(:η, 12))]
    (; dcm, population, objective, ps, st, observations) = fixture(; locals)
    d = locals.dimension
    @test size(ps.omega) == (d, d)
    @test all(μ -> length(μ) == d, ps.phi.μ)

    fit = fit_vem(Random.Xoshiro(4), objective, dcm, population, ps, st;
        local_optimizer=riemannian(), cycles=1, local_epochs=2,
        global_epochs=0, m_epochs=0, samples=4, verbose=false)

    L = fit.ps.phi.L[1]
    @test size(L) == (d, d)
    @test all(isfinite, fit.ps.phi.μ[1])
    @test all(factor -> isposdef(Symmetric(factor * factor')), fit.ps.phi.L)
end

# Probabilistic programming backends accumulate log densities in Float64, so the latent
# gradient comes back widened. The backend must narrow it rather than fail, otherwise the
# default element type of `setup` cannot be fitted at all.
@testset "Float32 parameters" begin
    (; dcm, population, objective, ps, st, observations) = fixture(; T=Float32)
    @test eltype(ps.phi.μ[1]) == Float32
    @test eltype(ps.omega) == Float32

    noise = vem_noise(dcm, ps)
    model = gaussian_model(population[1], noise)
    widened = ForwardDiff.gradient(Float32[0.1, -0.2]) do value
        -DeepCompartmentModels._individual_logjoint(model, LOCALS.pack(value))
    end
    @test eltype(widened) == Float64

    fit = fit_vem(Random.Xoshiro(4), objective, dcm, population, ps, st;
        local_optimizer=riemannian(Float32), cycles=1, local_epochs=2,
        global_epochs=0, m_epochs=0, samples=4, verbose=false)
    @test eltype(fit.ps.phi.μ[1]) == Float32
    @test eltype(fit.ps.phi.L[1]) == Float32
    @test all(isfinite, fit.ps.phi.μ[1])

    # A rule written with Float64 literals is adapted to the element type of the
    # parameters rather than widening them.
    mixed = fit_vem(Random.Xoshiro(4), objective, dcm, population, ps, st;
        local_optimizer=riemannian(Float64), cycles=1, local_epochs=2,
        global_epochs=0, m_epochs=0, samples=4, verbose=false)
    @test eltype(mixed.ps.phi.μ[1]) == Float32
    @test eltype(mixed.ps.phi.L[1]) == Float32
    @test all(isfinite, mixed.ps.phi.μ[1])
end

end
