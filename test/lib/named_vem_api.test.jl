module NamedVEMAPITests

using DeepCompartmentModels, DynamicPPL, LinearAlgebra, Random, Test
using ForwardDiff, Zygote

@test Base.get_extension(DeepCompartmentModels, :DCMDynamicPPLExt) !== nothing

DynamicPPL.@model function api_model(theta, noise, observation=0.0)
    η ~ MvNormal(zeros(2), noise.Ω)
    observation ~ Normal(theta.CL + η[1], noise.σ)
    return (; theta, noise)
end

DynamicPPL.@model function extra_site_model(theta, noise)
    η ~ MvNormal(zeros(2), noise.Ω)
    unwanted ~ Normal()
end

DynamicPPL.@model function no_site_model(theta, noise)
    return theta
end

@testset "Named VEM API" begin
    locals = LocalVariables(:η, (:CL, :V))
    dcm = DCM(one_comp_abs!, 2, Lux.Chain(), AdditiveError(0.1);
        target=2, parameter_names=(:Ka, :CL, :V))
    cb = generate_dosing_callback(reshape([0.0, 1.0], 1, 2), Float64)
    person = Individual("1", [1.0, 0.2, 2.0], [0.5, 1.0], zeros(2), cb, Float64)
    population = Population([person])
    builder = (individual, theta, noise) -> api_model(theta, noise)
    objective = VariationalEM(builder; local_variables=locals)
    ps, st = setup(objective, Random.Xoshiro(1), dcm, population, Float64)

    @test locals.dimension == 2
    @test locals([0.1, 0.2]) == (; CL=0.1, V=0.2)
    @test locals.pack([0.1, 0.2]) == (; η=[0.1, 0.2])
    latent_loss = x -> locals(x).CL^2 + sum(abs2, x)
    @test only(Zygote.gradient(latent_loss, [0.1, 0.2])) ≈
        ForwardDiff.gradient(latent_loss, [0.1, 0.2])
    @test LocalVariables(:z, (:V, :CL))([0.1, 0.2]) == (; V=0.1, CL=0.2)
    @test LocalVariables(:η, (:CL,))([0.1]) == (; CL=0.1)
    @test_throws DimensionMismatch locals([0.1])
    @test_throws ArgumentError LocalVariables(:η, (:CL, :CL))
    @test_throws ArgumentError LocalVariables(:η, ())
    @test_throws ArgumentError LocalVariables(:η, (:CL, "V"))

    theta = named_parameters(dcm, [1.0, 0.2, 2.0])
    @test theta == (; Ka=1.0, CL=0.2, V=2.0)
    parameter_loss = x -> named_parameters(dcm, x).CL^2 + sum(abs2, x)
    @test only(Zygote.gradient(parameter_loss, [1.0, 0.2, 2.0])) ≈
        ForwardDiff.gradient(parameter_loss, [1.0, 0.2, 2.0])
    @test_throws DimensionMismatch named_parameters(dcm, [1.0, 0.2])
    @test_throws ArgumentError DCM(one_comp_abs!, 2, Lux.Chain(); parameter_names=(:CL, :CL))
    @test DeepCompartmentModels.predict(dcm, person, (; V=2.0, Ka=1.0, CL=0.2)) ≈
        DeepCompartmentModels.predict(dcm, person, [1.0, 0.2, 2.0])
    @test_throws ArgumentError DeepCompartmentModels.predict(dcm, person, (; Ka=1.0, CL=0.2))
    @test_throws ArgumentError DeepCompartmentModels.predict(dcm, person, (; Ka=1.0, CL=0.2, volume=2.0))
    unnamed = DCM(one_comp_abs!, 2, Lux.Chain(), AdditiveError(0.1); target=2)
    @test_throws ArgumentError named_parameters(unnamed, [1.0, 0.2, 2.0])
    @test_throws ArgumentError DeepCompartmentModels.predict(unnamed, person, theta)
    @test_throws ArgumentError setup(objective, Random.Xoshiro(1), unnamed, population)

    noise = vem_noise(dcm, ps)
    @test noise.Ω == ps.omega
    @test noise.σ ≈ 0.1
    @test isfinite(objective(dcm, population, ps, st))
    returned = DynamicPPL.returned(builder(person, theta, noise), (; η=zeros(2)))
    @test returned.theta == theta
    @test returned.noise == noise
    custom_noise = (dcm, ps) -> (; Ω=ps.omega, σ=2 * Lux.softplus(only(ps.error.σ)))
    custom = VariationalEM(builder; local_variables=locals, noise=custom_noise)
    @test isfinite(custom(dcm, population, ps, st))
    @test custom(dcm, population, ps, st) != objective(dcm, population, ps, st)
    for error in (ProportionalError(0.1), CombinedError([0.1, 0.2]))
        other_dcm = DCM(one_comp_abs!, 2, Lux.Chain(), error)
        other_ps = (; omega=ps.omega, error=setup(error, nothing))
        other_noise = vem_noise(other_dcm, other_ps)
        if error isa CombinedError
            @test other_noise.σ_additive ≈ 0.1
            @test other_noise.σ_proportional ≈ 0.2
        else
            @test other_noise.σ ≈ 0.1
        end
    end
    @test_throws ArgumentError vem_noise(DCM(one_comp_abs!, 2, Lux.Chain(), CustomError([0.1])), ps)

    @test_throws UndefKeywordError VariationalEM(builder)

    wrong_site = VariationalEM(builder; local_variables=LocalVariables(:wrong, (:CL, :V)))
    @test_throws ErrorException setup(wrong_site, Random.Xoshiro(1), dcm, population)
    wrong_size = VariationalEM(builder; local_variables=LocalVariables(:η, (:CL,)))
    @test_throws DimensionMismatch setup(wrong_size, Random.Xoshiro(1), dcm, population)
    extra = VariationalEM((i, t, n) -> extra_site_model(t, n); local_variables=locals)
    @test_throws ErrorException setup(extra, Random.Xoshiro(1), dcm, population)
    absent = VariationalEM((i, t, n) -> no_site_model(t, n); local_variables=locals)
    @test_throws ArgumentError setup(absent, Random.Xoshiro(1), dcm, population)
end

end
