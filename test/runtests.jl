import Core.Compiler: return_type, isconcretetype

using Test
using DataFrames
@info "Loading local DeepCompartmentModels package..."
using DeepCompartmentModels
println("Done!")

@info "Starting tests..."

# Wrapped in an outer testset so that a failure in one group is reported but does not
# abort the remaining groups: only the outermost testset throws, and it does so at the end.
@testset "DeepCompartmentModels" begin
    # The extensions must not be loaded by `using DeepCompartmentModels` alone; the test
    # files that need them load their weak dependency themselves.
    @testset "Extension loading" begin
        @test Base.get_extension(DeepCompartmentModels, :DCMDynamicPPLExt) === nothing
        @test Base.get_extension(DeepCompartmentModels, :DCMNaturalOptimisersExt) === nothing
    end

    # NOTE: the groups below were unreachable until the include paths were corrected to
    # `lib/` and this outer testset was added, and have gone stale against the current
    # source in the meantime: `Objectives` calls a `make_dist` signature that no longer
    # exists and `DeepCompartmentModels._get_prior`, which was removed; `Initializers` and
    # `Mixed effect estimation` cover `init_omega`/`init_phi`/`make_etas`, also removed.
    # They are left running rather than skipped so the rot stays visible. Porting them is
    # a separate piece of work from the VEM extension.

    # TODO: Test generate_dosing_callback before individuals
    @testset "Populations and Individuals" begin
        include("lib/population.test.jl")
    end

    @testset "Objectives" begin
        include("lib/objectives.test.jl")
    end

    @testset "Initializers" begin
        include("lib/initializers.test.jl")
    end

    @testset "Mixed effect estimation" begin
        include("lib/mixed_effects.test.jl")
    end

    @testset "Model" begin
        include("lib/model.test.jl")
    end

    @testset "DCM" begin
        include("lib/dcm.test.jl")
    end

    @testset "Turing VEM" begin
        include("lib/named_vem_api.test.jl")
        include("lib/natural_vem_backend.test.jl")
    end
end
