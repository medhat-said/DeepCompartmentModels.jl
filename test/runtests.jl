import Core.Compiler: return_type, isconcretetype

using Test
using DataFrames
@info "Loading local DeepCompartmentModels package..."
using DeepCompartmentModels
println("Done!")
@test Base.get_extension(DeepCompartmentModels, :DCMDynamicPPLExt) === nothing
@test Base.get_extension(DeepCompartmentModels, :DCMNaturalOptimisersExt) === nothing

@info "Starting tests..."

begin
    # TODO: Test generate_dosing_callback before individuals
    @testset "Populations and Individuals" begin
        include("lib/population.test.jl")
    end
    
    @testset "Objectives" begin
        include("objectives.test.jl")
    end

    @testset "Initializers" begin
        include("initializers.test.jl")
    end

    @testset "Mixed effect estimation" begin
        include("mixed_effects.test.jl")
    end

    @testset "Model" begin
        include("model.test.jl")
    end

    @testset "DCM" begin
        include("dcm.test.jl")
    end

    @testset "Turing VEM" begin
        include("lib/named_vem_api.test.jl")
        include("lib/natural_vem_backend.test.jl")
    end
end
