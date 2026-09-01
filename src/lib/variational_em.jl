abstract type AbstractVariationalFamily end

"""Full-rank Gaussian approximation for individual latent variables."""
struct FullRankGaussian <: AbstractVariationalFamily end

"""Declare a vector-valued local site and its component names."""
struct LocalVariables{P,N}
    dimension::Int
    pack::P
    site::Union{Symbol,Nothing}
    names::N
    function LocalVariables(dimension::Int, pack::P, site=nothing, names::N=()) where {P,N}
        dimension > 0 || throw(ArgumentError("dimension must be positive"))
        new{P,N}(dimension, pack, site, names)
    end
end

function LocalVariables(site::Symbol, names::Tuple)
    _validate_names(names)
    return LocalVariables(length(names), η -> NamedTuple{(site,)}((η,)), site, names)
end

function (locals::LocalVariables)(η::AbstractVector)
    isempty(locals.names) && throw(ArgumentError("declare component names for named latent access"))
    length(η) == locals.dimension || throw(DimensionMismatch("latent vector must match component names"))
    return NamedTuple{locals.names}(ntuple(i -> η[i], locals.dimension))
end

"""Constrained VEM noise values passed to the user model."""
vem_noise(dcm, ps) = merge((Ω=ps.omega,), _residual_parameters(dcm.error, ps.error))
_residual_parameters(::Union{AdditiveError,ProportionalError}, ps) =
    (; σ=softplus(only(ps.σ)))
_residual_parameters(::CombinedError, ps) =
    (; σ_additive=softplus(ps.σ[1]), σ_proportional=softplus(ps.σ[2]))
_residual_parameters(::AbstractErrorModel, ps) =
    throw(ArgumentError("provide noise=(dcm, ps) -> constrained_values for this error parameterization"))

struct NamedVEMModel{M,N}
    builder::M
    noise::N
end

(adapter::NamedVEMModel)(dcm, individual, typical, ps) =
    adapter.builder(individual, named_parameters(dcm, typical), adapter.noise(dcm, ps))

"""
    VariationalEM(model_builder; local_variables, noise=vem_noise, path_deriv=true)

`model_builder(individual, theta, noise)` returns an individual probabilistic model.
The model backend is supplied by a package extension. Local variables must have a
centered Gaussian prior with covariance `ps.omega` for the analytic Ω update.
"""
struct VariationalEM{M,L,F,PD<:StaticBool} <: MixedObjective
    individual_model::M
    local_variables::L
    family::F
    path_deriv::PD
end

function VariationalEM(individual_model, local_variables::LocalVariables;
        family=FullRankGaussian(), path_deriv=true)
    family isa FullRankGaussian || error("Only FullRankGaussian is supported")
    return VariationalEM(individual_model, local_variables, family, static(path_deriv))
end

function VariationalEM(individual_model; local_variables::LocalVariables,
        noise=vem_noise, kwargs...)
    return VariationalEM(NamedVEMModel(individual_model, noise), local_variables; kwargs...)
end

_num_random_effects(objective::VariationalEM) = objective.local_variables.dimension

function _latent_values(objective::VariationalEM, η)
    length(η) == _num_random_effects(objective) ||
        throw(DimensionMismatch("latent sample does not match LocalVariables"))
    return objective.local_variables.pack(η)
end

_individual_model(objective::VariationalEM, dcm, individual, typical, ps) =
    objective.individual_model(dcm, individual, typical, ps)

_unsupported_model(model) = throw(ArgumentError(
    "no probabilistic-model extension is loaded for $(typeof(model)); load DynamicPPL"))
_validate_individual_model(model, locals, values) = _unsupported_model(model)
_individual_logjoint(model, values) = _unsupported_model(model)
_individual_loglikelihood(model, values) = _unsupported_model(model)

_validate_vem_setup(::MixedObjective, dcm, population, ps, st) = nothing

function _validate_vem_setup(objective::VariationalEM, dcm, population, ps, st)
    typical, _ = predict_typ_parameters(dcm, population, ps, st)
    locals = objective.local_variables
    values = _latent_values(objective, zeros(eltype(ps.omega), locals.dimension))
    for i in eachindex(population)
        model = _individual_model(objective, dcm, population[i], view(typical, :, i), ps)
        _validate_individual_model(model, locals, values)
    end
    return nothing
end

function _model_logjoint(objective::VariationalEM, dcm, population::Population, ps, st)
    typical, _ = predict_typ_parameters(dcm, population, ps, st)
    etas = sample_gaussian(ps.phi, st.phi)
    individuals = @ignore_derivatives population.data
    return sum(eachindex(individuals)) do i
        model = _individual_model(objective, dcm, individuals[i], view(typical, :, i), ps)
        _individual_logjoint(model, _latent_values(objective, etas[i]))
    end
end

_model_elbo(objective::VariationalEM, dcm, data, ps, st) =
    _model_logjoint(objective, dcm, data, ps, st) - logq(dcm, ps, st)

(objective::VariationalEM)(dcm, data, ps, st) =
    -_model_elbo(objective, dcm, data, ps, st)
