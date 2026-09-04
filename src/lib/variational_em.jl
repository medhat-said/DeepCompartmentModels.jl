abstract type AbstractVariationalFamily end

"""Full-rank Gaussian approximation for individual latent variables."""
struct FullRankGaussian <: AbstractVariationalFamily end

"""
    LocalVariables(site::Symbol, names::Tuple)
    LocalVariables(site::Symbol, dimension::Int)
    LocalVariables(dimension::Int, pack, site=nothing, names=())

Declare the vector-valued local site of an individual model, i.e. the random effects
`η` estimated per individual. `setup` sizes `Ω` and the variational posteriors from
this declaration.

Pass `names` to give the components names, enabling named access `locals(η).CL`. Pass
an integer `dimension` instead when the components are anonymous, which is convenient
for a large number of random effects.

# Arguments
- `site::Symbol`: Name the model samples the latent vector under, e.g. `:η`.
- `names::Tuple`: Component names in latent order. One name per dimension.
- `dimension::Int`: Number of latent dimensions, when the components are unnamed.
- `pack`: Function mapping a latent vector to the values passed to the model backend.

# Examples
```julia
LocalVariables(:η, (:CL, :V))  # two named random effects
LocalVariables(:η, 12)         # twelve anonymous random effects
```
"""
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
    return LocalVariables(length(names), _packer(site), site, names)
end

LocalVariables(site::Symbol, dimension::Int) =
    LocalVariables(dimension, _packer(site), site, ())

_packer(site::Symbol) = η -> NamedTuple{(site,)}((η,))

function (locals::LocalVariables)(η::AbstractVector)
    isempty(locals.names) && throw(ArgumentError("declare component names for named latent access"))
    length(η) == locals.dimension || throw(DimensionMismatch("latent vector must match component names"))
    return NamedTuple{locals.names}(ntuple(i -> η[i], locals.dimension))
end

"""
    vem_noise(dcm, ps)

Return the random-effect covariance `Ω` and constrained residual-error parameters
passed to the individual model.

# Arguments
- `dcm`: Model whose error model determines the residual parameters.
- `ps`: Parameters holding `omega` and `error`.
"""
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
The model backend is supplied by a package extension. The current analytic Ω update
requires the local site to have a centered Gaussian prior with covariance `noise.Ω`.

# Arguments
- `model_builder`: Builder called as `model_builder(individual, theta, noise)`.

# Keyword arguments
- `local_variables::LocalVariables`: Declaration of the local site and its dimension.
- `noise`: Function `(dcm, ps) -> constrained_values` reporting `Ω` and the residual error parameters. Default = [`vem_noise`](@ref).
- `family`: Variational family for `q(η)`. Default = `FullRankGaussian()`.
- `path_deriv`: Whether to use the path derivative estimator. Default = `true`.
"""
struct VariationalEM{M,L,F,PD<:StaticBool} <: MixedObjective
    individual_model::M
    local_variables::L
    family::F
    path_deriv::PD
end

function VariationalEM(individual_model, local_variables::LocalVariables;
        family=FullRankGaussian(), path_deriv::Bool=true)
    family isa FullRankGaussian ||
        throw(ArgumentError("only FullRankGaussian is currently supported"))
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

##### Probabilistic-model backend contract
#
# A backend extension (see ext/DCMDynamicPPLExt.jl) makes a probabilistic modelling
# package usable with `VariationalEM` by adding methods to the three functions below,
# dispatching on its own model type. Core never sees the backend's types: it only ever
# passes a model object it received from the user's builder back to these functions.
#
#   _validate_individual_model(model, locals, values) -> nothing
#       Throw if `model` does not sample exactly the local site declared by `locals`.
#       Called once per individual by `setup`.
#   _individual_logjoint(model, values) -> Real
#       Joint log density (log prior of the local site + log likelihood) at `values`.
#   _individual_loglikelihood(model, values) -> Real
#       Log likelihood of the observations only, used by the M-step.
#
# `values` is always the latent vector packed by `locals.pack`. Both densities must be
# differentiable by the AD backend used to fit the model.

_unsupported_model(model) = throw(ArgumentError(
    "no probabilistic-model backend is loaded for $(typeof(model))"))
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
