module DCMNaturalOptimisersExt

import DeepCompartmentModels as DCM
import Distributions: FullNormal
import ForwardDiff
import LinearAlgebra: LowerTriangular, Symmetric, cholesky, inv, isposdef
import NaturalOptimisers
import Optimisers

const FullNaturalDescent = NaturalOptimisers.NaturalDescent{FullNormal}
const SqrtManifold = Union{NaturalOptimisers.LieGroupManifold,
    NaturalOptimisers.EuclidianManifold}

function DCM._validate_local_vi(rule::FullNaturalDescent, ::DCM.FullRankGaussian, objective)
    rule.tau == 1 || throw(ArgumentError(
        "NaturalDescent requires tau=1 for this VI backend"))
    return nothing
end

function DCM._validate_local_vi(
        ::NaturalOptimisers.NaturalDescent{Q}, family::DCM.AbstractVariationalFamily,
        objective) where {Q}
    throw(ArgumentError(_unsupported_rule(Q, family)))
end

function DCM._setup_local_vi(rule::FullNaturalDescent, ::DCM.FullRankGaussian,
        objective, population, ps)
    site = objective.local_variables.site
    covariances = DCM._get_cov_matrix(ps.phi)
    items = [_initialise_state(rule, site, ps.phi.μ[i], covariances[i])
        for i in eachindex(population)]
    return (; parameters=[item.parameters for item in items],
        states=[item.state for item in items],
        individual_ids=[individual.id for individual in population])
end

function DCM._run_local_vi(rng, rule::FullNaturalDescent, ::DCM.FullRankGaussian,
        opt_state, objective, dcm, population, ps, st; epochs, samples)
    site = objective.local_variables.site
    _validate_state(opt_state, population, site, rule, eltype(first(ps.phi.μ)))
    typical, _ = DCM.predict_typ_parameters(dcm, population, ps, st)
    models = [DCM._individual_model(
        objective, dcm, population[i], view(typical, :, i), ps)
        for i in eachindex(population)]

    for _ in 1:epochs, i in eachindex(population)
        draws = NaturalOptimisers.sample(
            rng, opt_state.parameters[i], opt_state.states[i]; num_samples=samples)
        draws = samples == 1 ? [draws] : draws
        gradients = map(draws) do draw
            eta = draw[site]
            gradient = ForwardDiff.gradient(eta) do value
                -DCM._individual_logjoint(models[i], DCM._latent_values(objective, value))
            end
            NamedTuple{(site,)}((_narrow(gradient, eta),))
        end
        opt_state.states[i], opt_state.parameters[i] = Optimisers.update(
            opt_state.states[i], opt_state.parameters[i], gradients)
    end

    moments = [_extract_moments(state, site) for state in opt_state.states]
    phi = (; μ=[moment.μ for moment in moments],
        L=[moment.L for moment in moments])
    return opt_state, merge(ps, (; phi))
end

function DCM._run_local_vi(rng,
        ::NaturalOptimisers.NaturalDescent{Q}, family::DCM.AbstractVariationalFamily,
        opt_state, objective, dcm, population, ps, st; kwargs...) where {Q}
    throw(ArgumentError(_unsupported_rule(Q, family)))
end

function _validate_state(state, population, site, rule, ::Type{T}) where {T}
    hasproperty(state, :individual_ids) ||
        throw(ArgumentError("NaturalDescent state has no individual-ID mapping"))
    state.individual_ids == [individual.id for individual in population] ||
        throw(ArgumentError("NaturalDescent state does not match the supplied population order"))
    length(state.states) == length(population) == length(state.parameters) ||
        throw(DimensionMismatch("NaturalDescent state must contain one entry per individual"))
    expected_rule = _match_eltype(rule, T)
    all(item -> item[site].rule == expected_rule, state.states) ||
        throw(ArgumentError("NaturalDescent state was created with a different rule"))
    return nothing
end

_unsupported_rule(Q, family) =
    "NaturalDescent requires meanfield=false with $(typeof(family)); got $Q"

function _initialise_state(rule, site, μ, covariance)
    isposdef(covariance) || throw(ArgumentError("initial covariance must be positive definite"))
    parameters = NamedTuple{(site,)}((zeros(eltype(μ), length(μ)),))
    matched_rule = _match_eltype(rule, eltype(μ))
    state = Optimisers.setup(matched_rule, parameters)
    # NaturalOptimisers 0.2 has no public API for replacing or reading q moments.
    leaf = state[site]
    leaf.state = merge(leaf.state, (; q=_initial_q(matched_rule.manifold, μ, covariance)))
    return (; parameters, state)
end

function _extract_moments(state, site)
    leaf = state[site]
    μ, scale = leaf.state.q
    covariance = _covariance(leaf.rule.manifold, scale)
    isposdef(covariance) || throw(ArgumentError(
        "NaturalDescent covariance is not positive definite"))
    return (; μ=copy(μ),
        L=LowerTriangular(Matrix(cholesky(covariance).L)))
end

_initial_q(::NaturalOptimisers.RiemannianManifold, μ, covariance) =
    (copy(μ), Symmetric(inv(Matrix(covariance))))
_initial_q(::SqrtManifold, μ, covariance) =
    (copy(μ), LowerTriangular(Matrix(cholesky(covariance).L)))

_covariance(::NaturalOptimisers.RiemannianManifold, precision) =
    Symmetric(inv(Matrix(precision)))
_covariance(::SqrtManifold, L) = Symmetric(L * L')

function _match_eltype(rule::FullNaturalDescent, ::Type{T}) where T
    rule.eta isa T && return rule
    return NaturalOptimisers.NaturalDescent(T(rule.eta), T.(rule.beta);
        tau=T(rule.tau), scale=T(rule.init_scale), meanfield=false, manifold=rule.manifold)
end

_narrow(gradient::AbstractVector{T}, ::AbstractVector{T}) where T = gradient
_narrow(gradient::AbstractVector, eta::AbstractVector{T}) where T = convert(Vector{T}, gradient)

end
