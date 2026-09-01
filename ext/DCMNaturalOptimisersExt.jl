module DCMNaturalOptimisersExt

import DeepCompartmentModels as DCM
import ForwardDiff
import LinearAlgebra: LowerTriangular, Symmetric, cholesky, inv, isposdef
import NaturalOptimisers
import Optimisers

const FullRiemannian = NaturalOptimisers.NaturalDescent{
    DCM.FullNormal, NaturalOptimisers.RiemannianManifold}

function DCM._setup_local_vi(rule::FullRiemannian, ::DCM.FullRankGaussian,
        objective, population, ps)
    rule.tau == 1 || throw(ArgumentError("NaturalDescent requires tau=1 for VI"))
    site = objective.local_variables.site
    isnothing(site) && throw(ArgumentError("NaturalDescent requires a named local site"))
    covariances = DCM._get_cov_matrix(ps.phi)
    items = [_initialise_state(rule, site, ps.phi.μ[i], covariances[i])
        for i in eachindex(population)]
    return (; parameters=[item.parameters for item in items],
        states=[item.state for item in items])
end

function DCM._setup_local_vi(rule::NaturalOptimisers.NaturalDescent,
        family::DCM.AbstractVariationalFamily, objective, population, ps)
    throw(ArgumentError(
        "only full-covariance Riemannian NaturalDescent supports $(typeof(family))"))
end

function DCM._run_local_vi(rng, ::FullRiemannian, ::DCM.FullRankGaussian,
        opt_state, objective, dcm, population, ps, st; epochs, samples)
    site = objective.local_variables.site
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
                -DCM._individual_logjoint(
                    models[i], DCM._latent_values(objective, value))
            end
            NamedTuple{(site,)}((gradient,))
        end
        opt_state.states[i], opt_state.parameters[i] = Optimisers.update(
            opt_state.states[i], opt_state.parameters[i], gradients)
    end

    moments = [_extract_moments(state, site) for state in opt_state.states]
    phi = (; μ=[moment.μ for moment in moments],
        L=[moment.L for moment in moments])
    return opt_state, merge(ps, (; phi))
end

# NaturalOptimisers 0.2 stores a Riemannian Gaussian as (mean, precision).
function _initialise_state(rule, site, μ, covariance)
    isposdef(covariance) || throw(ArgumentError("initial covariance must be positive definite"))
    parameters = NamedTuple{(site,)}((zeros(eltype(μ), length(μ)),))
    state = Optimisers.setup(rule, parameters)
    leaf = state[site]
    leaf.state = merge(leaf.state,
        (; q=(copy(μ), Symmetric(inv(Matrix(covariance))))))
    return (; parameters, state)
end

function _extract_moments(state, site)
    μ, precision = state[site].state.q
    isposdef(precision) || throw(ArgumentError("NaturalDescent precision is not positive definite"))
    covariance = Symmetric(inv(Matrix(precision)))
    return (; μ=copy(μ),
        L=LowerTriangular(Matrix(cholesky(covariance).L)))
end

end
