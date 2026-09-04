"""
    fit_vem(rng, objective, dcm, population, ps, st; local_optimizer, global_optimizer, ...)

Alternate local Gaussian-q updates, `ps.theta` updates, and the existing M-step.
`local_optimizer` selects the q backend; its state is returned for resuming.
`samples` applies to NaturalDescent and the M-step. The standard DCM q gradient uses
one reparameterized sample per local epoch.
"""
function fit_vem(rng::Random.AbstractRNG, objective::VariationalEM,
        dcm::DeepCompartmentModel, population::Population, ps, st;
        local_optimizer=Optimisers.Adam(1e-2),
        global_optimizer=Optimisers.Adam(1e-3),
        local_opt_state=nothing, global_opt_state=nothing,
        cycles::Int=3, local_epochs::Int=25, global_epochs::Int=25,
        m_epochs::Int=5, samples::Int=20, verbose::Bool=true)
    cycles >= 0 || throw(ArgumentError("cycles must be nonnegative"))
    local_epochs >= 0 || throw(ArgumentError("local_epochs must be nonnegative"))
    global_epochs >= 0 || throw(ArgumentError("global_epochs must be nonnegative"))
    m_epochs >= 0 || throw(ArgumentError("m_epochs must be nonnegative"))
    samples > 0 || throw(ArgumentError("samples must be positive"))

    _validate_local_vi(local_optimizer, objective.family, objective)
    local_state = isnothing(local_opt_state) ?
        _setup_local_vi(local_optimizer, objective.family, objective, population, ps) :
        local_opt_state
    global_state = if isnothing(global_opt_state) && global_epochs > 0
        Optimisers.setup(global_optimizer, ps.theta)
    else
        global_opt_state
    end
    history = NamedTuple[]

    for cycle in 1:cycles
        local_state, ps = _run_local_vi(rng, local_optimizer, objective.family,
            local_state, objective, dcm, population, ps, st;
            epochs=local_epochs, samples)

        for _ in 1:global_epochs
            update_epsilon!(rng, st)
            grad = gradient(objective, dcm, population, ps, st)
            global_state, theta = Optimisers.update(global_state, ps.theta, grad.theta)
            ps = merge(ps, (; theta))
        end

        if m_epochs > 0
            ps = m_step(objective, rng, dcm, population, ps, st;
                epochs=m_epochs, num_samples=samples, verbose=false)
        end
        update_epsilon!(rng, st)
        negative_elbo = objective(dcm, population, ps, st)
        push!(history, (; cycle, negative_elbo))
        verbose && println("cycle $cycle: negative ELBO = $negative_elbo")
    end

    settings = (; cycles, local_epochs, global_epochs, m_epochs, samples,
        local_optimizer, global_optimizer)
    return (; ps, st, local_opt_state=local_state,
        global_opt_state=global_state,
        individual_ids=[individual.id for individual in population],
        settings, history, completed_cycles=cycles)
end

# Local VI backend contract. Extensions dispatch on their rule and family; core treats
# the returned optimiser state as opaque.
#
#   _setup_local_vi(rule, family, objective, population, ps) -> opt_state
#   _run_local_vi(rng, rule, family, opt_state, objective, dcm, population, ps, st;
#                 epochs, samples) -> (opt_state, ps)
#
# The exchange format is `ps.phi = (μ, L)`, with `Σ = L * L'`.

_validate_local_vi(::Optimisers.AbstractRule, ::AbstractVariationalFamily, objective) = nothing

_setup_local_vi(optimizer::Optimisers.AbstractRule, ::AbstractVariationalFamily,
        objective, population, ps) = Optimisers.setup(optimizer, ps.phi)

function _run_local_vi(rng, optimizer::Optimisers.AbstractRule,
        ::AbstractVariationalFamily, opt_state, objective, dcm, population, ps, st;
        epochs, samples)
    for _ in 1:epochs
        update_epsilon!(rng, st)
        grad = gradient(objective, dcm, population, ps, st)
        opt_state, phi = Optimisers.update(opt_state, ps.phi, grad.phi)
        ps = merge(ps, (; phi))
    end
    return opt_state, ps
end

function m_step(obj::VariationalELBO, rng::Random.AbstractRNG, dcm::DeepCompartmentModel{P,M}, population::Population, ps, st; kwargs...) where {P<:SciMLBase.AbstractDEProblem,M<:Lux.AbstractLuxLayer}
    @info "Optimising residual error parameters"
    ps = optimise_residual_error(obj, rng, dcm, population, ps, st; kwargs...)
    @info "Optimising omega based on Variational posteriors"
    omega_opt = optimise_omega(ps)
    
    return Accessors.@set ps.omega = omega_opt
end

function m_step(obj::VariationalEM, rng::Random.AbstractRNG,
        dcm::DeepCompartmentModel, population::Population, ps, st; kwargs...)
    @info "Optimising residual error parameters"
    ps = optimise_residual_error(obj, rng, dcm, population, ps, st; kwargs...)
    @info "Optimising omega based on Variational posteriors"
    return Accessors.@set ps.omega = optimise_omega(ps)
end

function optimise_residual_error(obj::Union{<:LogLikelihood,<:MixedObjective}, rng, dcm, data, ps, st; opt=Optimisers.Adam(1e-2), epochs=100, verbose::Bool = true, kwargs...)
    opt_state = Optimisers.setup(opt, ps)
    for epoch in 1:epochs
        loss, grad = residual_error_value_and_gradient(rng, obj, dcm, data, ps, st; kwargs...)
        if verbose
            println("Epoch $epoch, NLL = $(loss)")
        end
        opt_state, ps = Optimisers.update(opt_state, ps, grad)
    end

    return ps
end

function optimise_omega(ps::NamedTuple{(:theta,:error,:omega,:phi)})
    μμᵀ = map(ps.phi.μ) do μ
        μ * μ'
    end
    return mean(μμᵀ + _get_cov_matrix(ps.phi))
end

# These all assume that the variance parameters are vectors of parameters
_get_cov_matrix(ps::NamedTuple{(:μ,:Σ)}) = ps.Σ
_get_cov_matrix(ps::NamedTuple{(:μ,:L)}) = map(ps.L) do L
    Symmetric(L * L')
end
_get_cov_matrix(ps::NamedTuple{(:μ,:σ)}) = map(ps.σ) do σ
    collect(Diagonal(softplus.(σ).^2))
end
_get_cov_matrix(ps::NamedTuple{(:μ,:σ²)}) = map(ps.σ²) do σ²
    collect(Diagonal(σ²))
end
