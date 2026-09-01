# Turing model + NaturalOptimisers discussion POC

These two standalone scripts use the same two-ETA PK `DynamicPPL.@model` and
synthetic population. They require the proposed DCM changes on this branch.

```sh
julia --project=examples/poc -e 'using Pkg; Pkg.instantiate()'
julia --project=examples/poc examples/poc/turing_model_vem.jl
julia --project=examples/poc examples/poc/turing_natural_vem.jl
```

- `turing_model_vem.jl` calls `fit_vem` with Adam for q.
- `turing_natural_vem.jl` calls the same API with NaturalDescent. DCM retains
  Adam for typical parameters, its residual update, and its analytic Ω update.

The natural POC keeps persistent state per individual and exports moment copies
to `ps.phi`. Population parameters remain point estimates.

## What the POC shows

1. A user-written individual `@model`.
2. DCM supplies each individual, named typical parameters, Ω, and residual scale.
3. `fit_vem(...; local_optimizer=...)` selects the local-q backend.
4. NaturalDescent samples each q, differentiates `-logjoint` at those ETAs,
   and keeps one persistent optimizer state per individual.
5. Only exported q moments enter DCM's typical-parameter, residual, Ω, and
   prediction paths; the two optimizers never both update q.

`tau=1` lets NaturalDescent include the Gaussian entropy term. The model prior is
already in `logjoint`, so neither prior nor entropy is added again. The POC uses a full-covariance Riemannian update without momentum.

## Current limitations

- This is a two-individual, two-ETA synthetic data example, not a fit assessment.
- There is no convergence rule, ELBO history, multi-seed comparison, or runtime
  benchmark in these two shareable scripts.
- Typical parameters, Ω, and residual scale are point estimates; this is not fully
  Bayesian population inference and does not propagate their uncertainty.
- The selectable NaturalDescent backend currently supports only a full-covariance
  Riemannian Gaussian with `tau=1`.
- Its extension isolates, but still depends on, NaturalOptimisers 0.2 internals
  for Gaussian initialization and moment extraction.
- Checkpoint serialization, GPU use, arbitrary latent structures, missing data,
  real PK data, NN covariates, and larger/stiff ODE systems are not demonstrated.
- The existing DCM residual M-step is reused but its statistical objective has not
  been validated here; successful execution does not validate the full VEM loop.

## Questions before a PR

1. Should this model-builder/`LocalVariables` boundary become the public DCM API?
2. Should NaturalOptimisers expose public Gaussian initialization and moment
   extraction so the extension no longer accesses its state representation?
3. Should the backend own individual state while DCM receives only exported
   `(μ, Σ)` moments at M-step and prediction boundaries?
4. Should covariance, precision, or a Cholesky factor be the exchange format?
5. Is `fit_vem(...; local_optimizer=...)` the desired public backend-selection API?
