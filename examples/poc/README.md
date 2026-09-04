# Turing model + NaturalOptimisers discussion POC

These two standalone scripts use the same two-ETA PK `DynamicPPL.@model` and
synthetic population. They require the proposed DCM changes on this branch.

```sh
julia --project=examples/poc -e 'using Pkg; Pkg.instantiate()'
julia --project=examples/poc examples/poc/turing_model_vem.jl
julia --project=examples/poc examples/poc/turing_natural_vem.jl
```

For the real Theophylline comparison, open `theophylline_vem_comparison.jl` in
VS Code and run **Julia: Execute Active File**, or use:

```sh
julia --project=examples/poc examples/poc/theophylline_vem_comparison.jl
```

It fits IIV on Ka, CL, and V with both local-q backends, writes a parameter
comparison CSV, plots a per-cycle MC negative ELBO, and overlays their individual
95% curve bands with observations. Edit `FIT_OPTIONS` at the top of the script to
set method-specific optimization effort and learning rates.

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

- These scripts are a two-individual, two-ETA synthetic data example, not a fit
  assessment. The number of ETAs is not a limitation of the API: it follows the
  `LocalVariables` declaration. Tests cover 1, 2, 3, 5, and 12 dimensions; the
  two-dimensional case is compared with an analytic posterior.
- `fit.history` contains a stochastic negative-ELBO trace, but there is no automatic
  convergence rule, multi-seed assessment, or runtime benchmark.
- Typical parameters, Ω, and residual scale are point estimates; this is not fully
  Bayesian population inference and does not propagate their uncertainty.
- The selectable NaturalDescent backend supports full-covariance Gaussian rules on
  its Riemannian, Lie-group, and Euclidean manifolds, with `tau=1`.
- Its extension isolates, but still depends on, NaturalOptimisers 0.2 internals
  for Gaussian initialization and moment extraction.
- Checkpoint serialization, GPU use, missing data, real PK data, NN covariates,
  and larger/stiff ODE systems are not demonstrated.
- The existing DCM residual M-step is reused but its statistical objective has not
  been validated here; successful execution does not validate the full VEM loop.

## Questions before a PR

1. Should this model-builder/`LocalVariables` boundary become the public DCM API?
2. Should NaturalOptimisers expose public Gaussian initialization and moment
   extraction so the extension no longer accesses its state representation?
3. Is `fit_vem(...; local_optimizer=...)` the desired public backend-selection API?

The current contract keeps backend state outside `ps` and exchanges `μ` plus the
covariance Cholesky factor `L` at DCM boundaries.
