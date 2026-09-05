# DynamicPPL + NaturalOptimisers VEM example

`theophylline_natural_vem_minimal.jl` demonstrates a user-written individual
`DynamicPPL.@model` with full-covariance NaturalDescent updates of `q(ηᵢ)`.
DCM retains the updates of typical parameters, Ω, and residual error.

Open the script in VS Code and run **Julia: Execute Active File**, or run:

```sh
julia --project=examples/poc -e 'using Pkg; Pkg.instantiate()'
julia --project=examples/poc examples/poc/theophylline_natural_vem_minimal.jl
```

The default data path is `.scratch/theophylline_nmready.csv`; override it with
`DCM_DATA_FILE`. Edit `ETA`, `INITIAL`, `FIT`, the individual model, and the DCM
construction at the marked `# CHANGE:` locations for another project.

This is experimental: individual posteriors are Gaussian while typical
parameters, Ω, and residual parameters remain point estimates.
