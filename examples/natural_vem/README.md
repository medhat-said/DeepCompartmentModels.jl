# DynamicPPL + NaturalOptimisers VEM example

`theophylline.jl` demonstrates a user-written individual
`DynamicPPL.@model` with full-covariance NaturalDescent updates of `q(ηᵢ)`.
DCM retains the updates of typical parameters, Ω, and residual error.
The pinned NaturalOptimisers revision currently requires Julia 1.12.

Open the script in VS Code and run **Julia: Execute Active File**, or run:

```sh
julia --project=examples/natural_vem -e 'using Pkg; Pkg.instantiate()'
julia --project=examples/natural_vem examples/natural_vem/theophylline.jl
```

for the data, set `DCM_DATA_FILE` to use another path.
Edit `ETA`, `INITIAL`, `FIT`, the individual model, and the DCM construction at
the marked `# CHANGE:` locations for another project.

This is experimental: individual posteriors are Gaussian while typical
parameters, Ω, and residual parameters remain point estimates.
