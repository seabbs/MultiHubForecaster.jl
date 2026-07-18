# MultiHubForecaster.jl

Julia forecasting engine for hubverse forecast hubs, trained jointly across
their data.
The package holds the final model and the reusable machinery: data ingest,
models, inference, hubverse-format I/O, and scoring.

Experiments, the local hub, the upstream-hub registry, and research notes live
in the companion repo `MultiHubForecastExperiments`, which links this package
as a submodule.

## Development

- `julia --project=. -e 'using Pkg; Pkg.instantiate()'`
- `julia --project=. -e 'using Pkg; Pkg.test()'`
