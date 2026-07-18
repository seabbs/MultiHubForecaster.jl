"""
MultiHubForecaster provides reusable tooling for forecasting into hubverse
forecast hubs, trained jointly across their data.

It bundles hubverse submission I/O (`model-output`/`model-metadata` writers),
forecast scoring (the weighted interval score with its decomposition, central
interval coverage, and the multivariate energy score), a hub-registry loader,
an R-backed hub-validation wrapper, and an abstract forecasting-model
interface. This package ships the tooling and the model interface only; no
concrete forecasting model lives here.
"""
module MultiHubForecaster

# All genuine `using`/`import` statements live here (import centralisation);
# component files use these qualified. DocStringExtensions supplies the
# docstring templates registered in `docstrings.jl`.
using DocStringExtensions: @template, DOCSTRING, EXPORTS, IMPORTS,
                           TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES

import CSV
import DataFrames
import LinearAlgebra
import Parquet2
import TOML
import YAML

include("docstrings.jl")

# Scoring rules.
include("scoring.jl")
export weighted_interval_score, interval_coverage, energy_score

# Hubverse submission I/O.
include("hubio.jl")
export HUBVERSE_COLUMN_ORDER, order_hub_columns, write_submission,
       write_model_metadata

# Hub registry loader.
include("registry.jl")
export HubConfig, load_registry

# Hub-validation wrapper (shells out to R `hubValidations`).
include("validation.jl")
export HubValidationResult, validate_submission

# Forecasting-model interface (contract only, no concrete model).
include("model_interface.jl")
export AbstractForecastModel, fit, forecast

end # module
