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

import ADTypes
import Arrow
import CSV
import DataFrames
import Dates
import Distributions
import DynamicPPL
import FlexiChains
import LinearAlgebra
import LogExpFunctions
import Parquet2
import Random
import Statistics
import TOML
import Turing
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

# Model components as structs with dispatch (abstract types + concrete structs
# with per-component priors, EpiAware-style) and the generic `generate` function.
# All `public` (Julia 1.11 keyword) but not exported; reached as
# `MultiHubForecaster.<name>`.
include("components.jl")
public AbstractLink, LogLink, LogitLink, apply_link, inverse_link
public AbstractLatent, ARGrowthRate
public AbstractSeasonality, FourierSeasonality
public AbstractObservation, NegativeBinomialObs, BetaObs, LogNormalObs
public generate

# The generic `Skeleton` framing and the `Baseline` model built on it. Both are
# `public` (Julia 1.11 keyword) but not exported; reached as
# `MultiHubForecaster.Skeleton` / `.Baseline`.
include("skeleton.jl")
public Skeleton

include("baseline.jl")
public Baseline, BaselineFit

# Backtesting / experiment harness: time-ordered splits, walk-forward folds,
# and the scoring runner. Reuses the compiled Turing model across folds in one
# session (see the efficiency note in `backtest.jl`).
include("backtest.jl")
export DateSplit, date_split, partition, Fold, walk_forward_folds,
       run_backtest, BacktestResult

# Experiment storage: ZSTD-compressed Arrow tables plus a TOML run manifest.
include("storage.jl")
export save_experiment, load_experiment, write_scores, read_scores,
       write_forecasts, read_forecasts, write_manifest, read_manifest

end # module
