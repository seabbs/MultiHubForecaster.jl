# Model interface (contract only — no concrete model lives here). A forecasting
# model plugs into the MultiHubForecaster tooling by subtyping
# `AbstractForecastModel` and adding methods to `fit` and `forecast`. The
# hubverse I/O and scoring tools consume the forecast table `forecast` returns.

"""
`AbstractForecastModel` is the supertype every MultiHubForecaster forecasting
model subtypes.

A concrete model `M <: AbstractForecastModel` implements two methods:

  - [`fit`](@ref)`(model::M, target_data; kwargs...)` — calibrate the model to
    target data as known at a given `as_of` date, returning a fitted object.
  - [`forecast`](@ref)`(fitted, target_spec; kwargs...)` — produce a forecast
    table for the requested target.

The tooling in this package (hubverse I/O, scoring) consumes the forecast
table without knowing the model type, so any model satisfying this contract
integrates with the same submission and scoring pipeline. This package defines
the interface only and ships no concrete model.
"""
abstract type AbstractForecastModel end

"""
Calibrate a forecasting model to target data.

`fit` is part of the [`AbstractForecastModel`](@ref) interface. A concrete
model adds a method

    fit(model::MyModel, target_data; kwargs...)

where `target_data` is a target-data table keyed by an `as_of` date (the data
as it would have been known at that date) for the target the model forecasts.
It returns a fitted object that [`forecast`](@ref) consumes. This package
defines the generic function only; it adds no methods.
"""
function fit end

"""
Produce a forecast table from a fitted model.

`forecast` is part of the [`AbstractForecastModel`](@ref) interface. A concrete
model adds a method

    forecast(fitted, target_spec; kwargs...)

where `fitted` is the object returned by [`fit`](@ref) and `target_spec`
describes the target, horizons, locations and output types requested. It
returns a forecast table (hubverse schema: task-id columns, `output_type`,
`output_type_id`, `value`) of samples or quantiles, ready for the scoring and
hubverse-I/O tools. This package defines the generic function only; it adds no
methods.
"""
function forecast end
