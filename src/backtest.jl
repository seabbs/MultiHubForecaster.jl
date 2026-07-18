# Backtesting / experiment harness. A reusable, efficient walk-forward
# cross-validation runner over any `AbstractForecastModel`. It turns a dated
# target-data table into a time-ordered train/validation/test split, generates
# expanding-window walk-forward folds over the training span, refits and
# forecasts the model at each fold origin, and scores the forecasts into a tidy
# long table. Storage of the results lives in `storage.jl`.
#
# All external symbols are used qualified (`DataFrames.`, `Dates.`,
# `Statistics.`, `Random.`); the `import`s live in the main module file.
#
# --- Efficiency: compile the model once, reuse it across folds ----------------
#
# The precompile/latency lesson (steer-log "Backtesting, scoring, selection"):
# a Turing `@model` macro-expands *once*, at package load. Each fold then only
# constructs a fresh `DynamicPPL.Model` *instance* by calling the model builder
# with different data — it does NOT re-expand or recompile the model, provided
# the instance's *type* is identical across folds. `run_backtest` guarantees
# this by:
#
#   - running every fold in a single Julia process, in one loop, reusing the
#     same `model` object (no per-fold `eval`, no respawning Julia, no
#     regenerating the model type);
#   - standardising the fit table to stable column element types every fold —
#     the value column is always `Float64` and the time column always `Int`, so
#     the observation submodel's captured-data type (and hence the whole
#     `Skeleton` model type and the specialised NUTS log-density) never changes,
#     and Turing/DynamicPPL specialises the sampler once and reuses it;
#   - keeping the loop free of closures whose captured types vary per fold.
#
# Only the data *values* and series lengths differ between folds; their *types*
# do not, so the compiled sampling code is shared. Folds are independent and
# may be threaded (`fold_parallel`); chains are threaded inside `fit`
# (`MCMCThreads`). Memory stays low: each fold's fitted object (and its
# posterior draws) goes out of scope at the end of the iteration — only the
# tidy scores and the small forecast-quantile table are retained, never the
# full chains.

# --- three-way, time-ordered split --------------------------------------------

@doc raw"""
$(TYPEDEF)

A three-way, time-ordered split of a dated data table, applied globally by
date. Relative to a `present` date, the data partitions into

  - `train`: everything up to and including `train_end` (`present` minus two
    years);
  - `validation`: the second-most-recent year, `train_end < date <= val_end`
    (`val_end` is `present` minus one year);
  - `test`: the most recent year, `val_end < date <= present`.

Construct one with [`date_split`](@ref) and slice a table with
[`partition`](@ref). The boundaries are stored as dates so a fold generator and
the tests can check them directly.

---
## Fields
$(TYPEDFIELDS)
"""
struct DateSplit
    "The present date the split is taken relative to (the test upper bound)."
    present::Dates.Date
    "Inclusive upper bound of the training span (`present` minus two years)."
    train_end::Dates.Date
    "Inclusive upper bound of the validation span (`present` minus one year)."
    val_end::Dates.Date
end

@doc """
$(TYPEDSIGNATURES)

`date_split` builds a [`DateSplit`](@ref) from `data`.

`present` defaults to the maximum value of the `date_col` column. The training
span ends two years before `present`, the validation span covers the following
year, and the test span the most recent year up to `present`.

# Arguments
  - `data`: any `Tables.jl`/`DataFrames` source with a date column.

# Keyword arguments
  - `present`: the present date; defaults to `maximum(data[date_col])`.
  - `date_col`: the date column name (default `:date`).
"""
function date_split(data; present::Union{Nothing, Dates.Date} = nothing,
        date_col::Symbol = :date)
    df = DataFrames.DataFrame(data)
    p = present === nothing ? maximum(df[!, date_col]) : present
    return DateSplit(p, p - Dates.Year(2), p - Dates.Year(1))
end

# Row mask for one span of a `DateSplit`.
function _span_mask(split::DateSplit, dates, span::Symbol)
    if span === :train
        return dates .<= split.train_end
    elseif span === :validation
        return (dates .> split.train_end) .& (dates .<= split.val_end)
    elseif span === :test
        return (dates .> split.val_end) .& (dates .<= split.present)
    else
        throw(ArgumentError("span must be :train, :validation or :test"))
    end
end

@doc """
$(TYPEDSIGNATURES)

Partition `data` by a [`DateSplit`](@ref) into its three time-ordered spans.

Returns a `NamedTuple` `(; train, validation, test)` of `DataFrame`s, each the
rows of `data` whose `date_col` falls in the corresponding span. The spans are
disjoint and, together, cover every row up to `split.present`; rows after
`present` are dropped.

# Arguments
  - `split`: the [`DateSplit`](@ref) to apply.
  - `data`: the table to partition.

# Keyword arguments
  - `date_col`: the date column name (default `:date`).
"""
function partition(split::DateSplit, data; date_col::Symbol = :date)
    df = DataFrames.DataFrame(data)
    dates = df[!, date_col]
    return (;
        train = df[_span_mask(split, dates, :train), :],
        validation = df[_span_mask(split, dates, :validation), :],
        test = df[_span_mask(split, dates, :test), :])
end

# --- walk-forward folds -------------------------------------------------------

@doc raw"""
$(TYPEDEF)

One walk-forward cross-validation fold: a forecast origin date and the rows of
the source table available for training at that origin.

`train_rows` holds integer row indices into the table passed to
[`walk_forward_folds`](@ref). `origin_date` is the fold's last observed week
(the `reference_date`); `train_rows` contains only rows at or before it (and,
when a vintage column is used, only rows whose vintage is at or before it), and
forecasts run from horizon 1 = one step after `origin_date` — so a fold never
sees data after its origin. No leakage.

---
## Fields
$(TYPEDFIELDS)
"""
struct Fold
    "The forecast origin date for this fold."
    origin_date::Dates.Date
    "Row indices (into the source table) available for training at the origin."
    train_rows::Vector{Int}
end

# Row indices available at `origin` (the last observed week): dated at or before
# the origin, and (when `as_of_col` is given) with a vintage at or before it.
# The origin week IS the reference date, so its data is used and forecasts run
# from one step after it — see `Fold`.
function _available_rows(df, origin, date_col, as_of_col)
    dates = df[!, date_col]
    if as_of_col === nothing
        return findall(<=(origin), dates)
    end
    asof = df[!, as_of_col]
    return findall(i -> dates[i] <= origin && asof[i] <= origin,
        eachindex(dates))
end

@doc """
$(TYPEDSIGNATURES)

`walk_forward_folds` generates expanding-window walk-forward folds over the
training span of `split`.

Forecast origins step through the training span by `step` (default weekly),
starting `min_train` after the earliest training date and ending early enough
that `horizon` steps of forecasts still land inside the training span (so
validation and test data stay untouched during cross-validation). Each fold is
an expanding window: all data strictly before the origin is training data (see
[`Fold`](@ref) for the no-leakage guarantee). Folds with no available training
data are skipped.

# Arguments
  - `data`: the dated target-data table (same table later passed to
    [`run_backtest`](@ref)).
  - `split`: the [`DateSplit`](@ref) whose training span is walked.

# Keyword arguments
  - `step`: spacing between origins (a `Dates.Period`, default `Dates.Week(1)`).
  - `min_train`: minimum span before the first origin (a `Dates.Period`,
    default `Dates.Week(52)`).
  - `horizon`: number of forecast steps used later; caps the last origin so
    forecasts stay within the training span (default `4`).
  - `date_col`: the date column name (default `:date`).
  - `as_of_col`: an optional vintage column name; when given, only rows with a
    vintage at or before an origin are available at that origin.
"""
function walk_forward_folds(data, split::DateSplit;
        step::Dates.Period = Dates.Week(1),
        min_train::Dates.Period = Dates.Week(52),
        horizon::Int = 4, date_col::Symbol = :date,
        as_of_col::Union{Nothing, Symbol} = nothing)
    df = DataFrames.DataFrame(data)
    dates = df[!, date_col]
    train_dates = dates[dates .<= split.train_end]
    isempty(train_dates) &&
        throw(ArgumentError("no training data at or before train_end"))
    first_origin = minimum(train_dates) + min_train
    last_origin = split.train_end - horizon * step
    folds = Fold[]
    first_origin > last_origin && return folds
    for origin in first_origin:step:last_origin
        rows = _available_rows(df, origin, date_col, as_of_col)
        isempty(rows) && continue
        push!(folds, Fold(origin, rows))
    end
    return folds
end

# --- fit-table construction (dates -> weekly integer time index) --------------

# The weekly integer time index of `date` relative to `epoch` (whole weeks).
_week_index(date, epoch) = round(Int, Dates.value(date - epoch) / 7)

# Build the `(location, time, value)` fit table for one training slice. Dates
# map to a weekly integer index from a fixed `epoch` so seasonality aligns
# across folds and locations. When `as_of_col` is given, keep only the latest
# vintage per (location, date). The value column is always `Float64` so the
# fitted model type is identical across folds (see the efficiency note above).
function _fit_table(sub, epoch, date_col, location_col, value_col, as_of_col)
    d = DataFrames.DataFrame(sub)
    if as_of_col !== nothing
        g = DataFrames.groupby(d, [location_col, date_col])
        d = DataFrames.combine(g) do s
            s[argmax(s[!, as_of_col]), :]
        end
    end
    return DataFrames.DataFrame(
        location = d[!, location_col],
        time = _week_index.(d[!, date_col], epoch),
        value = Float64.(d[!, value_col]))
end

# Final-vintage truth: value scored against, keyed by (location, date). With a
# vintage column, take the latest vintage per (location, date); otherwise the
# value as given.
function _truth_lookup(df, date_col, location_col, value_col, as_of_col)
    d = DataFrames.DataFrame(df)
    if as_of_col !== nothing
        g = DataFrames.groupby(d, [location_col, date_col])
        d = DataFrames.combine(g) do s
            s[argmax(s[!, as_of_col]), :]
        end
    end
    truth = Dict{Tuple{Any, Dates.Date}, Float64}()
    for r in DataFrames.eachrow(d)
        truth[(r[location_col], r[date_col])] = Float64(r[value_col])
    end
    return truth
end

# --- scoring one forecast cell to tidy metric rows ----------------------------

# Score one predictive sample vector `samples` against the realised value `y`.
# Returns a vector of `(metric::Symbol, value::Float64)` pairs: the energy
# score (sample CRPS), the WIS and its three-way decomposition, the sample
# bias, and central-interval coverage indicators at each `coverage_levels`.
# `bias = 1 - 2 F̂(y)` with the mid-point empirical CDF, in `[-1, 1]`: positive
# when the forecast sits below the observation.
function _score_cell(samples::AbstractVector{<:Real}, y::Real,
        quantile_levels, coverage_levels)
    qs = Statistics.quantile(samples, quantile_levels)
    wis = weighted_interval_score(y, qs, quantile_levels)
    es = energy_score(samples, y)
    below = Statistics.mean(<(y), samples)
    at = Statistics.mean(==(y), samples)
    bias = 1 - 2 * (below + 0.5 * at)
    rows = Tuple{Symbol, Float64}[
        (:energy_score, es), (:wis, wis.wis),
        (:wis_dispersion, wis.dispersion),
        (:wis_overprediction, wis.overprediction),
        (:wis_underprediction, wis.underprediction), (:bias, bias)]
    for c in coverage_levels
        lo = Statistics.quantile(samples, (1 - c) / 2)
        hi = Statistics.quantile(samples, (1 + c) / 2)
        cov = interval_coverage(y, lo, hi) ? 1.0 : 0.0
        push!(rows, (Symbol("coverage_", round(Int, 100c)), cov))
    end
    return rows
end

# Empty tidy scores table with the canonical column schema.
function _empty_scores()
    return DataFrames.DataFrame(
        model = String[], origin_date = Dates.Date[], location = Any[],
        target = String[], horizon = Int[], metric = String[],
        value = Float64[])
end

# Score all sample forecasts in `fc` for one fold. Returns `(scores,
# forecasts)`: the tidy long scores for every (location, horizon) cell with a
# known truth, and the forecast-quantile rows tagged with the model id (the
# small artefact kept for storage; full chains are never retained).
function _score_fold(fc, origin, truth, model_id, target,
        quantile_levels, coverage_levels)
    scores = _empty_scores()
    samp = fc[fc.output_type .== "sample", :]
    for g in DataFrames.groupby(samp, [:location, :horizon])
        loc = g[1, :location]
        h = g[1, :horizon]
        ted = g[1, :target_end_date]
        haskey(truth, (loc, ted)) || continue
        y = truth[(loc, ted)]
        s = Float64.(g.value)
        for (metric, value) in _score_cell(
            s, y, quantile_levels, coverage_levels)
            push!(scores, (model_id, origin, loc, target,
                h, String(metric), value))
        end
    end
    quant = fc[fc.output_type .== "quantile", :]
    forecasts = DataFrames.insertcols(quant, 1, :model => model_id)
    return scores, forecasts
end

# --- the backtest result and runner -------------------------------------------

@doc raw"""
$(TYPEDEF)

The result of a walk-forward backtest run.

`model_id` is the run's model identifier; `scores` is the tidy long table with
columns `(model, origin_date, location, target, horizon, metric, value)`;
`forecasts` holds the forecast quantiles tagged with the model id; `summary`
collects run-level metadata and headline mean metrics. Persist and reload the whole thing with the helpers in
`storage.jl` ([`save_experiment`](@ref) / [`load_experiment`](@ref)).

---
## Fields
$(TYPEDFIELDS)
"""
struct BacktestResult
    "Model identifier the run was tagged with."
    model_id::String
    "Tidy long scores `(model, origin_date, location, target, horizon,
     metric, value)`."
    scores::DataFrames.DataFrame
    "Forecast quantiles tagged with the model id."
    forecasts::DataFrames.DataFrame
    "Run metadata and headline mean metrics."
    summary::Dict{String, Any}
end

# Mean value of one metric across the tidy scores (NaN if the metric is absent).
function _mean_metric(scores, metric)
    rows = scores[scores.metric .== metric, :]
    return DataFrames.nrow(rows) == 0 ? NaN : Statistics.mean(rows.value)
end

@doc """
$(TYPEDSIGNATURES)

`run_backtest` runs a walk-forward backtest of `model` over `folds` and scores
the forecasts.

For each fold (in one Julia session, reusing the compiled model — see the
efficiency note at the top of this file) the training slice is turned into a
`(location, time, value)` table, `model` is refit on it, forecasts are drawn
for `horizon` steps from the fold origin, and each (location, horizon) cell is
scored against the final-vintage truth. Scores accumulate into a tidy long
table; posterior draws are discarded per fold to keep memory low. Returns a
[`BacktestResult`](@ref).

# Arguments
  - `model`: the [`AbstractForecastModel`](@ref) to backtest.
  - `data`: the dated target-data table (the one passed to
    [`walk_forward_folds`](@ref)).
  - `folds`: the [`Fold`](@ref)s to run.

# Keyword arguments
  - `horizon`: number of forecast steps per fold (default `4`).
  - `target`: the hubverse target name (default `"target"`).
  - `model_id`: identifier tagged onto every score and forecast row
    (default `"model"`).
  - `date_col`, `location_col`, `value_col`: source column names.
  - `as_of_col`: optional vintage column; respected for both training
    availability and truth selection.
  - `epoch`: reference date for the weekly time index (default the minimum
    `date_col` in `data`).
  - `quantile_levels`: quantile grid for WIS and stored forecasts.
  - `coverage_levels`: nominal central-interval coverages to score
    (default `(0.5, 0.9)`).
  - `ndraws`, `nchains`, `adtype`, `n_samples`: forwarded to `fit`/`forecast`.
  - `fit_kwargs`: extra keyword arguments merged into each `fit` call.
  - `warmstart`: optional hook `(model, fit_table; rng) -> NamedTuple` whose
    result is merged into the `fit` keyword arguments (e.g. Pathfinder inits).
  - `fold_parallel`: thread across folds (default `false`; chains are threaded
    inside `fit` regardless).
  - `rng`: random number generator; per-fold generators are derived from it so
    runs are deterministic and thread-safe.
"""
function run_backtest(model, data, folds;
        horizon::Int = 4, target::AbstractString = "target",
        model_id::AbstractString = "model", date_col::Symbol = :date,
        location_col::Symbol = :location, value_col::Symbol = :value,
        as_of_col::Union{Nothing, Symbol} = nothing,
        epoch::Union{Nothing, Dates.Date} = nothing,
        quantile_levels = DEFAULT_QUANTILES,
        coverage_levels = (0.5, 0.9), ndraws::Int = 1000,
        nchains::Int = 4, adtype = ADTypes.AutoForwardDiff(),
        n_samples::Int = 200, fit_kwargs::NamedTuple = (;),
        warmstart = nothing, fold_parallel::Bool = false,
        rng::Random.AbstractRNG = Random.default_rng())
    df = DataFrames.DataFrame(data)
    ep = epoch === nothing ? minimum(df[!, date_col]) : epoch
    truth = _truth_lookup(df, date_col, location_col, value_col, as_of_col)
    tid = String(model_id)
    tgt = String(target)
    nf = length(folds)
    seeds = rand(rng, UInt, max(nf, 1))
    score_parts = Vector{DataFrames.DataFrame}(undef, nf)
    fc_parts = Vector{DataFrames.DataFrame}(undef, nf)

    run_fold = function (i)
        fold = folds[i]
        frng = Random.Xoshiro(seeds[i])
        train = df[fold.train_rows, :]
        ft = _fit_table(train, ep, date_col, location_col, value_col,
            as_of_col)
        extra = warmstart === nothing ? fit_kwargs :
                merge(fit_kwargs, warmstart(model, ft; rng = frng))
        fitted = fit(model, ft; adtype = adtype, ndraws = ndraws,
            nchains = nchains, rng = frng, location_col = :location,
            time_col = :time, value_col = :value, extra...)
        spec = (; horizon = horizon, reference_date = fold.origin_date,
            target = tgt, output_types = (:quantile, :sample),
            quantile_levels = quantile_levels, n_samples = n_samples)
        fc = forecast(fitted, spec; rng = frng)
        score_parts[i], fc_parts[i] = _score_fold(
            fc, fold.origin_date, truth, tid, tgt, quantile_levels,
            coverage_levels)
        return nothing
    end

    if fold_parallel
        Threads.@threads for i in 1:nf
            run_fold(i)
        end
    else
        for i in 1:nf
            run_fold(i)
        end
    end

    scores = nf == 0 ? _empty_scores() : reduce(vcat, score_parts)
    forecasts = nf == 0 ? DataFrames.DataFrame() :
                reduce(vcat, fc_parts)
    summary = Dict{String, Any}(
        "model_id" => tid, "target" => tgt, "n_folds" => nf,
        "horizon" => horizon, "ndraws" => ndraws, "nchains" => nchains,
        "mean_energy_score" => _mean_metric(scores, "energy_score"),
        "mean_wis" => _mean_metric(scores, "wis"))
    return BacktestResult(tid, scores, forecasts, summary)
end
