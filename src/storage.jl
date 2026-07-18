# Experiment storage: persist a backtest run to disk with low disk and low RAM
# use. The tidy scores and the forecast quantiles are written as ZSTD-compressed
# Arrow (columnar, streamed by Arrow itself rather than held twice in RAM); the
# small run manifest (model config plus headline metrics) is written as TOML.
# Read helpers reload each artefact.
#
# All external symbols are used qualified (`Arrow.`, `DataFrames.`, `TOML.`);
# the `import`s live in the main module file.

# Standard artefact file names inside an experiment directory.
const _SCORES_FILE = "scores.arrow"
const _FORECASTS_FILE = "forecasts.arrow"
const _MANIFEST_FILE = "manifest.toml"

@doc """
$(TYPEDSIGNATURES)

`write_scores` writes the tidy `scores` table to `path` as ZSTD-compressed
Arrow. Returns `path`.

# Arguments
  - `path`: destination `.arrow` file.
  - `scores`: any `Tables.jl`/`DataFrames` source (the `scores` field of a
    [`BacktestResult`](@ref)).
"""
function write_scores(path::AbstractString, scores)
    Arrow.write(path, DataFrames.DataFrame(scores); compress = :zstd)
    return path
end

@doc """
$(TYPEDSIGNATURES)

`read_scores` reads a scores table written by [`write_scores`](@ref) back into
a `DataFrame`.

# Arguments
  - `path`: the `.arrow` file to read.
"""
function read_scores(path::AbstractString)
    return DataFrames.DataFrame(Arrow.Table(path))
end

@doc """
$(TYPEDSIGNATURES)

`write_forecasts` writes the `forecasts` table to `path` as ZSTD-compressed
Arrow. Returns `path`.

# Arguments
  - `path`: destination `.arrow` file.
  - `forecasts`: any `Tables.jl`/`DataFrames` source (the `forecasts` field of
    a [`BacktestResult`](@ref)).
"""
function write_forecasts(path::AbstractString, forecasts)
    Arrow.write(path, DataFrames.DataFrame(forecasts); compress = :zstd)
    return path
end

@doc """
$(TYPEDSIGNATURES)

`read_forecasts` reads a forecasts table written by [`write_forecasts`](@ref)
into a `DataFrame`.

# Arguments
  - `path`: the `.arrow` file to read.
"""
function read_forecasts(path::AbstractString)
    return DataFrames.DataFrame(Arrow.Table(path))
end

# Coerce a manifest value to something TOML can serialise (dates and periods
# become strings; everything else is passed through).
_manifest_value(x::Dates.Date) = string(x)
_manifest_value(x::Dates.Period) = string(x)
_manifest_value(x) = x

@doc """
$(TYPEDSIGNATURES)

`write_manifest` writes a small run `manifest` (model config plus summary
metrics) to `path` as TOML. Returns `path`.

# Arguments
  - `path`: destination `.toml` file.
  - `manifest`: a `Dict` or `NamedTuple`; values are coerced to
    TOML-serialisable forms (dates and periods become strings).
"""
function write_manifest(path::AbstractString,
        manifest::Union{AbstractDict, NamedTuple})
    pairs_iter = manifest isa NamedTuple ? pairs(manifest) : manifest
    d = Dict{String, Any}(
        string(k) => _manifest_value(v) for (k, v) in pairs_iter)
    open(path, "w") do io
        TOML.print(io, d)
    end
    return path
end

@doc """
$(TYPEDSIGNATURES)

`read_manifest` reads a manifest written by [`write_manifest`](@ref) into a
`Dict{String, Any}`.

# Arguments
  - `path`: the `.toml` file to read.
"""
function read_manifest(path::AbstractString)
    return TOML.parsefile(path)
end

@doc """
$(TYPEDSIGNATURES)

`save_experiment` saves a whole [`BacktestResult`](@ref) under directory `dir`,
writing `scores.arrow`, `forecasts.arrow` (both ZSTD-compressed Arrow) and
`manifest.toml` (the run summary). Creates `dir` if needed and returns it.

# Arguments
  - `dir`: the output directory.
  - `result`: the [`BacktestResult`](@ref) to persist.
"""
function save_experiment(dir::AbstractString, result::BacktestResult)
    mkpath(dir)
    write_scores(joinpath(dir, _SCORES_FILE), result.scores)
    write_forecasts(joinpath(dir, _FORECASTS_FILE), result.forecasts)
    write_manifest(joinpath(dir, _MANIFEST_FILE), result.summary)
    return dir
end

@doc """
$(TYPEDSIGNATURES)

`load_experiment` loads an experiment saved by [`save_experiment`](@ref) from
directory `dir`. Returns a `NamedTuple` `(; scores, forecasts, manifest)` with
the two tables as `DataFrame`s and the manifest as a `Dict`.

# Arguments
  - `dir`: the directory to load from.
"""
function load_experiment(dir::AbstractString)
    return (;
        scores = read_scores(joinpath(dir, _SCORES_FILE)),
        forecasts = read_forecasts(joinpath(dir, _FORECASTS_FILE)),
        manifest = read_manifest(joinpath(dir, _MANIFEST_FILE)))
end
