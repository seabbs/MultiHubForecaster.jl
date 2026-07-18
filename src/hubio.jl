# Hubverse submission I/O: write a standard forecast table into a hubverse-hub
# clone (`model-output/`, `model-metadata/`). Generalised from the SISMID ILI
# sandbox to any hubverse hub — no target-specific columns are assumed. See the
# hubverse model-output schema: task-id columns, then `output_type`,
# `output_type_id`, `value`.
#
# All external symbols are used qualified (`DataFrames.`, `CSV.`, `Parquet2.`,
# `YAML.`); the `import`s live in the main module file.

# Canonical hubverse task-id column order. A hub may use a subset; unknown
# task-id columns a hub adds are kept and placed after these, before the
# output columns.
const HUBVERSE_TASK_COLUMNS = [
    :reference_date, :origin_date, :target, :horizon,
    :target_end_date, :location
]

# The hubverse output columns, always written after the task-id columns and in
# this order, with `value` last.
const HUBVERSE_OUTPUT_COLUMNS = [:output_type, :output_type_id, :value]

"""
`HUBVERSE_COLUMN_ORDER` is the canonical hubverse model-output column order:
the known task-id columns followed by `output_type`, `output_type_id`,
`value`. Used by [`order_hub_columns`](@ref) as the default ordering.
"""
const HUBVERSE_COLUMN_ORDER = [HUBVERSE_TASK_COLUMNS; HUBVERSE_OUTPUT_COLUMNS]

@doc raw"""
Return a copy of `forecast` with its columns in hubverse order.

`order_hub_columns` places the known task-id columns first (in
[`HUBVERSE_COLUMN_ORDER`](@ref) order), then any extra task-id columns the hub
adds (in their existing order), then `output_type`, `output_type_id` and
`value` last. A `model_id` column is dropped (the model is implied by the file
path). Only columns actually present are included.

# Arguments
  - `forecast`: a forecast table (any `Tables.jl`/`DataFrames` source).

# Keyword Arguments
  - `drop`: columns to exclude from the output (default `[:model_id]`).
"""
function order_hub_columns(forecast; drop = [:model_id])
    df = DataFrames.DataFrame(forecast)
    present = propertynames(df)
    dropset = Set(Symbol.(drop))
    keep(c) = c in present && !(c in dropset)

    lead = [c for c in HUBVERSE_TASK_COLUMNS if keep(c)]
    known = Set(HUBVERSE_COLUMN_ORDER)
    extras = [c for c in present
              if !(c in known) && !(c in dropset)]
    tail = [c for c in HUBVERSE_OUTPUT_COLUMNS if keep(c)]
    return DataFrames.select(df, vcat(lead, extras, tail))
end

# Group a forecast table into one `(; model_id, reference_date, df)` per
# submission file. When `model_id` is given every row belongs to it and the
# table is split by `reference_date_col` alone; otherwise the table is split by
# `(:model_id, reference_date_col)` and the `model_id` column must be present.
function _submission_groups(forecast, model_id, reference_date_col)
    df = DataFrames.DataFrame(forecast)
    groups = NamedTuple[]
    if model_id === nothing
        hasproperty(df, :model_id) || throw(ArgumentError(
            "forecast needs a `model_id` column when `model_id` is not given"))
        by = [:model_id, reference_date_col]
        for sub in DataFrames.groupby(df, by; sort = true)
            mid = string(sub[1, :model_id])
            ref = sub[1, reference_date_col]
            push!(groups, (; model_id = mid, reference_date = ref,
                out = order_hub_columns(sub)))
        end
    else
        mid = string(model_id)
        for sub in DataFrames.groupby(df, [reference_date_col]; sort = true)
            ref = sub[1, reference_date_col]
            push!(groups, (; model_id = mid, reference_date = ref,
                out = order_hub_columns(sub)))
        end
    end
    return groups
end

@doc raw"""
Write a forecast table as a hubverse submission under `hub_path`.

For each `(model_id, reference_date)` group, `write_submission` writes
`<hub_path>/model-output/<model_id>/<reference_date>-<model_id>.<ext>` for each
requested format, with columns in hubverse order (see
[`order_hub_columns`](@ref)) and no `model_id` column. Returns a vector of
`(; model_id, reference_date, paths, out)` named tuples, one per group, where
`paths` maps each format to the file written.

# Arguments
  - `forecast`: a forecast table in the hubverse schema (task-id columns,
    `output_type`, `output_type_id`, `value`, and a `reference_date` and/or
    `model_id` column as needed).
  - `hub_path`: the root of the hubverse-hub clone to write into.

# Keyword Arguments
  - `model_id`: the model identifier. When `nothing` (default), it is read from
    a `model_id` column, allowing several models in one table.
  - `reference_date_col`: the column holding the reference (origin) date, used
    for grouping and the file name (default `:reference_date`).
  - `formats`: an iterable of output formats, any of `:csv` and `:parquet`
    (default `(:csv, :parquet)`).
  - `dry_run`: when `true`, compute the groups and paths but write nothing.
"""
function write_submission(forecast, hub_path::AbstractString;
        model_id::Union{Nothing, AbstractString} = nothing,
        reference_date_col::Symbol = :reference_date,
        formats = (:csv, :parquet), dry_run::Bool = false)
    groups = _submission_groups(forecast, model_id, reference_date_col)
    results = NamedTuple[]
    for (; model_id, reference_date, out) in groups
        dir = joinpath(hub_path, "model-output", model_id)
        stem = string(reference_date) * "-" * model_id
        paths = Dict{Symbol, String}()
        for fmt in formats
            fmt in (:csv, :parquet) ||
                throw(ArgumentError("unknown format $fmt (use :csv, :parquet)"))
            ext = fmt === :csv ? ".csv" : ".parquet"
            path = joinpath(dir, stem * ext)
            if !dry_run
                mkpath(dir)
                if fmt === :csv
                    CSV.write(path, out; quotestrings = true)
                else
                    Parquet2.writefile(path, out)
                end
            end
            paths[fmt] = path
        end
        push!(results, (; model_id, reference_date, paths, out))
    end
    return results
end

# Normalise metadata into a `Dict{String, Any}` for YAML writing.
_metadata_dict(m::AbstractDict) = Dict{String, Any}(string(k) => v
for (k, v) in m)
_metadata_dict(m::NamedTuple) = Dict{String, Any}(string(k) => v
for (k, v) in pairs(m))

@doc raw"""
Write hubverse model metadata to `<hub_path>/model-metadata/<model_id>.yml`.

`write_model_metadata` serialises `metadata` (a `Dict` or `NamedTuple`, e.g.
`(; team_abbr, model_abbr, designated_model)`) to YAML, matching the layout of
a hubverse hub's existing metadata files. Returns the path written.

# Arguments
  - `model_id`: the model identifier; names the metadata file.
  - `hub_path`: the root of the hubverse-hub clone.
  - `metadata`: the metadata fields to write, as a `Dict` or `NamedTuple`.
"""
function write_model_metadata(model_id::AbstractString,
        hub_path::AbstractString, metadata::Union{AbstractDict, NamedTuple})
    dir = joinpath(hub_path, "model-metadata")
    mkpath(dir)
    path = joinpath(dir, model_id * ".yml")
    YAML.write_file(path, _metadata_dict(metadata))
    return path
end
