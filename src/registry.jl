# Registry loader: parse a hubverse-hub registry TOML into typed hub configs.
# The registry lists the upstream hubs to train across and submit to, one
# `[[hub]]` table each. The loader takes a caller-given path so it is not tied
# to any particular experiments repo.
#
# `TOML` is imported in the main module file and used qualified here.

"""
`HubConfig` is a typed description of one upstream hubverse hub, parsed from a
`[[hub]]` table of a registry TOML by [`load_registry`](@ref).

## Fields

  - `id`: short hub identifier (the `[[hub]]` table's `id`).
  - `repo`: the hub's GitHub `owner/name` slug.
  - `disease`: the disease or indicator the hub covers.
  - `targets`: every forecast target the hub offers.
  - `our_targets`: the subset of `targets` we actually forecast.
  - `output_types`: accepted hubverse output types (e.g. `quantile`, `sample`).
  - `geography`: a description of the hub's spatial units.
  - `horizons`: a description of the hub's forecast horizons.
  - `cadence`: the submission cadence.
  - `status`: free-text repository-activity note.
  - `role`: our roles for the hub (e.g. `train`, `submit`).
"""
struct HubConfig
    "Short hub identifier."
    id::String
    "GitHub `owner/name` slug."
    repo::String
    "Disease or indicator covered."
    disease::String
    "Every forecast target the hub offers."
    targets::Vector{String}
    "The subset of `targets` we forecast."
    our_targets::Vector{String}
    "Accepted hubverse output types."
    output_types::Vector{String}
    "Description of the hub's spatial units."
    geography::String
    "Description of the hub's forecast horizons."
    horizons::String
    "Submission cadence."
    cadence::String
    "Repository-activity note."
    status::String
    "Our roles for the hub (e.g. `train`, `submit`)."
    role::Vector{String}
end

# Read a string field from a parsed `[[hub]]` table, defaulting to `""`.
_getstr(tbl, key) = haskey(tbl, key) ? string(tbl[key]) : ""

# Read a string-vector field, accepting a scalar or a list, defaulting to `[]`.
function _getvec(tbl, key)
    haskey(tbl, key) || return String[]
    v = tbl[key]
    return v isa AbstractVector ? String[string(x) for x in v] :
           String[string(v)]
end

# Build a `HubConfig` from one parsed `[[hub]]` table.
function _hub_config(tbl)
    return HubConfig(
        _getstr(tbl, "id"),
        _getstr(tbl, "repo"),
        _getstr(tbl, "disease"),
        _getvec(tbl, "targets"),
        _getvec(tbl, "our_targets"),
        _getvec(tbl, "output_types"),
        _getstr(tbl, "geography"),
        _getstr(tbl, "horizons"),
        _getstr(tbl, "cadence"),
        _getstr(tbl, "status"),
        _getvec(tbl, "role")
    )
end

"""
Parse the hubverse-hub registry TOML at `path` into a vector of
[`HubConfig`](@ref).

`load_registry` expects a registry with an array of `[[hub]]` tables, each
carrying at least an `id` and `repo`; missing optional fields default to an
empty string or empty vector. The path is a caller-supplied argument so the
loader is independent of any particular experiments repository.

# Arguments
  - `path`: filesystem path to the registry TOML file.
"""
function load_registry(path::AbstractString)
    isfile(path) || throw(ArgumentError("registry file not found: $path"))
    data = TOML.parsefile(path)
    hubs = get(data, "hub", Any[])
    hubs isa AbstractVector || throw(ArgumentError(
        "registry must contain an array of `[[hub]]` tables"))
    return HubConfig[_hub_config(tbl) for tbl in hubs]
end
