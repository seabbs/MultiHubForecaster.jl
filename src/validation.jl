# Hub-validation wrapper: run the real hubverse validators (the R package
# `hubValidations`) against a submission by shelling out to `Rscript`. If R or
# `hubValidations` is unavailable, return a structured "unavailable" result
# rather than raising, so a caller without the R toolchain degrades gracefully.
#
# R dependency: this needs `Rscript` on `PATH` and the `hubValidations` R
# package installed (`install.packages("hubValidations")` or from the hubverse
# R-universe). Everything else uses only `Base`.

"""
`HubValidationResult` is the structured outcome of a hubverse submission
validation run (see [`validate_submission`](@ref)).

## Fields

  - `available`: whether the R validator could run at all (`Rscript` present
    and `hubValidations` installed).
  - `passed`: whether validation passed; always `false` when `available` is
    `false`.
  - `messages`: human-readable lines from the validator or the reason it was
    unavailable.
  - `output`: the raw combined stdout/stderr captured from `Rscript`.
"""
struct HubValidationResult
    "Whether the R validator could run."
    available::Bool
    "Whether validation passed (`false` when unavailable)."
    passed::Bool
    "Human-readable validator messages or the unavailability reason."
    messages::Vector{String}
    "Raw combined `Rscript` output."
    output::String
end

# The R driver script: check for hubValidations, run the requested validator,
# and print machine-readable markers the Julia side parses.
const _R_VALIDATION_SCRIPT = raw"""
args <- commandArgs(trailingOnly = TRUE)
hub_path <- args[[1]]
file_path <- args[[2]]
validator <- args[[3]]
if (!requireNamespace("hubValidations", quietly = TRUE)) {
  cat("MHF_UNAVAILABLE\n")
  quit(status = 0)
}
res <- tryCatch(
  {
    fn <- getExportedValue("hubValidations", validator)
    fn(hub_path = hub_path, file_path = file_path)
  },
  error = function(e) {
    cat("MHF_ERROR:", conditionMessage(e), "\n")
    quit(status = 0)
  }
)
print(res)
passed <- tryCatch(
  {
    hubValidations::check_for_errors(res)
    TRUE
  },
  error = function(e) FALSE
)
if (isTRUE(passed)) cat("MHF_PASS\n") else cat("MHF_FAIL\n")
"""

# Lines of `text` that carry information for a caller: non-empty and not one of
# the internal machine markers.
function _validator_messages(text::AbstractString)
    msgs = String[]
    for line in split(text, '\n')
        s = strip(line)
        (isempty(s) || startswith(s, "MHF_PASS") || startswith(s, "MHF_FAIL") ||
         startswith(s, "MHF_UNAVAILABLE")) && continue
        push!(msgs, String(s))
    end
    return msgs
end

"""
Validate a hubverse submission by running the R `hubValidations` package.

`validate_submission` shells out to `Rscript`, invoking
`hubValidations::validate_submission` (or another `hubValidations` entry point
named by `validator`, e.g. `"validate_model_data"`) on `hub_path` and
`submission_file`, and returns a [`HubValidationResult`](@ref). When `Rscript`
is not on `PATH` or the `hubValidations` package is not installed, the result
has `available = false` and `passed = false` rather than raising, so callers
without the R toolchain degrade gracefully.

# Arguments
  - `hub_path`: path to the hubverse-hub clone (its root directory).
  - `submission_file`: path to the model-output file to validate.

# Keyword Arguments
  - `validator`: the `hubValidations` function to call, as a `String` (default
    `"validate_submission"`).
"""
function validate_submission(hub_path::AbstractString,
        submission_file::AbstractString;
        validator::AbstractString = "validate_submission")
    if Sys.which("Rscript") === nothing
        return HubValidationResult(false, false,
            ["Rscript not found on PATH; install R to run hub validation"], "")
    end
    script = tempname() * ".R"
    write(script, _R_VALIDATION_SCRIPT)
    outfile = tempname()
    text = ""
    try
        open(outfile, "w") do io
            cmd = `Rscript $script $hub_path $submission_file $validator`
            run(pipeline(ignorestatus(cmd); stdout = io, stderr = io))
        end
        text = read(outfile, String)
    catch err
        return HubValidationResult(false, false,
            ["failed to run Rscript: $(err)"], text)
    finally
        rm(script; force = true)
        rm(outfile; force = true)
    end

    if occursin("MHF_UNAVAILABLE", text)
        return HubValidationResult(false, false,
            ["hubValidations R package is not installed"], text)
    elseif occursin("MHF_ERROR", text)
        return HubValidationResult(true, false, _validator_messages(text), text)
    elseif occursin("MHF_PASS", text)
        return HubValidationResult(true, true, _validator_messages(text), text)
    elseif occursin("MHF_FAIL", text)
        return HubValidationResult(true, false, _validator_messages(text), text)
    else
        return HubValidationResult(false, false,
            ["unrecognised validator output"], text)
    end
end
