# PACKAGE-OWNED — scaffold writes this once and never overwrites it.
#
# Benchmark suite for MultiHubForecaster. Defines a BenchmarkTools
# `BenchmarkGroup` named `SUITE` that the managed `run.jl` / `compare.jl`
# consume.
#
# Groups:
#   "Sampling"     — a short NUTS run fitting a `Baseline` count model, and the
#       out-of-sample `forecast` extension over a horizon.
#   "AD gradients" — gradient of the `Baseline` log-density across AD backends,
#       keyed `["AD gradients"][scenario][backend]` so `compare.jl` folds it into
#       a per-(scenario × backend) matrix.

using BenchmarkTools
using MultiHubForecaster
using DataFrames
using Dates: Date
using Distributions: NegativeBinomial
using Random: MersenneTwister
using DynamicPPL: LogDensityFunction, VarInfo, link, getlogjoint
import LogDensityProblems as LDP
import DifferentiationInterface as DI
using ADTypes: AutoForwardDiff, AutoMooncake, AutoEnzyme
import ForwardDiff, Mooncake, Enzyme

const MHF = MultiHubForecaster
const SUITE = BenchmarkGroup()

# --- shared fixtures --------------------------------------------------------

const N = 60

# A representative simulated count series.
function _sim_counts(n; seed = 42, level = 4.0, ρ = 0.5, σ = 0.15, r = 15.0)
    rng = MersenneTwister(seed)
    z = zeros(n)
    for t in 2:n
        z[t] = ρ * z[t - 1] + σ * randn(rng)
    end
    return [rand(rng, NegativeBinomial(r,
                r / (r + exp(level + 0.4 * sin(2π * t / 52) + z[t]))))
            for t in 1:n]
end

const _Y = _sim_counts(N)
const _DF = DataFrame(location = "a", time = 1:N, value = _Y)
const _MODEL = MHF.Baseline(;
    target_type = :count, p = 2, n_harmonics = 2, backfill_lag = 2)

# --- Sampling ---------------------------------------------------------------

let samp_grp = SUITE["Sampling"] = BenchmarkGroup()
    # A short single-chain NUTS fit; `seconds` in run.jl caps wall time.
    samp_grp["NUTS (Baseline count, 100 draws)"] = @benchmarkable MHF.fit(
        $_MODEL, $_DF; adtype = AutoForwardDiff(), ndraws = 100, nchains = 1,
        rng = MersenneTwister(1))
    # The out-of-sample forecast extension over a horizon from a fitted model.
    fitted = MHF.fit(_MODEL, _DF; adtype = AutoForwardDiff(),
        ndraws = 100, nchains = 1, rng = MersenneTwister(1))
    spec = (; horizon = 4, reference_date = Date(2024, 1, 6), target = "t")
    samp_grp["forecast (horizon 4)"] = @benchmarkable MHF.forecast(
        $fitted, $spec; rng = MersenneTwister(2))
end

# --- AD gradients -----------------------------------------------------------

# A real differentiable log-density: the linked log-joint over unconstrained ℝᵈ.
function _logdensity(model; seed::Int = 1)
    ldf = LogDensityFunction(model, getlogjoint, link(VarInfo(model), model))
    θ0 = 0.3 .* randn(MersenneTwister(seed), LDP.dimension(ldf))
    return (θ -> LDP.logdensity(ldf, θ)), θ0
end

const _ENZYME = AutoEnzyme(;
    mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
    function_annotation = Enzyme.Const)

# (backend display name, backend, runs_on_backfill_scenario?). Enzyme trips on
# the `arraydist` backfill prior (see test/ADFixtures), so it is skipped there.
const _AD_BACKENDS = [
    ("ForwardDiff", AutoForwardDiff(), true),
    ("Mooncake reverse", AutoMooncake(; config = nothing), true),
    ("Enzyme reverse", _ENZYME, false)
]

# (scenario name, model, uses_backfill?)
function _ad_scenarios()
    y = _sim_counts(14)
    count_model = MHF.Baseline(;
        target_type = :count, p = 2, n_harmonics = 1, backfill_lag = 0)
    count_bf = MHF.Baseline(;
        target_type = :count, p = 2, n_harmonics = 1, backfill_lag = 2)
    return [
        ("Baseline count posterior",
            MHF._series_model(count_model, y, 1), false),
        ("Baseline count+backfill posterior",
            MHF._series_model(count_bf, y, 1), true)
    ]
end

let ad_grp = SUITE["AD gradients"] = BenchmarkGroup()
    for (i, (sname, model, uses_bf)) in enumerate(_ad_scenarios())
        f, θ0 = _logdensity(model; seed = i)
        ad_grp[sname] = BenchmarkGroup()
        for (bname, backend, runs_bf) in _AD_BACKENDS
            (uses_bf && !runs_bf) && continue
            prep = DI.prepare_gradient(f, backend, θ0)
            ad_grp[sname][bname] = @benchmarkable DI.gradient(
                $f, $prep, $backend, $θ0)
        end
    end
end
