# `Baseline`: a minimal-complete instance of the [`Skeleton`](@ref). It is the
# canonical comparison baseline every experiment runs first. The model is a
# non-centred AR(p) trend plus a few Fourier harmonics on the linear-predictor
# scale, a minimal reporting-completion (backfill) submodel over the most recent
# weeks, and a target-type-selected observation (negative binomial for counts,
# Beta for proportions). Inference is NUTS through ADTypes; forecasting continues
# each fitted non-centred innovation stream with a fresh prior-draw tail (the
# out-of-sample technique from ComposableTuringIDModels PR #128) and draws the
# horizon observations.
#
# All external symbols are used qualified; the `import`s live in the main module.

# --- shared forward passes (used by both the submodels and `forecast`) --------

# Non-centred AR(p) path from initial conditions `z_init` (length `p`), damping
# `damp` (length `p`), process SD `σ` and standard-normal innovations `ε` (length
# `n - p`). Returns the length-`n` path `z_t = Σ_i damp_i z_{t-i} + σ ε_{t-p}`.
function _ar_path(z_init, damp, σ, ε)
    p = length(z_init)
    n = p + length(ε)
    T = promote_type(eltype(z_init), eltype(damp), typeof(σ), eltype(ε))
    z = Vector{T}(undef, n)
    @inbounds for i in 1:p
        z[i] = z_init[i]
    end
    @inbounds for t in (p + 1):n
        acc = zero(T)
        for i in 1:p
            acc += damp[i] * z[t - i]
        end
        z[t] = acc + σ * ε[t - p]
    end
    return z
end

# Fourier seasonal effect at `times` from `2K` coefficients `β` with `period`.
# `β[2k-1]` weights the `k`-th sine, `β[2k]` the `k`-th cosine.
function _fourier(β, times, K::Int, period::Real)
    return map(times) do t
        s = zero(eltype(β))
        for k in 1:K
            ω = 2 * π * k * t / period
            s += β[2k - 1] * sin(ω) + β[2k] * cos(ω)
        end
        s
    end
end

# Exponential link with the linear predictor clamped to a wide finite range, so
# an extreme warmup draw cannot overflow the mean to `Inf` (and hand AD a `NaN`
# gradient). The bounds sit far outside any plausible log-count.
_safe_exp(η) = exp(clamp(η, -30.0, 30.0))

# Negative binomial with mean `μ > 0` and size (dispersion) `r > 0`; variance is
# `μ + μ²/r`, so larger `r` is closer to Poisson. `p` is clamped just inside
# `(0, 1)` for numerical safety.
function _nbinom(r, μ)
    p = clamp(r / (r + μ), 1.0e-10, 1 - 1.0e-10)
    return Distributions.NegativeBinomial(r, p)
end

# --- component submodels ------------------------------------------------------

DynamicPPL.@model function _ar_process(n, p, damp_prior, init_prior, sd_prior)
    σ ~ sd_prior
    damp ~ damp_prior
    ar_init ~ init_prior
    ε ~ Turing.filldist(Distributions.Normal(), n - p)
    return _ar_path(ar_init, damp, σ, ε)
end

DynamicPPL.@model function _fourier_seasonality(
        n, K, period, coef_prior, level_prior, t0)
    level ~ level_prior
    β ~ Turing.filldist(coef_prior, 2K)
    times = t0 .+ (0:(n - 1))
    return level .+ _fourier(β, times, K, period)
end

DynamicPPL.@model function _backfill_completion(n, L, comp_prior)
    frac_recent ~ comp_prior
    T = eltype(frac_recent)
    frac = ones(T, n)
    @inbounds for d in 1:L
        frac[n - L + d] = frac_recent[d]
    end
    return frac
end

DynamicPPL.@model function _no_backfill(n)
    return ones(n)
end

DynamicPPL.@model function _count_obs(η, completion, y, disp_prior)
    r ~ disp_prior
    μ = _safe_exp.(η) .* completion
    for t in eachindex(y)
        y[t] ~ _nbinom(r, μ[t])
    end
    return μ
end

DynamicPPL.@model function _prop_obs(η, completion, y, prec_prior)
    φ ~ prec_prior
    μ = LogExpFunctions.logistic.(η) .* completion
    for t in eachindex(y)
        m = clamp(μ[t], 1.0e-6, 1 - 1.0e-6)
        y[t] ~ Distributions.Beta(m * φ, (1 - m) * φ)
    end
    return μ
end

# --- prior defaults -----------------------------------------------------------

# Weakly-informative default priors. Non-centred throughout: the AR innovations
# are standard normal (scaled by `ar_sd`), the seasonal coefficients are small,
# and the level is centred at run time on the observed scale (see `_level_prior`).
function _default_priors()
    return (;
        damp = Distributions.truncated(Distributions.Normal(0, 0.5), -1, 1),
        ar_init = Distributions.Normal(0, 1.0),
        ar_sd = Distributions.truncated(Distributions.Normal(0, 0.5), 0, Inf),
        seas_coef = Distributions.Normal(0, 0.3),
        level_sd = 1.0,
        nb_disp = Distributions.truncated(
            Distributions.Normal(0, 20), 0, Inf),
        beta_prec = Distributions.truncated(
            Distributions.Normal(0, 200), 0, Inf))
end

# Per-lag reporting-completion prior over the most recent `L` weeks: independent
# Betas whose mean falls from ~0.95 (oldest of the recent weeks) to ~0.6 (the
# most recent, least-complete week). Minimal and swappable.
function _backfill_prior(L::Int)
    κ = 20.0
    means = L == 1 ? [0.7] : collect(range(0.95, 0.6; length = L))
    return Turing.arraydist([Distributions.Beta(m * κ, (1 - m) * κ)
                             for m in means])
end

# Data-scaled, weakly-informative prior on the linear-predictor level (log mean
# for counts, logit mean for proportions). Centring on the observed scale keeps
# the AR trend and seasonal effect centred on zero.
function _level_prior(model, y)
    ys = collect(skipmissing(y))
    sd = model.priors.level_sd
    if model.target_type === :count
        m = log(max(Statistics.mean(ys), 1.0))
    else
        p = clamp(Statistics.mean(ys), 1.0e-3, 1 - 1.0e-3)
        m = LogExpFunctions.logit(p)
    end
    return Distributions.Normal(m, sd)
end

# --- the model type -----------------------------------------------------------

@doc raw"""
$(TYPEDEF)

A minimal-complete forecasting model built on the [`Skeleton`](@ref): the
canonical MultiHubForecaster baseline.

Its components are a non-centred AR(`p`) trend and `n_harmonics` Fourier
harmonics (`period`) on the linear-predictor scale, a minimal reporting-
completion submodel over the most recent `backfill_lag` weeks, and an
observation model chosen by `target_type` (negative binomial for `:count`
targets, Beta for `:proportion` targets). Latent trajectories are per location;
no cross-location pooling is added yet.

Its `priors` field carries the prior settings (see
`MultiHubForecaster._default_priors`).

`Baseline <: `[`AbstractForecastModel`](@ref) and adds [`fit`](@ref) and
[`forecast`](@ref) methods. It is `public` (reached as
`MultiHubForecaster.Baseline`) but not exported.

# Fields
$(TYPEDFIELDS)

# Examples
```julia
using MultiHubForecaster
model = MultiHubForecaster.Baseline(; target_type = :count, p = 2)
```
"""
struct Baseline{P <: NamedTuple} <: AbstractForecastModel
    "Target type driving the observation model: `:count` or `:proportion`."
    target_type::Symbol
    "AR order `p`."
    p::Int
    "Number of annual Fourier harmonics."
    n_harmonics::Int
    "Seasonal period in time steps (e.g. `52` for weekly data)."
    period::Float64
    "Number of most-recent weeks given a reporting-completion adjustment."
    backfill_lag::Int
    "Prior settings (see `MultiHubForecaster._default_priors`)."
    priors::P
end

@doc """
$(TYPEDSIGNATURES)

Construct a [`Baseline`](@ref) with weakly-informative defaults. `target_type`
is `:count` (negative binomial) or `:proportion` (Beta); `p` is the AR order,
`n_harmonics` the number of Fourier harmonics, `period` the seasonal period, and
`backfill_lag` the number of most-recent weeks given a reporting-completion
adjustment (`0` disables it while keeping the slot).
"""
function Baseline(; target_type::Symbol = :count, p::Int = 2,
        n_harmonics::Int = 2, period::Real = 52.0, backfill_lag::Int = 3,
        priors::NamedTuple = _default_priors())
    target_type in (:count, :proportion) ||
        throw(ArgumentError("target_type must be :count or :proportion"))
    p ≥ 1 || throw(ArgumentError("p must be ≥ 1"))
    n_harmonics ≥ 1 || throw(ArgumentError("n_harmonics must be ≥ 1"))
    backfill_lag ≥ 0 || throw(ArgumentError("backfill_lag must be ≥ 0"))
    return Baseline{typeof(priors)}(
        target_type, p, n_harmonics, Float64(period), backfill_lag, priors)
end

# The four `Skeleton` component builders for one location's series `y` starting
# at time index `t0`.
function _component_builders(model::Baseline, y, t0)
    p, K = model.p, model.n_harmonics
    # Clamp the backfill window to the series length so a short location series
    # cannot index out of range (degrades to no-op past `n` rather than a
    # `BoundsError` deep in sampling).
    L = min(model.backfill_lag, length(y))
    pr = model.priors
    damp_prior = Turing.filldist(pr.damp, p)
    init_prior = Turing.filldist(pr.ar_init, p)
    coef_prior = pr.seas_coef
    level_prior = _level_prior(model, y)
    disp_prior = model.target_type === :count ? pr.nb_disp : pr.beta_prec
    lat = n -> _ar_process(n, p, damp_prior, init_prior, pr.ar_sd)
    seas = n -> _fourier_seasonality(
        n, K, model.period, coef_prior, level_prior, t0)
    back = L == 0 ? (n -> _no_backfill(n)) :
           (n -> _backfill_completion(n, L, _backfill_prior(L)))
    obs = model.target_type === :count ?
          ((η, c) -> _count_obs(η, c, y, disp_prior)) :
          ((η, c) -> _prop_obs(η, c, y, disp_prior))
    return lat, seas, back, obs
end

# Build the `Skeleton` model for one location's series `y` at length `n`.
function _series_model(model::Baseline, y, t0)
    lat, seas, back, obs = _component_builders(model, y, t0)
    return Skeleton(lat, seas, back, obs, (; n = length(y)))
end

@doc """
$(TYPEDEF)

The fitted [`Baseline`](@ref): the model, the per-location posterior chains
(FlexiChains `VNChain`s), and per-location metadata (series length, start time,
observed values). Consumed by [`forecast`](@ref).

# Fields
$(TYPEDFIELDS)
"""
struct BaselineFit{M <: Baseline}
    "The fitted model configuration."
    model::M
    "Per-location posterior chains, keyed by location."
    chains::Dict{Any, Any}
    "Per-location metadata `(; n, t0, y)`, keyed by location."
    meta::Dict{Any, NamedTuple}
end

@doc """
$(TYPEDSIGNATURES)

Fit a [`Baseline`](@ref) to `target_data`, one independent per-location NUTS fit.

`target_data` is any `Tables.jl`/`DataFrames` source with a location column, an
integer time-index column, and a value column (`location_col`, `time_col`,
`value_col`). Each location's series is sorted by time and sampled with
multi-threaded NUTS through `adtype` (an `ADTypes` backend; default
`AutoForwardDiff`). Returns a [`BaselineFit`](@ref).

# Arguments
  - `model`: the [`Baseline`](@ref) configuration to fit.
  - `target_data`: the target-data table (see above).

# Keyword arguments
  - `adtype`: the AD backend (e.g. `ADTypes.AutoMooncake(; config = nothing)`).
  - `ndraws`, `nchains`: total post-warmup draws and number of chains.
  - `target_acceptance`: NUTS target acceptance rate.
  - `rng`: random number generator.
  - `location_col`, `time_col`, `value_col`: column names.
"""
function fit(model::Baseline, target_data;
        adtype = ADTypes.AutoForwardDiff(),
        ndraws::Int = 1000, nchains::Int = 4,
        target_acceptance::Real = 0.8,
        rng::Random.AbstractRNG = Random.default_rng(),
        location_col::Symbol = :location, time_col::Symbol = :time,
        value_col::Symbol = :value)
    df = DataFrames.DataFrame(target_data)
    locs = unique(df[!, location_col])
    chains = Dict{Any, Any}()
    meta = Dict{Any, NamedTuple}()
    perchain = cld(ndraws, nchains)
    for loc in locs
        sub = DataFrames.sort(
            df[df[!, location_col] .== loc, :], time_col)
        y = collect(sub[!, value_col])
        t0 = Int(first(sub[!, time_col]))
        m = _series_model(model, y, t0)
        sampler = Turing.NUTS(target_acceptance; adtype = adtype)
        chain = Turing.sample(rng, m, sampler, Turing.MCMCThreads(),
            perchain, nchains; chain_type = FlexiChains.VNChain,
            progress = false)
        chains[loc] = chain
        meta[loc] = (; n = length(y), t0 = t0, y = y)
    end
    return BaselineFit(model, chains, meta)
end

# Per-draw values of the single-varname parameter `sym` in `chain`, flattened
# across iterations and chains. Returns a vector whose elements are scalars or
# vectors depending on the parameter.
function _draws(chain, sym::Symbol)
    for vn in FlexiChains.parameters(chain)
        if Symbol(string(vn)) === sym
            return vec(collect(chain[vn]))
        end
    end
    throw(ArgumentError("parameter :$sym not found in chain"))
end

# Standard hubverse quantile levels (the FluSight 23-quantile grid).
const DEFAULT_QUANTILES = [
    0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]

# Out-of-sample horizon draws for one location: continue each posterior draw's
# latent path past the fit with a fresh prior-draw innovation tail, then draw the
# horizon observations. Returns a `horizon × ndraws` matrix.
function _forecast_draws(model::Baseline, chain, meta, horizon::Int,
        rng::Random.AbstractRNG)
    p, K = model.p, model.n_harmonics
    n, t0 = meta.n, meta.t0
    damp = _draws(chain, :damp)
    ar_init = _draws(chain, :ar_init)
    σ = _draws(chain, :σ)
    ε = _draws(chain, :ε)
    level = _draws(chain, :level)
    β = _draws(chain, :β)
    disp = model.target_type === :count ?
           _draws(chain, :r) : _draws(chain, :φ)
    D = length(σ)
    out = Matrix{Float64}(undef, horizon, D)
    times = t0 .+ (0:(n + horizon - 1))
    for d in 1:D
        # Extend the non-centred innovation stream with a fresh prior tail so the
        # latent path continues the fit rather than being redrawn (PR #128).
        ε_ext = vcat(ε[d], randn(rng, horizon))
        z = _ar_path(ar_init[d], damp[d], σ[d], ε_ext)
        s = level[d] .+ _fourier(β[d], times, K, model.period)
        η = z .+ s
        for k in 1:horizon
            ηk = η[n + k]
            if model.target_type === :count
                μ = _safe_exp(ηk)
                out[k, d] = rand(rng, _nbinom(disp[d], μ))
            else
                μ = clamp(LogExpFunctions.logistic(ηk), 1.0e-6, 1 - 1.0e-6)
                out[k, d] = rand(rng,
                    Distributions.Beta(μ * disp[d], (1 - μ) * disp[d]))
            end
        end
    end
    return out
end

@doc """
$(TYPEDSIGNATURES)

Produce a hubverse forecast table from a [`BaselineFit`](@ref).

For each fitted location, `forecast` continues every posterior draw's latent
path over `spec.horizon` weeks — extending its non-centred innovation stream with
a fresh prior-draw tail so the trajectory continues the fit — then draws the
horizon observations. It returns a `DataFrames.DataFrame` in the hubverse
model-output schema (`reference_date`, `target`, `horizon`, `target_end_date`,
`location`, `output_type`, `output_type_id`, `value`), ready for the scoring and
hubverse-I/O tools.

# Arguments
  - `fitted`: the [`BaselineFit`](@ref) returned by [`fit`](@ref).
  - `spec::NamedTuple`: the forecast request, with fields:
      + `horizon`: number of future weeks (required).
      + `reference_date::Dates.Date`: the forecast origin (required).
      + `target::AbstractString`: the hubverse target name (required).
      + `quantile_levels`: quantile grid for `:quantile` output (default the
        FluSight grid).
      + `output_types`: any of `:quantile`, `:sample` (default `(:quantile,)`).
      + `n_samples`: cap on `:sample` rows per task (default `100`).

# Keyword arguments
  - `rng`: random number generator for the horizon innovation tails and draws.
"""
function forecast(fitted::BaselineFit, spec;
        rng::Random.AbstractRNG = Random.default_rng())
    horizon = spec.horizon
    reference_date = spec.reference_date
    target = String(spec.target)
    levels = get(spec, :quantile_levels, DEFAULT_QUANTILES)
    output_types = get(spec, :output_types, (:quantile,))
    n_samples = get(spec, :n_samples, 100)
    model = fitted.model

    ref = Vector{Dates.Date}()
    tgt = String[]
    hz = Int[]
    ted = Vector{Dates.Date}()
    locv = Any[]
    otype = String[]
    otid = String[]
    val = Float64[]

    for (loc, chain) in fitted.chains
        draws = _forecast_draws(
            model, chain, fitted.meta[loc], horizon, rng)
        D = size(draws, 2)
        for k in 1:horizon
            row = view(draws, k, :)
            end_date = reference_date + Dates.Day(7 * k)
            if :quantile in output_types
                qs = Statistics.quantile(row, levels)
                for (lev, q) in zip(levels, qs)
                    push!(ref, reference_date);
                    push!(tgt, target)
                    push!(hz, k);
                    push!(ted, end_date);
                    push!(locv, loc)
                    push!(otype, "quantile")
                    push!(otid, string(lev));
                    push!(val, q)
                end
            end
            if :sample in output_types
                nsamp = min(D, n_samples)
                for s in 1:nsamp
                    push!(ref, reference_date);
                    push!(tgt, target)
                    push!(hz, k);
                    push!(ted, end_date);
                    push!(locv, loc)
                    push!(otype, "sample")
                    push!(otid, string(s));
                    push!(val, row[s])
                end
            end
        end
    end
    return DataFrames.DataFrame(
        reference_date = ref, target = tgt, horizon = hz,
        target_end_date = ted, location = locv,
        output_type = otype, output_type_id = otid, value = val)
end
