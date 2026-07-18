# `Baseline`: a minimal-complete instance of the [`Skeleton`](@ref). It is the
# canonical comparison baseline every experiment runs first: a single-series
# model (per location, no pooling yet) built from component structs — a
# non-centred growth-rate AR(`p`) latent on the linear-predictor scale plus a few
# Fourier harmonics, a per-data-type link (log for counts, logit for
# proportions), and a target-type-selected observation (negative binomial for
# counts, Beta for proportions). Backfill is not modelled yet.
#
# Inference is NUTS through ADTypes; forecasting continues each fitted
# growth-rate innovation stream with a fresh prior-draw tail (the out-of-sample
# technique from ComposableTuringIDModels PR #128) and draws the horizon
# observations.
#
# The shared forward passes and the component structs live in `components.jl`;
# all external symbols are used qualified.

# --- prior defaults -----------------------------------------------------------

# Weakly-informative default priors. Non-centred throughout: the growth-rate AR
# innovations are standard normal (scaled by `growth_sd`), the seasonal
# coefficients are small, and the level is centred at run time on the observed
# scale (see `_level_prior`). The `growth_sd` prior is a half-Cauchy (scale
# 0.2): its mode at zero still regularises the latent towards a smooth trend on
# well-behaved series (keeping observation dispersion identifiable), but its
# heavy tail lets the innovation SD be learned large when the data demand fast
# growth. The earlier tight half-normal (scale 0.1) under-dispersed count
# forecasts and, at very large counts, could not track the rise so the NB size
# collapsed to zero and the logpdf threw under AD (issue #6); the heavy tail
# fixes both while the half-Cauchy's zero mode preserves identifiability.
function _default_priors()
    return (;
        damp = Distributions.truncated(Distributions.Normal(0, 0.5), -1, 1),
        growth_init = Distributions.Normal(0, 0.2),
        growth_sd = Distributions.truncated(
            Distributions.Cauchy(0, 0.2), 0, Inf),
        seas_coef = Distributions.Normal(0, 0.3),
        level_sd = 1.0,
        nb_disp = Distributions.truncated(
            Distributions.Normal(0, 20), 0, Inf),
        beta_prec = Distributions.truncated(
            Distributions.Normal(0, 200), 0, Inf),
        logn_sigma = Distributions.truncated(
            Distributions.Normal(0, 1), 0, Inf))
end

# Data-scaled, weakly-informative prior on the initial linear-predictor level
# (log mean for counts, logit mean for proportions), obtained by pushing the
# observed mean back through the link's inverse. Centring on the observed scale
# keeps the growth-rate latent and the seasonal effect centred on zero.
function _level_prior(link::AbstractLink, y, sd::Real)
    ys = collect(skipmissing(y))
    m0 = link isa LogLink ? max(Statistics.mean(ys), 1.0) :
         clamp(Statistics.mean(ys), 1.0e-3, 1 - 1.0e-3)
    return Distributions.Normal(inverse_link(link, m0), sd)
end

# The link for a target type: log for counts and rates (both positive, log-scale
# latent), logit for proportions in `(0, 1)`.
_link_for(target_type::Symbol) = target_type === :proportion ? LogitLink() : LogLink()

# --- the model type -----------------------------------------------------------

@doc raw"""
$(TYPEDEF)

A minimal-complete forecasting model built on the [`Skeleton`](@ref): the
canonical MultiHubForecaster baseline. It fits each series (location)
independently — no cross-location pooling yet.

Its components are a non-centred growth-rate AR(`p`) latent
([`ARGrowthRate`](@ref)) and `n_harmonics` Fourier harmonics (`period`,
[`FourierSeasonality`](@ref)) on the shared linear-predictor scale, a per-data-
type link ([`LogLink`](@ref) for `:count` and `:rate`, [`LogitLink`](@ref) for
`:proportion`), and a matching observation chosen by `target_type`:
[`NegativeBinomialObs`](@ref) for counts, [`BetaObs`](@ref) for proportions, and
[`LogNormalObs`](@ref) for unbounded positive rates. Backfill is not modelled.

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
    "Target type driving the link and observation: `:count`, `:proportion`, or
    `:rate`."
    target_type::Symbol
    "Growth-rate AR order `p`."
    p::Int
    "Number of annual Fourier harmonics."
    n_harmonics::Int
    "Seasonal period in time steps (e.g. `52` for weekly data)."
    period::Float64
    "Prior settings (see `MultiHubForecaster._default_priors`)."
    priors::P
end

@doc """
$(TYPEDSIGNATURES)

Construct a [`Baseline`](@ref) with weakly-informative defaults. `target_type`
is `:count` (log link, negative binomial), `:proportion` (logit link, Beta), or
`:rate` (log link, log-normal — for unbounded positive rate/incidence targets);
`p` is the growth-rate AR order, `n_harmonics` the number of Fourier harmonics,
and `period` the seasonal period.
"""
function Baseline(; target_type::Symbol = :count, p::Int = 2,
        n_harmonics::Int = 2, period::Real = 52.0,
        priors::NamedTuple = _default_priors())
    target_type in (:count, :proportion, :rate) || throw(ArgumentError(
        "target_type must be :count, :proportion, or :rate"))
    p ≥ 1 || throw(ArgumentError("p must be ≥ 1"))
    n_harmonics ≥ 1 || throw(ArgumentError("n_harmonics must be ≥ 1"))
    return Baseline{typeof(priors)}(
        target_type, p, n_harmonics, Float64(period), priors)
end

# The `Skeleton` component structs for one location's series `y` starting at time
# index `t0`: a data-centred growth-rate latent, Fourier seasonality, the target-
# type link, and the matching observation.
function _components(model::Baseline, y, t0)
    pr = model.priors
    link = _link_for(model.target_type)
    latent = ARGrowthRate(model.p, pr.damp, pr.growth_sd,
        _level_prior(link, y, pr.level_sd), pr.growth_init)
    seas = FourierSeasonality(model.n_harmonics, model.period, pr.seas_coef, t0)
    obs = if model.target_type === :count
        NegativeBinomialObs(pr.nb_disp)
    elseif model.target_type === :proportion
        BetaObs(pr.beta_prec)
    else
        LogNormalObs(pr.logn_sigma)
    end
    return latent, seas, link, obs
end

# Build the `Skeleton` model for one location's series `y` at length `n`.
function _series_model(model::Baseline, y, t0)
    latent, seas, link, obs = _components(model, y, t0)
    return Skeleton(latent, seas, link, obs, (; n = length(y), y = y))
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
# growth-rate path past the fit with a fresh prior-draw innovation tail, integrate
# it to extend the non-stationary level, then draw the horizon observations.
# Returns a `horizon × ndraws` matrix.
function _forecast_draws(model::Baseline, chain, meta, horizon::Int,
        rng::Random.AbstractRNG)
    K = model.n_harmonics
    n, t0 = meta.n, meta.t0
    link = _link_for(model.target_type)
    damp = _draws(chain, :damp)
    growth_init = _draws(chain, :growth_init)
    σ = _draws(chain, :σ)
    ε = _draws(chain, :ε)
    level = _draws(chain, :level)
    β = _draws(chain, :β)
    disp = if model.target_type === :count
        _draws(chain, :r)
    elseif model.target_type === :proportion
        _draws(chain, :φ)
    else
        _draws(chain, :ν)
    end
    D = length(σ)
    out = Matrix{Float64}(undef, horizon, D)
    times = t0 .+ (0:(n + horizon - 1))
    for d in 1:D
        # Extend the non-centred growth innovation stream with a fresh prior tail
        # so the growth path (and the integrated level) continues the fit rather
        # than being redrawn (PR #128).
        ε_ext = vcat(ε[d], randn(rng, horizon))
        g = _ar_path(growth_init[d], damp[d], σ[d], ε_ext)
        z = _integrate(level[d], g)
        s = _fourier(β[d], times, K, model.period)
        η = z .+ s
        for k in 1:horizon
            μk = apply_link(link, η[n + k])
            if model.target_type === :count
                out[k, d] = rand(rng, _nbinom(disp[d], μk))
            elseif model.target_type === :proportion
                m = clamp(μk, 1.0e-6, 1 - 1.0e-6)
                out[k, d] = rand(rng,
                    Distributions.Beta(m * disp[d], (1 - m) * disp[d]))
            else
                m = clamp(μk, _MIN_SIZE, _MAX_MEAN)
                out[k, d] = rand(rng,
                    Distributions.LogNormal(log(m), disp[d]))
            end
        end
    end
    return out
end

@doc """
$(TYPEDSIGNATURES)

Produce a hubverse forecast table from a [`BaselineFit`](@ref).

For each fitted location, `forecast` continues every posterior draw's growth-rate
path over `spec.horizon` weeks — extending its non-centred innovation stream with
a fresh prior-draw tail so the integrated level continues the fit — then draws the
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
