# Model components as structs with dispatch (EpiAware-style). Each modelling
# role has an abstract supertype; concrete structs carry their own prior fields.
# A single generic function [`generate`](@ref) is dispatched on the component
# type and returns that component's Turing submodel (a `DynamicPPL.Model`), which
# the [`Skeleton`](@ref) splices in with `to_submodel`. The link is the one
# deterministic component (a pure transform with no priors) and is applied via
# [`apply_link`](@ref) / [`inverse_link`](@ref) rather than a submodel.
#
# All external symbols are used qualified; the `import`s live in the main module.

# --- shared forward passes (used by both the submodels and `forecast`) --------

# Non-centred AR(p) path from initial conditions `z_init` (length `p`), damping
# `damp` (length `p`), process SD `σ` and standard-normal innovations `ε` (length
# `n - p`). Returns the length-`n` path `z_t = Σ_i damp_i z_{t-i} + σ ε_{t-p}`.
# Written as an explicit loop (not `arraydist`/`cumprod`) so it differentiates
# cleanly in Enzyme as well as Mooncake and ForwardDiff.
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

# Integrate a growth-rate path `g` from an initial level `level`: the returned
# non-stationary path is `η_t = level + Σ_{s≤t} g_s`. This is the "log transform
# plus differencing" of the growth-rate latent read forwards — the level is a
# random walk whose increments are the (AR-structured) growth rate. An explicit
# accumulator loop keeps it Enzyme-differentiable.
function _integrate(level, g)
    n = length(g)
    T = promote_type(typeof(level), eltype(g))
    η = Vector{T}(undef, n)
    acc = zero(T)
    @inbounds for t in 1:n
        acc += g[t]
        η[t] = level + acc
    end
    return η
end

# Fourier seasonal effect at `times` from `2K` coefficients `β` with `period`.
# `β[2k-1]` weights the `k`-th sine, `β[2k]` the `k`-th cosine. Zero-mean: the
# series level lives in the latent, not here.
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

# Largest natural-scale mean allowed through an observation. `exp(30) ≈ 1.07e13`
# is the widest the log link can produce, so clamping just below it keeps the
# mean finite (and its gradient defined) under any warmup draw while leaving all
# real targets — counts to ~1e5, rates to ~1e6 — far inside the bound.
const _MAX_MEAN = 1.0e12
# Smallest negative-binomial size allowed. The dispersion prior is truncated at
# `0`, but a warmup draw can still map to a size of exactly `0`, at which
# `NegativeBinomial` throws `DomainError` (`r > 0` fails) and AD then tries to
# differentiate the throw. Flooring `r` keeps the constructor in-domain.
const _MIN_SIZE = 1.0e-6

# Negative binomial with mean `μ > 0` and size (dispersion) `r > 0`; variance is
# `μ + μ²/r`, so larger `r` is closer to Poisson. The size is floored away from
# `0` and the mean clamped to a finite max so the constructor can never throw a
# `DomainError` under an extreme warmup draw (see the count under-dispersion /
# large-count divergence fix, issue #6); `p` is clamped just inside `(0, 1)`.
function _nbinom(r, μ)
    rc = max(r, _MIN_SIZE)
    μc = clamp(μ, _MIN_SIZE, _MAX_MEAN)
    p = clamp(rc / (rc + μc), 1.0e-10, 1 - 1.0e-10)
    return Distributions.NegativeBinomial(rc, p)
end

# --- link components ----------------------------------------------------------

@doc raw"""
Supertype of the link components: the deterministic transform mapping the shared
linear-predictor (latent) scale to a data type's natural-scale mean, so one
latent serves each dataset through its own link. A link carries no priors and no
random variables, so it has no [`generate`](@ref) submodel; it is applied with
[`apply_link`](@ref) (forwards, `η → μ`) and [`inverse_link`](@ref) (backwards,
`μ → η`, used to centre the latent level prior on the observed scale).
"""
abstract type AbstractLink end

"The log link: `μ = exp(η)`, for count data (the latent is the log-mean)."
struct LogLink <: AbstractLink end

"The logit link: `μ = logistic(η)`, for proportion data (latent is the logit)."
struct LogitLink <: AbstractLink end

@doc """
$(TYPEDSIGNATURES)

`apply_link` maps a linear-predictor value `η` to the natural-scale mean through
`link`. The log link clamps `η` to a wide finite range first, so an extreme
warmup draw cannot overflow the mean to `Inf` (and hand AD a `NaN` gradient).
See also [`inverse_link`](@ref).
"""
apply_link(::LogLink, η) = exp(clamp(η, -30.0, 30.0))
apply_link(::LogitLink, η) = LogExpFunctions.logistic(η)

@doc """
$(TYPEDSIGNATURES)

`inverse_link` maps a natural-scale mean `μ` back to the linear-predictor scale
through `link` (the inverse of [`apply_link`](@ref)). Used to centre the latent
level prior on the observed data scale.
"""
inverse_link(::LogLink, μ) = log(μ)
inverse_link(::LogitLink, μ) = LogExpFunctions.logit(μ)

# --- latent components --------------------------------------------------------

"""
Supertype of the latent components: a submodel returning a length-`n` path on the
shared linear-predictor scale. Concrete latents carry their own priors.
"""
abstract type AbstractLatent end

@doc raw"""
$(TYPEDEF)

A growth-rate latent on the linear-predictor scale: the level is a random walk
whose increments (the growth rate, i.e. the first difference of the level) follow
a non-centred AR(`p`) process. Read on the log scale for counts this is a log
transform plus differencing, matching the smooth non-stationary trend the EDA
favours over a stationary level AR.

The path is `η_t = level + Σ_{s≤t} g_s` with growth `g` an AR(`p`):
`g_t = Σ_i damp_i g_{t-i} + σ ε_t`, `ε_t ~ Normal(0, 1)` (non-centred). The AR
order is `p`; `damp_prior` is the prior on each growth AR coefficient,
`growth_sd_prior` the prior on the innovation SD, `init_level_prior` the prior on
the initial level, and `init_growth_prior` the prior on the initial growth
values.

# Fields
$(TYPEDFIELDS)
"""
struct ARGrowthRate{D, S, L, G} <: AbstractLatent
    "AR order `p` of the growth-rate process."
    p::Int
    "Prior on each growth-rate AR coefficient (`damp`)."
    damp_prior::D
    "Prior on the growth-rate innovation SD (`σ`)."
    growth_sd_prior::S
    "Prior on the initial level (`level`), centred on the observed scale."
    init_level_prior::L
    "Prior on each of the `p` initial growth-rate values."
    init_growth_prior::G
end

# --- seasonality components ---------------------------------------------------

"""
Supertype of the seasonality components: a submodel returning a length-`n`,
zero-mean seasonal effect on the linear-predictor scale, added to the latent.
"""
abstract type AbstractSeasonality end

@doc raw"""
$(TYPEDEF)

A zero-mean Fourier seasonality: `K` harmonics of the given `period`, evaluated
from the series start time `t0`. Non-centred through independent coefficient
priors (`coef_prior`).

# Fields
$(TYPEDFIELDS)
"""
struct FourierSeasonality{C} <: AbstractSeasonality
    "Number of annual Fourier harmonics."
    K::Int
    "Seasonal period in time steps (e.g. `52` for weekly data)."
    period::Float64
    "Prior on each of the `2K` Fourier coefficients."
    coef_prior::C
    "Series start time index (the seasonal phase reference)."
    t0::Int
end

# --- observation components ---------------------------------------------------

"""
Supertype of the observation components: a submodel that maps the linear
predictor `η` to a mean through an [`AbstractLink`](@ref) and observes (or
predicts) the data. Concrete observations carry their own dispersion prior and
select the data type through the link they are paired with.
"""
abstract type AbstractObservation end

@doc raw"""
$(TYPEDEF)

A negative-binomial observation for count targets: `y_t ~ NegativeBinomial` with
mean `μ_t = apply_link(link, η_t)` and size (dispersion) `r ~ disp_prior`. Pairs
with a [`LogLink`](@ref).

# Fields
$(TYPEDFIELDS)
"""
struct NegativeBinomialObs{D} <: AbstractObservation
    "Prior on the negative-binomial size/dispersion `r`."
    disp_prior::D
end

@doc raw"""
$(TYPEDEF)

A Beta observation for proportion/rate targets in `(0, 1)`: `y_t ~ Beta(μ_t φ,
(1 - μ_t) φ)` with mean `μ_t = apply_link(link, η_t)` and precision
`φ ~ prec_prior`. Pairs with a [`LogitLink`](@ref).

# Fields
$(TYPEDFIELDS)
"""
struct BetaObs{P} <: AbstractObservation
    "Prior on the Beta precision `φ`."
    prec_prior::P
end

@doc raw"""
$(TYPEDEF)

A log-normal observation for unbounded positive rate/incidence targets (e.g.
RespiCast ARI/ILI incidence): `y_t ~ LogNormal(log μ_t, ν)` with median
`μ_t = apply_link(link, η_t)` and log-scale SD `ν ~ sigma_prior`. Pairs with a
[`LogLink`](@ref): the latent is the log-median, so no `(0, 1)` bound (Beta) or
count support (negative binomial) is imposed and the target can take any positive
real value. The observation SD `ν` is named distinctly from the latent
innovation SD `σ` so the two never collide in the un-prefixed submodel namespace.

# Fields
$(TYPEDFIELDS)
"""
struct LogNormalObs{S} <: AbstractObservation
    "Prior on the log-normal observation SD `ν` (log scale)."
    sigma_prior::S
end

# --- the generic component function -------------------------------------------

@doc raw"""
The generic component function: dispatched on a component type, it returns that
component's Turing submodel (a `DynamicPPL.Model`) for the [`Skeleton`](@ref) to
splice in with `to_submodel`. Each method is itself a Turing `@model`.

The link is deterministic (no priors, no random variables) so it has no
`generate` method; it is applied inside the observation with [`apply_link`](@ref).

# Arguments

  - `generate(latent::AbstractLatent, n)` — a length-`n` latent path.
  - `generate(seas::AbstractSeasonality, n)` — a length-`n` seasonal effect.
  - `generate(obs::AbstractObservation, link::AbstractLink, η, y)` — the
    observation of `y` given the linear predictor `η` and its link (negative
    binomial for counts, Beta for proportions, log-normal for rates).
"""
function generate end

DynamicPPL.@model function generate(latent::ARGrowthRate, n)
    σ ~ latent.growth_sd_prior
    damp ~ Turing.filldist(latent.damp_prior, latent.p)
    growth_init ~ Turing.filldist(latent.init_growth_prior, latent.p)
    ε ~ Turing.filldist(Distributions.Normal(), n - latent.p)
    level ~ latent.init_level_prior
    g = _ar_path(growth_init, damp, σ, ε)
    return _integrate(level, g)
end

DynamicPPL.@model function generate(seas::FourierSeasonality, n)
    β ~ Turing.filldist(seas.coef_prior, 2 * seas.K)
    times = seas.t0 .+ (0:(n - 1))
    return _fourier(β, times, seas.K, seas.period)
end

DynamicPPL.@model function generate(
        obs::NegativeBinomialObs, link::AbstractLink, η, y)
    r ~ obs.disp_prior
    μ = map(x -> apply_link(link, x), η)
    for t in eachindex(y)
        y[t] ~ _nbinom(r, μ[t])
    end
    return μ
end

DynamicPPL.@model function generate(
        obs::BetaObs, link::AbstractLink, η, y)
    φ ~ obs.prec_prior
    μ = map(x -> apply_link(link, x), η)
    for t in eachindex(y)
        m = clamp(μ[t], 1.0e-6, 1 - 1.0e-6)
        y[t] ~ Distributions.Beta(m * φ, (1 - m) * φ)
    end
    return μ
end

DynamicPPL.@model function generate(
        obs::LogNormalObs, link::AbstractLink, η, y)
    ν ~ obs.sigma_prior
    μ = map(x -> apply_link(link, x), η)
    for t in eachindex(y)
        # Clamp the median to a finite positive range so `log` (the LogNormal
        # location) stays finite under any warmup draw; `apply_link`'s own clamp
        # already bounds `μ`, this guards the inverse.
        m = clamp(μ[t], _MIN_SIZE, _MAX_MEAN)
        y[t] ~ Distributions.LogNormal(log(m), ν)
    end
    return μ
end
