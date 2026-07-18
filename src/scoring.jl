# Forecast scoring rules: the weighted interval score (WIS) and its
# decomposition for quantile forecasts, central-interval coverage, and the
# multivariate energy score for sample forecasts. Dependency-light: only
# `Statistics` and `LinearAlgebra` (both imported in the main module file).
#
# This is generalised tooling for any hubverse hub; it carries no
# target-specific (e.g. ILI) logic. Callers pass quantile levels, values and
# observations directly.

# Interval score for one central prediction interval (Gneiting & Raftery,
# 2007). `lower`/`upper` are the interval bounds, `observation` the realised
# value and `coverage` the interval's nominal coverage (e.g. `0.9` for the
# 90% interval). Not exported; used inside `weighted_interval_score`.
function _interval_score(lower::Real, upper::Real, observation::Real,
        coverage::Real)
    alpha = 1 - coverage
    width = upper - lower
    penalty = (2 / alpha) * max(lower - observation, 0.0) +
              (2 / alpha) * max(observation - upper, 0.0)
    return width + penalty
end

@doc raw"""
Weighted interval score (WIS) for a single quantile forecast.

`levels` are quantile levels symmetric about the median (e.g.
`[0.025, 0.25, 0.5, 0.75, 0.975]`) with corresponding predicted `values`,
scored against a scalar `observation`. Each symmetric pair `(a, 1 - a)` for
`a < 0.5` forms a central prediction interval; the ``K`` intervals are combined
with the median absolute error following Bracher et al. (2021):

```math
\mathrm{WIS} = \frac{1}{K + 1/2} \left( \frac{1}{2} |y - m| +
    \sum_{k=1}^{K} \frac{\alpha_k}{2} \mathrm{IS}_{\alpha_k} \right)
```

where ``m`` is the median, ``\alpha_k = 2a_k`` is the miscoverage of the
interval formed by level ``a_k`` and its counterpart ``1 - a_k``, and
``\mathrm{IS}`` is the interval score.

`weighted_interval_score` returns a named tuple
`(; wis, dispersion, overprediction, underprediction)`; the three components
sum to `wis` and decompose it into interval width versus one-sided misses
above and below, following the same reference.

# Arguments
  - `observation`: the realised value the forecast is scored against.
  - `values`: predicted quantile values, one per entry of `levels`.
  - `levels`: quantile levels in ``[0, 1]``, including the median `0.5` and at
    least one symmetric pair around it.
"""
function weighted_interval_score(observation::Real,
        values::AbstractVector{<:Real}, levels::AbstractVector{<:Real})
    length(values) == length(levels) || throw(DimensionMismatch(
        "values and levels must have the same length"))

    lv = collect(Float64, levels)
    val = collect(Float64, values)
    tol = 1e-8

    median_idx = findfirst(a -> abs(a - 0.5) < tol, lv)
    median_idx === nothing &&
        throw(ArgumentError("levels must include the median (0.5)"))
    median = val[median_idx]

    lower_levels = filter(a -> a < 0.5 - tol, lv)
    K = length(lower_levels)
    K == 0 && throw(ArgumentError(
        "levels must include at least one central interval below 0.5"))

    is_sum = 0.0
    dispersion = 0.0
    overprediction = 0.0
    underprediction = 0.0
    for a in lower_levels
        lower_idx = findfirst(x -> abs(x - a) < tol, lv)
        upper_idx = findfirst(x -> abs(x - (1 - a)) < tol, lv)
        upper_idx === nothing && throw(ArgumentError(
            "level $a has no symmetric upper counterpart $(1 - a)"))
        lower = val[lower_idx]
        upper = val[upper_idx]
        alpha_k = 2 * a
        coverage = 1 - alpha_k

        is_k = _interval_score(lower, upper, observation, coverage)
        is_sum += (alpha_k / 2) * is_k

        dispersion += (alpha_k / 2) * (upper - lower)
        overprediction += max(lower - observation, 0.0)
        underprediction += max(observation - upper, 0.0)
    end

    denom = K + 0.5
    median_term = 0.5 * abs(observation - median)
    wis_total = (median_term + is_sum) / denom

    dispersion /= denom
    over_tail = 0.5 * max(median - observation, 0.0)
    under_tail = 0.5 * max(observation - median, 0.0)
    overprediction = (overprediction + over_tail) / denom
    underprediction = (underprediction + under_tail) / denom

    return (wis = wis_total, dispersion = dispersion,
        overprediction = overprediction, underprediction = underprediction)
end

@doc raw"""
Whether `observation` falls inside the central prediction interval `[lower,
upper]`.

`interval_coverage` returns `true` when `lower <= observation <= upper`.
Averaging this indicator over many forecast tasks gives the empirical coverage
of the interval, which is compared against its nominal coverage for
calibration assessment.

# Arguments
  - `observation`: the realised value.
  - `lower`: the interval's lower bound.
  - `upper`: the interval's upper bound.
"""
function interval_coverage(observation::Real, lower::Real, upper::Real)
    return lower <= observation <= upper
end

@doc raw"""
Whether `observation` falls inside the central prediction interval at nominal
coverage `coverage`, read off a quantile forecast.

The interval is formed from the quantile at level ``(1 - c)/2`` and its
counterpart ``(1 + c)/2`` for coverage ``c``. Both levels must be present in
`levels` (to a tolerance of `1e-8`).

# Arguments
  - `observation`: the realised value.
  - `values`: predicted quantile values, one per entry of `levels`.
  - `levels`: quantile levels in ``[0, 1]``.
  - `coverage`: the interval's nominal coverage in ``(0, 1)``, e.g. `0.9`.
"""
function interval_coverage(observation::Real,
        values::AbstractVector{<:Real}, levels::AbstractVector{<:Real},
        coverage::Real)
    length(values) == length(levels) || throw(DimensionMismatch(
        "values and levels must have the same length"))
    0 < coverage < 1 || throw(ArgumentError("coverage must be in (0, 1)"))
    lv = collect(Float64, levels)
    tol = 1e-8
    lo_level = (1 - coverage) / 2
    hi_level = (1 + coverage) / 2
    lo_idx = findfirst(x -> abs(x - lo_level) < tol, lv)
    hi_idx = findfirst(x -> abs(x - hi_level) < tol, lv)
    (lo_idx === nothing || hi_idx === nothing) && throw(ArgumentError(
        "levels must include $lo_level and $hi_level for coverage $coverage"))
    return interval_coverage(observation, values[lo_idx], values[hi_idx])
end

@doc raw"""
Multivariate energy score for a sample forecast, estimated from predictive
samples.

The energy score generalises the CRPS to multiple dimensions (Gneiting &
Raftery, 2007):

```math
\mathrm{ES}(F, y) = \mathbb{E}\|X - y\| -
    \tfrac{1}{2}\, \mathbb{E}\|X - X'\|
```

for independent draws ``X, X' \sim F``. `energy_score` estimates it from
`samples` with

```math
\widehat{\mathrm{ES}} = \frac{1}{m} \sum_{i=1}^{m} \|x_i - y\| -
    \frac{1}{2 m^2} \sum_{i=1}^{m} \sum_{j=1}^{m} \|x_i - x_j\|
```

Lower is better; the score is minimised in expectation by the true predictive
distribution.

# Arguments
  - `samples`: a `d × m` matrix whose `m` columns are `d`-dimensional
    predictive draws.
  - `observation`: the realised `d`-dimensional value (length `d`).
"""
function energy_score(samples::AbstractMatrix{<:Real},
        observation::AbstractVector{<:Real})
    d, m = size(samples)
    length(observation) == d || throw(DimensionMismatch(
        "observation length $(length(observation)) != sample dimension $d"))
    m == 0 && throw(ArgumentError("samples must have at least one column"))

    term1 = 0.0
    for j in 1:m
        term1 += LinearAlgebra.norm(view(samples, :, j) .- observation)
    end
    term1 /= m

    term2 = 0.0
    for i in 1:m, j in 1:m

        term2 += LinearAlgebra.norm(view(samples, :, i) .- view(samples, :, j))
    end
    term2 /= (2 * m^2)

    return term1 - term2
end

@doc raw"""
Univariate energy score for a sample forecast; equal to the sample estimate of
the CRPS.

A one-dimensional convenience for [`energy_score`](@ref): `samples` is a vector
of scalar draws and `observation` a scalar realised value.

# Arguments
  - `samples`: predictive draws (a vector of scalars).
  - `observation`: the realised scalar value.
"""
function energy_score(samples::AbstractVector{<:Real}, observation::Real)
    return energy_score(reshape(collect(Float64, samples), 1, :),
        [Float64(observation)])
end
