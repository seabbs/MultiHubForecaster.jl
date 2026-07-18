# The `Skeleton` generic model. This is the framing every MultiHubForecaster
# model is built on: a pure-Turing `@model` whose components are injected as
# arguments (dependency injection) and spliced in with `to_submodel`, with their
# settings gathered in a single `params::NamedTuple`. It carries no EpiAware
# dependency and no target-specific logic.
#
# All external symbols are used qualified (`DynamicPPL.`, `Distributions.`, ...);
# the `import`s live in the main module file.

@doc raw"""
The generic forecasting-model skeleton: a pure-Turing `@model` composed from four
injected component builders.

`Skeleton` is the framing every MultiHubForecaster model is built on. Rather than
hard-coding its parts, it takes them as arguments (dependency injection) and
splices each in as a Turing submodel (`to_submodel`, prefix off). A concrete
model iterates by swapping a component builder or an entry of `params` rather
than by editing this function.

The four components are each a *builder* — a callable returning a
`DynamicPPL.Model` — so a component whose submodel depends on a value drawn
earlier (the observation, which needs the latent linear predictor) can be built
against that value at run time.

The model returns `(; η, completion, obs)`: the linear predictor, the completion
vector, and the observation submodel's return value.

`Skeleton` is `public` (reached as `MultiHubForecaster.Skeleton`) but not
exported; [`Baseline`](@ref) is a minimal-complete instance of it.

# Arguments
  - `latent(n)`: a builder returning a submodel for a length-`n` latent path on
    the linear-predictor (log or logit) scale.
  - `seasonality(n)`: a builder returning a submodel for a length-`n` seasonal
    effect on the same scale, added to the latent path.
  - `backfill(n)`: a builder returning a submodel for a length-`n` reporting-
    completion vector in ``(0, 1]`` (``1`` where a week is fully reported),
    applied multiplicatively to the natural-scale mean.
  - `observation(η, completion)`: a builder returning a submodel that applies its
    link to the linear predictor `η`, scales by `completion`, and observes (or
    predicts) the data.
  - `params::NamedTuple`: the shared settings; must provide `params.n`, the
    series length.
"""
DynamicPPL.@model function Skeleton(
        latent, seasonality, backfill, observation, params::NamedTuple)
    n = params.n
    trend ~ DynamicPPL.to_submodel(latent(n), false)
    seas ~ DynamicPPL.to_submodel(seasonality(n), false)
    η = trend .+ seas
    completion ~ DynamicPPL.to_submodel(backfill(n), false)
    obs ~ DynamicPPL.to_submodel(observation(η, completion), false)
    return (; η, completion, obs)
end
