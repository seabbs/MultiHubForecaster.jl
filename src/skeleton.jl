# The `Skeleton` generic model. This is the framing every MultiHubForecaster
# model is built on: a pure-Turing `@model` whose components are injected as
# component STRUCTS (dependency injection) and spliced in with `to_submodel` via
# the generic [`generate`](@ref) function, with their shared settings gathered in
# a single `params::NamedTuple`. It carries no EpiAware dependency and no
# target-specific logic.
#
# All external symbols are used qualified (`DynamicPPL.`, ...); the `import`s
# live in the main module file.

@doc raw"""
The generic forecasting-model skeleton: a pure-Turing `@model` assembled from
injected component structs.

`Skeleton` is the framing every MultiHubForecaster model is built on. Rather than
hard-coding its parts, it takes them as component structs (dependency injection)
and splices each in as a Turing submodel. Each part is an
[`AbstractLatent`](@ref), [`AbstractSeasonality`](@ref), [`AbstractLink`](@ref)
or [`AbstractObservation`](@ref); the generic [`generate`](@ref) function
dispatches on the struct type to build the corresponding submodel, so a concrete
model iterates by swapping a component struct (with its own priors) or an entry
of `params` rather than by editing this function.

The latent and seasonality submodels return length-`n` paths on the shared
linear-predictor scale; their sum `η` is the linear predictor. The observation
submodel maps `η` to the natural scale through the injected `link` and observes
(or predicts) the data.

The model returns `(; η, obs)`: the linear predictor and the observation
submodel's return value.

`Skeleton` is `public` (reached as `MultiHubForecaster.Skeleton`) but not
exported; [`Baseline`](@ref) is a minimal-complete instance of it.

# Arguments
  - `latent::AbstractLatent`: the latent-path component.
  - `seasonality::AbstractSeasonality`: the seasonal-effect component, added to
    the latent path.
  - `link::AbstractLink`: the transform from the linear-predictor scale to the
    natural-scale mean.
  - `observation::AbstractObservation`: the observation component, which applies
    `link` to `η` and observes (or predicts) the data.
  - `params::NamedTuple`: the shared settings; must provide `params.n`, the
    series length, and `params.y`, the observed data.
"""
DynamicPPL.@model function Skeleton(
        latent::AbstractLatent, seasonality::AbstractSeasonality,
        link::AbstractLink, observation::AbstractObservation,
        params::NamedTuple)
    n = params.n
    trend ~ DynamicPPL.to_submodel(generate(latent, n), false)
    seas ~ DynamicPPL.to_submodel(generate(seasonality, n), false)
    η = trend .+ seas
    obs ~ DynamicPPL.to_submodel(
        generate(observation, link, η, params.y), false)
    return (; η, obs)
end
