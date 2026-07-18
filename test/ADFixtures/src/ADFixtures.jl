# PACKAGE-OWNED — scaffold writes this once and never overwrites it.
#
# AD-fixture registry implementing the EpiAwarePackageTools `ADRegistry`
# contract. The scenarios are REAL differentiable log-densities from the
# package: the linked log-joint of `Baseline` models conditioned on simulated
# data — the gradients an AD backend must get right for NUTS to work. Each
# scenario carries a ForwardDiff reference gradient. The shared harness (driven
# from `test/ad/setup.jl`) consumes this registry.
module ADFixtures

using ADTypes: AutoForwardDiff
using DifferentiationInterface: DifferentiationInterface
import DifferentiationInterfaceTest as DIT
import ForwardDiff
using MultiHubForecaster
using Distributions
using Random: Random, MersenneTwister
using DynamicPPL: DynamicPPL, LogDensityFunction, VarInfo, link, getlogjoint
import LogDensityProblems as LDP

export scenarios, backends, broken_scenario_names,
       backend_broken_scenarios, backend_skip_scenarios

const MHF = MultiHubForecaster

# Turn a `Baseline` Skeleton model into a real differentiable scalar log-density.
# Linking the `VarInfo` maps every constrained variable (positive SDs, damping in
# (-1, 1), completion in (0, 1), ...) to an unconstrained coordinate, so the
# returned `f(θ)` is the log-joint (with the linking log-Jacobian) at the flat
# unconstrained vector — exactly the target NUTS differentiates. Returns
# `(f, θ0, dim)`.
function _logdensity(model; seed::Int = 1)
    vi = link(VarInfo(model), model)
    ldf = LogDensityFunction(model, getlogjoint, vi)
    dim = LDP.dimension(ldf)
    f = θ -> LDP.logdensity(ldf, θ)
    θ0 = 0.3 .* randn(MersenneTwister(seed), dim)
    return f, θ0, dim
end

_logistic(x) = 1 / (1 + exp(-x))

# Simulate a count series from the Baseline generative process (fixed seed).
function _sim_counts(n; seed::Int = 42, level = 4.0, ρ = 0.5,
        σ = 0.15, r = 15.0)
    rng = MersenneTwister(seed)
    z = zeros(n)
    for t in 2:n
        z[t] = ρ * z[t - 1] + σ * randn(rng)
    end
    return [rand(rng, NegativeBinomial(r,
                r / (r + exp(level + 0.4 * sin(2π * t / 52) + z[t]))))
            for t in 1:n]
end

# Simulate a proportion series from the Baseline generative process.
function _sim_props(n; seed::Int = 7, φ = 80.0)
    rng = MersenneTwister(seed)
    z = zeros(n)
    for t in 2:n
        z[t] = 0.5 * z[t - 1] + 0.15 * randn(rng)
    end
    return map(1:n) do t
        m = _logistic(-1.0 + 0.3 * sin(2π * t / 52) + z[t])
        rand(rng, Beta(m * φ, (1 - m) * φ))
    end
end

# Build the registry's models once, conditioned on simulated data.
function _models()
    n = 14
    yc = _sim_counts(n)
    yp = _sim_props(n)
    count_model = MHF.Baseline(;
        target_type = :count, p = 2, n_harmonics = 1, backfill_lag = 0)
    count_bf = MHF.Baseline(;
        target_type = :count, p = 2, n_harmonics = 1, backfill_lag = 2)
    prop_model = MHF.Baseline(;
        target_type = :proportion, p = 1, n_harmonics = 1, backfill_lag = 0)
    return [
        ("Baseline count posterior",
            MHF._series_model(count_model, yc, 1)),
        ("Baseline count+backfill posterior",
            MHF._series_model(count_bf, yc, 1)),
        ("Baseline proportion posterior",
            MHF._series_model(prop_model, yp, 1))
    ]
end

@doc """
    scenarios(; with_reference = false, category = :posterior)

The AD gradient scenarios — each a `DIT.Scenario{:gradient, :out}` over a real
`Baseline` log-density conditioned on simulated data. When
`with_reference = true` each scenario carries its ForwardDiff reference gradient
in `res1`. `category` is accepted for the harness's group selector; all
scenarios are in the single `:posterior` group.
"""
function scenarios(;
        with_reference::Bool = false, category::Symbol = :posterior)
    out = DIT.Scenario{:gradient, :out}[]
    for (i, (name, model)) in enumerate(_models())
        f, θ0, _ = _logdensity(model; seed = i)
        ref = with_reference ?
              DifferentiationInterface.gradient(f, AutoForwardDiff(), θ0) :
              nothing
        push!(out,
            DIT.Scenario{:gradient, :out}(f, θ0; name = name, res1 = ref))
    end
    return out
end

@doc """
    backends()

The AD backends exercised against the scenarios, as `(; name, backend)` named
tuples: ForwardDiff (the reference), Mooncake, and Enzyme reverse. Per-backend
brokenness is recorded honestly in [`backend_broken_scenarios`](@ref) rather than
by trimming this list.
"""
function backends()
    return [
        (name = "ForwardDiff", backend = _forwarddiff()),
        (name = "Mooncake reverse", backend = _mooncake()),
        (name = "Enzyme reverse", backend = _enzyme())
    ]
end

_forwarddiff() = AutoForwardDiff()
function _mooncake()
    ADT = Base.require(Base.PkgId(
        Base.UUID("47edcb42-4c32-4615-8424-f2b9edc5f35b"), "ADTypes"))
    return ADT.AutoMooncake(; config = nothing)
end
function _enzyme()
    ADT = Base.require(Base.PkgId(
        Base.UUID("47edcb42-4c32-4615-8424-f2b9edc5f35b"), "ADTypes"))
    Enzyme = Base.require(Base.PkgId(
        Base.UUID("7da242da-08ed-463a-9acd-ee780be4f1d9"), "Enzyme"))
    # `function_annotation = Enzyme.Const`: the log-density closures carry no
    # derivative data, and without this Enzyme raises a mutability exception on
    # every DynamicPPL log-density.
    return ADT.AutoEnzyme(;
        mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
        function_annotation = Enzyme.Const)
end

"Scenario names broken on every backend (none — all are FD-differentiable)."
broken_scenario_names() = String[]

@doc """
    backend_broken_scenarios()

Per-backend broken scenario names (`Dict{String, Set{String}}`), populated
honestly from the actual `test/ad` run rather than by silencing.
"""
function backend_broken_scenarios()
    return Dict{String, Set{String}}()
end

"Per-backend scenario names too unstable to even run (segfault/hang)."
backend_skip_scenarios() = Dict{String, Set{String}}()

end # module ADFixtures
