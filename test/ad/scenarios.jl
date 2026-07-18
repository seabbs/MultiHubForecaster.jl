# PACKAGE-OWNED — scaffold writes this once and never overwrites it.
#
# Per-backend AD gradient test items. Each backend is its own `@testitem`,
# tagged so the per-backend CI can select it with a tag filter (e.g.
# `julia test/ad/runtests.jl enzyme_reverse`). The harness wiring lives in the
# managed `setup.jl`; the SCENARIOS come from the package's own `ADFixtures`
# registry (the linked log-joints of `Baseline` models on simulated data).

@testitem "ForwardDiff gradients (posterior)" tags=[:ad, :forwarddiff] setup=[ADHelpers] begin
    test_working_backend("ForwardDiff"; category = :posterior)
end

@testitem "Mooncake reverse gradients (posterior)" tags=[:ad, :mooncake, :mooncake_reverse] setup=[ADHelpers] begin
    test_working_backend("Mooncake reverse"; category = :posterior)
end

@testitem "Enzyme reverse gradients (posterior)" tags=[:ad, :enzyme, :enzyme_reverse] setup=[ADHelpers] begin
    test_working_backend("Enzyme reverse"; category = :posterior)
end
