@testitem "weighted_interval_score: known example" begin
    using MultiHubForecaster: weighted_interval_score
    levels = [0.025, 0.25, 0.5, 0.75, 0.975]
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    # Observation at the median, inside every interval: only dispersion.
    r = weighted_interval_score(3.0, values, levels)
    @test r.wis ≈ 0.24
    @test r.dispersion ≈ 0.24
    @test r.overprediction ≈ 0.0
    @test r.underprediction ≈ 0.0

    # Observation above every interval: hand-computed WIS = 2.04.
    r2 = weighted_interval_score(6.0, values, levels)
    @test r2.wis ≈ 2.04
    @test r2.dispersion ≈ 0.24
    @test r2.underprediction ≈ 1.8
    @test r2.overprediction ≈ 0.0
    # Components sum to the total.
    @test r2.dispersion + r2.overprediction + r2.underprediction ≈ r2.wis
end

@testitem "weighted_interval_score: input validation" begin
    using MultiHubForecaster: weighted_interval_score
    @test_throws DimensionMismatch weighted_interval_score(
        1.0, [1.0, 2.0], [0.5])
    @test_throws ArgumentError weighted_interval_score(
        1.0, [1.0, 2.0], [0.25, 0.75])           # no median
    @test_throws ArgumentError weighted_interval_score(
        1.0, [1.0], [0.5])                        # no interval
end

@testitem "interval_coverage" begin
    using MultiHubForecaster: interval_coverage
    @test interval_coverage(3.0, 1.0, 5.0)
    @test !interval_coverage(6.0, 1.0, 5.0)

    levels = [0.025, 0.25, 0.5, 0.75, 0.975]
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    @test interval_coverage(3.0, values, levels, 0.95)     # [1, 5]
    @test !interval_coverage(4.5, values, levels, 0.5)     # [2, 4]
    @test_throws ArgumentError interval_coverage(
        3.0, values, levels, 0.9)                          # levels absent
end

@testitem "energy_score: sanity and CRPS" begin
    using MultiHubForecaster: energy_score
    using Random
    Random.seed!(42)

    # A sample cloud centred on the observation scores lower than the same
    # cloud shifted away from it.
    samples = randn(2, 800)
    y = [0.0, 0.0]
    centred = energy_score(samples, y)
    shifted = energy_score(samples .+ 5.0, y)
    @test centred < shifted
    @test centred > 0

    # Univariate convenience matches the matrix form.
    v = randn(500)
    @test energy_score(v, 0.3) ≈
          energy_score(reshape(v, 1, :), [0.3])

    @test_throws DimensionMismatch energy_score(randn(2, 10), [0.0])
end
