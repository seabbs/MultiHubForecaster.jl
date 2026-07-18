@testitem "Skeleton and Baseline: public, not exported, subtype" begin
    using MultiHubForecaster
    const M = MultiHubForecaster

    # Both are `public` but not exported.
    @test Base.ispublic(M, :Skeleton)
    @test Base.ispublic(M, :Baseline)
    @test !Base.isexported(M, :Skeleton)
    @test !Base.isexported(M, :Baseline)

    model = M.Baseline(; target_type = :count, p = 2)
    @test model isa M.AbstractForecastModel
    @test hasmethod(M.fit, Tuple{M.Baseline, Any})

    @test_throws ArgumentError M.Baseline(; target_type = :nonsense)
    @test_throws ArgumentError M.Baseline(; p = 0)
end

@testitem "Baseline: parameter recovery on simulated counts" tags=[:sample] begin
    using MultiHubForecaster, DataFrames, Random, Statistics
    const M = MultiHubForecaster
    D = M.Distributions
    Random.seed!(20240718)

    ρ_true, σ_true, level_true = 0.5, 0.15, 4.0
    βs, βc = 0.4, -0.3
    amp_true = sqrt(βs^2 + βc^2)
    r_true, P = 15.0, 52.0

    n = 150
    z = zeros(n)
    for t in 2:n
        z[t] = ρ_true * z[t - 1] + σ_true * randn()
    end
    y = map(1:n) do t
        ω = 2π * t / P
        μ = exp(level_true + βs * sin(ω) + βc * cos(ω) + z[t])
        rand(D.NegativeBinomial(r_true, r_true / (r_true + μ)))
    end
    df = DataFrame(location = "a", time = 1:n, value = y)

    model = M.Baseline(;
        target_type = :count, p = 1, n_harmonics = 1, backfill_lag = 0)
    fitted = M.fit(model, df; adtype = M.ADTypes.AutoForwardDiff(),
        ndraws = 600, nchains = 2, target_acceptance = 0.9,
        rng = MersenneTwister(1))

    ch = fitted.chains["a"]
    covers(draws, truth) = quantile(draws, 0.025) <= truth <=
                           quantile(draws, 0.975)

    damp = getindex.(M._draws(ch, :damp), 1)
    amp = [sqrt(b[1]^2 + b[2]^2) for b in M._draws(ch, :β)]
    r = M._draws(ch, :r)

    # AR coefficient, seasonal amplitude and observation dispersion are each
    # recovered: the 95% credible interval covers the simulating value.
    @test covers(damp, ρ_true)
    @test covers(amp, amp_true)
    @test covers(r, r_true)
    # Amplitude is well identified, so also check the point estimate is close.
    @test isapprox(median(amp), amp_true; atol = 0.25)
end

@testitem "Baseline: forecast returns a valid hubverse table" tags=[:sample] begin
    using MultiHubForecaster, DataFrames, Random, Dates
    const M = MultiHubForecaster
    D = M.Distributions
    Random.seed!(11)

    function series(n, level)
        z = zeros(n)
        for t in 2:n
            z[t] = 0.5 * z[t - 1] + 0.15 * randn()
        end
        [rand(D.NegativeBinomial(15,
             15 / (15 + exp(level + 0.4 * sin(2π * t / 52) + z[t]))))
         for t in 1:n]
    end
    n = 70
    df = vcat(
        DataFrame(location = "a", time = 1:n, value = series(n, 4.0)),
        DataFrame(location = "b", time = 1:n, value = series(n, 3.5)))

    model = M.Baseline(;
        target_type = :count, p = 2, n_harmonics = 1, backfill_lag = 2)
    fitted = M.fit(model, df; adtype = M.ADTypes.AutoForwardDiff(),
        ndraws = 100, nchains = 2, rng = MersenneTwister(2))

    horizon = 4
    levels = M.DEFAULT_QUANTILES
    fc = M.forecast(fitted,
        (; horizon = horizon, reference_date = Date(2024, 1, 6),
            target = "wk inc flu hosp", quantile_levels = levels);
        rng = MersenneTwister(3))

    # Hubverse schema columns present.
    for c in [:reference_date, :target, :horizon, :target_end_date,
        :location, :output_type, :output_type_id, :value]
        @test c in propertynames(fc)
    end
    @test all(fc.output_type .== "quantile")
    @test all(isfinite, fc.value)
    @test Set(fc.horizon) == Set(1:horizon)
    @test all(fc.target_end_date .==
              fc.reference_date .+ Day.(7 .* fc.horizon))

    # Quantiles are monotone non-decreasing within each task.
    for g in groupby(fc, [:location, :horizon])
        ord = sortperm(parse.(Float64, g.output_type_id))
        vals = g.value[ord]
        @test all(diff(vals) .>= -1e-8)
    end
end

@testitem "Baseline: write_submission round-trip and energy_score" tags=[:sample] begin
    using MultiHubForecaster, DataFrames, Random, Dates, CSV
    const M = MultiHubForecaster
    D = M.Distributions
    Random.seed!(21)

    n = 60
    z = zeros(n)
    for t in 2:n
        z[t] = 0.5 * z[t - 1] + 0.15 * randn()
    end
    y = [rand(D.NegativeBinomial(15,
             15 / (15 + exp(4 + 0.4 * sin(2π * t / 52) + z[t]))))
         for t in 1:n]
    df = DataFrame(location = "a", time = 1:n, value = y)

    model = M.Baseline(;
        target_type = :count, p = 2, n_harmonics = 1, backfill_lag = 0)
    fitted = M.fit(model, df; adtype = M.ADTypes.AutoForwardDiff(),
        ndraws = 120, nchains = 2, rng = MersenneTwister(4))

    horizon = 3
    fc = M.forecast(fitted,
        (; horizon = horizon, reference_date = Date(2024, 1, 6),
            target = "wk inc flu hosp",
            output_types = (:quantile, :sample), n_samples = 80);
        rng = MersenneTwister(5))

    # Round-trip a submission through `write_submission` and read it back.
    hub = mktempdir()
    res = M.write_submission(fc, hub; model_id = "baseline", formats = (:csv,))
    @test length(res) == 1
    path = only(values(res[1].paths))
    @test isfile(path)
    back = CSV.read(path, DataFrame)
    @test nrow(back) == nrow(fc)
    @test :model_id ∉ propertynames(back)     # dropped on write

    # Sanity energy score on the sample forecast: a cloud centred on an
    # observation scores below the same cloud shifted away.
    samp = fc[fc.output_type .== "sample", :]
    mat = Matrix{Float64}(undef, horizon,
        length(unique(samp.output_type_id)))
    for k in 1:horizon
        rows = samp[samp.horizon .== k, :]
        mat[k, :] = rows.value[1:size(mat, 2)]
    end
    obs = vec(M.Statistics.median(mat; dims = 2))
    centred = M.energy_score(mat, obs)
    shifted = M.energy_score(mat, obs .+ 500)
    @test isfinite(centred) && centred >= 0
    @test centred < shifted
end

@testitem "Baseline: proportion target samples and forecasts" tags=[:sample] begin
    using MultiHubForecaster, DataFrames, Random, Dates
    const M = MultiHubForecaster
    D = M.Distributions
    Random.seed!(31)

    logistic(x) = 1 / (1 + exp(-x))
    n = 60
    z = zeros(n)
    for t in 2:n
        z[t] = 0.5 * z[t - 1] + 0.15 * randn()
    end
    φ = 80.0
    y = map(1:n) do t
        m = logistic(-1.0 + 0.3 * sin(2π * t / 52) + z[t])
        rand(D.Beta(m * φ, (1 - m) * φ))
    end
    df = DataFrame(location = "a", time = 1:n, value = y)

    model = M.Baseline(;
        target_type = :proportion, p = 1, n_harmonics = 1, backfill_lag = 0)
    fitted = M.fit(model, df; adtype = M.ADTypes.AutoForwardDiff(),
        ndraws = 80, nchains = 2, rng = MersenneTwister(6))

    fc = M.forecast(fitted,
        (; horizon = 3, reference_date = Date(2024, 1, 6),
            target = "wk inc flu prop ed visits"); rng = MersenneTwister(7))
    @test all(isfinite, fc.value)
    @test all(0 .<= fc.value .<= 1)
end
