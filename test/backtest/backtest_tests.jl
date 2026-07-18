@testitem "date_split and partition: boundaries and no overlap" begin
    using MultiHubForecaster, DataFrames, Dates
    const M = MultiHubForecaster

    # Weekly dates over ~3.5 years so all three spans are populated.
    start = Date(2021, 1, 3)
    dates = start:Week(1):(start + Week(180))
    df = DataFrame(date = collect(dates), location = "a",
        value = 1.0:length(dates))

    split = M.date_split(df)
    @test split.present == maximum(df.date)
    @test split.train_end == split.present - Year(2)
    @test split.val_end == split.present - Year(1)

    parts = M.partition(split, df)
    # Spans are correctly bounded.
    @test all(parts.train.date .<= split.train_end)
    @test all(split.train_end .< parts.validation.date .<= split.val_end)
    @test all(split.val_end .< parts.test.date .<= split.present)
    # Disjoint and covering (no row lost, none double-counted).
    @test nrow(parts.train) + nrow(parts.validation) + nrow(parts.test) ==
          nrow(df)
end

@testitem "walk_forward_folds: expanding, no leakage" begin
    using MultiHubForecaster, DataFrames, Dates
    const M = MultiHubForecaster

    start = Date(2021, 1, 3)
    dates = start:Week(1):(start + Week(180))
    df = DataFrame(date = collect(dates), location = "a",
        value = 1.0:length(dates))
    split = M.date_split(df)

    folds = M.walk_forward_folds(df, split;
        step = Week(1), min_train = Week(52), horizon = 4)
    @test !isempty(folds)

    # Origins strictly increase and stay in the training span.
    origins = [f.origin_date for f in folds]
    @test issorted(origins)
    @test all(o -> o <= split.train_end, origins)

    # No leakage: every training row is strictly before its origin.
    for f in folds
        @test all(df.date[f.train_rows] .< f.origin_date)
    end
    # Expanding window: later folds have at least as many training rows.
    counts = [length(f.train_rows) for f in folds]
    @test issorted(counts)
end

@testitem "walk_forward_folds: as_of respects vintage" begin
    using MultiHubForecaster, DataFrames, Dates
    const M = MultiHubForecaster

    start = Date(2021, 1, 3)
    dates = collect(start:Week(1):(start + Week(160)))
    # Two vintages per observation date: a same-week snapshot and a late one.
    df = vcat(
        DataFrame(date = dates, as_of = dates, location = "a",
            value = 1.0),
        DataFrame(date = dates, as_of = dates .+ Week(6), location = "a",
            value = 2.0))
    split = M.date_split(df)
    folds = M.walk_forward_folds(df, split;
        min_train = Week(52), horizon = 2, as_of_col = :as_of)
    @test !isempty(folds)
    for f in folds
        @test all(df.date[f.train_rows] .< f.origin_date)
        @test all(df.as_of[f.train_rows] .<= f.origin_date)
    end
end

@testitem "run_backtest: tidy scores + Arrow round-trip" tags=[:sample] begin
    using MultiHubForecaster, DataFrames, Dates, Random, Statistics
    const M = MultiHubForecaster
    D = M.Distributions
    Random.seed!(2024)

    # Tiny synthetic hub: 2 locations, ~2.5 years of weekly counts.
    function series(n, level, seed)
        rng = MersenneTwister(seed)
        z = zeros(n)
        for t in 2:n
            z[t] = 0.5 * z[t - 1] + 0.15 * randn(rng)
        end
        [rand(rng, D.NegativeBinomial(15,
             15 / (15 + exp(level + 0.4 * sin(2π * t / 52) + z[t]))))
         for t in 1:n]
    end
    start = Date(2022, 1, 2)
    n = 200
    dts = collect(start:Week(1):(start + Week(n - 1)))
    df = vcat(
        DataFrame(date = dts, location = "a", value = series(n, 4.0, 1)),
        DataFrame(date = dts, location = "b", value = series(n, 3.5, 2)))

    # ~3.8 years of data leaves a training span (up to two years before the
    # present) with room for folds; take just the last two of them.
    present = maximum(df.date)
    split = M.date_split(df; present = present)
    folds = M.walk_forward_folds(df, split;
        step = Week(1), min_train = Week(30), horizon = 2)
    @test length(folds) >= 2
    folds = folds[(end - 1):end]

    model = M.Baseline(;
        target_type = :count, p = 1, n_harmonics = 1)
    result = M.run_backtest(model, df, folds;
        horizon = 2, target = "wk inc flu hosp", model_id = "baseline",
        adtype = M.ADTypes.AutoForwardDiff(), ndraws = 60, nchains = 2,
        n_samples = 60, rng = MersenneTwister(7))

    @test result isa M.BacktestResult
    sc = result.scores

    # Tidy schema: right columns, present and finite.
    for c in [:model, :origin_date, :location, :target, :horizon,
        :metric, :value]
        @test c in propertynames(sc)
    end
    @test nrow(sc) > 0
    @test all(isfinite, sc.value)
    @test Set(sc.model) == Set(["baseline"])
    @test Set(sc.target) == Set(["wk inc flu hosp"])
    @test Set(sc.horizon) ⊆ Set(1:2)
    @test Set(sc.location) ⊆ Set(["a", "b"])

    # Energy score and WIS are both present and non-negative.
    metrics = Set(sc.metric)
    @test "energy_score" in metrics
    @test "wis" in metrics
    es = sc[sc.metric .== "energy_score", :value]
    wis = sc[sc.metric .== "wis", :value]
    @test !isempty(es) && all(>=(0), es)
    @test !isempty(wis) && all(>=(0), wis)
    # WIS decomposition sums to WIS per cell.
    for g in groupby(sc, [:origin_date, :location, :horizon])
        val(m) = only(g[g.metric .== m, :value])
        @test val("wis_dispersion") + val("wis_overprediction") +
              val("wis_underprediction") ≈ val("wis") atol = 1e-8
    end

    # Summary carries headline metrics.
    @test result.summary["n_folds"] == length(folds)
    @test isfinite(result.summary["mean_energy_score"])

    # Arrow round-trip via save/load and the direct helpers.
    dir = mktempdir()
    M.save_experiment(dir, result)
    loaded = M.load_experiment(dir)
    @test nrow(loaded.scores) == nrow(sc)
    @test Set(propertynames(loaded.scores)) == Set(propertynames(sc))
    @test loaded.scores.value ≈ sc.value
    @test loaded.manifest["model_id"] == "baseline"
    @test nrow(loaded.forecasts) == nrow(result.forecasts)

    scpath = M.write_scores(joinpath(dir, "s2.arrow"), sc)
    @test nrow(M.read_scores(scpath)) == nrow(sc)
end
