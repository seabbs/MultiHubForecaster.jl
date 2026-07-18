@testitem "order_hub_columns: ordering and model_id drop" begin
    using MultiHubForecaster: order_hub_columns, HUBVERSE_COLUMN_ORDER
    using DataFrames
    using Dates

    df = DataFrame(
        value = [1.0],
        model_id = ["team-model"],
        output_type_id = [0.5],
        location = ["US"],
        target = ["wk inc flu hosp"],
        output_type = ["quantile"],
        reference_date = [Date("2026-01-03")],
        horizon = [1],
        target_end_date = [Date("2026-01-10")]
    )
    out = order_hub_columns(df)
    @test !("model_id" in names(out))
    @test names(out) == string.([
        :reference_date, :target, :horizon, :target_end_date, :location,
        :output_type, :output_type_id, :value])

    # An unknown task-id column is kept, before the output columns.
    df2 = copy(df)
    df2.age_group = ["65+"]
    out2 = order_hub_columns(df2)
    @test "age_group" in names(out2)
    @test findfirst(==("age_group"), names(out2)) <
          findfirst(==("output_type"), names(out2))
    @test names(out2)[end] == "value"
end

@testitem "write_submission: round-trip CSV and parquet" begin
    using MultiHubForecaster: write_submission, HUBVERSE_COLUMN_ORDER
    using DataFrames, CSV, Parquet2
    using Dates

    ref = Date("2026-01-03")
    df = DataFrame(
        model_id = fill("team-model", 3),
        reference_date = fill(ref, 3),
        target = fill("wk inc flu hosp", 3),
        horizon = fill(1, 3),
        target_end_date = fill(Date("2026-01-10"), 3),
        location = fill("US", 3),
        output_type = fill("quantile", 3),
        output_type_id = [0.25, 0.5, 0.75],
        value = [10.0, 20.0, 30.0]
    )

    hub = mktempdir()
    results = write_submission(df, hub)
    @test length(results) == 1
    r = results[1]
    @test r.model_id == "team-model"

    csv_path = joinpath(hub, "model-output", "team-model",
        "2026-01-03-team-model.csv")
    pq_path = joinpath(hub, "model-output", "team-model",
        "2026-01-03-team-model.parquet")
    @test r.paths[:csv] == csv_path
    @test isfile(csv_path)
    @test isfile(pq_path)

    back = CSV.read(csv_path, DataFrame)
    @test !("model_id" in names(back))
    @test names(back) == string.([
        :reference_date, :target, :horizon, :target_end_date, :location,
        :output_type, :output_type_id, :value])
    @test back.value == df.value
    @test back.output_type_id == df.output_type_id

    pq = DataFrame(Parquet2.Dataset(pq_path))
    @test pq.value == df.value
    @test names(pq) == names(back)
end

@testitem "write_submission: splits by model and reference date" begin
    using MultiHubForecaster: write_submission
    using DataFrames
    using Dates

    df = DataFrame(
        model_id = ["a", "a", "b"],
        reference_date = [Date("2026-01-03"), Date("2026-01-10"),
            Date("2026-01-03")],
        target = fill("t", 3),
        horizon = fill(1, 3),
        target_end_date = fill(Date("2026-01-17"), 3),
        location = fill("US", 3),
        output_type = fill("quantile", 3),
        output_type_id = fill(0.5, 3),
        value = [1.0, 2.0, 3.0]
    )
    hub = mktempdir()
    results = write_submission(df, hub; formats = (:csv,))
    @test length(results) == 3
    @test isfile(joinpath(hub, "model-output", "a",
        "2026-01-03-a.csv"))
    @test isfile(joinpath(hub, "model-output", "a",
        "2026-01-10-a.csv"))
    @test isfile(joinpath(hub, "model-output", "b",
        "2026-01-03-b.csv"))
end

@testitem "write_submission: explicit model_id and dry_run" begin
    using MultiHubForecaster: write_submission
    using DataFrames
    using Dates

    df = DataFrame(
        reference_date = fill(Date("2026-01-03"), 2),
        target = fill("t", 2),
        horizon = fill(1, 2),
        target_end_date = fill(Date("2026-01-10"), 2),
        location = fill("US", 2),
        output_type = fill("quantile", 2),
        output_type_id = [0.5, 0.9],
        value = [1.0, 2.0]
    )
    hub = mktempdir()
    results = write_submission(df, hub; model_id = "my-model",
        dry_run = true)
    @test length(results) == 1
    @test results[1].model_id == "my-model"
    # dry_run writes nothing.
    @test !isdir(joinpath(hub, "model-output"))
end

@testitem "write_model_metadata: round-trip YAML" begin
    using MultiHubForecaster: write_model_metadata
    using YAML

    hub = mktempdir()
    path = write_model_metadata("team-model", hub,
        (; team_abbr = "team", model_abbr = "model",
            designated_model = true))
    @test isfile(path)
    @test path == joinpath(hub, "model-metadata", "team-model.yml")

    meta = YAML.load_file(path)
    @test meta["team_abbr"] == "team"
    @test meta["model_abbr"] == "model"
    @test meta["designated_model"] == true

    # A Dict argument works too.
    path2 = write_model_metadata("d", hub, Dict("team_abbr" => "x"))
    @test YAML.load_file(path2)["team_abbr"] == "x"
end
