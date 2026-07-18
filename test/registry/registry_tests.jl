@testitem "load_registry: parses the registry fixture" begin
    using MultiHubForecaster: load_registry, HubConfig

    path = joinpath(@__DIR__, "registry.toml")
    hubs = load_registry(path)
    @test hubs isa Vector{HubConfig}
    @test length(hubs) == 5

    ids = [h.id for h in hubs]
    @test "flusight" in ids
    @test "covidhub" in ids

    flusight = hubs[findfirst(h -> h.id == "flusight", hubs)]
    @test flusight.repo == "cdcepi/FluSight-forecast-hub"
    @test flusight.disease == "influenza"
    @test "wk inc flu hosp" in flusight.targets
    ours = ["wk inc flu hosp", "wk inc flu prop ed visits"]
    @test flusight.our_targets == ours
    @test flusight.output_types == ["quantile", "sample"]
    @test flusight.role == ["train", "submit"]
    @test !isempty(flusight.geography)
    @test !isempty(flusight.cadence)
end

@testitem "load_registry: error handling" begin
    using MultiHubForecaster: load_registry
    @test_throws ArgumentError load_registry(joinpath(@__DIR__, "nope.toml"))

    # A registry with no [[hub]] tables yields an empty vector.
    tmp = tempname() * ".toml"
    write(tmp, "title = \"empty\"\n")
    @test isempty(load_registry(tmp))
    rm(tmp; force = true)
end
