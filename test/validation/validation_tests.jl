@testitem "validate_submission: runs or degrades gracefully" begin
    using MultiHubForecaster: validate_submission, HubValidationResult

    hub = mktempdir()
    sub = joinpath(hub, "submission.csv")
    write(sub, "reference_date,value\n2026-01-03,1.0\n")

    # Whether or not the R toolchain is present, this returns a structured
    # result rather than raising. A bogus hub directory can never validate, so
    # `passed` is false, and there is always at least one message.
    res = validate_submission(hub, sub)
    @test res isa HubValidationResult
    @test res.passed == false
    @test !isempty(res.messages)
end

@testitem "validate_submission: unavailable when Rscript is absent" begin
    using MultiHubForecaster: validate_submission, HubValidationResult

    hub = mktempdir()
    sub = joinpath(hub, "submission.csv")
    write(sub, "x\n1\n")

    # Force the "no R toolchain" path by hiding Rscript from PATH.
    old_path = get(ENV, "PATH", "")
    res = try
        ENV["PATH"] = ""
        validate_submission(hub, sub)
    finally
        ENV["PATH"] = old_path
    end
    @test res isa HubValidationResult
    @test res.available == false
    @test res.passed == false
    @test !isempty(res.messages)
    @test occursin("Rscript", first(res.messages))
end
