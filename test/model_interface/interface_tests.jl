@testitem "AbstractForecastModel: interface contract" begin
    using MultiHubForecaster: AbstractForecastModel, fit, forecast

    @test AbstractForecastModel isa Type
    @test isabstracttype(AbstractForecastModel)
    @test fit isa Function
    @test forecast isa Function

    # No concrete model ships with the package: the generics have no methods.
    @test isempty(methods(fit))
    @test isempty(methods(forecast))

    # A downstream model can subtype and add methods against the contract.
    struct DummyModel <: AbstractForecastModel end
    MultiHubForecaster.fit(::DummyModel, data; k = 1) = (fitted = data, k = k)
    MultiHubForecaster.forecast(fitted; horizon = 1) = fitted.fitted .+ horizon

    f = fit(DummyModel(), [1, 2, 3]; k = 2)
    @test f.k == 2
    @test forecast(f; horizon = 10) == [11, 12, 13]
end
