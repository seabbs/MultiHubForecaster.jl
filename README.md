# MultiHubForecaster.jl <img src="docs/src/assets/logo.svg" width="150" alt="MultiHubForecaster logo" align="right">

<!-- badges:start -->
| **Documentation** | **Build Status** | **Code Quality** | **License & DOI** | **Downloads** |
|:-----------------:|:----------------:|:----------------:|:-----------------:|:-------------:|
| [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://epiaware.org/MultiHubForecaster.jl/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://epiaware.org/MultiHubForecaster.jl/dev/) | [![Test](https://github.com/seabbs/MultiHubForecaster.jl/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/seabbs/MultiHubForecaster.jl/actions/workflows/test.yaml) [![codecov](https://codecov.io/gh/seabbs/MultiHubForecaster.jl/graph/badge.svg)](https://codecov.io/gh/seabbs/MultiHubForecaster.jl) | [![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle) [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) [![JET](https://img.shields.io/badge/%E2%9C%88%EF%B8%8F%20tested%20with%20-%20JET.jl%20-%20red)](https://github.com/aviatesk/JET.jl) | [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) | [![Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FMultiHubForecaster&query=total_requests&label=Downloads)](https://juliapkgstats.com/pkg/MultiHubForecaster) [![Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Fmonthly_downloads%2FMultiHubForecaster&query=total_requests&suffix=%2Fmonth&label=Downloads)](https://juliapkgstats.com/pkg/MultiHubForecaster) |
<!-- badges:end -->

Julia forecasting engine for hubverse forecast hubs, trained jointly across
their data.
The package holds the final model and the reusable machinery: data ingest,
models, inference, hubverse-format I/O, and scoring.

Experiments, the local hub, the upstream-hub registry, and research notes live
in the companion repo `MultiHubForecastExperiments`, which links this package
as a submodule.

## Development

- `julia --project=. -e 'using Pkg; Pkg.instantiate()'`
- `julia --project=. -e 'using Pkg; Pkg.test()'`

<!-- standard-sections:start -->
<!-- MANAGED by EpiAwarePackageTools.scaffold — do not edit between the
     markers. These standard sections are re-rendered on every scaffold_update;
     edit the package-owned sections outside them, or CITATION.cff. -->

## Contributing

We welcome contributions and new contributors! Please open an issue or pull request on [GitHub](https://github.com/seabbs/MultiHubForecaster.jl). This package follows [ColPrac](https://github.com/SciML/ColPrac) and the [SciML style](https://github.com/SciML/SciMLStyle).

## How to cite

If you use MultiHubForecaster in your work, please cite it. Citation metadata lives in [`CITATION.cff`](https://github.com/seabbs/MultiHubForecaster.jl/blob/main/CITATION.cff), which GitHub renders as a "Cite this repository" button on the repository page.

## Code of conduct

Please note that the MultiHubForecaster project is released with a [Contributor Code of Conduct](https://github.com/seabbs/.github/blob/main/CODE_OF_CONDUCT.md). By contributing, you agree to abide by its terms.
<!-- standard-sections:end -->
