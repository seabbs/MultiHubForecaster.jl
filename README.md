# MultiHubForecaster.jl <img src="docs/src/assets/logo.svg" width="150" alt="MultiHubForecaster logo" align="right">

<!-- badges:start -->
| **Documentation** | **Build Status** | **Code Quality** | **License & DOI** | **Downloads** |
|:-----------------:|:----------------:|:----------------:|:-----------------:|:-------------:|
| [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://seabbs.github.io/MultiHubForecaster.jl/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://seabbs.github.io/MultiHubForecaster.jl/dev/) | [![Test](https://github.com/seabbs/MultiHubForecaster.jl/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/seabbs/MultiHubForecaster.jl/actions/workflows/test.yaml) [![codecov](https://codecov.io/gh/seabbs/MultiHubForecaster.jl/graph/badge.svg)](https://codecov.io/gh/seabbs/MultiHubForecaster.jl) [![AD](https://github.com/seabbs/MultiHubForecaster.jl/actions/workflows/ad.yaml/badge.svg?branch=main)](https://github.com/seabbs/MultiHubForecaster.jl/actions/workflows/ad.yaml) | [![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle) [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) [![JET](https://img.shields.io/badge/%E2%9C%88%EF%B8%8F%20tested%20with%20-%20JET.jl%20-%20red)](https://github.com/aviatesk/JET.jl) | [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) | [![Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FMultiHubForecaster&query=total_requests&label=Downloads)](https://juliapkgstats.com/pkg/MultiHubForecaster) [![Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Fmonthly_downloads%2FMultiHubForecaster&query=total_requests&suffix=%2Fmonth&label=Downloads)](https://juliapkgstats.com/pkg/MultiHubForecaster) |

| ForwardDiff | ReverseDiff (tape) | Enzyme forward | Enzyme reverse | Mooncake reverse | Mooncake forward |
|:---:|:---:|:---:|:---:|:---:|:---:|
| [![cov ForwardDiff](https://codecov.io/gh/seabbs/MultiHubForecaster.jl/graph/badge.svg?flag=ad-forwarddiff)](https://app.codecov.io/gh/seabbs/MultiHubForecaster.jl?flags%5B0%5D=ad-forwarddiff) | [![cov ReverseDiff](https://codecov.io/gh/seabbs/MultiHubForecaster.jl/graph/badge.svg?flag=ad-reversediff)](https://app.codecov.io/gh/seabbs/MultiHubForecaster.jl?flags%5B0%5D=ad-reversediff) | [![cov Enzyme forward](https://codecov.io/gh/seabbs/MultiHubForecaster.jl/graph/badge.svg?flag=ad-enzyme-forward)](https://app.codecov.io/gh/seabbs/MultiHubForecaster.jl?flags%5B0%5D=ad-enzyme-forward) | [![cov Enzyme reverse](https://codecov.io/gh/seabbs/MultiHubForecaster.jl/graph/badge.svg?flag=ad-enzyme-reverse)](https://app.codecov.io/gh/seabbs/MultiHubForecaster.jl?flags%5B0%5D=ad-enzyme-reverse) | [![cov Mooncake reverse](https://codecov.io/gh/seabbs/MultiHubForecaster.jl/graph/badge.svg?flag=ad-mooncake-reverse)](https://app.codecov.io/gh/seabbs/MultiHubForecaster.jl?flags%5B0%5D=ad-mooncake-reverse) | [![cov Mooncake forward](https://codecov.io/gh/seabbs/MultiHubForecaster.jl/graph/badge.svg?flag=ad-mooncake-forward)](https://app.codecov.io/gh/seabbs/MultiHubForecaster.jl?flags%5B0%5D=ad-mooncake-forward) |
<!-- badges:end -->

Julia forecasting engine for hubverse forecast hubs, trained jointly across
their data.
The package holds the final model and the reusable machinery: data ingest,
models, inference, hubverse-format I/O, and scoring.

Experiments, the local hub, the upstream-hub registry, and research notes live
in the companion repo `MultiHubForecastExperiments`, which links this package
as a submodule.

## Overview

The package provides reusable tooling for working with hubverse forecast hubs,
independent of any particular model:

- hubverse submission I/O: write a forecast table to
  `model-output/<model>/<ref_date>-<model>.csv` (and `.parquet`) plus
  `model-metadata/<model>.yml`, in hubverse column order.
- scoring: the weighted interval score with its decomposition, central
  interval coverage, and the multivariate energy score for sample forecasts.
- a registry loader that parses a hubverse-hub registry TOML into typed hub
  configs.
- a hub-validation wrapper that runs the R `hubValidations` package.
- an abstract forecasting-model interface (`AbstractForecastModel`, `fit`,
  `forecast`); no concrete model ships here.

## Getting started

```julia
using MultiHubForecaster

# Score a quantile forecast.
levels = [0.025, 0.25, 0.5, 0.75, 0.975]
values = [1.0, 2.0, 3.0, 4.0, 5.0]
weighted_interval_score(3.0, values, levels)

# Write a submission into a hubverse-hub clone.
write_submission(forecast_table, hub_path)
```

- `julia --project=. -e 'using Pkg; Pkg.instantiate()'`
- `julia --project=. -e 'using Pkg; Pkg.test()'`

## Documentation

Rendered documentation is published at
[seabbs.github.io/MultiHubForecaster.jl](https://seabbs.github.io/MultiHubForecaster.jl/stable/),
with the public API reference and a getting-started guide.

<!-- standard-sections:start -->
<!-- MANAGED by EpiAwarePackageTools.scaffold — do not edit between the
     markers. These standard sections are re-rendered on every scaffold_update;
     edit the package-owned sections outside them, or CITATION.cff. -->

## Contributing

We welcome contributions and new contributors! Please open an issue or pull request on [GitHub](https://github.com/seabbs/MultiHubForecaster.jl). This package follows [ColPrac](https://github.com/SciML/ColPrac) and the [SciML style](https://github.com/SciML/SciMLStyle).

## How to cite

If you use MultiHubForecaster in your work, please cite it. Citation metadata lives in [`CITATION.cff`](https://github.com/seabbs/MultiHubForecaster.jl/blob/main/CITATION.cff), which GitHub renders as a "Cite this repository" button on the repository page.

## Code of conduct

Please note that the MultiHubForecaster project is released with a [Contributor Code of Conduct](https://github.com/EpiAware/.github/blob/main/CODE_OF_CONDUCT.md). By contributing, you agree to abide by its terms.
<!-- standard-sections:end -->
