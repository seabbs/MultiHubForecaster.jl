# GitHub Pages and CI secrets setup

This file documents the one-off repository-admin steps that `seabbs-bot`
cannot perform (they need repo-owner/admin rights).
The in-repo CI fixes on `feat/ci-fix` are done; these steps finish the job.

## Background

`MultiHubForecaster.jl` is a `seabbs`-owned package that reuses the shared
EpiAware CI infrastructure.
The managed workflows call reusable workflows that live only in
`EpiAware/.github` (there is no `seabbs/.github` repo), and the docs are
served from GitHub Pages at `https://seabbs.github.io/MultiHubForecaster.jl/`.

The `feat/ci-fix` branch repointed every reusable-workflow reference, the
docs kit source, and the docs URLs (see the branch diff).
What remains needs admin access.

## 1. Enable GitHub Pages (required for docs to publish)

The reusable `documentation.yml` deploys with `julia-actions/julia-docdeploy`,
which pushes the built site to a `gh-pages` branch (Documenter's classic
branch deploy). It does **not** use the newer Pages-artifact flow.

Order matters — the `gh-pages` branch is created by the first successful docs
deploy, and only then can Pages be pointed at it:

1. Make sure the `Documenter` workflow has run once on `main` and succeeded
   (it will, once this branch merges). That run creates the `gh-pages`
   branch.
2. Repo Settings -> Pages:
   - Source: **Deploy from a branch**
   - Branch: **`gh-pages`** / **`/ (root)`**
   - Leave the custom-domain field **empty** (the site is served at the
     default `seabbs.github.io/MultiHubForecaster.jl/`; the docs are built
     with `deploy_url = nothing`, so DocumenterVitepress derives the base
     path `/MultiHubForecaster.jl/` from the repo name — a custom domain here
     would break asset paths).

Do **not** choose "GitHub Actions" as the Pages source: this repo deploys via
the `gh-pages` branch, not the `actions/deploy-pages` artifact upload.

## 2. `DOCUMENTER_KEY` deploy secret (recommended)

`julia-docdeploy` authenticates the `gh-pages` push with `DOCUMENTER_KEY`
(an SSH deploy key) if present, otherwise it falls back to `GITHUB_TOKEN`.
The SSH key is the robust path and avoids Pages-rebuild edge cases.

From a checkout of this repo, in Julia:

```julia
using DocumenterTools
DocumenterTools.genkeys(user="seabbs", repo="MultiHubForecaster.jl")
```

Then follow its printed instructions:

- Add the **public** key as a repo **Deploy key** (Settings -> Deploy keys ->
  Add deploy key), name it `documenter`, and tick **Allow write access**.
- Add the **private** key as an **Actions secret** (Settings -> Secrets and
  variables -> Actions -> New repository secret) named **`DOCUMENTER_KEY`**.

The `document.yaml` caller already forwards secrets with `secrets: inherit`,
so no workflow change is needed once the secret exists.

## 3. Workflow permissions

Repo Settings -> Actions -> General -> Workflow permissions:

- Select **Read and write permissions** (lets `GITHUB_TOKEN` push `gh-pages`
  when `DOCUMENTER_KEY` is absent, and lets the coverage/version jobs write).
- Tick **Allow GitHub Actions to create and approve pull requests** (needed by
  the Auto Version Increment and Template sync workflows, which open PRs).

## 4. Codecov (only if coverage upload is not already working)

The coverage and AD workflows upload to Codecov. Public repos usually work
tokenless, but if uploads are rejected, add a `CODECOV_TOKEN` Actions secret
from the repo's Codecov settings (`app.codecov.io/gh/seabbs/MultiHubForecaster.jl`).

## Notes / follow-ups (not blocking)

- **Auto Version Increment** was failing on every push. Its failing step is a
  composite action, not a reusable-org caller, so the ref fix does not touch
  it; it is most likely resolved by step 3 (PR-create permission). Confirm
  after merge and, if it still fails, read its `--log-failed` for the real
  step.
- **Durability caveat (kit limitation).** The EpiAwarePackageTools scaffold
  couples one `org` input to three things at once: the package identity
  (repo/badges/reviewer), the reusable-workflow host, and the docs apex
  domain (`DOCS_PAGES_APEX = "epiaware.org"`, hardcoded). It cannot express
  "EpiAware infra + seabbs identity + seabbs.github.io docs", so the fixes on
  this branch were made by hand on the managed files. The weekly
  `Template sync` workflow re-runs `scaffold_update(".")` and would revert
  them (reusable refs back to `seabbs/.github`, docs badge back to
  `epiaware.org`). Options: (a) disable the `Template sync` schedule for this
  repo, or (b) teach the kit to take a separate reusable-workflow-host / docs
  apex, or create a `seabbs/.github`. Until then, re-apply this branch's edits
  after any template sync.
