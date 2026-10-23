# ESPEI documentation

The ESPEI documentation is a [Quarto](https://quarto.org) website. See the
[contributing guide](developer/contributing.qmd) for details on building
and deploying it.

## Building the docs

1. Install `quarto` from [quarto.org](https://quarto.org/docs/get-started/)
2. `uv sync --dev` from the root of the repository (installs ESPEI, `quartodoc` and `jupyter`)
3. `uv run quartodoc build && uv run quarto preview` from this directory to generate the `reference/api` docs and preview the site

## TODO:
- Update png logos to the versions with transparency (for dark mode)
- Enable "tabbed" mode in the Cu-Mg tutorials for comparing command line vs. running interactively with Python code
- API docs automation: make API docs more automated in terms of discovery. Using `__all__` and/or pointing to modules with things well documented should simplify.
- Content to add:
  - CLI reference document
  - Need to document equilibrium thermochemical data
  - 2022 workshop Cr-Ni example (with tabbed mode for command line vs. Python)
  - 2023 workshop Cu-Mg example (update Cu-Mg example with UQ steps)
  - Explanation: parameter selection, incorporate details from the 2024 paper
  - Explanation: likelihood functions: incorporate details from dissertation, 2022 Calphad presentation and Courtney's 2024 paper (specifically on ZPF residual)
  - Explanation: MCMC how it works and propagation, specifically the ensemble method used (if we incorporate HMC/NUTS/(insert gradient based MCMC) soon, then we should just wait until that's done)
