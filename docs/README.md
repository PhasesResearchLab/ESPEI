# espei.org Quarto site

## Building the docs
1. Install `quarto` from [quarto.org](https://quarto.org/docs/get-started/)
2. `pip install -r requirements.txt` (installs ESPEI and `quartodoc`)
3. `quartodoc build && quarto preview` to generate `reference/api` docs and preview the site

## TODO:
- Update png logos to the versions with transparency (for dark mode)
- Enable "tabbed" mode in the Cu-Mg tutorials for comparing command line vs. running interactively with Python code
- API docs automation:
  - Make API docs more automated in terms of discovery. Using `__all__` and/or pointing to modules with things well documented should simplify.
  - Make API docs more automated in terms of integrating the outputs of `quartodoc build` into the site (see the notes in `_quarto.yml`)
- Content to add:
  - CLI reference document
  - Need to document equilibrium thermochemical data
  - 2022 workshop Cr-Ni example (with tabbed mode for command line vs. Python)
  - 2023 workshop Cu-Mg example (update Cu-Mg example with UQ steps)
  - Explanation: parameter selection, incorporate details from the 2024 paper
  - Explanation: likelihood functions: incorporate details from dissertation, 2022 Calphad presentation and Courtney's 2024 paper (specifically on ZPF residual)
  - Explanation: MCMC how it works and propagation, specifically the ensemble method used (if we incorporate HMC/NUTS/(insert gradient based MCMC) soon, then we should just wait until that's done)
