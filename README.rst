=====
ESPEI
=====

ESPEI, or Extensible Self-optimizing Phase Equilibria Infrastructure, is a tool for automated thermodynamic database development within the CALPHAD method.

The ESPEI package is based on a fork of `pycalphad-fitting`_ and uses `pycalphad`_ for calculating Gibbs free energies of thermodynamic models.
The implementation for ESPEI involves first fitting single-phase data by calculating parameters in thermodynamic models that are linearly described by the single-phase input data.
Then Markov Chain Monte Carlo (MCMC) is used to optimize the candidate models from the single-phase fitting to multi-phase zero-phase fraction data.
Single-phase and multi-phase fitting methods are described in Chapter 3 of `Richard Otis's thesis`_.

The benefit of this approach is the automated, simultaneous fitting for many parameters that yields uncertainty quantification, as shown in Otis and Liu High-Throughput Thermodynamic Modeling and Uncertainty Quantification for ICME. `Jom 69, (2017)`_.

The name and idea of ESPEI are originally based off of Shang, Wang, and Liu, ESPEI: Extensible, Self-optimizing Phase Equilibrium Infrastructure for Magnesium Alloys Magnes. Technol. 2010 617–622 (2010).  

.. _pycalphad-fitting: https://github.com/richardotis/pycalphad-fitting
.. _pycalphad: http://pycalphad.org
.. _Richard Otis's thesis: https://etda.libraries.psu.edu/catalog/s1784k73d
.. _Jom 69, (2017): http://dx.doi.org/10.1007/s11837-017-2318-6

License
=======

ESPEI is MIT licensed. See LICENSE.
