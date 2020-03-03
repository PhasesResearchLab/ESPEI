.. ESPEI documentation master file, created by
   sphinx-quickstart on Sat Jun 24 22:30:49 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |logo| image:: _static/ESPEI-logo.png
          :height: 30pt
          :width: 30pt
          :alt: Logo

============
|logo| ESPEI
============

.. raw:: latex

   \part{Introduction}

.. toctree::
   :maxdepth: 1

   self
   installation
   quickstart

ESPEI, or Extensible Self-optimizing Phase Equilibria Infrastructure, is a tool for thermodynamic database development within the CALPHAD method. It uses `pycalphad`_ for calculating Gibbs free energies of thermodynamic models.


.. admonition:: Upcoming Workshops

    * May 24, 2020 - `CALPHAD XLIX <https://calphad.org/calphad-2020>`_: A pre-conference workshop on using `pyiron <https://pyiron.github.io/index.html>`_, `pycalphad`_ and ESPEI will be given at the 2020 CALPHAD meeting.
    * September 16, 2020 - `IMAT 2020 <https://www.asminternational.org/web/imat>`_: A workshop on using `pycalphad`_ and ESPEI will be given at the IMAT 2020 meeting as part of a one day workshop series on Python-based tools for materials modeling.


What is ESPEI?
--------------

1. ESPEI parameterizes CALPHAD models with enthalpy, entropy, and heat capacity data using the corrected Akiake Information Criterion (AICc). This parameter generation step augments the CALPHAD modeler by providing tools for data-driven model selection, rather than relying on a modeler's intuition alone.
2. ESPEI optimizes CALPHAD model parameters to thermochemical and phase boundary data and quantifies the uncertainty of the model parameters using Markov Chain Monte Carlo (MCMC). This is similar to the PARROT module of Thermo-Calc, but goes beyond by adjusting all parameters simultaneously and evaluating parameter uncertainty.

Details on the implmementation of ESPEI can be found in the publication: B. Bocklund *et al.*, MRS Communications 9(2) (2019) 1–10. doi:`10.1557/mrc.2019.59 <https://doi.org/10.1557/mrc.2019.59>`_.

What ESPEI can do?
------------------

ESPEI can be used to generate model parameters for CALPHAD models of the Gibbs energy that follow the temperature-dependent polynomial by Dinsdale (CALPHAD 15(4) 1991 317-425) within the compound energy formalism (CEF) for endmembers and Redlich-Kister-Mugganu excess mixing parameters in unary, binary and ternary systems.

All thermodynamic quantities are computed by pycalphad. The MCMC-based Bayesian parameter estimation can optimize data for any model that is supported by pycalphad, including models beyond the endmember Gibbs energies Redlich-Kister-Mugganiu excess terms, such as parameters in the ionic liquid model, magnetic, or two-state models. Performing Bayesian parameter estimation for arbitrary multicomponent thermodynamic data is supported.


Goals
-----

1. Offer a free and open-source tool for users to develop multicomponent databases with quantified uncertainty
2. Enable development of CALPHAD-type models for Gibbs energy, thermodynamic or kinetic properties
3. Provide a platform to build and apply novel model selection, optimization, and uncertainty quantification methods

The implementation for ESPEI involves first performing parameter generation by calculating parameters in thermodynamic models that are linearly described by :ref:`non-equilibrium thermochemical data <non_equilibrium_thermochemical_data>`.
Then Markov Chain Monte Carlo (MCMC) is used to optimize the candidate models from the parameter generation to :ref:`phase boundary data <phase_boundary_data>`.


.. figure:: _static/cu-mg-mcmc-phase-diagram.png
    :alt: Cu-Mg phase diagram
    :scale: 100%

    Cu-Mg phase diagram from a database created with and optimized by ESPEI.
    See the :ref:`Cu-Mg Example`.


History
-------

The ESPEI package is based on a fork of `pycalphad-fitting`_. The name and idea of ESPEI are originally based off of Shang, Wang, and Liu, ESPEI: Extensible, Self-optimizing Phase Equilibrium Infrastructure for Magnesium Alloys `Magnes. Technol. 2010 617-622 (2010)`_.

Implementation details for ESPEI have been described in the following publications:

- B. Bocklund *et al.*, MRS Communications 9(2) (2019) 1–10. doi:`10.1557/mrc.2019.59 <https://doi.org/10.1557/mrc.2019.59>`_
- R. Otis *et al.*, JOM 69 (2017) doi:`10.1007/s11837-017-2318-6 <http://doi.org/10.1007/s11837-017-2318-6>`_
- Richard Otis's `thesis <https://etda.libraries.psu.edu/catalog/s1784k73d>`_


Change log
----------

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: What's new

   CHANGES

See `what's new <CHANGES.html>`_!


.. raw:: latex

   \part{Tutorials}

.. toctree::
   :maxdepth: 1
   :caption: Tutorials


   cu-mg-example

.. raw:: latex

   \part{How-to}

.. toctree::
   :maxdepth: 2
   :caption: How-to

   writing_input
   specifying_priors
   input_data
   recipes


.. raw:: latex

   \part{Reference}

.. toctree::
   :maxdepth: 1
   :caption: Reference

   advanced_schedulers
   api/modules


.. raw:: latex

   \part{Developer}

.. toctree::
   :maxdepth: 1
   :caption: Developer

   contributing
   design


.. raw:: latex

   \part{Appendix}

.. raw:: latex

   \chapter{Appendices}

Getting Help
============

For help on installing and using ESPEI, please join the `PhasesResearchLab/ESPEI Gitter room <https://gitter.im/PhasesResearchLab/ESPEI>`_.

Bugs and software issues should be reported on `GitHub <https://github.com/PhasesResearchLab/ESPEI/issues>`_.


License
=======

ESPEI is MIT licensed.

.. code-block:: none

   The MIT License (MIT)

   Copyright (c) 2015-2018 Richard Otis
   Copyright (c) 2017-2018 Brandon Bocklund
   Copyright (c) 2018-2019 Materials Genome Foundation

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.


Citing ESPEI
============

If you use ESPEI for work presented in a publication, we ask that you cite the following publication:

B. Bocklund, R. Otis, A. Egorov, A. Obaied, I. Roslyakova, Z.-K. Liu, ESPEI for efficient thermodynamic database development, modification, and uncertainty quantification: application to Cu–Mg, MRS Commun. (2019) 1–10. doi:`10.1557/mrc.2019.59 <https://doi.org/10.1557/mrc.2019.59>`_.

.. code-block:: none

   @article{Bocklund2019ESPEI,
            archivePrefix = {arXiv},
            arxivId = {1902.01269},
            author = {Bocklund, Brandon and Otis, Richard and Egorov, Aleksei and Obaied, Abdulmonem and Roslyakova, Irina and Liu, Zi-Kui},
            doi = {10.1557/mrc.2019.59},
            eprint = {1902.01269},
            issn = {2159-6859},
            journal = {MRS Communications},
            month = {jun},
            pages = {1--10},
            title = {{ESPEI for efficient thermodynamic database development, modification, and uncertainty quantification: application to Cu–Mg}},
            year = {2019}
   }


.. only:: html

   Indices and tables
   ==================

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`

.. _pycalphad-fitting: https://github.com/richardotis/pycalphad-fitting
.. _pycalphad: http://pycalphad.org
.. _Richard Otis's thesis: https://etda.libraries.psu.edu/catalog/s1784k73d
.. _Jom 69, (2017): http://dx.doi.org/10.1007/s11837-017-2318-6
.. _Magnes. Technol. 2010 617-622 (2010): http://www.phases.psu.edu/wp-content/uploads/2010-Shang-Shunli-MagTech-ESPEI-0617-1.pdf
