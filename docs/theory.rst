.. raw:: latex

   \chapter{Theory}

.. _Theory:

======
Theory
======

ESPEI has two types of fitting -- parameter generation and MCMC
optimization. The parameter generation step uses experimental and DFT
data of the Gibbs free energy derivatives (:math:`C_P, H, S`) for each
phase and for the mixing energies within sublattices for each phase to
generate and fit parameters of given CALPHAD models. The MCMC
optimization step uses a Bayesian optimization procedure to fit
parameters in a Database to experimental phase equilibria.

Parameter generation
====================

A simple model with few parameters is better than a complex model that
describes the same data marginally better. Parameter generation in
ESPEI aims to achieve a balance of a simple parameterization and
goodness of fit in the Redlich-Kister polynomial used in CALPHAD
assessments. To achieve this, parameters are selected using the corrected
`Akaike information criterion <https://en.wikipedia.org/wiki/Akaike_information_criterion>`_
(AICc) to choose an optimal set of parameters from canditate
parameterizations.

The general Redlich Kister polynomial has the form :math:`G = a + bT +
cT\ln T + \sum_n d_n T^n`. Different parameterizations, e.g. only
considering :math:`a`, considering :math:`a` and :math:`b`, :math:`a`
and :math:`c`, etc. are fit to all of the input formation or mixing
data (depending on the parameter being selected) by a least squares
pseudo-inverse optimization.

Each parameterization is compared in the AICc and the most suitable
optimization balances the goodness of fit and the number of
parameters. The key aspect of this is that ESPEI will avoid
overfitting your data and will not add parameters you do not have data
for.

This is important for phases that would have a temperature dependent
contribution to the Gibbs energy, but the input data only gives 0K
formation energies. ESPEI cannot add temperature dependence to the
parameterized model. Because of this, an abundance of single-phase
data is critical to provide enough degrees of freedom in later
optimization.


MCMC optimization
=================

Details of Markov Chain Monte Carlo as an algorithm are better covered
elsewhere. A good example is MacKay's (free) book: `Information
Theory, Inference, and Learning Algorithms
<http://www.inference.org.uk/itprnn/book.pdf>`_.

Using MCMC for optimizing CALPHAD models might appear to have several
drawbacks. The parameters in the models are correlated and due to the
nature of single phase first-principles data the shape and size of the
posterior distribution for each parameter is not known before fitting.
Traditional Metropolis-Hastings MCMC algorithms require the a prior to
be defined for each parameter, which is a challenge for parameters in
CALPHAD models which vary over more than 6 orders of magnitude.

ESPEI solves these potential problems by using an Ensemble sampler, as
introduced by Goodman and Weare [1]_, rather than the
Metropolis-Hastings algorithm. Ensemble samplers have the property of
affine invariance, which uses multiple (:math:`\geq 2 N` for :math:`N`
parameters) parallel chains to scale new proposal parameters by linear
transforms. These chains, together an ensemble, define a proposal
distribution to sample parameters from that is scaled to the magnitude
and sensitivity of each parameter. Thus, Ensemble samplers directly
address the challenges we expect to encounter with traditional MCMC.

ESPEI uses an Ensemble sampler algorithm by using the `emcee
<http://dan.iel.fm/emcee/current/>`_ package that implements
parallelizable ensemble samplers. To use emcee, ESPEI defines the
initial ensemble of chains and a function that returns the error as a
log-probability. ESPEI defines the error as the mean square error
between experimental phase equilibria and the equilibria calculated by
the CALPHAD database.

Here, again, it is critical to point out the importance of abundant
phase equilibria data. Traditional CALPHAD modeling has involved the
modeler participating in tight feedback loops between updates to
parameters and the resulting phase diagram. ESPEI departs from this by
optimizing just a single scalar error function based on phase
equilibria. The implication of this is that if there are phase
equilibria that are observed to exist, but they are not in the
datasets that are considered by ESPEI, those equilibria cannot be
optimized against and may deviate from 'known' equilibria. A possible
approach to address this in ESPEI is to estimate the points for the
equilibria.

References
==========

.. [1] Goodman, J. & Weare, J. Ensemble samplers with affine invariance. Commun. Appl. Math. Comput. Sci. 5, 65–80 (2010). doi:`10.2140/camcos.2010.5.65 <https://doi.org/10.2140/camcos.2010.5.65>`_.
