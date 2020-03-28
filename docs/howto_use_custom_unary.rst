.. raw:: latex

   \chapter{Use a custom unary reference state}

.. _UseCustomUnary:

==================================
Use a custom unary reference state
==================================

By default, ESPEI uses the SGTE91

You may be interested in using custom unary reference states for parameter
generation in ESPEI if you are developing unary descriptions of the Gibbs
energy and/or lattice stability for pure elements. This is useful for any
unary function developed outside of ESPEI, available in the literature.

Instead of generating parameters with:

.. code-block:: yaml

    system:
      phase_models: my-phases.json
      datasets: my-input-datasets
    generate_parameters:
      excess_model: linear
      ref_state: SGTE91

You can create a small Python package that provides a reference state called
``MyCustomReferenceState``, which can be used as follows:

.. code-block:: yaml

    system:
      phase_models: my-phases.json
      datasets: my-input-datasets
    generate_parameters:
      excess_model: linear
      ref_state: MyCustomReferenceState

Here ``MyCustomReferenceState`` provides the Gibbs energy for the pure
elements, i.e. the ``GHSERXX`` function for some element ``XX``.
``MyCustomReferenceState`` can also provide lattice stabilities if the lattice
stability will not be fit using your input data.

.. _quickstart_unary_skeleton:

Quickstart: Skeleton package
============================

If you are not comfortable developing a Python package using the details
below, that's okay! We have provided a
`skeleton package <https://github.com/PhasesResearchLab/ESPEI-unary-refstate-skeleton>`_
that can be downloaded and installed to give you a working example.

Running the example
-------------------

Following these steps will give you a working unary reference state for Al and
Ag named ``CustomRefstate2020``. Starting from a command line, with ``git``
installed:

#. Clone the skeleton repository: ``git clone https://github.com/PhasesResearchLab/ESPEI-unary-refstate-skeleton``
#. Enter the downloaded repository: ``cd ESPEI-unary-refstate-skeleton``
#. Install the package using ``pip``: ``pip install -e .``

This will install the packaged, named ``espei_refstate_customrefstate2020``,
and provide a reference state named ``CustomRefstate2020``.

We can use that by passing using ``ref_state: CustomRefstate2020`` in the
``generate_parameters`` heading in :ref:`ESPEI's YAML input <input_yaml_generate_ref_state>`.
If you have ESPEI installed already, you can test that this works by:

#. Enter the ``espei-example`` directory: ``cd espei-example``
#. Run the YAML input file using ESPEI (note: it's safe to ignore a warning that no datsets were found - we aren't fitting any parameters to data here): ``espei --in gen_Ag-Al.yaml``

If it was successful, you should have ran the YAML file:

.. code-block:: yaml

   system:
     phase_models: Ag-Al.json
     datasets: input-datasets
   generate_parameters:
     excess_model: linear
     ref_state: CustomRefstate2020

and generated a database, ``out.tdb``, containing our custom ``GHSERAG`` function (among others):

.. code-block:: none

   FUNCTION GHSERAG 298.15 118.202013*T - 7209.512; 1234.93 Y 190.266404*T -
      15095.252; 3000.0 N !


and lattice stabilities for phases defined in the reference state and the system ``Ag-Al.json``, such as ``GHCPAG``.

Finally, since this reference state is probably not useful for developing any databases, uninstall the package by running ``pip uninstall espei_refstate_customrefstate2020`` and removing the directory ``espei_refstate_customrefstate2020.egg-info`` from the root directory if one exists.

Using the skeleton to create your own database
----------------------------------------------

If you want to use the skeleton to create your own reference state to provide
ESPEI, you can follow the steps below. To keep the steps concrete, we'll create
a reference state for Cu called ``Bocklund2019`` following the unary
description published for Cu in Bocklund *et al.* [#espei_paper]_. within the
segmented regression approach by Roslyakova
*et al.* [#segmented_regression_paper]_.


Assuming that you are fresh (without the skeleton downloaded yet):

#. Clone the skeleton repository: ``git clone https://github.com/PhasesResearchLab/ESPEI-unary-refstate-skeleton``
#. Enter the downloaded repository: ``cd ESPEI-unary-refstate-skeleton``
#. Update the ``NAME = 'CustomRefstate2020'`` parameter in ``setup.py`` to ``NAME = 'Bocklund2019'``
#. In the ``refstate.py`` module, create the ``Bockund2019Stable`` and ``Bockund2019`` dictionaries (see :ref:`creating_refstate_dicts` for more details)


   #. Delete the ``CustomRefstate2020Stable`` and ``CustomRefstate2020`` variables
   #. Add the stable phase Gibbs energy for Cu to the ``Bockund2019Stable`` variable

      .. code-block:: python

         Bocklund2019Stable = OrderedDict([
             ('CU',
             Piecewise((-0.0010514335*T**2 + 8.7685671186*T*log(exp(155.1404/T) - 1.0) + 16.1968683846*T*log(exp(290.9421/T) - 1.0) - 11038.0904080745, And(T >= 0.01, T < 103.57591)), (-2.15621953171362e-6*T**3 + 0.000288560900942072*T**2 - 0.13879113947248*T*log(T) + 8.7685671186*T*log(exp(155.1404/T) - 1.0) + 16.1968683846*T*log(exp(290.9421/T) - 1.0) + 0.574637617323048*T - 11042.8822142647, And(T >= 103.57591, T < 210.33309)), (-0.002432585*T**2 + 0.4335558862135*T*log(T) + 8.7685671186*T*log(exp(155.1404/T) - 1.0) + 16.1968683846*T*log(exp(290.9421/T) - 1.0) - 2.20049706600083*T - 11002.7543747764, And(T >= 210.33309, T < 1357.77)), (-31.38*T*log(T) + 183.555483717662*T - 12730.2995781851 + 7.42232714807953e+28/T**9, And(T >= 1357.77, T < 3200.0)), (0, True))),
         ])

   #. Add the lattice stability for all elements, including the stable phase, to the ``Bocklund2019`` variable

      .. code-block::

         Bocklund2019 = OrderedDict([
             (('CU', 'HCP_A3'), Piecewise((-3.38438862938597e-7*T**3 - 0.00121182291077191*T**2 + 8.7685671186*T*log(exp(155.1404/T) - 1.0) + 16.1968683846*T*log(exp(290.9421/T) - 1.0) - 0.321147237334052*T - 10441.4393392344, And(T >= 0.01, T < 298.15)), (1.29223e-7*T**3 - 0.00265684*T**2 - 24.112392*T*log(T) + 130.685235*T - 7170.458 + 52478/T, And(T >= 298.15, T < 1357.77)), (-31.38*T*log(T) + 184.003828*T - 12942.0252504739 + 3.64167e+29/T**9, And(T >= 1357.77, T < 3200.0)), (0, True))),
             (('CU', 'FCC_A1'), Piecewise((Symbol('GHSERCU'), And(T < 10000.0, T >= 1.0)))),
             (('CU', 'LIQUID'), Piecewise((-3.40056501515466e-7*T**3 - 0.00121066539331185*T**2 + 8.7685671186*T*log(exp(155.1404/T) - 1.0) + 16.1968683846*T*log(exp(290.9421/T) - 1.0) - 10.033338832193*T + 2379.36422209194, And(T >= 0.01, T < 298.15)), (-5.8489e-21*T**7 + 1.29223e-7*T**3 - 0.00265684*T**2 - 24.112392*T*log(T) + 120.973331*T + 5650.32106235287 + 52478/T, And(T >= 298.15, T < 1357.77)), (-31.38*T*log(T) + 173.881484*T + 409.498458129716, And(T >= 1357.77, T < 3200.0)), (0, True))),
         ])

#. Install the package using ``pip``: ``pip install -e .``
#. You can now use your reference state in ESPEI, and even change the definitions on the fly.



.. _creating_refstate_dicts:

Creating the reference state dictionaries
=========================================

To define the reference state and lattice stabilities, you must define two
ordered dictionaries, one ``<NAME>Stable`` and one ``<NAME>`` with the Gibbs
energies of the stable phase at 298.15 K and the lattice stabilities,
respectively. Note that ``OrderedDict`` is defined in the ``collections``
module in the Python standard library.

The Gibbs energy functions defined here must be defined as valid symbolic
expressions using SymPy ``Symbol`` objects and pycalphad ``StateVariable``
objects (e.g. ``pycalphad.variables.T``, ``pycalphad.variables.P``), which can
be (but are not required to be) piecewise in temperature. Any SymPy functions
can be used (``exp``, ``log``, ``Piecewise``, ...) and syntax/functions can be
used which are not available in commercial software (for example, direct
exponentiation to non-integer powers). Anything supported by pycalphad
``Model`` objects can be written, but note that the TDB objects that ESPEI
writes using these expressions may not be compatible with commercial software.

The ``<NAME>Stable`` dictionary defines the function corresponding to the
``GHSERXX`` function, you should interpret this function as defining a
``Symbol(GHSERXX)`` (a SymPy ``Symbol`` object). The ``<NAME>Stable``
dictionary directly maps pure element names to SymPy functions (note that
``OrderedDict`` syntax means construction a dict from a list of tuples).

The ``<NAME>`` dictionary maps tuples of ``("XX", "PHASE_NAME")`` to lattice
stability Gibbs energy functions, where ``XX`` is a pure element string. The
stable phase at 298.15 K should be defined by setting the energy to
``Symbol(GHSERXX)`` is the implictly defined ``GHSER`` function, again for
element ``XX``. The lattice stabilities, if desired, can be referenced to
the ``Symbol(GHSERXX)`` function, although they are not here except for the
stable ``FCC_A1``.


   .. code-block:: python

      from collections import OrderedDict
      from sympy import *
      from pycalphad.variables import P, T

      Bocklund2019Stable = OrderedDict([
          ('CU',
          Piecewise((-0.0010514335*T**2 + 8.7685671186*T*log(exp(155.1404/T) - 1.0) + 16.1968683846*T*log(exp(290.9421/T) - 1.0) - 11038.0904080745, And(T >= 0.01, T < 103.57591)), (-2.15621953171362e-6*T**3 + 0.000288560900942072*T**2 - 0.13879113947248*T*log(T) + 8.7685671186*T*log(exp(155.1404/T) - 1.0) + 16.1968683846*T*log(exp(290.9421/T) - 1.0) + 0.574637617323048*T - 11042.8822142647, And(T >= 103.57591, T < 210.33309)), (-0.002432585*T**2 + 0.4335558862135*T*log(T) + 8.7685671186*T*log(exp(155.1404/T) - 1.0) + 16.1968683846*T*log(exp(290.9421/T) - 1.0) - 2.20049706600083*T - 11002.7543747764, And(T >= 210.33309, T < 1357.77)), (-31.38*T*log(T) + 183.555483717662*T - 12730.2995781851 + 7.42232714807953e+28/T**9, And(T >= 1357.77, T < 3200.0)), (0, True))),
      ])

      Bocklund2019 = OrderedDict([
          (('CU', 'HCP_A3'), Piecewise((-3.38438862938597e-7*T**3 - 0.00121182291077191*T**2 + 8.7685671186*T*log(exp(155.1404/T) - 1.0) + 16.1968683846*T*log(exp(290.9421/T) - 1.0) - 0.321147237334052*T - 10441.4393392344, And(T >= 0.01, T < 298.15)), (1.29223e-7*T**3 - 0.00265684*T**2 - 24.112392*T*log(T) + 130.685235*T - 7170.458 + 52478/T, And(T >= 298.15, T < 1357.77)), (-31.38*T*log(T) + 184.003828*T - 12942.0252504739 + 3.64167e+29/T**9, And(T >= 1357.77, T < 3200.0)), (0, True))),
          (('CU', 'FCC_A1'), Piecewise((Symbol('GHSERCU'), And(T < 10000.0, T >= 1.0)))),
          (('CU', 'LIQUID'), Piecewise((-3.40056501515466e-7*T**3 - 0.00121066539331185*T**2 + 8.7685671186*T*log(exp(155.1404/T) - 1.0) + 16.1968683846*T*log(exp(290.9421/T) - 1.0) - 10.033338832193*T + 2379.36422209194, And(T >= 0.01, T < 298.15)), (-5.8489e-21*T**7 + 1.29223e-7*T**3 - 0.00265684*T**2 - 24.112392*T*log(T) + 120.973331*T + 5650.32106235287 + 52478/T, And(T >= 298.15, T < 1357.77)), (-31.38*T*log(T) + 173.881484*T + 409.498458129716, And(T >= 1357.77, T < 3200.0)), (0, True))),
      ])



Detailed Information
====================

Setting up setup.py
-------------------

If you're comfortable creaing your own package or want to go dig deeper into
how the skeleton works, ESPEI uses the
`entry_points <https://packaging.python.org/guides/creating-and-discovering-plugins/#using-package-metadata>`_
feature of ``setuptools`` to treat additional reference states as plugins.

A package providing a reference state to ESPEI should provide a module that has
two ``OrderedDict`` objects named ``<NAME>Stable`` and ``<NAME>``, according to
the :ref:`creating_refstate_dicts` section above. The module can have any name,
``<MODULE>``, (the skeleton has ``refstate.py``). ESPEI looks for the
``entry_point`` called ``espei.reference_states`` following the example from
the `setuptools documentation <https://setuptools.readthedocs.io/en/latest/setuptools.html#dynamic-discovery-of-services-and-plugins>`_.
Concretely, the ``entry_point`` should be described by:

.. code-block:: python

   # setup.py

   from setuptools import setup

   setup(# ...
       entry_points={'espei.reference_states': '<NAME> = <MODULE>'}
   )

where ``<NAME>`` and ``<MODULE>`` are replaced by the corresponding name of the
reference state and the name of the module with the reference states defined.

Interested readers may also find the `entry_points specification here <https://packaging.python.org/specifications/entry-points/>`_.

Debugging
---------

If you want to test whether your modules are found, you can run the following Python code to show what reference states were found

.. code-block:: python

   import espei
   print(espei.refdata.INSERTED_USER_REFERENCE_STATES)

If you do this after installing the unchanged
:ref:`skeleton package <quickstart_unary_skeleton>` package from this
repository, you should find CustomRefstate2020 is printed and the
dictionaries ``espei.refdata.CustomRefstate2020Stable`` and
``espei.refdata.CustomRefstate2020`` should be defined in the ``espei.refdata``
module. For more details on the implementation, see the
``espei.refdata.find_and_insert_user_refstate`` function.


References
==========

.. [#espei_paper] B. Bocklund *et al.*, MRS Communications 9(2) (2019) 1–10. doi:`10.1557/mrc.2019.59 <https://doi.org/10.1557/mrc.2019.59>`_
.. [#segmented_regression_paper] I. Roslyakova *et al.*, Calphad 55 (2016) 165–180. doi:`10.1016/j.calphad.2016.09.001 <https://doi.org/10.1016/j.calphad.2016.09.001>`_
