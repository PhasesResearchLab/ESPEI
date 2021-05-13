.. raw:: latex

   \chapter{Use a custom unary reference state}

.. _UseCustomUnary:

==================================
Use a custom unary reference state
==================================

Unary reference state data provides three pieces of information about pure elements to ESPEI:

#. Stable element reference (SER) data, giving the stable phase, atomic mass, :math:`H298` and :math:`S298` of each element
#. Gibbs energy of each element in its SER phase
#. Optional lattice stability functions for selected phases, often relative to the stable element reference energy

By default, ESPEI uses the SGTE91 reference state using the functions defined
in [Dinsdale1991]_.

ESPEI can be provided custom unary reference states for parameter generation.
Some common use cases for custom reference states are:

* Testing new unary descriptions in an assessment
* Using ESPEI to model systems with ficticious components, like a pseudo-binary

Using a custom reference state is very easy once it is created or installed.
Instead of generating parameters with:

.. code-block:: yaml

    system:
      phase_models: my-phases.json
      datasets: my-input-datasets
    generate_parameters:
      excess_model: linear
      ref_state: SGTE91

A custom reference state called ``MyCustomReferenceState`` could be used by:

.. code-block:: yaml

    system:
      phase_models: my-phases.json
      datasets: my-input-datasets
    generate_parameters:
      excess_model: linear
      ref_state: MyCustomReferenceState

Using the template package below, you can create a Python package of custom
reference data to install for yourself, and optionally to distribute to others.

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
Ag named ``CustomRefstate2020``. Assuming you have ``git`` and a working Python
environment, starting from the command line:

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

::

   FUNCTION GHSERAG 298.15 118.202013*T - 7209.512; 1234.93 Y 190.266404*T -
      15095.252; 3000.0 N !


and lattice stabilities for phases defined in the reference state and the system ``Ag-Al.json``, such as ``GHCPAG``.

Finally, since this reference state is probably not useful for developing any databases, uninstall the package by running ``pip uninstall espei_refstate_customrefstate2020`` and removing the directory ``espei_refstate_customrefstate2020.egg-info`` from the root directory if one exists.

Using the skeleton to create your own database
----------------------------------------------

If you want to use the skeleton to create your own reference state to provide
ESPEI, you can follow the steps below. To keep the steps concrete, we'll create
a reference state for Cu called ``Bocklund2019`` following the unary
description published for Cu in [Bocklund2019]_. within the
segmented regression approach by [Roslyakova2016]_.


Assuming that you are fresh (without the skeleton downloaded yet):

#. Clone the skeleton repository: ``git clone https://github.com/PhasesResearchLab/ESPEI-unary-refstate-skeleton``
#. Enter the downloaded repository: ``cd ESPEI-unary-refstate-skeleton``
#. Update the ``NAME = 'CustomRefstate2020'`` parameter in ``setup.py`` to ``NAME = 'Bocklund2019'``
#. In the ``refstate.py`` module, create the ``Bockund2019Stable``, ``Bockund2019``, and (optionally) ``Bocklund2019SER`` dictionaries (see :ref:`creating_refstate_dicts` for more details)


   #. Delete the ``CustomRefstate2020Stable`` and ``CustomRefstate2020`` variables
   #. Add the stable phase Gibbs energy for Cu to the ``Bockund2019Stable``
      variable. Note that ``OrderedDict`` is defined in the ``collections``
      module in the Python standard library.


      .. code-block:: python

         Bocklund2019Stable = OrderedDict([
             ('CU',
             Piecewise((-0.0010514335*T**2 + 8.7685671186*T*log(exp(155.1404/T) - 1.0) + 16.1968683846*T*log(exp(290.9421/T) - 1.0) - 11038.0904080745, And(T >= 0.01, T < 103.57591)), (-2.15621953171362e-6*T**3 + 0.000288560900942072*T**2 - 0.13879113947248*T*log(T) + 8.7685671186*T*log(exp(155.1404/T) - 1.0) + 16.1968683846*T*log(exp(290.9421/T) - 1.0) + 0.574637617323048*T - 11042.8822142647, And(T >= 103.57591, T < 210.33309)), (-0.002432585*T**2 + 0.4335558862135*T*log(T) + 8.7685671186*T*log(exp(155.1404/T) - 1.0) + 16.1968683846*T*log(exp(290.9421/T) - 1.0) - 2.20049706600083*T - 11002.7543747764, And(T >= 210.33309, T < 1357.77)), (-31.38*T*log(T) + 183.555483717662*T - 12730.2995781851 + 7.42232714807953e+28/T**9, And(T >= 1357.77, T < 3200.0)), (0, True))),
         ])

   #. Add the lattice stability for all elements, including the stable phase, to the ``Bocklund2019`` variable

      .. code-block::

         Bocklund2019 = OrderedDict([
             (('CU', 'HCP_A3'), Piecewise((-3.38438862938597e-7*T**3 - 0.00121182291077191*T**2 + 8.7685671186*T*log(exp(155.1404/T) - 1.0) + 16.1968683846*T*log(exp(290.9421/T) - 1.0) - 0.321147237334052*T - 10441.4393392344, And(T >= 0.01, T < 298.15)), (1.29223e-7*T**3 - 0.00265684*T**2 - 24.112392*T*log(T) + 130.685235*T - 7170.458 + 52478/T, And(T >= 298.15, T < 1357.77)), (-31.38*T*log(T) + 184.003828*T - 12942.0252504739 + 3.64167e+29/T**9, And(T >= 1357.77, T < 3200.0)), (0, True))),
             (('CU', 'FCC_A1'), Piecewise((Symbol('GHSERCU'), And(T < 10000.0, T >= 1.0)), (0, True))),
             (('CU', 'LIQUID'), Piecewise((-3.40056501515466e-7*T**3 - 0.00121066539331185*T**2 + 8.7685671186*T*log(exp(155.1404/T) - 1.0) + 16.1968683846*T*log(exp(290.9421/T) - 1.0) - 10.033338832193*T + 2379.36422209194, And(T >= 0.01, T < 298.15)), (-5.8489e-21*T**7 + 1.29223e-7*T**3 - 0.00265684*T**2 - 24.112392*T*log(T) + 120.973331*T + 5650.32106235287 + 52478/T, And(T >= 298.15, T < 1357.77)), (-31.38*T*log(T) + 173.881484*T + 409.498458129716, And(T >= 1357.77, T < 3200.0)), (0, True))),
         ])

   #. (Optional) Add the SER data. If you don't add this data, the SGTE91 data will be used as a fallback

      .. code-block:: python

         Bocklund2019SER = OrderedDict([
            ('CU', {'phase': 'FCC_A1', 'mass': 63.546, 'H298': 5004.1, 'S298': 33.15}),
         ])


#. Install the package as editable using ``pip``: ``pip install -e .``
#. You can now use your reference state in ESPEI, and even change the definitions on the fly.


.. _creating_refstate_dicts:

Creating the reference state dictionaries
=========================================

To define the reference Gibbs energy, lattice stabilities, and SER data you
must define three ordered dictionaries:

* ``<NAME>SER``, mapping element names to dictionaries of SER data
* ``<NAME>Stable``, mapping element names to Gibbs energy expressions for the stable phase
* ``<NAME>``, mapping pairs of ``(element name, phase name)`` to lattice stability expressions

The Gibbs energy expressions must be defined as valid symbolic expressions
using SymPy ``Symbol`` objects and pycalphad ``StateVariable`` objects (e.g.
``v.T``, ``v.P``), which can be (but are not required to be) piecewise in
temperature. Any SymPy functions can be used (``exp``, ``log``, ``Piecewise``,
...). Any valid Python syntax or functions can be used, including those not
available in commercial software (for example, direct exponentiation with
non-integer powers). Any expression supported by pycalphad ``Model`` objects
can be used, but note that the TDB files that ESPEI writes using these
expressions may not be compatible with commercial software.

It's important to note that the users probably want to add a ``(0, True)``
expression/condition pair to the end of any Piecewise expressions used. Since
pycalphad does not automatically extrapolate the piecewise expressions outside
of thier valid ranges, this condition will allow the solver to be numerically
stable, returning zero instead of `NaN`.

For ``<NAME>`` lattice stability data, all `GHSER` symbols will have a two
letter element name, regardless of how many letters the element name has. This
is to prevent abbreviation name clashes in commercial software. For example,
`GHSERC` could represent the Gibbs energy for carbon (`C`), but also be a
valid abbreviation for calcium (`CA`). Using `GHSERCC` automatically fixes this
issue, but be aware to use `Symbol("GHSERCC")` in the case of single letter
phase names.


Detailed Information
====================

Setting up setup.py
-------------------

If you want to go dig deeper into how the skeleton works, ESPEI uses the
`entry_points <https://packaging.python.org/guides/creating-and-discovering-plugins/#using-package-metadata>`_
feature of ``setuptools`` to treat additional reference states as plugins.

A package providing a reference state to ESPEI should provide a module that has
three  dictionaries: ``<NAME>Stable``, ``<NAME>``, and (optional) ``<NAME>SER``,
according to the :ref:`creating_refstate_dicts` section above. The module can
have any name, ``<MODULE>``, (the skeleton uses ``refstate.py``). ESPEI looks
for the ``entry_point`` called ``espei.reference_states`` following the example
from the `setuptools documentation <https://setuptools.readthedocs.io/en/latest/setuptools.html#dynamic-discovery-of-services-and-plugins>`_.
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
