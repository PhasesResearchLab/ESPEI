.. _Input Files:

===================
Writing input files
===================

JSON Format
===========

ESPEI has a single input style in JSON format that is used for all data entry.
Single-phase and multi-phase input files are almost identical, but detailed descriptions and key differences can be found in the following sections.
For those unfamiliar with JSON, it is fairly similar to Python dictionaries with some rigid requirements

	•	All string quotes must be double quotes. Use ``"key"`` instead of ``'key'``.
	•	Numbers should not have leading zeros. ``00.123`` should be ``0.123`` and ``012.34`` must be ``12.34``.
	•	Lists and nested key-value pairs cannot have trailing commas. ``{"nums": [1,2,3,],}`` is invalid and should be ``{"nums": [1,2,3]}``.

These errors can be challenging to track down, particularly if you are only reading the JSON error messages in Python. 
A visual editor is encouraged for debugging JSON files such as `JSONLint`_.
A quick reference to the format can be found at `Learn JSON in Y minutes <https://learnxinyminutes.com/docs/json/>`_.

ESPEI has support for checking all of your input datasets for errors, which you should always use before you attempt to run ESPEI.
This error checking will report all of the errors at once and all errors should be fixed.
Errors in the datasets will prevent fitting.
To check the datasets at path ``my-input-data/`` you can run ``espei --check-datasets my-input-data``.

.. _JSONLint: https://jsonlint.com


Phase Descriptions
==================

The JSON file for describing CALPHAD phases is conceptually similar to a setup file in Thermo-Calc's PARROT module.
At the top of the file there is the ``refdata`` key that describes which reference state you would like to choose.
Currently the reference states are strings referring to dictionaries in ``pycalphad.refdata`` only ``"SGTE91"`` is implemented.

Each phase is described with the phase name as they key in the dictionary of phases.
The details of that phase is a dictionary of values for that key.
There are 4 possible entries to describe a phase: ``sublattice_model``, ``sublattice_site_ratios``, ``equivalent_sublattices``, and ``aliases``.
``sublattice_model`` is a list of lists, where each internal list contains all of the components in that sublattice.
The ``BCC_B2`` sublattice model is  ``[["AL", "NI", "VA"], ["AL", "NI", "VA"], ["VA"]]``, thus there are three sublattices where the first two have Al, Ni, and vacancies.
``sublattice_site_ratios`` should be of the same length as the sublattice model (e.g. 3 for ``BCC_B2``).
The sublattice site ratios can be fractional or integers and do not have to sum to unity.

The optional ``equivalent_sublattices`` key is a list of lists that describe which sublattices are symmetrically equivalent.
Each sub-list in ``equivalent_sublattices`` describes the indices (zero-indexed) of sublattices that are equivalent.
For ``BCC_B2`` the equivalent sublattices are ``[[0, 1]]``, meaning that the sublattice at index 0 and index 1 are equivalent.
There can be multiple different sets (multiple sub-lists) of equivalent sublattices and there can be many equivalent sublattices within a sublattice (see ``FCC_L12``).
If no ``equivalent_sublattice`` key exists, it is assumed that there are none.a

Finally, the ``aliases`` key is used to refer to other phases that this sublattice model can describe when symmetry is accounted for.
Aliases are used here to describe the ``BCC_A2`` and ``FCC_A1``, which are the disordered phases of ``BCC_B2`` and ``FCC_L12``, respectively.
Notice that the aliased phases are not otherwise described in the input file.
Multiple phases can exist with aliases to the same phase, e.g. ``FCC_L12`` and ``FCC_L10`` can both have ``FCC_A1`` as an alias.

.. code-block:: JSON

    {
      "refdata": "SGTE91",
      "components": ["AL", "NI", "VA"],
      "phases": {
          "LIQUID" : {
          "sublattice_model": [["AL", "NI"]],
          "sublattice_site_ratios": [1]
          },
          "BCC_B2": {
          "aliases": ["BCC_A2"],
          "sublattice_model": [["AL", "NI", "VA"], ["AL", "NI", "VA"], ["VA"]],
          "sublattice_site_ratios": [0.5, 0.5, 1],
          "equivalent_sublattices": [[0, 1]]
          },
          "FCC_L12": {
                "aliases": ["FCC_A1"],
          "sublattice_model": [["AL", "NI"], ["AL", "NI"], ["AL", "NI"], ["AL", "NI"], ["VA"]],
          "sublattice_site_ratios": [0.25, 0.25, 0.25, 0.25, 1],
          "equivalent_sublattices": [[0, 1, 2, 3]]
          },
          "AL3NI1": {
          "sublattice_site_ratios": [0.75, 0.25],
          "sublattice_model": [["AL"], ["NI"]]
          },
          "AL3NI2": {
          "sublattice_site_ratios": [3, 2, 1],
          "sublattice_model": [["AL"], ["AL", "NI"], ["NI", "VA"]]
          },
          "AL3NI5": {
          "sublattice_site_ratios": [0.375, 0.625],
          "sublattice_model": [["AL"], ["NI"]]
          }
        }
    }


Single-phase Data
=================

Two example of ESPEI input file for single-phase data follow.
The first dataset has some data for the formation heat capacity for BCC_B2.

The ``components`` and ``phases`` keys simply describe those found in this entry.
Use the ``reference`` key for bookkeeping the source of the data.
In ``solver`` the sublattice configuration and site ratios are described for the phase.

``sublattice_configurations`` is a list of different configurations, that should correspond to the sublattices for the phase descriptions.
Non-mixing sublattices are represented as a string, while mixing sublattices are represented as a lists.
Thus an endmember for ``BCC_B2`` (as in this example) is ``["AL", "NI", VA"]`` and if there were mixing (as in the next example) it might be ``["AL", ["AL", "NI"], "VA"]``.
Mixing also means that the ``sublattice_occupancies`` key must be specified, but that is not the case in this example.
Regardless of whether there is mixing or not, the length of this list should always equal the number of sublattices in the phase, though the sub-lists can have mixing up to the number of components in that sublattice.
Note that the ``sublattice_configurations`` is a *list* of these lists.
That is, there can be multiple sublattice configurations in a single dataset.
See the second example in this section for such an example.

The ``conditions`` describe temperatures (``T``) and pressures (``P``) as either scalars or one-dimensional lists.
Most important to describing data are the ``output`` and ``values`` keys.
The type of quantity is expressed using the ``output`` key.
This can in principle be any thermodynamic quantity, but currently only ``CPM*``, ``SM*``, and ``HM*`` (where ``*`` is either nothing, ``_MIX`` or ``_FORM``) are supported.
Support for changing reference states planned but not yet implemented, so all thermodynamic quantities must be formation quantities (e.g. ``HM_FORM`` or ``HM_MIX``, etc.).

The ``values`` key is the most complicated and care must be taken to avoid mistakes.
``values`` is a 3-dimensional array where each value is the ``output`` for a specific condition of pressure, temperature, and sublattice configurations from outside to inside.
Alternatively, the size of the array must be ``(len(P), len(T), len(subl_config))``.
In the example below, the shape of the ``values`` array is (1, 12, 1) as there is one pressure scalar, one sublattice configuration, and 12 temperatures.
The formatting of this can be tricky, and it is suggested to use a NumPy array and reshape or add axes using ``np.newaxis`` indexing.

.. code-block:: JSON

    {
      "reference": "Yi Wang et al 2009",
      "components": ["AL", "NI", "VA"],
      "phases": ["BCC_B2"],
      "solver": {
	      "sublattice_site_ratios": [0.5, 0.5, 1],
	      "sublattice_configurations": [["AL", "NI", "VA"]],
	      "comment": "NiAl sublattice configuration (2SL)"
      },
      "conditions": {
	      "P": 101325,
	      "T": [  0,  10,  20,  30,  40,  50,  60,  70,  80,  90, 100, 110]
      },
      "output": "CPM_FORM",
      "values":   [[[ 0      ],
		    [-0.0173 ],
		    [-0.01205],
		    [ 0.12915],
		    [ 0.24355],
		    [ 0.13305],
		    [-0.1617 ],
		    [-0.51625],
		    [-0.841  ],
		    [-1.0975 ],
		    [-1.28045],
		    [-1.3997 ]]]
    }
    
    
In the second example below, there is formation enthalpy data for multiple sublattice configurations.
All of the keys and values are conceptually similar.
Here, instead of describing how the ``output`` quantity changes with temperature or pressure, we are instead only comparing ``HM_FORM`` values for different sublattice configurations.
The key differences from the previous example are that there are 9 different sublattice configurations described by ``sublattice_configurations`` and ``sublattice_occupancies``.
Note that the ``sublattice_configurations`` and ``sublattice_occupancies`` should have exactly the same shape.
Sublattices without mixing should have single strings and occupancies of one.
Sublattices that do have mixing should have a site ratio for each active component in that sublattice.
If the sublattice of a phase is ``["AL", "NI", "VA"]``, it should only have two occupancies if only ``["AL", "NI"]`` are active in the sublattice configuration.

The last difference to note is the shape of the ``values`` array.
Here there is one pressure, one temperature, and 9 sublattice configurations to give a shape of (1, 1, 9).

.. code-block:: JSON

    {
      "reference": "C. Jiang 2009 (constrained SQS)",
      "components": ["AL", "NI", "VA"],
      "phases": ["BCC_B2"],
      "solver": {
	      "sublattice_occupancies": [
				         [1, [0.5, 0.5], 1],
				         [1, [0.75, 0.25], 1],
				         [1, [0.75, 0.25], 1],
				         [1, [0.5, 0.5], 1],
				         [1, [0.5, 0.5], 1],
				         [1, [0.25, 0.75], 1],
				         [1, [0.75, 0.25], 1],
				         [1, [0.5, 0.5], 1],
				         [1, [0.5, 0.5], 1]
				        ],
	      "sublattice_site_ratios": [0.5, 0.5, 1],
	      "sublattice_configurations": [
				            ["AL", ["NI", "VA"], "VA"],
				            ["AL", ["NI", "VA"], "VA"],
				            ["NI", ["AL", "NI"], "VA"],
				            ["NI", ["AL", "NI"], "VA"],
				            ["AL", ["AL", "NI"], "VA"],
				            ["AL", ["AL", "NI"], "VA"],
				            ["NI", ["AL", "VA"], "VA"],
				            ["NI", ["AL", "VA"], "VA"],
				            ["VA", ["AL", "NI"], "VA"]
				           ],
	      "comment": "BCC_B2 sublattice configuration (2SL)"
      },
      "conditions": {
	      "P": 101325,
	      "T": 300
      },
      "output": "HM_FORM",
      "values":   [[[-40316.61077, -56361.58554,
	             -49636.39281, -32471.25149, -10890.09929,
	             -35190.49282, -38147.99217, -2463.55684,
	             -15183.13371]]]
    }



Multi-phase Data
================

The difference between single- and multi-phase is data is in the absence of the ``solver`` key, since we are no longer concerned with individual site configurations, and the ``values`` key where we need to represent phase equilibria rather than thermodynamic quantities.
Notice that the type of data we are entering in the ``output`` key is ``ZPF`` (zero-phase fraction) rather than ``CP_FORM`` or ``H_MIX``.
Each entry in the ZPF list is a list of all phases in equilibrium, here ``[["AL3NI2", ["NI"], [0.4083]], ["BCC_B2", ["NI"], [0.4340]]]`` where each phase entry has the name of the phase, the composition element, and the composition of the tie line point.
If there is no corresponding tie line point, such as on a liquidus line, then one of the compositions will be ``null``: ``[["LIQUID", ["NI"], [0.6992]], ["BCC_B2", ["NI"],  [null]]]``.
Three- or n-phase equilibria are described as expected: ``[["LIQUID", ["NI"], [0.752]], ["BCC_B2", ["NI"], [0.71]], ["FCC_L12", ["NI"], [0.76]]]``.

Note that for higher-order systems the component names and compositions are lists and should be of length ``c-1``, where ``c`` is the number of components.

.. code-block:: JSON

    {
      "components": ["AL", "NI"],
      "phases": ["AL3NI2", "BCC_B2"],
      "conditions": {
	      "P": 101325,
	      "T": [1348, 1176, 977]
      },
      "output": "ZPF",
      "values":   [
             [["AL3NI2", ["NI"], [0.4083]], ["BCC_B2", ["NI"], [0.4340]]],
	           [["AL3NI2", ["NI"], [0.4114]], ["BCC_B2", ["NI"], [0.4456]]],
	           [["AL3NI2", ["NI"], [0.4114]], ["BCC_B2", ["NI"], [0.4532]]]
                  ],
      "reference": "37ALE"
    }

