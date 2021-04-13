.. raw:: latex

   \chapter{Contributing to ESPEI}

.. _Contributing guide:

=====================
Contributing to ESPEI
=====================

This is the place to start as a new ESPEI contributor. This guide assumes you have
:ref:`installed a development version of ESPEI <installing-development-versions>`.

The next sections lay out the basics of getting an ESPEI development set up and the development standards.
Then the :ref:`Software design` sections walk through the key parts of the codebase.


Tests
=====

Even though much of ESPEI is devoted to being a multi-core, stochastic user tool, we strive to test all logic and functionality.
We are continuously maintaining tests and writing tests for previously untested code.
As a general rule, any time you write a new function or modify an existing function you should write or maintain a test for that function.

Some tips for testing:

* Ideally you would practicing test driven development by writing tests of your intended results before you write the function.
* If possible, keep the tests small and fast.
* See the `NumPy/SciPy testing guidelines <https://github.com/numpy/numpy/blob/master/doc/TESTS.rst.txt>`_ for more tips.

Running Tests
-------------

ESPEI uses `pytest <https://pytest.org>`_ as a test runner. The tests can be run from the root directory of the cloned repository:

.. code-block:: bash

   pytest


Style
=====

Code style
----------

For most naming and style, follow `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_.
One exception to PEP8 is regarding the line length, which we suggest a 120 character maximum, but may be longer within reason.

Code documentation
------------------

ESPEI uses the `NumPy documentation <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_ style.
All functions and classes should be documented with at least a description, parameters, and return values, if applicable.

Using ``Examples`` in the documentation is especially encouraged for utilities that are likely to be run by users.
See :py:func:`espei.analysis.truncate_arrays` for an example.

Web documention
---------------

Documentation on ESPEI is split into user tutorials, reference and developer documentation.

* Tutorials are resources for users new to ESPEI or new to certain features of ESPEI to be *guided* through typical actions.
* Reference pages should be concise articles that explain how to complete specific goals for users who know what they want to accomplish.
* Developer documentation should describe what should be considered when contributing source code back to ESPEI.

You can check changes you make to the documentation by going to the documentation folder in the root repository ``cd docs/``.
Running the command ``make html && cd build/html && python3 -m http.server && cd ../.. && make clean`` from that folder will build the docs and run them on a local HTTP server.
You can see the documentation when the server is running by
visting the URL at the end of the output, usually ``localhost port 8000 <http://0.0.0.0:8000>``_.
When you are finished, type ``Ctrl-C`` to stop the server and the command will clean up the build for you.

Make sure to fix any warnings that come up if you are adding documentation.


Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

The docs can be built by running the docs/Makefile (or docs/make.bat on
Windows). Then Python can be used to serve the html files in the _build
directory and you can visit ``http://localhost:8000`` in your broswer to
see the built documentation.

For Unix systems:

.. code-block:: bash

   cd docs
   make html
   cd _build/html
   python -m http.server

Windows:

.. code-block:: bash

   cd docs
   make.bat html
   cd _build\html
   python -m http.server


Logging
=======

Since ESPEI is intended to be run by users, we must provide useful feedback on how their runs are progressing.
ESPEI uses the logging module to allow control over verbosity of the output.

There are 5 different logging levels provided by Python.
They should be used as follows:

Critical or Error (``logging.critical`` or ``logging.error``)
  Never use these. These log levels would only be used when there is an unrecoverable error that requires the run to be stopped.
  In that case, it is better to ``raise`` an appropriate error instead.
Warning (``logging.warning``)
  Warnings are best used when we are able to recover from something bad that has happened.
  The warning should inform the user about potentially incorrect results or let them know about something they have the potential to fix.
  Again, anything unrecoverable should not be logged and should instead be raised with a good error message.
Info (``logging.info``)
  Info logging should report on the progress of the program.
  Usually info should give feedback on milestones of a run or on actions that were taken as a result of a user setting.
  An example of a milestone is starting and finishing parameter generation.
  An example of an action taken as a result of a user setting is the logging of the number of chains in an mcmc run.
Debug (``logging.debug``)
  Debugging is the lowest level of logging we provide in ESPEI.
  Debug messages should consist of possibly useful information that is beyond the user's direct control.
  Examples are the values of initial parameters, progress of checking datasets and building phase models, and the acceptance ratios of MCMC iterations.

