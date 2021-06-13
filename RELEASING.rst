Releasing ESPEI
===============

Use this checklist to create a new release of ESPEI and distribute the package
to PyPI and conda-forge. All steps are intended to be run from the root
directory of the repository (i.e. the one containing ``docs/``, ``espei/``,
``setup.py``, ...).

Create a release of espei
--------------------------
To release a new version of espei:

These steps assume that ``0.1`` is the most recently tagged version number and ``0.2`` is the next version number to be released.
Replace their values with the last public release's version number and the new version number as appropriate.

#. Determine what the next version number should be using `semantic versioning <https://semver.org/>`_.
#. Resolve or defer all pull requests and issues tagged with the upcoming version milestone.
#. ``git stash`` to save any uncommitted work.
#. ``git checkout master``
#. ``git pull`` to make sure you haven't missed any last-minute commits. **After this point, nothing else is making it into this version.**
#. ``pytest`` to ensure that all tests pass locally.
#. ``sphinx-apidoc -f -H 'API Documentation' -o docs/api/ espei`` to regenerate the API documentation.
#. Update ``CHANGES.rst`` with a human-readable list of changes since the last commit.
   ``git log --oneline --no-decorate --color 0.1^..master`` can be used to list the changes since the last version.
#. ``git add docs/api CHANGES.rst`` to stage the updated documentation.
#. ``git commit -m "REL: 0.2"`` to commit the changes.
#. ``git push origin master``
#. **Verify that all continuous integration test and build workflows pass.**
#. Create a release on GitHub

   #. Go to https://github.com/phasesresearchlab/espei/releases/new
   #. Set the "Tag version" field to ``0.2``.
   #. Set the branch target to ``master``.
   #. Set the "Release title" to ``espei 0.2``.
   #. Leave the description box blank.
   #. If this version is a pre-release, check the "This is a pre-release" box.
   #. Click "Publish release".
#. The new version will be available on PyPI when the ``Build and deploy to PyPI`` workflow on GitHub Actions finishes successfully.

Now the public package must be built and distributed.

Deploy to PyPI (manually)
-------------------------

.. warning::

   DO NOT FOLLOW THESE STEPS unless the GitHub Actions deployment workflow is broken.
   Creating a GitHub release should trigger the ``Build and deploy to PyPI`` workflow on GitHub Actions that will upload source and platform-dependent wheel distributions automatically.

To release a source distribution to PyPI:

#. If deploying for the first time: ``pip install twine build``
#. ``rm -R dist/*`` on Linux/OSX or ``del dist/*`` on Windows
#. ``git checkout master`` to checkout the latest version
#. ``git pull``
#. ``git log`` to verify the repository state matches the newly created tag
#. ``python -m build --sdist``
#. **Make sure that the script correctly detected the new version exactly and not a dirty / revised state of the repo.**
#. ``twine upload dist/*`` to upload (assumes a `correctly configured <https://packaging.python.org/specifications/pypirc/>`_ ``~/.pypirc`` file)


Updating the conda-forge feedstock
----------------------------------

conda-forge is a community-developed platform for distributing packages to the
`conda-forge channel on Anaconda Cloud`_. Metadata for the packages are hosted
in *feedstocks* and built using `conda-build`_ in a continuous integration
pipeline.

`conda-build`_ is driven by a ``recipe/meta.yaml`` configuration file, which
specifies the package metadata and dependencies. The ``meta.yaml`` file is
updated via pull requests to the `conda-forge/espei-feedstock`_. A pull request
is usually opened automatically by the conda-forge autotick bot, but pull
requests can be opened manually as well. Both methods are detailed below.

After updating the ``meta.yaml`` file and merging the pull request, the
conda-forge continuous integration pipeline will run from the master branch and
upload build artifacts to the `conda-forge channel on Anaconda Cloud`_. Uploaded
build artifacts are usually available to download and install in about 1 hour
after the continuous integration pipeline completes on the master branch.

The availability of a particular ESPEI package on conda-forge can be verified by
running ``conda search -c conda-forge --override-channels espei``.

conda-forge autotick bot (preferred)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. The `conda-forge autotick bot`_ will automatically open a pull request in
   the `conda-forge/espei-feedstock`_ repository after the package has been
   uploaded to PyPI. This usually happens in less than 10 minutes after the
   PyPI release.
#. Verify that the ``recipe/meta.yaml`` requirements match the dependencies in ``environment-dev.yml``.
#. Once all the checks pass, merge the pull request.


Manually updating
~~~~~~~~~~~~~~~~~

If the `conda-forge autotick bot`_ does not open a pull request automatically,
the `conda-forge/espei-feedstock`_ can still be updated manually with a pull
request that updates the ``recipe/meta.yaml`` file.

1. Get the sha-256 hash of the tarball via ``openssl sha256 dist/espei-0.3.1.tar.gz``
   or by viewing the hashes for the release at https://pypi.org/project/espei/#files.
#. Fork the `conda-forge/espei-feedstock`_ repository.
#. Update the version number and hash in the ``recipe/meta.yaml`` file and set
   the build number to zero if the version number changed.
#. Verify that the ``recipe/meta.yaml`` requirements match the dependencies in ``environment-dev.yml``.
#. Open a PR against the `conda-forge/espei-feedstock`_ repository
#. Once all the checks pass, merge the pull request.

.. _conda-forge autotick bot: https://github.com/regro-cf-autotick-bot
.. _conda-forge/espei-feedstock: https://github.com/conda-forge/espei-feedstock
.. _conda-forge channel on Anaconda Cloud: https://anaconda.org/conda-forge
.. _conda-build: https://docs.conda.io/projects/conda-build
