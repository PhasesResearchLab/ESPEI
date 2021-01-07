Releasing ESPEI
===============

Use this checklist to create a new release of ESPEI and distribute the package
to PyPI and conda-forge. All steps are intended to be run from the root directory of the repository (i.e.
the one containing ``docs/``, ``espei/``, ``setup.py``, ...).

Creating a new release
----------------------

These steps will create a new tagged version in the GitHub repository.

1. ``git pull`` to make sure you haven't missed any last-minute commits. **After this point, nothing else is making it into this version.**
#. ``pytest -v --doctest-modules espei tests`` to ensure that all tests pass locally.
#. ``sphinx-apidoc -f -o docs/api/ espei/ -H 'API Documentation'`` to
   regenerate API documentation.
#. Update the ``version`` and ``release`` variables of ``docs/conf.py`` to the new version.
#. Commit the updated API documentation and ``conf.py`` file.
#. ``git push`` and verify all tests pass on all CI services.
#. Generate a list of commits since the last version with
   ``git --no-pager log --oneline --no-decorate 0.1^..origin/master``.
   Replace ``0.1`` with the tag of the last public version.
#. Condense the change list into something user-readable. Update and commit
   ``CHANGES.rst`` with the release date.
#. ``git tag 0.2 master -m "0.2"`` Replace ``0.2`` with the new version.
#. ``git show 0.2`` to ensure the correct commit was tagged.
#. ``git push origin master --tags`` to push the tag to GitHub.

Now the public package must be built and distributed.

Uploading to PyPI
-----------------

1. ``rm -R dist/*`` on Linux/OSX or ``del dist/*`` on Windows.
#. ``python setup.py sdist`` to create a source distribution.
#. Make sure that the script correctly detected the new version exactly and not a
   dirty or revised state of the repository.
#. ``twine upload -r pypi -u bocklund dist/*`` to upload to PyPI.


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