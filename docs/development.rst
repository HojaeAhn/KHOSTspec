Development and documentation
=============================

Source organization
-------------------

The command-line entry point is ``spec_reduction_parallel.py``. Scientific
stages and the run controller are in ``speclibparallel.py``. The API reference
in :doc:`api/index` is generated from the callable signatures and docstrings.

Keep function docstrings accurate about units, array shapes, return values,
side effects, and failure behavior. Source comments are written in English.
Changes to the scientific implementation should preserve the documented
quality and uncertainty policies or explicitly update their validation.

Build the documentation
-----------------------

Run these commands from the repository root:

.. code-block:: console

   python -m pip install -r requirements.txt -r requirements-docs.txt
   python docs/build.py

The API build imports the runtime modules, so the scientific dependencies are
needed as well as Sphinx. The build helper treats warnings as failures and
keeps its doctree cache in ``docs/_build/doctrees``, separate from the
publishable HTML in ``docs/_build/html``. Open
``docs/_build/html/index.html`` to read the generated site locally. HTML output
is generated content and should not be mixed with the maintained source pages.

The equivalent manual build command is:

.. code-block:: console

   python -m sphinx -W --keep-going -d docs/_build/doctrees -b html docs docs/_build/html

For command-line usage independent of the documentation build:

.. code-block:: console

   python spec_reduction_parallel.py --help

Tests
-----

.. code-block:: console

   python -m pip install -r requirements-dev.txt
   python -m pytest tests

The public suite contains six tests covering path handling and entry-module
delegation. These tests do not require observation files and are not a complete
scientific regression suite.

The pre-release local review also passed 308 regression tests. Refactoring was
compared on two real observation strips, each 61 by 6,216 pixels, with 256
Monte Carlo realizations. All 30 compared output arrays and 41 diagnostic values
per frame were identical before and after that refactor. Observation-dependent
fixtures and historical study archives are not bundled in the source release.

Publishing source and documentation
-----------------------------------

Keep local settings, observations, results, and study archives out of a source
release. The project ignore rules select the maintained files explicitly;
include new documentation sources deliberately when adding pages.

Generated FITS headers, run logs, and manifests can retain local paths and
observation metadata. Ignore rules do not remove information from existing Git
history. A source-only export has no repository history.

GitHub Actions and Pages
------------------------

The maintained workflow is ``.github/workflows/documentation.yml``. A push to
``main`` runs the portable tests and a strict Sphinx build, then publishes
the resulting HTML to GitHub Pages. Pull requests targeting ``main`` run
the same tests and build but do not deploy. Merging a pull request into
``main`` creates a push event and updates the published site.

Use ``develop`` for ongoing work and open a pull request into ``main`` when
the changes are ready for publication. A push to ``develop`` does not run this
workflow. There is no manual deployment trigger; an existing failed run can
still be retried after its underlying problem is resolved. A local ``git pull``
downloads commits and does not produce a GitHub workflow event.

Initial repository setup requires the following:

1. Create a public personal repository named ``KHOSTspec`` and push the reviewed
   source-only export to ``main``. Set ``main`` as the default branch and
   create ``develop`` for ongoing development.
2. In ``Settings > Pages``, set the publishing source to ``GitHub Actions``.
3. In ``Settings > Environments > github-pages``, ensure that deployment rules
   permit only ``main``. Remove an older rule permitting ``develop``.
4. The initial ``main`` push starts ``Sphinx documentation``. If Pages was not
   enabled before that run reached deployment, finish the settings and retry
   its failed jobs from the Actions tab.

The optional ``tools/publish_github.ps1`` helper performs these setup steps for
a new repository after a local GitHub CLI login. It is intended for a clean
source export and refuses existing local or remote repositories. It obtains
the authenticated account name from GitHub CLI and uses the account's numeric
ID and login to form a GitHub no-reply commit address. It does not request a
password or print an access token.

The helper requires permission to execute a reviewed local PowerShell script.
It has passed syntax checks; the validation workstation's script execution
policy prevented an execution test. If your environment blocks it, use an
approved publishing route or the repository settings described above.

The workflow uses read-only source access during the build. Only the deployment
job receives ``pages: write`` and ``id-token: write``, and only
``docs/_build/html`` is uploaded; doctree caches remain outside the artifact.
The default site URL is ``https://<your-login>.github.io/KHOSTspec/``. The deploy
job provides the final URL. Local builds do not upload or publish anything.

If a run fails, inspect dependency installation, the test/build output, Pages
enablement, and the environment's allowed branches. A successful local build
does not by itself confirm that repository permissions or deployment settings
are correct. See `GitHub's custom Pages workflow guide
<https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages>`_
for the required repository settings and permissions.

Sphinx references
-----------------

The build uses `autodoc <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html>`_
for runtime API introspection, `Napoleon <https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html>`_
for NumPy-style docstrings, and the built-in source viewer. See the
`Sphinx build command reference <https://www.sphinx-doc.org/en/master/man/sphinx-build.html>`_
for additional output formats and build options.
