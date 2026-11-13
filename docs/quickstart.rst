Quick start
===========

Install the runtime
-------------------

The validated environment uses **Python 3.13.9**. Other Python versions have not
been validated. ``requirements.txt`` records the tested direct dependency
versions; it is not a complete environment lock.

From the repository root, install the dependencies in your Python environment:

.. code-block:: console

   python -m pip install -r requirements.txt

Prepare the observations
------------------------

1. Copy ``config.example.yaml`` to ``config.local.yaml``.
2. Set the file paths, observatory position, detector gain/read noise, trim
   range, and extraction settings for the actual observations. The supplied
   values describe an example QSI configuration.
3. Place bias frames and any same-night dark frames at the top level of the
   observation directory. Supply science, standard-star, and arc observations
   as described in :doc:`inputs`.
4. Review ``TARGETS`` and ``REFERENCES`` in ``speclibparallel.py`` and supply
   the template files required by the selected reference stars.

Use distinct observation, working, and output directories. The template and
dark-library directories are configured separately; templates and master
darks are not bundled with the source.

Run from the command line
-------------------------

.. code-block:: console

   python spec_reduction_parallel.py observations/night_001 config.local.yaml

The first argument is the observation directory and the second is the YAML
configuration file. Both may be absolute or relative. **All relative paths,
including paths inside the YAML configuration, use the current working
directory** rather than the directory containing the configuration file.
Quote paths containing spaces.

Display the command-line help with:

.. code-block:: console

   python spec_reduction_parallel.py --help

After execution, inspect the run status, warnings, and scientific diagnostics.
See :doc:`outputs` for publication conditions and the location of incomplete
runs.

Call from Python
----------------

.. code-block:: python

   from speclibparallel import main

   if __name__ == "__main__":
       report = main("observations/night_001", "config.local.yaml")
       print(report["status"])

Keep the call inside the ``__main__`` guard when using worker processes,
particularly on Windows. A completed reduction returns a run-report dictionary;
per-stage failures and warnings are recorded in that report. File/configuration
errors that prevent initialization can raise before a run report is created.

For callable signatures and docstrings, see :doc:`api/index`. To build this
guide locally, follow :doc:`development`.
