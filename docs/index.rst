KHOSTspec
=========

Reduce optical FITS observations into extracted spectra, wavelength-calibrated
spectra, and response-corrected **relative-flux spectra with statistical
uncertainties**. The pipeline processes independent frames in parallel and
records calibration choices, warnings, and product provenance for each run.

Start with :doc:`quickstart` to run an observation directory. Read
:doc:`inputs` and :doc:`configuration` before adapting the example to another
instrument or observing field. :doc:`science` explains the extraction,
normalization, and uncertainty assumptions that affect scientific use.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   quickstart
   inputs
   workflow
   configuration
   science
   outputs
   development

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index

Source layout
-------------

``spec_reduction_parallel.py`` provides the command-line entry point and
compatible worker aliases. ``speclibparallel.py`` contains the scientific
stages and run controller. There is no separate batch-extraction module:
512-column Horne and derivative kernels are included in the library.

The source distribution includes an example configuration and an arc line list.
Observation files, dark masters, and standard-star templates must be supplied
separately.
