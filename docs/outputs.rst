Outputs and run status
======================

Working and published directories
---------------------------------

Each run creates ``<workpath>/run-<UTC timestamp>-<id>/``. When all target
reductions finish without failed stage events, selected products are published
to ``<savepath>/<observation-directory-name>/`` and verified. Successful cleanup
then removes the working directory.

Incomplete runs retain their working products and do not install a new completed
product set. Publication and cleanup failures are recorded in the run manifest.
Use separate observation, work, and output directories. Existing output
collisions are handled by the controller's staged publication policy.

.. list-table:: Product files
   :header-rows: 1
   :widths: 45 55

   * - Filename
     - Content
   * - ``T<group>_<target>_relflux_<suffix>.fits``
     - Final relative-flux spectrum and uncertainty realizations.
   * - ``*.ec.fits``
     - Extracted spectrum on native detector columns.
   * - ``*.ec_dc.fits``
     - Dispersion-corrected spectrum.
   * - ``*.res.fits``
     - Reference-star response function.
   * - ``*.png``, ``*.pdf``
     - Enabled trace, wavelength, and response diagnostic plots.
   * - ``run_manifest.json``, ``pipeline.log``
     - Status, stage events, warnings, and recorded provenance.
   * - ``effective_config.yaml``, ``grouping_summary.json``
     - Applied configuration and frame associations.

Final relative-flux FITS file
-----------------------------

The ``SPECTRUM`` binary table contains:

.. list-table:: Spectrum columns
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Interpretation
   * - ``WAVELENGTH``
     - Air wavelength in Angstrom. Inspect wavelength-frame metadata for the
       velocity correction state.
   * - ``FLUX``
     - Dimensionless relative flux, normalized in the fixed wavelength window.
   * - ``FLUXERR``
     - Sample standard deviation of the normalized Monte Carlo realizations.
   * - ``MASK``
     - Invalid-bin mask.
   * - ``RESP_SCATTER``
     - Reference-frame log-response scatter indicator, distinct from ``FLUXERR``.
   * - ``QUALITY``
     - Combined target and reference quality information.
   * - ``TARGQUAL``, ``REFQUAL``
     - Separate target and reference quality information.

A nonzero quality value is not automatically fatal: some flags describe
retained data. Use ``MASK`` and the quality definitions, including the
``FATALBIT`` header, when interpreting support.

``MC_RELFLUX`` stores the normalized uncertainty realizations and ``MC_INDEX``
stores their identifiers. ``NORMTYPE``, ``NORMWLO``, ``NORMWHI``, ``NORMNPIX``,
and ``NORMCOV`` describe the normalization. ``ERRMODEL``, ``SYSERR``, and
``CROSSCAL`` document the uncertainty assumptions discussed in :doc:`science`.

Run status and failure inspection
---------------------------------

The controller returns the run-report dictionary. Its status is one of:

.. list-table:: Final status
   :header-rows: 1
   :widths: 35 65

   * - Status
     - Meaning
   * - ``completed``
     - The run completed without recorded warnings or failed stage events.
   * - ``completed_with_warnings``
     - Completed outputs are available, with warnings requiring review.
   * - ``partial_success``
     - Some target outputs exist but the run is incomplete or has failed events.
   * - ``failed``
     - The run did not complete successfully; inspect its recorded failures.

Consult the manifest and its events rather than inferring success from the
presence of an individual FITS file. For example, a successful extraction may
remain available even if arc calibration or response correction fails later.
Publication and cleanup outcomes are recorded separately.

Generated headers, manifests, and logs can contain original observation
metadata and local paths. Review reduced products separately before sharing
them; the source-only documentation and release do not anonymize observations.
