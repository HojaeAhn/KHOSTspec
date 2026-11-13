Configuration
=============

Copy ``config.example.yaml`` to a local configuration and adjust it for the
observations. The example detector values must be verified against the actual
instrument. All relative paths use the current working directory.

Paths, detector, and observatory
--------------------------------

.. list-table:: Main settings
   :header-rows: 1
   :widths: 35 65

   * - Key
     - Meaning
   * - ``workpath``, ``savepath``
     - Parents of per-run working directories and completed product directories.
   * - ``darkpath``, ``refpath``
     - Master-dark library and standard-template library.
   * - ``memory_limit``
     - Memory budget in decimal GB; limits worker allocation.
   * - ``longitude``, ``latitude``, ``elevation``
     - Observatory position in degrees east, degrees north, and metres.
   * - ``CCD.gain``, ``CCD.rdnoise``
     - Electrons per ADU and read noise in electrons.
   * - ``CCD.datay_start``, ``CCD.datay_end``
     - Trimmed detector-row range, start inclusive and end exclusive.
   * - ``CCD.saturation``
     - Optional ADU threshold; otherwise available detector metadata are used.

Tracing and wavelengths
-----------------------

.. list-table:: Trace and wavelength settings
   :header-rows: 1
   :widths: 35 65

   * - Key
     - Meaning
   * - ``Trace.naper``
     - Use ``1`` for the operating pipeline.
   * - ``Trace.dW``
     - Dispersion median-slice half-width in pixels. Trace steps are ``2*dW``.
   * - ``Trace.dH``
     - Spatial search-box half-width in pixels.
   * - ``Trace.fwhm``
     - Legacy name for the spatial peak-detection filter width, not a fitted
       Gaussian FWHM.
   * - ``Trace.threshold``
     - Traced peak excess above the local median. The initial central search
       uses its own fixed threshold.
   * - ``Trace.order``, ``Wavecal.order``
     - Trace and wavelength polynomial degrees.
   * - ``Wavecal.refpath``
     - Arc line-list path.
   * - ``Wavecal.lindisp``
     - Initial dispersion estimate in Angstrom per pixel.
   * - ``Wavecal.window``
     - Gaussian peak-fitting half-window in pixels.
   * - ``Wavecal.wmin``, ``Wavecal.wmax``
     - Accepted Gaussian sigma range in pixels.
   * - ``Wavecal.prominence``
     - Arc peak prominence threshold.
   * - ``Trace.plotyn``, ``Wavecal.plotyn``, ``Resgen.plotyn``
     - Write the corresponding diagnostic plots when enabled.

Extraction
----------

.. list-table:: ``Extract`` settings
   :header-rows: 1
   :widths: 35 65

   * - Key
     - Meaning
   * - ``method``
     - ``ac`` by default; ``horne`` retains the earlier extraction path.
   * - ``raper``
     - Aperture half-width in pixels.
   * - ``bgin``, ``bgout``
     - Background-sideband radii; require ``raper < bgin < bgout``.
   * - ``crreject``, ``niter``
     - Positive cosmic-ray threshold in sigma and maximum rejections per
       column. ``niter: 0`` disables rejection.
   * - ``nk``, ``xbin``, ``xsmooth``
     - Profile basis count, initial training-bin width, and smoothing scale.
   * - ``profile_floor``
     - Positive floor for profile weights.
   * - ``ac_training_bin``
     - Training columns per independent fold; default ``64``.
   * - ``ac_window``
     - Anchor pooling half-width in original wavelength columns; default ``128``.
   * - ``ac_cr_refinements``
     - Maximum final cosmic-ray refinement passes; default ``3``.
   * - ``mc_samples``
     - Optional extraction draw count; otherwise uses ``Resgen.mc_samples``.
   * - ``mc_seed``
     - Optional extraction Monte Carlo seed.

Response and normalization
--------------------------

.. list-table:: ``Resgen`` settings
   :header-rows: 1
   :widths: 35 65

   * - Key
     - Meaning
   * - ``mc_samples``
     - Monte Carlo draw count, default ``256`` and minimum ``8``.
   * - ``order``
     - Response spline interval setting. The spline is cubic; this is not its
       polynomial degree.
   * - ``lines``
     - Optional masked regions as ``name: {center, width}`` in Angstrom.
       Defaults include Balmer and selected telluric bands.
   * - ``clip_sigma``, ``fit_iterations``, ``n_shifts``
     - Clipping threshold, clipping iterations, and shifted knot grids;
       defaults ``2.5``, ``5``, and ``3``.
   * - ``seed``
     - Response Monte Carlo base seed; default ``42``.
   * - ``match_resolution``
     - Match observation and template resolution when supported; default ``true``.
   * - ``observed_fwhm_angstrom``
     - Optional fixed observed spectral FWHM instead of the arc estimate.
   * - ``relflux_norm_window``
     - Fixed science-grid interval in Angstrom; default ``[5400.0, 5600.0]``.
   * - ``relflux_norm_min_coverage``
     - Required supported wavelength-width fraction of the window; default ``0.9``.
   * - ``relflux_norm_min_bins``
     - Required valid bins inside the window; default ``20``.

Use consistent extraction and response draw counts: downstream propagation
uses their common sample IDs. Legacy ``Resgen.lambda``, ``pvalue``, and
``dualyn`` options are not used by the current response-file pipeline.

Runtime, coordinates, and templates
-----------------------------------

.. code-block:: yaml

   Runtime:
       max_workers: 4
       offline: true
       update_dark_library: false

``max_workers`` remains subject to memory and task-count limits.
``offline`` disables online name resolution and IERS automatic downloads for
the relevant stages. ``update_dark_library`` controls copying newly generated
local dark masters to the library.

Explicit coordinates use ``Coordinates.<object>.ra_deg`` and
``Coordinates.<object>.dec_deg``. Template metadata can be supplied through
``Templates.libraries.<library>`` and
``Templates.files.<template_file>``. Coordinate precedence and template
metadata conventions are described in :doc:`inputs`.

Complete example
----------------

The following is included directly from the maintained configuration file:

.. literalinclude:: ../config.example.yaml
   :language: yaml
