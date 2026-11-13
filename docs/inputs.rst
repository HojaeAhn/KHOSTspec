Input data
==========

FITS layout and headers
-----------------------

Bias and same-night dark discovery uses the top level of the observation
directory. Science, reference-star, and arc group discovery also searches
subdirectories and recognizes ``.fit``, ``.fits``, and ``.fts``. Bias frames are
required. A compatible library dark can be used when a suitable same-night
master is unavailable.

.. list-table:: Header conventions
   :header-rows: 1
   :widths: 30 70

   * - Header
     - Use
   * - ``IMAGETYP``
     - ``Bias Frame``, ``Dark Frame``, ``Flat Field``, or ``Light Frame``.
   * - ``OBJECT``
     - ``Arc`` for lamp frames; otherwise a configured target or reference name.
   * - ``JD``, ``JD-OBS``, ``MJD``, ``MJD-OBS``
     - Group ordering. ``JD`` takes priority; modified Julian dates are converted.
   * - ``EXPTIME``
     - Exposure time in seconds, used by the calibration path.
   * - ``CCD-TEMP``, ``SET-TEMP``, ``CCDTEMP``
     - Detector temperature for calibration compatibility; measured temperature
       is preferred.
   * - ``DATE-OBS``
     - Time for airmass, velocity correction, and dark-library selection.
   * - ``OBJCTRA`` / ``OBJCTDEC``
     - Optional coordinates: RA as hour angle and Dec in degrees.
   * - ``RA`` / ``DEC``
     - Alternative coordinates, both interpreted as numeric degrees.

Group discovery skips unreadable, unrecognized, or undated frames.
``DATE-OBS`` alone does not supply the Julian date needed for grouping. An
invalid preferred ``JD`` value also prevents the frame from being grouped.
Although some helpers recognize exposure aliases, the operating calibration
path requires ``EXPTIME``.

Object catalogs and grouping
----------------------------

``TARGETS`` in ``speclibparallel.py`` lists the recognized science objects.
``REFERENCES`` maps each standard-star name to its template library and file.
Object names are matched without regard to case or spaces. Unrecognized science
objects are ignored.

Runs are formed from consecutive frames of the same **frame class**, ordered
by time. A change of ``OBJECT`` alone does not split a run. Keep each consecutive
target or reference run limited to one object, and inspect
``grouping_summary.json`` for mixed-object observations. Reference groups
contain at most three frames.

Standard-star templates
-----------------------

Provide each required template using this directory layout:

.. code-block:: text

   <refpath>/<library>/<template_file>.fits

The ``REFERENCES`` entry supplies the library and template filename without
the ``.fits`` suffix. Supported formats are:

* **Pickles and CALSPEC:** a FITS binary table with ``WAVELENGTH`` and ``FLUX``
  columns. Available statistical-error and quality metadata are used.
* **MILES:** a one-dimensional FITS image with header wavelength coordinates.

Declare the wavelength medium and resolution from the actual template
provenance. ``Templates.libraries.<library>`` accepts ``wavelength_medium``
(``air`` or ``vacuum``), and either ``fwhm_angstrom`` or ``resolving_power``.
``Templates.files.<template_file>`` supplies file-specific overrides. Metadata
embedded in the corresponding ``REFERENCES`` entry takes precedence.

Optical vacuum templates are converted to air with the corresponding flux
density transformation. If the wavelength convention is unknown, the input
axis is retained and a warning is recorded. Missing template statistical errors
also produce a warning: the template is then treated as fixed in the response
uncertainty calculation. See :doc:`science`.

Arc line list
-------------

``Wavecal.refpath`` names a text file with one reference wavelength in Angstrom
per line and no header. The supplied ``NeAr.csv`` follows this format.

Dark library
------------

Library masters use filenames such as
``Mdark_T<temp>_E<exptime>_<YYYYMMDD>.fit``. Selection checks the FITS metadata
and detector compatibility rather than treating the filename as authoritative.
The default temperature tolerance is 0.5 degrees C.

The normal selection preference is:

1. Compatible same-night master with the exact exposure.
2. Compatible library master with the exact exposure.
3. Compatible scaled same-night master.
4. Compatible scaled library master.

If a same-exposure local dark was rejected as incompatible, an exact compatible
replacement is required; scaling does not bypass that rejection. New local
masters are written to ``darkpath`` unless
``Runtime.update_dark_library: false`` is configured.

Coordinates and offline operation
---------------------------------

Resolution order is explicit ``Coordinates`` configuration,
``OBJCTRA``/``OBJCTDEC``, numeric ``RA``/``DEC``, then online name lookup.
Configuration coordinates use ``Coordinates.<object>.ra_deg`` and
``Coordinates.<object>.dec_deg`` in degrees.

With ``Runtime.offline: true``, online name lookup and IERS automatic downloads
are disabled for those processing stages. Coordinates must be available from
the configuration or FITS headers.
