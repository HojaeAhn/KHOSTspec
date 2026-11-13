Extraction and uncertainty model
================================

Read this page when interpreting relative flux, quality flags, or uncertainties.
The API details and array conventions are documented in :doc:`api/index`.

A+C extraction
--------------

The default extraction combines a positive spatial profile with signed-anchor
calibration trained independently of the extracted columns. The signed anchor
is built once after the final cosmic-ray refinement pass. Horne fits and their
finite-iteration derivatives are evaluated in blocks of up to 512 wavelength
columns.

Each column retains its own convergence state and sequential cosmic-ray
rejections. Column batching changes the execution layout, not the wavelength
sampling. Profile-training bins, anchor pooling windows, and Monte Carlo draw
batches are distinct settings or execution mechanisms.

Insufficient independent anchor support retains the frame-level failure policy;
the pipeline does not install a local-anchor recovery scheme. Nonconverged
cosmic-ray/profile refinement is also represented by a fatal quality state.
Large negative residuals can be retained with a quality warning rather than
being silently clipped as positive cosmic rays.

Physical detector variance and the variance used to construct fitting weights
have separate roles. The finite-iteration derivatives follow the actual capped
weight-update path rather than assuming an exactly converged fixed point.

Relative-flux normalization
---------------------------

Let ``C`` be the response-corrected nominal spectrum, and let ``W`` select valid
bins in the configured science-grid wavelength window. The reported nominal
relative flux is:

.. code-block:: text

   F_rel(lambda) = C(lambda) / median(C[W])

The default window is **5400--5600 Angstrom** on the output science wavelength
grid. At least 20 valid bins and 90% supported window width are required.
Coverage uses the sum of supported bin widths within the window, not simply
the fraction of valid bin centers. Inadequate support produces an error; the
code does not replace the fixed window with a full-range normalization.

Every Monte Carlo realization is divided by its own median over the same
selected bins. The sample standard deviation of those normalized realizations
is reported as ``FLUXERR``. Thus normalization uncertainty and its correlation
with the spectrum are included in that reported statistical uncertainty.

Changing the normalization window changes the numerical reference scale.
A lower relative-flux or error value alone does not indicate improved
observational signal-to-noise. The output is relative flux, not an absolute
spectrophotometric calibration.

Modeled and unmodeled uncertainty
---------------------------------

The Monte Carlo propagation includes modeled spectral correlations and
response/normalization uncertainty. Upstream marginal variance includes
master-calibration variance. The final uncertainties remain **conditional
statistical errors**:

* Template mismatch and other unmodeled systematic errors are not included.
* Cross-frame covariance caused by sharing calibration masters is not modeled.
* When template statistical errors are unavailable, the template is treated
  as fixed and the omission is reported.
* If native extraction-noise provenance is unavailable, an independence
  approximation for saved bins is reported rather than silently claimed to
  preserve all correlations.

Relevant output headers include ``ERRMODEL``, ``COVAR``, ``SYSERR``,
``CROSSCAL``, ``TPLSTAT``, and ``NATIVEMC``. See :doc:`outputs` for the final
table and uncertainty extensions.

Wavelength and response interpretation
--------------------------------------

The science wavelength convention is air in Angstrom. A heliocentric correction
is attempted, but a failed correction leaves the uncorrected wavelength grid
and a recorded warning. Inspect ``SPECSYS`` and the correction metadata before
combining spectra across observations.

Reference responses are interpolated in log response using airmass. Extrapolation
and poorly separated reference airmasses produce warnings. Resolution matching
requires usable information about both the observations and templates. Unknown
template wavelength conventions are not guessed.

These assumptions should be considered alongside the trace, wavelength, and
response diagnostics when assessing a reduction.
