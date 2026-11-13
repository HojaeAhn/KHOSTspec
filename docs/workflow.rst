Processing workflow
===================

The controller coordinates calibration, independent frame tasks, and final
response correction. Scientific stages and orchestration reside in
``speclibparallel.py``; ``spec_reduction_parallel.py`` exposes the command-line
entry point.

Reduction stages
----------------

1. **Discover groups and build global masters.** Time-order recognized frames,
   associate target and reference groups with calibration observations, combine
   biases, and combine compatible darks grouped by exposure time.
2. **Build shared arcs.** Combine each distinct associated arc set once so that
   applicable frame tasks can reuse it.
3. **Reduce science and reference frames.** Apply bias and dark correction,
   trace the aperture, and extract each spectrum. Extract the associated arc
   along the same trace.
4. **Calibrate wavelengths.** Fit the arc wavelength solution and apply
   dispersion correction. Attempt the heliocentric velocity correction; on
   failure, retain the uncorrected wavelength grid and record the warning.
5. **Fit reference responses.** Compare each reference spectrum with its
   template, matching spectral resolution when supported by the metadata, and
   propagate response uncertainty.
6. **Correct target spectra.** Interpolate the bracketing reference groups' log
   responses in airmass, divide each target spectrum by the response, and
   normalize the nominal flux and uncertainty realizations in a fixed window.
7. **Publish completed products.** Verify selected products before installing
   the completed observation directory. Retain incomplete working runs for
   inspection.

Flat frames are recognized and grouped, but **flat division is not applied**.
The operating pipeline uses one aperture. Review :doc:`inputs` for the current
grouping rules and :doc:`science` for the assumptions behind extraction and
uncertainty propagation.

Parallel execution
------------------

Independent frame tasks use separate worker processes. The default CPU-based
worker limit is ``max(1, CPU_count // 2 - 2)``. Task count and the configured
memory budget further limit concurrency. Setting ``Runtime.max_workers``
overrides the CPU-based preference, but does not remove the task-count or
memory constraints. Numerical threads are limited within the worker stages.

Column batching is a separate mechanism within extraction. Horne fits and
finite-iteration derivatives run in blocks of up to 512 columns. Each column
keeps its own convergence and rejection states. These blocks do not merge
wavelength samples and are separate from profile-training bins and Monte Carlo
draw batches.

Reviewing a run
---------------

Inspect the trace diagnostics, arc-fit residuals, response diagnostics, and the
recorded stage warnings. Wavelength-frame metadata describe whether the
heliocentric correction succeeded. Airmass extrapolation and weak reference
airmass separation are reported rather than silently replaced by a different
interpolation rule.

``run_manifest.json`` contains the final status and per-stage events.
Publication and retained-work behavior are described in :doc:`outputs`.
