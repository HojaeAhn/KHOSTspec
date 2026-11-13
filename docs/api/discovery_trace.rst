Discovery and aperture tracing
==============================

Read and classify observation metadata, associate frame groups, and trace
the spatial aperture. The operating controller uses one aperture. Object
catalogs and grouping constraints are described in :doc:`../inputs`.

.. currentmodule:: speclibparallel

Public interfaces and helpers
-----------------------------

.. autofunction:: gaussian_jit

.. autofunction:: gaussian

.. autofunction:: inv_median

.. autofunction:: fit_gaussian_peak

.. autofunction:: get_ccd_temp

.. autofunction:: parse_library_filename

.. autofunction:: find_nearest_dark_exposure

.. autofunction:: norm_obj

.. autofunction:: get_header_any

.. autofunction:: get_jd

.. autofunction:: get_exptime

.. autofunction:: classify

.. autoclass:: FileRec

.. autoclass:: Group
   :members: finalize
   :member-order: bysource

.. autoclass:: Summary

.. autofunction:: scan_fits

.. autofunction:: build_runs

.. autofunction:: pick_before_after

.. autofunction:: attach_nearest

.. autofunction:: expand_target_results

.. autofunction:: file_sort

.. autofunction:: run_trace

.. autofunction:: discover_files

.. autofunction:: parse_trace_coeffs

.. autofunction:: get_lambda_from_WLCAL

.. autofunction:: make_edges_from_centers

.. autofunction:: cr_reject

.. autofunction:: find_emission

