Continuum utilities and legacy response helpers
===============================================

These continuum and polynomial-response routines remain available for
direct callers and earlier workflows. The current file-reduction controller
uses :func:`speclibparallel.generate_response_file` and the response
propagation described in :doc:`response`.

.. currentmodule:: speclibparallel

Public interfaces and helpers
-----------------------------

.. autofunction:: robust_sigma_logspec

.. autofunction:: get_variable_mask

.. autofunction:: get_adaptive_p

.. autofunction:: als_continuum

.. autofunction:: baseline_als

.. autofunction:: dual_als_continuum_core

.. autofunction:: als_continuum_ref_QSI

.. autofunction:: als_continuum_ref_Touptek

.. autofunction:: generate_response_polyfit

.. autofunction:: make_response

