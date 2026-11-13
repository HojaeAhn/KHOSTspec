Horne extraction and bounded column kernels
===========================================

Extract independent detector columns with propagated variance and
sequential positive cosmic-ray rejection. The bounded kernels process up
to 512 columns at a time without combining wavelengths or sharing
convergence states. :func:`speclibparallel.optimal_extract_fits` is the
file-level extraction interface; its A+C path also uses :doc:`anchor_covariance`.

.. currentmodule:: speclibparallel

Public interfaces and helpers
-----------------------------

.. autofunction:: batch_model_weighted_fit

.. autofunction:: batch_extract_physical

.. autofunction:: batch_iterated_weight_jacobian

.. autofunction:: chunked_extract_physical

.. autofunction:: chunked_iterated_weight_jacobian

.. autoclass:: ExtractionResult

.. autofunction:: aperture_geometry

.. autofunction:: estimate_background

.. autofunction:: column_background

.. autofunction:: extract_column

.. autofunction:: optimal_extract_math

.. autofunction:: optimal_extract_one

.. autofunction:: build_profile_map

.. autofunction:: optimal_extract_fits

Internal implementation helpers
-------------------------------

Names beginning with an underscore document implementation details.
They may change without preserving an external calling contract.

.. autofunction:: _extraction_control

.. autofunction:: _batch_optional_gain

.. autofunction:: _batch_rows

.. autofunction:: _batch_vector

.. autofunction:: _batch_masked_sum

.. autofunction:: _batch_weighted_rows

.. autofunction:: _batch_model_rows

.. autofunction:: _batch_physical_rows

.. autofunction:: _batch_rejection_rows

.. autofunction:: _batch_packed

.. autofunction:: _batch_unpacked

.. autofunction:: _batch_chunked

.. autofunction:: _normalise

.. autofunction:: _weighted_fit

.. autofunction:: _model_weighted_fit

.. autofunction:: _rejection_scale

.. autofunction:: _align_for_training

.. autofunction:: _nanmedian

.. autofunction:: _compact_profile_model

.. autofunction:: _read_calibrated

