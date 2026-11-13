Signed anchors and correlated uncertainty propagation
=====================================================

Construct independently trained profiles and signed anchors, apply their
calibration, and propagate correlated statistical uncertainty through the
actual finite Horne iteration. See :doc:`../science` for normalization,
support requirements, and uncertainty limitations.

.. currentmodule:: speclibparallel

Public interfaces and helpers
-----------------------------

.. autofunction:: expected_map

.. autofunction:: expected_detector_terms

.. autofunction:: pixel_variance

.. autofunction:: extract_column_physical

.. autofunction:: feature_training_exclusions

.. autofunction:: build_anchor

.. autofunction:: crossfit_profiles

.. autofunction:: build_profile

.. autofunction:: pooling_operator

.. autofunction:: iterated_weight_jacobian

.. autofunction:: calibrated_gradients

.. autofunction:: spatial_operators

.. autofunction:: physical_anchor_covariance

.. autofunction:: correlated_extraction_samples

.. autofunction:: band_covariance

.. autoclass:: ExtractionStageError

.. autofunction:: calibrated_column

.. autofunction:: extract_signed

.. autofunction:: write_ac_extraction

Internal implementation helpers
-------------------------------

Names beginning with an underscore document implementation details.
They may change without preserving an external calling contract.

.. autofunction:: _ring_footprint

.. autofunction:: _ring_median

.. autofunction:: _pooled

.. autofunction:: _local_polynomial_pool

.. autofunction:: _align

.. autofunction:: _anchor_linear_kernel

.. autofunction:: _anchor_fold_columns

.. autofunction:: _anchor_fast_aligned

.. autofunction:: _broadcast_map

.. autofunction:: _calibrate_extraction_result

.. autofunction:: _calibrate_extraction_columns

.. autofunction:: _positive_rejection_pass

.. autofunction:: _pixel_map

.. autofunction:: _validate_settings

.. autofunction:: _negative_pixel_table

.. autofunction:: _sample_provenance

