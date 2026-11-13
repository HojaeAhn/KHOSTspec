Detector calibration and master selection
=========================================

Validate detector compatibility, select dark masters, and propagate
calibration variance and masks. The master-selection policy and required
observation metadata are described in :doc:`../inputs`.

.. currentmodule:: speclibparallel

Public interfaces and helpers
-----------------------------

.. autoclass:: CalibrationMismatchError

.. autofunction:: validate_calibration

.. autofunction:: load_ccd

.. autofunction:: make_bias_master

.. autofunction:: make_dark_master

.. autofunction:: library_dark_candidates

.. autofunction:: select_dark_master

.. autofunction:: basic_calibrate

.. autofunction:: combine_arc

.. autofunction:: combine_flat

.. autofunction:: calibrate_science_frames

Internal implementation helpers
-------------------------------

Names beginning with an underscore document implementation details.
They may change without preserving an external calling contract.

.. autofunction:: _header_value

.. autofunction:: _temperature

.. autofunction:: _variance

.. autofunction:: _mask

.. autofunction:: _detector_parameters

.. autofunction:: _trim

.. autofunction:: _master_id

.. autofunction:: _read_raw_strip

.. autofunction:: _clipped_weights

.. autofunction:: _clipped_mean_variance_factor

.. autofunction:: _combine_raw_master

.. autofunction:: _required_variance

.. autofunction:: _combine_calibrated

