Spectral grids, responses, and relative flux
============================================

Read extracted spectra, apply wavelength and density transformations, fit
reference-star responses, and produce fixed-window relative-flux products.
See :doc:`../outputs` for FITS conventions and :doc:`../science` for the
conditional statistical error model.

.. currentmodule:: speclibparallel

Public interfaces and helpers
-----------------------------

.. autoclass:: ResponseError

.. autoclass:: ArraySpectrum

.. autofunction:: bin_edges

.. autofunction:: density_operator

.. autofunction:: resample_density

.. autofunction:: smoothing_operator

.. autofunction:: read_array_spectrum

.. autofunction:: read_spectrum

.. autofunction:: get_airmass

.. autofunction:: dispcor

.. autofunction:: convert_vacuum_density

.. autofunction:: load_template

.. autofunction:: get_ref_template

.. autofunction:: estimate_arc_fwhm

.. autofunction:: generate_response_spline

.. autofunction:: generate_response_file

.. autofunction:: get_group_response

.. autofunction:: response_correction

Internal implementation helpers
-------------------------------

Names beginning with an underscore document implementation details.
They may change without preserving an external calling contract.

.. autofunction:: _quality_header

.. autofunction:: _quality_array

.. autofunction:: _project_quality

.. autofunction:: _read_extraction_samples

.. autofunction:: _selected_samples

.. autofunction:: _warning

.. autofunction:: _seed

.. autofunction:: _frame_identity

.. autofunction:: _linear_operator

.. autofunction:: _wavelength_header

.. autofunction:: _grid_from_header

.. autofunction:: _resolve_coordinate

.. autofunction:: _dispcor_impl

.. autofunction:: _native_source

.. autofunction:: _template_settings

.. autofunction:: _gaussian

.. autofunction:: _line_mask

.. autofunction:: _spline_fit

.. autofunction:: _prepare_response

.. autofunction:: _response_ensemble

.. autofunction:: _plot_response

.. autofunction:: _load_response

.. autofunction:: _group_response

.. autofunction:: _project_group

.. autofunction:: _noise_realizations

.. autofunction:: _relative_normalization_window

.. autofunction:: _correct_one

.. autofunction:: _response_correction_impl

