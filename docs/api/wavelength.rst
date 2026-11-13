Arc wavelength fitting
======================

Match detected arc peaks to reference wavelengths and fit the wavelength
solution. Applying that solution, resampling, and velocity-frame handling
are documented under :doc:`response`. Inspect wavelength diagnostics before
using the calibrated products.

.. currentmodule:: speclibparallel

Public interfaces and helpers
-----------------------------

.. autofunction:: robust_wavelength_solution

.. autofunction:: get_wavelength_solution

Internal implementation helpers
-------------------------------

Names beginning with an underscore document implementation details.
They may change without preserving an external calling contract.

.. autofunction:: _wavelength_vote

