API reference
=============

The reference below reads signatures and docstrings from the installed local
source during the Sphinx build. No observation reduction is executed. Public
interfaces appear before internal implementation helpers on each page.

All module-level functions and classes defined in the two production modules
are included. Nested local helpers are implementation details of their parent
functions and can be inspected through the source-code links.

.. toctree::
   :maxdepth: 2

   entrypoint
   runtime
   discovery_trace
   calibration
   horne
   anchor_covariance
   wavelength
   response
   legacy_continuum

Modules
-------

.. automodule:: speclibparallel

.. automodule:: spec_reduction_parallel
