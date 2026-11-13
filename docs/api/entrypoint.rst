Command-line entry point
========================

The entry module parses paths and forwards work to the library controller.
It also preserves stage aliases for existing callers; those aliases are
documented once under their defining functions in ``speclibparallel``.
For executable examples, see :doc:`../quickstart`.

.. currentmodule:: spec_reduction_parallel

Public interfaces and helpers
-----------------------------

.. autofunction:: make_group_masters_worker

.. autofunction:: process_single_frame_worker

.. autofunction:: main

.. autofunction:: cli

