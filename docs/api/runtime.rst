Run orchestration and product publication
=========================================

Schedule frame-level work, record provenance and failures, and publish
completed products through verified staging.
:func:`speclibparallel.main` is the programmatic pipeline entry point.
For normal use, start with :doc:`../quickstart`.

.. currentmodule:: speclibparallel

Public interfaces and helpers
-----------------------------

.. autofunction:: json_default

.. autofunction:: file_sha256

.. autofunction:: write_json

.. autofunction:: new_run

.. autofunction:: failure

.. autofunction:: warning_text

.. autofunction:: worker_count

.. autofunction:: completed_jobs

.. autofunction:: final_status

.. autofunction:: copy_verified

.. autofunction:: publication_files

.. autofunction:: publish_products

.. autofunction:: relocate_published_paths

.. autofunction:: checked_archive_tree

.. autofunction:: publication_workspace

.. autofunction:: preserve_unmanaged_products

.. autofunction:: install_products

.. autofunction:: cleanup_run_work

.. autoclass:: RunJournal
   :members: register_input_files, emit, record, flush, finish, finish_work_cleanup, record_storage_failure
   :member-order: bysource

.. autofunction:: make_site

.. autofunction:: frame_memory_estimate

.. autofunction:: make_group_masters_worker

.. autofunction:: process_single_frame_worker

.. autofunction:: find_library_darks

.. autofunction:: arc_key

.. autofunction:: main

