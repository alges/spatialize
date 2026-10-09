.. _troubleshooting:

###############
Troubleshooting
###############

Problems met when installing or running Spatialize, with their cause and their fix.

Symbol not found: ``___kmpc_...``
=================================

**Symptom.** On macOS, importing Spatialize fails with

.. code-block:: text

   ImportError: dlopen(.../libspatialize.cpython-312-darwin.so, 0x0002):
   symbol not found in flat namespace '___kmpc_dispatch_deinit'

or, from version 1.3, with a :class:`~spatialize.SpatializeError` that explains the same.

**Cause.** The compiled extension runs in parallel with OpenMP, whose runtime is the library
``libomp.dylib``. A recent compiler, Apple clang 17 or LLVM 19 and later, emits calls to functions
of the runtime that older runtimes lack, such as ``__kmpc_dispatch_deinit``. Inside a conda
environment the linker takes the environment's runtime, the package ``llvm-openmp``, before
Homebrew's. When the environment holds an older ``llvm-openmp`` than the compiler, the extension
builds and installs, then fails to load.

**Fix.** Install a recent runtime in the same environment,

.. code-block:: bash

   conda install -n <environment> "llvm-openmp>=19"

and run Python again. Spatialize need not be reinstalled, since the extension loads the
environment's runtime by name. The packages that depend on ``llvm-openmp``, such as scikit-learn and
numba, keep working, since a newer runtime serves code compiled against an older one.

**Checking.** The runtime the extension loads, and whether it has the function,

.. code-block:: bash

   otool -L $(python -c "import importlib.util as u; print(u.find_spec('libspatialize').origin)")
   nm -gU "$CONDA_PREFIX/lib/libomp.dylib" | grep kmpc_dispatch_deinit

The first command lists ``@rpath/libomp.dylib``, resolved in the environment's ``lib`` directory.
The second prints the function when the runtime is recent enough.

From version 1.3 the installation stops before building, with the same fix, when it detects an
environment whose runtime is older than the compiler.

Pointing the extension at Homebrew's runtime instead does not help, since scikit-learn already loads
the environment's runtime, and two OpenMP runtimes in one process abort (next section).

OMP: Error #15, ``libomp.dylib`` already initialized
====================================================

**Symptom.** A run stops with ``OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib
already initialized``.

**Cause.** Two copies of the OpenMP runtime were loaded in one process, for instance Homebrew's by
one library and a conda environment's by another.

**Fix.** Keep one runtime per process. In a conda environment, use the environment's
``llvm-openmp`` for every package, building Spatialize inside the environment. The environment
variable ``KMP_DUPLICATE_LIB_OK=TRUE`` lets the run continue, at the risk of wrong results or
crashes, so it serves only as a temporary workaround.

Spatialize was built without OpenMP
===================================

**Symptom.** A warning says that Spatialize was built without OpenMP and runs on a single thread.

**Cause.** No OpenMP runtime was found when the extension was built, so its parallel loops run
serially. ``libspatialize.build_info()`` reports ``'openmp': False``.

**Fix.** Install a runtime, ``brew install libomp`` on macOS or the GNU runtime (``libgomp``) on
Linux, then reinstall Spatialize from source,

.. code-block:: bash

   pip install --force-reinstall --no-binary spatialize spatialize

To run on one thread on purpose and silence the warning, set
``spatialize.session.set(parallel=False)`` (:doc:`reference/session`).

Messages and progress bars
==========================

- **Warnings do not appear, or too many messages do.** The session setting ``verbosity`` sets the
  lowest level shown, warnings by default. ``spatialize.session.set(verbosity="info")`` adds what
  the functions do, ``"error"`` keeps only the errors.
- **Progress bars garble a log file or the output of a continuous integration.** Spatialize draws
  live bars only on a terminal and writes plain lines elsewhere. A terminal whose output is captured
  can be forced to plain lines with ``spatialize.session.set(display="plain")``, or to nothing with
  ``display="silent"``.
- **No progress bar in a notebook.** The bars of a notebook are HTML, which needs IPython's display
  machinery. A front end that shows no HTML can use ``display="plain"``.

Building from source
====================

These concern developers working on Spatialize itself (:doc:`development/index`).

- **An edited header has no effect.** ``python setup.py build_ext`` does not track the headers under
  ``include/spatialize``, so after editing one only ``--force`` rebuilds the extension. The unit tests
  and the guard checks refuse a binary older than the sources.
- **The in-place build loads the wrong runtime.** In a conda environment with an old
  ``llvm-openmp``, an extension built in place against Homebrew's runtime loads the environment's
  one. Update the environment's runtime as in the first section, or run with
  ``DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib``.
