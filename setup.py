import glob
import os
import sys

from setuptools import setup, find_packages #, Extension
from pybind11.setup_helpers import Pybind11Extension

# the version lives in one place, the package's _version.py
with open(os.path.join(os.path.dirname(__file__), 'src', 'python', 'spatialize', '_version.py')) as vh:
    version = vh.read().split('=')[1].strip().strip('"\'')

libsptlzsrc = os.path.join('src', 'c++', 'libspatialize.cpp')
macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]

# Header files, tracked as build dependencies.
libsptlzheaders = sorted(
    glob.glob(os.path.join('include', 'spatialize', '**', '*.hpp'), recursive=True)
    + glob.glob(os.path.join('src', 'c++', '*.hpp'))
)

extra_compile_args = ['-std=c++17']
extra_link_args = []

def _check_conda_openmp():
    """Inside a conda environment, the linker picks the environment's libomp.dylib (llvm-openmp)
    before Homebrew's. A recent clang (Apple clang 17 or LLVM 19 and later) emits calls to
    __kmpc_dispatch_deinit, which an older llvm-openmp lacks, so the extension would build but fail
    to load ("symbol not found in flat namespace '___kmpc_dispatch_deinit'"). Stop the build with
    the fix instead. See the documentation, Troubleshooting."""
    import re
    import subprocess
    prefix = os.environ.get("CONDA_PREFIX")
    runtime = os.path.join(prefix, "lib", "libomp.dylib") if prefix else None
    if not runtime or not os.path.exists(runtime):
        return
    try:
        version = subprocess.check_output([os.environ.get("CC", "clang"), "--version"],
                                          stderr=subprocess.DEVNULL).decode()
        symbols = subprocess.check_output(["nm", "-gU", runtime], stderr=subprocess.DEVNULL).decode()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return
    m = re.search(r"(Apple )?clang version (\d+)", version)
    if not m:
        return
    major = int(m.group(2))
    emits = major >= 17 if m.group(1) else major >= 19
    if emits and "__kmpc_dispatch_deinit" not in symbols:
        sys.exit(f"""
Spatialize cannot be built against this conda environment's OpenMP runtime:
  {runtime} lacks __kmpc_dispatch_deinit, which {m.group(0)} needs,
  so the extension would install but fail to load. Update the runtime first:

      conda install "llvm-openmp>=19"

  then build again. See the documentation, Troubleshooting.""")


# the '-Wno-error=c++11-narrowing' argument is needed for
# compiling with CLang (OS X)
if sys.platform == 'darwin':
    extra_compile_args += ['-Wno-error=c++11-narrowing']
    # macOS with Clang: use libomp from Homebrew
    import subprocess
    try:
        # Get Homebrew prefix for libomp
        brew_prefix = subprocess.check_output(
            ['brew', '--prefix', 'libomp'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        extra_compile_args += ['-Xpreprocessor', '-fopenmp', f'-I{brew_prefix}/include']
        extra_link_args += [f'-L{brew_prefix}/lib', '-lomp']
        _check_conda_openmp()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback: Homebrew not available or libomp not installed
        # OpenMP pragmas will be ignored at compile time
        print("Warning: OpenMP not found. Install with: brew install libomp")
        pass
elif sys.platform == 'linux':
    # Linux: use standard OpenMP
    extra_compile_args += ['-fopenmp']
    extra_link_args += ['-fopenmp']
elif sys.platform == 'win32':
    # Windows with MSVC
    extra_compile_args = ['/std:c++17', '/Ox']
    extra_compile_args += ['/openmp']

libspatialize_extensions = [
    Pybind11Extension(
        "libspatialize",
        sources=[libsptlzsrc],
        depends=libsptlzheaders,
        include_dirs=[ os.path.join('.', 'include')],#, numpy.get_include()],
        extra_compile_args= extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=macros,
    ),
    # Dawid-Skene EM aggregation of categorical ensembles (spatialize.gs.cat_esi)
    Pybind11Extension(
        "spatialize.gs.cat_esi._dawid_skene",
        sources=[os.path.join('src', 'c++', 'dawid_skene.cpp')],
        include_dirs=[os.path.join('.', 'include')],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

if __name__ == '__main__':
    setup(
        name='spatialize',
        version=version,
        author='ALGES Laboratory',
        author_email='dev@alges.cl',
        description='Python Library for Generative Geostatistics and Spatial Analysis',
        keywords="ESI ESS ensemble spatial analysis",
        url="http://www.alges.cl/",
        long_description=open(os.path.join(os.path.dirname(os.path.realpath(
            __file__)), "README.md")).read(),
        ext_modules=libspatialize_extensions,
        package_dir={'spatialize': os.path.join('src', 'python', 'spatialize')},
        packages=find_packages(os.path.join('src', 'python'), exclude=[".DS_Store", "__pycache__"]),
        include_package_data=True,
        scripts=[],
    )
