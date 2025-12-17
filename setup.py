import os
import sys

from setuptools import setup, find_packages #, Extension
from pybind11.setup_helpers import Pybind11Extension

with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as reqh:
    install_requires = reqh.readlines()

libsptlzsrc = os.path.join('src', 'c++', 'libspatialize.cpp')
macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]

extra_compile_args = ['-std=c++17']
extra_link_args = []

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
    extra_compile_args += ['/openmp']

libspatialize_extensions = [
    Pybind11Extension(
        "libspatialize",
        sources=[libsptlzsrc],
        include_dirs=[ os.path.join('.', 'include')],#, numpy.get_include()],
        extra_compile_args= extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=macros,
    ),
]

if __name__ == '__main__':
    setup(
        name='spatialize',
        version='1.0.4',
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
        install_requires=install_requires,
    )
