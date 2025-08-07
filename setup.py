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
        version='1.0.3',
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
