import os
import sys
import glob
import pybind11

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class get_pybind_include:
    def __str__(self):
        return pybind11.get_include()


def find_include(lib: str):
    include_dirs = ['/usr/local/include', '/usr/include']
    for include_dir in include_dirs:
        eigen_path = os.path.join(include_dir, lib)
        if os.path.exists(eigen_path):
            return eigen_path
    raise RuntimeError(f'{lib} library not found. Please install {lib} and make sure it is in a standard include path.')

    
boost_include_dir = find_include('boost')
eigen_include_dir = find_include('eigen3')


ext_modules = [
    Extension(
        'TurnpikeMM',
        sources=['src/TurnpikeMM/TurnpikeMM.cpp'],
        include_dirs=[get_pybind_include(), eigen_include_dir, boost_include_dir],
        library_dirs=[boost_library_dir],
        libraries=['boost_system', 'boost_filesystem'],
        language='c++',
        extra_compile_args=['-std=c++20', '-fopenmp-simd', '-fopenmp']
    ),
]


class BuildExt(build_ext):
    def build_extensions(self):
        ct = self.compiler.compiler_type
        if ct == 'unix':
            for e in self.extensions:
                e.extra_compile_args.append('-DVERSION_INFO="%s"' % self.distribution.get_version())

        super(BuildExt, self).build_extensions()


setup(
    name='TurnpikeMM',
    version='0.1',
    author='C. Shane Elder',
    author_email='celder@andrew.cmu.edu',
    description='TurnpikeMM Solver Implementation',
    ext_modules=ext_modules,
    cmdclass=dict(build_ext=BuildExt),
    zip_safe=False,
    install_requires=[
        'pybind11',
        'scipy'
    ]
)
