import numpy as np
from numpy.distutils.core import setup, Extension
from Cython.Build import cythonize

ext_modules = []

ext_modules += cythonize(
    Extension(
        "excipy.clib.retain_full_residues",
        ["excipy/cython/retain_full_residues.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    )
)

ext_modules += cythonize(
    Extension(
        "excipy.clib.distances_diffmask",
        ["excipy/cython/distances_diffmask.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    )
)

ext_modules += cythonize(
    Extension(
        "excipy.clib.map_polarizable_atoms",
        ["excipy/cython/map_polarizable_atoms.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    )
)

ext_modules += cythonize(
    Extension(
        "excipy.clib.tmu",
        ["excipy/cython/tmu.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-fopenmp"],
        extra_link_args=["-lgomp"],
    )
)

setup(ext_modules=ext_modules)
