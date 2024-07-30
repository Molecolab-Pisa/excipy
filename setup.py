import numpy as np
from numpy.distutils.core import setup, Extension
from Cython.Build import cythonize


ext_modules = [
    Extension(
        name="excipy.tmu",
        sources=["excipy/tmu.f90"],
        extra_f90_compile_args=["-O3", "-fopenmp"],
        extra_link_args=["-lgomp"],
    ),
]

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


package_data = {
    "excipy": [
        "database/atom_names/*.json",
        "database/parameters/*/*.json",
        "database/rescalings/*/*.json",
        "database/models/*/*/*.npz",
    ]
}


packages = [
    "excipy",
    "excipy.clib",
    "excipy.models",
]


entry_points = {
    "console_scripts": [
        "excipy=excipy.cli:main",
        "excipy2exat=excipy.cli:excipy2exat",
        "excipy-scan=excipy.cli:excipy_scan",
    ],
}


setup(
    name="excipy",
    version="1.5.1",
    author="Edoardo Cignoni, Elena Betti, Lorenzo Cupellini, Benedetta Mennucci",
    author_email="edoardo.cignoni96@gmail.com,elena15.be@gmail.com",
    packages=packages,
    package_data=package_data,
    ext_modules=ext_modules,
    entry_points=entry_points,
    description="Machine learning models for a fast estimation of excitonic Hamiltonians",
    long_description=open("README.md").read(),
    setup_requires=["numpy", "cython"],
    install_requires=[
        "pytraj",
        "numpy",
        "scipy",
        "tqdm",
        "cython",
    ],
)
