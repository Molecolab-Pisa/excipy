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
        "excipy.retain_full_residues",
        ["excipy/cython/retain_full_residues.pyx"],
        include_dirs=[np.get_include()],
    )
)


package_data = {
    "excipy": [
        "database/atom_names/*.json",
        "database/parameters/*.json",
        "database/rescalings/*.json",
        "database/models/*",
        "database/models/*/*/*",
        "database/models/*/*.pb",
    ]
}


packages = [
    "excipy",
]


entry_points = {
    "console_scripts": [
        "excipy=excipy.cli:main",
        "excipy2exat=excipy.cli:excipy2exat",
    ],
}


setup(
    name="excipy",
    version="1.5.0",
    author="Edoardo Cignoni, Lorenzo Cupellini, Benedetta Mennucci",
    author_email="edoardo.cignoni96@gmail.com",
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
        "tensorflow",
        "gpflow",
        "cython",
    ],
)
