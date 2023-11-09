
# excipy

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Authors**: Edoardo Cignoni, Lorenzo Cupellini, Benedetta Mennucci

Machine-learning models for a fast estimation of excitonic Hamiltonians [1].
Excipy provides estimates of site energies and excitonic couplings for chlorophylls *a* and *b* in vacuum, in electrostatic embedding QM/MM (EE QM/MM), and in polarizable embedding QM/MM (QM/MMPol).

## Install

You may want to create a dedicated Python environment for `excipy`.
Currently, Python 3.7 is needed in order to use Amber's `pytraj`.
`excipy` requires NumPy and Cython installed before proceeding with the installation.

For example, you can create a dedicated environment with conda:

```bash
conda create -n excipy -c conda-forge python=3.7 numpy cython
conda activate excipy
```

At this point you can install this package via the standard procedure with `pip`.

```bash
pip install .
```

At this point you should be ready to go.
You can type `excipy` to check if the command line executable works fine.


## Command line interface (CLI)

`excipy` comes with a command line interface.
If you have installed `excipy` as described above, an executable called `excipy` should be available to you.
The executable computes:

* Vacuum, EE QM/MM, and QM/MMPol transition (TrEsp) charges;

* Vacuum, EE QM/MM, and QM/MMPol Coulomb couplings;

* Vacuum, and EE QM/MM transition dipoles;

* Vacuum, EE QM/MM, and QM/MMPol excitation energies;

Currently, `excipy` models are trained on chlorophylls *a* and *b*.
For a description of the ML models implemented in `excipy`, we refer to Refs. [1] and __FILLME__.
These quantities can be computed along all-atom Molecular Dynamics trajectories.

#### CLI: usage

```bash
    8888888888 Y88b   d88P  .d8888b. 8888888 8888888b. Y88b   d88P
    888         Y88b d88P  d88P  Y88b  888   888   Y88b Y88b d88P
    888          Y88o88P   888    888  888   888    888  Y88o88P
    8888888       Y888P    888         888   888   d88P   Y888P
    888           d888b    888         888   8888888P"     888
    888          d88888b   888    888  888   888           888
    888         d88P Y88b  Y88b  d88P  888   888           888
    8888888888 d88P   Y88b  "Y8888P" 8888888 888           888

usage: /home/software/miniconda3/envs/excipy/bin/excipy [-h] -c COORDINATES -p
                                                        PARAMETERS -r
                                                        RESIDUE_IDS
                                                        [RESIDUE_IDS ...]
                                                        [-f FRAMES [FRAMES ...]]
                                                        [-t TURNOFF_MASK]
                                                        [--cutoff CUTOFF]
                                                        [--no_coup]
                                                        [--no_siten]
                                                        [--pol_cutoff POL_CUTOFF]
                                                        [--no_coup_pol]
                                                        [--no_site_pol]
                                                        [--no_site_env]
                                                        [--database_folder DATABASE_FOLDER]
                                                        [--outfile OUTFILE]

Regression-based estimation of the Coulomb coupling.

optional arguments:
  -h, --help            show this help message and exit

required:
  -c COORDINATES, --coordinates COORDINATES
                        Molecular Dynamics trajectory coordinates (.nc, .xtc,
                        ...)
  -p PARAMETERS, --parameters PARAMETERS
                        Molecular Dynamics parameters/topology (.prmtop, .gro,
                        ...)
  -r RESIDUE_IDS [RESIDUE_IDS ...], --residue_ids RESIDUE_IDS [RESIDUE_IDS ...]
                        Residue IDs (e.g., 1 to select the first residue)

optional:
  -f FRAMES [FRAMES ...], --frames FRAMES [FRAMES ...]
                        Frames to load, [start, stop, step] format, one-based
                        count.
  -t TURNOFF_MASK, --turnoff_mask TURNOFF_MASK
                        Amber mask to turn off (no electrostatics) some part
                        of the MM region (e.g., ':1' to turn off
                        electrostatics of resid 1)
  --cutoff CUTOFF       Distance cutoff in Angstrom (no coupling for molecules
                        farther than cutoff). NOTE: if a list of coupling is
                        provided by the --coup_list option, the cutoff is
                        ignored.
  --coup_list COUP_LIST [COUP_LIST ...]
                        Residues pairs you want to compute the couplings on,
                        e.g. 664_665, 664_666. Has the priority on --cutoff
                        option.
  --no_coup             Whether to skip computing Coulomb couplings between
                        chlorophylls.
  --no_siten            Whether to skip computing site energies between
                        chlorophylls.
  --pol_cutoff POL_CUTOFF
                        Polarization cutoff in Angstrom (no polarization for
                        molecules farther than cutoff).
  --no_coup_pol         Whether to not compute the MMPol contribution to the
                        Coulomb coupling.
  --no_site_pol         Whether to not compute the MMPol contribution to the
                        site energies.
  --no_site_env         Whether to not compute the chlorophyll site energy in
                        environment.
  --database_folder DATABASE_FOLDER
                        Absolute path to a custom database folder.

output files:
  --outfile OUTFILE     Output file (HDF5).
```

The `examples` folder contains several examples on how to use `excipy` to post-process a molecular dynamics simulation.
The output file of `excipy` can be read with excipy utilities (see `examples/ex7`and `examples/ex8`).


#### NOTE: Atom names

`excipy` comes with a database where atom names and regression parameters are specified for both chlorophyll *a* (molecule type `CLA`) and chlorophyll *b* (molecule type `CHL`). Atom names are loaded from the database based on the specified molecule `type`. When a molecule is loaded from the trajectory, its coordinates are loaded following the same ordering of the atom names specified in the database: this ensures that the TrEsp charges are estimated correctly. If your trajectories have different atom names for the chlorophyll atoms, you need to modify the atom names in the database in order to match the atom names of your trajectories.

As an example, below is the `CLA.json` file in `excipy/database/atom_names/` where atom names and identical atoms are specified for chlorophyll *a*:

```json
{"names": "MG CHA CHB CHC CHD NA C1A C2A C3A C4A CMA CAA CBA NB C1B C2B C3B C4B CMB CAB CBB NC C1C C2C C3C C4C CMC CAC CBC ND C1D C2D C3D C4D CMD CAD OBD CBD CGD O1D O2D CED HHB HHC HHD H2A H3A 1HMA 2HMA 3HMA 1HAA 2HAA 1HMB 2HMB 3HMB HAB 1HBB 2HBB 1HMC 2HMC 3HMC 1HAC 2HAC 1HBC 2HBC 3HBC 1HMD 2HMD 3HMD HBD 1HED 2HED 3HED", "identical": [["1HMA", "2HMA", "3HMA"], ["1HMB", "2HMB", "3HMB"], ["1HMC", "2HMC", "3HMC"], ["1HBC", "2HBC", "3HBC"], ["1HMD", "2HMD", "3HMD"], ["1HED", "2HED", "3HED"]]}
```
so if, e.g., your methyl hydrogens "1HMA", "2HMA", and "3HMA" are instead named "HMA1", "HMA2", and "HMA3", you only need to change their names in the `CLA.json` file. Note that their name should also be changed in the `identical` entry, which specifies which atoms are considered identical.


## References

[1]: Cignoni, Edoardo, Lorenzo Cupellini, and Benedetta Mennucci. “A Fast Method for Electronic Couplings in Embedded Multichromophoric Systems.” Journal of Physics: Condensed Matter (May 2022).

[2]: Cignoni, Edoardo, Lorenzo Cupellini, and Benedetta Mennucci. "Machine Learning Exciton Hamiltonians in Light-Harvesting Complexes." Journal of Chemical Theory and Computation (January 2023).
