# README for Diradical ExROPPP Repository

## Overview
The **Diradical ExROPPP** repository is a computational framework designed for semiempirical excited-state calculations based on the Pariser-Parr-Pople (PPP) model for open-shell diradical systems. 
The repository supports two types of calculations in two separate codes:
1. **`ExROPPP.py`**: Extended to support open-shell radicals, implementing **XCIS** (Extended Configuration Interaction Singles - D. Maurice and M. Head-Gordon, J. Phys. Chem. 1996, 100, 15, 6131–6137) for calculating excited states of radicals.

---

## Structure of the Codebase

### 1. **`ExROPPP_settings_opt.py`**
- **Purpose**: Provides configuration and constants for open-shell calculations of the ExROPPP Hamiltonian.
- **Key Components**:
  - **State Cutoff**: Defines how many states are computed during diagonalization.
  - **Parameters**: Optimized PPP parameters for specific atom types (carbon, nitrogen, etc.).
  - **Spectra Options**:
    - Lorentzian broadening for spectra.
    - Wavelength-based Full Width at Half Maximum (FWHM) adjustments.
  - **Fundamental Constants**: Includes Planck’s constant, speed of light, etc., for unit conversions.
  - **Bonds cutoffs**: Includes bond cutoffs to compute the relevant dihedral angles.

---

### 2. **`Diradical_ExROPPP.py`** (Open-Shell Systems)
- **Purpose**: Implements **XCIS** (Extended Configuration Interaction Singles) to compute the electronic spectra of open-shell diradicals.
- **Dependencies**:
  - Imports settings and constants from `ExROPPP_settings_opt.py`.
- **Output**:
  - Excited states `.xyz` files.
  - Broadened-spectrum `.gnuplot` visualization files.
  - Converged Fock orbitals `.out` files.
- **Features**:
  - **Spin Purity**: Ensures spin-pure configurations by including single and selected double excitations.
  - Computes vertical excitation energies for organic radicals.
  - Implements state cutoffs (e.g., `states_to_print`) to control eigenvalue computation.
  - Uses Lorentzian wavelength broadening for spectrum generation.

---

### 3. **`main.py`**
- **Purpose**: Entry point for running the framework.
- **Parameters**: Includes pre-configured (optimized) parameter sets for testing.
- **Structure**:
  - Parses molecular geometry from the command line (`python main.py molecule_name`).
  - Imports `ExROPPP`.
  - Calls:
    - `rad_calc` from `ExROPPP.py` for orbital computation and spectra calculation.
    - `gnu_Exroppp` from `ExROPPP.py` for spectra plotting using the output of `rad_calc`.
- **Output**:
  - `gnuplot_script_file`: Gnuplot-compatible visualizations (also compatible with `spectra_plotter_matplotlib.py`).
  - `file.out`: Detailed report of the generated converged Fock orbitals.
  - `file_excitedstates.xyz`: Detailed report of the contributing configurations (in excited basis) for the vertical excitation.

---

### 4. **Supporting Files and Folders**
- **`Converged_orbitals/`**:
  - Stores `.out` files containing Fock orbitals.
- **`Excited_States/`**:
  - Contains `.xyz` files with eigenvalues of XCIS Hamiltonians.
- **`Gnuplots/`**:
  - Houses Gnuplot visualization scripts for generated spectra.
- **`Supporting_Script/spectra_plotter_matplotlib.py`**:
  - Converts Gnuplot data into matplotlib visualizations.

---


## Execution Workflow

1. **Single Molecule Calculation**  
   - **Command**: 
   ```bash
   python main.py molecule_name  
   ```
   - **Input**: Molecular geometry file.
   - **Output**:
     - Converged Fock orbitals.
     - Excited state spectra (`.gnuplot` and `.xyz`).
     - Visualized spectra.

---


## Installation
1. move ```Open_Source``` folder to somewhere on your machine
2. **Install required Python packages**:
   ```bash
   pip install numpy matplotlib scipy
   ```
## License
This project is licensed under the creative commons CC-BY licence.


## Author
Main code written by Joey Kielty was adapted from the original ExROPPP code written by Kevin Ma and James D. Green. 
Initial PPP code adapted from initial closed-shell code of Timothy J. H. Hele and Eric G. Fuemmeler. 
MO visualiser code written by Kevin Ma. 

- Cite as: J. Chem. Phys. 160, 164110 (2024) and arXiv:2412.10149 [physics.chem-ph]