# README

## Overview
This repository contains Python scripts designed for data analysis and visualization for ExROPPP. Below are the details for each script:

## Prerequisites

Before running any of the Python scripts described below, make sure you are in the `Supporting_Script` directory:

```bash
cd Supporting_Script

### 1. `data_analysis.py`
This script analyzes experimental and calculated data for D1 and Bright states, calculates statistical metrics (e.g., RMSD, MAD, Spearman Rank, $\text{R}^2$), and generates scatter plots.

#### Features:
Automatically identifies `D1` and `Bright` energy files in the given directory.
- Performs statistical calculations:
  - RMSD (Root Mean Square Deviation)
  - MAD (Mean Absolute Deviation)
  - Spearman Rank Correlation
  - $\text{R}^2$ Correlation
- Generates three types of scatter plots:
    1. **Combined** plot: D1 and Bright states together.
    2. **D1-only** plot.
    3. **Bright-only** plot.
- Saves results and plots in the input directory.

#### Usage:
```bash
python data_analysis.py input_dir
```

#### Arguments:
- `<input_dir>`: (required): The directory/folder name containing the D1 and Bright state files.


#### Output:
- A text file `combined_results.txt` containing RMSD, MAD, Spearman Rank, and $\text{R}^2$ for:
  - Combined data
  - D1 state
  - Bright state
- Scatter plots:
  - `combined_combined_scatter.pdf` for combined data
  - `d1_D1_scatter.pdf` for D1 state
  - `bright_Bright_scatter.pdf` for Bright state

---

### 2. `plot_fitness.py`
This script reads a file containing fitness values across iterations and plots fitness against iterations.

#### Features:
- Parses fitness data from a file.
- Plots fitness as a function of iterations.

#### Usage:
```bash
python read_fitness.py <filepath> --data_start <data_start> --data_end <data_end>
```

#### Arguments:
- `<filepath>`(required): Path to the input file containing fitness data.
- `<data_start>`(optional, type = int): Start index for slicing the data (default: 0, beginning of the iteration).
- `<data_end>`(optional, type = int): End index for slicing the data (default: -1, end of the iteration).

#### Output:
- A plot of fitness vs. iterations for the specified range displayed in the default matplotlib viewer 

#### Example:
This plots the fitness values for all 200 iterations.
```bash
python read_fitness.py fitness_log.txt
```
To plot fitness values for iterations 50 to 150:
```bash
python read_fitness.py fitness_data.txt 10 50
```

---

### 3. `plot_parameters.py`
This script extracts and visualizes specific parameters from machine learning output files.

#### Features:
- Extracts a specific parameter from machine learning output.
- Supports parameter types (`alpha`, `A`, `B`, `U`, `r0`) and atomic types (`C`, `N1`, `N2`, `Cl`).
- Plots the parameter values across iterations.

#### Usage:
```bash
python read_parameters.py <output_file> <param>
```

#### Arguments:
- `<output_file>`(required): Path to the output file from the machine learning process.
- `<param>`(required): Parameter to extract and plot, formatted as `<param_type>-<atom_type>` (e.g., `U-Cl`, `alpha-C`).

#### Output:
- A plot showing the extracted parameter values across iterations.

#### Example:
```bash
python read_parameters.py ml_output.txt U-Cl
```

---

### 4. `MO_visualizer_circle.py`

#### Features:
- Extracts and visualizes the converged Fock orbitals as Spheres obtained with ExROPPP.py or ExROPPP_train.py in Converged_orbiatls folder
- Plot can be saved as PDF by changing `savefic` to `True`in `MO_input.py`

#### Usage:
```bash
python MO_visualizer_circle.py MO_input
```

#### Output:
- A plot showing the converged focked orbitals



---

### 5. `MO_visualizer_spherical_harmonics.py`

#### Features:
- Extracts and visualizes the converged Fock orbitals using Spherical Harmonics functions (2Pz Orbitals) obtained with ExROPPP.py or ExROPPP_train.py in Converged_orbiatls folder
- Plot can be saved as PDF by changing `savefic` to `True`in `MO_input.py`

#### Usage:
```bash
python MO_visualizer_spherical_harmonics.py MO_input
```

#### Output:
- A plot showing the converged focked orbitals


---

## Installation
1. Clone the repository:
```bash
git clone <repository_url>
```
2. Install the required Python packages:
```bash
pip install numpy matplotlib scipy
```

---

## License
This project is licensed under the --- License.

---

## Author
Developed by 
