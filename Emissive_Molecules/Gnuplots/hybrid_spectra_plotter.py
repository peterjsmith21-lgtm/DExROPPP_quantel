import numpy as np
import matplotlib.pyplot as plt
import argparse
import re

def parse_spectrum(inputfile):
    '''
    Parse a gnuplot file and return the plot command string.

    Args:
        inputfile (str): Path to the gnuplot file.
    Returns:
        plot_cmd (str): Cleaned plot command string.
    '''
    with open(inputfile, 'r') as file:
        lines = file.readlines()

    last_line = lines[-1]
    start_index = last_line.find('p ') + 2
    end_index = last_line.find(' lw 3 dt 1')

    plot_cmd = last_line[start_index:end_index]
    plot_cmd = plot_cmd.replace('0inf', '0')
    print(f"Cleaned expression: {plot_cmd[:100]}...")
    return plot_cmd


def evaluate_spectrum(plot_cmd, wavelength):
    '''
    Evaluate a plot command over a wavelength array.

    Args:
        plot_cmd (str): Cleaned gnuplot plot command string.
        wavelength (ndarray): Array of wavelength values in nm.
    Returns:
        broad (ndarray): Array of broadened spectral intensities.
    '''
    broad = np.zeros_like(wavelength)
    for i, x in enumerate(wavelength):
        broad[i] = eval(plot_cmd)
    return broad

def boltzmann_weights(delta_e_ev, temperature=298.15):
    '''
    Compute Boltzmann weights for singlet and triplet states.

    Args:
        delta_e_ev (float): Singlet-triplet energy gap in eV (E_singlet - E_triplet).
                            Positive means triplet is lower in energy.
        temperature (float): Temperature in Kelvin. Default 298.15 K.
    Returns:
        w_singlet (float): Boltzmann weight of singlet state.
        w_triplet (float): Boltzmann weight of triplet state.
    '''
    kb = 8.617333e-5  # Boltzmann constant in eV/K
    kT = kb * temperature

    z_triplet = 3.0 * np.exp(0.0 / kT)
    z_singlet = 1.0 * np.exp(-delta_e_ev / kT)
    z_total = z_triplet + z_singlet

    w_triplet = z_triplet / z_total
    w_singlet = z_singlet / z_total

    print(f'\nBoltzmann weights at {temperature} K (delta_E = {delta_e_ev:.4f} eV):')
    print(f'  Singlet weight: {w_singlet:.4f}')
    print(f'  Triplet weight: {w_triplet:.4f}\n')

    return w_singlet, w_triplet

# Argument parser setup
parser = argparse.ArgumentParser(description="Plots the computed spectrum produced by ExROPPP")
parser.add_argument('singlet_file', type=str, help="Gnuplot file for singlet spectrum")
parser.add_argument('triplet_file', type=str, help="Gnuplot file for triplet spectrum")
parser.add_argument('molecule_name', type=str, help="The name of the molecule")
parser.add_argument('delta_e', type=float, help="Singlet-triplet energy gap in eV (E_singlet - E_triplet).")
parser.add_argument('--temperature', type=float, default=298.15,
                    help="Temperature in Kelvin for Boltzmann weighting. Default 298.15 K.")
parser.add_argument('--xmin', type=float, default=250,
                    help="Minimum wavelength to plot in nm. Default 300.")
parser.add_argument('--xmax', type=float, default=800,
                    help="Maximum wavelength to plot in nm. Default 800.")
parser.add_argument('--expfile', type=str, help="Experimental data file")
args = parser.parse_args()

# Evaluate over the full spectral range (200-1000 nm) regardless of plot window,
# so that normalisation is never affected by the chosen x-axis limits.
wavelength = np.linspace(200, 1000, 1601)

# Parse spectra
singlet_cmd = parse_spectrum(args.singlet_file)
triplet_cmd = parse_spectrum(args.triplet_file)

mask = (wavelength >= args.xmin) & (wavelength <= args.xmax)

# Normalise using the maximum raw oscillator strength across both spectra.
# This is independent of wavelength range and Lorentzian overlap artifacts.

broad_singlet = evaluate_spectrum(singlet_cmd, wavelength) 
broad_triplet = evaluate_spectrum(triplet_cmd, wavelength)

# Compute Boltzmann weights and combine
w_singlet, w_triplet = boltzmann_weights(args.delta_e, args.temperature)
broad_combined = w_singlet * broad_singlet + w_triplet * broad_triplet

singlet_max = np.max(broad_singlet[mask])
triplet_max = np.max(broad_triplet[mask])
single_max = max(singlet_max, triplet_max)
combined_max = np.max(broad_combined[mask])

# Normalise combined spectrum to 1
broad_singlet = broad_singlet / single_max
broad_triplet = broad_triplet / single_max
broad_combined = broad_combined / combined_max

# Plot — xlim is now freely settable without affecting normalisation
plt.style.use('seaborn-v0_8-paper')
fig, ax = plt.subplots(figsize=(6, 4.8))

ax.plot(wavelength, broad_singlet, color='darkblue',  linestyle='dotted', linewidth=1.2, label='Singlet')
ax.plot(wavelength, broad_triplet, color='darkgreen', linestyle='dotted', linewidth=1.2, label='Triplet')
ax.plot(wavelength, broad_combined, color='firebrick', linewidth=2, label='Weighted Combination')

if args.expfile is not None:
    exp_wl, exp_abs = [], []
    with open(args.expfile, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip().strip('"')
            parts = line.split(',')
            if len(parts) == 2:
                try:
                    exp_wl.append(float(parts[0].strip()))
                    exp_abs.append(float(parts[1].strip()))
                except ValueError:
                    pass
    print(f"Parsed {len(exp_wl)} rows")
    exp_wl = np.array(exp_wl)
    exp_abs = np.array(exp_abs)
    #exp_abs = exp_abs / np.max(exp_abs)
    ax.plot(exp_wl, exp_abs, color='black', linestyle='--', label='Experimental', linewidth=2)

ax.set_title(f'{args.molecule_name}')
ax.set_xlabel('Wavelength / nm')
ax.set_ylabel('Normalised Absorbance')
ax.set_xlim(args.xmin, args.xmax)
ax.set_ylim(0, 1.05)
ax.legend()

plt.tight_layout()
plt.savefig(f'{args.molecule_name}_spectrum.png', dpi=400, transparent=False)
print(f"Spectrum saved to {args.molecule_name}_spectrum.png")