import numpy as np
import matplotlib.pyplot as plt
import argparse

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

    # Triplet has degeneracy 3, singlet has degeneracy 1
    # Setting triplet energy as reference (E=0), singlet is at delta_e_ev
    z_triplet = np.exp(0.0 / kT)
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
args = parser.parse_args()

wavelength = np.linspace(250, 850, 605)

# Parse and evaluate both spectra
singlet_cmd = parse_spectrum(args.singlet_file)
triplet_cmd = parse_spectrum(args.triplet_file)

broad_singlet = evaluate_spectrum(singlet_cmd, wavelength)
broad_triplet = evaluate_spectrum(triplet_cmd, wavelength)

# Normalise each spectrum independently before weighting
if np.max(broad_singlet) > 0:
    broad_singlet = broad_singlet / np.max(broad_singlet)
if np.max(broad_triplet) > 0:
    broad_triplet = broad_triplet / np.max(broad_triplet)

# Compute Boltzmann weights and combine
w_singlet, w_triplet = boltzmann_weights(args.delta_e, args.temperature)
broad_combined = w_singlet * broad_singlet + w_triplet * broad_triplet

# Normalise combined spectrum
if np.max(broad_combined) > 0:
    broad_combined = broad_combined / np.max(broad_combined)

# Plot
plt.style.use('seaborn-v0_8-paper')
fig, ax = plt.subplots()

ax.plot(wavelength, broad_singlet, color='steelblue',  linestyle='--', linewidth=1.5, label='Singlet')
ax.plot(wavelength, broad_triplet, color='firebrick',  linestyle='--', linewidth=1.5, label='Triplet')
ax.plot(wavelength, broad_combined, color='teal', linewidth=2.0, label='Weighted Combination')

ax.set_title(f'{args.molecule_name}')
ax.set_xlabel('Wavelength / nm')
ax.set_ylabel('Normalised Absorbance')
ax.legend()

plt.tight_layout()
plt.savefig(f'{args.molecule_name}_spectrum.png', dpi=300)
print(f"Spectrum saved to {args.molecule_name}_spectrum.png")