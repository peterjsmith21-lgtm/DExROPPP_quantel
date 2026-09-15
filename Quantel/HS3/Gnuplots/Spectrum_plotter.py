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


# Argument parser setup
parser = argparse.ArgumentParser(
    description="Plots a single computed spectrum produced by ExROPPP. "
                "Use this script when E_ST is large and only one spin state "
                "contributes (no Boltzmann weighting needed)."
)
parser.add_argument('spectrum_file', type=str, help="Gnuplot file for the spectrum")
parser.add_argument('molecule_name', type=str, help="The name of the molecule")
parser.add_argument('--state', type=str, default='Calculated',
                    help="Spin state label for the legend, e.g. Singlet or Triplet. Default: 'Calculated'.")
parser.add_argument('--xmin', type=float, default=250,
                    help="Minimum wavelength to plot in nm. Default 250.")
parser.add_argument('--xmax', type=float, default=800,
                    help="Maximum wavelength to plot in nm. Default 800.")
parser.add_argument('--expfile', type=str, help="Experimental data file (CSV, TSV, or space-separated .txt)")
args = parser.parse_args()

# Evaluate over the full spectral range (200-1000 nm) regardless of plot window,
# so that normalisation is never affected by the chosen x-axis limits.
wavelength = np.linspace(200, 1000, 1601)

# Parse and evaluate the spectrum
plot_cmd = parse_spectrum(args.spectrum_file)
broad = evaluate_spectrum(plot_cmd, wavelength)

# Normalise to 1 using the maximum within the plot window.
mask = (wavelength >= args.xmin) & (wavelength <= args.xmax)
spec_max = np.max(broad[mask])
broad = broad / spec_max

# Plot
plt.style.use('seaborn-v0_8-paper')
fig, ax = plt.subplots(figsize=(6, 4.8))

ax.plot(wavelength, broad, color='darkgreen', linewidth=2, label=args.state)

if args.expfile is not None:
    exp_wl, exp_abs = [], []
    with open(args.expfile, 'r', encoding='utf-8-sig') as f:
        for line in f:
            # Strip leading/trailing whitespace and stray quote characters,
            # then split on any mix of commas and whitespace so the same
            # loader handles CSV, TSV, space-separated .txt files, and
            # formats with trailing commas.
            line = line.strip().strip('"')
            tokens = [t.strip().strip('"') for t in re.split(r'[,\s]+', line) if t.strip()]
            if len(tokens) < 2:
                continue
            try:
                exp_wl.append(float(tokens[0]))
                exp_abs.append(float(tokens[1]))
            except ValueError:
                pass   # skip header/label lines
    print(f"Parsed {len(exp_wl)} rows from '{args.expfile}'")
    exp_wl = np.array(exp_wl)
    exp_abs = np.array(exp_abs)
    exp_abs = (exp_abs - np.min(exp_abs)) / (np.max(exp_abs) - np.min(exp_abs))   # normalise to [0, 1]
    ax.plot(exp_wl, exp_abs, color='black', linestyle='--', linewidth=2, label='Experimental')

#ax.set_title(f'{args.molecule_name}')
ax.set_xlabel('Wavelength / nm')
ax.set_ylabel('Normalised Absorbance')
ax.set_xlim(args.xmin, args.xmax)
ax.set_ylim(0, 1.05)
ax.legend()

plt.tight_layout()
plt.savefig(f'{args.molecule_name}_spectrum.png', dpi=400, transparent=False)
print(f"Spectrum saved to {args.molecule_name}_spectrum.png")
