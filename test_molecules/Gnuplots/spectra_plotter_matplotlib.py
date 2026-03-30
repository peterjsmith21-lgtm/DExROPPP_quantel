import numpy as np
import matplotlib.pyplot as plt
import argparse

# Argument parser setup
parser = argparse.ArgumentParser(description="Plots the computed spectrum produced by ExROPPP")
parser.add_argument('inputfile', type=str, help="Gnuplot file to be plotted")
parser.add_argument('molecule_name', type=str, help="The name of the molecule")
args = parser.parse_args()
wavelength = np.linspace(250, 850, 605)
broad = np.zeros_like(wavelength)

with open(args.inputfile, 'r') as file:
    lines = file.readlines(   )

    # Find the start and end of the plotting command
    last_line = lines[-1]
    start_index = last_line.find('p ') + 2
    end_index = last_line.find(' lw 3 dt 1')
    
    # Extract and sanitize the string
    plot_cmd = last_line[start_index:end_index]
    
    # 1. Replace gnuplot's "0inf" with "0"
    plot_cmd = plot_cmd.replace('0inf', '0')
    print(f"Cleaned expression: {plot_cmd[:100]}...")

for i, x in enumerate(wavelength):
    broad[i] = eval(plot_cmd)
if np.max(broad) > 0:
    broad = broad / np.max(broad)
plt.style.use('seaborn-v0_8-paper')
plt.plot(wavelength, broad, color = 'teal')
plt.title(f'{args.molecule_name}')  
plt.xlabel('Wavelength/nm')
plt.ylabel('Normalised Absorbance')
plt.savefig(f'{args.molecule_name} Spectrum XCIS')