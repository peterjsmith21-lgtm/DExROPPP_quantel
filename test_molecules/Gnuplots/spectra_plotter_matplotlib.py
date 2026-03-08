import numpy as np
import matplotlib.pyplot as plt
import argparse

# Argument parser setup
parser = argparse.ArgumentParser(description="Plots the computed spectrum produced by ExROPPP")
parser.add_argument('inputfile', type=str, help="Gnuplot file to be plotted")
parser.add_argument('molecule_name', type=str, help="The name of the molecule")
args = parser.parse_args()
wavelength = np.linspace(250, 650, 405)
broad = np.zeros_like(wavelength)

with open(args.inputfile, 'r') as file:
    lines = file.readlines(   )

    end_index = lines[-1].find(' lw 3 dt 1')
    newline = lines[-1][2:end_index]
    print(newline)
    str = newline
for i, x in enumerate(wavelength):
    broad[i] = eval(str)
broad / np.max(broad)
plt.style.use('seaborn-v0_8-paper')
plt.plot(wavelength, broad, color = 'teal')
plt.title(f'{args.molecule_name}')  
plt.xlabel('Wavelength/nm')
plt.ylabel('Normalised Absorbance')
plt.savefig(f'{args.molecule_name} Spectrum')