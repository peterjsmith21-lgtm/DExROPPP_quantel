import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv

# Argument parser setup
parser = argparse.ArgumentParser(description="Plots the computed spectrum produced by ExROPPP")
parser.add_argument('inputfile', type=str, help="Gnuplot file to be plotted")
parser.add_argument('molecule_name', type=str, help="The name of the molecule")
parser.add_argument('--expfile', type=str, help="Experimental data file")
args = parser.parse_args()
wavelength = np.linspace(200, 600, 800)
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
plt.figure(figsize=(10, 6))
plt.plot(wavelength, broad, color = 'darkgreen', label='D-ExROPPP', linewidth=2.5)

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
    plt.plot(exp_wl, exp_abs, color='black', linestyle='--', label='Experimental', linewidth=2.5)
    plt.legend(fontsize=36)

#plt.title(f'UV absorption spectrum for {args.molecule_name}', fontsize=18)  
#plt.xlabel('Wavelength / nm', fontsize=18)
plt.xlim(380, 480)
#plt.ylabel('Normalised Absorbance', fontsize=18)
plt.ylim(0.006, 0.018)
plt.xticks(fontsize=22)
#plt.yticks(fontsize=22)
plt.yticks(np.arange(0.008, 0.018, 0.004), fontsize=22)
plt.savefig(f'{args.molecule_name} Spectrum', transparent =True, dpi=400)