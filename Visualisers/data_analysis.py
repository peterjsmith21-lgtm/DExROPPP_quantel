import numpy as np
from scipy.stats import spearmanr
import argparse
import matplotlib.pyplot as plt
import os

def read_data_from_file(filename):
    data = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            values = list(map(float, lines[i + 1].strip()[1:-1].split(',')))
            if values[0] == '0' or values[0] == 0:
                continue
            if len(values) == 2:
                data.append([values[0], values[1]])
            elif len(values) == 4:
                data.append([values[0], values[1]])
                data.append([values[2], values[3]])
    return np.array(data)

def calculate_rmsd(data):
    squared_diffs = (data[:, 1] - data[:, 0])**2
    return np.sqrt(np.mean(squared_diffs))

def calculate_mad(data):
    abs_diffs = np.abs(data[:, 1] - data[:, 0])
    return np.mean(abs_diffs)

def calculate_spearman_rank(data):
    experimental = data[:, 0]
    calculated = data[:, 1]
    spearman_corr, _ = spearmanr(calculated, experimental)
    return spearman_corr

def calculate_r2(data):
    experimental = data[:, 0]
    calculated = data[:, 1]
    numerator = np.sum((experimental - calculated)**2)
    denominator = np.sum((experimental - np.mean(experimental))**2)
    R2 = 1 - numerator / denominator
    return R2

def plot_bright(data, label, filename, directory):
    plt.scatter(data[:, 0], data[:, 1], label=label)
    min_value = min(np.min(data[:, 0]), np.min(data[:, 1]))
    max_value = max(np.max(data[:, 0]), np.max(data[:, 1]))
    plt.plot([min_value, max_value], [min_value, max_value], color='royalblue', linestyle='--', label='Ideal')
    plt.xlabel('Experimental Energy / eV', fontsize=20)
    plt.ylabel('Calculated Energy / eV', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(prop={'size': 16})
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.savefig(os.path.join(directory, f'{filename}_scatter.pdf'))
    plt.close()

def plot_d1(data, label, filename, directory):
    plt.scatter(data[:, 0], data[:, 1], label=label)
    min_value = min(np.min(data[:, 0]), np.min(data[:, 1]))
    max_value = max(np.max(data[:, 0]), np.max(data[:, 1]))
    plt.plot([min_value, max_value], [min_value, max_value], color='mediumseagreen', linestyle='--', label='Ideal')
    plt.xlabel('Experimental Energy / eV', fontsize=20)
    plt.ylabel('Calculated Energy / eV', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(prop={'size': 16})
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.savefig(os.path.join(directory, f'{filename}_scatter.pdf'))
    plt.close()

def plot_combined_data(d1_data, bright_data, filename, directory):
    title_font_style = {'family': 'arial',
        'color':  'black',
        'weight': 'normal',
        'size': 30,
        }
    plt.scatter(d1_data[:, 0], d1_data[:, 1], color='green', label='D1 State')
    plt.scatter(bright_data[:, 0], bright_data[:, 1], color='blue', label='Bright State')
    min_value = min(np.min(d1_data[:, 0]), np.min(bright_data[:, 0]), np.min(d1_data[:, 1]), np.min(bright_data[:, 1]))
    max_value = max(np.max(d1_data[:, 0]), np.max(bright_data[:, 0]), np.max(d1_data[:, 1]), np.max(bright_data[:, 1]))
    plt.plot([min_value, max_value], [min_value, max_value], color='orangered', linestyle='--')
    plt.title(f'Literature Parameters', fontdict=title_font_style)
    plt.xlabel('Experimental Energy / eV', fontsize=20, fontdict ={'family': 'arial'} )
    plt.ylabel('Calculated Energy / eV', fontsize=20, fontdict ={'family': 'arial'})
           
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    x_min = np.floor(min_value)
    x_max = np.ceil(max_value)

    # Generate tick locations with a step of 1:
    ticks = np.arange(x_min, x_max + 1, 1)

    plt.xticks(ticks)   # set x-axis ticks at intervals of 1
    plt.yticks(ticks)   # set y-axis ticks at intervals of 1
    plt.legend(prop={'size': 16})
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.savefig(os.path.join(directory, f'{filename}_combined_scatter.svg'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(prog='data analyzer',description="Process D1 and Bright state files.")
    parser.add_argument('input_dir', type=str, help="Path to the directory containing D1 and Bright state files.")
    args = parser.parse_args()

    # Set output directory to input directory
    output_dir = args.input_dir

    # Identify D1 and Bright files
    d1_file = None
    bright_file = None
    for file in os.listdir(args.input_dir):
        if any(keyword in file.lower() for keyword in ['d1']):
            d1_file = os.path.join(args.input_dir, file)
        elif any(keyword in file.lower() for keyword in ['bright', 'brght']):
            bright_file = os.path.join(args.input_dir, file)

    if not d1_file or not bright_file:
        print("Error: Could not find both D1 and Bright state files in the specified directory.")
        return


    # Read data
    d1_data = read_data_from_file(d1_file)
    bright_data = read_data_from_file(bright_file)

    # Combined data
    combined_data = np.vstack([d1_data, bright_data])

    # Perform calculations on combined
    rmsd_combined = calculate_rmsd(combined_data)
    mad_combined = calculate_mad(combined_data)
    spearman_combined = calculate_spearman_rank(combined_data)
    r2_combined = calculate_r2(combined_data)

    # Perform calculations on D1
    rmsd_d1 = calculate_rmsd(d1_data)
    mad_d1 = calculate_mad(d1_data)
    spearman_d1 = calculate_spearman_rank(d1_data)
    r2_d1 = calculate_r2(d1_data)

    # Perform calculations on Bright
    rmsd_bright = calculate_rmsd(bright_data)
    mad_bright = calculate_mad(bright_data)
    spearman_bright = calculate_spearman_rank(bright_data)
    r2_bright = calculate_r2(bright_data)

    # Save combined results
    output_file = os.path.join(output_dir, 'combined_results.txt')
    with open(output_file, 'w') as out_file:
        out_file.write("Combined Results:\n")
        out_file.write(f"RMSD: {rmsd_combined}\n")
        out_file.write(f"MAD: {mad_combined}\n")
        out_file.write(f"Spearman Rank Correlation: {spearman_combined}\n")
        out_file.write(f"R2 Correlation: {r2_combined}\n\n")

        out_file.write("D1 Results:\n")
        out_file.write(f"RMSD: {rmsd_d1}\n")
        out_file.write(f"MAD: {mad_d1}\n")
        out_file.write(f"Spearman Rank Correlation: {spearman_d1}\n")
        out_file.write(f"R2 Correlation: {r2_d1}\n\n")

        out_file.write("Bright Results:\n")
        out_file.write(f"RMSD: {rmsd_bright}\n")
        out_file.write(f"MAD: {mad_bright}\n")
        out_file.write(f"Spearman Rank Correlation: {spearman_bright}\n")
        out_file.write(f"R2 Correlation: {r2_bright}\n")

    # Plot data
    plot_combined_data(d1_data, bright_data, 'combined', output_dir)
    plot_d1(d1_data, 'D1', 'd1', output_dir)
    plot_bright(bright_data, 'Bright', 'bright', output_dir)

    print(f"Processing completed. Results saved to {output_file} and plots saved to {output_dir}.")

if __name__ == "__main__":
    main()
