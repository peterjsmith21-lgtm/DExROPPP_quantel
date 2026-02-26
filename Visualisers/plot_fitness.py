import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser(description='Process a file and plot fitness vs iteration.')
parser.add_argument('filepath', type=str, help='Path to the input file')
parser.add_argument('--data_start', type=int, default= 0, help='Start index for slicing data')
parser.add_argument('--data_end', type=int, default=-1, help='End index for slicing data')

args = parser.parse_args()
filepath = args.filepath
data_start = args.data_start
data_end = args.data_end


cur_dir = os.path.dirname(os.path.abspath(__file__))
joined_path = os.path.join(cur_dir, '..','Params_training',f'{filepath}')

with open(joined_path, 'r') as file:
    count = 0
    lines = file.readlines()
    iteration = []
    fitness = []
    for line in lines:
        if line.startswith('iteration'):
            line = line.split()
            #iteration.append(float(line[1]))
            fitness.append(float(line[-1]))
            
            count += 1
    iteration = np.arange(1, count + 1)
    
plt.plot(iteration[data_start:data_end], np.array(fitness[data_start:data_end]), color = 'teal', label = f'{filepath}')
plt.xlabel('iteration')
plt.ylabel('fitness')
plt.legend()
plt.show()