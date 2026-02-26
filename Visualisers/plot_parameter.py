import argparse
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('output_file',type=str,help='output file from machine learning to search through')
parser.add_argument('param',type=str,help='parameter to extract and plot e.g. U-Cl for Uclcl')
args = parser.parse_args()
output_file=args.output_file
param=args.param

# read parameter
atom=param.split("-")[1]
if atom in ["C","c"]:
    paramindex1=0
if atom in ["N1","n1"]:
    paramindex1=1
if atom in ["N2","n2"]:
    paramindex2=2
if atom in ["CL","cl","Cl"]:
    paramindex1=3
    
paramtyp=param.split("-")[0]
if paramtyp in ["alpha"]:
    paramindex2=0
if paramtyp in ["A","a"]:
    paramindex2=1
if paramtyp in ["B","b"]:
    paramindex2=2    
if paramtyp in ["u","U"]:
    paramindex2=3
if paramtyp in ["r0"]:
    paramindex2=4

all_params=[]
our_param=[]
f = open(output_file,'r')
file=f.readlines()
for i,line in enumerate(file):
    if "ttm_1cz is using" in line:
        params_iter=[]
        for j in range(1,5):
            pline=file[i+j]
            pline=pline.split("[")
            if j==1:
                n=2
            else:
                n=1
            pline=pline[n].split()
            pline[4]=pline[4].split("]")
            pline2=[pline[0],pline[1],pline[2],pline[3]]
            pline2+=pline[4]
            pline2=pline2[:5]
            params_iter.append(pline2)
        all_params.append(params_iter)
        our_param.append(params_iter[paramindex1][paramindex2])
    
all_params=np.array(all_params)
all_params=all_params.astype(np.float64)
our_param=np.array(our_param)
our_param=our_param.astype(np.float64)
print(our_param)

plt.plot(our_param)
plt.xlabel("Iteration")
plt.ylabel("%s"%param)
plt.show()
        
