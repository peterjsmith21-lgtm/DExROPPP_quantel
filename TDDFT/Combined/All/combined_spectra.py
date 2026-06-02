import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import re
from matplotlib.transforms import blended_transform_factory



def read_array(filepath):
    with open(filepath) as f:
        content = f.read()
    return np.array([float(x) for x in content.replace('[', '').replace(']', '').split()])



# Evaluate over the full spectral range (200-1000 nm) regardless of plot window,
# so that normalisation is never affected by the chosen x-axis limits.
wavelength = np.linspace(200, 1000, 1601)

diradicals = ["5-7ICzTTM2", "5-8ICzTTM2", "5-11ICzTTM2", "DTA", "PCzPyBTM2",
              "pseudo-o-PCP-TTM2", "pseudo-p-PCP-TTM2", "PyBTM-Hex", "PyBTM-Ph",
              "THDBA-PyBTM", "TPA-Me", "TPA-OMe", "TTM-Ph-TTM", "TTM-TTM",
              "TTMmTTM"]
EST = {"5-7ICzTTM2": -0.00054, "5-8ICzTTM2": 0., "5-11ICzTTM2": 0., "DTA": -0.0143, "PCzPyBTM2": -0.0021,
       "pseudo-o-PCP-TTM2": 0., "pseudo-p-PCP-TTM2": 0., "PyBTM-Hex": -0.0105, "PyBTM-Ph": -0.014, 
       "THDBA-PyBTM": 0.0007, "TPA-Me": -0.00148, "TPA-OMe": -0.0014, "TTM-Ph-TTM": -0.0654, "TTM-TTM": -0.135, "TTMmTTM": 0.}   

xrange = {"5-7ICzTTM2": (250, 800), "5-8ICzTTM2": (250, 800), "5-11ICzTTM2": (250, 800), "DTA": (300, 1000), "PCzPyBTM2": (250, 800),
          "pseudo-o-PCP-TTM2": (275, 800), "pseudo-p-PCP-TTM2": (275, 800), "PyBTM-Hex": (250, 800), "PyBTM-Ph": (250, 700), 
          "THDBA-PyBTM": (250, 700), "TPA-Me": (220, 1000), "TPA-OMe": (220, 1000), "TTM-Ph-TTM": (300, 800), "TTM-TTM": (300, 800), "TTMmTTM": (300, 800)}

np.set_printoptions(threshold=sys.maxsize)
plt.style.use('seaborn-v0_8-paper')

if __name__ == "__main__":
    x = np.linspace(200, 1000, 1600)
   
    for molecule in diradicals:
        try:
            xmin, xmax = xrange[molecule]
            mask = (x >= xmin) & (x <= xmax) # wavelength grid in nm — adjust to your range
            
            B3LYP_file = Path("../B3LYP") / f"{molecule}_LYPC.gnu"
            CAM_file = Path("../CAM") / f"{molecule}_CAMC.gnu"
            PBE0_file = Path("../PBE0") / f"{molecule}_PBEC.gnu"

            fig, ax = plt.subplots(figsize=(6, 4.8))

            B3LYP_spectrum = read_array(B3LYP_file)
            CAM_spectrum = read_array(CAM_file)
            PBE0_spectrum = read_array(PBE0_file)
            
            


            ax.plot(x, B3LYP_spectrum, color='darkred', linewidth=1.5, label='B3LYP')
            ax.plot(x, CAM_spectrum, color='darkorange', linewidth=1.5, label='CAM-B3LYP')
            ax.plot(x, PBE0_spectrum, color='darkblue', linewidth=2.5, label='PBE0')
            
            
            ax.set_xlabel('Wavelength / nm')
            ax.set_ylabel('Normalised Absorbance')
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(0, 1.05)
            ax.legend()

            plt.tight_layout()
            plt.savefig(f"{molecule}_spectrum.png", dpi=300)
            plt.close()

        except Exception as e:
            print(f"Error processing {molecule}: {e}")
            
