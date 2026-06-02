import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import re
from matplotlib.transforms import blended_transform_factory



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

def read_gnu(filepath, x):
    with open(filepath) as f:
        eqn = f.read().strip()
    return eval(eqn)

def extract_wavelengths_from_gnu(filepath, xmin, xmax):
    with open(filepath) as f:
        content = f.read()
    # Match osc and centre wavelength from each Lorentzian term
    matches = re.findall(r'\+([0-9.e+-]+)\*1/\(1\+\(\(([0-9.]+)-x\)', content)
    wavelengths = np.array([
        float(wl) for osc, wl in matches
        if float(osc) > 0 and xmin <= float(wl) <= xmax
    ])
    return wavelengths


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

functional = 'CAM'

np.set_printoptions(threshold=sys.maxsize)
plt.style.use('seaborn-v0_8-paper')

if __name__ == "__main__":
    x = np.linspace(200, 1000, 1600)
   
    for molecule in diradicals:
        try:
            xmin, xmax = xrange[molecule]
            mask = (x >= xmin) & (x <= xmax) # wavelength grid in nm — adjust to your range
            
            singlet_file = Path("../../BS_Singlet/CAM/Plots") / f"{molecule}_{functional}1.gnu"
            triplet_file = Path("../../SF_Triplets/CAM/Plots") / f"{molecule}_{functional}3.gnu"

            wls_s = extract_wavelengths_from_gnu(singlet_file, xmin, xmax)
            wls_t = extract_wavelengths_from_gnu(triplet_file, xmin, xmax)

            min_wl_s = wls_s.min() if len(wls_s) > 0 else None
            min_wl_t = wls_t.min() if len(wls_t) > 0 else None

            fig, ax = plt.subplots(figsize=(6, 4.8))

            singlet_spectrum = spectrum = read_gnu(singlet_file, x)
            triplet_spectrum = spectrum = read_gnu(triplet_file, x)
            w_singlet, w_triplet = boltzmann_weights(EST[molecule])
            combined_spectrum = w_singlet * singlet_spectrum + w_triplet * triplet_spectrum

            singlet_max = np.max(singlet_spectrum[mask])
            triplet_max = np.max(triplet_spectrum[mask])
            single_max = max(singlet_max, triplet_max)
            combined_max = np.max(combined_spectrum[mask])

            singlet_spectrum = singlet_spectrum / single_max
            triplet_spectrum = triplet_spectrum / single_max
            combined_spectrum = combined_spectrum / combined_max

            ax.plot(x, singlet_spectrum, color='darkblue',  linestyle='--', linewidth=1.2, label='Singlet')
            ax.plot(x, triplet_spectrum, color='darkgreen', linestyle='--', linewidth=1.2, label='Triplet')
            ax.plot(x, combined_spectrum, color='firebrick', linewidth=2.5, label='Weighted Combination')

            trans = blended_transform_factory(ax.transData, ax.transAxes)
            if min_wl_s is not None:
                ax.plot([min_wl_s, min_wl_s], [-0.02, 0.02], color='darkblue',
                        lw=2, transform=trans, clip_on=False)
            if min_wl_t is not None:
                ax.plot([min_wl_t, min_wl_t], [-0.02, 0.02], color='darkgreen',
                        lw=2, transform=trans, clip_on=False)
            
            
            ax.set_xlabel('Wavelength / nm')
            ax.set_ylabel('Normalised Absorbance')
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(0, 1.05)
            ax.legend()

            plt.tight_layout()
            plt.savefig(f"{molecule}_spectrum.png", dpi=300)
            plt.close()

            with open(f"{molecule}_{functional}C.gnu", 'w') as f:
                f.write(f"{combined_spectrum}")
                f.close()

        except Exception as e:
            print(f"Error processing {molecule}: {e}")
            
