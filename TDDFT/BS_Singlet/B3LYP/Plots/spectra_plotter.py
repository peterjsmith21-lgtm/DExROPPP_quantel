import cclib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


cmtoeV  = 1.23984e-4   # cm⁻¹ → eV
echarge=1.602176634e-19 # C
planck=6.62607015e-34 # Js
clight=299792458 # m/s
evtonm=planck*clight*10**9/echarge

brdn_typ = ['energy','wavelength'][0]
line_typ = ['gaussian','lorentzian'][1]
FWHM = (1/(300-10)-1/(300+10))


def _is_float(s):
    try: float(s); return True
    except ValueError: return False

def parse_orca(filepath):
    wavelengths, oscs = [], []

    with open(filepath) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if 'ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' in lines[i]:
            block_wl, block_osc = [], []
            j = i + 1

            while j < len(lines):
                stripped = lines[j].strip()
                parts = stripped.split()

                if (not parts or set(stripped) <= {'-'}) and block_wl:
                    break
                if not parts or set(stripped) <= {'-'} or \
                   any(h in stripped for h in ('State', 'Transition', '(cm', '(eV', 'au')):
                    j += 1
                    continue

                floats = [float(p) for p in parts if _is_float(p)]
                try:
                    block_wl.append(floats[2])   # nm — same index for both regular and SF
                    block_osc.append(floats[3])  # fosc
                except (IndexError, ValueError):
                    pass
                j += 1

            if block_wl:
                wavelengths, oscs = block_wl, block_osc
        i += 1

    if not wavelengths:
        raise ValueError(f"No absorption data found in {filepath}")

    return np.array(wavelengths), np.array(oscs)


'''
def parse_orca(filepath):
    """Returns excitation wavelengths (nm) and oscillator strengths."""
    data = cclib.io.ccread(filepath)
    energies_nm = 1e7 / data.etenergies   # cclib gives energies in cm⁻¹
    return energies_nm, data.etoscs
'''
def broaden(FWHM, osc, energy, energy_unit='eV'):
    if energy_unit == 'cm-1':
        energy = energy * cmtoeV          # convert ORCA/cclib cm⁻¹ to eV

    if brdn_typ == 'wavelength' and line_typ == 'lorentzian':
        eqn = "+%04.3f*1/(1+((%04.3f-x)/(%s/2))**2)" % (osc, evtonm/energy, FWHM)
    elif brdn_typ == 'energy' and line_typ == 'lorentzian':
        eqn = "+%04.3f*1/(1+((%04.3f-x)/(0.5*%s*%04.3f*x))**2)" % (osc, evtonm/energy, FWHM, evtonm/energy)
    elif brdn_typ == 'energy' and line_typ == 'gaussian':
        eqn = "+%04.3f*exp(-((%04.3f-x)/(0.5*%s*%04.3f*x))**2)" % (osc, evtonm/energy, FWHM, evtonm/energy)
    return eqn

def write_spectrum(wavelengths_nm, oscs, output_file, x):
    spectrum_str = "0"
    for wl, osc in zip(wavelengths_nm, oscs):
        energy_eV = evtonm / wl          # nm → eV for broaden
        spectrum_str += broaden(FWHM, osc, energy_eV, energy_unit='eV')
    with open(output_file, 'w') as f:
        f.write(spectrum_str)
    return eval(spectrum_str)

plt.style.use('seaborn-v0_8-paper')

if __name__ == "__main__":
    x = np.linspace(200, 1000, 1600)   # wavelength grid in nm — adjust to your range

    for outfile in sorted(Path("../").glob("*.out")):
        try:
            plt.style.use('seaborn-v0_8-paper')
            fig, ax = plt.subplots(figsize=(6, 4.8))

            wavelenghts, oscs = parse_orca(outfile)
            #print(f"{outfile.stem}: transitions at {wavelenghts.min():.0f}–{wavelenghts.max():.0f} nm, max fosc = {oscs.max():.5f}")
            spectrum = write_spectrum(wavelenghts, oscs, Path(outfile.stem + '_LYP1.gnu'), x)
            spectrum = spectrum / spectrum.max()
            ax.plot(x, spectrum, label=outfile.stem, alpha=1)
        
            ax.set_xlabel('Wavelength / nm')
            ax.set_ylabel('Normalised Absorbance')
            ax.legend(fontsize=6, ncol=3)
            plt.tight_layout()
            plt.savefig(f"{outfile.stem}_spectrum.png", dpi=300)
            plt.close()

        except Exception as e:
            print(f"Error processing {outfile}: {e}")
