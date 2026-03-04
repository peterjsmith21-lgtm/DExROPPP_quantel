#Some settings that can be altered for various ppp calculations

mixing=True # controls correction to dipole moments from ground state mixing 
alt=False # invert orbital signs according to alternacy

states_cutoff_option = ['none','states','energy'][1]  #cutoff for no of CIS states
states_to_print = 40 # WARNING! This will effect no. of states to be used in simulation of spectra as well as no. of states to be printed
energy_cutoff = 20 #6.5 5eV = 250nm, 6.5eV = 190nm # # WARNING! This will effect no. of states to be used in simulation of spectra as well as no. of states to be printed

charge=0 #not truly set up to go away from half-filling but can be made to easily

# extra parameters (not currently optimised)
tt=-2.8 #triple bond hopping / eV

# hopping term cutoffs
ccl_cutoff = 1.77
cutoff = 1.6 #angstrom
single_bond_cutoff = 1.43 #changed back to 1.43
single_bond_cutoff_cn = 1.4 #ONLY FOR PYRROLE TYPE N
triple_bond_cutoff = 1.3 #angstrom
#repulsion parameterization type
#parameterization=["Ohno","MN","Pople"][1]

#Overlap between nearest neighbors---0 for standard PPP calculation
nn_overlap=0

# ----- Fundamental Constants ------
toev=27.211
tobohr=1.889725989
echarge=1.602176634e-19 # C
planck=6.62607015e-34 # Js
clight=299792458 # m/s
evtonm=planck*clight*10**9/echarge

# ----- graph plotting options ------
brdn_typ = ['energy','wavelength'][0]
line_typ = ['gaussian','lorentzian'][1]
if brdn_typ == 'energy':
    FWHM = (1/(300-10)-1/(300+10)) #FWHM in terms of wavenumber for 20 nm splitting at 300 nm
if brdn_typ == 'wavelength':
    FWHM = 20.0 # nm 
