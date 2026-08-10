import numpy as np
from scipy.linalg import block_diag
from ExROPPP_settings_opt import *

'''
File containing helper functions for building the CI Matrix for a diradical system
'''

def build_singlet_ref_block(ndocc, energy0, orb_energies, rep_tens):
    '''
    Function to build the CI matrix for 3 singlet reference states for a diradical system.
    These are the Open-Shell Singlet (OS1) and the  +/- Combinations of Zwitterion states (ZW+ and ZW-).
    Args: 
        ndocc (int): Number of doubly occupied orbitals
        energy0 (float): Base energy of the mean-field reference state
        orb_energies (numpy.ndarray): Orbital energies for the system
        rep_tens (numpy.ndarray): Representation tensor for the system
    Returns:
        numpy.ndarray: CI matrix for the diradical system
    '''
    # Calculate the size of the CI matrix based on the number of doubly occupied orbitals
    CI = np.zeros((3, 3))  # Initialize a 3x3 CI matrix
    SOMO1 = ndocc # Index of SOMO1
    SOMO2 = ndocc + 1 # Index of SOMO2

    # <OS1|H|OS1>
    CI[0,0] = energy0 - 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) + (1.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1])
    # <OS1|H|ZW->
    CI[0,1] = rep_tens[SOMO1,SOMO2,SOMO1,SOMO1] - rep_tens[SOMO1,SOMO2,SOMO2,SOMO2]
    # <OS1|H|ZW+> = 0
    
    # <ZW-|H|ZW->
    CI[1,1] = energy0 + 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) - 0.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1] - rep_tens[SOMO1, SOMO1, SOMO2, SOMO2]
    # <ZW-|H|ZW+>
    CI[1,2] = orb_energies[SOMO1] - orb_energies[SOMO2]
    
    # <ZW+|H|ZW+>
    CI[2,2] = energy0 + 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) + 1.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1] - rep_tens[SOMO1, SOMO1, SOMO2, SOMO2]

    return CI

def build_singlet_CS_SV_block(ndocc, norbs, energy0, orb_energies, rep_tens):
    '''
    Function to build the upper diagonal of the CI matrix for singlet reference states for a diradical system - the closed-shell singlet (CS) and the single-reference singlet (SV).
    Args:
        ndocc (int): Number of doubly occupied orbitals
        norbs (int): Total number of orbitals
        energy0 (float): Base energy of the mean-field reference state
        orb_energies (numpy.ndarray): Orbital energies for the system
        rep_tens (numpy.ndarray): Representation tensor for the system
    Returns:
        numpy.ndarray: CI matrix for the diradical system
    '''
    # Calculate the size of the CI matrix based on the number of doubly occupied orbitals
    SOMO1 = ndocc # Index of SOMO1
    SOMO2 = ndocc + 1 # Index of SOMO2
    nvirt = norbs - ndocc - 2 # Number of virtual orbitals
    
    row_dim = 2 * ndocc + 2 * nvirt + 3
    col_dim = 2 * ndocc + 2 * nvirt
    CI = np.zeros((row_dim, col_dim))  # Initialize CI Block

    # <OS1|H|CS0> (CHECKED)
    for col in range(0, ndocc):
        o_orb = col
        CI[0,col] = 1.5 * rep_tens[o_orb,SOMO2,SOMO2,SOMO1] - 0.5 * rep_tens[o_orb,SOMO1,SOMO1,SOMO1]
    # <OS1|H|CS0'> 
    for col in range(ndocc, 2 * ndocc):
        o_orb = col - ndocc
        CI[0,col] = 0.5 * rep_tens[o_orb,SOMO2,SOMO2,SOMO2] - 1.5 * rep_tens[o_orb,SOMO1,SOMO1,SOMO2]
    # <OS1|H|SV0>
    for col in range(2 * ndocc, 2 * ndocc + nvirt):
        v_orb = col - (2 * ndocc) + (SOMO2 + 1)
        CI[0,col] = 1.5 * rep_tens[v_orb,SOMO2,SOMO2,SOMO1] - 0.5 * rep_tens[v_orb,SOMO1,SOMO1,SOMO1]
    # <OS1|H|SV0'>
    for col in range(2 * ndocc + nvirt, 2 * ndocc + 2 * nvirt):
        v_orb = col - (2 * ndocc + nvirt) + (SOMO2 + 1)
        CI[0,col] = 0.5 * rep_tens[v_orb,SOMO2,SOMO2,SOMO2] - 1.5 * rep_tens[v_orb,SOMO1,SOMO1,SOMO2]
    
    # <ZW-|H|CS0> (CHECKED)
    for col in range(0, ndocc):
        o_orb = col
        CI[1,col] = rep_tens[o_orb,SOMO2,SOMO1,SOMO1] + 0.5 * rep_tens[o_orb,SOMO1,SOMO1,SOMO2] - 0.5 * rep_tens[o_orb,SOMO2,SOMO2,SOMO2]
    # <ZW-|H|CS0'>
    for col in range(ndocc, 2 * ndocc):
        o_orb = col - ndocc
        CI[1,col] = rep_tens[o_orb,SOMO1,SOMO2,SOMO2] + 0.5 * rep_tens[o_orb,SOMO2,SOMO2,SOMO1] - 0.5 * rep_tens[o_orb,SOMO1,SOMO1,SOMO1]
    # <ZW-|H|SV0>
    for col in range(2 * ndocc, 2 * ndocc + nvirt):
        v_orb = col - (2 * ndocc) + (SOMO2 + 1)
        CI[1,col] = rep_tens[v_orb,SOMO2,SOMO1,SOMO1] + 0.5 * rep_tens[v_orb,SOMO1,SOMO1,SOMO2] - 0.5 * rep_tens[v_orb,SOMO2,SOMO2,SOMO2]
    # <ZW-|H|SV0'>
    for col in range(2 * ndocc + nvirt, 2 * ndocc + 2 * nvirt):
        v_orb = col - (2 * ndocc + nvirt) + (SOMO2 + 1)
        CI[1,col] = rep_tens[v_orb,SOMO1,SOMO2,SOMO2] + 0.5 * rep_tens[v_orb,SOMO2,SOMO2,SOMO1] - 0.5 * rep_tens[v_orb,SOMO1,SOMO1,SOMO1]
        
    # <ZW+|H|CS0> (CHECKED)
    for col in range(0, ndocc):
        o_orb = col
        CI[2,col] = rep_tens[o_orb,SOMO2,SOMO1,SOMO1] - 1.5 * rep_tens[o_orb,SOMO1,SOMO1,SOMO2] - 0.5 * rep_tens[o_orb,SOMO2,SOMO2,SOMO2]
    # <ZW+|H|CS0'>
    for col in range(ndocc, 2 * ndocc):
        o_orb = col - ndocc
        CI[2,col] = 1.5 * rep_tens[o_orb,SOMO2,SOMO2,SOMO1] + 0.5 * rep_tens[o_orb,SOMO1,SOMO1,SOMO1] - rep_tens[o_orb,SOMO1,SOMO2,SOMO2]
    # <ZW+|H|SV0>
    for col in range(2 * ndocc, 2 * ndocc + nvirt):
        v_orb = col - (2 * ndocc) + (SOMO2 + 1)
        CI[2,col] = 1.5 * rep_tens[v_orb,SOMO1,SOMO1,SOMO2] + 0.5 * rep_tens[v_orb,SOMO2,SOMO2,SOMO2] - rep_tens[v_orb,SOMO2,SOMO1,SOMO1]
    # <ZW+|H|SV0'>
    for col in range(2 * ndocc + nvirt, 2 * ndocc + 2 * nvirt):
        v_orb = col - (2 * ndocc + nvirt) + (SOMO2 + 1)
        CI[2,col] = rep_tens[v_orb,SOMO1,SOMO2,SOMO2] - 1.5 * rep_tens[v_orb,SOMO2,SOMO2,SOMO1] - 0.5 * rep_tens[v_orb,SOMO1,SOMO1,SOMO1]
    
    row_index = 3
    for row in range(row_index, row_index + ndocc):
        o_orb1 = row - row_index
        # <CS0|H|CS0>
        for col in range(row - row_index, ndocc):
            o_orb2 = col
            if o_orb1 == o_orb2:
                CI[row, col] = energy0 + orb_energies[SOMO1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, SOMO1, SOMO1] + 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] \
                             - 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] + 1.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb1] + 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb1]
            else:    
                CI[row, col] = 0.5 * rep_tens[o_orb2, SOMO1, SOMO1, o_orb1] + 1.5 * rep_tens[o_orb2, SOMO2, SOMO2, o_orb1] - rep_tens[o_orb2, o_orb1, SOMO1, SOMO1]
        # <CS0|H|CS0'>
        for col in range(ndocc, 2*ndocc):
            o_orb2 = col - ndocc
            if o_orb1 == o_orb2:
                CI[row, col] = 0.5 * rep_tens[SOMO1, SOMO2, SOMO1, SOMO1] + 0.5 * rep_tens[SOMO1, SOMO2, SOMO2, SOMO2] - rep_tens[SOMO1, o_orb1, o_orb1, SOMO2] - rep_tens[o_orb1, o_orb1, SOMO1, SOMO2]
            else:    
                CI[row, col] = - rep_tens[o_orb1, o_orb2, SOMO1, SOMO2] - rep_tens[o_orb1, SOMO1, SOMO2, o_orb2]
        # <CS0|H|SV0>
        for col in range(2*ndocc, 2*ndocc + nvirt):
            v_orb = col - 2*ndocc + (SOMO2 + 1)
            CI[row, col] = - rep_tens[o_orb1, SOMO1, SOMO1, v_orb] 
        # <CS0|H|SV0'>
        for col in range(2*ndocc + nvirt, 2*ndocc + 2*nvirt):
            v_orb = col - (2*ndocc + nvirt) + (SOMO2 + 1)
            CI[row, col] = rep_tens[o_orb1, SOMO1, SOMO2, v_orb] - 2 * rep_tens[o_orb1, SOMO2, SOMO1, v_orb]
            
    
    row_index = ndocc + 3
    for row in range(row_index, row_index + ndocc):
        o_orb1 = row - row_index
        # <CS0'|H|CS0'>
        for col in range(row - row_index + ndocc, 2*ndocc):
            o_orb2 = col - ndocc
            if o_orb1 == o_orb2:
                CI[row, col] = energy0 + orb_energies[SOMO2] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, SOMO2, SOMO2] + 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] \
                             - 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] + 1.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb1] + 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb1]
            else:    
                CI[row, col] = 1.5 * rep_tens[o_orb2, SOMO1, SOMO1, o_orb1] + 0.5 * rep_tens[o_orb2, SOMO2, SOMO2, o_orb1] - rep_tens[o_orb2, o_orb1, SOMO2, SOMO2]
        # <CS0'|H|SV0>
        for col in range(2*ndocc, 2*ndocc + nvirt):
            v_orb = col - 2*ndocc + (SOMO2 + 1)
            CI[row, col] = rep_tens[o_orb1, SOMO2, SOMO1, v_orb] - 2 * rep_tens[o_orb1, SOMO1, SOMO2, v_orb]
        # <CS0'|H|SV0'>
        for col in range(2*ndocc + nvirt,  2*ndocc + 2*nvirt):
            v_orb = col - (2*ndocc + nvirt) + (SOMO2 + 1)
            CI[row, col] = - rep_tens[o_orb1, SOMO2, SOMO2, v_orb]
    
    row_index = 2 * ndocc + 3
    for row in range(row_index, row_index + nvirt):
        v_orb1 = row - row_index + (SOMO2 + 1)
        # <SV0|H|SV0>
        for col in range(row - row_index + 2*ndocc, 2*ndocc + nvirt):
            v_orb2 = col - (2*ndocc) + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                CI[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[SOMO1] - rep_tens[v_orb1, v_orb1, SOMO1, SOMO1] + 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] \
                             - 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] + 1.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb1] + 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb1]
            else:    
                CI[row, col] = 1.5 * rep_tens[v_orb2, SOMO2, SOMO2, v_orb1] + 0.5 * rep_tens[v_orb2, SOMO1, SOMO1, v_orb1] -  rep_tens[v_orb2, v_orb1, SOMO1, SOMO1]
        # <SV0|H|SV0'>
        for col in range(2*ndocc + nvirt, 2*ndocc + 2*nvirt):
            v_orb2 = col - (2*ndocc + nvirt) + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                CI[row, col] = 0.5 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO2] + 0.5 * rep_tens[SOMO1, SOMO2, SOMO2, SOMO2] - rep_tens[v_orb1, v_orb1, SOMO1, SOMO2] - rep_tens[v_orb1, SOMO1, SOMO2, v_orb1]
            else:    
                CI[row, col] = - rep_tens[v_orb1, v_orb2, SOMO1, SOMO2] - rep_tens[v_orb1, SOMO1, SOMO2, v_orb2]
    
    row_index = 2 * ndocc + nvirt + 3
    # <SV0'|H|SV0'>
    for row in range(row_index, row_index + nvirt):
        v_orb1 = row - row_index + (SOMO2 + 1)
        for col in range(row - row_index + 2*ndocc + nvirt, 2*ndocc + 2*nvirt):
            v_orb2 = col - (2*ndocc + nvirt) + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                CI[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[SOMO2] - rep_tens[v_orb1, v_orb1, SOMO2, SOMO2] + 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] \
                             - 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] + 1.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb1] + 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb1]
            else:    
                CI[row, col] = 1.5 * rep_tens[v_orb2, SOMO1, SOMO1, v_orb1] + 0.5 * rep_tens[v_orb2, SOMO2, SOMO2, v_orb1] -  rep_tens[v_orb2, v_orb1, SOMO2, SOMO2]

    return CI


def build_singlet_HL_block(ndocc, norbs, energy0, orb_energies, rep_tens):
    '''
    Function to build the upper diagonal of the CI matrix for HOMO to LUMO excited states for a diradical system.
    Args:
        ndocc (int): Number of doubly occupied orbitals
        norbs (int): Total number of orbitals
        energy0 (float): Base energy of the mean-field reference state
        orb_energies (numpy.ndarray): Orbital energies for the system
        rep_tens (numpy.ndarray): Representation tensor for the system
    Returns:
        numpy.ndarray: CI matrix for the diradical system
    '''
    # Calculate the size of the CI matrix based on the number of doubly occupied orbitals
    SOMO1 = ndocc # Index of SOMO1
    SOMO2 = ndocc + 1 # Index of SOMO2
    nvirt = norbs - ndocc - 2 # Number of virtual orbitals
    npairs = ndocc * nvirt
    
    row_dim = 4 * (npairs) + 2 * ndocc + 2 * nvirt + 3
    col_dim = 4 * (npairs)
    CI = np.zeros((row_dim, col_dim))  # Initialize CI Block

    # <OS1|H|HL1> = 0
    # <OS1|H|HL2>
    for col in range(npairs, 2 * npairs):
        o_orb = (col - npairs) // nvirt # Increase o_orb after every ndocc cols
        v_orb = (col - npairs) % nvirt + (SOMO2 + 1) # Increase v_orb for every col then reset after ndocc cols
        CI[0,col] = np.sqrt(1.5) * (rep_tens[o_orb, SOMO2, SOMO2, v_orb] - rep_tens[o_orb, SOMO1, SOMO1, v_orb])
    # <OS1|H|ZHL1> 
    for col in range(2*npairs, 3*npairs):
        o_orb = (col - 2*npairs) // nvirt
        v_orb = (col - 2*npairs) % nvirt + (SOMO2 + 1)
        CI[0,col] = 2 * rep_tens[o_orb, v_orb, SOMO1, SOMO2] - rep_tens[o_orb, SOMO1, SOMO2, v_orb]
    # <OS1|H|ZHL2>
    for col in range(3*npairs, 4*npairs):
        o_orb = (col - 3*npairs) // nvirt
        v_orb = (col - 3*npairs) % nvirt + (SOMO2 + 1)
        CI[0,col] = 2 * rep_tens[o_orb, v_orb, SOMO1, SOMO2] - rep_tens[o_orb, SOMO2, SOMO1, v_orb]

    
    # <ZW-|H|HL1>
    for col in range(0, npairs):
        o_orb = col // nvirt
        v_orb = col % nvirt + (SOMO2 + 1)
        CI[1,col] = (1 / np.sqrt(2)) * (rep_tens[o_orb, SOMO1, SOMO2, v_orb] - rep_tens[o_orb, SOMO2, SOMO1, v_orb])
    # <ZW-|H|HL2>
    for col in range(npairs, 2 * npairs):
        o_orb = (col - npairs) // nvirt 
        v_orb = (col - npairs) % nvirt + (SOMO2 + 1) 
        CI[1,col] = np.sqrt(1.5) * (rep_tens[o_orb, SOMO1, SOMO2, v_orb] + rep_tens[o_orb, SOMO2, SOMO1, v_orb])
    # <ZW-|H|ZHL1> 
    for col in range(2*npairs, 3*npairs):
        o_orb = (col - 2*npairs) // nvirt
        v_orb = (col - 2*npairs) % nvirt + (SOMO2 + 1)
        CI[1,col] = rep_tens[o_orb, v_orb, SOMO1, SOMO1] - rep_tens[o_orb, v_orb, SOMO2, SOMO2] + 0.5 * rep_tens[o_orb, SOMO2, SOMO2, v_orb] - 0.5 * rep_tens[o_orb, SOMO1, SOMO1, v_orb]
    # <ZW-|H|ZHL2>
    for col in range(3*npairs, 4*npairs):
        o_orb = (col - 3*npairs) // nvirt
        v_orb = (col - 3*npairs) % nvirt + (SOMO2 + 1)
        CI[1,col] = rep_tens[o_orb, v_orb, SOMO1, SOMO1] - rep_tens[o_orb, v_orb, SOMO2, SOMO2] + 0.5 * rep_tens[o_orb, SOMO2, SOMO2, v_orb] - 0.5 * rep_tens[o_orb, SOMO1, SOMO1, v_orb]
    
    # <ZW+|H|HL1>
    for col in range(0, npairs):
        o_orb = col // nvirt
        v_orb = col % nvirt + (SOMO2 + 1)
        CI[2,col] = (1 / np.sqrt(2)) * (4 * rep_tens[o_orb, v_orb, SOMO1, SOMO2] - rep_tens[o_orb, SOMO2, SOMO1, v_orb] - rep_tens[o_orb, SOMO1, SOMO2, v_orb])
    # <ZW+|H|HL2>
    for col in range(npairs, 2*npairs):
        o_orb = (col - npairs) // nvirt 
        v_orb = (col - npairs) % nvirt + (SOMO2 + 1) 
        CI[2,col] = np.sqrt(1.5) * (rep_tens[o_orb, SOMO2, SOMO1, v_orb] - rep_tens[o_orb, SOMO1, SOMO2, v_orb])
    # <ZW+|H|ZHL1> 
    for col in range(2*npairs, 3*npairs):
        o_orb = (col - 2*npairs) // nvirt
        v_orb = (col - 2*npairs) % nvirt + (SOMO2 + 1)
        CI[2,col] = rep_tens[o_orb, v_orb, SOMO1, SOMO1] - rep_tens[o_orb, v_orb, SOMO2, SOMO2] + 0.5 * rep_tens[o_orb, SOMO2, SOMO2, v_orb] - 0.5 * rep_tens[o_orb, SOMO1, SOMO1, v_orb]
    # <ZW+|H|ZHL2>
    for col in range(3*npairs, 4*npairs):
        o_orb = (col - 3*npairs) // nvirt
        v_orb = (col - 3*npairs) % nvirt + (SOMO2 + 1)
        CI[2,col] = rep_tens[o_orb, v_orb, SOMO2, SOMO2] - rep_tens[o_orb, v_orb, SOMO1, SOMO1] + 0.5 * rep_tens[o_orb, SOMO1, SOMO1, v_orb] - 0.5 * rep_tens[o_orb, SOMO2, SOMO2, v_orb]

    
    row_index = 3
    for row in range(row_index, row_index + ndocc):
        o_orb1 = row - row_index
        # <CS0|H|HL1>
        for col in range(0, npairs):
            o_orb2 = col // nvirt
            v_orb = col % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                CI[row, col] = (1 / np.sqrt(2)) * (rep_tens[o_orb1, o_orb1, SOMO1, v_orb] + 1.5 * rep_tens[SOMO1, SOMO2, SOMO2, v_orb] - 0.5 * rep_tens[SOMO1, SOMO1, SOMO1, v_orb] - 2 * rep_tens[SOMO1, o_orb1, o_orb1, v_orb])
            else:    
                CI[row, col] = (1 / np.sqrt(2)) * (rep_tens[SOMO1, v_orb, o_orb1, o_orb2] -  2 * rep_tens[SOMO1, o_orb1, o_orb2, v_orb])
        # <CS0|H|HL2>
        for col in range(npairs, 2 * npairs):
            o_orb2 = (col - npairs) // nvirt
            v_orb = (col - npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                CI[row, col] = np.sqrt(1.5) * (0.5 * rep_tens[SOMO1, SOMO2, SOMO2, v_orb] + 0.5 * rep_tens[SOMO1, SOMO1, SOMO1, v_orb] - rep_tens[o_orb1, o_orb1, SOMO1, v_orb])
            else:    
                CI[row, col] = - np.sqrt(1.5) * rep_tens[SOMO1, v_orb, o_orb1, o_orb2]
        # <CS0|H|ZHL1>
        for col in range(2*npairs, 3*npairs):
            o_orb2 = (col - 2*npairs) // nvirt
            v_orb = (col - 2*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                CI[row, col] = 2 * rep_tens[SOMO2, o_orb1, o_orb1, v_orb] + rep_tens[SOMO2, v_orb, SOMO1, SOMO1] - rep_tens[o_orb1, o_orb1, SOMO2, v_orb] - 0.5 * rep_tens[SOMO2, SOMO1, SOMO1, v_orb] - 0.5 * rep_tens[SOMO2, SOMO2, SOMO2, v_orb] 
            else:    
                CI[row, col] = 2 * rep_tens[SOMO2, o_orb1, o_orb2, v_orb] - rep_tens[SOMO2, v_orb, o_orb1, o_orb2] 
        # <CS0|H|ZHL2>
        for col in range(3*npairs, 4*npairs):
            o_orb2 = (col - 3*npairs) // nvirt
            v_orb = (col - 3*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                CI[row, col] = - rep_tens[SOMO2, SOMO1, SOMO1, v_orb]
    
    row_index = ndocc + 3
    for row in range(row_index, row_index + ndocc):
        o_orb1 = row - row_index
        # <CS0'|H|HL1>
        for col in range(0, npairs):
            o_orb2 = col // nvirt
            v_orb = col % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                CI[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[SOMO2, o_orb1, o_orb1, v_orb] + 0.5 * rep_tens[SOMO2, SOMO2, SOMO2, v_orb] - rep_tens[o_orb1, o_orb1, SOMO2, v_orb] - 1.5 * rep_tens[SOMO2, SOMO1, SOMO1, v_orb])
            else:
                CI[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[SOMO2, o_orb1, o_orb2, v_orb] - rep_tens[SOMO2, v_orb, o_orb1, o_orb2])
        # <CS0'|H|HL2>
        for col in range(npairs, 2 * npairs):
            o_orb2 = (col - npairs) // nvirt
            v_orb = (col - npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                CI[row, col] = np.sqrt(1.5) * (0.5 * rep_tens[SOMO2, SOMO1, SOMO1, v_orb] + 0.5 * rep_tens[SOMO2, SOMO2, SOMO2, v_orb] - rep_tens[o_orb1, o_orb1, SOMO2, v_orb])
            else:    
                CI[row, col] = - np.sqrt(1.5) * rep_tens[SOMO2, v_orb, o_orb1, o_orb2]
        # <CS0'|H|ZHL1>
        for col in range(2*npairs, 3*npairs):
            o_orb2 = (col - 2*npairs) // nvirt
            v_orb = (col - 2*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                CI[row, col] = rep_tens[SOMO1, SOMO2, SOMO2, v_orb]
        # <CS0'|H|ZHL2>
        for col in range(3*npairs, 4*npairs):
            o_orb2 = (col - 3*npairs) // nvirt
            v_orb = (col - 3*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                CI[row, col] = rep_tens[SOMO1, v_orb, o_orb1, o_orb1] + 0.5 * rep_tens[SOMO1, SOMO1, SOMO1, v_orb] + 0.5 * rep_tens[SOMO1, SOMO2, SOMO2, v_orb] - 2 * rep_tens[SOMO1, o_orb1, o_orb1, v_orb] - rep_tens[SOMO1, v_orb, SOMO2, SOMO2]
            else:
                CI[row, col] = rep_tens[SOMO1, v_orb, o_orb1, o_orb2] - 2 * rep_tens[SOMO1, o_orb1, o_orb2, v_orb]
    
    row_index = 2 * ndocc + 3
    for row in range(row_index, row_index + nvirt):
        v_orb1 = row - row_index + (SOMO2 + 1)
        # <SV0|H|HL1>
        for col in range(0, npairs):
            o_orb = col // nvirt
            v_orb2 = col % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                CI[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[o_orb, v_orb1, v_orb1, SOMO1] + 0.5 * rep_tens[SOMO1, SOMO1, SOMO1, o_orb] - rep_tens[v_orb1, v_orb1, SOMO1, o_orb] - 1.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO1])
            else:    
                CI[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[o_orb, v_orb2, v_orb1, SOMO1] - rep_tens[v_orb1, v_orb2, SOMO1, o_orb])
        # <SV0|H|HL2>
        for col in range(npairs, 2*npairs):
            o_orb = (col - npairs) // nvirt
            v_orb2 = (col - npairs) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                CI[row, col] = np.sqrt(1.5) * (0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO1] + 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO1] - rep_tens[o_orb, SOMO1, v_orb1, v_orb1])
            else:    
                CI[row, col] = - np.sqrt(1.5) * rep_tens[o_orb, SOMO1, v_orb2, v_orb1]
        # <SV0|H|ZHL1>
        for col in range(2*npairs, 3*npairs):
            o_orb = (col - 2*npairs) // nvirt
            v_orb2 = (col - 2*npairs) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                CI[row, col] = - rep_tens[o_orb, SOMO1, SOMO1, SOMO2]
        # <SV0|H|ZHL2>
        for col in range(3*npairs, 4*npairs):
            o_orb = (col - 3*npairs) // nvirt
            v_orb2 = (col - 3*npairs) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                CI[row, col] = 2 * rep_tens[SOMO2, v_orb1, v_orb1, o_orb] + rep_tens[o_orb, SOMO2, SOMO1, SOMO1] - rep_tens[o_orb, SOMO2, v_orb1, v_orb1] - 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO2] - 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO2]
            else:
                CI[row, col] = 2 * rep_tens[SOMO2, v_orb1, v_orb2, o_orb] - rep_tens[o_orb, SOMO2, v_orb1, v_orb2]
                
    row_index = 2 * ndocc + nvirt + 3
    for row in range(row_index, row_index + nvirt):
        v_orb1 = row - row_index + (SOMO2 + 1)
        # <SV0'|H|HL1>
        for col in range(0, npairs):
            o_orb = col // nvirt
            v_orb2 = col % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                CI[row, col] = (1 / np.sqrt(2)) * (1.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO2] + rep_tens[o_orb, SOMO2, v_orb1, v_orb1] - 2 * rep_tens[o_orb, v_orb1, v_orb1, SOMO2] - 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO2])
            else:    
                CI[row, col] = (1 / np.sqrt(2)) * (rep_tens[o_orb, SOMO2, v_orb2, v_orb1] - 2 * rep_tens[SOMO2, v_orb1, v_orb2, o_orb])
        # <SV0'|H|HL2>
        for col in range(npairs, 2*npairs):
            o_orb = (col - npairs) // nvirt
            v_orb2 = (col - npairs) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                CI[row, col] = np.sqrt(1.5) * (0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO2] + 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO2] - rep_tens[o_orb, SOMO2, v_orb1, v_orb1])
            else:    
                CI[row, col] = - np.sqrt(1.5) * rep_tens[o_orb, SOMO2, v_orb2, v_orb1]
        # <SV0'|H|ZHL1>
        for col in range(2*npairs, 3*npairs):
            o_orb = (col - 2*npairs) // nvirt
            v_orb2 = (col - 2*npairs) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                CI[row, col] = rep_tens[o_orb, SOMO1, v_orb1, v_orb1] + 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO1] + 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO1] - 2 * rep_tens[SOMO1, v_orb1, v_orb1, o_orb] - rep_tens[o_orb, SOMO1, SOMO2, SOMO2]
            else:
                CI[row, col] = rep_tens[o_orb, SOMO1, v_orb1, v_orb2] - 2 * rep_tens[SOMO1, v_orb1, v_orb2, o_orb]
        # <SV0'|H|ZHL2>
        for col in range(3*npairs, 4*npairs):
            o_orb = (col - 3*npairs) // nvirt
            v_orb2 = (col - 3*npairs) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                CI[row, col] = rep_tens[o_orb, SOMO2, SOMO2, SOMO1]
    
    row_index = 2 * ndocc + 2 * nvirt + 3
    for row in range(row_index, row_index + npairs):
        o_orb1 = (row - row_index) // nvirt
        v_orb1 = (row - row_index) % nvirt + (SOMO2 + 1)
        # <HL1|H|HL1>
        for col in range(row - row_index, npairs):
            o_orb2 = col // nvirt
            v_orb2 = col % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                CI[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb1] - 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) \
                    + 1.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1] + 2 * rep_tens[o_orb1, v_orb1, v_orb1, o_orb1]
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                CI[row, col] =  2 * rep_tens[o_orb1, v_orb1, v_orb1, o_orb2] - rep_tens[o_orb1, o_orb2, v_orb1, v_orb1]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                CI[row, col] = 2 * rep_tens[v_orb1, o_orb1, o_orb1, v_orb2] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb2]
            else:
                CI[row, col] = 2 * rep_tens[o_orb1, v_orb1, o_orb2, v_orb2] - rep_tens[o_orb1, o_orb2, v_orb1, v_orb2]
        # <HL1|H|HL2>
        for col in range(npairs, 2*npairs):
            o_orb2 = (col - npairs) // nvirt
            v_orb2 = (col - npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                CI[row, col] = (np.sqrt(3) / 2) * (rep_tens[o_orb1,SOMO1,SOMO1,o_orb1] - rep_tens[o_orb1,SOMO2,SOMO2,o_orb1] + rep_tens[v_orb1,SOMO2,SOMO2,v_orb1] - rep_tens[v_orb1,SOMO1,SOMO1,v_orb1])
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                CI[row, col] =  (np.sqrt(3) / 2) * (rep_tens[o_orb1, SOMO1, SOMO1, o_orb2] - rep_tens[o_orb1, SOMO2, SOMO2, o_orb2])
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                CI[row, col] =  (np.sqrt(3) / 2) * (rep_tens[v_orb1, SOMO2, SOMO2, v_orb2] - rep_tens[v_orb1, SOMO1, SOMO1, v_orb2])
        #<HL1|H|ZHL1>
        for col in range(2*npairs, 3*npairs):
            o_orb2 = (col - 2*npairs) // nvirt
            v_orb2 = (col - 2*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                CI[row, col] = np.sqrt(2) * (rep_tens[SOMO1, SOMO2, v_orb1, v_orb1] - rep_tens[SOMO1, SOMO2, o_orb1, o_orb1] + 0.5 * rep_tens[SOMO1, o_orb1, o_orb1, SOMO2] - 0.5 * rep_tens[SOMO1, v_orb1, v_orb1, SOMO2] \
                                + 0.5 * rep_tens[SOMO2, SOMO1, SOMO1, SOMO1] - 0.5 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO1])
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:
                CI[row, col] = np.sqrt(2) * (0.5 * rep_tens[o_orb1, SOMO2, SOMO1, o_orb2] - rep_tens[SOMO1, SOMO2, o_orb1, o_orb2])
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                CI[row, col] = np.sqrt(2) * (rep_tens[SOMO1, SOMO2, v_orb1, v_orb2] - 0.5 * rep_tens[v_orb1, SOMO1, SOMO2, v_orb2])
        #<HL1|H|ZHL2>
        for col in range(3*npairs, 4*npairs):
            o_orb2 = (col - 3*npairs) // nvirt
            v_orb2 = (col - 3*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                CI[row, col] = np.sqrt(2) * (rep_tens[SOMO1, SOMO2, v_orb1, v_orb1] - rep_tens[SOMO1, SOMO2, o_orb1, o_orb1] + 0.5 * rep_tens[SOMO1, o_orb1, o_orb1, SOMO2] - 0.5 * rep_tens[SOMO1, v_orb1, v_orb1, SOMO2] \
                                - 0.5 * rep_tens[SOMO2, SOMO1, SOMO1, SOMO1] + 0.5 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO1])
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:
                CI[row, col] = np.sqrt(2) * (0.5 * rep_tens[o_orb1, SOMO1, SOMO2, o_orb2] - rep_tens[SOMO1, SOMO2, o_orb1, o_orb2])
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                CI[row, col] = np.sqrt(2) * (rep_tens[SOMO1, SOMO2, v_orb1, v_orb2] - 0.5 * rep_tens[v_orb1, SOMO2, SOMO1, v_orb2])
    
    row_index = npairs + 2 * ndocc + 2 * nvirt + 3
    for row in range(row_index, row_index + npairs):
        o_orb1 = (row - row_index) // nvirt
        v_orb1 = (row - row_index) % nvirt + (SOMO2 + 1)
        # <HL2|H|HL2>
        for col in range(row - row_index + npairs, 2*npairs):
            o_orb2 = (col - npairs) // nvirt
            v_orb2 = (col - npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                CI[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb1] - 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) - 0.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1] \
                                    + rep_tens[o_orb1, SOMO1, SOMO1, o_orb1] + rep_tens[o_orb1, SOMO2, SOMO2, o_orb1] + rep_tens[v_orb1, SOMO1, SOMO1, v_orb1] + rep_tens[v_orb1, SOMO2, SOMO2, v_orb1]
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                CI[row, col] = rep_tens[o_orb1, SOMO1, SOMO1, o_orb2] + rep_tens[o_orb1, SOMO2, SOMO2, o_orb2] - rep_tens[o_orb1, o_orb2, v_orb1, v_orb1]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                CI[row, col] = rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] + rep_tens[v_orb1, SOMO2, SOMO2, v_orb2] - rep_tens[v_orb1, v_orb2, o_orb1, o_orb1]
            else:
                CI[row, col] = - rep_tens[o_orb1, o_orb2, v_orb1, v_orb2]
        #<HL2|H|ZHL1>
        for col in range(2*npairs, 3*npairs):
            o_orb2 = (col - 2*npairs) // nvirt
            v_orb2 = (col - 2*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                CI[row, col] = np.sqrt(1.5) * (rep_tens[v_orb1, SOMO1, SOMO2, v_orb1] - rep_tens[SOMO1, o_orb1, o_orb1, SOMO2])
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:
                CI[row, col] = - np.sqrt(1.5) * rep_tens[SOMO2, o_orb1, o_orb2, SOMO1]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                CI[row, col] = np.sqrt(1.5) * rep_tens[SOMO1, v_orb1, v_orb2, SOMO2]
        #<HL2|H|ZHL2>
        for col in range(3*npairs, 4*npairs):
            o_orb2 = (col - 3*npairs) // nvirt
            v_orb2 = (col - 3*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                CI[row, col] = - np.sqrt(1.5) * (rep_tens[v_orb1, SOMO2, SOMO1, v_orb1] - rep_tens[SOMO2, o_orb1, o_orb1, SOMO1])
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:
                CI[row, col] = np.sqrt(1.5) * rep_tens[SOMO1, o_orb1, o_orb2, SOMO2]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                CI[row, col] = - np.sqrt(1.5) * rep_tens[SOMO2, v_orb1, v_orb2, SOMO1]

    row_index = 2 * npairs + 2 * ndocc + 2 * nvirt + 3
    for row in range(row_index, row_index + npairs):
        o_orb1 = (row - row_index) // nvirt
        v_orb1 = (row - row_index) % nvirt + (SOMO2 + 1)
        # <ZHL1|H|ZHL1>
        for col in range(row - row_index + 2*npairs, 3*npairs):
            o_orb2 = (col - 2*npairs) // nvirt
            v_orb2 = (col - 2*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                CI[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[o_orb1] + orb_energies[SOMO1] - orb_energies[SOMO2] + rep_tens[v_orb1, v_orb1, SOMO1, SOMO1] + rep_tens[o_orb1, o_orb1, SOMO2, SOMO2] - rep_tens[o_orb1, o_orb1, SOMO1, SOMO1] - rep_tens[v_orb1, v_orb1, SOMO2, SOMO2] \
                                + 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb1] + 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb1] - 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb1] - 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb1] - rep_tens[SOMO1, SOMO1, SOMO2, SOMO2] + 0.5 * rep_tens[SOMO1, SOMO2, SOMO2, SOMO1] \
                                + 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) + 2 * rep_tens[o_orb1, v_orb1, v_orb1, o_orb1] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb1]
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                CI[row, col] = 2 * rep_tens[o_orb1, v_orb1, v_orb1, o_orb2] + rep_tens[o_orb1, o_orb2, SOMO2, SOMO2] + 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb2] - rep_tens[o_orb1, o_orb2, SOMO1, SOMO1] - rep_tens[o_orb1, o_orb2, v_orb1, v_orb1] \
                                - 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb2]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                CI[row, col] = 2 * rep_tens[v_orb1, o_orb1, o_orb1, v_orb2] + rep_tens[v_orb1, v_orb2, SOMO1, SOMO1] + 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb2] - rep_tens[v_orb1, v_orb2, SOMO2, SOMO2] - rep_tens[v_orb1, v_orb2, o_orb1, o_orb1] \
                                - 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb2]
            else:
                CI[row, col] = 2 * rep_tens[o_orb1, v_orb1, o_orb2, v_orb2] - rep_tens[o_orb1, o_orb2, v_orb1, v_orb2]
        #<ZHL1|H|ZHL2>
        for col in range(3*npairs, 4*npairs):
            o_orb2 = (col - 3*npairs) // nvirt
            v_orb2 = (col - 3*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                CI[row, col] = rep_tens[SOMO1, SOMO2, SOMO2, SOMO1]

    row_index = 3 * npairs + 2 * ndocc + 2 * nvirt + 3
    for row in range(row_index, row_index + npairs):
        o_orb1 = (row - row_index) // nvirt
        v_orb1 = (row - row_index) % nvirt + (SOMO2 + 1)
        # <ZHL2|H|ZHL2>
        for col in range(row - row_index + 3*npairs, 4*npairs):
            o_orb2 = (col - 3*npairs) // nvirt
            v_orb2 = (col - 3*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                CI[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[o_orb1] + orb_energies[SOMO2] - orb_energies[SOMO1] + rep_tens[v_orb1, v_orb1, SOMO2, SOMO2] + rep_tens[o_orb1, o_orb1, SOMO1, SOMO1] - rep_tens[o_orb1, o_orb1, SOMO2, SOMO2] - rep_tens[v_orb1, v_orb1, SOMO1, SOMO1] \
                                + 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb1] + 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb1] - 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb1] - 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb1] - rep_tens[SOMO1, SOMO1, SOMO2, SOMO2] + 0.5 * rep_tens[SOMO1, SOMO2, SOMO2, SOMO1] \
                                + 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) + 2 * rep_tens[o_orb1, v_orb1, v_orb1, o_orb1] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb1]
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                CI[row, col] = 2 * rep_tens[o_orb1, v_orb1, v_orb1, o_orb2] + rep_tens[o_orb1, o_orb2, SOMO1, SOMO1] + 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb2] - rep_tens[o_orb1, o_orb2, SOMO2, SOMO2] - rep_tens[o_orb1, o_orb2, v_orb1, v_orb1] \
                                - 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb2]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                CI[row, col] = 2 * rep_tens[v_orb1, o_orb1, o_orb1, v_orb2] + rep_tens[v_orb1, v_orb2, SOMO2, SOMO2] + 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] - rep_tens[v_orb1, v_orb2, SOMO1, SOMO1] - rep_tens[v_orb1, v_orb2, o_orb1, o_orb1] \
                                - 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb2]
            else:
                CI[row, col] = 2 * rep_tens[o_orb1, v_orb1, o_orb2, v_orb2] - rep_tens[o_orb1, o_orb2, v_orb1, v_orb2]
    
    return CI



def build_singlet_D_block(ndocc, norbs, energy0, orb_energies, rep_tens):
    '''
    Function to build the upper diagonal of the CI matrix for doubly excited states (HOMO to SOMO & SOMO to LUMO) for a diradical system.
    Args:
        ndocc (int): Number of doubly occupied orbitals
        norbs (int): Total number of orbitals
        energy0 (float): Base energy of the mean-field reference state
        orb_energies (numpy.ndarray): Orbital energies for the system
        rep_tens (numpy.ndarray): Representation tensor for the system
    Returns:
        numpy.ndarray: CI matrix for the diradical system
    '''
    # Calculate the size of the CI matrix based on the number of doubly occupied orbitals
    SOMO1 = ndocc # Index of SOMO1
    SOMO2 = ndocc + 1 # Index of SOMO2
    nvirt = norbs - ndocc - 2 # Number of virtual orbitals
    npairs = ndocc * nvirt
    ndcs = int((ndocc ** 2 + ndocc) / 2) # Number of doubly excited core to SOMO singlet CSFs
    ndsv = int((nvirt ** 2 + nvirt) / 2) # Number of doubly excited SOMO to virtual singlet CSFs
    
    row_dim = ndcs + ndsv + 4 * (npairs) + 2 * ndocc + 2 * nvirt + 3
    col_dim = ndcs + ndsv
    CI = np.zeros((row_dim, col_dim))  # Initialize CI Block

    # <OS1|H|CSD>
    o_orb1 = 0
    o_orb2 = 0
    for col in range(0, ndcs):
        if o_orb1 == o_orb2:
            CI[0,col] = - np.sqrt(2) * rep_tens[o_orb1, SOMO1, SOMO2, o_orb1]
        else:
            CI[0,col] = - rep_tens[o_orb1, SOMO1, SOMO2, o_orb2] - rep_tens[o_orb1, SOMO2, SOMO1, o_orb2]
        o_orb2 += 1
        if o_orb2 == ndocc:
            o_orb1 += 1
            o_orb2 = o_orb1
        
    # <OS1|H|SVD>
    v_orb1 = SOMO2 + 1
    v_orb2 = SOMO2 + 1
    for col in range(ndcs, ndcs + ndsv):
        if v_orb1 == v_orb2:
            CI[0,col] = np.sqrt(2) * rep_tens[v_orb1, SOMO1, SOMO2, v_orb1]
        else:
            CI[0,col] = rep_tens[v_orb1, SOMO1, SOMO2, v_orb2] + rep_tens[v_orb1, SOMO2, SOMO1, v_orb2]
        v_orb2 += 1
        if v_orb2 == norbs:
            v_orb1 += 1
            v_orb2 = v_orb1
            
    
    # <ZW-|H|CSD>
    o_orb1 = 0
    o_orb2 = 0
    for col in range(0, ndcs):
        if o_orb1 == o_orb2:
            CI[1,col] = (1 / np.sqrt(2)) * (rep_tens[o_orb1, SOMO2, SOMO2, o_orb1] - rep_tens[o_orb1, SOMO1, SOMO1, o_orb1])
        else:
            CI[1,col] = rep_tens[o_orb1, SOMO2, SOMO2, o_orb2] - rep_tens[o_orb1, SOMO1, SOMO1, o_orb2]
        o_orb2 += 1
        if o_orb2 == ndocc:
            o_orb1 += 1
            o_orb2 = o_orb1
    # <ZW-|H|SVD>
    v_orb1 = SOMO2 + 1
    v_orb2 = SOMO2 + 1
    for col in range(ndcs, ndcs + ndsv):
        if v_orb1 == v_orb2:
            CI[1,col] = (1 / np.sqrt(2)) * (rep_tens[v_orb1, SOMO1, SOMO1, v_orb1] - rep_tens[v_orb1, SOMO2, SOMO2, v_orb1])
        else:
            CI[1,col] = rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] - rep_tens[v_orb1, SOMO2, SOMO2, v_orb2]
        v_orb2 += 1
        if v_orb2 == norbs:
            v_orb1 += 1
            v_orb2 = v_orb1
        
    
    # <ZW+|H|CSD>
    o_orb1 = 0
    o_orb2 = 0
    for col in range(0, ndcs):
        if o_orb1 == o_orb2:
            CI[2,col] = (1 / np.sqrt(2)) * (rep_tens[o_orb1, SOMO1, SOMO1, o_orb1] + rep_tens[o_orb1, SOMO2, SOMO2, o_orb1])
        else:
            CI[2,col] = rep_tens[o_orb1, SOMO1, SOMO1, o_orb2] + rep_tens[o_orb1, SOMO2, SOMO2, o_orb2]
        o_orb2 += 1
        if o_orb2 == ndocc:
            o_orb1 += 1
            o_orb2 = o_orb1
    # <ZW+|H|SVD>
    v_orb1 = SOMO2 + 1
    v_orb2 = SOMO2 + 1
    for col in range(ndcs, ndcs + ndsv):
        if v_orb1 == v_orb2:
            CI[2,col] = (1 / np.sqrt(2)) * (rep_tens[v_orb1, SOMO2, SOMO2, v_orb1] + rep_tens[v_orb1, SOMO1, SOMO1, v_orb1])
        else:
            CI[2,col] = rep_tens[v_orb1, SOMO2, SOMO2, v_orb2] + rep_tens[v_orb1, SOMO1, SOMO1, v_orb2]
        v_orb2 += 1
        if v_orb2 == norbs:
            v_orb1 += 1
            v_orb2 = v_orb1

    
    row_index = 3
    for row in range(row_index, row_index + ndocc):
        o_orb1 = row - row_index
        o_orb2 = 0
        o_orb3 = 0
        # <CS0|H|CSD>
        for col in range(0, ndcs):
            if o_orb2 == o_orb3:
                if o_orb1 == o_orb2:
                    CI[row, col] = np.sqrt(2) * (rep_tens[o_orb1, SOMO2, SOMO1, SOMO1] - rep_tens[o_orb1, SOMO2, o_orb1, o_orb1] + 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, SOMO2] - 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, SOMO2])
                else:
                    CI[row, col] = - np.sqrt(2) * rep_tens[o_orb2, SOMO2, o_orb1, o_orb2]
            else:
                if o_orb1 == o_orb2:
                    CI[row, col] = rep_tens[o_orb3, SOMO2, SOMO1, SOMO1] - rep_tens[o_orb3, SOMO2, o_orb1, o_orb1] - rep_tens[o_orb3, o_orb1, o_orb1, SOMO2] + 0.5 * rep_tens[o_orb3, SOMO2, SOMO2, SOMO2] - 0.5 * rep_tens[o_orb3, SOMO1, SOMO1, SOMO2]
                elif o_orb1 == o_orb3:
                    CI[row, col] = rep_tens[o_orb2, SOMO2, SOMO1, SOMO1] - rep_tens[o_orb2, SOMO2, o_orb1, o_orb1] - rep_tens[o_orb2, o_orb1, o_orb1, SOMO2] + 0.5 * rep_tens[o_orb2, SOMO2, SOMO2, SOMO2] - 0.5 * rep_tens[o_orb2, SOMO1, SOMO1, SOMO2]
                else:
                    CI[row, col] = - rep_tens[o_orb2, SOMO2, o_orb1, o_orb3] - rep_tens[o_orb3, SOMO2, o_orb1, o_orb2]
            o_orb3 += 1
            if o_orb3 == ndocc:
                o_orb2 += 1
                o_orb3 = o_orb2
        # <CS0|H|SVD> = 0
    
    row_index = ndocc + 3
    for row in range(row_index, row_index + ndocc):
        o_orb1 = row - row_index
        o_orb2 = 0
        o_orb3 = 0
        # <CS0'|H|CSD> FIX
        for col in range(0, ndcs):
            if o_orb2 == o_orb3:
                if o_orb1 == o_orb2:
                    CI[row, col] = - np.sqrt(2) * (rep_tens[o_orb1, SOMO1, SOMO2, SOMO2] - rep_tens[o_orb1, SOMO1, o_orb1, o_orb1] + 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, SOMO1] - 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, SOMO1])
                else:
                    CI[row, col] = np.sqrt(2) * rep_tens[o_orb2, SOMO1, o_orb1, o_orb2]
            else:
                if o_orb1 == o_orb2:
                    CI[row, col] = - rep_tens[o_orb3, SOMO1, SOMO2, SOMO2] + rep_tens[o_orb3, SOMO1, o_orb1, o_orb1] + rep_tens[o_orb3, o_orb1, o_orb1, SOMO1] + 0.5 * rep_tens[o_orb3, SOMO1, SOMO1, SOMO1] - 0.5 * rep_tens[o_orb3, SOMO2, SOMO2, SOMO1]
                elif o_orb1 == o_orb3:
                    CI[row, col] = - rep_tens[o_orb2, SOMO1, SOMO2, SOMO2] + rep_tens[o_orb2, SOMO1, o_orb1, o_orb1] + rep_tens[o_orb2, o_orb1, o_orb1, SOMO1] + 0.5 * rep_tens[o_orb2, SOMO1, SOMO1, SOMO1] - 0.5 * rep_tens[o_orb2, SOMO2, SOMO2, SOMO1]
                else:
                    CI[row, col] = rep_tens[o_orb2, SOMO1, o_orb1, o_orb3] + rep_tens[o_orb3, SOMO1, o_orb1, o_orb2]
            o_orb3 += 1
            if o_orb3 == ndocc:
                o_orb2 += 1
                o_orb3 = o_orb2
        # <CS0'|H|SVD> = 0
    
    row_index = 2 * ndocc + 3
    for row in range(row_index, row_index + nvirt):
        # <SV0|H|CSD> = 0
        v_orb1 = row - row_index + (SOMO2 + 1)
        v_orb2 = SOMO2 + 1
        v_orb3 = SOMO2 + 1
        # <SV0|H|SVD>
        for col in range(ndcs, ndcs + ndsv):
            if v_orb2 == v_orb3:
                if v_orb1 == v_orb2:
                    CI[row, col] = np.sqrt(2) * (rep_tens[v_orb1, v_orb1, v_orb1, SOMO2] - rep_tens[v_orb1, SOMO2, SOMO1, SOMO1] + 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, SOMO2] - 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, SOMO2])
                else:
                    CI[row, col] = np.sqrt(2) * rep_tens[v_orb2, SOMO2, v_orb2, v_orb1] 
            else:
                if v_orb1 == v_orb2:
                    CI[row, col] = rep_tens[v_orb3, SOMO2, v_orb1, v_orb1] + rep_tens[SOMO2, v_orb1, v_orb1, v_orb3] - rep_tens[v_orb3, SOMO2, SOMO1, SOMO1] + 0.5 * rep_tens[v_orb3, SOMO1, SOMO1, SOMO2] - 0.5 * rep_tens[v_orb3, SOMO2, SOMO2, SOMO2]
                elif v_orb1 == v_orb3:
                    CI[row, col] = rep_tens[v_orb2, SOMO2, v_orb1, v_orb1] + rep_tens[SOMO2, v_orb1, v_orb1, v_orb2] - rep_tens[v_orb2, SOMO2, SOMO1, SOMO1] + 0.5 * rep_tens[v_orb2, SOMO1, SOMO1, SOMO2] - 0.5 * rep_tens[v_orb2, SOMO2, SOMO2, SOMO2]
                else:
                    CI[row, col] = rep_tens[v_orb2, SOMO2, v_orb1, v_orb3] + rep_tens[v_orb3, SOMO2, v_orb1, v_orb2]
            v_orb3 += 1
            if v_orb3 == norbs:
                v_orb2 += 1
                v_orb3 = v_orb2

                
    row_index = 2 * ndocc + nvirt + 3
    for row in range(row_index, row_index + nvirt):
        # <SV0'|H|CSD> = 0
        v_orb1 = row - row_index + (SOMO2 + 1)
        v_orb2 = SOMO2 + 1
        v_orb3 = SOMO2 + 1
        # <SV0'|H|SVD>
        for col in range(ndcs, ndcs + ndsv):
            if v_orb2 == v_orb3:
                if v_orb1 == v_orb2:
                    CI[row, col] = np.sqrt(2) * (rep_tens[v_orb1, SOMO1, SOMO2, SOMO2] - rep_tens[v_orb1, v_orb1, v_orb1, SOMO1] + 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, SOMO1] - 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, SOMO1])
                else:
                    CI[row, col] = - np.sqrt(2) * rep_tens[v_orb2, SOMO1, v_orb2, v_orb1] 
            else:
                if v_orb1 == v_orb2:
                    CI[row, col] = rep_tens[v_orb3, SOMO1, SOMO2, SOMO2] - rep_tens[v_orb3, SOMO1, v_orb1, v_orb1] - rep_tens[SOMO1, v_orb1, v_orb1, v_orb3] + 0.5 * rep_tens[v_orb3, SOMO1, SOMO1, SOMO1] - 0.5 * rep_tens[v_orb3, SOMO2, SOMO2, SOMO1]
                elif v_orb1 == v_orb3:
                    CI[row, col] = rep_tens[v_orb2, SOMO1, SOMO2, SOMO2] - rep_tens[v_orb2, SOMO1, v_orb1, v_orb1] - rep_tens[SOMO1, v_orb1, v_orb1, v_orb2] + 0.5 * rep_tens[v_orb2, SOMO1, SOMO1, SOMO1] - 0.5 * rep_tens[v_orb2, SOMO2, SOMO2, SOMO1]
                else:
                    CI[row, col] = - rep_tens[v_orb2, SOMO1, v_orb1, v_orb3] - rep_tens[v_orb3, SOMO1, v_orb1, v_orb2]
            v_orb3 += 1
            if v_orb3 == norbs:
                v_orb2 += 1
                v_orb3 = v_orb2
    
    row_index = 2 * ndocc + 2 * nvirt + 3
    for row in range(row_index, row_index + npairs):
        o_orb1 = (row - row_index) // nvirt
        o_orb2 = 0
        o_orb3 = 0
        v_orb1 = (row - row_index) % nvirt + (SOMO2 + 1)
        v_orb2 = SOMO2 + 1
        v_orb3 = SOMO2 + 1
        # <HL1|H|CSD>
        for col in range(0, ndcs):
            if o_orb2 == o_orb3:
                if o_orb1 == o_orb2:
                    CI[row, col] = - rep_tens[o_orb1, SOMO2, SOMO1, v_orb1] - rep_tens[o_orb1, SOMO1, SOMO2, v_orb1]
            else:
                if o_orb1 == o_orb2:
                    CI[row, col] = - (np.sqrt(2) / 2) * (rep_tens[o_orb3, SOMO1, SOMO2, v_orb1] + rep_tens[o_orb3, SOMO2, SOMO1, v_orb1])
                elif o_orb1 == o_orb3:
                    CI[row, col] = - (np.sqrt(2) / 2) * (rep_tens[o_orb2, SOMO1, SOMO2, v_orb1] + rep_tens[o_orb2, SOMO2, SOMO1, v_orb1])
            o_orb3 += 1
            if o_orb3 == ndocc:
                o_orb2 += 1
                o_orb3 = o_orb2
        # <HL1|H|SVD>
        for col in range(ndcs, ndcs + ndsv):
            if v_orb2 == v_orb3:
                if v_orb1 == v_orb2:
                    CI[row, col] = - rep_tens[o_orb1, SOMO2, SOMO1, v_orb1] - rep_tens[o_orb1, SOMO1, SOMO2, v_orb1]
            else:
                if v_orb1 == v_orb2:
                    CI[row, col] = - (np.sqrt(2) / 2) * (rep_tens[o_orb1, SOMO1, SOMO2, v_orb3] + rep_tens[o_orb1, SOMO2, SOMO1, v_orb3])
                elif v_orb1 == v_orb3:
                    CI[row, col] = - (np.sqrt(2) / 2) * (rep_tens[o_orb1, SOMO1, SOMO2, v_orb2] + rep_tens[o_orb1, SOMO2, SOMO1, v_orb2])
            v_orb3 += 1
            if v_orb3 == norbs:
                v_orb2 += 1
                v_orb3 = v_orb2
    
    row_index = npairs + 2 * ndocc + 2 * nvirt + 3
    for row in range(row_index, row_index + npairs):
        o_orb1 = (row - row_index) // nvirt
        o_orb2 = 0
        o_orb3 = 0
        v_orb1 = (row - row_index) % nvirt + (SOMO2 + 1)
        v_orb2 = SOMO2 + 1
        v_orb3 = SOMO2 + 1
        # <HL2|H|CSD>
        for col in range(0, ndcs):
            if o_orb2 == o_orb3:
                if o_orb1 == o_orb2:
                    CI[row, col] = np.sqrt(3) * (rep_tens[o_orb1, SOMO2, SOMO1, v_orb1] - rep_tens[o_orb1, SOMO1, SOMO2, v_orb1])
            else:
                if o_orb1 == o_orb2:
                    CI[row, col] = (np.sqrt(1.5)) * (rep_tens[o_orb3, SOMO2, SOMO1, v_orb1] - rep_tens[o_orb3, SOMO1, SOMO2, v_orb1])
                elif o_orb1 == o_orb3:
                    CI[row, col] = (np.sqrt(1.5)) * (rep_tens[o_orb2, SOMO2, SOMO1, v_orb1] - rep_tens[o_orb2, SOMO1, SOMO2, v_orb1])
            o_orb3 += 1
            if o_orb3 == ndocc:
                o_orb2 += 1
                o_orb3 = o_orb2
        #<HL2|H|SLD>
        for col in range(ndcs, ndcs + ndsv):
            if v_orb2 == v_orb3:
                if v_orb1 == v_orb2:
                    CI[row, col] = np.sqrt(3) * (rep_tens[o_orb1, SOMO2, SOMO1, v_orb1] - rep_tens[o_orb1, SOMO1, SOMO2, v_orb1])
            else:
                if v_orb1 == v_orb2:
                    CI[row, col] = np.sqrt(1.5) * (rep_tens[o_orb1, SOMO2, SOMO1, v_orb3] - rep_tens[o_orb1, SOMO1, SOMO2, v_orb3])
                elif v_orb1 == v_orb3:
                    CI[row, col] = np.sqrt(1.5) * (rep_tens[o_orb1, SOMO2, SOMO1, v_orb2] - rep_tens[o_orb1, SOMO1, SOMO2, v_orb2])
            v_orb3 += 1
            if v_orb3 == norbs:
                v_orb2 += 1
                v_orb3 = v_orb2

    row_index = 2 * npairs + 2 * ndocc + 2 * nvirt + 3
    for row in range(row_index, row_index + npairs):
        o_orb1 = (row - row_index) // nvirt
        o_orb2 = 0
        o_orb3 = 0
        v_orb1 = (row - row_index) % nvirt + (SOMO2 + 1)
        v_orb2 = SOMO2 + 1
        v_orb3 = SOMO2 + 1
        # <ZHL1|H|CSD>
        for col in range(0, ndcs):
            if o_orb2 == o_orb3:
                if o_orb1 == o_orb2:
                    CI[row, col] = np.sqrt(2) * rep_tens[o_orb1, SOMO2, SOMO2, v_orb1]
            else:
                if o_orb1 == o_orb2:
                    CI[row, col] = rep_tens[o_orb3, SOMO2, SOMO2, v_orb1]
                elif o_orb1 == o_orb3:
                    CI[row, col] = rep_tens[o_orb2, SOMO2, SOMO2, v_orb1]
            o_orb3 += 1
            if o_orb3 == ndocc:
                o_orb2 += 1
                o_orb3 = o_orb2
        # <ZHL1|H|SVD>
        for col in range(ndcs, ndcs + ndsv):
            if v_orb2 == v_orb3:
                if v_orb1 == v_orb2:
                    CI[row, col] = - np.sqrt(2) * rep_tens[o_orb1, SOMO1, SOMO1, v_orb1]
            else:
                if v_orb1 == v_orb2:
                    CI[row, col] = - rep_tens[o_orb1, SOMO1, SOMO1, v_orb3]
                elif v_orb1 == v_orb3:
                    CI[row, col] = - rep_tens[o_orb1, SOMO1, SOMO1, v_orb2]
            v_orb3 += 1
            if v_orb3 == norbs:
                v_orb2 += 1
                v_orb3 = v_orb2

    row_index = 3 * npairs + 2 * ndocc + 2 * nvirt + 3
    for row in range(row_index, row_index + npairs):
        o_orb1 = (row - row_index) // nvirt
        o_orb2 = 0
        o_orb3 = 0
        v_orb1 = (row - row_index) % nvirt + (SOMO2 + 1)
        v_orb2 = SOMO2 + 1
        v_orb3 = SOMO2 + 1
        # <ZHL2|H|CSD>
        for col in range(0, ndcs):
            if o_orb2 == o_orb3:
                if o_orb1 == o_orb2:
                    CI[row, col] = np.sqrt(2) * rep_tens[o_orb1, SOMO1, SOMO1, v_orb1]
            else:
                if o_orb1 == o_orb2:
                    CI[row, col] = rep_tens[o_orb3, SOMO1, SOMO1, v_orb1]
                elif o_orb1 == o_orb3:
                    CI[row, col] = rep_tens[o_orb2, SOMO1, SOMO1, v_orb1]
            o_orb3 += 1
            if o_orb3 == ndocc:
                o_orb2 += 1
                o_orb3 = o_orb2
        # <ZHL2|H|SVD>
        for col in range(ndcs, ndcs + ndsv):
            if v_orb2 == v_orb3:
                if v_orb1 == v_orb2:
                    CI[row, col] = - np.sqrt(2) * rep_tens[o_orb1, SOMO2, SOMO2, v_orb1]
            else:
                if v_orb1 == v_orb2:
                    CI[row, col] = - rep_tens[o_orb1, SOMO2, SOMO2, v_orb3]
                elif v_orb1 == v_orb3:
                    CI[row, col] = - rep_tens[o_orb1, SOMO2, SOMO2, v_orb2]
            v_orb3 += 1
            if v_orb3 == norbs:
                v_orb2 += 1
                v_orb3 = v_orb2
    
    
    row_index = 4 * npairs + 2 * ndocc + 2 * nvirt + 3
    o_orb1 = 0
    o_orb2 = 0
    for row in range(row_index, row_index + ndcs):
        o_orb3 = o_orb1
        o_orb4 = o_orb2
        # <HSD|H|HSD>
        for col in range(row - row_index, ndcs):
            if o_orb1 == o_orb2 and o_orb3 == o_orb4:
                if o_orb1 == o_orb3:
                    CI[row, col] = energy0 - 2 * orb_energies[o_orb1] + orb_energies[SOMO1] + orb_energies[SOMO2] + rep_tens[o_orb1, o_orb1, o_orb1, o_orb1] - 2 * rep_tens[o_orb1,o_orb1,SOMO1,SOMO1] - 2 * rep_tens[o_orb1,o_orb1,SOMO2,SOMO2] \
                        + rep_tens[o_orb1, SOMO1, SOMO1, o_orb1] + rep_tens[o_orb1, SOMO2, SOMO2, o_orb1] + rep_tens[SOMO1,SOMO1,SOMO2,SOMO2] - 0.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1] + 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2])
                else:
                    CI[row, col] = rep_tens[o_orb1,o_orb3,o_orb3,o_orb1]
            elif o_orb1 == o_orb2 and o_orb3 != o_orb4:
                if o_orb1 == o_orb3:
                    CI[row, col] = np.sqrt(2) * (rep_tens[o_orb1, o_orb4, o_orb1, o_orb1] - rep_tens[o_orb1, o_orb4, SOMO1, SOMO1] - rep_tens[o_orb1, o_orb4, SOMO2, SOMO2] + 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb4] + 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb4])
                elif o_orb1 == o_orb4:
                    CI[row, col] = np.sqrt(2) * (rep_tens[o_orb1, o_orb3, o_orb1, o_orb1] - rep_tens[o_orb1, o_orb3, SOMO1, SOMO1] - rep_tens[o_orb1, o_orb3, SOMO2, SOMO2] + 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb3] + 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb3])
                else:
                    CI[row, col] = np.sqrt(2) * rep_tens[o_orb1, o_orb3, o_orb4, o_orb1]
            elif o_orb3 == o_orb4 and o_orb1 != o_orb2:
                if o_orb3 == o_orb1:
                    CI[row, col] = np.sqrt(2) * (rep_tens[o_orb3, o_orb2, o_orb3, o_orb3] - rep_tens[o_orb3, o_orb2, SOMO1, SOMO1] - rep_tens[o_orb3, o_orb2, SOMO2, SOMO2] + 0.5 * rep_tens[o_orb3, SOMO1, SOMO1, o_orb2] + 0.5 * rep_tens[o_orb3, SOMO2, SOMO2, o_orb2])
                elif o_orb3 == o_orb2:
                    CI[row, col] = np.sqrt(2) * (rep_tens[o_orb3, o_orb1, o_orb3, o_orb3] - rep_tens[o_orb3, o_orb1, SOMO1, SOMO1] - rep_tens[o_orb3, o_orb1, SOMO2, SOMO2] + 0.5 * rep_tens[o_orb3, SOMO1, SOMO1, o_orb1] + 0.5 * rep_tens[o_orb3, SOMO2, SOMO2, o_orb1])
                else:
                    CI[row, col] = np.sqrt(2) * rep_tens[o_orb3, o_orb1, o_orb2, o_orb3]
            else:
                if o_orb1 == o_orb3 and o_orb2 == o_orb4:
                    CI[row, col] = energy0 - orb_energies[o_orb1] - orb_energies[o_orb2] + orb_energies[SOMO1] + orb_energies[SOMO2] + rep_tens[o_orb1, o_orb1, o_orb2, o_orb2] - rep_tens[o_orb1, o_orb1, SOMO1, SOMO1] - rep_tens[o_orb1, o_orb1, SOMO2, SOMO2] \
                        - rep_tens[o_orb2, o_orb2, SOMO1, SOMO1] - rep_tens[o_orb2, o_orb2, SOMO2, SOMO2] + 0.5 * (rep_tens[o_orb1, SOMO1, SOMO1, o_orb1] + rep_tens[o_orb1, SOMO2, SOMO2, o_orb1] + rep_tens[o_orb2, SOMO1, SOMO1, o_orb2] + rep_tens[o_orb2, SOMO2, SOMO2, o_orb2]) \
                        + 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) + rep_tens[SOMO1,SOMO1,SOMO2,SOMO2] - 0.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1] + rep_tens[o_orb1, o_orb2, o_orb2, o_orb1]
                elif o_orb1 == o_orb3 and o_orb2 != o_orb4:
                    CI[row,col] = rep_tens[o_orb2, o_orb4, o_orb1, o_orb1] + rep_tens[o_orb2, o_orb1, o_orb1, o_orb4] - rep_tens[o_orb2, o_orb4, SOMO1, SOMO1] - rep_tens[o_orb2, o_orb4, SOMO2, SOMO2]+ 0.5 * rep_tens[o_orb2, SOMO1, SOMO1, o_orb4] + 0.5 * rep_tens[o_orb2, SOMO2, SOMO2, o_orb4]
                elif o_orb2 == o_orb4 and o_orb1 != o_orb3:
                    CI[row,col] = rep_tens[o_orb1, o_orb3, o_orb2, o_orb2] + rep_tens[o_orb1, o_orb2, o_orb2, o_orb3] - rep_tens[o_orb1, o_orb3, SOMO1, SOMO1] - rep_tens[o_orb1, o_orb3, SOMO2, SOMO2]+ 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb3] + 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb3]
                else:
                    CI[row,col] = rep_tens[o_orb1, o_orb3, o_orb2, o_orb4] + rep_tens[o_orb1, o_orb4, o_orb2, o_orb3]
            o_orb4 += 1
            if o_orb4 == ndocc:
                o_orb3 += 1
                o_orb4 = o_orb3
        # <HSD|H|SVD> = 0
        o_orb2 += 1
        if o_orb2 == ndocc:
            o_orb1 += 1
            o_orb2 = o_orb1
    
    row_index = ndcs + 4 * npairs + 2 * ndocc + 2 * nvirt + 3
    v_orb1 = SOMO2 + 1
    v_orb2 = SOMO2 + 1
    for row in range(row_index, row_index + ndsv):
        v_orb3 = v_orb1
        v_orb4 = v_orb2
        # <SVD|H|SVD>
        for col in range(row - row_index + ndcs, ndcs + ndsv):
            if v_orb1 == v_orb2 and v_orb3 == v_orb4:
                if v_orb1 == v_orb3:
                    CI[row, col] = energy0 + 2 * orb_energies[v_orb1] - orb_energies[SOMO1] - orb_energies[SOMO2] + rep_tens[v_orb1, v_orb1, v_orb1, v_orb1] - 2 * rep_tens[v_orb1,v_orb1,SOMO1,SOMO1] - 2 * rep_tens[v_orb1,v_orb1,SOMO2,SOMO2] \
                        + rep_tens[v_orb1, SOMO1, SOMO1, v_orb1] + rep_tens[v_orb1, SOMO2, SOMO2, v_orb1] + rep_tens[SOMO1,SOMO1,SOMO2,SOMO2] - 0.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1] + 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2])
                else:
                    CI[row, col] = rep_tens[v_orb1,v_orb3,v_orb3,v_orb1]
            elif v_orb1 == v_orb2 and v_orb3 != v_orb4:
                if v_orb1 == v_orb3:
                    CI[row, col] = np.sqrt(2) * (rep_tens[v_orb1, v_orb4, v_orb1, v_orb1] - rep_tens[v_orb1, v_orb4, SOMO1, SOMO1] - rep_tens[v_orb1, v_orb4, SOMO2, SOMO2] + 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb4] + 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb4])
                elif v_orb1 == v_orb4:
                    CI[row, col] = np.sqrt(2) * (rep_tens[v_orb1, v_orb3, v_orb1, v_orb1] - rep_tens[v_orb1, v_orb3, SOMO1, SOMO1] - rep_tens[v_orb1, v_orb3, SOMO2, SOMO2] + 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb3] + 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb3])
                else:
                    CI[row,col] = np.sqrt(2) * rep_tens[v_orb1,v_orb3,v_orb4,v_orb1]
            elif v_orb1 != v_orb2 and v_orb3 == v_orb4:
                if v_orb3 == v_orb1:
                    CI[row, col] = np.sqrt(2) * (rep_tens[v_orb3, v_orb2, v_orb3, v_orb3] - rep_tens[v_orb3, v_orb2, SOMO1, SOMO1] - rep_tens[v_orb3, v_orb2, SOMO2, SOMO2] + 0.5 * rep_tens[v_orb3, SOMO1, SOMO1, v_orb2] + 0.5 * rep_tens[v_orb3, SOMO2, SOMO2, v_orb2])
                elif v_orb3 == v_orb2:
                    CI[row, col] = np.sqrt(2) * (rep_tens[v_orb3, v_orb1, v_orb3, v_orb3] - rep_tens[v_orb3, v_orb1, SOMO1, SOMO1] - rep_tens[v_orb3, v_orb1, SOMO2, SOMO2] + 0.5 * rep_tens[v_orb3, SOMO1, SOMO1, v_orb1] + 0.5 * rep_tens[v_orb3, SOMO2, SOMO2, v_orb1])
                else:
                    CI[row,col] = np.sqrt(2) * rep_tens[v_orb3,v_orb1,v_orb2,v_orb3]
            else:
                if v_orb1 == v_orb3 and v_orb2 == v_orb4:
                    CI[row, col] = energy0 + orb_energies[v_orb1] + orb_energies[v_orb2] - orb_energies[SOMO1] - orb_energies[SOMO2] + rep_tens[v_orb1, v_orb1, v_orb2, v_orb2] - rep_tens[v_orb1, v_orb1, SOMO1, SOMO1] - rep_tens[v_orb1, v_orb1, SOMO2, SOMO2] \
                        - rep_tens[v_orb2, v_orb2, SOMO1, SOMO1] - rep_tens[v_orb2, v_orb2, SOMO2, SOMO2] + 0.5 * (rep_tens[v_orb1, SOMO1, SOMO1, v_orb1] + rep_tens[v_orb1, SOMO2, SOMO2, v_orb1] + rep_tens[v_orb2, SOMO1, SOMO1, v_orb2] + rep_tens[v_orb2, SOMO2, SOMO2, v_orb2]) \
                        + 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) + rep_tens[SOMO1,SOMO1,SOMO2,SOMO2] - 0.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1] + rep_tens[v_orb1, v_orb2, v_orb2, v_orb1]
                elif v_orb1 == v_orb3 and v_orb2 != v_orb4:
                    CI[row, col] = rep_tens[v_orb2, v_orb4, v_orb1, v_orb1] + rep_tens[v_orb2, v_orb1, v_orb1, v_orb4] - rep_tens[v_orb2, v_orb4, SOMO1, SOMO1] - rep_tens[v_orb2, v_orb4, SOMO2, SOMO2]+ 0.5 * rep_tens[v_orb2, SOMO1, SOMO1, v_orb4] + 0.5 * rep_tens[v_orb2, SOMO2, SOMO2, v_orb4]
                elif v_orb1 != v_orb3 and v_orb2 == v_orb4:
                    CI[row, col] = rep_tens[v_orb1, v_orb3, v_orb2, v_orb2] + rep_tens[v_orb1, v_orb2, v_orb2, v_orb3] - rep_tens[v_orb1, v_orb3, SOMO1, SOMO1] - rep_tens[v_orb1, v_orb3, SOMO2, SOMO2]+ 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb3] + 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb3]
                else:
                    CI[row, col] = rep_tens[v_orb1, v_orb3, v_orb2, v_orb4] + rep_tens[v_orb1, v_orb4, v_orb2, v_orb3]
            v_orb4 += 1
            if v_orb4 == norbs:
                v_orb3 += 1
                v_orb4 = v_orb3
        v_orb2 += 1
        if v_orb2 == norbs:
            v_orb1 += 1
            v_orb2 = v_orb1
    
    return CI


def build_triplet_ref_block(ndocc, energy0, rep_tens):
    '''
    Function to build the CI matrix for 1 triplet reference states for a diradical system - the open-shell triplet (OS3).
    Args: 
        ndocc (int): Number of doubly occupied orbitals
        energy0 (float): Base energy of the mean-field reference state
        rep_tens (numpy.ndarray): Representation tensor for the system
    Returns:
        numpy.ndarray: CI matrix for the diradical system
    '''
    # Calculate the size of the CI matrix based on the number of doubly occupied orbitals
    CI = np.zeros((1, 1))  # Initialize a 1x1 CI matrix
    SOMO1 = ndocc # Index of SOMO1
    SOMO2 = ndocc + 1 # Index of SOMO2

    # <OS3|H|OS3>
    CI[0,0] = energy0 - 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) - (0.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1])

    return CI


def build_triplet_CS_SV_block(ndocc, norbs, energy0, orb_energies, rep_tens):
    '''
    Function to build the upper diagonal of the CI matrix for singlet reference states for a diradical system - the closed-shell singlet (CS) and the single-reference singlet (SV).
    Args:
        ndocc (int): Number of doubly occupied orbitals
        norbs (int): Total number of orbitals
        energy0 (float): Base energy of the mean-field reference state
        orb_energies (numpy.ndarray): Orbital energies for the system
        rep_tens (numpy.ndarray): Representation tensor for the system
    Returns:
        numpy.ndarray: CI matrix for the diradical system
    '''
    # Calculate the size of the CI matrix based on the number of doubly occupied orbitals
    SOMO1 = ndocc # Index of SOMO1
    SOMO2 = ndocc + 1 # Index of SOMO2
    nvirt = norbs - ndocc - 2 # Number of virtual orbitals
    
    row_dim = 2 * ndocc + 2 * nvirt + 1
    col_dim = 2 * ndocc + 2 * nvirt
    CI = np.zeros((row_dim, col_dim))  # Initialize CI Block
    
    # <OS3|H|CS0> (CHECKED)
    for col in range(0, ndocc):
        o_orb = col
        CI[0,col] = 0.5 * rep_tens[o_orb,SOMO2,SOMO2,SOMO1] + 0.5 * rep_tens[o_orb,SOMO1,SOMO1,SOMO1]
    # <OS3|H|CS0'> 
    for col in range(ndocc, 2 * ndocc):
        o_orb = col - ndocc
        CI[0,col] = 0.5 * rep_tens[o_orb,SOMO2,SOMO2,SOMO2] + 0.5 * rep_tens[o_orb,SOMO1,SOMO1,SOMO2]
    # <OS3|H|SV0>
    for col in range(2 * ndocc, 2 * ndocc + nvirt):
        v_orb = col - (2 * ndocc) + (SOMO2 + 1)
        CI[0,col] = - 0.5 * rep_tens[v_orb,SOMO2,SOMO2,SOMO1] - 0.5 * rep_tens[v_orb,SOMO1,SOMO1,SOMO1]
    # <OS3|H|SV0'>
    for col in range(2 * ndocc + nvirt, 2 * ndocc + 2 * nvirt):
        v_orb = col - (2 * ndocc + nvirt) + (SOMO2 + 1)
        CI[0,col] = 0.5 * rep_tens[v_orb,SOMO2,SOMO2,SOMO2] + 0.5 * rep_tens[v_orb,SOMO1,SOMO1,SOMO2]

    row_index = 1
    for row in range(row_index, row_index + ndocc):
        o_orb1 = row - row_index
        # <CS0|H|CS0>
        for col in range(row - row_index, ndocc):
            o_orb2 = col
            if o_orb1 == o_orb2:
                CI[row, col] = energy0 + orb_energies[SOMO1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, SOMO1, SOMO1] + 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] \
                             - 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] - 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb1] + 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb1]
            else:    
                CI[row, col] = 0.5 * rep_tens[o_orb2, SOMO1, SOMO1, o_orb1] - 0.5 * rep_tens[o_orb2, SOMO2, SOMO2, o_orb1] - rep_tens[o_orb2, o_orb1, SOMO1, SOMO1]
        # <CS0|H|CS0'>
        for col in range(ndocc, 2*ndocc):
            o_orb2 = col - ndocc
            if o_orb1 == o_orb2:
                CI[row, col] = 0.5 * rep_tens[SOMO1, SOMO2, SOMO1, SOMO1] + 0.5 * rep_tens[SOMO1, SOMO2, SOMO2, SOMO2] + rep_tens[SOMO1, o_orb1, o_orb1, SOMO2] - rep_tens[o_orb1, o_orb1, SOMO1, SOMO2]
            else:    
                CI[row, col] = rep_tens[o_orb1, SOMO1, SOMO2, o_orb2] - rep_tens[o_orb1, o_orb2, SOMO1, SOMO2]
        # <CS0|H|SV0>
        for col in range(2*ndocc, 2*ndocc + nvirt):
            v_orb = col - 2*ndocc + (SOMO2 + 1)
            CI[row, col] = rep_tens[o_orb1, SOMO1, SOMO1, v_orb] 
        # <CS0|H|SV0'>
        for col in range(2*ndocc + nvirt, 2*ndocc + 2*nvirt):
            v_orb = col - (2*ndocc + nvirt) + (SOMO2 + 1)
            CI[row, col] = - rep_tens[o_orb1, SOMO1, SOMO2, v_orb]
            
    
    row_index = ndocc + 1
    for row in range(row_index, row_index + ndocc):
        o_orb1 = row - row_index
        # <CS0'|H|CS0'>
        for col in range(row - row_index + ndocc, 2*ndocc):
            o_orb2 = col - ndocc
            if o_orb1 == o_orb2:
                CI[row, col] = energy0 + orb_energies[SOMO2] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, SOMO2, SOMO2] + 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] \
                             - 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] - 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb1] + 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb1]
            else:    
                CI[row, col] =  0.5 * rep_tens[o_orb2, SOMO2, SOMO2, o_orb1] - rep_tens[o_orb2, o_orb1, SOMO2, SOMO2] - 0.5 * rep_tens[o_orb2, SOMO1, SOMO1, o_orb1]
        # <CS0'|H|SV0>
        for col in range(2*ndocc, 2*ndocc + nvirt):
            v_orb = col - 2*ndocc + (SOMO2 + 1)
            CI[row, col] = rep_tens[o_orb1, SOMO2, SOMO1, v_orb]
        # <CS0'|H|SV0'>
        for col in range(2*ndocc + nvirt,  2*ndocc + 2*nvirt):
            v_orb = col - (2*ndocc + nvirt) + (SOMO2 + 1)
            CI[row, col] = - rep_tens[o_orb1, SOMO2, SOMO2, v_orb]
    
    row_index = 2 * ndocc + 1
    for row in range(row_index, row_index + nvirt):
        v_orb1 = row - row_index + (SOMO2 + 1)
         # <SV0|H|SV0>
        for col in range(row - row_index + 2*ndocc, 2*ndocc + nvirt):
            v_orb2 = col - (2*ndocc) + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                CI[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[SOMO1] - rep_tens[v_orb1, v_orb1, SOMO1, SOMO1] + 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] \
                             - 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] - 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb1] + 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb1]
            else:    
                CI[row, col] = 0.5 * rep_tens[v_orb2, SOMO1, SOMO1, v_orb1] - 0.5 * rep_tens[v_orb2, SOMO2, SOMO2, v_orb1] - rep_tens[v_orb2, v_orb1, SOMO1, SOMO1]
        # <SV0|H|SV0'>
        for col in range(2*ndocc + nvirt, 2*ndocc + 2*nvirt):
            v_orb2 = col - (2*ndocc + nvirt) + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                CI[row, col] = rep_tens[v_orb1, v_orb1, SOMO1, SOMO2] - 0.5 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO2] - 0.5 * rep_tens[SOMO1, SOMO2, SOMO2, SOMO2]  - rep_tens[v_orb1, SOMO1, SOMO2, v_orb1]
            else:    
                CI[row, col] = rep_tens[v_orb1, v_orb2, SOMO1, SOMO2] - rep_tens[v_orb1, SOMO1, SOMO2, v_orb2]
    
    row_index = 2 * ndocc + nvirt + 1
    # <SV0'|H|SV0'>
    for row in range(row_index, row_index + nvirt):
        v_orb1 = row - row_index + (SOMO2 + 1)
        for col in range(row - row_index + 2*ndocc + nvirt, 2*ndocc + 2*nvirt):
            v_orb2 = col - (2*ndocc + nvirt) + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                CI[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[SOMO2] - rep_tens[v_orb1, v_orb1, SOMO2, SOMO2] + 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] \
                             - 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] - 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb1] + 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb1]
            else:    
                CI[row, col] =  0.5 * rep_tens[v_orb2, SOMO2, SOMO2, v_orb1] - 0.5 * rep_tens[v_orb2, SOMO1, SOMO1, v_orb1] - rep_tens[v_orb2, v_orb1, SOMO2, SOMO2]

    return CI


def build_triplet_HL_block(ndocc, norbs, energy0, orb_energies, rep_tens):
    '''
    Function to build the upper diagonal of the CI matrix for HOMO to LUMO excited states for a diradical system.
    Args:
        ndocc (int): Number of doubly occupied orbitals
        norbs (int): Total number of orbitals
        energy0 (float): Base energy of the mean-field reference state
        orb_energies (numpy.ndarray): Orbital energies for the system
        rep_tens (numpy.ndarray): Representation tensor for the system
    Returns:
        numpy.ndarray: CI matrix for the diradical system
    '''
    # Calculate the size of the CI matrix based on the number of doubly occupied orbitals
    SOMO1 = ndocc # Index of SOMO1
    SOMO2 = ndocc + 1 # Index of SOMO2
    nvirt = norbs - ndocc - 2 # Number of virtual orbitals
    npairs = ndocc * nvirt
    
    row_dim = 5 * (npairs) + 2 * ndocc + 2 * nvirt + 1
    col_dim = 5 * (npairs)
    CI = np.zeros((row_dim, col_dim))  # Initialize CI Block

    # <OS3|H|HL1> = 0
    # <OS3|H|HL2>
    for col in range(npairs, 2*npairs):
        o_orb = (col - npairs) // nvirt # Increase o_orb after every ndocc cols
        v_orb = (col - npairs) % nvirt + (SOMO2 + 1) # Increase v_orb for every col then reset after ndocc cols
        CI[0,col] = (1 / np.sqrt(2)) * (rep_tens[o_orb, SOMO1, SOMO1, v_orb] - rep_tens[o_orb, SOMO2, SOMO2, v_orb])
    # <OS3|H|HL3>
    for col in range(2*npairs, 3*npairs):
        o_orb = (col - 2*npairs) // nvirt
        v_orb = (col - 2*npairs) % nvirt + (SOMO2 + 1)
        CI[0,col] = rep_tens[o_orb, SOMO1, SOMO1, v_orb] + rep_tens[o_orb, SOMO2, SOMO2, v_orb]
    # <OS3|H|ZHL1> 
    for col in range(3*npairs, 4*npairs):
        o_orb = (col - 3*npairs) // nvirt
        v_orb = (col - 3*npairs) % nvirt + (SOMO2 + 1)
        CI[0,col] = - rep_tens[o_orb, SOMO1, SOMO2, v_orb]
    # <OS3|H|ZHL2>
    for col in range(4*npairs, 5*npairs):
        o_orb = (col - 4*npairs) // nvirt
        v_orb = (col - 4*npairs) % nvirt + (SOMO2 + 1)
        CI[0,col] = rep_tens[o_orb, SOMO2, SOMO1, v_orb]


    row_index = 1
    for row in range(row_index, row_index + ndocc):
        o_orb1 = row - row_index
        # <CS0|H|HL1>
        for col in range(0, npairs):
            o_orb2 = col // nvirt
            v_orb = col % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                CI[row, col] = (1 / np.sqrt(2)) * (2*rep_tens[SOMO1, o_orb1, o_orb1, v_orb] + 0.5 * rep_tens[SOMO1, SOMO2, SOMO2, v_orb] + 0.5 * rep_tens[SOMO1, SOMO1, SOMO1, v_orb] - rep_tens[SOMO1, v_orb, o_orb1, o_orb1])
            else:    
                CI[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[SOMO1, o_orb1, o_orb2, v_orb] - rep_tens[SOMO1, v_orb, o_orb1, o_orb2])
        # <CS0|H|HL2>
        for col in range(npairs, 2*npairs):
            o_orb2 = (col - npairs) // nvirt
            v_orb = (col - npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                CI[row, col] = (1 / np.sqrt(2)) * (0.5 * rep_tens[SOMO1, SOMO1, SOMO1, v_orb] - 1.5 * rep_tens[SOMO1, SOMO2, SOMO2, v_orb] - rep_tens[o_orb1, o_orb1, SOMO1, v_orb])
            else:    
                CI[row, col] = - (1 / np.sqrt(2)) * rep_tens[SOMO1, v_orb, o_orb1, o_orb2]
        # <CS0|H|HL3>
        for col in range(2*npairs, 3*npairs):
            o_orb2 = (col - 2*npairs) // nvirt
            v_orb = (col - 2*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                CI[row, col] = 0.5 * rep_tens[SOMO1, SOMO1, SOMO1, v_orb] + 0.5 * rep_tens[SOMO1, SOMO2, SOMO2, v_orb] - rep_tens[o_orb1, o_orb1, SOMO1, v_orb]
            else:    
                CI[row, col] = - rep_tens[SOMO1, v_orb, o_orb1, o_orb2]
        # <CS0|H|ZHL1>
        for col in range(3*npairs, 4*npairs):
            o_orb2 = (col - 3*npairs) // nvirt
            v_orb = (col - 3*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                CI[row, col] = rep_tens[o_orb1, o_orb1, SOMO2, v_orb] + 0.5 * rep_tens[SOMO2, SOMO1, SOMO1, v_orb] + 0.5 * rep_tens[SOMO2, SOMO2, SOMO2, v_orb] - rep_tens[SOMO2, v_orb, SOMO1, SOMO1]
            else:    
                CI[row, col] = rep_tens[SOMO2, v_orb, o_orb1, o_orb2] 
        # <CS0|H|ZHL2>
        for col in range(4*npairs, 5*npairs):
            o_orb2 = (col - 4*npairs) // nvirt
            v_orb = (col - 4*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                CI[row, col] = rep_tens[SOMO2, SOMO1, SOMO1, v_orb]
    
    row_index = ndocc + 1
    for row in range(row_index, row_index + ndocc):
        o_orb1 = row - row_index
        # <CS0'|H|HL1>
        for col in range(0, npairs):
            o_orb2 = col // nvirt
            v_orb = col % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                CI[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[SOMO2, o_orb1, o_orb1, v_orb] + 0.5 * rep_tens[SOMO2, SOMO2, SOMO2, v_orb] - rep_tens[o_orb1, o_orb1, SOMO2, v_orb] + 0.5 * rep_tens[SOMO2, SOMO1, SOMO1, v_orb])
            else:
                CI[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[SOMO2, o_orb1, o_orb2, v_orb] - rep_tens[SOMO2, v_orb, o_orb1, o_orb2])
        # <CS0'|H|HL2>
        for col in range(npairs, 2 * npairs):
            o_orb2 = (col - npairs) // nvirt
            v_orb = (col - npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                CI[row, col] = (1 / np.sqrt(2)) * (1.5 * rep_tens[SOMO2, SOMO1, SOMO1, v_orb] + rep_tens[o_orb1, o_orb1, SOMO2, v_orb] - 0.5 * rep_tens[SOMO2, SOMO2, SOMO2, v_orb])
            else:    
                CI[row, col] = (1 / np.sqrt(2)) * rep_tens[SOMO2, v_orb, o_orb1, o_orb2]
        # <CS0'|H|HL3>
        for col in range(2*npairs, 3*npairs):
            o_orb2 = (col - 2*npairs) // nvirt
            v_orb = (col - 2*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                CI[row, col] = 0.5 * rep_tens[SOMO2, SOMO2, SOMO2, v_orb] + 0.5 * rep_tens[SOMO2, SOMO1, SOMO1, v_orb] - rep_tens[o_orb1, o_orb1, SOMO2, v_orb]
            else:    
                CI[row, col] = - rep_tens[SOMO2, v_orb, o_orb1, o_orb2]
        # <CS0'|H|ZHL1>
        for col in range(3*npairs, 4*npairs):
            o_orb2 = (col - 3*npairs) // nvirt
            v_orb = (col - 3*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                CI[row, col] = - rep_tens[SOMO1, SOMO2, SOMO2, v_orb]
        # <CS0'|H|ZHL2>
        for col in range(4*npairs, 5*npairs):
            o_orb2 = (col - 4*npairs) // nvirt
            v_orb = (col - 4*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                CI[row, col] = rep_tens[SOMO1, v_orb, SOMO2, SOMO2] - rep_tens[SOMO1, v_orb, o_orb1, o_orb1] - 0.5 * rep_tens[SOMO1, SOMO1, SOMO1, v_orb] - 0.5 * rep_tens[SOMO1, SOMO2, SOMO2, v_orb]
            else:
                CI[row, col] = - rep_tens[SOMO1, v_orb, o_orb1, o_orb2]
    
    row_index = 2 * ndocc + 1
    for row in range(row_index, row_index + nvirt):
        v_orb1 = row - row_index + (SOMO2 + 1)
        # <SV0|H|HL1>
        for col in range(0, npairs):
            o_orb = col // nvirt
            v_orb2 = col % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                CI[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[o_orb, v_orb1, v_orb1, SOMO1] + 0.5 * rep_tens[SOMO1, SOMO1, SOMO1, o_orb] - rep_tens[v_orb1, v_orb1, SOMO1, o_orb] + 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO1])
            else:    
                CI[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[o_orb, v_orb2, v_orb1, SOMO1] - rep_tens[v_orb1, v_orb2, SOMO1, o_orb])
        # <SV0|H|HL2>
        for col in range(npairs, 2*npairs):
            o_orb = (col - npairs) // nvirt
            v_orb2 = (col - npairs) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                CI[row, col] = (1 / np.sqrt(2)) * (1.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO1] + rep_tens[o_orb, SOMO1, v_orb1, v_orb1] - 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO1])
            else:    
                CI[row, col] = (1 / np.sqrt(2)) * rep_tens[o_orb, SOMO1, v_orb2, v_orb1]
        # <SV0|H|HL3>
        for col in range(2*npairs, 3*npairs):
            o_orb = (col - 2*npairs) // nvirt
            v_orb2 = (col - 2*npairs) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                CI[row, col] =  rep_tens[o_orb, SOMO1, v_orb1, v_orb1] - 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO1] - 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO1]
            else:    
                CI[row, col] = rep_tens[o_orb, SOMO1, v_orb2, v_orb1]
        # <SV0|H|ZHL1>
        for col in range(3*npairs, 4*npairs):
            o_orb = (col - 3*npairs) // nvirt
            v_orb2 = (col - 3*npairs) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                CI[row, col] = rep_tens[o_orb, SOMO1, SOMO1, SOMO2]
        # <SV0|H|ZHL2>
        for col in range(4*npairs, 5*npairs):
            o_orb = (col - 4*npairs) // nvirt
            v_orb2 = (col - 4*npairs) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                CI[row, col] = rep_tens[o_orb, SOMO2, v_orb1, v_orb1] + 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO2] + 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO2] - rep_tens[o_orb, SOMO2, SOMO1, SOMO1]
            else:
                CI[row, col] = rep_tens[o_orb, SOMO2, v_orb1, v_orb2]
                
    row_index = 2 * ndocc + nvirt + 1
    for row in range(row_index, row_index + nvirt):
        v_orb1 = row - row_index + (SOMO2 + 1)
        # <SV0'|H|HL1>
        for col in range(0, npairs):
            o_orb = col // nvirt
            v_orb2 = col % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                CI[row, col] = (1 / np.sqrt(2)) * (rep_tens[o_orb, SOMO2, v_orb1, v_orb1] - 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO2] - 2 * rep_tens[o_orb, v_orb1, v_orb1, SOMO2] - 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO2])
            else:    
                CI[row, col] = (1 / np.sqrt(2)) * (rep_tens[o_orb, SOMO2, v_orb2, v_orb1] - 2 * rep_tens[SOMO2, v_orb1, v_orb2, o_orb])
        # <SV0'|H|HL2>
        for col in range(npairs, 2*npairs):
            o_orb = (col - npairs) // nvirt
            v_orb2 = (col - npairs) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                CI[row, col] = (1 / np.sqrt(2)) * (rep_tens[o_orb, SOMO2, v_orb1, v_orb1] + 1.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO2] - 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO2])
            else:    
                CI[row, col] = (1 / np.sqrt(2)) * rep_tens[o_orb, SOMO2, v_orb2, v_orb1]
        # <SV0'|H|HL3>
        for col in range(2*npairs, 3*npairs):
            o_orb = (col - 2*npairs) // nvirt
            v_orb2 = (col - 2*npairs) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                CI[row, col] = 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO2] + 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO2] - rep_tens[o_orb, SOMO2, v_orb1, v_orb1]
            else:    
                CI[row, col] = - rep_tens[o_orb, SOMO2, v_orb2, v_orb1]
        # <SV0'|H|ZHL1>
        for col in range(3*npairs, 4*npairs):
            o_orb = (col - 3*npairs) // nvirt
            v_orb2 = (col - 3*npairs) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                CI[row, col] = rep_tens[o_orb, SOMO1, v_orb1, v_orb1] + 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO1] + 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO1] - rep_tens[o_orb, SOMO1, SOMO2, SOMO2]
            else:
                CI[row, col] = rep_tens[o_orb, SOMO1, v_orb1, v_orb2]
        # <SV0'|H|ZHL2>
        for col in range(4*npairs, 5*npairs):
            o_orb = (col - 4*npairs) // nvirt
            v_orb2 = (col - 4*npairs) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                CI[row, col] = rep_tens[o_orb, SOMO2, SOMO2, SOMO1]
    
    
    row_index = 2 * ndocc + 2 * nvirt + 1
    for row in range(row_index, row_index + npairs):
        o_orb1 = (row - row_index) // nvirt
        v_orb1 = (row - row_index) % nvirt + (SOMO2 + 1)
        # <HL1|H|HL1>
        for col in range(row - row_index, npairs):
            o_orb2 = col // nvirt
            v_orb2 = col % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                CI[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb1] - 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) \
                    - 0.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1] + 2 * rep_tens[o_orb1, v_orb1, v_orb1, o_orb1]
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                CI[row, col] =  2 * rep_tens[o_orb1, v_orb1, v_orb1, o_orb2] - rep_tens[o_orb1, o_orb2, v_orb1, v_orb1]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                CI[row, col] = 2 * rep_tens[v_orb1, o_orb1, o_orb1, v_orb2] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb2]
            else:
                CI[row, col] = 2 * rep_tens[o_orb1, v_orb1, o_orb2, v_orb2] - rep_tens[o_orb1, o_orb2, v_orb1, v_orb2]
        # <HL1|H|HL2>
        for col in range(npairs, 2*npairs):
            o_orb2 = (col - npairs) // nvirt
            v_orb2 = (col - npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                CI[row, col] = 0.5 * (rep_tens[o_orb1,SOMO2,SOMO2,o_orb1] - rep_tens[o_orb1,SOMO1,SOMO1,o_orb1] + rep_tens[v_orb1,SOMO1,SOMO1,v_orb1] - rep_tens[v_orb1,SOMO2,SOMO2,v_orb1])
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                CI[row, col] =  0.5 * (rep_tens[o_orb1, SOMO2, SOMO2, o_orb2] - rep_tens[o_orb1, SOMO1, SOMO1, o_orb2])
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                CI[row, col] =  0.5 * (rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] - rep_tens[v_orb1, SOMO2, SOMO2, v_orb2])
        # <HL1|H|HL3>
        for col in range(2*npairs, 3*npairs):
            o_orb2 = (col - 2*npairs) // nvirt
            v_orb2 = (col - 2*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                CI[row, col] = (1 / np.sqrt(2)) * (rep_tens[v_orb1,SOMO1,SOMO1,v_orb1] + rep_tens[v_orb1,SOMO2,SOMO2,v_orb1] - rep_tens[o_orb1,SOMO2,SOMO2,o_orb1] - rep_tens[o_orb1,SOMO1,SOMO1,o_orb1])
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                CI[row, col] =  (1 / np.sqrt(2)) * (- rep_tens[o_orb1, SOMO1, SOMO1, o_orb2] - rep_tens[o_orb1, SOMO2, SOMO2, o_orb2])
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                CI[row, col] =  (1 / np.sqrt(2)) * (rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] + rep_tens[v_orb1, SOMO2, SOMO2, v_orb2])
        #<HL1|H|ZHL1>
        for col in range(3*npairs, 4*npairs):
            o_orb2 = (col - 3*npairs) // nvirt
            v_orb2 = (col - 3*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                CI[row, col] = (1 / np.sqrt(2)) * (rep_tens[o_orb1, SOMO1, SOMO2, o_orb1] - rep_tens[v_orb1, SOMO1, SOMO2, v_orb1])
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:
                CI[row, col] = (1 / np.sqrt(2)) * rep_tens[o_orb1, SOMO2, SOMO1, o_orb2]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                CI[row, col] = - (1 / np.sqrt(2)) *  rep_tens[v_orb1, SOMO1, SOMO2, v_orb2]
        #<HL1|H|ZHL2>
        for col in range(4*npairs, 5*npairs):
            o_orb2 = (col - 4*npairs) // nvirt
            v_orb2 = (col - 4*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                CI[row, col] = (1 / np.sqrt(2)) * (rep_tens[v_orb1, SOMO1, SOMO2, v_orb1] - rep_tens[o_orb1, SOMO1, SOMO2, o_orb1])
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:
                CI[row, col] = - (1 / np.sqrt(2)) * rep_tens[o_orb1, SOMO1, SOMO2, o_orb2]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                CI[row, col] = (1 / np.sqrt(2)) *  rep_tens[v_orb1, SOMO2, SOMO1, v_orb2]
    
    row_index = npairs + 2 * ndocc + 2 * nvirt + 1
    for row in range(row_index, row_index + npairs):
        o_orb1 = (row - row_index) // nvirt
        v_orb1 = (row - row_index) % nvirt + (SOMO2 + 1)
        # <HL2|H|HL2>
        for col in range(row - row_index + npairs, 2*npairs):
            o_orb2 = (col - npairs) // nvirt
            v_orb2 = (col - npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                CI[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb1] - 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) + 1.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1]
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                CI[row, col] = - rep_tens[o_orb1, o_orb2, v_orb1, v_orb1]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                CI[row, col] = - rep_tens[v_orb1, v_orb2, o_orb1, o_orb1]
            else:
                CI[row, col] = - rep_tens[o_orb1, o_orb2, v_orb1, v_orb2]
        # <HL2|H|HL3>
        for col in range(2*npairs, 3*npairs):
            o_orb2 = (col - 2*npairs) // nvirt
            v_orb2 = (col - 2*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                CI[row, col] = (1 / np.sqrt(2)) * (rep_tens[v_orb1,SOMO1,SOMO1,v_orb1] - rep_tens[v_orb1,SOMO2,SOMO2,v_orb1] - rep_tens[o_orb1,SOMO2,SOMO2,o_orb1] + rep_tens[o_orb1,SOMO1,SOMO1,o_orb1])
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                CI[row, col] =  (1 / np.sqrt(2)) * (rep_tens[o_orb1, SOMO1, SOMO1, o_orb2] - rep_tens[o_orb1, SOMO2, SOMO2, o_orb2])
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                CI[row, col] =  (1 / np.sqrt(2)) * (rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] - rep_tens[v_orb1, SOMO2, SOMO2, v_orb2])
        #<HL2|H|ZHL1>
        for col in range(3*npairs, 4*npairs):
            o_orb2 = (col - 3*npairs) // nvirt
            v_orb2 = (col - 3*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                CI[row, col] = np.sqrt(2) * (rep_tens[SOMO1, SOMO2, v_orb1, v_orb1] - rep_tens[SOMO1, SOMO2, o_orb1, o_orb1] + 0.5 * rep_tens[o_orb1, SOMO1, SOMO2, o_orb1] - 0.5 * rep_tens[v_orb1, SOMO1, SOMO2, v_orb1] \
                               + 0.5 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO2] - 0.5 * rep_tens[SOMO1, SOMO2, SOMO2, SOMO2])
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:
                CI[row, col] = np.sqrt(2) * (0.5 * rep_tens[o_orb1, SOMO2, SOMO1, o_orb2] - rep_tens[o_orb1, o_orb2, SOMO1, SOMO2])
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                CI[row, col] = np.sqrt(2) * (rep_tens[SOMO1, SOMO2, v_orb1, v_orb2] - 0.5 * rep_tens[v_orb1, SOMO1, SOMO2, v_orb2])
        #<HL2|H|ZHL2>
        for col in range(4*npairs, 5*npairs):
            o_orb2 = (col - 4*npairs) // nvirt
            v_orb2 = (col - 4*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                CI[row, col] = np.sqrt(2) * (rep_tens[SOMO1, SOMO2, v_orb1, v_orb1] - rep_tens[SOMO1, SOMO2, o_orb1, o_orb1] + 0.5 * rep_tens[o_orb1, SOMO1, SOMO2, o_orb1] - 0.5 * rep_tens[v_orb1, SOMO1, SOMO2, v_orb1] \
                                + 0.5 * rep_tens[SOMO1, SOMO2, SOMO2, SOMO2] - 0.5 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO2])
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:
                CI[row, col] = np.sqrt(2) * (0.5 * rep_tens[o_orb1, SOMO1, SOMO2, o_orb2] - rep_tens[o_orb1, o_orb2, SOMO1, SOMO2])
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                CI[row, col] = np.sqrt(2) * (rep_tens[SOMO1, SOMO2, v_orb1, v_orb2] - 0.5 * rep_tens[v_orb1, SOMO2, SOMO1, v_orb2])
                
    row_index = 2 * npairs + 2 * ndocc + 2 * nvirt + 1
    for row in range(row_index, row_index + npairs):
        o_orb1 = (row - row_index) // nvirt
        v_orb1 = (row - row_index) % nvirt + (SOMO2 + 1)
        # <HL3|H|HL3>
        for col in range(row - row_index + 2*npairs, 3*npairs):
            o_orb2 = (col - 2*npairs) // nvirt
            v_orb2 = (col - 2*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                CI[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb1] - 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) - 0.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1] \
                                    + 0.5 * (rep_tens[o_orb1, SOMO1, SOMO1, o_orb1] + rep_tens[o_orb1, SOMO2, SOMO2, o_orb2] + rep_tens[v_orb1, SOMO1, SOMO1, v_orb1] + rep_tens[v_orb1, SOMO2, SOMO2, v_orb1])
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                CI[row, col] = 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb2] + 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb2] - rep_tens[o_orb1, o_orb2, v_orb1, v_orb1]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                CI[row, col] = 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] + 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb2] - rep_tens[v_orb1, v_orb2, o_orb1, o_orb1]
            else:
                CI[row, col] = - rep_tens[o_orb1, o_orb2, v_orb1, v_orb2]
        #<HL3|H|ZHL1>
        for col in range(3*npairs, 4*npairs):
            o_orb2 = (col - 3*npairs) // nvirt
            v_orb2 = (col - 3*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                CI[row, col] = - rep_tens[v_orb1, SOMO1, SOMO2, v_orb1] - rep_tens[SOMO1, o_orb1, o_orb1, SOMO2]
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:
                CI[row, col] = - rep_tens[o_orb1, SOMO2, SOMO1, o_orb2]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                CI[row, col] = - rep_tens[v_orb1, SOMO1, SOMO2, v_orb2]
        #<HL3|H|ZHL2>
        for col in range(4*npairs, 5*npairs):
            o_orb2 = (col - 4*npairs) // nvirt
            v_orb2 = (col - 4*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                CI[row, col] = rep_tens[v_orb1, SOMO1, SOMO2, v_orb1] + rep_tens[SOMO1, o_orb1, o_orb1, SOMO2]
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:
                CI[row, col] = rep_tens[o_orb1, SOMO1, SOMO2, o_orb2]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                CI[row, col] = rep_tens[v_orb1, SOMO2, SOMO1, v_orb2]

    row_index = 3 * npairs + 2 * ndocc + 2 * nvirt + 1
    for row in range(row_index, row_index + npairs):
        o_orb1 = (row - row_index) // nvirt
        v_orb1 = (row - row_index) % nvirt + (SOMO2 + 1)
        # <ZHL1|H|ZHL1>
        for col in range(row - row_index + 3*npairs, 4*npairs):
            o_orb2 = (col - 3*npairs) // nvirt
            v_orb2 = (col - 3*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                CI[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[o_orb1] + orb_energies[SOMO1] - orb_energies[SOMO2] + rep_tens[v_orb1, v_orb1, SOMO1, SOMO1] + rep_tens[o_orb1, o_orb1, SOMO2, SOMO2] - rep_tens[o_orb1, o_orb1, SOMO1, SOMO1] - rep_tens[v_orb1, v_orb1, SOMO2, SOMO2] \
                                + 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb1] + 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb1] - 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb1] - 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb1] - rep_tens[SOMO1, SOMO1, SOMO2, SOMO2] + 0.5 * rep_tens[SOMO1, SOMO2, SOMO2, SOMO1] \
                                + 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) - rep_tens[o_orb1, o_orb1, v_orb1, v_orb1]
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                CI[row, col] = rep_tens[o_orb1, o_orb2, SOMO2, SOMO2] + 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb2] - rep_tens[o_orb1, o_orb2, SOMO1, SOMO1] - rep_tens[o_orb1, o_orb2, v_orb1, v_orb1] - 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb2]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                CI[row, col] = rep_tens[v_orb1, v_orb2, SOMO1, SOMO1] + 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb2] - rep_tens[v_orb1, v_orb2, SOMO2, SOMO2] - rep_tens[v_orb1, v_orb2, o_orb1, o_orb1] - 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb2]
            else:
                CI[row, col] = - rep_tens[o_orb1, o_orb2, v_orb1, v_orb2]
        #<ZHL1|H|ZHL2>
        for col in range(4*npairs, 5*npairs):
            o_orb2 = (col - 4*npairs) // nvirt
            v_orb2 = (col - 4*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                CI[row, col] = rep_tens[SOMO1, SOMO2, SOMO2, SOMO1]

    row_index = 4 * npairs + 2 * ndocc + 2 * nvirt + 1
    for row in range(row_index, row_index + npairs):
        o_orb1 = (row - row_index) // nvirt
        v_orb1 = (row - row_index) % nvirt + (SOMO2 + 1)
        # <ZHL2|H|ZHL2>
        for col in range(row - row_index + 4*npairs, 5*npairs):
            o_orb2 = (col - 4*npairs) // nvirt
            v_orb2 = (col - 4*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                CI[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[o_orb1] + orb_energies[SOMO2] - orb_energies[SOMO1] + rep_tens[v_orb1, v_orb1, SOMO2, SOMO2] + rep_tens[o_orb1, o_orb1, SOMO1, SOMO1] - rep_tens[o_orb1, o_orb1, SOMO2, SOMO2] - rep_tens[v_orb1, v_orb1, SOMO1, SOMO1] \
                                + 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb1] + 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb1] - 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb1] - 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb1] - rep_tens[SOMO1, SOMO1, SOMO2, SOMO2] + 0.5 * rep_tens[SOMO1, SOMO2, SOMO2, SOMO1] \
                                + 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) - rep_tens[o_orb1, o_orb1, v_orb1, v_orb1]
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                CI[row, col] = rep_tens[o_orb1, o_orb2, SOMO1, SOMO1] + 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb2] - rep_tens[o_orb1, o_orb2, SOMO2, SOMO2] - rep_tens[o_orb1, o_orb2, v_orb1, v_orb1] \
                                - 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb2]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                CI[row, col] = rep_tens[v_orb1, v_orb2, SOMO2, SOMO2] + 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] - rep_tens[v_orb1, v_orb2, SOMO1, SOMO1] - rep_tens[v_orb1, v_orb2, o_orb1, o_orb1] \
                                - 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb2]
            else:
                CI[row, col] = - rep_tens[o_orb1, o_orb2, v_orb1, v_orb2]
    
    return CI

def build_triplet_D_block(ndocc, norbs, energy0, orb_energies, rep_tens):
    '''
    Function to build the upper diagonal of the CI matrix for doubly excited states (HOMO to SOMO & SOMO to LUMO) for a diradical system.
    Args:
        ndocc (int): Number of doubly occupied orbitals
        norbs (int): Total number of orbitals
        energy0 (float): Base energy of the mean-field reference state
        orb_energies (numpy.ndarray): Orbital energies for the system
        rep_tens (numpy.ndarray): Representation tensor for the system
    Returns:
        numpy.ndarray: CI matrix for the diradical system
    '''
    # Calculate the size of the CI matrix based on the number of doubly occupied orbitals
    SOMO1 = ndocc # Index of SOMO1
    SOMO2 = ndocc + 1 # Index of SOMO2
    nvirt = norbs - ndocc - 2 # Number of virtual orbitals
    npairs = ndocc * nvirt
    ndcs = int((ndocc ** 2 - ndocc) / 2) # Number of doubly excited core to SOMO singlet CSFs
    ndsv = int((nvirt ** 2 - nvirt) / 2) # Number of doubly excited SOMO to virtual singlet CSFs
    
    row_dim = ndcs + ndsv + 5 * (npairs) + 2 * ndocc + 2 * nvirt + 1
    col_dim = ndcs + ndsv
    CI = np.zeros((row_dim, col_dim))  # Initialize CI Block

    # <OS3|H|CSD>
    o_orb1 = 0
    o_orb2 = 1
    for col in range(0, ndcs):
        CI[0,col] = rep_tens[o_orb1, SOMO1, SOMO2, o_orb2] - rep_tens[o_orb1, SOMO2, SOMO1, o_orb2]
        o_orb2 += 1
        if o_orb2 >= ndocc:
            o_orb1 += 1
            o_orb2 = o_orb1 + 1
        
    # <OS1|H|SVD>
    v_orb1 = SOMO2 + 1
    v_orb2 = SOMO2 + 2
    for col in range(ndcs, ndcs + ndsv):
        CI[0,col] = rep_tens[v_orb1, SOMO1, SOMO2, v_orb2] - rep_tens[v_orb1, SOMO2, SOMO1, v_orb2]
        v_orb2 += 1
        if v_orb2 >= norbs:
            v_orb1 += 1
            v_orb2 = v_orb1 + 1

    
    row_index = 1
    for row in range(row_index, row_index + ndocc):
        o_orb1 = row - row_index
        o_orb2 = 0
        o_orb3 = 1
        # <CS0|H|CSD>
        for col in range(0, ndcs):
            if o_orb1 == o_orb2:
                CI[row, col] = rep_tens[o_orb3, o_orb1, o_orb1, SOMO2] + rep_tens[o_orb3, SOMO1, SOMO1, SOMO2] - rep_tens[o_orb3, SOMO2, o_orb1, o_orb1] + 0.5 * rep_tens[o_orb3, SOMO2, SOMO2, SOMO2] - 0.5 * rep_tens[o_orb3, SOMO1, SOMO1, SOMO2]
            elif o_orb1 == o_orb3:
                CI[row, col] = - (rep_tens[o_orb2, o_orb1, o_orb1, SOMO2] + rep_tens[o_orb2, SOMO1, SOMO1, SOMO2] - rep_tens[o_orb2, SOMO2, o_orb1, o_orb1] + 0.5 * rep_tens[o_orb2, SOMO2, SOMO2, SOMO2] - 0.5 * rep_tens[o_orb2, SOMO1, SOMO1, SOMO2])
            else:
                CI[row, col] = rep_tens[o_orb2, SOMO2, o_orb1, o_orb3] - rep_tens[o_orb3, SOMO2, o_orb1, o_orb2]
            o_orb3 += 1
            if o_orb3 >= ndocc:
                o_orb2 += 1
                o_orb3 = o_orb2 + 1
        # <CS0|H|SVD> = 0
    
    row_index = ndocc + 1
    for row in range(row_index, row_index + ndocc):
        o_orb1 = row - row_index
        o_orb2 = 0
        o_orb3 = 1
        # <CS0'|H|HL1>
        for col in range(0, ndcs):
            if o_orb1 == o_orb2:
                CI[row, col] = - (rep_tens[o_orb3, o_orb1, o_orb1, SOMO1] + rep_tens[o_orb3, SOMO2, SOMO2, SOMO1] - rep_tens[o_orb3, SOMO1, o_orb1, o_orb1] + 0.5 * rep_tens[o_orb3, SOMO1, SOMO1, SOMO1] - 0.5 * rep_tens[o_orb3, SOMO2, SOMO2, SOMO1])
            elif o_orb1 == o_orb3:
                CI[row, col] = rep_tens[o_orb2, o_orb1, o_orb1, SOMO1] + rep_tens[o_orb2, SOMO2, SOMO2, SOMO1] - rep_tens[o_orb2, SOMO1, o_orb1, o_orb1] + 0.5 * rep_tens[o_orb2, SOMO1, SOMO1, SOMO1] - 0.5 * rep_tens[o_orb2, SOMO2, SOMO2, SOMO1]
            else:
                CI[row, col] = rep_tens[o_orb3, SOMO1, o_orb1, o_orb2] - rep_tens[o_orb2, SOMO1, o_orb1, o_orb3]
            o_orb3 += 1
            if o_orb3 >= ndocc:
                o_orb2 += 1
                o_orb3 = o_orb2 + 1
        # <CS0'|H|SVD> = 0
    
    row_index = 2 * ndocc + 1
    for row in range(row_index, row_index + nvirt):
        # <SV0|H|HSD> = 0
        v_orb1 = row - row_index + (SOMO2 + 1)
        v_orb2 = SOMO2 + 1
        v_orb3 = SOMO2 + 2
        # <SV0|H|SVD>
        for col in range(ndcs, ndcs + ndsv):
            if v_orb1 == v_orb2:
                CI[row, col] = - (rep_tens[v_orb3, SOMO2, SOMO1, SOMO1] - rep_tens[v_orb3, SOMO2, v_orb1, v_orb1] + rep_tens[SOMO2, v_orb1, v_orb1, v_orb3] + 0.5 * rep_tens[v_orb3, SOMO2, SOMO2, SOMO2] - 0.5 * rep_tens[v_orb3, SOMO1, SOMO1, SOMO2])
            elif v_orb1 == v_orb3:
                CI[row, col] = - (rep_tens[v_orb2, SOMO2, SOMO1, SOMO1] - rep_tens[v_orb2, SOMO2, v_orb1, v_orb1] + rep_tens[SOMO2, v_orb1, v_orb1, v_orb2] + 0.5 * rep_tens[v_orb2, SOMO2, SOMO2, SOMO2] - 0.5 * rep_tens[v_orb2, SOMO1, SOMO1, SOMO2])
            else:
                CI[row, col] = rep_tens[v_orb3, SOMO2, v_orb1, v_orb2] - rep_tens[v_orb2, SOMO2, v_orb1, v_orb3]
            v_orb3 += 1
            if v_orb3 >= norbs:
                v_orb2 += 1
                v_orb3 = v_orb2 + 1

                
    row_index = 2 * ndocc + nvirt + 1
    for row in range(row_index, row_index + nvirt):
        # <SV0'|H|HSD> = 0
        v_orb1 = row - row_index + (SOMO2 + 1)
        v_orb2 = SOMO2 + 1
        v_orb3 = SOMO2 + 2
        # <SV0'|H|SVD>
        for col in range(ndcs, ndcs + ndsv):
            if v_orb1 == v_orb2:
                CI[row, col] = rep_tens[v_orb3, SOMO1, SOMO2, SOMO2] - rep_tens[v_orb3, SOMO1, v_orb1, v_orb1] + rep_tens[SOMO1, v_orb1, v_orb1, v_orb3] + 0.5 * rep_tens[v_orb3, SOMO1, SOMO1, SOMO1] - 0.5 * rep_tens[v_orb3, SOMO2, SOMO2, SOMO1]
            elif v_orb1 == v_orb3:
                CI[row, col] = rep_tens[v_orb2, SOMO1, SOMO2, SOMO2] - rep_tens[v_orb2, SOMO1, v_orb1, v_orb1] + rep_tens[SOMO1, v_orb1, v_orb1, v_orb2] + 0.5 * rep_tens[v_orb2, SOMO1, SOMO1, SOMO1] - 0.5 * rep_tens[v_orb2, SOMO2, SOMO2, SOMO1]
            else:
                CI[row, col] = rep_tens[v_orb2, SOMO1, v_orb1, v_orb3] - rep_tens[v_orb3, SOMO1, v_orb1, v_orb2]
            v_orb3 += 1
            if v_orb3 >= norbs:
                v_orb2 += 1
                v_orb3 = v_orb2 + 1
    
    row_index = 2 * ndocc + 2 * nvirt + 1
    for row in range(row_index, row_index + npairs):
        o_orb1 = (row - row_index) // nvirt
        o_orb2 = 0
        o_orb3 = 1
        v_orb1 = (row - row_index) % nvirt + (SOMO2 + 1)
        v_orb2 = SOMO2 + 1
        v_orb3 = SOMO2 + 2
        # <HL1|H|HSD>
        for col in range(0, ndcs):
            if o_orb1 == o_orb2:
                CI[row, col] = (np.sqrt(2) / 2) * (rep_tens[o_orb3, SOMO2, SOMO1, v_orb1] - rep_tens[o_orb3, SOMO1, SOMO2, v_orb1])
            elif o_orb1 == o_orb3:
                CI[row, col] = - (np.sqrt(2) / 2) * (rep_tens[o_orb2, SOMO2, SOMO1, v_orb1] - rep_tens[o_orb2, SOMO1, SOMO2, v_orb1])
            o_orb3 += 1
            if o_orb3 >= ndocc:
                o_orb2 += 1
                o_orb3 = o_orb2 + 1
        # <HL1|H|SVD>
        for col in range(ndcs, ndcs + ndsv):
            if v_orb1 == v_orb2:
                CI[row, col] = (np.sqrt(2) / 2) * (rep_tens[o_orb1, SOMO2, SOMO1, v_orb3] - rep_tens[o_orb1, SOMO1, SOMO2, v_orb3])
            elif v_orb1 == v_orb3:
                CI[row, col] = (np.sqrt(2) / 2) * (rep_tens[o_orb1, SOMO1, SOMO2, v_orb2] - rep_tens[o_orb1, SOMO2, SOMO1, v_orb2])
            v_orb3 += 1
            if v_orb3 >= norbs:
                v_orb2 += 1
                v_orb3 = v_orb2 + 1
    
    
    row_index = npairs + 2 * ndocc + 2 * nvirt + 1
    for row in range(row_index, row_index + npairs):
        o_orb1 = (row - row_index) // nvirt
        o_orb2 = 0
        o_orb3 = 1
        v_orb1 = (row - row_index) % nvirt + (SOMO2 + 1)
        v_orb2 = SOMO2 + 1
        v_orb3 = SOMO2 + 2
        # <HL3|H|HSD>
        for col in range(0, ndcs):
            if o_orb1 == o_orb2:
                CI[row, col] = - (np.sqrt(2) / 2) * (rep_tens[o_orb3, SOMO2, SOMO1, v_orb1] + rep_tens[o_orb3, SOMO1, SOMO2, v_orb1])
            elif o_orb1 == o_orb3:
                CI[row, col] = (np.sqrt(2) / 2) * (rep_tens[o_orb2, SOMO2, SOMO1, v_orb1] + rep_tens[o_orb2, SOMO1, SOMO2, v_orb1])
            o_orb3 += 1
            if o_orb3 >= ndocc:
                o_orb2 += 1
                o_orb3 = o_orb2 + 1
        # <HL3|H|SLD>
        for col in range(ndcs, ndcs + ndsv):
            if v_orb1 == v_orb2:
                CI[row, col] = (np.sqrt(2) / 2) * (rep_tens[o_orb1, SOMO2, SOMO1, v_orb3] + rep_tens[o_orb1, SOMO1, SOMO2, v_orb3])
            elif v_orb1 == v_orb3:
                CI[row, col] = - (np.sqrt(2) / 2) * (rep_tens[o_orb1, SOMO2, SOMO1, v_orb2] + rep_tens[o_orb1, SOMO1, SOMO2, v_orb2])
            v_orb3 += 1
            if v_orb3 >= norbs:
                v_orb2 += 1
                v_orb3 = v_orb2 + 1
    
    
    row_index = 2 * npairs + 2 * ndocc + 2 * nvirt + 1
    for row in range(row_index, row_index + npairs):
        o_orb1 = (row - row_index) // nvirt
        o_orb2 = 0
        o_orb3 = 1
        v_orb1 = (row - row_index) % nvirt + (SOMO2 + 1)
        v_orb2 = SOMO2 + 1
        v_orb3 = SOMO2 + 2
        # <HL3|H|HSD>
        for col in range(0, ndcs):
            if o_orb1 == o_orb2:
                CI[row, col] = (np.sqrt(2) / 2) * (rep_tens[o_orb3, SOMO2, SOMO1, v_orb1] - rep_tens[o_orb3, SOMO1, SOMO2, v_orb1])
            elif o_orb1 == o_orb3:
                CI[row, col] = - (np.sqrt(2) / 2) * (rep_tens[o_orb2, SOMO2, SOMO1, v_orb1] - rep_tens[o_orb2, SOMO1, SOMO2, v_orb1])
            o_orb3 += 1
            if o_orb3 >= ndocc:
                o_orb2 += 1
                o_orb3 = o_orb2 + 1
        # <HL3|H|SLD>
        for col in range(ndcs, ndcs + ndsv):
            if v_orb1 == v_orb2:
                CI[row, col] = rep_tens[o_orb1, SOMO1, SOMO2, v_orb3] - rep_tens[o_orb1, SOMO2, SOMO1, v_orb3]
            elif v_orb1 == v_orb3:
                CI[row, col] = rep_tens[o_orb1, SOMO2, SOMO1, v_orb2] - rep_tens[o_orb1, SOMO1, SOMO2, v_orb2]
            v_orb3 += 1
            if v_orb3 >= norbs:
                v_orb2 += 1
                v_orb3 = v_orb2 + 1
    


    row_index = 3 * npairs + 2 * ndocc + 2 * nvirt + 1
    for row in range(row_index, row_index + npairs):
        o_orb1 = (row - row_index) // nvirt
        o_orb2 = 0
        o_orb3 = 1
        v_orb1 = (row - row_index) % nvirt + (SOMO2 + 1)
        v_orb2 = SOMO2 + 1
        v_orb3 = SOMO2 + 2
        # <ZHL1|H|CSD>
        for col in range(0, ndcs):
            if o_orb1 == o_orb2:
                CI[row, col] = rep_tens[o_orb3, SOMO2, SOMO2, v_orb1]
            elif o_orb1 == o_orb3:
                CI[row, col] = rep_tens[o_orb2, SOMO2, SOMO2, v_orb1]
            o_orb3 += 1
            if o_orb3 >= ndocc:
                o_orb2 += 1
                o_orb3 = o_orb2 + 1
        # <ZHL1|H|SVD>
        for col in range(ndcs, ndcs + ndsv):
            if v_orb1 == v_orb2:
                CI[row, col] = rep_tens[o_orb1, SOMO1, SOMO1, v_orb3]
            elif v_orb1 == v_orb3:
                CI[row, col] = rep_tens[o_orb1, SOMO1, SOMO1, v_orb2]
            v_orb3 += 1
            if v_orb3 >= norbs:
                v_orb2 += 1
                v_orb3 = v_orb2 + 1

    row_index = 4 * npairs + 2 * ndocc + 2 * nvirt + 1
    for row in range(row_index, row_index + npairs):
        o_orb1 = (row - row_index) // nvirt
        o_orb2 = 0
        o_orb3 = 1
        v_orb1 = (row - row_index) % nvirt + (SOMO2 + 1)
        v_orb2 = SOMO2 + 1
        v_orb3 = SOMO2 + 2
        # <ZHL2|H|CSD>
        for col in range(0, ndcs):
            if o_orb1 == o_orb2:
                CI[row, col] = rep_tens[o_orb3, SOMO1, SOMO1, v_orb1]
            elif o_orb1 == o_orb3:
                CI[row, col] = rep_tens[o_orb2, SOMO1, SOMO1, v_orb1]
            o_orb3 += 1
            if o_orb3 >= ndocc:
                o_orb2 += 1
                o_orb3 = o_orb2 + 1
        # <ZHL2|H|SVD>
        for col in range(ndcs, ndcs + ndsv):
            if v_orb1 == v_orb2:
                CI[row, col] = rep_tens[o_orb1, SOMO2, SOMO2, v_orb3]
            elif v_orb1 == v_orb3:
                CI[row, col] = rep_tens[o_orb1, SOMO2, SOMO2, v_orb2]
            v_orb3 += 1
            if v_orb3 >= norbs:
                v_orb2 += 1
                v_orb3 = v_orb2 + 1
    
    
    row_index = 5 * npairs + 2 * ndocc + 2 * nvirt + 1
    o_orb1 = 0
    o_orb2 = 1
    for row in range(row_index, row_index + ndcs):
        o_orb3 = o_orb1
        o_orb4 = o_orb2
        # <HSD|H|HSD>
        for col in range(row - row_index, ndcs):
            if o_orb1 == o_orb3 and o_orb2 == o_orb4:
                CI[row, col] = energy0 - orb_energies[o_orb1] - orb_energies[o_orb2] + orb_energies[SOMO1] + orb_energies[SOMO2] + rep_tens[o_orb1, o_orb1, o_orb2, o_orb2] - rep_tens[o_orb1, o_orb1, SOMO1, SOMO1] - rep_tens[o_orb1, o_orb1, SOMO2, SOMO2] \
                    - rep_tens[o_orb2, o_orb2, SOMO1, SOMO1] - rep_tens[o_orb2, o_orb2, SOMO2, SOMO2] + 0.5 * (rep_tens[o_orb1, SOMO1, SOMO1, o_orb1] + rep_tens[o_orb1, SOMO2, SOMO2, o_orb1] + rep_tens[o_orb2, SOMO1, SOMO1, o_orb2] + rep_tens[o_orb2, SOMO2, SOMO2, o_orb2]) \
                    + 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) + rep_tens[SOMO1,SOMO1,SOMO2,SOMO2] - 0.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1] - rep_tens[o_orb1, o_orb2, o_orb2, o_orb1]
            elif o_orb1 == o_orb3 and o_orb2 != o_orb4:
                CI[row,col] = rep_tens[o_orb2, o_orb1, o_orb1, o_orb4] - rep_tens[o_orb2, o_orb4, o_orb1, o_orb1] + rep_tens[o_orb2, o_orb4, SOMO1, SOMO1] + rep_tens[o_orb2, o_orb4, SOMO2, SOMO2] - 0.5 * rep_tens[o_orb2, SOMO1, SOMO1, o_orb4] - 0.5 * rep_tens[o_orb2, SOMO2, SOMO2, o_orb4]
            elif o_orb2 == o_orb4 and o_orb1 != o_orb3:
                CI[row,col] = -(rep_tens[o_orb1, o_orb2, o_orb2, o_orb3] - rep_tens[o_orb1, o_orb3, o_orb2, o_orb2] + rep_tens[o_orb1, o_orb3, SOMO1, SOMO1] + rep_tens[o_orb1, o_orb3, SOMO2, SOMO2] - 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb3] - 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb3])
            else:
                CI[row,col] = rep_tens[o_orb1, o_orb3, o_orb2, o_orb4] - rep_tens[o_orb1, o_orb4, o_orb2, o_orb3]
            o_orb4 += 1
            if o_orb4 >= ndocc:
                o_orb3 += 1
                o_orb4 = o_orb3 + 1
        # <HSD|H|SVD> = 0
        o_orb2 += 1
        if o_orb2 >= ndocc:
            o_orb1 += 1
            o_orb2 = o_orb1 + 1
    
    row_index = ndcs + 5 * npairs + 2 * ndocc + 2 * nvirt + 1
    v_orb1 = SOMO2 + 1
    v_orb2 = SOMO2 + 2
    for row in range(row_index, row_index + ndsv):
        v_orb3 = v_orb1
        v_orb4 = v_orb4
        # <SVD|H|SVD>
        for col in range(row - row_index + ndcs, ndcs + ndsv):
            if v_orb1 == v_orb3 and v_orb2 == v_orb4:
                CI[row, col] = energy0 + orb_energies[v_orb1] + orb_energies[v_orb2] - orb_energies[SOMO1] - orb_energies[SOMO2] + rep_tens[v_orb1, v_orb1, v_orb2, v_orb2] - rep_tens[v_orb1, v_orb1, SOMO1, SOMO1] - rep_tens[v_orb1, v_orb1, SOMO2, SOMO2] \
                    - rep_tens[v_orb2, v_orb2, SOMO1, SOMO1] - rep_tens[v_orb2, v_orb2, SOMO2, SOMO2] + 0.5 * (rep_tens[v_orb1, SOMO1, SOMO1, v_orb1] + rep_tens[v_orb1, SOMO2, SOMO2, v_orb1] + rep_tens[v_orb2, SOMO1, SOMO1, v_orb2] + rep_tens[v_orb2, SOMO2, SOMO2, v_orb2]) \
                    + 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) + rep_tens[SOMO1,SOMO1,SOMO2,SOMO2] - 0.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1] - rep_tens[v_orb1, v_orb2, v_orb2, v_orb1]
            elif v_orb1 == v_orb3 and v_orb2 != v_orb4:
                CI[row, col] = rep_tens[v_orb2, v_orb4, v_orb1, v_orb1] - rep_tens[v_orb2, v_orb1, v_orb1, v_orb4] - rep_tens[v_orb2, v_orb4, SOMO1, SOMO1] - rep_tens[v_orb2, v_orb4, SOMO2, SOMO2] + 0.5 * rep_tens[v_orb2, SOMO1, SOMO1, v_orb4] + 0.5 * rep_tens[v_orb2, SOMO2, SOMO2, v_orb4]
            elif v_orb1 != v_orb3 and v_orb2 == v_orb4:
                CI[row, col] = rep_tens[v_orb1, v_orb3, v_orb2, v_orb2] - rep_tens[v_orb1, v_orb2, v_orb2, v_orb3] - rep_tens[v_orb1, v_orb3, SOMO1, SOMO1] - rep_tens[v_orb1, v_orb3, SOMO2, SOMO2] + 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb3] + 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb3]
            else:
                CI[row, col] = rep_tens[v_orb1, v_orb3, v_orb2, v_orb4] - rep_tens[v_orb1, v_orb4, v_orb2, v_orb3]
            v_orb4 += 1
            if v_orb4 >= norbs:
                v_orb3 += 1
                v_orb4 = v_orb3 + 1
        v_orb2 += 1
        if v_orb2 >= norbs:
            v_orb1 += 1
            v_orb2 = v_orb1 + 1
    
    return CI


def build_quintet_block(ndocc, norbs, energy0, orb_energies, rep_tens):
    
    SOMO1 = ndocc # Index of SOMO1
    SOMO2 = ndocc + 1 # Index of SOMO2
    nvirt = norbs - ndocc - 2 # Number of virtual orbitals
    npairs = ndocc * nvirt
    
    CI = np.zeros((npairs, npairs))  # Initialize CI Block

    for row in range(0, npairs):
        o_orb1 = row // nvirt
        v_orb1 = row % nvirt + (SOMO2 + 1)
        # <HL1|H|HL1>
        for col in range(row, npairs):
            o_orb2 = col // nvirt
            v_orb2 = col % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                CI[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb1] - 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) - 0.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1] \
                                 - 0.5 * (rep_tens[o_orb1, SOMO1, SOMO1, o_orb1] + rep_tens[o_orb1, SOMO2, SOMO2, o_orb1] + rep_tens[v_orb1, SOMO1, SOMO1, v_orb1] + rep_tens[v_orb1, SOMO2, SOMO2, v_orb1])
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                CI[row, col] =  - rep_tens[o_orb1, o_orb2, v_orb1, v_orb1] - 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb2] - 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb2]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                CI[row, col] = - rep_tens[v_orb1, v_orb2, o_orb1, o_orb1] - 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] - 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb2]
            else:
                CI[row, col] = - rep_tens[o_orb1, o_orb2, v_orb1, v_orb2]
            CI[col, row] = CI[row, col]
    return CI

def build_singlet_CIMatrix(ndocc, norbs, energy0, orb_energies, rep_tens, ci_level):
    
    if ci_level == 0:
        Singlet_CI = build_singlet_ref_block(ndocc, energy0, orb_energies, rep_tens)
        
    elif ci_level == 1:
        ref_block = build_singlet_ref_block(ndocc, energy0, orb_energies, rep_tens)
        cs_sv_block = build_singlet_CS_SV_block(ndocc, norbs, energy0, orb_energies, rep_tens)
        
        Singlet_CI = np.zeros((cs_sv_block.shape[0], cs_sv_block.shape[0]))
        Singlet_CI[:ref_block.shape[0], :ref_block.shape[1]] = ref_block
        Singlet_CI[:, ref_block.shape[1]:] = cs_sv_block
        
    elif ci_level == 2:
        ref_block = build_singlet_ref_block(ndocc, energy0, orb_energies, rep_tens)
        cs_sv_block = build_singlet_CS_SV_block(ndocc, norbs, energy0, orb_energies, rep_tens)
        hl_block = build_singlet_HL_block(ndocc, norbs, energy0, orb_energies, rep_tens)
        
        Singlet_CI = np.zeros((hl_block.shape[0], hl_block.shape[0]))
        Singlet_CI[:ref_block.shape[0], :ref_block.shape[1]] = ref_block
        Singlet_CI[:cs_sv_block.shape[0], ref_block.shape[1]:(ref_block.shape[1]+cs_sv_block.shape[1])] = cs_sv_block
        Singlet_CI[:, (ref_block.shape[1]+cs_sv_block.shape[1]):] = hl_block
    
    elif ci_level == 3:
        ref_block = build_singlet_ref_block(ndocc, energy0, orb_energies, rep_tens)
        cs_sv_block = build_singlet_CS_SV_block(ndocc, norbs, energy0, orb_energies, rep_tens)
        hl_block = build_singlet_HL_block(ndocc, norbs, energy0, orb_energies, rep_tens)
        d_block = build_singlet_D_block(ndocc, norbs, energy0, orb_energies, rep_tens)
        
        Singlet_CI = np.zeros((d_block.shape[0], d_block.shape[0]))
        Singlet_CI[:ref_block.shape[0], :ref_block.shape[1]] = ref_block
        Singlet_CI[:cs_sv_block.shape[0], ref_block.shape[1]:(ref_block.shape[1]+cs_sv_block.shape[1])] = cs_sv_block
        Singlet_CI[:hl_block.shape[0], (ref_block.shape[1]+cs_sv_block.shape[1]):(ref_block.shape[1]+cs_sv_block.shape[1]+hl_block.shape[1])] = hl_block
        Singlet_CI[:, (ref_block.shape[1]+cs_sv_block.shape[1]+hl_block.shape[1]):] = d_block
    
    Singlet_CI = Singlet_CI + Singlet_CI.T - np.diag(np.diag(Singlet_CI))  # Fill the lower diagonal
    
    return Singlet_CI


def build_triplet_CIMatrix(ndocc, norbs, energy0, orb_energies, rep_tens, ci_level):
    '''
    Function to build the Triplet CI matrix from a defined number of blocks
    '''
    if ci_level == 0:
        Triplet_CI = build_triplet_ref_block(ndocc, energy0, rep_tens)
    
    elif ci_level == 1:
        ref_block = build_triplet_ref_block(ndocc, energy0, rep_tens)
        cs_sv_block = build_triplet_CS_SV_block(ndocc, norbs, energy0, orb_energies, rep_tens)
        
        Triplet_CI = np.zeros((cs_sv_block.shape[0], cs_sv_block.shape[0]))
        Triplet_CI[:ref_block.shape[0], :ref_block.shape[1]] = ref_block
        Triplet_CI[:, ref_block.shape[1]:] = cs_sv_block
        
    elif ci_level == 2:
        ref_block = build_triplet_ref_block(ndocc, energy0, rep_tens)
        cs_sv_block = build_triplet_CS_SV_block(ndocc, norbs, energy0, orb_energies, rep_tens)
        hl_block = build_triplet_HL_block(ndocc, norbs, energy0, orb_energies, rep_tens)
        
        Triplet_CI = np.zeros((hl_block.shape[0], hl_block.shape[0]))
        Triplet_CI[:ref_block.shape[0], :ref_block.shape[1]] = ref_block
        Triplet_CI[:cs_sv_block.shape[0], ref_block.shape[1]:(ref_block.shape[1]+cs_sv_block.shape[1])] = cs_sv_block
        Triplet_CI[:, (ref_block.shape[1]+cs_sv_block.shape[1]):] = hl_block

    elif ci_level == 3:
        ref_block = build_triplet_ref_block(ndocc, energy0, orb_energies, rep_tens)
        cs_sv_block = build_triplet_CS_SV_block(ndocc, norbs, energy0, orb_energies, rep_tens)
        hl_block = build_triplet_HL_block(ndocc, norbs, energy0, orb_energies, rep_tens)
        d_block = build_triplet_D_block(ndocc, norbs, energy0, orb_energies, rep_tens)
        
        Triplet_CI = np.zeros((d_block.shape[0], d_block.shape[0]))
        Triplet_CI[:ref_block.shape[0], :ref_block.shape[1]] = ref_block
        Triplet_CI[:cs_sv_block.shape[0], ref_block.shape[1]:(ref_block.shape[1]+cs_sv_block.shape[1])] = cs_sv_block
        Triplet_CI[:hl_block.shape[0], (ref_block.shape[1]+cs_sv_block.shape[1]):(ref_block.shape[1]+cs_sv_block.shape[1]+hl_block.shape[1])] = hl_block
        Triplet_CI[:, (ref_block.shape[1]+cs_sv_block.shape[1]+hl_block.shape[1]):] = d_block
    
    Triplet_CI = Triplet_CI + Triplet_CI.T - np.diag(np.diag(Triplet_CI))  # Fill the lower diagonal
    
    return Triplet_CI

def get_full_CIMatrix(ndocc, norbs, energy0, orb_energies, rep_tens, ci_level):
    '''
    Function to build the CI matrix from a number of excitation blocks given by ci_level.
    ci_level = 0 -> Only the reference block is included
    ci_level = 1 -> The reference block and the CS/SV block
    ci_level = 2 -> The reference block, the CS/SV block, and the CV block
    ci_level = 3 -> The reference block, the CS/SV block, the CV block, and the double CS/double SV block
    '''
    
    singlet_block = build_singlet_CIMatrix(ndocc, norbs, energy0, orb_energies, rep_tens, ci_level)
    triplet_block = build_triplet_CIMatrix(ndocc, norbs, energy0, orb_energies, rep_tens, ci_level)
    if ci_level < 2:
        return block_diag(singlet_block, triplet_block), singlet_block, triplet_block
    else:
        quintet_block = build_quintet_block(ndocc, norbs, energy0, orb_energies, rep_tens)
        return block_diag(singlet_block, triplet_block, quintet_block), singlet_block, triplet_block, quintet_block
    
    
    
def print_ci_info(out_file, ci_energies, ci_coeffs, ndocc, norbs, tdms, rng, cutoff_energy, ci_level, csf_tol=0.01):
    print("Energy of the lowest CI state:", ci_energies[0])
    osc_array1 = np.zeros_like(ci_energies)
    osc_array3 = np.zeros_like(ci_energies)
    s2_array = np.zeros_like(ci_energies)
    nvirt = norbs - ndocc - 2
    npairs = ndocc * nvirt
    ndoc1 = int((ndocc ** 2 + ndocc) / 2)
    ndcv1 = int((nvirt ** 2 + nvirt) / 2)
    strng3 = ""
    strng1 = ""
    for i in range(rng): # Loop over CIS states
        if ci_energies[i] - ci_energies[0] > cutoff_energy:
            break
        print("\nState %s %04.3f eV " % (i, ci_energies[i] - ci_energies[0]))
        print("Excitation    CI Coef")
        out_file.write("State %s %04.3f eV \n" % (i, ci_energies[i] - ci_energies[0]))
        out_file.write("Excitation    CI Coef\n")
        spin = 0 # initialise total spin
        for j in range (ci_coeffs.shape[0]): # Loop over configurations in each CIS state
                    
            if ci_level == 0:
                if j == 0: 
                    str = "|1^OS>"
                    # S^2 = 0
                elif j == 1:
                    str = "|ZW->"
                    # S^2 = 0
                elif j == 2:
                    str = "|ZW+>"
                    # S^2 = 0
                elif j == 3:
                    str = "|3^OS>"
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
                
                if np.absolute(ci_coeffs[j,i]) > csf_tol:
                    print("%s %10.5f" %(str, ci_coeffs[j,i]))
                    out_file.write("%s %10.5f \n" %(str, ci_coeffs[j,i]))
            
            elif ci_level == 1:
            ### SINGLET CSFS ### 
            # Open shell singlet ground state (|OS1>)
                if j == 0: 
                    str = "|1^OS>"
                    # S^2 = 0
            # Zwitterion - (|ZW->)    
                elif j == 1:
                    str = "|1^ZW->"
                    # S^2 = 0
            # Zwitterion 0' (|ZW+>)   
                elif j == 2:
                    str = "|1^ZW+>"
                    # S^2 = 0
            # Singlet core to SOMO 0 (|1^CS0>)
                elif j > 2 and j <= ndocc + 2:
                    iorb = ndocc + 3 - j
                    str = f"|1^CS({iorb}->0)>" 
                    # S^2 = 0 
            # Singlet core to SOMO 0' (|1^CS0'>)
                elif j > ndocc + 2 and j <= (2 * ndocc + 2):
                    iorb = 2 * ndocc + 3 - j
                    str = f"|1^CS({iorb}->0')>" 
                    # S^2 = 0
            # Singlet SOMO 0 to virtual (|1^SV0>)
                elif j > (2 * ndocc + 2) and j <= (nvirt + 2 * ndocc + 2):
                    iorb = j - (2 * ndocc + 2)
                    str = f"|1^SV(0->{iorb}')>"
                    # S^2 = 0
            # Singlet SOMO 0' to virtual (|1^SV0'>)
                elif j > (nvirt + 2 * ndocc + 2) and j <= (2 * nvirt + 2 * ndocc + 2):
                    iorb = j - (nvirt + 2 * ndocc + 2)
                    str = f"|1^SV(0'->{iorb}')>"
                    # S^2 = 0
                    
            ### TRIPLET CSFs ###
            # Triplet ground state (|OS3>)
                elif j == (2 * nvirt + 2 * ndocc + 3): 
                    str = "|3^OS>"
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet core to SOMO 0 (|3^CS0>)
                elif j > (2 * nvirt + 2 * ndocc + 3) and j <= (2 * nvirt + 3 * ndocc + 3):
                    iorb = (2 * nvirt + 3 * ndocc + 4) - j
                    str = f"|3^CS({iorb}->0)>" 
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1) 
            # Triplet core to SOMO 0' (|3^CS0'>)
                elif j > (2 * nvirt + 3 * ndocc + 3) and j <= (2 * nvirt + 4 * ndocc + 3):
                    iorb = (2 * nvirt + 4 * ndocc + 4) - j
                    str = f"|3^CS({iorb}->0')>" 
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet SOMO 0 to virtual (|3^SV0>)
                elif j > (2 * nvirt + 4 * ndocc + 3) and j <= (3 * nvirt + 4 * ndocc + 3):
                    iorb = j - (2 * nvirt + 4 * ndocc + 3)
                    str = f"|3^SV(0->{iorb}')>"
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet SOMO 0' to virtual (|3^SV0'>)
                elif j > (3 * nvirt + 4 * ndocc + 3) and j <= (4 * nvirt + 4 * ndocc + 3):
                    iorb = j - (3 * nvirt + 4 * ndocc + 3)
                    str = f"|3^SV(0'->{iorb}')>"
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
                
                if np.absolute(ci_coeffs[j,i]) > csf_tol:
                    print("%s %10.5f" %(str, ci_coeffs[j,i]))
                    out_file.write("%s %10.5f \n" %(str, ci_coeffs[j,i]))
            
            elif ci_level == 2:
            ########## SINGLET CSFS ##########   
            # Open shell singlet ground state (|OS1>)
                if j == 0: 
                    str = "|1^OS>"
                    # S^2 = 0
            # Zwitterion - (|ZW->)    
                elif j == 1:
                    str = "|1^ZW->"
                    # S^2 = 0
            # Zwitterion 0' (|ZW+>)   
                elif j == 2:
                    str = "|1^ZW+>"
                    # S^2 = 0
            # Singlet core to SOMO 0 (|1^CS>)
                elif j > 2 and j <= ndocc + 2:
                    iorb = ndocc + 3 - j
                    str = f"|1^CS({iorb}->0)>" 
                    # S^2 = 0 
            # Singlet core to SOMO 0' (|1^CS>)
                elif j > ndocc + 2 and j <= (2 * ndocc + 2):
                    iorb = 2 * ndocc + 3 - j
                    str = f"|1^CS({iorb}->0')>" 
                    # S^2 = 0
            # Singlet SOMO 0 to virtual (|1^SV>)
                elif j > (2 * ndocc + 2) and j <= (nvirt + 2 * ndocc + 2):
                    iorb = j - (2 * ndocc + 2)
                    str = f"|1^SV(0->{iorb}')>"
                    # S^2 = 0
            # Singlet SOMO 0' to virtual (|1^SV>)
                elif j > (nvirt + 2 * ndocc + 2) and j <= (2 * nvirt + 2 * ndocc + 2):
                    iorb = j - (nvirt + 2 * ndocc + 2)
                    str = f"|1^SV(0'->{iorb}')>"
                    # S^2 = 0
            # Singlet Core to Virtual 1 (|1S^CV>)
                elif j > (2 * nvirt + 2 * ndocc + 2) and j <= ((npairs) + 2 * nvirt + 2 * ndocc + 2):
                    o_orb = ndocc - ((j - (2 * nvirt + 2 * ndocc + 3)) // nvirt)
                    v_orb = ((j - (2 * nvirt + 2 * ndocc + 3)) % nvirt) + 1
                    str = f"|1S^CV({o_orb}->{v_orb}')>" 
                    # S^2 = 0
            # Singlet Core to Virtual 2 (|1T^CV>)
                elif j > ((npairs) + 2 * nvirt + 2 * ndocc + 2) and j <= (2 * (npairs) + 2 * nvirt + 2 * ndocc + 2):
                    o_orb = ndocc - ((j - ((npairs) + 2 * nvirt + 2 * ndocc + 3)) // nvirt)
                    v_orb = ((j - ((npairs) + 2 * nvirt + 2 * ndocc + 3)) % nvirt) + 1
                    str = f"|1T^CV({o_orb}->{v_orb}')>" 
                    # S^2 = 0
            # Singlet Zwitterionic Core to Virtual 0 (|1^ZCV0>)
                elif j > (2*npairs + 2 * nvirt + 2 * ndocc + 2) and j <= (3 * npairs + 2 * nvirt + 2 * ndocc + 2):
                    o_orb = ndocc - ((j - (2*npairs + 2 * nvirt + 2 * ndocc + 3)) // nvirt)
                    v_orb = ((j - (2*npairs + 2 * nvirt + 2 * ndocc + 3)) % nvirt) + 1
                    str = f"|1^ZCV0({o_orb}->{v_orb}')>" 
                    # S^2 = 0
            # Singlet Zwitterionic Core to Virtual 0' (|1^ZCV0'>)
                elif j > (3*npairs + 2 * nvirt + 2 * ndocc + 2) and j <= (4 * npairs + 2 * nvirt + 2 * ndocc + 2):
                    o_orb = ndocc - ((j - (3*npairs + 2 * nvirt + 2 * ndocc + 3)) // nvirt)
                    v_orb = ((j - (3*npairs + 2 * nvirt + 2 * ndocc + 3)) % nvirt) + 1
                    str = f"|1^ZCV0'({o_orb}->{v_orb}')>" 
                    # S^2 = 0
            ########### TRIPLET CSFs ###########
            # Triplet ground state (|OS3>)
                elif j == (4 * npairs + 2 * nvirt + 2 * ndocc + 3): 
                    str = "|3^OS>"
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet core to SOMO 0 (|3^CS>)
                elif j > (4 * npairs + 2 * nvirt + 2 * ndocc + 3) and j <= (4 * npairs + 2 * nvirt + 3 * ndocc + 3):
                    iorb = (4 * npairs + 2 * nvirt + 3 * ndocc + 4) - j
                    str = f"|3^CS({iorb}->0)>" 
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1) 
            # Triplet core to SOMO 0' (|3^CS>)
                elif j > (4 * npairs + 2 * nvirt + 3 * ndocc + 3) and j <= (4 * npairs + 2 * nvirt + 4 * ndocc + 3):
                    iorb = (4 * npairs + 2 * nvirt + 4 * ndocc + 4) - j
                    str = f"|3^CS({iorb}->0')>" 
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet SOMO 0 to virtual (|3^SV>)
                elif j > (4 * npairs + 2 * nvirt + 4 * ndocc + 3) and j <= (4 * npairs + 3 * nvirt + 4 * ndocc + 3):
                    iorb = j - (4 * npairs + 2 * nvirt + 4 * ndocc + 3)
                    str = f"|3^SV(0->{iorb}')>"
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet SOMO 0' to virtual (|3^SV>)
                elif j > (4 * npairs + 3 * nvirt + 4 * ndocc + 3) and j <= (4 * npairs + 4 * nvirt + 4 * ndocc + 3):
                    iorb = j - (4 * npairs + 3 * nvirt + 4 * ndocc + 3)
                    str = f"|3^SV(0'->{iorb}')>"
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet Core to Virtual 1 (|3T^CV>)
                elif j > (4 * npairs + 4 * nvirt + 4 * ndocc + 3) and j <= (5 * npairs + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (4 * npairs + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (4 * npairs + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|3T^CV({o_orb}->{v_orb}')>" 
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet Core to Virtual 2 (|3S^CV>)
                elif j > (5 * npairs + 4 * nvirt + 4 * ndocc + 3) and j <= (6 * npairs + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (5 * npairs + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (5 * npairs + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|3S^CV({o_orb}->{v_orb}')>" 
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet Core to Virtual 3 (|3X^CV>)
                elif j > (6 * npairs + 4 * nvirt + 4 * ndocc + 3) and j <= (7 * npairs + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (6 * npairs + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (6 * npairs + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|3X^CV({o_orb}->{v_orb}')>" 
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet Zwitterionic Core to Virtual 0 (|3^ZCV0>)
                elif j > (7 * npairs + 4 * nvirt + 4 * ndocc + 3) and j <= (8 * npairs + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (7 * npairs + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (7 * npairs + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|3^ZCV0({o_orb}->{v_orb}')>" 
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet Zwitterionic Core to Virtual 0' (|3^ZCV0'>)
                elif j > (8 * npairs + 4 * nvirt + 4 * ndocc + 3) and j <= (9 * npairs + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (9 * npairs + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (9 * npairs + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|3^ZCV0'({o_orb}->{v_orb}')>" 
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Quintet Core to Virtual (|5^CV>)
                elif j > (9 * npairs + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (9 * npairs + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (9 * npairs + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|5^CV({o_orb}->{v_orb}')>" 
                    spin += 6 * ci_coeffs[j,i]**2 # (S=2)
                    
                if np.absolute(ci_coeffs[j,i]) > csf_tol:
                    print("%s %10.5f" %(str, ci_coeffs[j,i]))
                    out_file.write("%s %10.5f \n" %(str, ci_coeffs[j,i]))
            
            elif ci_level == 3:
            ########## SINGLET CSFS ##########   
            # Open shell singlet ground state (|OS1>)
                if j == 0: 
                    str = "|1^OS>"
                    # S^2 = 0
            # Zwitterion - (|ZW->)    
                elif j == 1:
                    str = "|1^ZW->"
                    # S^2 = 0
            # Zwitterion 0' (|ZW+>)   
                elif j == 2:
                    str = "|1^ZW+>"
                    # S^2 = 0
            # Singlet core to SOMO 0 (|1^CS>)
                elif j > 2 and j <= ndocc + 2:
                    iorb = ndocc + 3 - j
                    str = f"|1^CS({iorb}->0)>" 
                    # S^2 = 0 
            # Singlet core to SOMO 0' (|1^CS>)
                elif j > ndocc + 2 and j <= (2 * ndocc + 2):
                    iorb = 2 * ndocc + 3 - j
                    str = f"|1^CS({iorb}->0')>" 
                    # S^2 = 0
            # Singlet SOMO 0 to virtual (|1^SV>)
                elif j > (2 * ndocc + 2) and j <= (nvirt + 2 * ndocc + 2):
                    iorb = j - (2 * ndocc + 2)
                    str = f"|1^SV(0->{iorb}')>"
                    # S^2 = 0
            # Singlet SOMO 0' to virtual (|1^SV>)
                elif j > (nvirt + 2 * ndocc + 2) and j <= (2 * nvirt + 2 * ndocc + 2):
                    iorb = j - (nvirt + 2 * ndocc + 2)
                    str = f"|1^SV(0'->{iorb}')>"
                    # S^2 = 0
            # Singlet Core to Virtual 1 (|1S^CV>)
                elif j > (2 * nvirt + 2 * ndocc + 2) and j <= (npairs + 2 * nvirt + 2 * ndocc + 2):
                    o_orb = ndocc - ((j - (2 * nvirt + 2 * ndocc + 3)) // nvirt)
                    v_orb = ((j - (2 * nvirt + 2 * ndocc + 3)) % nvirt) + 1
                    str = f"|1S^CV({o_orb}->{v_orb}')>" 
                    # S^2 = 0
            # Singlet Core to Virtual 2 (|1T^CV>)
                elif j > (npairs + 2 * nvirt + 2 * ndocc + 2) and j <= (2 * npairs + 2 * nvirt + 2 * ndocc + 2):
                    o_orb = ndocc - ((j - (npairs + 2 * nvirt + 2 * ndocc + 3)) // nvirt)
                    v_orb = ((j - (npairs + 2 * nvirt + 2 * ndocc + 3)) % nvirt) + 1
                    str = f"|1T^CV({o_orb}->{v_orb}')>" 
                    # S^2 = 0
            # Singlet Zwitterionic Core to Virtual 0 (|1^ZCV0>)
                elif j > (2*npairs + 2 * nvirt + 2 * ndocc + 2) and j <= (3 * npairs + 2 * nvirt + 2 * ndocc + 2):
                    o_orb = ndocc - ((j - (2*npairs + 2 * nvirt + 2 * ndocc + 3)) // nvirt)
                    v_orb = ((j - (2*npairs + 2 * nvirt + 2 * ndocc + 3)) % nvirt) + 1
                    str = f"|1^ZCV0({o_orb}->{v_orb}')>" 
                    # S^2 = 0
            # Singlet Zwitterionic Core to Virtual 0' (|1^ZCV0'>)
                elif j > (3*npairs + 2 * nvirt + 2 * ndocc + 2) and j <= (4 * npairs + 2 * nvirt + 2 * ndocc + 2):
                    o_orb = ndocc - ((j - (3*npairs + 2 * nvirt + 2 * ndocc + 3)) // nvirt)
                    v_orb = ((j - (3*npairs + 2 * nvirt + 2 * ndocc + 3)) % nvirt) + 1
                    str = f"|1^ZCV0'({o_orb}->{v_orb}')>" 
                    # S^2 = 0
            # Singlet Double Core to SOMO (|1^CSD>)
                elif j > (4 * npairs + 2 * nvirt + 2 * ndocc + 2) and j <= (ndoc1 + 4 * npairs + 2 * nvirt + 2 * ndocc + 2):
                    block_start = 4 * npairs + 2 * nvirt + 2 * ndocc + 3
                    k = j - block_start
                    o_orb1 = ndocc 
                    o_orb2 = ndocc
                    temp_k = k
                    row_size = ndocc
                    while temp_k >= row_size:
                        temp_k -= row_size
                        o_orb1 -= 1
                        row_size -= 1
                    o_orb2 = o_orb1 - temp_k
                    str = f"|1^CSD_({o_orb1},{o_orb2})>"
                    # S^2 = 0
            # Singlet Double SOMO to Virtual (|1^SVD>)
                elif j > (ndoc1 + 4 * npairs + 2 * nvirt + 2 * ndocc + 2) and j <= (ndcv1 + ndoc1 + 4 * npairs + 2 * nvirt + 2 * ndocc + 2):
                    block_start = ndoc1 + 4 * npairs + 2 * nvirt + 2 * ndocc + 3
                    k = j - block_start
                    v_orb1 = 1
                    v_orb2 = 1
                    temp_k = k
                    row_size = nvirt
                    while temp_k >= row_size:
                        temp_k -= row_size
                        v_orb1 += 1
                        row_size -= 1
                    v_orb2 = v_orb1 + temp_k
                    str = f"|1^SVD_({v_orb1}',{v_orb2}')>"
                    # S^2 = 0
            ########### TRIPLET CSFs ###########
            # Triplet ground state (|OS3>)
                elif j == (ndcv1 + ndoc1 + 4 * npairs + 2 * nvirt + 2 * ndocc + 3): 
                    str = "|3^OS>"
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet core to SOMO 0 (|3^CS>)
                elif j > (ndcv1 + ndoc1 + 4 * npairs + 2 * nvirt + 2 * ndocc + 3) and j <= (ndcv1 + ndoc1 + 4 * npairs + 2 * nvirt + 3 * ndocc + 3):
                    iorb = (ndcv1 + ndoc1 + 4 * npairs + 2 * nvirt + 3 * ndocc + 4) - j
                    str = f"|3^CS({iorb}->0)>" 
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1) 
            # Triplet core to SOMO 0' (|3^CS>)
                elif j > (ndcv1 + ndoc1 + 4 * npairs + 2 * nvirt + 3 * ndocc + 3) and j <= (ndcv1 + ndoc1 + 4 * npairs + 2 * nvirt + 4 * ndocc + 3):
                    iorb = (ndcv1 + ndoc1 + 4 * npairs + 2 * nvirt + 4 * ndocc + 4) - j
                    str = f"|3^CS({iorb}->0')>" 
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet SOMO 0 to virtual (|3^SV>)
                elif j > (ndcv1 + ndoc1 + 4 * npairs + 2 * nvirt + 4 * ndocc + 3) and j <= (ndcv1 + ndoc1 + 4 * npairs + 3 * nvirt + 4 * ndocc + 3):
                    iorb = j - (ndcv1 + ndoc1 + 4 * npairs + 2 * nvirt + 4 * ndocc + 3)
                    str = f"|3^SV(0->{iorb}')>"
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet SOMO 0' to virtual (|3^SV>)
                elif j > (ndcv1 + ndoc1 + 4 * npairs + 3 * nvirt + 4 * ndocc + 3) and j <= (ndcv1 + ndoc1 + 4 * npairs + 4 * nvirt + 4 * ndocc + 3):
                    iorb = j - (ndcv1 + ndoc1 + 4 * npairs + 3 * nvirt + 4 * ndocc + 3)
                    str = f"|3^SV(0'->{iorb}')>"
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet Core to Virtual 1 (|3T^CV>)
                elif j > (ndcv1 + ndoc1 + 4 * npairs + 4 * nvirt + 4 * ndocc + 3) and j <= (ndcv1 + ndoc1 + 5 * npairs + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (ndcv1 + ndoc1 + 4 * npairs + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (ndcv1 + ndoc1 + 4 * npairs + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|3T^CV({o_orb}->{v_orb}')>" 
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet Core to Virtual 2 (|3S^CV>)
                elif j > (ndcv1 + ndoc1 + 5 * npairs + 4 * nvirt + 4 * ndocc + 3) and j <= (ndcv1 + ndoc1 + 6 * npairs + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (ndcv1 + ndoc1 + 5 * npairs + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (ndcv1 + ndoc1 + 5 * npairs + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|3S^CV({o_orb}->{v_orb}')>" 
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet Core to Virtual 3 (|3X^CV>)
                elif j > (ndcv1 + ndoc1 + 6 * npairs + 4 * nvirt + 4 * ndocc + 3) and j <= (ndcv1 + ndoc1 + 7 * npairs + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (ndcv1 + ndoc1 + 6 * npairs + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (ndcv1 + ndoc1 + 6 * npairs + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|3X^CV({o_orb}->{v_orb}')>" 
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet Zwitterionic Core to Virtual 0 (|3^ZCV0>)
                elif j > (ndcv1 + ndoc1 + 7 * npairs + 4 * nvirt + 4 * ndocc + 3) and j <= (ndcv1 + ndoc1 + 8 * npairs + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (ndcv1 + ndoc1 + 7 * npairs + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (ndcv1 + ndoc1 + 7 * npairs + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|3^ZCV0({o_orb}->{v_orb}')>" 
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet Zwitterionic Core to Virtual 0' (|3^ZCV0'>)
                elif j > (ndcv1 + ndoc1 + 8 * npairs + 4 * nvirt + 4 * ndocc + 3) and j <= (ndcv1 + ndoc1 + 9 * npairs + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (ndcv1 + ndoc1 + 9 * npairs + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (ndcv1 + ndoc1 + 9 * npairs + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|3^ZCV0'({o_orb}->{v_orb}')>" 
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
                elif j > (ndcv1 + ndoc1 + 9 * npairs + 4 * nvirt + 4 * ndocc + 3) and j <= (ndcv1 + ndocc ** 2 + 9 * npairs + 4 * nvirt + 4 * ndocc + 3):
                    block_start = ndcv1 + ndoc1 + 9 * npairs + 4 * nvirt + 4 * ndocc + 4
                    k = j - block_start
                    o_orb1 = ndocc
                    temp_k = k
                    row_size = o_orb1 - 1
                    while temp_k >= row_size and row_size > 0:
                        temp_k -= row_size
                        o_orb1 -= 1
                        row_size = o_orb1 - 1
                    o_orb2 = (o_orb1 - 1) - temp_k
                    str = f"|3^CSD({o_orb1},{o_orb2})>"
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet double SOMO to Virtual    
                elif j > (ndcv1 + ndocc ** 2 + 9 * npairs + 4 * nvirt + 4 * ndocc + 3) and j <= (nvirt ** 2 + ndocc ** 2 + 9 * npairs + 4 * nvirt + 4 * ndocc + 3):
                    block_start = ndcv1 + ndocc ** 2 + 9 * npairs + 4 * nvirt + 4 * ndocc + 4
                    k = j - block_start
                    o_orb1 = 1
                    temp_k = k
                    row_size = nvirt - 1 
                    while temp_k >= row_size and row_size > 0:
                        temp_k -= row_size
                        o_orb1 += 1
                        row_size = nvirt - o_orb1
                    o_orb2 = o_orb1 + 1 + temp_k
                    str = f"|3^SVD({o_orb1},{o_orb2})>"
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Quintet Core to Virtual (|5^CV>)
                elif j > (nvirt ** 2 + ndocc ** 2 + 9 * npairs + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (nvirt ** 2 + ndocc ** 2 + 9 * npairs + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (nvirt ** 2 + ndocc ** 2 + 9 * npairs + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|5^CV({o_orb}->{v_orb}')>" 
                    spin += 6 * ci_coeffs[j,i]**2 # (S=2)
                
                if np.absolute(ci_coeffs[j,i]) > csf_tol:
                    print("%s %10.5f" %(str, ci_coeffs[j,i]))
                    out_file.write("%s %10.5f \n" %(str, ci_coeffs[j,i]))
            
        if i == 0:
            if spin > 1:
                print('Ground state is triplet')
                triplet = 0
                singlet = 1
            else:
                print('Ground state is singlet')
                singlet = 0
                triplet = 1

        osc3 = 2.0/3.0 * ((ci_energies[i] - ci_energies[triplet]) / toev) * (tdms[triplet][i,0]**2 + tdms[triplet][i,1]**2 + tdms[triplet][i,2]**2)  # Calculating Oscillator Strength with Triplet Ground state
        osc1 = 2.0/3.0 * ((ci_energies[i] - ci_energies[singlet]) / toev) * (tdms[singlet][i,0]**2 + tdms[singlet][i,1]**2 + tdms[singlet][i,2]**2)  # Calculating Oscillator Strength with Singlet Ground state
        osc_array3[i] = osc3
        osc_array1[i] = osc1
        s2_array[i] = spin
        print("TDMs with Triplet 'Ground' state")
        print("TDMX:%04.3f   TDMY:%04.3f   TDMZ:%04.3f   Oscillator Strength:%04.5f   <S**2>: %04.3f" % (tdms[triplet][i,0], tdms[triplet][i,1], tdms[triplet][i,2], osc3, spin))
        print("--------------------------------------------------------------------")
        print("TDMs with Singlet 'Ground' state")
        print("TDMX:%04.3f   TDMY:%04.3f   TDMZ:%04.3f   Oscillator Strength:%04.5f   <S**2>: %04.3f" % (tdms[singlet][i,0], tdms[singlet][i,1], tdms[singlet][i,2], osc1, spin))
        print("--------------------------------------------------------------------\n")
        out_file.write("TDMs with Triplet 'Ground' state")
        out_file.write("TDMX:%04.3f   TDMY:%04.3f   TDMZ:%04.3f   Oscillator Strength:%04.5f   <S**2>: %04.3f" % (tdms[triplet][i,0], tdms[triplet][i,1], tdms[triplet][i,2], osc3, spin))
        out_file.write("--------------------------------------------------------------------")
        out_file.write("TDMs with Singlet 'Ground' state")
        out_file.write("TDMX:%04.3f   TDMY:%04.3f   TDMZ:%04.3f   Oscillator Strength:%04.5f   <S**2>: %04.3f" % (tdms[singlet][i,0], tdms[singlet][i,1], tdms[singlet][i,2], osc1, spin))
        out_file.write("--------------------------------------------------------------------\n")
        #strng3 = strng3 + broaden(20.0,osc3,ci_energies[i]-ci_energies[triplet])
        strng3 = strng3 + broaden(FWHM,osc3,ci_energies[i]-ci_energies[triplet])
        strng1 = strng1 + broaden(FWHM,osc1,ci_energies[i]-ci_energies[singlet])
    
    return (strng3, strng1), (osc_array3, osc_array1), s2_array

def broaden(FWHM,osc,energy):
    if brdn_typ == 'wavelength' and line_typ == 'lorentzian':
        eqn="+%04.3f*1/(1+((%04.3f-x)/(%s/2))**2)" %(osc,evtonm/energy,FWHM)
    elif brdn_typ == 'energy' and line_typ == 'lorentzian':
        eqn="+%04.3f*1/(1+((%04.3f-x)/(0.5*%s*%04.3f*x))**2)"  %(osc,evtonm/energy,FWHM,evtonm/energy)
    elif brdn_typ == 'energy' and line_typ == 'gaussian':
        eqn="+%04.3f*exp(-((%04.3f-x)/(0.5*%s*%04.3f*x))**2)" %(osc,evtonm/energy,FWHM,evtonm/energy)
    return eqn