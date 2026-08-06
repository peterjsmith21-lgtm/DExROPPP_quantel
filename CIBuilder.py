import numpy as np
from scipy.linalg import block_diag

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
            CI[row, col] = - rep_tens[o_orb, SOMO1, SOMO1, v_orb] 
        # <CS0|H|SV0'>
        for col in range(2*ndocc + nvirt, 2*ndocc + 2*nvirt):
            v_orb = col - (2*ndocc + nvirt) + (SOMO2 + 1)
            CI[row, col] = rep_tens[o_orb, SOMO1, SOMO2, v_orb] - 2 * rep_tens[o_orb, SOMO2, SOMO1, v_orb]
            
    
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
            CI[row, col] = rep_tens[o_orb, SOMO2, SOMO1, v_orb] - 2 * rep_tens[o_orb, SOMO1, SOMO2, v_orb]
        # <CS0'|H|SV0'>
        for col in range(2*ndocc + nvirt,  2*ndocc + 2*nvirt):
            v_orb = col - (2*ndocc + nvirt) + (SOMO2 + 1)
            CI[row, col] = - rep_tens[o_orb, SOMO2, SOMO2, v_orb]
    
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
        CI[2,col] = np.sqrt(1.5) * (rep_tens[o_orb, SOMO1, SOMO2, v_orb] - rep_tens[o_orb, SOMO2, SOMO1, v_orb])
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
                CI[row, col] = (1 / np.sqrt(2)) * (rep_tens[SOMO1, v_orb, o_orb1, o_orb2] -  2 * rep_tens[SOMO1, o_orb2, o_orb1, v_orb])
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
                CI[row, col] = 2 * rep_tens[SOMO2, o_orb2, o_orb1, v_orb] - rep_tens[SOMO2, v_orb, o_orb1, o_orb2] 
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
                CI[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[SOMO2, o_orb2, o_orb1, v_orb] - rep_tens[SOMO2, v_orb, o_orb1, o_orb2])
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
                CI[row, col] = rep_tens[SOMO1, v_orb, o_orb1, o_orb2] - 2 * rep_tens[SOMO1, o_orb2, o_orb1, v_orb] - rep_tens[SOMO1, o_orb2, o_orb1, v_orb]
    
    row_index = 2 * ndocc + 3
    for row in range(row_index, row_index + nvirt):
        v_orb1 = row - row_index + (SOMO2 + 1)
        # <SV0|H|HL1>
        for col in range(0, npairs):
            o_orb = col // nvirt
            v_orb2 = col % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                CI[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[o_orb, v_orb1, v_orb1, SOMO1] + 0.5 * rep_tens[SOMO1, SOMO1, SOMO1, v_orb1] - rep_tens[v_orb1, v_orb1, SOMO1, o_orb] - 1.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO1])
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
                CI[row, col] = 2 * rep_tens[SOMO2, v_orb1, v_orb1, o_orb] + rep_tens[o_orb, SOMO1, SOMO1, SOMO2] - rep_tens[o_orb, SOMO2, v_orb1, v_orb1] - 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO2] - 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO2]
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
                CI[row, col] = - rep_tens[o_orb, SOMO2, SOMO2, SOMO1]
    
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
                CI[row, col] = rep_tens[v_orb1, o_orb1, o_orb1, v_orb2] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb2]
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
                CI[row, col] = np.sqrt(2) * (rep_tens[SOMO1, SOMO2, v_orb1, v_orb1] - rep_tens[SOMO1, SOMO2, o_orb1, o_orb1] + 0.5 * rep_tens[SOMO1, o_orb1, o_orb1, SOMO2] - 0.5 * rep_tens[SOMO1, v_orb1, v_orb1, SOMO2] \
                                - 0.5 * rep_tens[SOMO2, SOMO1, SOMO1, SOMO1] + 0.5 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO1])
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:
                CI[row, col] = np.sqrt(2) * (0.5 * rep_tens[o_orb1, SOMO1, SOMO2, o_orb2] - rep_tens[SOMO1, SOMO2, o_orb1, o_orb2])
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                CI[row, col] = np.sqrt(2) * (rep_tens[SOMO1, SOMO2, v_orb1, v_orb2] - 0.5 * rep_tens[v_orb1, SOMO2, SOMO1, v_orb2])

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
            CI[row, col] = rep_tens[o_orb, SOMO1, SOMO1, v_orb] 
        # <CS0|H|SV0'>
        for col in range(2*ndocc + nvirt, 2*ndocc + 2*nvirt):
            v_orb = col - (2*ndocc + nvirt) + (SOMO2 + 1)
            CI[row, col] = - rep_tens[o_orb, SOMO1, SOMO2, v_orb]
            
    
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
            CI[row, col] = rep_tens[o_orb, SOMO2, SOMO1, v_orb]
        # <CS0'|H|SV0'>
        for col in range(2*ndocc + nvirt,  2*ndocc + 2*nvirt):
            v_orb = col - (2*ndocc + nvirt) + (SOMO2 + 1)
            CI[row, col] = - rep_tens[o_orb, SOMO2, SOMO2, v_orb]
    
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
    
    row_dim = 5 * (npairs) + 2 * ndocc + 2 * nvirt + 3
    col_dim = 5 * (npairs)
    CI = np.zeros((row_dim, col_dim))  # Initialize CI Block

    # <OS3|H|HL1> = 0
    # <OS3|H|HL2>
    for col in range(npairs, 2*npairs):
        o_orb = (col - npairs) // nvirt # Increase o_orb after every ndocc cols
        v_orb = (col - npairs) % nvirt + (SOMO2 + 1) # Increase v_orb for every col then reset after ndocc cols
        CI[0,col] = np.sqrt(1.5) * (rep_tens[o_orb, SOMO2, SOMO2, v_orb] - rep_tens[o_orb, SOMO1, SOMO1, v_orb])
    # <OS3|H|HL3>
    for col in range(2*npairs, 3*npairs):
        o_orb = (col - 2*npairs) // nvirt
        v_orb = (col - 2*npairs) % nvirt + (SOMO2 + 1)
        CI[0,col] = 1
    # <OS3|H|ZHL1> 
    for col in range(3*npairs, 4*npairs):
        o_orb = (col - 3*npairs) // nvirt
        v_orb = (col - 3*npairs) % nvirt + (SOMO2 + 1)
        CI[0,col] = 2 * rep_tens[o_orb, v_orb, SOMO1, SOMO2] - rep_tens[o_orb, SOMO1, SOMO2, v_orb]
    # <OS3|H|ZHL2>
    for col in range(4*npairs, 5*npairs):
        o_orb = (col - 4*npairs) // nvirt
        v_orb = (col - 4*npairs) % nvirt + (SOMO2 + 1)
        CI[0,col] = 2 * rep_tens[o_orb, v_orb, SOMO1, SOMO2] - rep_tens[o_orb, SOMO2, SOMO1, v_orb]


    row_index = 1
    for row in range(row_index, row_index + ndocc):
        o_orb1 = row - row_index
        # <CS0|H|HL1>
        for col in range(0, npairs):
            o_orb2 = col // nvirt
            v_orb = col % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                CI[row, col] = (1 / np.sqrt(2)) * (rep_tens[o_orb1, o_orb1, SOMO1, v_orb] + 1.5 * rep_tens[SOMO1, SOMO2, SOMO2, v_orb] - 0.5 * rep_tens[SOMO1, SOMO1, SOMO1, v_orb] - 2 * rep_tens[SOMO1, o_orb1, o_orb1, v_orb])
            else:    
                CI[row, col] = (1 / np.sqrt(2)) * (rep_tens[SOMO1, v_orb, o_orb1, o_orb2] -  2 * rep_tens[SOMO1, o_orb2, o_orb1, v_orb])
        # <CS0|H|HL2>
        for col in range(npairs, 2*npairs):
            o_orb2 = (col - npairs) // nvirt
            v_orb = (col - npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                CI[row, col] = np.sqrt(1.5) * (0.5 * rep_tens[SOMO1, SOMO2, SOMO2, v_orb] + 0.5 * rep_tens[SOMO1, SOMO1, SOMO1, v_orb] - rep_tens[o_orb1, o_orb1, SOMO1, v_orb])
            else:    
                CI[row, col] = - np.sqrt(1.5) * rep_tens[SOMO1, v_orb, o_orb1, o_orb2]
        # <CS0|H|HL3>
        for col in range(2*npairs, 3*npairs):
            o_orb2 = (col - 2*npairs) // nvirt
            v_orb = (col - 2*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                CI[row, col] = 1
            else:    
                CI[row, col] = 1
        # <CS0|H|ZHL1>
        for col in range(3*npairs, 4*npairs):
            o_orb2 = (col - 3*npairs) // nvirt
            v_orb = (col - 3*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                CI[row, col] = 2 * rep_tens[SOMO2, o_orb1, o_orb1, v_orb] + rep_tens[SOMO2, v_orb, SOMO1, SOMO1] - rep_tens[o_orb1, o_orb1, SOMO2, v_orb] - 0.5 * rep_tens[SOMO2, SOMO1, SOMO1, v_orb] - 0.5 * rep_tens[SOMO2, SOMO2, SOMO2, v_orb] 
            else:    
                CI[row, col] = 2 * rep_tens[SOMO2, o_orb2, o_orb1, v_orb] - rep_tens[SOMO2, v_orb, o_orb1, o_orb2] 
        # <CS0|H|ZHL2>
        for col in range(4*npairs, 5*npairs):
            o_orb2 = (col - 4*npairs) // nvirt
            v_orb = (col - 4*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                CI[row, col] = - rep_tens[SOMO2, SOMO1, SOMO1, v_orb]
    
    row_index = ndocc + 1
    for row in range(row_index, row_index + ndocc):
        o_orb1 = row - row_index
        # <CS0'|H|HL1>
        for col in range(0, npairs):
            o_orb2 = col // nvirt
            v_orb = col % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                CI[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[SOMO2, o_orb1, o_orb1, v_orb] + 0.5 * rep_tens[SOMO2, SOMO2, SOMO2, v_orb] - rep_tens[o_orb1, o_orb1, SOMO2, v_orb] - 1.5 * rep_tens[SOMO2, SOMO1, SOMO1, v_orb])
            else:
                CI[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[SOMO2, o_orb2, o_orb1, v_orb] - rep_tens[SOMO2, v_orb, o_orb1, o_orb2])
        # <CS0'|H|HL2>
        for col in range(npairs, 2 * npairs):
            o_orb2 = (col - npairs) // nvirt
            v_orb = (col - npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                CI[row, col] = np.sqrt(1.5) * (0.5 * rep_tens[SOMO2, SOMO1, SOMO1, v_orb] + 0.5 * rep_tens[SOMO2, SOMO2, SOMO2, v_orb] - rep_tens[o_orb1, o_orb1, SOMO2, v_orb])
            else:    
                CI[row, col] = - np.sqrt(1.5) * rep_tens[SOMO2, v_orb, o_orb1, o_orb2]
        # <CS0'|H|HL3>
        for col in range(2*npairs, 3*npairs):
            o_orb2 = (col - 2*npairs) // nvirt
            v_orb = (col - 2*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                CI[row, col] = 1
            else:    
                CI[row, col] = 1
        # <CS0'|H|ZHL1>
        for col in range(3*npairs, 4*npairs):
            o_orb2 = (col - 3*npairs) // nvirt
            v_orb = (col - 3*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                CI[row, col] = rep_tens[SOMO1, SOMO2, SOMO2, v_orb]
        # <CS0'|H|ZHL2>
        for col in range(4*npairs, 5*npairs):
            o_orb2 = (col - 4*npairs) // nvirt
            v_orb = (col - 4*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                CI[row, col] = rep_tens[SOMO1, v_orb, o_orb1, o_orb1] + 0.5 * rep_tens[SOMO1, SOMO1, SOMO1, v_orb] + 0.5 * rep_tens[SOMO1, SOMO2, SOMO2, v_orb] - 2 * rep_tens[SOMO1, o_orb1, o_orb1, v_orb] - rep_tens[SOMO1, v_orb, SOMO2, SOMO2]
            else:
                CI[row, col] = rep_tens[SOMO1, v_orb, o_orb1, o_orb2] - 2 * rep_tens[SOMO1, o_orb2, o_orb1, v_orb] - rep_tens[SOMO1, o_orb2, o_orb1, v_orb]
    
    row_index = 2 * ndocc + 1
    for row in range(row_index, row_index + nvirt):
        v_orb1 = row - row_index + (SOMO2 + 1)
        # <SV0|H|HL1>
        for col in range(0, npairs):
            o_orb = col // nvirt
            v_orb2 = col % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                CI[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[o_orb, v_orb1, v_orb1, SOMO1] + 0.5 * rep_tens[SOMO1, SOMO1, SOMO1, v_orb1] - rep_tens[v_orb1, v_orb1, SOMO1, o_orb] - 1.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO1])
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
        # <SV0|H|HL3>
        for col in range(2*npairs, 3*npairs):
            o_orb = (col - 2*npairs) // nvirt
            v_orb2 = (col - 2*npairs) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                CI[row, col] = 2
            else:    
                CI[row, col] = 2
        # <SV0|H|ZHL1>
        for col in range(3*npairs, 4*npairs):
            o_orb = (col - 3*npairs) // nvirt
            v_orb2 = (col - 3*npairs) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                CI[row, col] = - rep_tens[o_orb, SOMO1, SOMO1, SOMO2]
        # <SV0|H|ZHL2>
        for col in range(4*npairs, 5*npairs):
            o_orb = (col - 4*npairs) // nvirt
            v_orb2 = (col - 4*npairs) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                CI[row, col] = 2 * rep_tens[SOMO2, v_orb1, v_orb1, o_orb] + rep_tens[o_orb, SOMO1, SOMO1, SOMO2] - rep_tens[o_orb, SOMO2, v_orb1, v_orb1] - 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO2] - 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO2]
            else:
                CI[row, col] = 2 * rep_tens[SOMO2, v_orb1, v_orb2, o_orb] - rep_tens[o_orb, SOMO2, v_orb1, v_orb2]
                
    row_index = 2 * ndocc + nvirt + 1
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
        # <SV0'|H|HL3>
        for col in range(2*npairs, 3*npairs):
            o_orb = (col - 2*npairs) // nvirt
            v_orb2 = (col - 2*npairs) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                CI[row, col] = 2
            else:    
                CI[row, col] = 2
        # <SV0'|H|ZHL1>
        for col in range(3*npairs, 4*npairs):
            o_orb = (col - 3*npairs) // nvirt
            v_orb2 = (col - 3*npairs) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                CI[row, col] = rep_tens[o_orb, SOMO1, v_orb1, v_orb1] + 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO1] + 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO1] - 2 * rep_tens[SOMO1, v_orb1, v_orb1, o_orb] - rep_tens[o_orb, SOMO1, SOMO2, SOMO2]
            else:
                CI[row, col] = rep_tens[o_orb, SOMO1, v_orb1, v_orb2] - 2 * rep_tens[SOMO1, v_orb1, v_orb2, o_orb]
        # <SV0'|H|ZHL2>
        for col in range(4*npairs, 5*npairs):
            o_orb = (col - 4*npairs) // nvirt
            v_orb2 = (col - 4*npairs) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                CI[row, col] = - rep_tens[o_orb, SOMO2, SOMO2, SOMO1]
    
    
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
                    + 1.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1] + 2 * rep_tens[o_orb1, v_orb1, v_orb1, o_orb1]
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                CI[row, col] =  2 * rep_tens[o_orb1, v_orb1, v_orb1, o_orb2] - rep_tens[o_orb1, o_orb2, v_orb1, v_orb1]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                CI[row, col] = rep_tens[v_orb1, o_orb1, o_orb1, v_orb2] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb2]
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
        # <HL1|H|HL3>
        for col in range(2*npairs, 3*npairs):
            o_orb2 = (col - 2*npairs) // nvirt
            v_orb2 = (col - 2*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                CI[row, col] = 2
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                CI[row, col] =  2
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                CI[row, col] =  2
        #<HL1|H|ZHL1>
        for col in range(3*npairs, 4*npairs):
            o_orb2 = (col - 3*npairs) // nvirt
            v_orb2 = (col - 3*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                CI[row, col] = np.sqrt(2) * (rep_tens[SOMO1, SOMO2, v_orb1, v_orb1] - rep_tens[SOMO1, SOMO2, o_orb1, o_orb1] + 0.5 * rep_tens[SOMO1, o_orb1, o_orb1, SOMO2] - 0.5 * rep_tens[SOMO1, v_orb1, v_orb1, SOMO2] \
                                + 0.5 * rep_tens[SOMO2, SOMO1, SOMO1, SOMO1] - 0.5 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO1])
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:
                CI[row, col] = np.sqrt(2) * (0.5 * rep_tens[o_orb1, SOMO2, SOMO1, o_orb2] - rep_tens[SOMO1, SOMO2, o_orb1, o_orb2])
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                CI[row, col] = np.sqrt(2) * (rep_tens[SOMO1, SOMO2, v_orb1, v_orb2] - 0.5 * rep_tens[v_orb1, SOMO1, SOMO2, v_orb2])
        #<HL1|H|ZHL2>
        for col in range(4*npairs, 5*npairs):
            o_orb2 = (col - 4*npairs) // nvirt
            v_orb2 = (col - 4*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                CI[row, col] = np.sqrt(2) * (rep_tens[SOMO1, SOMO2, v_orb1, v_orb1] - rep_tens[SOMO1, SOMO2, o_orb1, o_orb1] + 0.5 * rep_tens[SOMO1, o_orb1, o_orb1, SOMO2] - 0.5 * rep_tens[SOMO1, v_orb1, v_orb1, SOMO2] \
                                - 0.5 * rep_tens[SOMO2, SOMO1, SOMO1, SOMO1] + 0.5 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO1])
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:
                CI[row, col] = np.sqrt(2) * (0.5 * rep_tens[o_orb1, SOMO1, SOMO2, o_orb2] - rep_tens[SOMO1, SOMO2, o_orb1, o_orb2])
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                CI[row, col] = np.sqrt(2) * (rep_tens[SOMO1, SOMO2, v_orb1, v_orb2] - 0.5 * rep_tens[v_orb1, SOMO2, SOMO1, v_orb2])
    
    row_index = npairs + 2 * ndocc + 2 * nvirt + 1
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
        # <HL2|H|HL3>
        for col in range(2*npairs, 3*npairs):
            o_orb2 = (col - 2*npairs) // nvirt
            v_orb2 = (col - 2*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                CI[row, col] = 2
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                CI[row, col] =  2
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                CI[row, col] =  2
        #<HL2|H|ZHL1>
        for col in range(3*npairs, 4*npairs):
            o_orb2 = (col - 3*npairs) // nvirt
            v_orb2 = (col - 3*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                CI[row, col] = np.sqrt(1.5) * (rep_tens[v_orb1, SOMO1, SOMO2, v_orb1] - rep_tens[SOMO1, o_orb1, o_orb1, SOMO2])
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:
                CI[row, col] = - np.sqrt(1.5) * rep_tens[SOMO2, o_orb1, o_orb2, SOMO1]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                CI[row, col] = np.sqrt(1.5) * rep_tens[SOMO1, v_orb1, v_orb2, SOMO2]
        #<HL2|H|ZHL2>
        for col in range(4*npairs, 5*npairs):
            o_orb2 = (col - 4*npairs) // nvirt
            v_orb2 = (col - 4*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                CI[row, col] = np.sqrt(2) * (rep_tens[SOMO1, SOMO2, v_orb1, v_orb1] - rep_tens[SOMO1, SOMO2, o_orb1, o_orb1] + 0.5 * rep_tens[SOMO1, o_orb1, o_orb1, SOMO2] - 0.5 * rep_tens[SOMO1, v_orb1, v_orb1, SOMO2] \
                                - 0.5 * rep_tens[SOMO2, SOMO1, SOMO1, SOMO1] + 0.5 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO1])
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:
                CI[row, col] = np.sqrt(2) * (0.5 * rep_tens[o_orb1, SOMO1, SOMO2, o_orb2] - rep_tens[SOMO1, SOMO2, o_orb1, o_orb2])
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                CI[row, col] = np.sqrt(2) * (rep_tens[SOMO1, SOMO2, v_orb1, v_orb2] - 0.5 * rep_tens[v_orb1, SOMO2, SOMO1, v_orb2])
                
    row_index = 2 * npairs + 2 * ndocc + 2 * nvirt + 1
    for row in range(row_index, row_index + npairs):
        o_orb1 = (row - row_index) // nvirt
        v_orb1 = (row - row_index) % nvirt + (SOMO2 + 1)
        # <HL2|H|HL2>
        for col in range(row - row_index + 2*npairs, 3*npairs):
            o_orb2 = (col - 2*npairs) // nvirt
            v_orb2 = (col - 2*npairs) % nvirt + (SOMO2 + 1)
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
        for col in range(3*npairs, 4*npairs):
            o_orb2 = (col - 3*npairs) // nvirt
            v_orb2 = (col - 3*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                CI[row, col] = np.sqrt(1.5) * (rep_tens[v_orb1, SOMO1, SOMO2, v_orb1] - rep_tens[SOMO1, o_orb1, o_orb1, SOMO2])
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:
                CI[row, col] = - np.sqrt(1.5) * rep_tens[SOMO2, o_orb1, o_orb2, SOMO1]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                CI[row, col] = np.sqrt(1.5) * rep_tens[SOMO1, v_orb1, v_orb2, SOMO2]
        #<HL2|H|ZHL2>
        for col in range(4*npairs, 5*npairs):
            o_orb2 = (col - 4*npairs) // nvirt
            v_orb2 = (col - 4*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                CI[row, col] = np.sqrt(2) * (rep_tens[SOMO1, SOMO2, v_orb1, v_orb1] - rep_tens[SOMO1, SOMO2, o_orb1, o_orb1] + 0.5 * rep_tens[SOMO1, o_orb1, o_orb1, SOMO2] - 0.5 * rep_tens[SOMO1, v_orb1, v_orb1, SOMO2] \
                                - 0.5 * rep_tens[SOMO2, SOMO1, SOMO1, SOMO1] + 0.5 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO1])
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:
                CI[row, col] = np.sqrt(2) * (0.5 * rep_tens[o_orb1, SOMO1, SOMO2, o_orb2] - rep_tens[SOMO1, SOMO2, o_orb1, o_orb2])
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                CI[row, col] = np.sqrt(2) * (rep_tens[SOMO1, SOMO2, v_orb1, v_orb2] - 0.5 * rep_tens[v_orb1, SOMO2, SOMO1, v_orb2])

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
        Singlet_CI[:cs_sv_block.shape[0], ref_block.shape[1]:(ref_block.shape[1]+cs_sv_block.shape[0])] = cs_sv_block
        Singlet_CI[:hl_block.shape[0], (ref_block.shape[1]+cs_sv_block.shape[0]):] = hl_block
    
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
        ref_block = build_triplet_ref_block(ndocc, energy0, orb_energies, rep_tens)
        cs_sv_block = build_triplet_CS_SV_block(ndocc, norbs, energy0, orb_energies, rep_tens)
        hl_block = build_triplet_HL_block(ndocc, norbs, energy0, orb_energies, rep_tens)
        
        Triplet_CI = np.zeros((hl_block.shape[0], hl_block.shape[0]))
        Triplet_CI[:ref_block.shape[0], :ref_block.shape[1]] = ref_block
        Triplet_CI[:cs_sv_block.shape[0], ref_block.shape[1]:(ref_block.shape[1]+cs_sv_block.shape[0])] = cs_sv_block
        Triplet_CI[:hl_block.shape[0], (ref_block.shape[1]+cs_sv_block.shape[0]):] = hl_block
    
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