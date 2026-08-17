import numpy as np
from scipy.linalg import block_diag
from ExROPPP_settings_opt import *

'''
File containing helper functions for building the Transition Dipole Matrix for a diradical system
'''

def cartesian_operators(coords, hf_orbs):
    
    dip1el = np.einsum('ui,uk,uj->ijk', hf_orbs, coords, hf_orbs) * tobohr
    
    x_operator = dip1el[:, :, 0]
    y_operator = dip1el[:, :, 1]
    z_operator = dip1el[:, :, 2]
    full_cartesian_operator = dip1el
    
    return full_cartesian_operator, x_operator, y_operator, z_operator

def build_singlet_ref_block(ndocc, dip1el):
    '''
    Function to build the CI matrix for 3 singlet reference states for a diradical system.
    These are the Open-Shell Singlet (OS1) and the  +/- Combinations of Zwitterion states (ZW+ and ZW-).
    Args: 
        ndocc (int): Number of doubly occupied orbitals
        dip1el (numpy.ndarray): Dipole moment operator for the system
    Returns:
        numpy.ndarray: CI matrix for the diradical system
    '''
    # Calculate the size of the CI matrix based on the number of doubly occupied orbitals
    TDM = np.zeros((3,3,3))  # Initialize a 3x3x3 Transition Dipole Matrix
    SOMO1 = ndocc # Index of SOMO1
    SOMO2 = ndocc + 1 # Index of SOMO2
    
    ndocc_sum = - 2.0 * dip1el[np.arange(ndocc), np.arange(ndocc), :].sum(axis=0)

    # <OS1|mu|OS1>
    TDM[0,0,:] = ndocc_sum
    TDM[0,0,:] -= (dip1el[SOMO1,SOMO1,:] + dip1el[SOMO2,SOMO2,:]) #Adding contribution from SOMOs
    # <OS1|mu|ZW->  = 0
    # <OS1|mu|ZW+> 
    TDM[0,2,:] = - 2 * dip1el[SOMO1,SOMO2,:]
    
    # <ZW-|mu|ZW->
    TDM[1,1,:] = ndocc_sum
    TDM[1,1,:] -= (dip1el[SOMO1,SOMO1,:] + dip1el[SOMO2,SOMO2,:]) #Adding contribution from SOMOs
    # <ZW-|mu|ZW+>
    TDM[1,2,:] = - (dip1el[SOMO1,SOMO1,:] - dip1el[SOMO2,SOMO2,:])
    
    # <ZW+|mu|ZW+>
    TDM[2,2,:] = ndocc_sum
    TDM[2,2,:] -= (dip1el[SOMO1,SOMO1,:] + dip1el[SOMO2,SOMO2,:]) #Adding contribution from SOMOs
    
    return TDM

def build_singlet_CS_SV_block(ndocc, norbs, dip1el):
    '''
    Function to build the upper diagonal of the CI matrix for singlet reference states for a diradical system - the closed-shell singlet (CS) and the single-reference singlet (SV).
    Args:
        ndocc (int): Number of doubly occupied orbitals
        norbs (int): Total number of orbitals
        dip1el (numpy.ndarray): Dipole moment operator for the system
    Returns:
        numpy.ndarray: CI matrix for the diradical system
    '''

    SOMO1 = ndocc # Index of SOMO1
    SOMO2 = ndocc + 1 # Index of SOMO2
    nvirt = norbs - ndocc - 2 # Number of virtual orbitals
    
    row_dim = 2 * ndocc + 2 * nvirt + 3
    col_dim = 2 * ndocc + 2 * nvirt
    TDM = np.zeros((row_dim, col_dim, 3))  # Initialize TDM Block

    ndocc_sum = - 2.0 * dip1el[np.arange(ndocc), np.arange(ndocc), :].sum(axis=0)
    
    # <OS1|mu|CS0>
    for col in range(0, ndocc):
        o_orb = col
        TDM[0,col,:] = dip1el[o_orb, SOMO1, :]
    # <OS1|mu|CS0'> 
    for col in range(ndocc, 2 * ndocc):
        o_orb = col - ndocc
        TDM[0,col,:] = - dip1el[o_orb, SOMO2, :]
    # <OS1|mu|SV0>
    for col in range(2 * ndocc, 2 * ndocc + nvirt):
        v_orb = col - (2 * ndocc) + (SOMO2 + 1)
        TDM[0,col,:] = - dip1el[v_orb, SOMO1, :]
    # <OS1|mu|SV0'>
    for col in range(2 * ndocc + nvirt, 2 * ndocc + 2 * nvirt):
        v_orb = col - (2 * ndocc + nvirt) + (SOMO2 + 1)
        TDM[0,col,:] = dip1el[v_orb, SOMO2, :]
    
    # <ZW-|mu|CS0>
    for col in range(0, ndocc):
        o_orb = col
        TDM[1,col,:] = - dip1el[o_orb, SOMO2, :]
    # <ZW-|mu|CS0'>
    for col in range(ndocc, 2 * ndocc):
        o_orb = col - ndocc
        TDM[1,col,:] = - dip1el[o_orb, SOMO1, :]
    # <ZW-|mu|SV0>
    for col in range(2 * ndocc, 2 * ndocc + nvirt):
        v_orb = col - (2 * ndocc) + (SOMO2 + 1)
        TDM[1,col,:] = dip1el[v_orb, SOMO2, :]
    # <ZW-|mu|SV0'>
    for col in range(2 * ndocc + nvirt, 2 * ndocc + 2 * nvirt):
        v_orb = col - (2 * ndocc + nvirt) + (SOMO2 + 1)
        TDM[1,col,:] = dip1el[v_orb, SOMO1, :]
        
    # <ZW+|mu|CS0>
    for col in range(0, ndocc):
        o_orb = col
        TDM[2,col,:] = - dip1el[o_orb, SOMO2, :]
    # <ZW+|mu|CS0'>
    for col in range(ndocc, 2 * ndocc):
        o_orb = col - ndocc
        TDM[2,col,:] = dip1el[o_orb, SOMO1, :]
    # <ZW+|mu|SV0>
    for col in range(2 * ndocc, 2 * ndocc + nvirt):
        v_orb = col - (2 * ndocc) + (SOMO2 + 1)
        TDM[2,col,:] = - dip1el[v_orb, SOMO2, :]
    # <ZW+|mu|SV0'>
    for col in range(2 * ndocc + nvirt, 2 * ndocc + 2 * nvirt):
        v_orb = col - (2 * ndocc + nvirt) + (SOMO2 + 1)
        TDM[2,col,:] = dip1el[v_orb, SOMO1, :]
    
    row_index = 3
    for row in range(row_index, row_index + ndocc):
        o_orb1 = row - row_index
        # <CS0|H|CS0>
        for col in range(row - row_index, ndocc):
            o_orb2 = col
            if o_orb1 == o_orb2:
                TDM[row,col,:] = ndocc_sum
                TDM[row,col,:] += dip1el[o_orb1, o_orb1, :] # Remove contribution from 1e in HOMO i
                TDM[row,col,:] -= 2 * dip1el[SOMO1, SOMO1, :] # Add contribution from 2e in SOMO1
                TDM[row,col,:] -= dip1el[SOMO2, SOMO2, :] # Add contribution from 1e in SOMO2
            else:    
                TDM[row,col,:] = dip1el[o_orb1, o_orb2, :] 
        # <CS0|H|CS0'>
        for col in range(ndocc, 2*ndocc):
            o_orb2 = col - ndocc
            if o_orb1 == o_orb2:
                 TDM[row,col,:] = -dip1el[SOMO1, SOMO2, :] # Only diagonal elements are non-zero
        # <CS0|H|SV0> = 0
        # <CS0|H|SV0'> = 0
            
    row_index = ndocc + 3
    for row in range(row_index, row_index + ndocc):
        o_orb1 = row - row_index
        # <CS0'|H|CS0'>
        for col in range(row - 3, 2*ndocc):
            o_orb2 = col - ndocc
            if o_orb1 == o_orb2:
                TDM[row,col,:] = ndocc_sum
                TDM[row,col,:] += dip1el[o_orb1, o_orb1, :] # Remove contribution from 1e in HOMO i
                TDM[row,col,:] -= 2 * dip1el[SOMO2, SOMO2, :] # Add contribution from 2e in SOMO2
                TDM[row,col,:] -= dip1el[SOMO1, SOMO1, :] # Add contribution from 1e in SOMO1
            else:    
                TDM[row,col,:] = dip1el[o_orb1, o_orb2, :]
        # <CS0'|H|SV0> = 0
        # <CS0'|H|SV0'> = 0
    
    row_index = 2 * ndocc + 3
    for row in range(row_index, row_index + nvirt):
        v_orb1 = row - row_index + (SOMO2 + 1)
        # <SV0|H|SV0>
        for col in range(row - 3, 2*ndocc + nvirt):
            v_orb2 = col - (2*ndocc) + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                TDM[row,col,:] = ndocc_sum
                TDM[row,col,:] -= dip1el[v_orb1, v_orb1, :] # Add contribution from 1e in LUMO j
                TDM[row,col,:] -= dip1el[SOMO2, SOMO2, :] # Add contribution from 1e in SOMO2
            else:    
                TDM[row,col,:] = -dip1el[v_orb1, v_orb2, :]
        # <SV0|H|SV0'>
        for col in range(2*ndocc + nvirt, 2*ndocc + 2*nvirt):
            v_orb2 = col - (2*ndocc + nvirt) + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                TDM[row,col,:] = dip1el[SOMO1,SOMO2,:] 
    
    row_index = 2 * ndocc + nvirt + 3
    # <SV0'|H|SV0'>
    for row in range(row_index, row_index + nvirt):
        v_orb1 = row - row_index + (SOMO2 + 1)
        for col in range(row - 3, 2*ndocc + 2*nvirt):
            v_orb2 = col - (2*ndocc + nvirt) + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                TDM[row,col,:] = ndocc_sum
                TDM[row,col,:] -= dip1el[v_orb1, v_orb1, :] # Add contribution from 1e in LUMO j
                TDM[row,col,:] -= dip1el[SOMO1, SOMO1, :] # Add contribution from 1e in SOMO1
            else:    
                TDM[row,col,:] = -dip1el[v_orb1, v_orb2, :]

    return TDM



def build_singlet_HL_block(ndocc, norbs, dip1el):
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
    TDM = np.zeros((row_dim, col_dim, 3))  # Initialize CI Block

    ndocc_sum = - 2.0 * dip1el[np.arange(ndocc), np.arange(ndocc), :].sum(axis=0)
    
    # <OS1|H|HL1>
    for col in range(0, npairs):
        o_orb = col // nvirt # Increase o_orb after every ndocc cols
        v_orb = col % nvirt + (SOMO2 + 1) # Increase v_orb for every col then reset after ndocc cols
        TDM[0,col,:] = - np.sqrt(2) * dip1el[o_orb, v_orb, :]
    # <OS1|H|HL2> = 0
    # <OS1|H|ZHL1> = 0
    # <OS1|H|ZHL2> = 0

    
    # <ZW-|H|HL1> = 0
    # <ZW-|H|HL2> = 0
    # <ZW-|H|ZHL1> 
    for col in range(2*npairs, 3*npairs):
        o_orb = (col - 2*npairs) // nvirt
        v_orb = (col - 2*npairs) % nvirt + (SOMO2 + 1)
        TDM[1,col,:] = - dip1el[o_orb, v_orb, :]
    # <ZW-|H|ZHL2>
    for col in range(3*npairs, 4*npairs):
        o_orb = (col - 3*npairs) // nvirt
        v_orb = (col - 3*npairs) % nvirt + (SOMO2 + 1)
        TDM[1,col,:] = dip1el[o_orb, v_orb, :]
    
    # <ZW+|H|HL1> = 0
    # <ZW+|H|HL2> = 0
    # <ZW+|H|ZHL1> 
    for col in range(2*npairs, 3*npairs):
        o_orb = (col - 2*npairs) // nvirt
        v_orb = (col - 2*npairs) % nvirt + (SOMO2 + 1)
        TDM[2,col,:] = - dip1el[o_orb, v_orb, :]
    # <ZW+|H|ZHL2>
    for col in range(3*npairs, 4*npairs):
        o_orb = (col - 3*npairs) // nvirt
        v_orb = (col - 3*npairs) % nvirt + (SOMO2 + 1)
        TDM[2,col,:] = - dip1el[o_orb, v_orb, :]

    
    row_index = 3
    for row in range(row_index, row_index + ndocc):
        o_orb1 = row - row_index
        # <CS0|H|HL1>
        for col in range(0, npairs):
            o_orb2 = col // nvirt
            v_orb = col % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                TDM[row,col,:] = 1 / np.sqrt(2) * dip1el[SOMO1, v_orb, :]
        # <CS0|H|HL2>
        for col in range(npairs, 2 * npairs):
            o_orb2 = (col - npairs) // nvirt
            v_orb = (col - npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                TDM[row, col, :] = - np.sqrt(1.5) * dip1el[SOMO1, v_orb, :]
        # <CS0|H|ZHL1>
        for col in range(2*npairs, 3*npairs):
            o_orb2 = (col - 2*npairs) // nvirt
            v_orb = (col - 2*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                TDM[row, col, :] = - dip1el[SOMO2, v_orb, :]
        # <CS0|H|ZHL2> = 0
    
    row_index = ndocc + 3
    for row in range(row_index, row_index + ndocc):
        o_orb1 = row - row_index
        # <CS0'|H|HL1>
        for col in range(0, npairs):
            o_orb2 = col // nvirt
            v_orb = col % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                TDM[row, col, :] = - (1 / np.sqrt(2)) * dip1el[SOMO2, v_orb, :]
        # <CS0'|H|HL2>
        for col in range(npairs, 2 * npairs):
            o_orb2 = (col - npairs) // nvirt
            v_orb = (col - npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                TDM[row, col, :] = - np.sqrt(1.5) * dip1el[SOMO2, v_orb, :]
        # <CS0'|H|ZHL1> = 0
        # <CS0'|H|ZHL2>
        for col in range(3*npairs, 4*npairs):
            o_orb2 = (col - 3*npairs) // nvirt
            v_orb = (col - 3*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                TDM[row, col, :] = dip1el[SOMO1, v_orb, :]
    
    row_index = 2 * ndocc + 3
    for row in range(row_index, row_index + nvirt):
        v_orb1 = row - row_index + (SOMO2 + 1)
        # <SV0|H|HL1>
        for col in range(0, npairs):
            o_orb = col // nvirt
            v_orb2 = col % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                TDM[row, col, :] = (1 / np.sqrt(2)) * dip1el[SOMO1, o_orb, :]
        # <SV0|H|HL2>
        for col in range(npairs, 2*npairs):
            o_orb = (col - npairs) // nvirt
            v_orb2 = (col - npairs) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                TDM[row, col, :] = np.sqrt(1.5) * dip1el[SOMO1, o_orb, :]
        # <SV0|H|ZHL1> = 0
        # <SV0|H|ZHL2>
        for col in range(3*npairs, 4*npairs):
            o_orb = (col - 3*npairs) // nvirt
            v_orb2 = (col - 3*npairs) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                TDM[row, col, :] = dip1el[o_orb, SOMO2, :]
                
    row_index = 2 * ndocc + nvirt + 3
    for row in range(row_index, row_index + nvirt):
        v_orb1 = row - row_index + (SOMO2 + 1)
        # <SV0'|H|HL1>
        for col in range(0, npairs):
            o_orb = col // nvirt
            v_orb2 = col % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                TDM[row, col, :] = - (1 / np.sqrt(2)) * dip1el[SOMO2, o_orb, :]
        # <SV0'|H|HL2>
        for col in range(npairs, 2*npairs):
            o_orb = (col - npairs) // nvirt
            v_orb2 = (col - npairs) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                TDM[row, col, :] = np.sqrt(1.5) * dip1el[SOMO2, o_orb, :]
        # <SV0'|H|ZHL1>
        for col in range(2*npairs, 3*npairs):
            o_orb = (col - 2*npairs) // nvirt
            v_orb2 = (col - 2*npairs) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                TDM[row, col, :] = - dip1el[o_orb, SOMO1, :]
        # <SV0'|H|ZHL2> = 0
    
    row_index = 2 * ndocc + 2 * nvirt + 3
    for row in range(row_index, row_index + npairs):
        o_orb1 = (row - row_index) // nvirt
        v_orb1 = (row - row_index) % nvirt + (SOMO2 + 1)
        # <HL1|H|HL1>
        for col in range(row - row_index, npairs):
            o_orb2 = col // nvirt
            v_orb2 = col % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                TDM[row,col,:] = ndocc_sum
                TDM[row,col,:] += dip1el[o_orb1, o_orb1, :]
                TDM[row,col,:] -= dip1el[SOMO1, SOMO1, :]
                TDM[row,col,:] -= dip1el[SOMO2, SOMO2, :]
                TDM[row,col,:] -= dip1el[v_orb1, v_orb1, :]
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                TDM[row, col, :] =  dip1el[o_orb1, o_orb2, :]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                TDM[row, col, :] = -dip1el[v_orb1, v_orb2, :]
        #<HL1|H|HL2> = 0
        #<HL1|H|ZHL1>
        for col in range(2*npairs, 3*npairs):
            o_orb2 = (col - 2*npairs) // nvirt
            v_orb2 = (col - 2*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                TDM[row, col, :] = - np.sqrt(2) * dip1el[SOMO1, SOMO2, :]
        #<HL1|H|ZHL2>
        for col in range(3*npairs, 4*npairs):
            o_orb2 = (col - 3*npairs) // nvirt
            v_orb2 = (col - 3*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                TDM[row, col, :] = - np.sqrt(2) * dip1el[SOMO1, SOMO2, :]
    
    row_index = npairs + 2 * ndocc + 2 * nvirt + 3
    for row in range(row_index, row_index + npairs):
        o_orb1 = (row - row_index) // nvirt
        v_orb1 = (row - row_index) % nvirt + (SOMO2 + 1)
        # <HL2|H|HL2>
        for col in range(row - row_index + npairs, 2*npairs):
            o_orb2 = (col - npairs) // nvirt
            v_orb2 = (col - npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                TDM[row,col,:] = ndocc_sum
                TDM[row,col,:] += dip1el[o_orb1, o_orb1, :]
                TDM[row,col,:] -= dip1el[SOMO1, SOMO1, :]
                TDM[row,col,:] -= dip1el[SOMO2, SOMO2, :]
                TDM[row,col,:] -= dip1el[v_orb1, v_orb1, :]
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                TDM[row, col, :] = dip1el[o_orb1, o_orb2, :]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                TDM[row, col, :] = -dip1el[v_orb1, v_orb2, :]
        #<HL2|H|ZHL1> = 0
        #<HL2|H|ZHL2> = 0

    row_index = 2 * npairs + 2 * ndocc + 2 * nvirt + 3
    for row in range(row_index, row_index + npairs):
        o_orb1 = (row - row_index) // nvirt
        v_orb1 = (row - row_index) % nvirt + (SOMO2 + 1)
        # <ZHL1|H|ZHL1>
        for col in range(row - row_index + 2*npairs, 3*npairs):
            o_orb2 = (col - 2*npairs) // nvirt
            v_orb2 = (col - 2*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                TDM[row,col,:] = ndocc_sum
                TDM[row,col,:] += dip1el[o_orb1, o_orb1, :]
                TDM[row,col,:] -= 2 * dip1el[SOMO1, SOMO1, :]
                TDM[row,col,:] -= dip1el[v_orb1, v_orb1, :]
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                TDM[row, col, :] = dip1el[o_orb1, o_orb2, :]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                TDM[row, col, :] = - dip1el[v_orb1, v_orb2, :]
        #<ZHL1|H|ZHL2> = 0

    row_index = 3 * npairs + 2 * ndocc + 2 * nvirt + 3
    for row in range(row_index, row_index + npairs):
        o_orb1 = (row - row_index) // nvirt
        v_orb1 = (row - row_index) % nvirt + (SOMO2 + 1)
        # <ZHL2|H|ZHL2>
        for col in range(row - row_index + 3*npairs, 4*npairs):
            o_orb2 = (col - 3*npairs) // nvirt
            v_orb2 = (col - 3*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                TDM[row,col,:] = ndocc_sum
                TDM[row,col,:] += dip1el[o_orb1, o_orb1, :]
                TDM[row,col,:] -= 2 * dip1el[SOMO2, SOMO2, :]
                TDM[row,col,:] -= dip1el[v_orb1, v_orb1, :]
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                TDM[row, col, :] = dip1el[o_orb1, o_orb2, :]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                TDM[row, col, :] = - dip1el[v_orb1, v_orb2, :]
    
    return TDM


def build_singlet_D_block(ndocc, norbs, dip1el):
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
    TDM = np.zeros((row_dim, col_dim, 3))  # Initialize CI Block

    ndocc_sum = - 2.0 * dip1el[np.arange(ndocc), np.arange(ndocc), :].sum(axis=0)
    
    # <OS1|H|CSD> = 0
    # <OS1|H|SVD> = 0
    
    # <ZW-|H|CSD> = 0
    # <ZW-|H|SVD> = 0
        
    # <ZW+|H|CSD> = 0
    # <ZW+|H|SVD> = 0

    
    row_index = 3
    for row in range(row_index, row_index + ndocc):
        o_orb1 = row - row_index
        o_orb2 = 0
        o_orb3 = 0
        # <CS0|H|CSD>
        for col in range(0, ndcs):
            if o_orb2 == o_orb3:
                if o_orb1 == o_orb2:
                    TDM[row, col, :] = - np.sqrt(2) * dip1el[o_orb1, SOMO2, :]
            else:
                if o_orb1 == o_orb2:
                    TDM[row, col, :] = - dip1el[o_orb3, SOMO2, :]
                elif o_orb1 == o_orb3:
                    TDM[row, col, :] = - dip1el[o_orb2, SOMO2, :]
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
        # <CS0'|H|CSD>
        for col in range(0, ndcs):
            if o_orb2 == o_orb3:
                if o_orb1 == o_orb2:
                    TDM[row, col, :] = np.sqrt(2) * dip1el[o_orb1, SOMO1, :]
            else:
                if o_orb1 == o_orb2:
                    TDM[row, col, :] = dip1el[o_orb3, SOMO1, :]
                elif o_orb1 == o_orb3:
                    TDM[row, col, :] = dip1el[o_orb2, SOMO1, :]
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
                    TDM[row, col, :] = - np.sqrt(2) * dip1el[SOMO2, v_orb1, :]
            else:
                if v_orb1 == v_orb2:
                    TDM[row, col, :] = - dip1el[SOMO2, v_orb3, :]
                elif v_orb1 == v_orb3:
                    TDM[row, col, :] = - dip1el[SOMO2, v_orb2, :]
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
                    TDM[row, col, :] = np.sqrt(2) * dip1el[SOMO1, v_orb1, :]
            else:
                if v_orb1 == v_orb2:
                    TDM[row, col, :] = dip1el[SOMO1, v_orb3, :]
                elif v_orb1 == v_orb3:
                    TDM[row, col, :] = dip1el[SOMO1, v_orb2, :]
            v_orb3 += 1
            if v_orb3 == norbs:
                v_orb2 += 1
                v_orb3 = v_orb2
    
    # <HL1|H|CSD> = 0
    # <HL1|H|SVD> = 0

    # <HL2|H|CSD> = 0
    # <HL2|H|SLD> = 0

    # <ZHL1|H|CSD> = 0
    # <ZHL1|H|SVD> = 0
    
    # <ZHL2|H|CSD> = 0
    # <ZHL2|H|SVD> = 0
    
    
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
                    TDM[row,col,:] = ndocc_sum
                    TDM[row, col, :] += 2 * dip1el[o_orb1,o_orb1,:]
                    TDM[row, col, :] -= 2 * dip1el[SOMO1, SOMO1, :]
                    TDM[row, col, :] -= 2 * dip1el[SOMO2, SOMO2, :]
            elif o_orb1 == o_orb2 and o_orb3 != o_orb4:
                if o_orb1 == o_orb3:
                    TDM[row, col, :] = np.sqrt(2) * dip1el[o_orb1, o_orb4, :]
                elif o_orb1 == o_orb4:
                    TDM[row, col, :] = np.sqrt(2) * dip1el[o_orb1, o_orb3, :]
            elif o_orb3 == o_orb4 and o_orb1 != o_orb2:
                if o_orb3 == o_orb1:
                    TDM[row, col, :] = np.sqrt(2) * dip1el[o_orb3, o_orb2, :]
                elif o_orb3 == o_orb2:
                    TDM[row, col, :] = np.sqrt(2) * dip1el[o_orb3, o_orb1, :]
            else:
                if o_orb1 == o_orb3 and o_orb2 == o_orb4:
                    TDM[row,col,:] = ndocc_sum
                    TDM[row, col, :] += dip1el[o_orb1,o_orb1,:] + dip1el[o_orb2, o_orb2, :]
                    TDM[row, col, :] -= 2 * dip1el[SOMO1, SOMO1, :]
                    TDM[row, col, :] -= 2 * dip1el[SOMO2, SOMO2, :]
                elif o_orb1 == o_orb3 and o_orb2 != o_orb4:
                    TDM[row,col,:] = dip1el[o_orb2, o_orb4, :]
                elif o_orb2 == o_orb4 and o_orb1 != o_orb3:
                    TDM[row,col,:] = dip1el[o_orb1, o_orb3, :]
                elif o_orb1 == o_orb4 and o_orb2 != o_orb3:
                    TDM[row,col,:] = dip1el[o_orb2, o_orb3, :]
                elif o_orb2 == o_orb3 and o_orb1 != o_orb4:
                    TDM[row,col,:] = dip1el[o_orb1, o_orb4, :]
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
                    TDM[row, col, :] = ndocc_sum
                    TDM[row, col, :] -= 2 * dip1el[v_orb1, v_orb1, :]
            elif v_orb1 == v_orb2 and v_orb3 != v_orb4:
                if v_orb1 == v_orb3:
                    TDM[row, col, :] = - np.sqrt(2) * dip1el[v_orb1, v_orb4, :]
                elif v_orb1 == v_orb4:
                    TDM[row, col, :] = - np.sqrt(2) * dip1el[v_orb1, v_orb3, :]
            elif v_orb1 != v_orb2 and v_orb3 == v_orb4:
                if v_orb3 == v_orb1:
                    TDM[row, col, :] = - np.sqrt(2) * dip1el[v_orb3, v_orb2, :]
                elif v_orb3 == v_orb2:
                    TDM[row, col, :] = - np.sqrt(2) * dip1el[v_orb3, v_orb1, :]
            else:
                if v_orb1 == v_orb3 and v_orb2 == v_orb4:
                    TDM[row, col, :] = ndocc_sum
                    TDM[row, col, :] -= dip1el[v_orb1, v_orb1, :]
                    TDM[row, col, :] -= dip1el[v_orb2, v_orb2, :]
                elif v_orb1 == v_orb3 and v_orb2 != v_orb4:
                    TDM[row, col, :] = - dip1el[v_orb2, v_orb4, :]
                elif v_orb1 != v_orb3 and v_orb2 == v_orb4:
                    TDM[row, col, :] = - dip1el[v_orb1, v_orb3, :]
                elif v_orb1 == v_orb4 and v_orb2 != v_orb3:
                    TDM[row,col,:] = - dip1el[v_orb2, v_orb3, :]
                elif v_orb2 == v_orb3 and v_orb1 != v_orb4:
                    TDM[row,col,:] = - dip1el[v_orb1, v_orb4, :]
            v_orb4 += 1
            if v_orb4 == norbs:
                v_orb3 += 1
                v_orb4 = v_orb3
        v_orb2 += 1
        if v_orb2 == norbs:
            v_orb1 += 1
            v_orb2 = v_orb1
    
    return TDM



def build_triplet_ref_block(ndocc, dip1el):
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
    TDM = np.zeros((1, 1, 3))  # Initialize a 1x1 CI matrix
    SOMO1 = ndocc # Index of SOMO1
    SOMO2 = ndocc + 1 # Index of SOMO2
    
    ndocc_sum = - 2.0 * dip1el[np.arange(ndocc), np.arange(ndocc), :].sum(axis=0)

    # <OS3|H|OS3>
    TDM[0,0,:] = ndocc_sum
    TDM[0,0,:] -= (dip1el[SOMO1,SOMO1,:] + dip1el[SOMO2,SOMO2,:])

    return TDM


def build_triplet_CS_SV_block(ndocc, norbs, dip1el):
    '''
    Function to build the upper diagonal of the CI matrix for singlet reference states for a diradical system - the closed-shell singlet (CS) and the single-reference singlet (SV).
    Args:
        ndocc (int): Number of doubly occupied orbitals
        norbs (int): Total number of orbitals
        dip1el (numpy.ndarray): Dipole integrals for the system
    Returns:
        numpy.ndarray: CI matrix for the diradical system
    '''
    SOMO1 = ndocc # Index of SOMO1
    SOMO2 = ndocc + 1 # Index of SOMO2
    nvirt = norbs - ndocc - 2 # Number of virtual orbitals
    
    row_dim = 2 * ndocc + 2 * nvirt + 1
    col_dim = 2 * ndocc + 2 * nvirt
    TDM = np.zeros((row_dim, col_dim, 3))  # Initialize TDM Block
    
    ndocc_sum = - 2.0 * dip1el[np.arange(ndocc), np.arange(ndocc), :].sum(axis=0)
    
    # <OS3|H|CS0>
    for col in range(0, ndocc):
        o_orb = col
        TDM[0,col,:] = -dip1el[o_orb,SOMO1,:]
    # <OS3|H|CS0'> 
    for col in range(ndocc, 2 * ndocc):
        o_orb = col - ndocc
        TDM[0,col,:] = -dip1el[o_orb,SOMO2,:]
    # <OS3|H|SV0>
    for col in range(2 * ndocc, 2 * ndocc + nvirt):
        v_orb = col - (2 * ndocc) + (SOMO2 + 1)
        TDM[0,col,:] = -dip1el[v_orb,SOMO1,:]
    # <OS3|H|SV0'>
    for col in range(2 * ndocc + nvirt, 2 * ndocc + 2 * nvirt):
        v_orb = col - (2 * ndocc + nvirt) + (SOMO2 + 1)
        TDM[0,col,:] = dip1el[v_orb,SOMO2,:]

    row_index = 1
    for row in range(row_index, row_index + ndocc):
        o_orb1 = row - row_index
        # <CS0|H|CS0>
        for col in range(row - 1, ndocc):
            o_orb2 = col
            if o_orb1 == o_orb2:
                TDM[row,col,:] = ndocc_sum
                TDM[row,col,:] += dip1el[o_orb1, o_orb1, :] # Remove contribution from 1e in HOMO i
                TDM[row,col,:] -= 2 * dip1el[SOMO1, SOMO1, :] # Add contribution from 2e in SOMO1
                TDM[row,col,:] -= dip1el[SOMO2, SOMO2, :] # Add contribution from 1e in SOMO2
            else:    
                TDM[row,col,:] = dip1el[o_orb1, o_orb2, :] 
        # <CS0|H|CS0'>
        for col in range(ndocc, 2*ndocc):
            o_orb2 = col - ndocc
            if o_orb1 == o_orb2:
                TDM[row,col,:] = -dip1el[SOMO1, SOMO2, :]
        # <CS0|H|SV0> = 0
        # <CS0|H|SV0'> = 0
            
    row_index = ndocc + 1
    for row in range(row_index, row_index + ndocc):
        o_orb1 = row - row_index
        # <CS0'|H|CS0'>
        for col in range(row - 1, 2*ndocc):
            o_orb2 = col - ndocc
            if o_orb1 == o_orb2:
                TDM[row,col,:] = ndocc_sum
                TDM[row,col,:] += dip1el[o_orb1, o_orb1, :] # Remove contribution from 1e in HOMO i
                TDM[row,col,:] -= 2 * dip1el[SOMO2, SOMO2, :] # Add contribution from 2e in SOMO2
                TDM[row,col,:] -= dip1el[SOMO1, SOMO1, :] # Add contribution from 1e in SOMO1
            else:    
                TDM[row,col,:] = dip1el[o_orb1, o_orb2, :]
        # <CS0'|H|SV0> = 0
        # <CS0'|H|SV0'> = 0
    
    row_index = 2 * ndocc + 1
    for row in range(row_index, row_index + nvirt):
        v_orb1 = row - row_index + (SOMO2 + 1)
        # <SV0|H|SV0>
        for col in range(row - 1, 2*ndocc + nvirt):
            v_orb2 = col - (2*ndocc) + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                TDM[row,col,:] = ndocc_sum
                TDM[row,col,:] -= dip1el[v_orb1, v_orb1, :] # Add contribution from 1e in LUMO j
                TDM[row,col,:] -= dip1el[SOMO2, SOMO2, :] # Add contribution from 1e in SOMO2
            else:    
                TDM[row,col,:] = -dip1el[v_orb1, v_orb2, :]
        # <SV0|H|SV0'>
        for col in range(2*ndocc + nvirt, 2*ndocc + 2*nvirt):
            v_orb2 = col - (2*ndocc + nvirt) + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                TDM[row,col,:] = -dip1el[SOMO1,SOMO2,:]
    
    row_index = 2 * ndocc + nvirt + 1
    for row in range(row_index, row_index + nvirt):
        v_orb1 = row - row_index + (SOMO2 + 1)
        # <SV0'|H|SV0'>
        for col in range(row - 1, 2*ndocc + 2*nvirt):
            v_orb2 = col - (2*ndocc + nvirt) + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                TDM[row,col,:] = ndocc_sum
                TDM[row,col,:] -= dip1el[v_orb1, v_orb1, :] # Add contribution from 1e in LUMO j
                TDM[row,col,:] -= dip1el[SOMO1, SOMO1, :] # Add contribution from 1e in SOMO1
            else:    
                TDM[row,col,:] = -dip1el[v_orb1, v_orb2, :]

    return TDM


def build_triplet_HL_block(ndocc, norbs, dip1el):
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
    TDM = np.zeros((row_dim, col_dim, 3))  # Initialize CI Block
    
    ndocc_sum = - 2.0 * dip1el[np.arange(ndocc), np.arange(ndocc), :].sum(axis=0)

    # <OS3|H|HL1>
    for col in range(0, npairs):
        o_orb = col // nvirt
        v_orb = col % nvirt + (SOMO2 + 1)
        TDM[0, col, :] = - np.sqrt(2) * dip1el[o_orb, v_orb, :]
    # <OS3|H|HL2> = 0
    # <OS3|H|HL3> = 0
    # <OS3|H|ZHL1> = 0
    # <OS3|H|ZHL2> = 0


    row_index = 1
    for row in range(row_index, row_index + ndocc):
        o_orb1 = row - row_index
        # <CS0|H|HL1>
        for col in range(0, npairs):
            o_orb2 = col // nvirt
            v_orb = col % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                TDM[row, col, :] = - (1 / np.sqrt(2)) * dip1el[SOMO1, v_orb, :]
        # <CS0|H|HL2>
        for col in range(npairs, 2*npairs):
            o_orb2 = (col - npairs) // nvirt
            v_orb = (col - npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                TDM[row, col, :] = - (1 / np.sqrt(2)) * dip1el[SOMO1, v_orb, :]
        # <CS0|H|HL3>
        for col in range(2*npairs, 3*npairs):
            o_orb2 = (col - 2*npairs) // nvirt
            v_orb = (col - 2*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                TDM[row, col, :] = - dip1el[SOMO1, v_orb, :]
        # <CS0|H|ZHL1>
        for col in range(3*npairs, 4*npairs):
            o_orb2 = (col - 3*npairs) // nvirt
            v_orb = (col - 3*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                TDM[row, col, :] =  dip1el[SOMO2, v_orb, :]
        # <CS0|H|ZHL2> = 0
    
    row_index = ndocc + 1
    for row in range(row_index, row_index + ndocc):
        o_orb1 = row - row_index
        # <CS0'|H|HL1>
        for col in range(0, npairs):
            o_orb2 = col // nvirt
            v_orb = col % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                TDM[row, col, :] = - (1 / np.sqrt(2)) * dip1el[SOMO2, v_orb, :]
        # <CS0'|H|HL2>
        for col in range(npairs, 2 * npairs):
            o_orb2 = (col - npairs) // nvirt
            v_orb = (col - npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                TDM[row, col, :] = (1 / np.sqrt(2)) * dip1el[SOMO2, v_orb, :]
        # <CS0'|H|HL3>
        for col in range(2*npairs, 3*npairs):
            o_orb2 = (col - 2*npairs) // nvirt
            v_orb = (col - 2*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                TDM[row, col, :] = - dip1el[SOMO2, v_orb, :]
        # <CS0'|H|ZHL1> = 0
        # <CS0'|H|ZHL2>
        for col in range(4*npairs, 5*npairs):
            o_orb2 = (col - 4*npairs) // nvirt
            v_orb = (col - 4*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                TDM[row, col, :] = - dip1el[SOMO1, v_orb, :]
    
    row_index = 2 * ndocc + 1
    for row in range(row_index, row_index + nvirt):
        v_orb1 = row - row_index + (SOMO2 + 1)
        # <SV0|H|HL1>
        for col in range(0, npairs):
            o_orb = col // nvirt
            v_orb2 = col % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                TDM[row, col, :] = (1 / np.sqrt(2)) * dip1el[SOMO1, o_orb, :]
        # <SV0|H|HL2>
        for col in range(npairs, 2*npairs):
            o_orb = (col - npairs) // nvirt
            v_orb2 = (col - npairs) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                TDM[row, col, :] = - (1 / np.sqrt(2)) * dip1el[SOMO1, o_orb, :]
        # <SV0|H|HL3>
        for col in range(2*npairs, 3*npairs):
            o_orb = (col - 2*npairs) // nvirt
            v_orb2 = (col - 2*npairs) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                TDM[row, col, :] = - dip1el[SOMO1, o_orb, :]
        # <SV0|H|ZHL1> = 0
        # <SV0|H|ZHL2>
        for col in range(4*npairs, 5*npairs):
            o_orb = (col - 4*npairs) // nvirt
            v_orb2 = (col - 4*npairs) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                TDM[row, col, :] = - dip1el[SOMO2, o_orb, :]
                
    row_index = 2 * ndocc + nvirt + 1
    for row in range(row_index, row_index + nvirt):
        v_orb1 = row - row_index + (SOMO2 + 1)
        # <SV0'|H|HL1>
        for col in range(0, npairs):
            o_orb = col // nvirt
            v_orb2 = col % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                TDM[row, col, :] = - (1 / np.sqrt(2)) * dip1el[SOMO2, o_orb, :]
        # <SV0'|H|HL2>
        for col in range(npairs, 2*npairs):
            o_orb = (col - npairs) // nvirt
            v_orb2 = (col - npairs) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                TDM[row, col, :] = - (1 / np.sqrt(2)) * dip1el[SOMO2, o_orb, :]
        # <SV0'|H|HL3>
        for col in range(2*npairs, 3*npairs):
            o_orb = (col - 2*npairs) // nvirt
            v_orb2 = (col - 2*npairs) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                TDM[row, col, :] = dip1el[SOMO2, o_orb, :]
        # <SV0'|H|ZHL1>
        for col in range(3*npairs, 4*npairs):
            o_orb = (col - 3*npairs) // nvirt
            v_orb2 = (col - 3*npairs) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                TDM[row, col, :] = - dip1el[SOMO1, o_orb, :]
        # <SV0'|H|ZHL2> = 0
    
    
    row_index = 2 * ndocc + 2 * nvirt + 1
    for row in range(row_index, row_index + npairs):
        o_orb1 = (row - row_index) // nvirt
        v_orb1 = (row - row_index) % nvirt + (SOMO2 + 1)
        # <HL1|H|HL1>
        for col in range(row - row_index, npairs):
            o_orb2 = col // nvirt
            v_orb2 = col % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                TDM[row,col,:] = ndocc_sum
                TDM[row,col,:] += dip1el[o_orb1, o_orb1, :] # Remove contribution from 1e in HOMO i
                TDM[row,col,:] -= dip1el[SOMO1, SOMO1, :] # Add contribution from 1e in SOMO1
                TDM[row,col,:] -= dip1el[SOMO2, SOMO2, :] # Add contribution from 1e in SOMO2
                TDM[row,col,:] -= dip1el[v_orb1, v_orb1, :]
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                TDM[row, col, :] = dip1el[o_orb1, o_orb2, :]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                TDM[row, col, :] = - dip1el[v_orb1, v_orb2, :]

        # <HL1|H|HL2> = 0
        # <HL1|H|HL3> = 0
        # <HL1|H|ZHL1> = 0
        # <HL1|H|ZHL2> = 0
    
    row_index = npairs + 2 * ndocc + 2 * nvirt + 1
    for row in range(row_index, row_index + npairs):
        o_orb1 = (row - row_index) // nvirt
        v_orb1 = (row - row_index) % nvirt + (SOMO2 + 1)
        # <HL2|H|HL2>
        for col in range(row - row_index + npairs, 2*npairs):
            o_orb2 = (col - npairs) // nvirt
            v_orb2 = (col - npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                TDM[row,col,:] = ndocc_sum
                TDM[row,col,:] += dip1el[o_orb1, o_orb1, :] # Remove contribution from 1e in HOMO i
                TDM[row,col,:] -= dip1el[SOMO1, SOMO1, :] # Add contribution from 1e in SOMO1
                TDM[row,col,:] -= dip1el[SOMO2, SOMO2, :] # Add contribution from 1e in SOMO2
                TDM[row,col,:] -= dip1el[v_orb1, v_orb1, :]
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                TDM[row, col, :] = dip1el[o_orb1, o_orb2, :]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                TDM[row, col, :] = - dip1el[v_orb1, v_orb2, :]
        # <HL2|H|HL3> = 0
        #<HL2|H|ZHL1>
        for col in range(3*npairs, 4*npairs):
            o_orb2 = (col - 3*npairs) // nvirt
            v_orb2 = (col - 3*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                TDM[row, col, :] = - np.sqrt(2) * dip1el[SOMO1, SOMO2, :]
        #<HL2|H|ZHL2>
        for col in range(4*npairs, 5*npairs):
            o_orb2 = (col - 4*npairs) // nvirt
            v_orb2 = (col - 4*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                TDM[row, col, :] = - np.sqrt(2) * dip1el[SOMO1, SOMO2, :]
                
    row_index = 2 * npairs + 2 * ndocc + 2 * nvirt + 1
    for row in range(row_index, row_index + npairs):
        o_orb1 = (row - row_index) // nvirt
        v_orb1 = (row - row_index) % nvirt + (SOMO2 + 1)
        # <HL3|H|HL3>
        for col in range(row - row_index + 2*npairs, 3*npairs):
            o_orb2 = (col - 2*npairs) // nvirt
            v_orb2 = (col - 2*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                TDM[row,col,:] = ndocc_sum
                TDM[row,col,:] += dip1el[o_orb1, o_orb1, :] # Remove contribution from 1e in HOMO i
                TDM[row,col,:] -= dip1el[SOMO1, SOMO1, :] # Add contribution from 1e in SOMO1
                TDM[row,col,:] -= dip1el[SOMO2, SOMO2, :] # Add contribution from 1e in SOMO2
                TDM[row,col,:] -= dip1el[v_orb1, v_orb1, :]
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                TDM[row, col, :] = dip1el[o_orb1, o_orb2, :]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                TDM[row, col, :] = - dip1el[v_orb1, v_orb2, :]
        #<HL3|H|ZHL1> = 0
        #<HL3|H|ZHL2> = 0

    row_index = 3 * npairs + 2 * ndocc + 2 * nvirt + 1
    for row in range(row_index, row_index + npairs):
        o_orb1 = (row - row_index) // nvirt
        v_orb1 = (row - row_index) % nvirt + (SOMO2 + 1)
        # <ZHL1|H|ZHL1>
        for col in range(row - row_index + 3*npairs, 4*npairs):
            o_orb2 = (col - 3*npairs) // nvirt
            v_orb2 = (col - 3*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                TDM[row,col,:] = ndocc_sum
                TDM[row,col,:] += dip1el[o_orb1, o_orb1, :] # Remove contribution from 1e in HOMO i
                TDM[row,col,:] -= 2 * dip1el[SOMO1, SOMO1, :] # Add contribution from 1e in SOMO1
                TDM[row,col,:] -= dip1el[v_orb1, v_orb1, :]
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                TDM[row, col, :] = dip1el[o_orb1, o_orb2, :]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                TDM[row, col, :] = - dip1el[v_orb1, v_orb2, :]
        #<ZHL1|H|ZHL2> = 0

    row_index = 4 * npairs + 2 * ndocc + 2 * nvirt + 1
    for row in range(row_index, row_index + npairs):
        o_orb1 = (row - row_index) // nvirt
        v_orb1 = (row - row_index) % nvirt + (SOMO2 + 1)
        # <ZHL2|H|ZHL2>
        for col in range(row - row_index + 4*npairs, 5*npairs):
            o_orb2 = (col - 4*npairs) // nvirt
            v_orb2 = (col - 4*npairs) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                TDM[row,col,:] = ndocc_sum
                TDM[row,col,:] += dip1el[o_orb1, o_orb1, :] # Remove contribution from 1e in HOMO i
                TDM[row,col,:] -= 2 * dip1el[SOMO2, SOMO2, :] # Add contribution from 1e in SOMO1
                TDM[row,col,:] -= dip1el[v_orb1, v_orb1, :]
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                TDM[row, col, :] = dip1el[o_orb1, o_orb2, :]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                TDM[row, col, :] = - dip1el[v_orb1, v_orb2, :]
    
    return TDM

def build_triplet_D_block(ndocc, norbs, dip1el):
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
    TDM = np.zeros((row_dim, col_dim, 3))  # Initialize CI Block

    # <OS3|H|CSD> = 0
    # <OS1|H|SVD> = 0
    
    ndocc_sum = - 2.0 * dip1el[np.arange(ndocc), np.arange(ndocc), :].sum(axis=0)

    
    row_index = 1
    for row in range(row_index, row_index + ndocc):
        o_orb1 = row - row_index
        o_orb2 = 0
        o_orb3 = 1
        # <CS0|H|CSD>
        for col in range(0, ndcs):
            if o_orb1 == o_orb2:
                TDM[row, col, :] = - dip1el[o_orb3, SOMO2, :]
            elif o_orb1 == o_orb3:
                TDM[row, col, :] = dip1el[o_orb2, SOMO2, :]
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
        # <CS0'|H|CSD>
        for col in range(0, ndcs):
            if o_orb1 == o_orb2:
                TDM[row, col, :] = dip1el[o_orb3, SOMO1, :]
            elif o_orb1 == o_orb3:
                TDM[row, col, :] = - dip1el[o_orb2, SOMO1, :]
            o_orb3 += 1
            if o_orb3 >= ndocc:
                o_orb2 += 1
                o_orb3 = o_orb2 + 1
        # <CS0'|H|SVD> = 0
    
    row_index = 2 * ndocc + 1
    for row in range(row_index, row_index + nvirt):
        # <SV0|H|CSD> = 0
        v_orb1 = row - row_index + (SOMO2 + 1)
        v_orb2 = SOMO2 + 1
        v_orb3 = SOMO2 + 2
        # <SV0|H|SVD>
        for col in range(ndcs, ndcs + ndsv):
            if v_orb1 == v_orb2:
                TDM[row, col, :] = - dip1el[v_orb3, SOMO2, :]
            elif v_orb1 == v_orb3:
                TDM[row, col, :] = dip1el[v_orb2, SOMO2, :]
            v_orb3 += 1
            if v_orb3 >= norbs:
                v_orb2 += 1
                v_orb3 = v_orb2 + 1

                
    row_index = 2 * ndocc + nvirt + 1
    for row in range(row_index, row_index + nvirt):
        # <SV0'|H|CSD> = 0
        v_orb1 = row - row_index + (SOMO2 + 1)
        v_orb2 = SOMO2 + 1
        v_orb3 = SOMO2 + 2
        # <SV0'|H|SVD>
        for col in range(ndcs, ndcs + ndsv):
            if v_orb1 == v_orb2:
                TDM[row, col, :] = - dip1el[v_orb3, SOMO1, :]
            elif v_orb1 == v_orb3:
                TDM[row, col, :] = dip1el[v_orb2, SOMO1, :]
            v_orb3 += 1
            if v_orb3 >= norbs:
                v_orb2 += 1
                v_orb3 = v_orb2 + 1
    
    # <HL1|H|CSD> = 0
    # <HL1|H|SVD> = 0
    
    # <HL2|H|CSD> = 0
    # <HL2|H|SLD> = 0

    # <HL3|H|CSD> = 0
    # <HL3|H|SLD> = 0

    # <ZHL1|H|CSD> = 0
    # <ZHL1|H|SVD> = 0

    # <ZHL2|H|CSD> = 0
    # <ZHL2|H|SVD> = 0
    
    
    row_index = 5 * npairs + 2 * ndocc + 2 * nvirt + 1
    o_orb1 = 0
    o_orb2 = 1
    for row in range(row_index, row_index + ndcs):
        o_orb3 = o_orb1
        o_orb4 = o_orb2
        # <CSD|H|CSD>
        for col in range(row - row_index, ndcs):
            if o_orb1 == o_orb3 and o_orb2 == o_orb4:
                TDM[row, col, :] = ndocc_sum
                TDM[row, col, :] += dip1el[o_orb1,o_orb1,:] + dip1el[o_orb2,o_orb2,:]
                TDM[row, col, :] -= 2 * dip1el[SOMO1,SOMO1,:]
                TDM[row, col, :] -= 2 * dip1el[SOMO2,SOMO2,:]
            elif o_orb1 == o_orb3 and o_orb2 != o_orb4:
                TDM[row,col,:] = dip1el[o_orb2, o_orb4, :]
            elif o_orb2 == o_orb4 and o_orb1 != o_orb3:
                TDM[row,col,:] = dip1el[o_orb1, o_orb3, :]
            elif o_orb2 == o_orb3 and o_orb1 != o_orb4:
                TDM[row,col,:] = - dip1el[o_orb1, o_orb4, :]
            o_orb4 += 1
            if o_orb4 >= ndocc:
                o_orb3 += 1
                o_orb4 = o_orb3 + 1
        # <CSD|H|SVD> = 0
        o_orb2 += 1
        if o_orb2 >= ndocc:
            o_orb1 += 1
            o_orb2 = o_orb1 + 1
    
    row_index = ndcs + 5 * npairs + 2 * ndocc + 2 * nvirt + 1
    v_orb1 = SOMO2 + 1
    v_orb2 = SOMO2 + 2
    for row in range(row_index, row_index + ndsv):
        v_orb3 = v_orb1
        v_orb4 = v_orb2
        # <SVD|H|SVD>
        for col in range(row - row_index + ndcs, ndcs + ndsv):
            if v_orb1 == v_orb3 and v_orb2 == v_orb4:
                TDM[row, col, :] = ndocc_sum
                TDM[row, col, :] -= dip1el[v_orb1, v_orb1, :]
                TDM[row, col, :] -= dip1el[v_orb2, v_orb2, :]
            elif v_orb1 == v_orb3 and v_orb2 != v_orb4:
                TDM[row, col, :] = - dip1el[v_orb2, v_orb4, :]
            elif v_orb1 != v_orb3 and v_orb2 == v_orb4:
                TDM[row, col, :] = - dip1el[v_orb1, v_orb3, :]
            elif v_orb2 == v_orb3 and v_orb1 != v_orb4:
                TDM[row, col, :] = dip1el[v_orb1, v_orb4, :] 
            v_orb4 += 1
            if v_orb4 >= norbs:
                v_orb3 += 1
                v_orb4 = v_orb3 + 1
        v_orb2 += 1
        if v_orb2 >= norbs:
            v_orb1 += 1
            v_orb2 = v_orb1 + 1
    
    return TDM

def build_quintet_block(ndocc, norbs, dip1el):
    
    SOMO1 = ndocc # Index of SOMO1
    SOMO2 = ndocc + 1 # Index of SOMO2
    nvirt = norbs - ndocc - 2 # Number of virtual orbitals
    npairs = ndocc * nvirt
    
    TDM = np.zeros((npairs, npairs, 3))  # Initialize CI Block

    for row in range(0, npairs):
        o_orb1 = row // nvirt
        v_orb1 = row % nvirt + (SOMO2 + 1)
        # <HL1|H|HL1>
        for col in range(row, npairs):
            o_orb2 = col // nvirt
            v_orb2 = col % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                for o in range(ndocc):
                    TDM[row,col,:] -= 2 * dip1el[o,o,:]
                TDM[row,col,:] += dip1el[o_orb1, o_orb1, :] # Remove contribution from 1e in HOMO i
                TDM[row,col,:] -= dip1el[SOMO1, SOMO1, :] # Add contribution from 1e in SOMO1
                TDM[row,col,:] -= dip1el[SOMO2, SOMO2, :] # Add contribution from 1e in SOMO2
                TDM[row,col,:] -= dip1el[v_orb1, v_orb1, :]
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                TDM[row, col, :] = dip1el[o_orb1, o_orb2, :]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                TDM[row, col, :] = - dip1el[v_orb1, v_orb2, :]
            TDM[col, row, :] = TDM[row, col, :]
    
    return TDM


def build_singlet_TDM(ndocc, norbs, dip1el, ci_level):
    
    if ci_level == 0:
        Singlet_TDM = build_singlet_ref_block(ndocc, dip1el)
        
    elif ci_level == 1:
        ref_block = build_singlet_ref_block(ndocc, dip1el)
        cs_sv_block = build_singlet_CS_SV_block(ndocc, norbs, dip1el)
        
        Singlet_TDM = np.zeros((cs_sv_block.shape[0], cs_sv_block.shape[0], 3))
        Singlet_TDM[:ref_block.shape[0], :ref_block.shape[1], :] = ref_block
        Singlet_TDM[:, ref_block.shape[1]:, :] = cs_sv_block
    
    elif ci_level == 2:
        ref_block = build_singlet_ref_block(ndocc, dip1el)
        cs_sv_block = build_singlet_CS_SV_block(ndocc, norbs, dip1el)
        hl_block = build_singlet_HL_block(ndocc, norbs, dip1el)
        
        Singlet_TDM = np.zeros((hl_block.shape[0], hl_block.shape[0], 3))
        Singlet_TDM[:ref_block.shape[0], :ref_block.shape[1], :] = ref_block
        Singlet_TDM[:cs_sv_block.shape[0], ref_block.shape[1]:(ref_block.shape[1]+cs_sv_block.shape[1]), :] = cs_sv_block
        Singlet_TDM[:hl_block.shape[0], (ref_block.shape[1]+cs_sv_block.shape[1]):, :] = hl_block
    
    elif ci_level == 3:
        ref_block = build_singlet_ref_block(ndocc, dip1el)
        cs_sv_block = build_singlet_CS_SV_block(ndocc, norbs, dip1el)
        hl_block = build_singlet_HL_block(ndocc, norbs, dip1el)
        d_block = build_singlet_D_block(ndocc, norbs, dip1el)
        
        Singlet_TDM = np.zeros((d_block.shape[0], d_block.shape[0], 3))
        Singlet_TDM[:ref_block.shape[0], :ref_block.shape[1], :] = ref_block
        Singlet_TDM[:cs_sv_block.shape[0], ref_block.shape[1]:(ref_block.shape[1]+cs_sv_block.shape[1]), :] = cs_sv_block
        Singlet_TDM[:hl_block.shape[0], (ref_block.shape[1]+cs_sv_block.shape[1]):(ref_block.shape[1]+cs_sv_block.shape[1]+hl_block.shape[1]), :] = hl_block
        Singlet_TDM[:, (ref_block.shape[1]+cs_sv_block.shape[1]+hl_block.shape[1]):, :] = d_block
    
    
    idx = np.arange(Singlet_TDM.shape[0])
    Singlet_TDM_sym = Singlet_TDM + Singlet_TDM.transpose(1,0,2)
    Singlet_TDM_sym[idx, idx, :] = Singlet_TDM[idx, idx, :]
    
    return Singlet_TDM_sym


def build_triplet_TDM(ndocc, norbs, dip1el, ci_level):
    '''
    Function to build the Triplet TDM from a defined number of blocks
    '''
    if ci_level == 0:
        return build_triplet_ref_block(ndocc, dip1el)
    elif ci_level == 1:
        ref_block = build_triplet_ref_block(ndocc, dip1el)
        cs_sv_block = build_triplet_CS_SV_block(ndocc, norbs, dip1el)
        
        Triplet_TDM = np.zeros((cs_sv_block.shape[0], cs_sv_block.shape[0], 3))
        Triplet_TDM[:ref_block.shape[0], :ref_block.shape[1], :] = ref_block
        Triplet_TDM[:, ref_block.shape[1]:, :] = cs_sv_block
    
    elif ci_level == 2:
        ref_block = build_triplet_ref_block(ndocc, dip1el)
        cs_sv_block = build_triplet_CS_SV_block(ndocc, norbs, dip1el)
        hl_block = build_triplet_HL_block(ndocc, norbs, dip1el)
        
        Triplet_TDM = np.zeros((hl_block.shape[0], hl_block.shape[0], 3))
        Triplet_TDM[:ref_block.shape[0], :ref_block.shape[1], :] = ref_block
        Triplet_TDM[:cs_sv_block.shape[0], ref_block.shape[1]:(ref_block.shape[1]+cs_sv_block.shape[1]), :] = cs_sv_block
        Triplet_TDM[:hl_block.shape[0], (ref_block.shape[1]+cs_sv_block.shape[1]):, :] = hl_block
    
    elif ci_level == 3:
        ref_block = build_triplet_ref_block(ndocc, dip1el)
        cs_sv_block = build_triplet_CS_SV_block(ndocc, norbs, dip1el)
        hl_block = build_triplet_HL_block(ndocc, norbs, dip1el)
        d_block = build_triplet_D_block(ndocc, norbs, dip1el)

        Triplet_TDM = np.zeros((d_block.shape[0], d_block.shape[0], 3))
        Triplet_TDM[:ref_block.shape[0], :ref_block.shape[1], :] = ref_block
        Triplet_TDM[:cs_sv_block.shape[0], ref_block.shape[1]:(ref_block.shape[1]+cs_sv_block.shape[1]), :] = cs_sv_block
        Triplet_TDM[:hl_block.shape[0], (ref_block.shape[1]+cs_sv_block.shape[1]):(ref_block.shape[1]+cs_sv_block.shape[1]+hl_block.shape[1]), :] = hl_block
        Triplet_TDM[:, (ref_block.shape[1]+cs_sv_block.shape[1]+hl_block.shape[1]):, :] = d_block
    
    idx = np.arange(Triplet_TDM.shape[0])
    Triplet_TDM_sym = Triplet_TDM + Triplet_TDM.transpose(1,0,2)
    Triplet_TDM_sym[idx, idx, :] = Triplet_TDM[idx, idx, :]
    
    return Triplet_TDM_sym

def get_full_TDM(ndocc, norbs, coords, hf_orbs, ci_level):
    '''
    Function to build the TDM from a number of excitation blocks given by ci_level.
    ci_level = 0 -> Only the reference block is included
    ci_level = 1 -> The reference block and the CS/SV block
    ci_level = 2 -> The reference block, the CS/SV block, and the CV block
    ci_level = 3 -> The reference block, the CS/SV block, the CV block, and the double CS/ double SV block
    '''
    dip1el = cartesian_operators(coords, hf_orbs)[0]
    singlet_block = build_singlet_TDM(ndocc, norbs, dip1el, ci_level)
    triplet_block = build_triplet_TDM(ndocc, norbs, dip1el, ci_level)
    singlet_dim = singlet_block.shape[0]
    
    if ci_level < 2: 
        full_dim = singlet_dim + triplet_block.shape[0]
        
        full_TDM = np.zeros((full_dim, full_dim, 3))
        full_TDM[:singlet_dim, :singlet_dim, :] = singlet_block
        full_TDM[singlet_dim:, singlet_dim:, :] = triplet_block
    
    else:
        quintet_block = build_quintet_block(ndocc, norbs, dip1el)
        triplet_dim = triplet_block.shape[0]
        full_dim = singlet_dim + triplet_dim + quintet_block.shape[0]
                
        full_TDM = np.zeros((full_dim, full_dim, 3))
        full_TDM[:singlet_dim, :singlet_dim, :] = singlet_block
        full_TDM[singlet_dim:singlet_dim + triplet_dim, singlet_dim:singlet_dim + triplet_dim, :] = triplet_block
        full_TDM[singlet_dim + triplet_dim:, singlet_dim + triplet_dim:, :] = quintet_block

    return full_TDM, singlet_block, triplet_block