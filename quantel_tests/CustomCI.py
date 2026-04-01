
from quantel import MOintegrals
from quantel.wfn.cisolver import CustomCI
from Diradical_ExROPPP import main_scf
from Diradical_ExROPPP import transform
import numpy as np
import argparse


# TESTING CUSTOM CI SOLVER #

params = [[-22.71707507,   1.70561621 ,  8.42083845  , 1.17315691 ,  0.        ],
          [ -3.486745 ,  -25.23133814 ,  1.76801716,  12.80518166  , 1.20074375],
          [-17.68133786, -24.73720244 ,  1.43363853 , 17.97984271 ,  1.11179102],
          [-10.33567426, -26.02193733 ,  1.45186057 ,  9.64299129 ,  2.25331612]]

parser = argparse.ArgumentParser()
parser.add_argument('geometry', type = str, help = 'file containing geometry')
args = parser.parse_args()
optimized_geometry = args.geometry

def transform_1e(hopping, hf_orbs):
    '''
    Transforms a 1-body hopping matrix from the atomic orbital (AO) basis 
    to the molecular orbital (MO) basis.
    
    Args:
        hopping_ao: 2D array of hopping integrals in the AO basis. 
                    Shape (Natoms, Natoms)
        hf_orbs:    2D array of Hartree-Fock orbital coefficients (C matrix). 
                    Shape (Natoms, N_MO)
    
    Returns:
        hopping_mo: 2D array of hopping integrals in the MO basis. 
                    Shape (N_MO, N_MO)
    '''
    # Using einsum to stay consistent with your ERI function style:
    # i, j are AO indices; a, b are MO indices.
    # We transform i -> a and j -> b
    hopping_mo = np.einsum('ia, jb, ij -> ab', 
                           hf_orbs, hf_orbs, hopping, optimize='optimal')
    
    return hopping_mo



if __name__ == "__main__":
    
    coord,atoms_array,coord_w_h,dist_array,nelec,ndocc,n_list,natoms_c,natoms_n,natoms_cl,energy2,hopping,repulsion,evals,orbs,fock_mat = main_scf(optimized_geometry, params)
    
    print("\n--------------------------")
    print("Converged ROPPP Orbitals")
    print("--------------------------\n")
    natoms=np.shape(coord)[0]
    for iorb in range(natoms):
        print('orbital number', iorb + 1, 'energy', evals[iorb]-evals[int((nelec-1)/2)])
        print(np.around(orbs[:, iorb], decimals=2))
    print("--------------------------\n")
    
    
    nmo = 2 * ndocc + 2
    nalpha = ndocc + 1
    nbeta = ndocc + 1
 
    Vscalar = 0
    fac = 1 / np.sqrt(2)

    # h1e[a,b] = <psi_a|h1|psi_b>
    hopping_mo = transform_1e(hopping, orbs)
    h1e = hopping_mo

    # h2e_ab[i,j,k,l] = <ij|kl> 
    repulsion_mo = transform(repulsion, orbs) # 2e integrals in MO basis but in chemist's notation
    print("(10|01')", repulsion_mo[0,1,1,3])
    print("(10'|0'1')", repulsion_mo[0,2,2,3])
    print("(10|0'1')", repulsion_mo[0,1,2,3])
    print("(10'|01')", repulsion_mo[0,2,1,3])
    h2e = repulsion_mo.transpose(0, 2, 1, 3) # Transform into physicist's notation

    
    # Build MO integral object, which can be passed to CustomCI
    mo_ints = MOintegrals(Vscalar, h1e, h2e, nmo)
    #dip_ints = MOintegrals(0,mux,0,nmo)

    # Setup and solve FCI
    ci = CustomCI(mo_ints, ["2ab0", "2ba0", "2200", "2020", # Reference states
                            "a2b0", "b2a0", "ab20", "ba20", # HOMO to SOMO states
                            "2a0b", "2b0a", "20ab", "20ba", # SOMO to LUMO states
                            "aabb", "abab", "abba", "bbaa", "baba", "baab", # HOMO to LUMO states
                            "0220"], # Double Zwitterion State
                (nalpha,nbeta)) 
    

    ham = ci.get_hamiltonian()
    print(ham.shape)
    print("\nHamiltonian matrix in Determinant Basis:")
    print(ham)
    
    fac1 = 1 / (2*np.sqrt(3))
    fac2 = 1 / np.sqrt(6)
    
    csf_basis = np.zeros((19,19))
    # singlet xy
    csf_basis[[0,1],0] = [fac,fac]
    # x^2 - y^2
    csf_basis[[2,3],1] = [fac,-fac]
    # x^2 + y^2
    csf_basis[[2,3],2] = [fac,fac]
    # Ey
    csf_basis[[4,5],3] = [fac,fac] # HOMO to SOMO1
    csf_basis[[10,11],4] = [fac,fac] # SOMO1 to LUMO
    # Ex
    csf_basis[[6,7],5] = [fac,fac] # HOMO to SOMO2
    csf_basis[[8,9],6] = [fac,fac] # SOMO2 to LUMO
    # Homo to LUMO
    csf_basis[[12,13,15,16],7] = [0.5,0.5,0.5,0.5] # Bright HOMO to LUMO
    csf_basis[[12,13,14,15,16,17],8] = [-fac1,fac1,2*fac1,-fac1,fac1,2*fac1] # Dark HOMO to LUMO
    # Triplet xy
    csf_basis[[0,1],9] = [fac,-fac]
    # Triplet Ey
    csf_basis[[4,5],10] = [fac,-fac] # HOMO to SOMO1
    csf_basis[[10,11],11] = [fac,-fac] # SOMO1 to LUMO
    # Tripley Ex
    csf_basis[[6,7],12] = [fac,-fac] # HOMO to SOMO2
    csf_basis[[8,9],13] = [fac,-fac] # SOMO2 to LUMO
    csf_basis[[12,13,15,16],14] = [0.5,-0.5,-0.5,0.5] # Bright HOMO to LUMO
    csf_basis[[12,13,15,16],15] = [0.5,0.5,-0.5,-0.5] # Dark HOMO to LUMO 1
    csf_basis[[14,17],16] = [-fac,fac] # Dark HOMO to LUMO 2
    # Quintet
    csf_basis[[12,13,14,15,16,17],17] = [fac2,-fac2,fac2,fac2,-fac2,fac2]
    # Double Zwitterion
    csf_basis[18, 18] = 1
    ham_csf = csf_basis.T @ ham @ csf_basis
    #print(ham_csf)
    e,v = np.linalg.eigh(ham_csf[:9,:9])
    #print(e)
    #print(v)
    e,v = np.linalg.eigh(ham_csf[9:18,9:18])
    #print(e)
    #print(v)
    #quit()


    print("\nHamiltonian matrix in CSF Basis:")
    print(ham_csf)
    e,x = np.linalg.eigh(ham_csf)

    for i in range(10):
        x0 = np.copy(x[:,i])
        s2 = ci.get_s2(x0)
        print(f"--------------------------------------------------------------------")
        print(f"State {i}: E = {e[i]: 16.10f}, <S^2> = {s2: 8.6f}")
        print(x0)
        print(f"--------------------------------------------------------------------")
        #ci.cispace.print_vector(x0,1e-6)