import numpy as np
import scipy.optimize as optimize
import scipy.sparse.linalg as sp
import scipy.linalg as linalg
from datetime import datetime
from subprocess import getoutput
import sys
from ExROPPP_settings_opt import *
from SCF import *
from DipBuilder import *
from CIBuilder import *
import os





# parser = argparse.ArgumentParser()
# parser.add_argument('geometry', type = str, help = 'file containing geometry')
# args = parser.parse_args()
# optimized_geometry = args.geometry

# params=[[-22.53982183,   1.70115484 ,  8.47994446 ,  1.17367777,   0.        ],
#  [ -3.25983816 ,-24.50393011 ,  1.76554162 , 13.0486315  ,  1.18938422],
#  [-17.50211252 ,-23.67958463 ,  1.43383449 , 18.08184948 ,  1.12853335],
#  [-10.18396573 ,-26.36242115 ,  1.45855408 ,  9.61199125 ,  2.23245629]] 





def transform(two_body, hf_orbs):
    '''
    Places two-body terms (V_ij) into a four-index tensor (ij|kl) and performs a four-index transformation to the molecular orbital basis.
    
    Args:
        two_body: 2D array of two-body repulsion integrals in the atomic orbital basis. Usually repulsion array from v_term function. Shape (Natoms, Natoms)
        hf_orbs: 2D array of Hartree-Fock orbital coefficients in the atomic orbital basis. Shape (Natoms, Natoms)
    
    Returns:
        two_body_mo: 4D array of two-body repulsion integrals in the molecular orbital basis. Shape (Natoms, Natoms, Natoms, Natoms)
    '''
    Natoms = hf_orbs.shape[0]
    two_body_4i = np.zeros((Natoms, Natoms, Natoms, Natoms))
    ia = np.arange(Natoms)
    two_body_4i[ia[:, None], ia[:, None], ia[None, :], ia[None, :]] = two_body
    #four index transformation
    two_body_mo = np.einsum("ia, jb, kc, ld, ijkl -> abcd",
                             hf_orbs, hf_orbs, hf_orbs, hf_orbs, two_body_4i, optimize= 'optimal' )
    return two_body_mo



def write_gnu(strng,file):
    base_dir = os.path.dirname(file)
    mol_name = os.path.basename(file)
    output_path = os.path.join(base_dir, 'Gnuplots', f'gnuplot_script_{mol_name}')
    f=open(output_path,'w')
    f.write("#simulated spectrum\n")
    f.write("set term pdf size 6,4\n")
    f.write("unset key\n")
    f.write("set output '%s.pdf'\n" %(file))
    f.write("set xrange [200:700]\n")
    f.write("set samples 10000\n")
    f.write("set xlabel 'Wavelength / nm' font ',18'\n")
    f.write("set ylabel 'Absorbance / Arbitrary Units' font ',18'\n")
    f.write("set xtics font ',18'\n")
    f.write("set ytics font ',18'\n")
    f.write("set bmargin 4\n")
    f.write("p %s lw 3 dt 1" %strng)
    f.close()
    return

   

def diagonalise_xcis(ham_blocks, rng, nstates, out, ci_level):
    """
    Diagonalise the XCIS Hamiltonian by exploiting its block-diagonal structure
    (singlet / triplet / quintet blocks), then merge and sort the results by
    ascending energy.

    Args:
        ham_rot  : (nstates, nstates) ndarray — full XCIS Hamiltonian
        ndocc    : int — number of doubly-occupied orbitals
        nvirt    : int — number of virtual orbitals
        rng      : int — number of lowest states requested (sparse path if rng < nstates)
        nstates  : int — total number of states
        out      : file handle for log output
        ci_type  : str — type of CI calculation ('XCIS' or 'XCISD')

    Returns:
        ci_energies : (nstates,) or (rng,) ndarray — eigenvalues sorted low→high
        ci_coeffs   : (nstates, nstates) or (nstates, rng) ndarray — eigenvectors,
                      each column is a CI state in the full CSF basis
    """

    # Slice the three diagonal blocks
    H_s = ham_blocks[0]
    H_t = ham_blocks[1]
    n_singlet = H_s.shape[0]
    n_triplet = H_t.shape[0]
    
    if ci_level > 1:
        H_q = ham_blocks[2]
        n_quintet = H_q.shape[0]

    # ------------------------------------------------------------------
    # 1. Diagonalise each block
    # ------------------------------------------------------------------
    if rng < nstates:
        # Sparse path — request enough states from each block.
        # We over-request proportionally then trim after merging.
        # At minimum request 1 from each block, at most the full block size.
        k_s = max(1, min(n_singlet - 1, int(np.ceil(rng * n_singlet / nstates)) + 10))
        k_t = max(1, min(n_triplet - 1, int(np.ceil(rng * n_triplet / nstates)) + 10))
        k_q = max(1, min(n_quintet - 1, int(np.ceil(rng * n_quintet / nstates)) + 10))

        msg = (
            f"Using sparse solver (eigsh) — requesting "
            f"{k_s} singlets, {k_t} triplets, {k_q} quintets "
            f"(targeting {rng} states total)\n"
        )
        print(msg)
        out.write(msg)

        e_s, v_s = sp.eigsh(H_s, k=k_s, which="SA")
        e_t, v_t = sp.eigsh(H_t, k=k_t, which="SA")
        if ci_level > 1:
            e_q, v_q = sp.eigsh(H_q, k=k_q, which="SA")

    else:
        # Dense path — full diagonalisation of each block
        msg = "Using dense solver (eigh) on each block ...\n"
        print(msg)
        out.write(msg)

        e_s, v_s = linalg.eigh(H_s)
        e_t, v_t = linalg.eigh(H_t)
        if ci_level > 1:
            e_q, v_q = linalg.eigh(H_q)

    # ------------------------------------------------------------------
    # 2. Embed block eigenvectors into the full CSF basis
    #    Each column of ci_coeffs_block is a state vector of length nstates,
    #    with zeros outside the relevant block.
    # ------------------------------------------------------------------
    def embed(v, start, total):
        """Pad eigenvector matrix v into the full basis of size `total`."""
        n_basis, n_vecs = v.shape
        full = np.zeros((total, n_vecs))
        full[start:start + n_basis, :] = v
        return full          # shape: (nstates, n_vecs)

    V_s = embed(v_s, 0, nstates)   # (nstates, k_s or n_singlet)
    V_t = embed(v_t, n_singlet, nstates)   # (nstates, k_t or n_triplet)
    if ci_level > 1:
        V_q = embed(v_q, n_singlet+n_triplet, nstates)   # (nstates, k_q or n_quintet)

    # ------------------------------------------------------------------
    # 3. Concatenate all eigenvalues/vectors and sort by energy
    # ------------------------------------------------------------------
    
    if ci_level > 1:
        all_energies = np.concatenate([e_s, e_t, e_q])
        all_coeffs   = np.concatenate([V_s, V_t, V_q], axis=1)  # (nstates, total_vecs)
    else:
        all_energies = np.concatenate([e_s, e_t])
        all_coeffs   = np.concatenate([V_s, V_t], axis=1)  # (nstates, total_vecs)
        
    sort_idx = np.argsort(all_energies)
    all_energies = all_energies[sort_idx]
    all_coeffs   = all_coeffs[:, sort_idx]

    # ------------------------------------------------------------------
    # 4. Trim to rng states if using the sparse path
    # ------------------------------------------------------------------
    if rng < nstates:
        # Guard: if we didn't get enough states across blocks, warn and use what we have
        n_available = len(all_energies)
        if n_available < rng:
            msg = (
                f"Warning: only {n_available} states available after merging blocks "
                f"(requested {rng}). Consider increasing over-request buffer.\n"
            )
            print(msg)
            out.write(msg)
            rng = n_available

        all_energies = all_energies[:rng]
        all_coeffs   = all_coeffs[:, :rng]

    ci_energies = all_energies
    ci_coeffs   = all_coeffs

    msg = f"Diagonalisation complete. Returning {ci_energies.shape[0]} states.\n"
    print(msg)
    out.write(msg)

    return ci_energies, ci_coeffs



def ci_rot(ndocc,norbs,coords,atoms,energy0,repulsion,orb_energies,hf_orbs, file, ci_level):
    '''
    Calculates monoradical excited states in rotated (CSF) basis using the CIS or XCIS method. Used for molecules without Nitrogen or Chlorine present.
    
    Args:
        ndocc (int): Number of doubly occupied orbitals
        norbs (int): Total number of orbitals
        coords (array): Array of atomic coordinates
        atoms (array): Array of atomic symbols
        energy0 (float): Ground state energy
        repulsion (array): 2-electron repulsion integrals in AO basis
        orb_energies (array): HF orbital energies
        hf_orbs (array): HF molecular orbitals
        file (str): Name of file to write output to (without extension)

    '''
    base_dir = os.path.dirname(file)
    mol_name = os.path.basename(file)
    output_path = os.path.join(base_dir, 'Excited_States', f'{mol_name}_excitedstates.xyz')
    with open(output_path,'w') as out:
        print("")
        print("------------------------")
        print("Starting ExROPPP calculation for diradical in rotated basis")
        print("------------------------\n")

        out.write("")
        out.write("------------------------")
        out.write("Starting ExROPPP calculation for diradical in rotated basis")
        out.write("------------------------\n")

        # Transform 2-el ingrls into mo basis
        rep_tens = transform(repulsion,hf_orbs)
        print('Coulomb Matrix in MO basis, J_ij = (ii|jj):\n', np.einsum('iijj->ij', rep_tens))
        print('Coulomb Matrix in MO basis, K_ij = (ij|ji):\n', np.einsum('ijij->ij', rep_tens))
        # Get exchange and Coulomb terms for SOMOs
        '''
        print('Two-Electron Array')
        for p in range(norbs):
            for q in range(norbs):
                for r in range(norbs):
                    for s in range(norbs):
                        val = rep_tens[p, q, r, s]
                        # Only print significant values to avoid clutter
                        if abs(val) > 1e-8:
                            print(f"({p}, {q} | {r}, {s}) {val:15.8f}")
        '''
        # Construct CIS Hamiltonian
        if ci_level < 2:
            ham_rot, ham_blocks = get_full_CIMatrix(ndocc, norbs, energy0, orb_energies, rep_tens, ci_level)
        else:
            ham_rot, ham_blocks = get_full_CIMatrix(ndocc, norbs, energy0, orb_energies, rep_tens, ci_level)
        np.set_printoptions(precision=3, suppress=True)
        #print('CI Hamiltonian:\n', ham_rot)
            
        print('Dimensions of CI matrix:', ham_rot.shape)
        print("Checking that the Hamiltonian is symmetric (a value of zero means matrix is symmetric) ... ")
        print("Frobenius norm of matrix - matrix transpose = %f.\n" %(linalg.norm(ham_rot-ham_rot.T)))

        out.write("Checking that the Hamiltonian is symmetric (a value of zero means matrix is symmetric) ... \n")
        out.write("Frobenius norm of matrix - matrix transpose = %f.\n" %(linalg.norm(ham_rot-ham_rot.T)))
        
        # Print energies of CSFs
        #print_csf_info(out, ham_rot, ndocc, norbs, ci_level)
        
        # Set rng and cutoff_energy
        nstates = ham_rot.shape[0]
        if states_cutoff_option == 'states' and states_to_print <= nstates:
            rng = states_to_print
            print('Lowest %d states. WARNING - Some states may not be included in the spectrum.\n'%states_to_print)
            out.write('Lowest %d states. WARNING - Some states may not be included in the spectrum.\n'%states_to_print)
        else:
            rng = nstates
        if states_cutoff_option == 'energy':
            cutoff_energy = energy_cutoff
            print('Used energy cutoff of %04.2f eV for states. WARNING - Some states may not be included in spectrum.\n'%cutoff_energy)
            out.write('Used energy cutoff of %04.2f eV for states. WARNING - Some states may not be included in spectrum.\n'%cutoff_energy)
        else:
            cutoff_energy = 100
        
        ci_energies, ci_coeffs = diagonalise_xcis(ham_blocks, rng, nstates, out, ci_level)
        
        '''
        # Diagonalize CIS Hamiltonianfor first rng excited states
        if rng < nstates:
            print("Diagonalizing Hamiltonian using the sparse matrix method ...\n")
            out.write("Diagonalizing Hamiltonian using the sparse matrix method ...\n")

            ci_energies, ci_coeffs = sp.eigsh(ham_rot,k=rng,which="SA")
        elif rng == nstates:
            print("Diagonalizing Hamiltonian using the dense matrix method ...\n")
            out.write("Diagonalizing Hamiltonian using the dense matrix method ...\n")
            ci_energies, ci_coeffs = linalg.eigh(ham_rot)
        '''

        # Calculate transition dipole moment matrix
        dip_array = get_full_TDM(ndocc, norbs, coords, hf_orbs, ci_level)[0]

        
        print("Checking that the Dipole matrix is symmetric (a value of zero means matrix is symmetric) ... ")
        print(f"Frobenius norm of matrix - matrix transpose = {linalg.norm(dip_array[:, :, 0]-dip_array[:,:,0].T):.5f} \
            {linalg.norm(dip_array[:, :, 1]-dip_array[:,:,1].T):.5f}, {linalg.norm(dip_array[:, :, 2]-dip_array[:,:,2].T):.5f}.\n")
        out.write("Checking that the Dipole matrix is symmetric (a value of zero means matrix is symmetric) ... \n")
        out.write(f"Frobenius norm of matrix - matrix transpose = {linalg.norm(dip_array[:, :, 0]-dip_array[:,:,0].T):.5f} \
            {linalg.norm(dip_array[:, :, 1]-dip_array[:,:,1].T):.5f}, {linalg.norm(dip_array[:, :, 2]-dip_array[:,:,2].T):.5f}.\n")
        
        dip_couplings = np.einsum("ijx,jk",dip_array,ci_coeffs)
        state0_tdms = np.einsum("j,jix",ci_coeffs[:,0].T, dip_couplings)
        state1_tdms = np.einsum("j,jix",ci_coeffs[:,1].T, dip_couplings)
        tdms = (state0_tdms, state1_tdms) 
        
        # Print information about CI states
        strngs, osc_arrays, s2_array = print_ci_info(out, ci_energies, ci_coeffs, ndocc, norbs, tdms, rng, cutoff_energy, ci_level, csf_tol=0.05)
        strngs = (strngs[0][1:], strngs[1][1:])
    return strngs, ci_energies - ci_energies[0], osc_arrays, s2_array





def rad_calc(file,params):
    filename = os.path.basename(file)
    coord,atoms_array,coord_w_h,dist_array,nelec,ndocc,n_list,natoms_c,natoms_n,natoms_cl,energy0,one_body,two_body,orb_energy,hf_orbs,fock_mat=main_scf(file,params)
    com,coord = re_center(coord,atoms_array,coord_w_h)
    hf_orbs = orb_sign(hf_orbs,orb_energy,nelec,dist_array,natoms_c,alt)
    print("\n--------------------------")
    print("Converged ROPPP Orbitals")
    print("--------------------------\n")
    natoms=np.shape(coord)[0]
    for iorb in range(natoms):
        print('orbital number', iorb + 1, 'energy', orb_energy[iorb]-orb_energy[int((nelec-1)/2)])
        print(np.around(hf_orbs[:, iorb], decimals=2))

            #########################################################
             # PRINTING OF MOLECULAR ORBITALS BASED ON GAMESS OUTPUT #
             #########################################################
    atomic_numbers=[]
    for atom in atoms_array:
        number={"C":6.0,"c":6.0,"H":1.0,"h":1.0,"N":7.0,"n":7.0,"N1":7.0,"n1":7.0,"N2":7.0,"n2":7.0,"Cl":17.0,"cl":17.0,"CL":17.0}[atom[0]]
        atomic_numbers.append([atom[0],number])
    f=open('Converged_orbitals/%s.out'%filename,'w')
    f.write("\n")
    f.write("\nGAMESS COORDINATES FORMAT")
    f.write("\n")
    f.write("\n ATOM      ATOMIC                      COORDINATES (BOHR)")
    f.write("\n           CHARGE         X                   Y                   Z")
    #for i,atom in enumerate(atoms_array):
    for i in range(natoms_c+natoms_n+natoms_cl):
        f.write("\n %s           %d     %f            %f            %f"%(atoms_array[i][0],atomic_numbers[i][1],coord[i,0]*tobohr,coord[i,1]*tobohr,coord[i,2]*tobohr))
    f.write("\n                      ")
    f.write("\n     ATOMIC BASIS SET")
    f.write("\n     ----------------")
    f.write("\n ")
    f.write("\n ")
    f.write("\n ")
    f.write("\n  SHELL TYPE  PRIMITIVE        EXPONENT          CONTRACTION COEFFICIENT(S)")
    f.write("\n ")
    n1=1
    n2=1
    for i,atom in enumerate(atoms_array):
        if atom[0] == 'C':
            f.write("\n C         ")
            f.write("\n ")
            f.write("\n     %2s   S     %3s            27.3850330    0.430128498301"%(str(n1+i),str(n2+i)))
            f.write("\n     %2s   S     %3s             4.8745221    0.678913530502"%(str(n1+i),str(n2+i+1)))
            f.write("\n ")
            f.write("\n     %2s   L     %3s             1.1367482    0.049471769201    0.511540707616"%(str(n1+i+1),str(n2+i+2)))
            f.write("\n     %2s   L     %3s             0.2883094    0.963782408119    0.612819896119"%(str(n1+i+1),str(n2+i+3)))
            f.write("\n ")
            n1+=1
            n2+=3
        if atom[0] in ['N','N1','N2']:
            f.write("\n N         ")
            f.write("\n ")
            f.write("\n     %2s   S     %3s            27.3850330    0.430128498301"%(str(n1+i),str(n2+i)))
            f.write("\n     %2s   S     %3s             4.8745221    0.678913530502"%(str(n1+i),str(n2+i+1)))
            f.write("\n ")
            f.write("\n     %2s   L     %3s             1.1367482    0.049471769201    0.511540707616"%(str(n1+i+1),str(n2+i+2)))
            f.write("\n     %2s   L     %3s             0.2883094    0.963782408119    0.612819896119"%(str(n1+i+1),str(n2+i+3)))
            f.write("\n ")
            n1+=1
            n2+=3
        if atom[0] == 'Cl':
            f.write("\n Cl         ")
            f.write("\n ")
            f.write("\n     %2s   S     %3s           229.9441039    0.430128498301"%(n1+i,n2+i))
            f.write("\n     %2s   S     %3s            40.9299346    0.678913530502"%(n1+i,n2+i+1))
            f.write("\n ")
            f.write("\n     %2s   L     %3s            15.0576101    0.049471769201    0.511540707616"%(str(n1+i+1),str(n2+i+2)))
            f.write("\n     %2s   L     %3s             3.8190075    0.963782408119    0.612819896119"%(str(n1+i+1),str(n2+i+3)))
            f.write("\n ")
            f.write("\n     %2s   L     %3s             0.8883464   -0.298398604487    0.348047191182"%(str(n1+i+2),str(n2+i+4)))
            f.write("\n     %2s   L     %3s             0.3047828    1.227982887359    0.722252322062"%(str(n1+i+2),str(n2+i+5)))
            n1+=2
            n2+=5
        f.write("\n ")  
    for imo in range(hf_orbs.shape[0]):
        f.write("\n ")
        f.write("\n          ------------")
        f.write("\n          EIGENVECTORS")
        f.write("\n          ------------")
        f.write("\n ")
        f.write("\n                      %s    "%str(imo+1))
        f.write("\n                   %4f "%(orb_energy[imo]-orb_energy[int((nelec-1)/2)]))
        f.write("\n                     A     ")# symmetry (A is default for c1)
        kao=1
        for jatom, atom in enumerate(atoms_array):
            if atom[0]=='C':
                if file=='allyl' or file=='benzyl':
                    f.write("\n  %3s  C %2s  S    0.000000  "%(str(kao),str(jatom+1)))
                    f.write("\n  %3s  C %2s  S    0.000000"  %(str(kao+1),str(jatom+1)))
                    f.write("\n  %3s  C %2s  X    0.000000  "%(str(kao+2),str(jatom+1)))
                    #f.write("\n  %3s  C %2s  X    %6f"%(str(kao+2),str(jatom+1),hf_orbs[jatom,imo]))
                    #f.write("\n  %3s  C %2s  Y    0.000000  "%(str(kao+3),str(jatom+1)))
                    #f.write("\n  %3s  C %2s  Z    0.000000  "%(str(kao+4),str(jatom+1)))
                    f.write("\n  %3s  C %2s  Y    %6f"%(str(kao+3),str(jatom+1),hf_orbs[jatom,imo]))
                    #f.write("\n  %3s  C %2s  Z    %6f"%(str(kao+4),str(jatom+1),hf_orbs[jatom,imo]))
                    f.write("\n  %3s  C %2s  Z    0.000000  "%(str(kao+4),str(jatom+1)))
                    kao+=5
                elif file=='dpm' or file=='dpxm' or file=='pdxm':
                    f.write("\n  %3s  C %2s  S    0.000000  "%(str(kao),str(jatom+1)))
                    f.write("\n  %3s  C %2s  S    0.000000"  %(str(kao+1),str(jatom+1)))
                    f.write("\n  %3s  C %2s  X    %6f"%(str(kao+2),str(jatom+1),hf_orbs[jatom,imo]))
                    f.write("\n  %3s  C %2s  Y    0.000000  "%(str(kao+3),str(jatom+1)))
                    f.write("\n  %3s  C %2s  Z    0.000000  "%(str(kao+4),str(jatom+1)))
                    kao+=5
                else:
                    f.write("\n  %3s  C %2s  S    0.000000  "%(str(kao),str(jatom+1)))
                    f.write("\n  %3s  C %2s  S    0.000000"  %(str(kao+1),str(jatom+1)))
                    f.write("\n  %3s  C %2s  X    0.000000  "%(str(kao+2),str(jatom+1)))
                    f.write("\n  %3s  C %2s  Y    0.000000  "%(str(kao+3),str(jatom+1)))
                    f.write("\n  %3s  C %2s  Z    %6f"%(str(kao+4),str(jatom+1),hf_orbs[jatom,imo]))
                    kao+=5
            if atom[0] in ['N','N1','N2']:
                f.write("\n  %3s  N %2s  S    0.000000  "%(str(kao),str(jatom+1)))
                f.write("\n  %3s  N %2s  S    0.000000"  %(str(kao+1),str(jatom+1)))
                f.write("\n  %3s  N %2s  X    0.000000  "%(str(kao+2),str(jatom+1)))
                f.write("\n  %3s  N %2s  Y    0.000000  "%(str(kao+3),str(jatom+1)))
                f.write("\n  %3s  N %2s  Z    %6f"%(str(kao+4),str(jatom+1),hf_orbs[jatom,imo]))
                kao+=5
            if atom[0]=='Cl':
                f.write("\n  %3s  Cl%2s  S    0.000000  "%(str(kao),str(jatom+1)))
                f.write("\n  %3s  Cl%2s  S    0.000000  "%(str(kao+1),str(jatom+1)))
                f.write("\n  %3s  Cl%2s  X    0.000000  "%(str(kao+2),str(jatom+1)))
                f.write("\n  %3s  Cl%2s  Y    0.000000  "%(str(kao+3),str(jatom+1)))
                f.write("\n  %3s  Cl%2s  Z    0.000000  "%(str(kao+4),str(jatom+1)))
                f.write("\n  %3s  Cl%2s  S    0.000000  "%(str(kao+5),str(jatom+1)))
                f.write("\n  %3s  Cl%2s  X    0.000000  "%(str(kao+6),str(jatom+1)))
                f.write("\n  %3s  Cl%2s  Y    0.000000  "%(str(kao+7),str(jatom+1)))
                f.write("\n  %3s  Cl%2s  Z    %6f"%(str(kao+8),str(jatom+1),hf_orbs[jatom,imo]))
                kao+=9
        f.write("\n  ...... END OF ROHF CALCULATION ......")
    f.write("\n ")
    f.close()
    # check that fock matrix is diagonalized
    fock_mo = np.dot(hf_orbs.T,np.dot(fock_mat,hf_orbs))
    for i in range(fock_mo.shape[0]):
        for j in range(fock_mo.shape[0]):
            if i!=j and fock_mo[i,j] > 1e-4:
                print("Fock matrix not converged!")
                print("\nFock Matrix:")
                print(f'Large off-diagonal matrix element found at F_{i},{j}: {fock_mo[i,j]}')
                print(fock_mo)
                sys.exit()
    # check the density matrix
    dens_mat = density(hf_orbs, ndocc)
    dens_mo = np.dot(hf_orbs.T, np.dot(dens_mat, hf_orbs))
    print('\nOrbital occupation numbers:')
    for i in range(dens_mo.shape[0]):
        print("%d: %f"%(i+1,dens_mo[i,i]))
        for j in range(dens_mo.shape[0]):
            if i!=j and fock_mo[i,j] > 1e-4:
                print("Density matrix not converged!")
                print("\nDensity Matrix:")
                print(dens_mo)
                sys.exit()
    strngs, ci_energies_array, osc_arrays, s2_array = ci_rot(ndocc, natoms, coord, atoms_array, energy0, two_body, orb_energy, hf_orbs, file, ci_level=3)
    return strngs, ci_energies_array, osc_arrays, s2_array
