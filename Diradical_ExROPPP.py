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
from CIBuilderNew import *
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

        if ci_level > 1:
            k_q = max(1, min(n_quintet - 1, int(np.ceil(rng * n_quintet / nstates)) + 10))

        if ci_level > 1:
            msg = (
                f"Using sparse solver (eigsh) — requesting "
                f"{k_s} singlets, {k_t} triplets, {k_q} quintets "
                f"(targeting {rng} states total)\n"
            )
        else:
            msg = (
                f"Using sparse solver (eigsh) — requesting "
                f"{k_s} singlets, {k_t} triplets "
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

    # ------------------------------------------------------------------
    # 5. Calculate S^2 from the spin block containing each state
    # ------------------------------------------------------------------
    s2_array = np.zeros(len(ci_energies))

    for i in range(len(ci_energies)):

        singlet_weight = np.sum(
            ci_coeffs[:n_singlet, i]**2
        )

        triplet_weight = np.sum(
            ci_coeffs[n_singlet:n_singlet+n_triplet, i]**2
        )

        if ci_level > 1:
            quintet_weight = np.sum(
                ci_coeffs[n_singlet+n_triplet:, i]**2
            )
        else:
            quintet_weight = 0.0

        if quintet_weight > triplet_weight and quintet_weight > singlet_weight:
            s2_array[i] = 6.0

        elif triplet_weight > singlet_weight:
            s2_array[i] = 2.0

        else:
            s2_array[i] = 0.0

    msg = f"Diagonalisation complete. Returning {ci_energies.shape[0]} states.\n"
    print(msg)
    out.write(msg)

    return ci_energies, ci_coeffs, s2_array



def ci_rot(ndocc,norbs,coords,atoms,energy0,rep_tens,fock_mat,hf_orbs, file, ci_level):
    '''
    Calculates monoradical excited states in rotated (CSF) basis using the CIS or XCIS method. Used for molecules without Nitrogen or Chlorine present.
    
    Args:
        ndocc (int): Number of doubly occupied orbitals
        norbs (int): Total number of orbitals
        coords (array): Array of atomic coordinates
        atoms (array): Array of atomic symbols
        energy0 (float): Ground state energy
        rep_tens (array): 4-index two-electron repulsion integrals in MO basis
        fock_mat (array): Fock matrix in MO basis
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
        
        # Construct CIS Hamiltonian
        ham_rot, ham_blocks = get_full_CIMatrix(ndocc, norbs, energy0, fock_mat, rep_tens, ci_level)
            
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
        
        ci_energies, ci_coeffs, s2_array = diagonalise_xcis(ham_blocks, rng, nstates, out, ci_level)

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
        strngs, osc_arrays = print_ci_info(
            out, ci_energies, ci_coeffs, ndocc, norbs, tdms,
            dip_array, rng, cutoff_energy, ci_level, csf_tol=0.05)
        strngs = (strngs[0][1:], strngs[1][1:])

        # Print TDM Summary
        print_tdm_info(ndocc, norbs, dip_array, ci_level, ci_energies)
    return strngs, ci_energies - ci_energies[0], osc_arrays





def rad_calc(file,params,rotation_matrix=None,converged_orbs=None):
    filename = os.path.basename(file)
    coord,atoms_array,coord_w_h,dist_array,nelec,ndocc,n_list,natoms_c,natoms_n,natoms_cl,energy0,rep_tens,hf_orbs,fock_mat = main_scf(file,params,rotation_matrix,converged_orbs)
    com,coord = re_center(coord,atoms_array,coord_w_h)
    natoms=np.shape(coord)[0]


    fock_mo = np.dot(hf_orbs.T,np.dot(fock_mat,hf_orbs))
    
    dens_mat = density(hf_orbs, ndocc)
    dens_mo = np.dot(hf_orbs.T, np.dot(dens_mat, hf_orbs))
    print('\nOrbital occupation numbers:')
    for i in range(dens_mo.shape[0]):
        print("%d: %f"%(i+1,dens_mo[i,i]))
    strngs, ci_energies_array, osc_arrays = ci_rot(ndocc, natoms, coord, atoms_array, energy0, rep_tens, fock_mo, hf_orbs, file, ci_level=3)
    return strngs, ci_energies_array, osc_arrays




def print_tdm_info(ndocc, norbs, tdms, ci_level, ci_energies=None, tdm_threshold=1e-5):

    nvirt = norbs - ndocc - 2
    npairs = ndocc * nvirt
    ndoc1 = int((ndocc ** 2 + ndocc) / 2)
    ndcv1 = int((nvirt ** 2 + nvirt) / 2)
    OS1_index = 0
    ZWm_index = 1
    ZWp_index = 2
    if ci_level == 0:
        OS3_index = 0
    elif ci_level == 1:
        OS3_index = 2 * nvirt + 2 * ndocc + 3
    elif ci_level == 2:
        OS3_index = 4 * npairs + 2 * nvirt + 2 * ndocc + 3
    else:
        OS3_index = ndcv1 + ndoc1 + 4 * npairs + 2 * nvirt + 2 * ndocc + 3

    for i, state_label in [(OS1_index, "OS1"), (ZWm_index, "ZW_minus"), (ZWp_index, "ZW_plus"), (OS3_index, "OS3")]:  # Loop over lowest 'reference' states
        # --- FIX: OS3_index (and friends) is a position in the full CSF basis,
        # not an index into the (possibly truncated/sorted) ci_energies array
        # of diagonalised eigenstates. Only use ci_energies[i] when i is a
        # valid index into it; otherwise skip printing the energy rather than
        # crashing or printing a meaningless value.
        if ci_energies is not None and i < ci_energies.shape[0]:
            print("\nState %s %04.3f eV " % (state_label, ci_energies[i]))
        else:
            print("\nState %s (energy unavailable: index %d outside truncated ci_energies of size %s)"
                  % (state_label, i, ci_energies.shape[0] if ci_energies is not None else "N/A"))
        print("Excitation    TDM")
        for j in range(tdms.shape[0]):  # Loop over configurations in each CIS state

            # Skip rows whose TDM is essentially zero in all three components
            if not np.any(np.abs(tdms[i, j, :]) > tdm_threshold):
                continue

            if ci_level == 0:
                if j == 0:
                    str = "|1^OS>"
                elif j == 1:
                    str = "|ZW->"
                elif j == 2:
                    str = "|ZW+>"
                elif j == 3:
                    str = "|3^OS>"
                print("%s (%10.5f, %10.5f, %10.5f)" % (str, tdms[i, j, 0], tdms[i, j, 1], tdms[i, j, 2]))

            elif ci_level == 1:
                ### SINGLET CSFS ###
                if j == 0:
                    str = "|1^OS>"
                elif j == 1:
                    str = "|1^ZW->"
                elif j == 2:
                    str = "|1^ZW+>"
                elif j > 2 and j <= ndocc + 2:
                    iorb = ndocc + 3 - j
                    str = f"|1^CS({iorb}->0)>"
                elif j > ndocc + 2 and j <= (2 * ndocc + 2):
                    iorb = 2 * ndocc + 3 - j
                    str = f"|1^CS({iorb}->0')>"
                elif j > (2 * ndocc + 2) and j <= (nvirt + 2 * ndocc + 2):
                    iorb = j - (2 * ndocc + 2)
                    str = f"|1^SV(0->{iorb}')>"
                elif j > (nvirt + 2 * ndocc + 2) and j <= (2 * nvirt + 2 * ndocc + 2):
                    iorb = j - (nvirt + 2 * ndocc + 2)
                    str = f"|1^SV(0'->{iorb}')>"

                ### TRIPLET CSFs ###
                elif j == (2 * nvirt + 2 * ndocc + 3):
                    str = "|3^OS>"
                elif j > (2 * nvirt + 2 * ndocc + 3) and j <= (2 * nvirt + 3 * ndocc + 3):
                    iorb = (2 * nvirt + 3 * ndocc + 4) - j
                    str = f"|3^CS({iorb}->0)>"
                elif j > (2 * nvirt + 3 * ndocc + 3) and j <= (2 * nvirt + 4 * ndocc + 3):
                    iorb = (2 * nvirt + 4 * ndocc + 4) - j
                    str = f"|3^CS({iorb}->0')>"
                elif j > (2 * nvirt + 4 * ndocc + 3) and j <= (3 * nvirt + 4 * ndocc + 3):
                    iorb = j - (2 * nvirt + 4 * ndocc + 3)
                    str = f"|3^SV(0->{iorb}')>"
                elif j > (3 * nvirt + 4 * ndocc + 3) and j <= (4 * nvirt + 4 * ndocc + 3):
                    iorb = j - (3 * nvirt + 4 * ndocc + 3)
                    str = f"|3^SV(0'->{iorb}')>"

                print("%s (%10.5f, %10.5f, %10.5f)" % (str, tdms[i, j, 0], tdms[i, j, 1], tdms[i, j, 2]))

            elif ci_level == 2:
                ########## SINGLET CSFS ##########
                if j == 0:
                    str = "|1^OS>"
                elif j == 1:
                    str = "|1^ZW->"
                elif j == 2:
                    str = "|1^ZW+>"
                elif j > 2 and j <= ndocc + 2:
                    iorb = ndocc + 3 - j
                    str = f"|1^CS({iorb}->0)>"
                elif j > ndocc + 2 and j <= (2 * ndocc + 2):
                    iorb = 2 * ndocc + 3 - j
                    str = f"|1^CS({iorb}->0')>"
                elif j > (2 * ndocc + 2) and j <= (nvirt + 2 * ndocc + 2):
                    iorb = j - (2 * ndocc + 2)
                    str = f"|1^SV(0->{iorb}')>"
                elif j > (nvirt + 2 * ndocc + 2) and j <= (2 * nvirt + 2 * ndocc + 2):
                    iorb = j - (nvirt + 2 * ndocc + 2)
                    str = f"|1^SV(0'->{iorb}')>"
                elif j > (2 * nvirt + 2 * ndocc + 2) and j <= ((npairs) + 2 * nvirt + 2 * ndocc + 2):
                    o_orb = ndocc - ((j - (2 * nvirt + 2 * ndocc + 3)) // nvirt)
                    v_orb = ((j - (2 * nvirt + 2 * ndocc + 3)) % nvirt) + 1
                    str = f"|1S^CV({o_orb}->{v_orb}')>"
                elif j > ((npairs) + 2 * nvirt + 2 * ndocc + 2) and j <= (2 * (npairs) + 2 * nvirt + 2 * ndocc + 2):
                    o_orb = ndocc - ((j - ((npairs) + 2 * nvirt + 2 * ndocc + 3)) // nvirt)
                    v_orb = ((j - ((npairs) + 2 * nvirt + 2 * ndocc + 3)) % nvirt) + 1
                    str = f"|1T^CV({o_orb}->{v_orb}')>"
                elif j > (2 * npairs + 2 * nvirt + 2 * ndocc + 2) and j <= (3 * npairs + 2 * nvirt + 2 * ndocc + 2):
                    o_orb = ndocc - ((j - (2 * npairs + 2 * nvirt + 2 * ndocc + 3)) // nvirt)
                    v_orb = ((j - (2 * npairs + 2 * nvirt + 2 * ndocc + 3)) % nvirt) + 1
                    str = f"|1^ZCV0({o_orb}->{v_orb}')>"
                elif j > (3 * npairs + 2 * nvirt + 2 * ndocc + 2) and j <= (4 * npairs + 2 * nvirt + 2 * ndocc + 2):
                    o_orb = ndocc - ((j - (3 * npairs + 2 * nvirt + 2 * ndocc + 3)) // nvirt)
                    v_orb = ((j - (3 * npairs + 2 * nvirt + 2 * ndocc + 3)) % nvirt) + 1
                    str = f"|1^ZCV0'({o_orb}->{v_orb}')>"
                ########### TRIPLET CSFs ###########
                elif j == (4 * npairs + 2 * nvirt + 2 * ndocc + 3):
                    str = "|3^OS>"
                elif j > (4 * npairs + 2 * nvirt + 2 * ndocc + 3) and j <= (4 * npairs + 2 * nvirt + 3 * ndocc + 3):
                    iorb = (4 * npairs + 2 * nvirt + 3 * ndocc + 4) - j
                    str = f"|3^CS({iorb}->0)>"
                elif j > (4 * npairs + 2 * nvirt + 3 * ndocc + 3) and j <= (4 * npairs + 2 * nvirt + 4 * ndocc + 3):
                    iorb = (4 * npairs + 2 * nvirt + 4 * ndocc + 4) - j
                    str = f"|3^CS({iorb}->0')>"
                elif j > (4 * npairs + 2 * nvirt + 4 * ndocc + 3) and j <= (4 * npairs + 3 * nvirt + 4 * ndocc + 3):
                    iorb = j - (4 * npairs + 2 * nvirt + 4 * ndocc + 3)
                    str = f"|3^SV(0->{iorb}')>"
                elif j > (4 * npairs + 3 * nvirt + 4 * ndocc + 3) and j <= (4 * npairs + 4 * nvirt + 4 * ndocc + 3):
                    iorb = j - (4 * npairs + 3 * nvirt + 4 * ndocc + 3)
                    str = f"|3^SV(0'->{iorb}')>"
                elif j > (4 * npairs + 4 * nvirt + 4 * ndocc + 3) and j <= (5 * npairs + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (4 * npairs + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (4 * npairs + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|3T^CV({o_orb}->{v_orb}')>"
                elif j > (5 * npairs + 4 * nvirt + 4 * ndocc + 3) and j <= (6 * npairs + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (5 * npairs + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (5 * npairs + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|3S^CV({o_orb}->{v_orb}')>"
                elif j > (6 * npairs + 4 * nvirt + 4 * ndocc + 3) and j <= (7 * npairs + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (6 * npairs + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (6 * npairs + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|3X^CV({o_orb}->{v_orb}')>"
                elif j > (7 * npairs + 4 * nvirt + 4 * ndocc + 3) and j <= (8 * npairs + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (7 * npairs + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (7 * npairs + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|3^ZCV0({o_orb}->{v_orb}')>"
                elif j > (8 * npairs + 4 * nvirt + 4 * ndocc + 3) and j <= (9 * npairs + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (9 * npairs + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (9 * npairs + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|3^ZCV0'({o_orb}->{v_orb}')>"
                elif j > (9 * npairs + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (9 * npairs + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (9 * npairs + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|5^CV({o_orb}->{v_orb}')>"

                print("%s (%10.5f, %10.5f, %10.5f)" % (str, tdms[i, j, 0], tdms[i, j, 1], tdms[i, j, 2]))

            elif ci_level == 3:
                ########## SINGLET CSFS ##########
                if j == 0:
                    str = "|1^OS>"
                elif j == 1:
                    str = "|1^ZW->"
                elif j == 2:
                    str = "|1^ZW+>"
                elif j > 2 and j <= ndocc + 2:
                    iorb = ndocc + 3 - j
                    str = f"|1^CS({iorb}->0)>"
                elif j > ndocc + 2 and j <= (2 * ndocc + 2):
                    iorb = 2 * ndocc + 3 - j
                    str = f"|1^CS({iorb}->0')>"
                elif j > (2 * ndocc + 2) and j <= (nvirt + 2 * ndocc + 2):
                    iorb = j - (2 * ndocc + 2)
                    str = f"|1^SV(0->{iorb}')>"
                elif j > (nvirt + 2 * ndocc + 2) and j <= (2 * nvirt + 2 * ndocc + 2):
                    iorb = j - (nvirt + 2 * ndocc + 2)
                    str = f"|1^SV(0'->{iorb}')>"
                elif j > (2 * nvirt + 2 * ndocc + 2) and j <= (npairs + 2 * nvirt + 2 * ndocc + 2):
                    o_orb = ndocc - ((j - (2 * nvirt + 2 * ndocc + 3)) // nvirt)
                    v_orb = ((j - (2 * nvirt + 2 * ndocc + 3)) % nvirt) + 1
                    str = f"|1S^CV({o_orb}->{v_orb}')>"
                elif j > (npairs + 2 * nvirt + 2 * ndocc + 2) and j <= (2 * npairs + 2 * nvirt + 2 * ndocc + 2):
                    o_orb = ndocc - ((j - (npairs + 2 * nvirt + 2 * ndocc + 3)) // nvirt)
                    v_orb = ((j - (npairs + 2 * nvirt + 2 * ndocc + 3)) % nvirt) + 1
                    str = f"|1T^CV({o_orb}->{v_orb}')>"
                elif j > (2 * npairs + 2 * nvirt + 2 * ndocc + 2) and j <= (3 * npairs + 2 * nvirt + 2 * ndocc + 2):
                    o_orb = ndocc - ((j - (2 * npairs + 2 * nvirt + 2 * ndocc + 3)) // nvirt)
                    v_orb = ((j - (2 * npairs + 2 * nvirt + 2 * ndocc + 3)) % nvirt) + 1
                    str = f"|1^ZCV0({o_orb}->{v_orb}')>"
                elif j > (3 * npairs + 2 * nvirt + 2 * ndocc + 2) and j <= (4 * npairs + 2 * nvirt + 2 * ndocc + 2):
                    o_orb = ndocc - ((j - (3 * npairs + 2 * nvirt + 2 * ndocc + 3)) // nvirt)
                    v_orb = ((j - (3 * npairs + 2 * nvirt + 2 * ndocc + 3)) % nvirt) + 1
                    str = f"|1^ZCV0'({o_orb}->{v_orb}')>"
                elif j > (4 * npairs + 2 * nvirt + 2 * ndocc + 2) and j <= (ndoc1 + 4 * npairs + 2 * nvirt + 2 * ndocc + 2):
                    block_start = 4 * npairs + 2 * nvirt + 2 * ndocc + 3
                    k = j - block_start
                    o_orb1 = ndocc
                    temp_k = k
                    row_size = ndocc
                    while temp_k >= row_size:
                        temp_k -= row_size
                        o_orb1 -= 1
                        row_size -= 1
                    o_orb2 = o_orb1 - temp_k
                    str = f"|1^CSD_({o_orb1},{o_orb2})>"
                elif j > (ndoc1 + 4 * npairs + 2 * nvirt + 2 * ndocc + 2) and j <= (ndcv1 + ndoc1 + 4 * npairs + 2 * nvirt + 2 * ndocc + 2):
                    block_start = ndoc1 + 4 * npairs + 2 * nvirt + 2 * ndocc + 3
                    k = j - block_start
                    v_orb1 = 1
                    temp_k = k
                    row_size = nvirt
                    while temp_k >= row_size:
                        temp_k -= row_size
                        v_orb1 += 1
                        row_size -= 1
                    v_orb2 = v_orb1 + temp_k
                    str = f"|1^SVD_({v_orb1}',{v_orb2}')>"
                ########### TRIPLET CSFs ###########
                elif j == (ndcv1 + ndoc1 + 4 * npairs + 2 * nvirt + 2 * ndocc + 3):
                    str = "|3^OS>"
                elif j > (ndcv1 + ndoc1 + 4 * npairs + 2 * nvirt + 2 * ndocc + 3) and j <= (ndcv1 + ndoc1 + 4 * npairs + 2 * nvirt + 3 * ndocc + 3):
                    iorb = (ndcv1 + ndoc1 + 4 * npairs + 2 * nvirt + 3 * ndocc + 4) - j
                    str = f"|3^CS({iorb}->0)>"
                elif j > (ndcv1 + ndoc1 + 4 * npairs + 2 * nvirt + 3 * ndocc + 3) and j <= (ndcv1 + ndoc1 + 4 * npairs + 2 * nvirt + 4 * ndocc + 3):
                    iorb = (ndcv1 + ndoc1 + 4 * npairs + 2 * nvirt + 4 * ndocc + 4) - j
                    str = f"|3^CS({iorb}->0')>"
                elif j > (ndcv1 + ndoc1 + 4 * npairs + 2 * nvirt + 4 * ndocc + 3) and j <= (ndcv1 + ndoc1 + 4 * npairs + 3 * nvirt + 4 * ndocc + 3):
                    iorb = j - (ndcv1 + ndoc1 + 4 * npairs + 2 * nvirt + 4 * ndocc + 3)
                    str = f"|3^SV(0->{iorb}')>"
                elif j > (ndcv1 + ndoc1 + 4 * npairs + 3 * nvirt + 4 * ndocc + 3) and j <= (ndcv1 + ndoc1 + 4 * npairs + 4 * nvirt + 4 * ndocc + 3):
                    iorb = j - (ndcv1 + ndoc1 + 4 * npairs + 3 * nvirt + 4 * ndocc + 3)
                    str = f"|3^SV(0'->{iorb}')>"
                elif j > (ndcv1 + ndoc1 + 4 * npairs + 4 * nvirt + 4 * ndocc + 3) and j <= (ndcv1 + ndoc1 + 5 * npairs + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (ndcv1 + ndoc1 + 4 * npairs + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (ndcv1 + ndoc1 + 4 * npairs + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|3T^CV({o_orb}->{v_orb}')>"
                elif j > (ndcv1 + ndoc1 + 5 * npairs + 4 * nvirt + 4 * ndocc + 3) and j <= (ndcv1 + ndoc1 + 6 * npairs + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (ndcv1 + ndoc1 + 5 * npairs + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (ndcv1 + ndoc1 + 5 * npairs + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|3S^CV({o_orb}->{v_orb}')>"
                elif j > (ndcv1 + ndoc1 + 6 * npairs + 4 * nvirt + 4 * ndocc + 3) and j <= (ndcv1 + ndoc1 + 7 * npairs + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (ndcv1 + ndoc1 + 6 * npairs + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (ndcv1 + ndoc1 + 6 * npairs + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|3X^CV({o_orb}->{v_orb}')>"
                elif j > (ndcv1 + ndoc1 + 7 * npairs + 4 * nvirt + 4 * ndocc + 3) and j <= (ndcv1 + ndoc1 + 8 * npairs + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (ndcv1 + ndoc1 + 7 * npairs + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (ndcv1 + ndoc1 + 7 * npairs + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|3^ZCV0({o_orb}->{v_orb}')>"
                elif j > (ndcv1 + ndoc1 + 8 * npairs + 4 * nvirt + 4 * ndocc + 3) and j <= (ndcv1 + ndoc1 + 9 * npairs + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (ndcv1 + ndoc1 + 9 * npairs + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (ndcv1 + ndoc1 + 9 * npairs + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|3^ZCV0'({o_orb}->{v_orb}')>"
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
                elif j > (nvirt ** 2 + ndocc ** 2 + 9 * npairs + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (nvirt ** 2 + ndocc ** 2 + 9 * npairs + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (nvirt ** 2 + ndocc ** 2 + 9 * npairs + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|5^CV({o_orb}->{v_orb}')>"

                print("%s (%10.5f, %10.5f, %10.5f)" % (str, tdms[i, j, 0], tdms[i, j, 1], tdms[i, j, 2]))