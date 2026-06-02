import numpy as np
import scipy.optimize as optimize
import scipy.sparse.linalg as sp
import scipy.linalg as linalg
from datetime import datetime
from subprocess import getoutput
import sys
from ExROPPP_settings_opt import *
import os





# parser = argparse.ArgumentParser()
# parser.add_argument('geometry', type = str, help = 'file containing geometry')
# args = parser.parse_args()
# optimized_geometry = args.geometry

# params=[[-22.53982183,   1.70115484 ,  8.47994446 ,  1.17367777,   0.        ],
#  [ -3.25983816 ,-24.50393011 ,  1.76554162 , 13.0486315  ,  1.18938422],
#  [-17.50211252 ,-23.67958463 ,  1.43383449 , 18.08184948 ,  1.12853335],
#  [-10.18396573 ,-26.36242115 ,  1.45855408 ,  9.61199125 ,  2.23245629]] 





def read_geom(file):
    '''
    Read molecular geometry from file and returns various arrays and integers.
    
    Args:
        - file (str): File containing molecular geometries in ... format.
    Returns:
        - array (ndarray): 2D Array of atomic coordinates of all heavy atoms (C then N then Cl) in Angstrom. Shape (natoms, 3).
                         Used as the atomic coordinates array for electronic structure calculation.
        - atoms (ndarray): Array of atomic symbols and atomic numbers for all atoms. Listed in order of carbon then nitrogen 
                         then chlorine then hydrogen. Shape (natoms, 2).
        - array_all (ndarray): Array of atomic coordinates of all atoms including hydrogen in order to calculate number of bonds 
                         to nitrogen. Not used if N isn't present.
        - natoms_{c, n, cl} (int): Number of carbon, nitrogen and chlorine atoms respectively.
        - natoms (int): Total number of heavy atoms in the molecule.
    '''
    print("--------------------------------")
    print("Cartesian Coordinates / Angstrom")
    print("--------------------------------\n")
    f=open(file,'r')
    array=[]
    array_n=[]
    array_h=[] 
    array_cl=[] 
    array_all=[]
    atoms_c=[] 
    atoms_n=[]
    atoms_cl=[]
    atoms_h=[]
    natoms_c=0
    natoms_n=0 
    natoms_cl=0 
    natoms_h=0 
    for i, line in enumerate(f): # Read through lines of file
        index = i-2 # Set start of geometry to 2 lines before first line of geometry (to allow for different file formats)
        splt_ln=line.split()
        if line == '\n':
            break
        if splt_ln[0] in ["C","c"]:
            array.append(splt_ln[1:])
            atoms_c.append(['C', 12])
            print(line.rstrip('\n'))
            natoms_c += 1
        if splt_ln[0] in ["N","n"]:
            array_n.append(splt_ln[1:])
            atoms_n.append(['N', 14])
            print(line.rstrip('\n'))
            natoms_n += 1
        if splt_ln[0] in ["H","h"]:
            array_h.append(splt_ln[1:])
            atoms_h.append(['H', 1])
            natoms_h +=1
            print(line.rstrip('\n'))
        if splt_ln[0] in ["Cl","cl", 'CL']:
            array_cl.append(splt_ln[1:])
            atoms_cl.append(['Cl', 35.5])
            print(line.rstrip('\n'))
            natoms_cl += 1
    array = np.array(array)
    array = array.astype(np.float64)
    if natoms_n != 0:
        array_n = np.array(array_n)
        array_n = array_n.astype(np.float64)
        array = np.concatenate((array,array_n))
    if natoms_cl != 0:
        array_cl = np.array(array_cl)
        array_cl = array_cl.astype(np.float64)
        array = np.concatenate((array,array_cl))
    array_h = np.array(array_h)
    array_h = array_h.astype(np.float64)
    if natoms_h==0:
        array_all = array
    else:
        array_all = np.concatenate((array,array_h))
    atoms=atoms_c+atoms_n+atoms_cl+atoms_h
    natoms = natoms_c+natoms_n+natoms_cl
    
    return array, atoms, array_all, natoms_c, natoms_n, natoms_cl, natoms


def distance(array):
    '''
    Takes a list of atomic coordinates for n atoms and returns an (nxn) array of interatomic distances.
    
    Args:
        - array (ndarray): 2D Array of atomic coordinates of all heavy atoms (C then N then Cl) in Angstrom.
    Returns:
        -dist_array (ndarray): 2D Array of interatomic distances in Angstrom.
    '''
    n = array.shape[0]
    dist_array = np.zeros((n, n))
    
    # Get upper triangular indices
    upper_tri_indices = np.triu_indices(n, k=1)
    
    # Calculate the distances
    separations = np.linalg.norm(array[upper_tri_indices[0]] - array[upper_tri_indices[1]], axis=1)
    
    # Assign the distances to the upper triangular part
    dist_array[upper_tri_indices] = separations
    
    # Reflect the upper triangular part to the lower triangular part
    dist_array += dist_array.T
    
    return dist_array

def adjacency(dist_array, cutoff):
    '''
    Takes a 2D array of interatomic distances and a cutoff distance, and returns an adjacency matrix and bond list based on 
    whether the distance between atoms is less than the cutoff.
    
    Args:
        - dist_array (ndarray): 2D Array of interatomic distances in Angstrom.
        - cutoff (float): Cutoff distance in Angstrom for considering a bond.
    Returns:
        - adj_mat (ndarray): 2D adjacency matrix. Value of 1 if atoms are considered bonded, 0 otherwise.
        - bond_list (list): List of pairs of atom indices that are bonded.
    '''
    mask = (dist_array < cutoff) & (np.triu(np.ones_like(dist_array, dtype=bool), k=1))  # Upper triangle only (excluding diagonal)
    
    # Create the adjacency matrix
    adj_mat = np.zeros_like(dist_array, dtype=int)
    adj_mat[mask] = 1
    adj_mat = adj_mat + adj_mat.T  # Make it symmetric
    
    # Generate bond list
    bond_list = np.array(np.nonzero(mask)).T.tolist()  
    return adj_mat, bond_list


def array_intersect(lst1, lst2):
    list3 = list(set(lst1).intersection(set(lst2)))
    return list3


def compute_angle(dihedral, coords):
    '''
    Computes the dihedral angle between four atoms given their indices and coordinates.
    
    Args:
        - dihedral: List of four atom indices (k-i-j-l) for which to compute the dihedral angle.
        - coords: 2D array of atomic coordinates for all atoms in the molecule.
    Returns:
        theta: The dihedral angle in degrees.
    '''
    # angle k-i-j-l
    rij = coords[dihedral[2],:]-coords[dihedral[1],:]
    rik = coords[dihedral[0],:]-coords[dihedral[1],:]
    rjl = coords[dihedral[3],:]-coords[dihedral[2],:]
    r1 = np.cross(rij,rik)
    r2 = np.cross(rij,rjl)
    # r1.r2 = |r1||r2|cost
    theta = np.arccos(np.dot(r1,r2)/(linalg.norm(r1)*linalg.norm(r2))) * 180/np.pi
    if theta > 90:
        theta = 180 - theta
    if theta < 0:
        theta = -theta
    return theta


def dihedrals(natoms,atoms,coords,dist_array, cutoff=cutoff, single_bond_cutoff=single_bond_cutoff, single_bond_cutoff_cn=single_bond_cutoff_cn):
    '''
    Computes dihedral angles for all pairs of atoms that are 4 bonds apart and returns a dictionary of 
    average dihedral angles for each bond in the molecule.
    
    Args:
        - natoms (int): Number of atoms in the molecule.
        - atoms (ndarray): List of atom types for each atom in the molecule.
        - coords (ndarray): 2D array of atomic coordinates for all atoms in the molecule.
        - dist_array (ndarray): 2D array of interatomic distances in Angstrom.
        - cutoff (float): Cutoff distance in Angstrom for considering a bond.
        - single_bond_cutoff (float): Cutoff distance in Angstrom for considering a single bond between carbon atoms.
        - single_bond_cutoff_cn (float): Cutoff distance in Angstrom for considering a single bond between nitrogen atoms.
    Returns:
        angles (dict): Dictionary where keys are strings of the form 'i-j' representing a bond between atoms i and j, 
        and values are the average dihedral angle in degrees for that bond.
    '''
    a2, bond_list = adjacency(dist_array,cutoff) # get adjacency matrix and bond list for molecule (1 bond apart)
    a3=np.dot(a2,a2) # get paths of length 2 between atoms (2 bonds apart)
    a4=np.dot(a3,a2) # get paths of length 3 between atoms (3 bonds apart)
    lst=[]
    for i in range(natoms):
        for j in range(i+1,natoms):
            if a4[i,j]!=0 and a3[i,j]==0 and a2[i,j]==0: # Identify atom pairs that are exactly 3 bonds apart
                lst.append([i,j])
                lst.append([j,i])
    angles={}
    for dihedral in lst:
        for bond in bond_list:
            if a2[dihedral[0],bond[0]]==1 and a2[dihedral[1],bond[1]]==1: # Checking for continuous 4-atom chain
                if dist_array[bond[0],bond[1]]>single_bond_cutoff and atoms[bond[0]][0] in ['C','c'] and atoms[bond[1]][0] in ['C','c']:
                    theta=compute_angle([dihedral[0],bond[0],bond[1],dihedral[1]],coords)
                    if '%s-%s'%(bond[0],bond[1]) in angles:
                        angles['%s-%s'%(bond[0],bond[1])].append(theta) # Store angles associated with central bond in a list
                    else:
                        angles.update({'%s-%s'%(bond[0],bond[1]):[theta]})
                elif dist_array[bond[0],bond[1]]>single_bond_cutoff_cn and array_intersect([atoms[bond[0]][0],atoms[bond[1]][0]],['N','n','N2','n2']) in [['N'],['n'],['N2'],['n2']]:
                    theta=compute_angle([dihedral[0],bond[0],bond[1],dihedral[1]],coords)
                    if '%s-%s'%(bond[0],bond[1]) in angles:
                        angles['%s-%s'%(bond[0],bond[1])].append(theta)
                    else:
                        angles.update({'%s-%s'%(bond[0],bond[1]):[theta]})
    for bond in angles:
        avg_angle=sum(angles[bond])/len(angles[bond])
        angles.update({bond:avg_angle}) # Average over all dihedral angles associated with each bond to get one dihedral angle per bond
    return angles 


def re_center(coords, atoms, coords_h):
    '''
    Centers the coordinates of the molecule on the center of mass of the heavy atoms and returns the recentred coordinates and the center of mass.
    
    Args:
        - coords (ndarray): 2D array of atomic coordinates for all heavy atoms in the molecule.
        - atoms (ndarray): List of atomic symbols and atomic numbers for all atoms in the molecule.
        - coords_h (ndarray): 2D array of atomic coordinates for all atoms in the molecule including hydrogen.
    Returns:
        - com (ndarray): 1D array of the x, y, z coordinates of the center of mass of the heavy atoms in the molecule.
        - coords (ndarray): 2D array of atomic coordinates for all heavy atoms in the molecule, recentred with the COM at the origin.
    '''
    com = np.zeros(3)
    summass=0
    for i in range(coords_h.shape[0]):
        com[:] += atoms[i][1]*coords_h[i,:]
        summass += atoms[i][1]
    com[:] /= summass
    for i in range(coords.shape[0]):
        coords[i,:] -= com
    return com, coords

def ntype(array_all, atoms, natoms_c, natoms_n):
    ''' Classifies Nitrogen atoms based on their bonding coordination (number of bonds - 2).Calculates the number of neighbors within a cutoff distance for each 
        Nitrogen atom and returns a list of Nitrogen coordinations then updates the atom labels to reflect their connectivity.

    Args:
        - array_all (numpy.ndarray): (N, 3) array of all atom coordinates.
        - atoms (list): List of atom data, where atoms[i][0] is the element label.
        - natom_c (int): The number of Carbon atoms in the molecule.
        - natoms_n (int): The number of Nitrogen atoms in the molecule.

    Returns:
        - nlist (list): A list of coordination indices (nbonds - 2).
        - atoms (list): The updated atoms list with specific Nitrogen labels.
    '''
    nlist=[]
    for natom in range(natoms_n):
        nbonds=-1 #Prevent counting the N atom with itself
        for iatom in range(array_all.shape[0]):
            distn=0
            for k in range(3):
                distn += (array_all[natom + natoms_c,k]-array_all[iatom,k])**2
            distn=np.sqrt(distn)
            if distn < cutoff:
                nbonds+=1
        nlist.append(nbonds-2)
        atoms[natom+natoms_c][0]='N'+str(nbonds-1)
    return nlist, atoms   
   

def conec(ncarb, dist_array, natoms_c):
    '''
    Group atoms in alternant hydrocarbons into starred and unstarred lists.
    
    Args:
        - ncarb (int): Number of carbon atoms in the molecule.
        - dist_array (ndarray): 2D Array of interatomic distances in Angstrom.
    Returns:
        - star (list): List of indices of starred atoms.
        - unst (list): List of indices of unstarred atoms.
    '''
    ncarb_pi = natoms_c
    assignment = [-1] * ncarb_pi  # -1 = unassigned, 0 = unstarred, 1 = starred

    for start in range(ncarb_pi):
        if assignment[start] != -1:
            continue  # already assigned, handle disconnected fragments

        # BFS from this starting atom
        queue = [start]
        assignment[start] = 1  # starred

        while queue:
            atom = queue.pop(0)
            for neighbour in range(ncarb_pi):
                if neighbour == atom:
                    continue
                if dist_array[atom, neighbour] < cutoff:
                    if assignment[neighbour] == -1:
                        # Assign opposite set
                        assignment[neighbour] = 1 - assignment[atom]
                        queue.append(neighbour)
                    elif assignment[neighbour] == assignment[atom]:
                        raise ValueError(
                            f"Non-alternant system: atom {neighbour} conflicts "
                            f"with atom {atom} — odd-membered ring detected"
                        )

    star = [i for i in range(ncarb_pi) if assignment[i] == 1]
    unst = [i for i in range(ncarb_pi) if assignment[i] == 0]

    # Convention: starred set is the larger one
    if len(star) < len(unst):
        print('Swapping starred and unstarred atoms ...')
        star, unst = unst, star

    print(f'\nStarred atoms: {star}')
    print(f'Unstarred atoms: {unst}\n')
    return star, unst

# Routine to group bonding and antibonding orbitals into coulson-rushbrooke pairs
def order_orbs(ncarb, orbs, orb_energies, alt):
    '''
    Pairs bonding and antibonding orbitals based on Coulson-Rushbrooke symmetry.In alternant hydrocarbons, orbitals occur in pairs with energies 
    +/- E and identical coefficient magnitudes. This function identifies those pairs by matching energy levels and verifying orbital coefficient magnitudes.

    Args:
        ncarb (int): Number of carbon atoms/total orbitals.
        orbs (numpy.ndarray): Matrix of orbital coefficients (columns are orbitals).
        orb_energies (numpy.ndarray): Array of orbital energies.
        alt (bool): Current alternacy status of the molecule.

    Returns:
        - pairs_list (list of lists): Indices of paired [bonding, antibonding] orbitals.
        - alt (bool): Updated alternacy status (set to False if pairing fails).
    '''
    print(' ')
    nbond = int((ncarb-1)/2)
    anti_list = list(range(nbond+1,ncarb))
    anti_list.reverse()
    pairs_list = []           
    search = False      
    for ibond in range(nbond):
        if abs(orb_energies[ibond+1] - orb_energies[ibond]) < 1e-6:
            print("degenerate orbitals %d and %d!"%(ibond+1,ibond+2)) #CHECK
            search=True
        elif ibond > 0:
            if abs(orb_energies[ibond-1] - orb_energies[ibond]) < 1e-6:
                print("degenerate orbitals %d and %d!"%(ibond+1,ibond)) #CHECK
                search=True
        if search == False: # If orbital ibond is not degenerate, assign as Coulson-Rushbrooke pair with opposite orbital in energy ordering
            ianti = ncarb-ibond-1
            pairs_list.append([ibond,ianti])
            anti_list.remove(ianti)
            print('Coulson-Rushbrooke pair orbs %d, %d\n'%(ibond+1,ianti+1))
        if search == True:
            print('Searching for correct antibonding pair for orb %d ...'%(ibond+1))
            for ianti in anti_list: # guess orbital pair
                print("Trying antibonding orbital", ianti+1)
                if abs(abs(orb_energies[ianti]) - abs(orb_energies[ibond])) < 1e-6: #if energies match
                    print("Absolute energies %4f eV and %4f eV match, difference = %4f eV"%(orb_energies[ibond],orb_energies[ianti],abs(abs(orb_energies[ianti]) - abs(orb_energies[ibond]))))
                    pairs = tuple(zip(orbs[:,ibond],orbs[:,ianti])) # pairs of coeffs in bonding and antibonding orbital pair
                    for n,(icoeff,jcoeff) in enumerate(pairs): #compare coeffs
                        if abs(abs(icoeff) - abs(jcoeff)) > 1e-4: # if coeffs are not equal in magnitude start n loop again
                            print("Magnitude of coeffs not equal",abs(abs(icoeff) - abs(jcoeff)))
                            print('Searching for correct antibonding pair for orb %d ...'%(ibond+1))
                            break
                        if n == ncarb-1: # if all coefficients of two orbitals match in magnitude
                            pairs_list.append([ibond,ianti])
                            anti_list.remove(ianti)
                            print("Magnitudes of all orbital coefficients are within 1e-4")
                            print('Coulson-Rushbrooke pair orbs %d, %d\n'%(ibond+1,ianti+1))
                            search = False
                    if search == False:
                        break
                else:
                    print("absolute energies %4f eV and %4f eV do not match, difference = %4f eV"%(orb_energies[ibond],orb_energies[ianti],abs(abs(orb_energies[ianti]) - abs(orb_energies[ibond]))))
                if ianti==anti_list[len(anti_list)-1] and search==True: # if all antibonding orbitals are tried and none match the bonding orbital, warn user and switch off alternacy
                    print("\nWARNING!!: Could not find Coulson-Rushbrooke pair for orbital %d, switching off alternacy. If molecule is alternant, try lowering orbital coefficient matching threshold and re-run calculation. \
                          But examine the orbitals first!"%(ibond+1))
                    alt=False
                    return pairs_list, alt
    return pairs_list, alt                   
  


def orb_sign(orbs,orb_energies,nelec,dist_array,natoms_c,alt):
    '''
    Adjusts orbital phases to satisfy alternant hydrocarbon symmetry.Ensures that starred atoms retain their sign 
    across a pair, while unstarred atoms undergo a phase inversion in the antibonding orbital.

    Args:
        orbs (ndarray): Matrix of orbital coefficients (rows=atoms, cols=orbitals).
        orb_energies (ndarray): Array of orbital energies.
        nelec (int): Total number of electrons in the system.
        dist_array (ndarray): Matrix of inter-atomic distances.
        natoms_c (int): Number of carbon atoms in the molecule.
        alt (bool): Alternacy status flag.

    Returns:
        orbs (ndarray): The orbital coefficient matrix with standardized phases.
    '''
    if alt==True:
        print('\nGrouping orbitals according to alternacy symmetry...')
        ncarb = orbs.shape[0]
        average_somo_energy = (np.abs(orb_energies[int((nelec-1)/2)] - orb_energies[int((nelec+1)/2)])/2) / 2
        for i in range(orb_energies.shape[0]):
            orb_energies[i] = orb_energies[i] - average_somo_energy
        orb_list,alt = order_orbs(ncarb,orbs,orb_energies,alt)
    if alt==True:
        star,unst = conec(ncarb,dist_array,natoms_c)
        print('\nInverting orbital phases according to alternacy symmetry...\n')
        for i,ip in orb_list:
            for satom in star:
                if np.sign(orbs[satom,i]) != np.sign(orbs[satom,ip]):
                    orbs[satom,ip] = -1*orbs[satom,ip]
                    print('flipping sign orb '+str(ip)+' starred atom '+str(satom))
            for uatom in unst:
                if np.sign(orbs[uatom,i]) == np.sign(orbs[uatom,ip]):
                    print('flipping sign orb '+str(ip)+' unstarred atom '+str(uatom))
                    orbs[uatom,ip] = -1*orbs[uatom,ip]
    if np.sign(orbs[0,0]) == -1: # if orbital 0 has all -ve coeffs, make all +ve and invert all coeffs on all other orbitals
        orbs = np.multiply(orbs,-1) # as per Tim's alteration
    return orbs


def t_term(dist_array,natoms_c,natoms_n,natoms,n_list,theta,params):
    '''
    Forms off-diagonal hopping contribution for PPP Hamiltonian, using cutoff to determine nearest neighbors.
    
    Args:
    - dist_array (ndarray): 2D array of interatomic distances in Angstrom.
    - natoms_c (int): Number of carbon atoms in the molecule.
    - natoms_n (int): Number of nitrogen atoms in the molecule.
    - natoms (int): Total number of heavy atoms in the molecule.
    - n_list (list): List of nitrogen coordination indices (nbonds - 2).
    - theta (dict): Dictionary of average dihedral angles for each bond in the molecule, 
                    with keys as 'i-j' representing a bond between atoms i and j.
    - params (list of lists): List of PPP parameter sets for carbon, nitrogen and chlorine atoms, 
                              where each parameter set is a list containing values for A, b, alpha, U, r0 etc.
    
    Returns:
    - array (ndarray): 2D array representing the off-diagonal hopping contribution to the PPP Hamiltonian, with shape (natoms, natoms). 
                       Non-zero values correspond to hopping terms between atoms that are considered bonded based on the cutoff distance, 
                       and are calculated using the provided parameters and dihedral angles where applicable.
    '''
    A=params[0][0]
    b=params[0][1]
    alphan=params[1][0]
    Acn=params[1][1]
    bcn=params[1][2]
    alphan2=params[2][0]
    Acn2=params[2][1]
    bcn2=params[2][2] # change for cn2 hopping ratio
    alphacl=params[3][0]
    Accl=params[3][1]
    bccl=params[3][2]
    print("\nCarbon 1e params: A = %f b = %f"%(A,b))
    print("\nNitrogen 1e params: alphan2 = %f Acn2 = %f bcn2 = %f"%(alphan2,Acn2,bcn2))
    print("\nChlorine 1e params: alphacl = %f Accl = %f bccl=%f"%(alphacl,Accl,bccl))
    array=np.zeros_like(dist_array)
    # C-C hopping 
    ntheta=0
    for i in range (natoms_c):
        for j in range (i+1,natoms_c):
            if dist_array[i,j]<cutoff:
                if '%s-%s'%(i,j) in theta:
                    #print("Used Theta %d %f deg. atoms %d %d"%(ntheta,theta['%s-%s'%(i,j)],i+1,j+1))
                    array[i,j]=abs(np.cos(np.pi*theta['%s-%s'%(i,j)]/180))*A*np.exp(-b*dist_array[i,j])
                    ntheta+=1
                else:
                    array[i,j]=A*np.exp(-b*dist_array[i,j])
                array[j,i]=array[i,j]  
                #ntheta+=1
    # N and Cl hopping 
    # C-N hopping
    for i in range (natoms_c):
        for j in range (natoms_c,natoms_c+natoms_n):
            if dist_array[i,j]<cutoff:
                if n_list[j-natoms_c]==0:
                    if '%s-%s'%(i,j) in theta:
                        #print("Used Theta %d %f deg. atoms %d %d"%(ntheta,theta['%s-%s'%(i,j)],i+1,j+1))
                        array[i,j]=abs(np.cos(np.pi*theta['%s-%s'%(i,j)]/180))*Acn*np.exp(-bcn*dist_array[i,j])
                        ntheta+=1
                    else:
                       array[i,j]= Acn*np.exp(-bcn*dist_array[i,j])
                    print("C-N1 bond")
                elif n_list[j-natoms_c]==1:
                    if '%s-%s'%(i,j) in theta:
                        #print("Used Theta %d %f deg. atoms %d %d"%(ntheta,theta['%s-%s'%(i,j)],i+1,j+1))
                        array[i,j]=abs(np.cos(np.pi*theta['%s-%s'%(i,j)]/180))*Acn2*np.exp(-bcn2*dist_array[i,j])  
                        ntheta+=1
                    else:
                        array[i,j]=Acn2*np.exp(-bcn2*dist_array[i,j]) 
                    print("C-N2 bond")
                array[j,i] = array[i,j]
     # N-N hopping 
    # for i in range (natoms_c,natoms_c+natoms_n):
     #    for j in range (i+1,natoms_c+natoms_n):
        #     if dist_array[i,j]<cutoff:
        #         if n_list[i-natoms_c]==0 and n_list[j-natoms_c]==0:
           #          array[i,j] = tnn
             #        print("N1-N1 bond")
              #   elif n_list[i-natoms_c]+n_list[j-natoms_c]==1:
               #      array[i,j] = tnn2
               #      print("N1-N2 bond")
               #  elif n_list[i-natoms_c]==1 and n_list[j-natoms_c]==1:
                #     array[i,j] = tn2n2
                #     print("N2-N2 bond")
                # array[j,i] = array[i,j]  
    # C-Cl hopping
    for i in range (natoms_c):
        for j in range (natoms_c+natoms_n,natoms):
            if dist_array[i,j]<ccl_cutoff:
                array[i,j]=Accl*np.exp(-bccl*dist_array[i,j])
                array[j,i]=array[i,j]  
                ntheta+=1
                print('C-Cl bond')
                array[j,i] = array[i,j]
    # N alpha (diagonal) terms
    for i in range(natoms_c,natoms_c+natoms_n):
        if n_list[i-natoms_c]==0:
            array[i,i] += alphan 
            print("N1 atom %d"%(i+1))
        elif n_list[i-natoms_c]==1:
            array[i,i] += alphan2
            print("N2 atom %d"%(i+1))
    # Cl alpha (diagonal) terms
    for i in range(natoms_c+natoms_n,natoms):
        array[i,i]+=alphacl 
        print("Cl atom %d"%(i+1))
    return array

def v_term(dist_array,natoms_c,natoms_n,natoms,n_list,params):
    '''
    Forms two-body repulsion contribution for PPP Hamiltonian, with short and long range terms.
    
    Args:
    - dist_array (ndarray): 2D array of interatomic distances in Angstrom.
    - natoms_c (int): Number of carbon atoms in the molecule.
    - natoms_n (int): Number of nitrogen atoms in the molecule.
    - natoms (int): Total number of heavy atoms in the molecule.
    - n_list (list): List of nitrogen coordination indices (nbonds - 2).
    - params (list of lists): List of PPP parameter sets for carbon, nitrogen and chlorine atoms, 
                              where each parameter set is a list containing values for A, b, alpha, U, r0 etc.
    
    Returns:
    - array (ndarray): 2D array giving the repulsion contribution to the PPP Hamiltonian, with shape (natoms, natoms).
    '''
    U=params[0][2]
    r0=params[0][3]
    Unn=params[1][3]
    r0nn=params[1][4]
    Un2n2=params[2][3]
    r0n2n2=params[2][4]
    Uclcl=params[3][3]
    r0clcl=params[3][4]
    Ucn=(Unn+U)/2
    Ucn2=(Un2n2+U)/2
    Uccl=(U+Uclcl)/2
    Uncl=(Unn+Uclcl)/2
    Un2cl=(Un2n2+Uclcl)/2
    Unn2=(Un2n2+Unn)/2
    r0cn=(r0nn+r0)/2
    r0cn2=(r0n2n2+r0)/2
    r0ccl=(r0+r0clcl)/2
    r0ncl=(r0nn+r0clcl)/2
    r0n2cl=(r0n2n2+r0clcl)/2
    r0nn2=(r0n2n2+r0nn)/2
    print("\nCarbon 2e params: U = %f r0 = %f"%(U,r0))
    print("\nNitrogen 2e params: Un2n2 = %f r0n2n2 = %f"%(Un2n2,r0n2n2))
    print("\nChlorine 2e params: Uclcl= %f r0clcl = %f"%(Uclcl,r0clcl))
    print("\nMixed 2e params: Ucn2 = %f Uccl = %f Un2cl = %f r0cn2 =%f r0ccl = %f r0n2cl = %f"%(Ucn2,Uccl,Un2cl,r0cn2,r0ccl,r0n2cl))
    array=np.zeros_like(dist_array)
    # C-C repulsion
    for i in range (natoms_c):
        for j in range (i+1,natoms_c):
            array[i,j]=U/(1+dist_array[i,j]/r0)
            array[j,i]=array[i,j] 
        array[i,i]=U
    # C-N repulsion
    for i in range (natoms_c):
        for j in range (natoms_c,natoms_c+natoms_n):
             if n_list[j-natoms_c]==0:
                 array[i,j]=Ucn/(1+dist_array[i,j]/r0cn)
             elif n_list[j-natoms_c]==1:
                 array[i,j]=Ucn2/(1+dist_array[i,j]/r0cn2)
             array[j,i]=array[i,j] 
    # N-N repulsion
    for i in range (natoms_c,natoms_c+natoms_n):
        for j in range (i+1,natoms_c+natoms_n):
             if n_list[i-natoms_c]==0 and n_list[j-natoms_c]==0:
                 array[i,j]=Unn/(1+dist_array[i,j]/r0nn)
             elif n_list[i-natoms_c]+n_list[j-natoms_c]==1:
                 array[i,j]=Unn2/(1+dist_array[i,j]/r0nn2)
             elif n_list[i-natoms_c]==1 and n_list[j-natoms_c]==1:
                 array[i,j]=Un2n2/(1+dist_array[i,j]/r0n2n2)
             array[j,i]=array[i,j]
        # diagonal terms
        if n_list[i-natoms_c]==0:
            array[i,i]=Unn
        elif n_list[i-natoms_c]==1:
            array[i,i]=Un2n2
    # C-Cl repulsion
    for i in range(natoms_c):
        for j in range(natoms_c+natoms_n,natoms):
            array[i,j]=Uccl/(1+dist_array[i,j]/r0ccl)
            array[j,i]=array[i,j]
    # N-Cl repulsion
    for i in range(natoms_c,natoms_c+natoms_n):
        for j in range(natoms_c+natoms_n,natoms):
            if n_list[i-natoms_c]==0:
                array[i,j]=Uncl/(1+dist_array[i,j]/r0ncl)
            elif n_list[i-natoms_c]==1:
                array[i,j]=Un2cl/(1+dist_array[i,j]/r0n2cl)
            array[j,i]=array[i,j]
    # Cl-Cl repulsion
    for i in range(natoms_c+natoms_n,natoms):
        for j in range(i+1,natoms):
            array[i,j]=Uclcl/(1+dist_array[i,j]/r0clcl)
            array[j,i]=array[i,j]
        array[i,i]=Uclcl
    print(array[0,:])
    return array


def density(orbs, ndocc):
    '''
    Function to form and return density matrix for diradicals.
    
    Args:
        - orbs (ndarray): 2D array of orbital coefficients (rows=atoms, cols=MOs).
        - ndocc (int): Number of doubly-occupied MOs.
    Returns:
        - density: 2D array representing the density matrix of the system, with shape (natoms, natoms). 
                   The density matrix is constructed by summing the contributions from the doubly occupied orbitals twice and
                   once from the singly occupied molecular orbitals (SOMOs).
    '''
    prefactor_matrix = np.diag((np.full(ndocc + 2, 2)))
    prefactor_matrix[[ndocc, ndocc + 1], [ndocc, ndocc + 1]] = 1 # SOMOs are multiplied by 1 rather than 2.
    density = orbs[:,:ndocc + 2] @ (prefactor_matrix @ orbs[:,:ndocc + 2].T) # P = C_occ *  Prefactor * C_occ^T  
    return density


def fock(repulsion,hopping,density,natoms_c,natoms_n,natoms,nlist):
    '''
    Function to form and return open-shell Fock matrix
    
    Args:
        - repulsion (ndarray): 2D array representing the two-body repulsion integrals for the PPP Hamiltonian, with shape (natoms, natoms).
        - hopping (ndarray): 2D array representing the one-body hopping integrals for the PPP Hamiltonian, with shape (natoms, natoms).
        - density (ndarray): 2D array representing the density matrix of the system, with shape (natoms, natoms).
        - natoms_c (int): Number of carbon atoms in the molecule.
        - natoms_n (int): Number of nitrogen atoms in the molecule.
        - natoms (int): Total number of heavy atoms in the molecule.
        - nlist (list): List of nitrogen coordination indices (nbonds - 2).
    Returns:
        - fock_mat (ndarray): 2D array representing the open-shell Fock matrix of the system.
    '''
    fock_mat=np.zeros_like(repulsion)
    for i in range (natoms):
        for j in range (i,natoms):
            if i == j:
                mylist = list(range(natoms))
                mylist.remove(i)
                # Determining atom type
                for atom in mylist:
                    if atom >= natoms_c and atom < natoms_c + natoms_n: # N atom
                        zk = nlist[atom - natoms_c] + 1
                    elif atom >= natoms_c + natoms_n: # Cl atom
                        zk = 2
                    else: # Carbon
                        zk = 1
                    fock_mat[i,j] += (density[atom,atom] - zk) * repulsion[i,atom] # Contribution to electron energy on given atom from electrons on other atoms
                fock_mat[i,j] += 0.5 * density[i,j] * repulsion[i,j] # Self-interaction energy for electron energy on given atom
            else:
                fock_mat[i,j] = -0.5 * density[i,j] * repulsion[i,j] # Exchange contribution for electrons on different atoms
                fock_mat[j,i] = fock_mat[i,j]
    fock_mat = fock_mat + hopping
    return fock_mat


def energy(hopping,repulsion,fock_mat,density,orbs,ndocc):
    """Calculates the total SCF energy of a fictituous system that is close to the energy of the open-shell singlet ground state.

    Returns:
        float: The total calculated SCF energy of the system from PPP theory.
    """
    return 0.5 * (np.dot(density.flatten(), hopping.flatten()) + np.dot(density.flatten(), fock_mat.flatten()))

def cartesian_operators(coords,hf_orbs):
    
    dip1el = np.einsum('ui,uk,uj->ijk', hf_orbs, coords, hf_orbs) * tobohr
    
    x_operator = dip1el[:, :, 0]
    y_operator = dip1el[:, :, 1]
    z_operator = dip1el[:, :, 2]
    full_cartesian_operator = dip1el
    
    return full_cartesian_operator, x_operator, y_operator, z_operator

class DIIS:
    def __init__(self, max_iter=50):
        self.max_iter = max_iter
        self.fock_list = []
        self.error_list = []
    
    def get_extrapolated_fock(self, F, D):
        error = F @ D - D @ F
        
        self.fock_list.append(F.copy())
        self.error_list.append(error)
        
        if len(self.fock_list) > self.max_iter:
                self.fock_list.pop(0)
                self.error_list.pop(0)
        n = len(self.fock_list)
        if n < 2:
            return F

        B = np.zeros((n + 1, n + 1))
        for i in range(n):
            for j in range(i, n):
                val = np.sum(self.error_list[i] * self.error_list[j])
                B[i, j] = B[j, i] = val
        
        # Constraint: sum(c_i) = 1
        B[n, :n] = -1
        B[:n, n] = -1
        B[n, n] = 0

        rhs = np.zeros(n + 1)
        rhs[n] = -1

        try:
            coeffs = np.linalg.solve(B, rhs)
        except np.linalg.LinAlgError:
            return F

        F_ext = np.zeros_like(F)
        for i in range(n):
            F_ext += coeffs[i] * self.fock_list[i]

        return F_ext
    
    
def get_level_shifted_fock(Fock_mat, coeffs, ndocc, shift_virt, shift_somo=0):
    """
    F: Standard Fock matrix
    D: Current Density matrix (2 for Docc, 1 for SOMOs)
    shift_ext: The shift value 'b' (e.g., 0.2 Hartrees)
    """
    P_docc = coeffs[:, :ndocc] @ coeffs[:, :ndocc].T
    P_somo = coeffs[:, ndocc:ndocc+2] @ coeffs[:, ndocc:ndocc+2].T
    P_virt = coeffs[:, ndocc+2:] @ coeffs[:, ndocc+2:].T
    assert np.allclose(P_docc + P_somo + P_virt, np.eye(coeffs.shape[0]), atol=1e-8), \
    "Projectors do not sum to identity — check MO coefficients are orthonormal"
    
    F_shifted = Fock_mat + shift_virt * P_virt
    
    if shift_somo > 0:
        F_shifted += shift_somo * P_somo
    
    return F_shifted


def delocalise_somos(orbs, i, j):
    """
    i, j: indices of the two localized SOMOs
    """
    # Create copies to avoid overwriting mid-calculation
    L = orbs[:, i].copy()
    R = orbs[:, j].copy()
    
    # Apply the 45-degree unitary rotation
    orbs[:, i] = (L + R) / np.sqrt(2)
    orbs[:, j] = (L - R) / np.sqrt(2)
    
    return orbs


#Main HF function
def main_scf(file, params, maxcycles=5000, d_tol=5e-15):
    '''
    Main Hartree-Fock function to perform SCF calculation for a radical molecule using the ExROPPP method.
    For molecules that struggle to converge, a level shift can be applied...
    
    Args:
        - file (str): The filename of the input geometry file for the radical molecule.
        - params (dict): The dictionary of PPP parameters for Carbon, Nitrogen and Chlorine.
        - maxcycles (int): The maximum number of SCF cycles to perform.
        - d_tol (float): The convergence tolerance for the density matrix.
        
    '''
    print("                    ---------------------------------")
    print("                    | Radical ExROPPP Calculation |")
    print("                    ---------------------------------\n")
    print("Molecule: "+str(file)+" radical\n")
    #read in geometry and form distance matrix
    try:
        coord,atoms_array,coord_w_h,natoms_c,natoms_n,natoms_cl,natoms = read_geom(file)
    except FileNotFoundError:
        file = f'Molecules/{file}'
        coord,atoms_array,coord_w_h,natoms_c,natoms_n,natoms_cl,natoms = read_geom(file)
    dist_array = distance(coord)
    n_list,atoms = ntype(coord_w_h,atoms_array,natoms_c,natoms_n)
    nelec = natoms + sum(n_list) + natoms_cl #each pyrolle type N contributes 1 additional e-, so does Cl
    ndocc = int((nelec-1)/2) # no. of doubly-occupied orbitals
    print("\nThere are %d heavy atoms."%natoms)
    print("There are %d electrons in %d orbitals.\n"%(nelec,natoms))
    #compute array of dihedral angles for given molecule (originaly used predefined dictionary of angles but now they are computed directly)
    angles = dihedrals(natoms_c+natoms_n+natoms_cl,atoms_array,coord,dist_array)
    #call functions to get 1/2-body "integrals"
    hopping = t_term(dist_array,natoms_c,natoms_n,natoms,n_list,angles,params)
    repulsion = v_term(dist_array,natoms_c,natoms_n,natoms,n_list,params)
    #Diagonalize Huckel Hamiltonian to form initial density guess
    guess_evals, guess_orbs = np.linalg.eigh(hopping)
    guess_dens = density(guess_orbs,ndocc)
    #iterate until convergence 
    energy1=0
    diis = DIIS(max_iter=10)
    level_shift = False
    shift_virt = 0
    shift_somo = 0
    use_diis = False
    damping = False
    print("\n-------------------------------------")
    print("Restricted Open-shell PPP Calculation")
    print("-------------------------------------\n")
    print("Starting SCF cycle...\n")
    print("Iter   Energy        Dens Change      Energy Change")
    print("-----------------------------------------------------")
    for iter in range (maxcycles):
        if iter == maxcycles-1:
            print(f"\nEnergy not converged after {maxcycles} cycles")
            break
        fock_mat = fock(repulsion, hopping, guess_dens, natoms_c, natoms_n, natoms, n_list)
        if iter > 50 and not damping and conv_crit > 0.01:
            damping = True
            alpha = 0.1
            print(f'\n---Applying damping with alpha={alpha} to aid convergence---\n')
        if damping:
            if conv_crit < 1e-5:
                alpha = 0.2
            elif conv_crit < 1e-10:
                alpha = 0.5
        
        if iter > 500 and conv_crit > 0.001:
            use_diis = True
        if iter > 1000 and conv_crit > 1e-4:
            use_diis = True
        if iter > 2000 and conv_crit > 1e-5:
            use_diis = True
        
        if damping:
            fock_mat = alpha * fock_mat + (1 - alpha) * guess_fock
        if level_shift:
            fock_mat = get_level_shifted_fock(fock_mat, orbs, ndocc, shift_virt = shift_virt, shift_somo = shift_somo)
        if use_diis:
            fock_mat = diis.get_extrapolated_fock(fock_mat, guess_dens)
        evals, orbs = np.linalg.eigh(fock_mat)
        dens = density(orbs,ndocc)
        energy2 = energy(hopping, repulsion, fock_mat, dens, orbs, ndocc)
        conv_crit = np.absolute(guess_dens-dens).max()
        print(iter, energy2, conv_crit, energy2 - energy1)
        if conv_crit < d_tol:
            break
        energy1 = energy2
        guess_dens = dens
        guess_fock = fock_mat
        
        '''
        if iter == maxcycles - 5:
            print("\n--------------------------")
            print("Converged ROPPP Orbitals")
            print("--------------------------\n")
            natoms=np.shape(coord)[0]
            for iorb in range(natoms):
                print('orbital number', iorb + 1, 'energy', evals[iorb]-evals[int((nelec-1)/2)])
                print(np.around(orbs[:, iorb], decimals=2))
        '''
    
    SOMO1 = ndocc
    SOMO2 = ndocc + 1
    
    #assert abs(evals[SOMO1] - evals[SOMO2]) < 1e-12, "SOMOs are not degenerate!"
    
    '''
    print('\nEnforcing Spatial Symmetry in x for denerate SOMOs\n')
    x_operator = cartesian_operators(coord,orbs)[1]
    SOMOs_in_x_basis = x_operator[np.ix_([SOMO1, SOMO2], [SOMO1, SOMO2])]
    _, x_rotation = np.linalg.eigh(SOMOs_in_x_basis)
    SOMOs_x_rot = np.dot(orbs[:, [SOMO1, SOMO2]], x_rotation)
    orbs[:, [SOMO1, SOMO2]] = SOMOs_x_rot
    
    print('\nEnforcing Spatial Symmetry in y for denerate SOMOs\n')
    y_operator = cartesian_operators(coord,orbs)[2]
    SOMOs_in_y_basis = y_operator[np.ix_([SOMO1, SOMO2], [SOMO1, SOMO2])]
    _, y_rotation = np.linalg.eigh(SOMOs_in_y_basis)
    SOMOs_y_rot = np.dot(orbs[:, [SOMO1, SOMO2]], y_rotation)
    orbs[:, [SOMO1, SOMO2]] = SOMOs_y_rot
    
    print('\nEnforcing Spatial Symmetry in z for denerate SOMOs\n')
    z_operator = cartesian_operators(coord,orbs)[3]
    SOMOs_in_z_basis = z_operator[np.ix_([SOMO1, SOMO2], [SOMO1, SOMO2])]
    _, z_rotation = np.linalg.eigh(SOMOs_in_z_basis)
    SOMOs_z_rot = np.dot(orbs[:, [SOMO1, SOMO2]], z_rotation)
    orbs[:, [SOMO1, SOMO2]] = SOMOs_z_rot
    '''
    '''
    print('\nLocalising SOMOs')
    orbs = delocalise_somos(orbs, SOMO1, SOMO2)
    density_rot = density(orbs, ndocc)
    fock_mat = fock(repulsion, hopping, density_rot, natoms_c, natoms_n, natoms, n_list)
    energy2 = energy(hopping, repulsion, fock_mat, density_rot, orbs, ndocc)
    '''
    print('ENERGY0:', energy2)
    return coord,atoms_array,coord_w_h,dist_array,nelec,ndocc,n_list,natoms_c,natoms_n,natoms_cl,energy2,hopping,repulsion,evals,orbs,fock_mat



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
    for i in range (Natoms):
            for j in range (i,Natoms):
                    two_body_4i[i,i,j,j]=two_body[i,j]
                    two_body_4i[j,j,i,i]=two_body[i,j]
    #four index transformation
    two_body_mo = np.einsum("ia, jb, kc, ld, ijkl -> abcd",
                             hf_orbs, hf_orbs, hf_orbs, hf_orbs, two_body_4i, optimize= 'optimal' )
    return two_body_mo


    
def broaden(FWHM,osc,energy):
    if brdn_typ == 'wavelength' and line_typ == 'lorentzian':
        eqn="+%04.3f*1/(1+((%04.3f-x)/(%s/2))**2)" %(osc,evtonm/energy,FWHM)
    elif brdn_typ == 'energy' and line_typ == 'lorentzian':
        eqn="+%04.3f*1/(1+((%04.3f-x)/(0.5*%s*%04.3f*x))**2)"  %(osc,evtonm/energy,FWHM,evtonm/energy)
    elif brdn_typ == 'energy' and line_typ == 'gaussian':
        eqn="+%04.3f*exp(-((%04.3f-x)/(0.5*%s*%04.3f*x))**2)" %(osc,evtonm/energy,FWHM,evtonm/energy)
    return eqn


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




def hetero_xcis_ham_rot(ndocc, norbs, energy0, orb_energies, rep_tens):
    '''
    Form the CIS Hamiltonian matrix in the rotated CSF basis. Matrix elements on off-diagonals are typically 2e integrals, found in the working doc.
    Note that the basis is given in the working doc, i.e the ordering of CSFs. We have singlets then triplets, making the Hamiltonian block diagonal.
    
    Args:
        ndocc (int): Number of doubly occupied orbitals.
        energy0 (float): Ground state HF energy.
        orb_energies (array): HF orbital energies.
        rep_tens (array): 4D tensor giving two-electron repulsion integrals in the MO basis.
    Returns:
    '''
    SOMO1 = ndocc # Index of SOMO1
    SOMO2 = ndocc + 1 # Index of SOMO2
    nvirt = norbs - ndocc - 2 # Number of virtual orbitals
    nstates = 6 * (ndocc * nvirt) + 4 * ndocc + 4 * nvirt + 4  # 6 * ndocc * nvirt doubles (HOMO to LUMO), 4 * ndocc singles (HOMO to SOMO), 4 * nvirt singles (SOMO to LUMO), 4 reference configurations (OS GSs and Zwitterions)
    xcish = np.zeros((nstates,nstates))
    
    ################# SINGLET BLOCK ######################
    #1 <OS1|H|OS1>
    xcish[0,0] = energy0 - 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) + (1.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1])
    #2 <OS1|H|ZW->
    xcish[0,1] = rep_tens[SOMO1,SOMO2,SOMO1,SOMO1] - rep_tens[SOMO1,SOMO2,SOMO2,SOMO2]
    xcish[1,0] = xcish[0,1]
    #3 <OS1|H|ZW+>
    xcish[0,2] = 0
    xcish[2,0] = xcish[0,2]
    #4 <OS1|H|HS1> 
    block_index = 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        xcish[0,col] = 1.5 * rep_tens[o_orb,SOMO2,SOMO2,SOMO1] - 0.5 * rep_tens[o_orb,SOMO1,SOMO1,SOMO1]
        xcish[col,0] = xcish[0,col]
    #5 <OS1|H|HS2> 
    block_index = ndocc + 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        xcish[0,col] = 0.5 * rep_tens[o_orb,SOMO2,SOMO2,SOMO2] - 1.5 * rep_tens[o_orb,SOMO1,SOMO1,SOMO2]
        xcish[col,0] = xcish[0,col]
    #6 <OS1|H|SL1>
    block_index = 2 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        xcish[0,col] = 1.5 * rep_tens[v_orb,SOMO2,SOMO2,SOMO1] - 0.5 * rep_tens[v_orb,SOMO1,SOMO1,SOMO1]
        xcish[col,0] = xcish[0,col]
    #7 <OS1|H|SL2>
    block_index = nvirt + 2 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        xcish[0,col] = 0.5 * rep_tens[v_orb,SOMO2,SOMO2,SOMO2] - 1.5 * rep_tens[v_orb,SOMO1,SOMO1,SOMO2]
        xcish[col,0] = xcish[0,col]
    #8 <OS1|H|HL1> = 0
    #9 <OS1|H|HL2>
    block_index =  (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for col in range(block_index, block_index + (ndocc * nvirt)):
        o_orb = (col - block_index) // nvirt # Increase o_orb after every ndocc cols
        v_orb = (col - block_index) % nvirt + (SOMO2 + 1) # Increase v_orb then reset after ndocc cols
        xcish[0,col] = np.sqrt(1.5) * (rep_tens[o_orb, SOMO2, SOMO2, v_orb] - rep_tens[o_orb, SOMO1, SOMO1, v_orb])
        xcish[col,0] = xcish[0,col]

    
    #10 <ZW->|H|ZW->
    xcish[1,1] = energy0 + 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) - 0.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1] - rep_tens[SOMO1, SOMO1, SOMO2, SOMO2]
    #11 <ZW->|H|ZW+>
    xcish[1,2] = orb_energies[SOMO1] - orb_energies[SOMO2]
    xcish[2,1] = xcish[1,2]
    #12 <ZW->|H|HS1>
    block_index = 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        xcish[1,col] = rep_tens[o_orb,SOMO2,SOMO1,SOMO1] + 0.5 * rep_tens[o_orb,SOMO1,SOMO1,SOMO2] - 0.5 * rep_tens[o_orb,SOMO2,SOMO2,SOMO2]
        xcish[col,1] = xcish[1,col]
    #13 <ZW->|H|HS2>
    block_index = ndocc + 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        xcish[1,col] = rep_tens[o_orb,SOMO1,SOMO2,SOMO2] + 0.5 * rep_tens[o_orb,SOMO2,SOMO2,SOMO1] - 0.5 * rep_tens[o_orb,SOMO1,SOMO1,SOMO1]
        xcish[col, 1] = xcish[1,col]
    #14 <ZW-|H|SL1>
    block_index = 2 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        xcish[1,col] = rep_tens[v_orb,SOMO2,SOMO1,SOMO1] + 0.5 * rep_tens[v_orb,SOMO1,SOMO1,SOMO2] - 0.5 * rep_tens[v_orb,SOMO2,SOMO2,SOMO2]
        xcish[col,1] = xcish[1,col]
    #15 <ZW-|H|SL2>
    block_index = nvirt + 2 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        xcish[1,col] = rep_tens[v_orb,SOMO1,SOMO2,SOMO2] + 0.5 * rep_tens[v_orb,SOMO2,SOMO2,SOMO1] - 0.5 * rep_tens[v_orb,SOMO1,SOMO1,SOMO1]
        xcish[col,1] = xcish[1,col]
    #16 <ZW-|H|HL1>
    block_index = 2 * nvirt + 2 * ndocc + 3
    for col in range(block_index, block_index + (ndocc * nvirt)):
        o_orb = (col - block_index) // nvirt
        v_orb = (col - block_index) % nvirt + (SOMO2 + 1)
        xcish[1,col] = (1 / np.sqrt(2)) * (rep_tens[o_orb, SOMO1, SOMO2, v_orb] - rep_tens[o_orb, SOMO2, SOMO1, v_orb])
        xcish[col,1] = xcish[1,col]
    #17 <ZW-|H|HL2>
    block_index = (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for col in range(block_index, block_index + (ndocc * nvirt)):
        o_orb = (col - block_index) // nvirt
        v_orb = (col - block_index) % nvirt + (SOMO2 + 1)
        xcish[1,col] = np.sqrt(1.5) * (rep_tens[o_orb, SOMO1, SOMO2, v_orb] + rep_tens[o_orb, SOMO2, SOMO1, v_orb])
        xcish[col,1] = xcish[1,col]
        
        
    #18 <ZW+|H|ZW+>
    xcish[2,2] = energy0 + 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) + 1.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1] - rep_tens[SOMO1, SOMO1, SOMO2, SOMO2]
    #19 <ZW+|H|HS1>
    block_index = 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        xcish[2,col] = rep_tens[o_orb,SOMO2,SOMO1,SOMO1] - 1.5 * rep_tens[o_orb,SOMO1,SOMO1,SOMO2] - 0.5 * rep_tens[o_orb,SOMO2,SOMO2,SOMO2]
        xcish[col,2] = xcish[2,col]
    #20 <ZW+|H|HS2>
    block_index = ndocc + 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        xcish[2,col] = 1.5 * rep_tens[o_orb,SOMO2,SOMO2,SOMO1] + 0.5 * rep_tens[o_orb,SOMO1,SOMO1,SOMO1] - rep_tens[o_orb, SOMO1, SOMO2, SOMO2]
        xcish[col,2] = xcish[2,col]
    #21 <ZW0+|H|SL1>
    block_index = 2 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        xcish[2,col] = 1.5 * rep_tens[v_orb,SOMO1,SOMO1,SOMO2] + 0.5 * rep_tens[v_orb,SOMO2,SOMO2,SOMO2] - rep_tens[v_orb,SOMO2,SOMO1,SOMO1]
        xcish[col,2] = xcish[2,col]
    #22 <ZW0+|H|SL2>
    block_index = nvirt + 2 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        xcish[2,col] = rep_tens[v_orb,SOMO1,SOMO2,SOMO2] - 1.5 * rep_tens[v_orb,SOMO2,SOMO2,SOMO1] - 0.5 * rep_tens[v_orb,SOMO1,SOMO1,SOMO1]
        xcish[col,2] = xcish[2,col]
    #23 <ZW+|H|HL1>
    block_index = 2 * nvirt + 2 * ndocc + 3
    for col in range(block_index, block_index + (ndocc * nvirt)):
        o_orb = (col - block_index) // nvirt
        v_orb = (col - block_index) % nvirt + (SOMO2 + 1)
        xcish[2,col] = (1/np.sqrt(2)) * (4 * rep_tens[o_orb, v_orb, SOMO1, SOMO2] - rep_tens[o_orb, SOMO1, SOMO2, v_orb] - rep_tens[o_orb, SOMO2, SOMO1, v_orb])
        xcish[col,2] = xcish[2,col]
    #24 <ZW+|H|HL2>
    block_index = (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for col in range(block_index, block_index + (ndocc * nvirt)):
        o_orb = (col - block_index) // nvirt
        v_orb = (col - block_index) % nvirt + (SOMO2 + 1)
        xcish[2,col] = np.sqrt(1.5) * (rep_tens[o_orb, SOMO2, SOMO1, v_orb] - rep_tens[o_orb, SOMO1, SOMO2, v_orb])
        xcish[col,2] = xcish[2,col]
    
    
    row_block_index = 3
    #25 <HS1|H|HS1>
    col_block_index = 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(row, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                xcish[row, col] = energy0 + orb_energies[SOMO1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, SOMO1, SOMO1] + 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] \
                             - 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] + 1.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb1] + 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb1]
            else:    
                xcish[row, col] = 0.5 * rep_tens[o_orb2, SOMO1, SOMO1, o_orb1] + 1.5 * rep_tens[o_orb2, SOMO2, SOMO2, o_orb1] - rep_tens[o_orb2, o_orb1, SOMO1, SOMO1]
            xcish[col, row] = xcish[row,col]
    #26 <HS1|H|HS2>
    col_block_index = ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                xcish[row, col] = 0.5 * rep_tens[SOMO1, SOMO2, SOMO1, SOMO1] + 0.5 * rep_tens[SOMO1, SOMO2, SOMO2, SOMO2] - rep_tens[SOMO1, o_orb1, o_orb1, SOMO2] - rep_tens[o_orb1, o_orb1, SOMO1, SOMO2]
            else:    
                xcish[row, col] = - rep_tens[o_orb1, o_orb2, SOMO1, SOMO2] - rep_tens[o_orb2, SOMO1, SOMO2, o_orb1]
            xcish[col, row] = xcish[row, col]
    #27 <HS1|H|SL1>
    col_block_index = 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb = col - col_block_index + (SOMO2 + 1)
            xcish[row, col] = - rep_tens[o_orb, SOMO1, SOMO1, v_orb] 
            xcish[col, row] = xcish[row,col]
    #28 <HS1|H|SL2>
    col_block_index = nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb = col - col_block_index + (SOMO2 + 1)
            xcish[row, col] = rep_tens[o_orb, SOMO1, SOMO2, v_orb] - 2 * rep_tens[o_orb, SOMO2, SOMO1, v_orb]
            xcish[col, row] = xcish[row,col]
    #29 <HS1|H|HL1>
    col_block_index = 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (rep_tens[o_orb1, o_orb1, SOMO1, v_orb] + 1.5 * rep_tens[v_orb, SOMO2, SOMO2, SOMO1] - 0.5 * rep_tens[v_orb, SOMO1, SOMO1, SOMO1] \
                                   - 2 * rep_tens[v_orb, o_orb1, o_orb1, SOMO1])
            else:
                xcish[row, col] = (1 / np.sqrt(2)) * (rep_tens[v_orb, SOMO1, o_orb1, o_orb2] - 2 * rep_tens[v_orb, o_orb2, o_orb1, SOMO1])
            xcish[col,row] = xcish[row,col]
    #30 <HS1|H|HL2>
    col_block_index = (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                xcish[row, col] = np.sqrt(1.5) * (0.5 * rep_tens[v_orb, SOMO1, SOMO1, SOMO1] + 0.5 * rep_tens[v_orb, SOMO2, SOMO2, SOMO1] - rep_tens[v_orb, SOMO1, o_orb1, o_orb1])
            else:
                xcish[row, col] = - np.sqrt(1.5) * (rep_tens[v_orb, SOMO1, o_orb1, o_orb2])
            xcish[col,row] = xcish[row,col]

    
    row_block_index = ndocc + 3
    #31 <HS2|H|HS2>
    col_block_index = ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(row, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                xcish[row, col] = energy0 + orb_energies[SOMO2] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, SOMO2, SOMO2] + 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] \
                             - 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] + 1.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb1] + 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb1]
            else:    
                xcish[row, col] = 1.5 * rep_tens[o_orb2, SOMO1, SOMO1, o_orb1] + 0.5 * rep_tens[o_orb2, SOMO2, SOMO2, o_orb1] - rep_tens[o_orb2, o_orb1, SOMO2, SOMO2]
            xcish[col, row] = xcish[row,col]
    #32 <HS2|H|SL1>
    col_block_index = 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb = col - col_block_index + (SOMO2 + 1)
            xcish[row, col] = rep_tens[o_orb, SOMO2, SOMO1, v_orb] - 2 * rep_tens[o_orb, SOMO1, SOMO2, v_orb]
            xcish[col, row] = xcish[row, col]
    #33 <HS2|H|SL2>
    col_block_index = nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb = col - col_block_index + (SOMO2 + 1)
            xcish[row, col] = - rep_tens[o_orb, SOMO2, SOMO2, v_orb]
            xcish[col, row] = xcish[row,col]
    #34 <HS2|H|HL1>
    col_block_index = 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (2*rep_tens[v_orb,o_orb1,o_orb1,SOMO2] - rep_tens[o_orb1,o_orb1,SOMO2,v_orb] - 1.5*rep_tens[v_orb, SOMO1, SOMO1, SOMO2] + 0.5*rep_tens[v_orb,SOMO2,SOMO2,SOMO2])
            else:
                xcish[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[v_orb, o_orb2, o_orb1, SOMO2] - rep_tens[v_orb, SOMO2, o_orb1, o_orb2])
            xcish[col,row] = xcish[row,col]
    #35 <HS2|H|HL2>
    col_block_index = (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                xcish[row, col] = np.sqrt(1.5) * (0.5 * rep_tens[v_orb, SOMO2, SOMO2, SOMO2] + 0.5 * rep_tens[v_orb, SOMO1, SOMO1, SOMO2] - rep_tens[v_orb, SOMO2, o_orb1, o_orb1])
            else:
                xcish[row, col] = - np.sqrt(1.5) * (rep_tens[v_orb, SOMO2, o_orb1, o_orb2])
            xcish[col,row] = xcish[row,col]

    
    row_block_index = 2 * ndocc + 3
    #36 <SL1|H|SL1>
    col_block_index = 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(row, col_block_index + nvirt):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[SOMO1] - rep_tens[v_orb1, v_orb1, SOMO1, SOMO1] + 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] \
                             - 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] + 1.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb1] + 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb1]
            else:    
                xcish[row, col] = 1.5 * rep_tens[v_orb2, SOMO2, SOMO2, v_orb1] + 0.5 * rep_tens[v_orb2, SOMO1, SOMO1, v_orb1] -  rep_tens[v_orb2, v_orb1, SOMO2, SOMO2]
            xcish[col, row] = xcish[row,col]
    #37 <SL1|H|SL2>
    col_block_index = nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = 0.5 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO2] + 0.5 * rep_tens[SOMO1, SOMO2, SOMO2, SOMO2] - rep_tens[v_orb1, v_orb1, SOMO1, SOMO2] - rep_tens[v_orb1, SOMO1, SOMO2, v_orb1]
            else:    
                xcish[row, col] = - rep_tens[v_orb1, v_orb2, SOMO1, SOMO2] - rep_tens[v_orb1, SOMO2, SOMO1, v_orb2]
            xcish[col, row] = xcish[row,col]
    #38 <SL1|H|HL1>
    col_block_index = 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[o_orb, v_orb1, v_orb1, SOMO1] + 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO1] - rep_tens[o_orb, SOMO1, v_orb1, v_orb1] - 1.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO1])
            else:
                xcish[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[v_orb1, SOMO1, v_orb2, o_orb] - rep_tens[o_orb, SOMO1, v_orb1, v_orb2])
            xcish[col,row] = xcish[row,col]
    #39 <SL1|H|HL2>
    col_block_index = (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = np.sqrt(1.5) * (0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO1] + 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO1] - rep_tens[o_orb, SOMO1, v_orb1, v_orb1])
            else:
                xcish[row, col] = - np.sqrt(1.5) * (rep_tens[o_orb, SOMO1, v_orb1, v_orb2])
            xcish[col,row] = xcish[row,col]
    
    
    row_block_index = nvirt + 2 * ndocc + 3
    #40 <SL2|H|SL2>
    col_block_index = nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(row, col_block_index + nvirt):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[SOMO2] - rep_tens[v_orb1, v_orb1, SOMO2, SOMO2] + 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] \
                             - 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] + 1.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb1] + 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb1]
            else:    
                xcish[row, col] = 1.5 * rep_tens[v_orb2, SOMO2, SOMO2, v_orb1] + 0.5 * rep_tens[v_orb2, SOMO1, SOMO1, v_orb1] -  rep_tens[v_orb2, v_orb1, SOMO2, SOMO2]
            xcish[col, row] = xcish[row,col]
    #41 <SL2|H|HL1>
    col_block_index = 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (rep_tens[o_orb,SOMO2,v_orb1,v_orb1] + 1.5*rep_tens[o_orb,SOMO1,SOMO1,SOMO2] - 2*rep_tens[o_orb,v_orb1,v_orb1,SOMO2] - 0.5*rep_tens[o_orb,SOMO2,SOMO2,SOMO2])
            else:
                xcish[row, col] = (1 / np.sqrt(2)) * (rep_tens[o_orb, SOMO2, v_orb1, v_orb2] - 2 * rep_tens[v_orb1, SOMO2, v_orb2, o_orb])
            xcish[col,row] = xcish[row,col]
    #42 <SL2|H|HL2>
    col_block_index = (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = np.sqrt(1.5) * (0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO2] + 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO2] - rep_tens[o_orb, SOMO2, v_orb1, v_orb1])
            else:
                xcish[row, col] = - np.sqrt(1.5) * (rep_tens[o_orb, SOMO2, v_orb1, v_orb2])
            xcish[col,row] = xcish[row,col]
    
    
    row_block_index = 2 * nvirt + 2 * ndocc + 3
    #43 <HL1|H|HL1>
    col_block_index = 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + (ndocc * nvirt)):
        o_orb1 = (row - row_block_index) // nvirt
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                xcish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb1] - 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) + 1.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1] \
                                 + 2 * rep_tens[o_orb1, v_orb1, v_orb1, o_orb1]
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                xcish[row, col] =  2 * rep_tens[o_orb1, v_orb1, v_orb1, o_orb2] - rep_tens[o_orb1, o_orb2, v_orb1, v_orb1]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                xcish[row, col] = rep_tens[v_orb1, o_orb1, o_orb1, v_orb2] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb2]
            else:
                xcish[row, col] = 2 * rep_tens[o_orb1, v_orb1, o_orb2, v_orb2] - rep_tens[o_orb1, o_orb2, v_orb1, v_orb2]
            xcish[col, row] = xcish[row,col]
    #44 <HL1|H|HL2>
    col_block_index = (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + (ndocc * nvirt)):
        o_orb1 = (row - row_block_index) // nvirt
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                xcish[row, col] = (np.sqrt(3) / 2) * (rep_tens[o_orb1,SOMO1,SOMO1,o_orb1] - rep_tens[o_orb1,SOMO2,SOMO2,o_orb1] + rep_tens[v_orb1,SOMO2,SOMO2,v_orb1] - rep_tens[v_orb1,SOMO1,SOMO1,v_orb1])
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                xcish[row, col] =  (np.sqrt(3) / 2) * (rep_tens[o_orb1, SOMO1, SOMO1, o_orb2] - rep_tens[o_orb1, SOMO2, SOMO2, o_orb2])
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                xcish[row, col] =  (np.sqrt(3) / 2) * (rep_tens[v_orb1, SOMO2, SOMO2, v_orb2] - rep_tens[v_orb1, SOMO1, SOMO1, v_orb2])
            xcish[col, row] = xcish[row,col]
    
    row_block_index = (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    #45 <HL2|H|HL2>
    col_block_index = (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + (ndocc * nvirt)):
        o_orb1 = (row - row_block_index) // nvirt
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                xcish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb1] - 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) - 0.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1] \
                                 + rep_tens[o_orb1, SOMO1, SOMO1, o_orb1] + rep_tens[o_orb1, SOMO2, SOMO2, o_orb1] + rep_tens[v_orb1, SOMO1, SOMO1, v_orb1] + rep_tens[v_orb1, SOMO2, SOMO2, v_orb1]
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                xcish[row, col] = rep_tens[o_orb1, SOMO1, SOMO1, o_orb2] + rep_tens[o_orb1, SOMO2, SOMO2, o_orb2] - rep_tens[o_orb1, o_orb2, v_orb1, v_orb1]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                xcish[row, col] = rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] + rep_tens[v_orb1, SOMO2, SOMO2, v_orb2] - rep_tens[v_orb1, v_orb2, o_orb1, o_orb1]
            else:
                xcish[row, col] = - rep_tens[o_orb1, o_orb2, v_orb1, v_orb2]
            xcish[col, row] = xcish[row,col]
    
    
    ################# TRIPLET BLOCK ######################
    
    row_index = 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    #46 <OS3|H|OS3>
    xcish[row_index, row_index] = energy0 - 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) - (0.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1])
    #47 <OS3|H|HS1>
    col_index = 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 4
    for col in range(col_index, col_index + ndocc):
        o_orb = col - col_index
        xcish[row_index, col] = 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO1] + 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO1]
        xcish[col, row_index] = xcish[row_index,col]
    #48 <OS3|H|HS2>
    col_index = 2 * (ndocc * nvirt) + 2 * nvirt + 3 * ndocc + 4
    for col in range(col_index, col_index + ndocc):
        o_orb = col - col_index
        xcish[row_index, col] = 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO2] + 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO2]
        xcish[col, row_index] = xcish[row_index,col]
    #49 <OS3|H|SL1>
    col_index = 2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 4
    for col in range(col_index, col_index + nvirt):
        v_orb = col - col_index + (SOMO2 + 1)
        xcish[row_index, col] = - 0.5 * rep_tens[v_orb, SOMO1, SOMO1, SOMO1] - 0.5 * rep_tens[v_orb, SOMO2, SOMO2, SOMO1]
        xcish[col, row_index] = xcish[row_index,col]
    #50 <OS3|H|SL2>
    col_index = 2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 4
    for col in range(col_index, col_index + nvirt):
        v_orb = col - col_index + (SOMO2 + 1)
        xcish[row_index, col] = 0.5 * rep_tens[v_orb, SOMO1, SOMO1, SOMO2] + 0.5 * rep_tens[v_orb, SOMO2, SOMO2, SOMO2]
        xcish[col, row_index] = xcish[row_index, col]
    #51 <OS3|H|HL1> = 0
    #52 <OS3|H|HL2>
    col_index =  3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for col in range(col_index, col_index + (ndocc * nvirt)):
        o_orb = (col - col_index) // nvirt # Increase o_orb after every ndocc cols
        v_orb = (col - col_index) % nvirt + (SOMO2 + 1) # Increase v_orb then reset after ndocc cols
        xcish[row_index,col] = (1 / np.sqrt(2)) * (rep_tens[o_orb, SOMO1, SOMO1, v_orb] - rep_tens[o_orb, SOMO2, SOMO2, v_orb])
        xcish[col,row_index] = xcish[row_index,col]
    #53 <OS3|H|HL3>
    col_index =  4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for col in range(col_index, col_index + (ndocc * nvirt)):
        o_orb = (col - col_index) // nvirt # Increase o_orb after every ndocc cols
        v_orb = (col - col_index) % nvirt + (SOMO2 + 1) # Increase v_orb then reset after ndocc cols
        xcish[row_index,col] = rep_tens[o_orb, SOMO1, SOMO1, v_orb] + rep_tens[o_orb, SOMO2, SOMO2, v_orb]
        xcish[col,row_index] = xcish[row_index,col]
    
    row_block_index = 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 4
    #54 <HS1|H|HS1>
    col_block_index = 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(row, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                xcish[row, col] = energy0 + orb_energies[SOMO1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, SOMO1, SOMO1] + 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] \
                             - 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] - 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb1] + 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb1]
            else:    
                xcish[row, col] = 0.5 * rep_tens[o_orb2, SOMO1, SOMO1, o_orb1] - 0.5 * rep_tens[o_orb2, SOMO2, SOMO2, o_orb1] - rep_tens[o_orb2, o_orb1, SOMO1, SOMO1]
            xcish[col, row] = xcish[row,col]
    #55 <HS1|H|HS2>
    col_block_index = 2 * (ndocc * nvirt) + 2 * nvirt + 3 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                xcish[row, col] = rep_tens[SOMO1, o_orb1, o_orb1, SOMO2] - rep_tens[o_orb1, o_orb1, SOMO1, SOMO2] + 0.5 * rep_tens[SOMO2, SOMO1, SOMO1, SOMO1] \
                             + 0.5 * rep_tens[SOMO1, SOMO2, SOMO2, SOMO2]                                                               #CHECK RESULT, SAME AS SINGLET?
            else:    
                xcish[row, col] = rep_tens[o_orb2, SOMO1, SOMO2, o_orb1] - rep_tens[o_orb2, o_orb1, SOMO1, SOMO2]
            xcish[col, row] = xcish[row,col]
    #56 <HS1|H|SL1>
    col_block_index = 2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb = col - col_block_index + (SOMO2 + 1)
            xcish[row, col] = rep_tens[o_orb, SOMO1, SOMO1, v_orb]
            xcish[col, row] = xcish[row,col]
    #57 <HS1|H|SL2>
    col_block_index = 2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb = col - col_block_index + (SOMO2 + 1)
            xcish[row, col] = - rep_tens[o_orb, SOMO1, SOMO2, v_orb]
            xcish[col, row] = xcish[row,col]
    #58 <HS1|H|HL1>
    col_block_index = 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[v_orb, o_orb1, o_orb1, SOMO1] + 0.5 * rep_tens[v_orb, SOMO2, SOMO2, SOMO1] + 0.5 * rep_tens[v_orb, SOMO1, SOMO1, SOMO1] \
                                   - rep_tens[o_orb1, o_orb1, SOMO1, v_orb])
            else:
                xcish[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[v_orb, o_orb2, o_orb1, SOMO1] - rep_tens[v_orb, SOMO1, o_orb1, o_orb2])
            xcish[col,row] = xcish[row,col]
    #59 <HS1|H|HL2>
    col_block_index = 3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (0.5 * rep_tens[v_orb, SOMO1, SOMO1, SOMO1] - 1.5 * rep_tens[v_orb, SOMO2, SOMO2, SOMO1] - rep_tens[v_orb, SOMO1, o_orb1, o_orb1])
            else:
                xcish[row, col] = - (1 / np.sqrt(2)) * (rep_tens[v_orb, SOMO1, o_orb1, o_orb2])
            xcish[col,row] = xcish[row,col]
    #60 <HS1|H|HL3>
    col_block_index = 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                xcish[row, col] = 0.5 * rep_tens[v_orb, SOMO1, SOMO1, SOMO1] + 0.5 * rep_tens[v_orb, SOMO2, SOMO2, SOMO1] - rep_tens[v_orb, SOMO1, o_orb1, o_orb1]
            else:
                xcish[row, col] = - rep_tens[v_orb, SOMO1, o_orb1, o_orb2]
            xcish[col,row] = xcish[row,col]

    
    row_block_index = 2 * (ndocc * nvirt) + 2 * nvirt + 3 * ndocc + 4
    #61 <HS2|H|HS2>
    col_block_index = 2 * (ndocc * nvirt) + 2 * nvirt + 3 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(row, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                xcish[row, col] = energy0 + orb_energies[SOMO2] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, SOMO2, SOMO2] + 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] \
                             - 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] - 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb1] + 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb1]
            else:
                xcish[row, col] = rep_tens[o_orb1, o_orb2, SOMO2, SOMO2] - 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb2] - 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb2]
            xcish[col, row] = xcish[row,col]
    #62 <HS2|H|SL1>
    col_block_index = 2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb = col - col_block_index + (SOMO2 + 1)
            xcish[row, col] = rep_tens[o_orb, SOMO2, SOMO1, v_orb] 
            xcish[col, row] = xcish[row,col]
    #63 <HS2|H|SL2>
    col_block_index = 2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb = col - col_block_index + (SOMO2 + 1)
            xcish[row, col] = - rep_tens[o_orb, SOMO2, SOMO2, v_orb]
            xcish[col, row] = xcish[row,col]
    #64 <HS2|H|HL1>
    col_block_index = 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[v_orb, o_orb1, o_orb1, SOMO2] + 0.5 * rep_tens[v_orb, SOMO1, SOMO1, SOMO2] + 0.5 * rep_tens[v_orb, SOMO2, SOMO2, SOMO2] \
                                   - rep_tens[o_orb1, o_orb1, SOMO2, v_orb])
            else:
                xcish[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[v_orb, o_orb2, o_orb1, SOMO2] - rep_tens[v_orb, SOMO2, o_orb1, o_orb2])
            xcish[col,row] = xcish[row,col]
    #65 <HS2|H|HL2>
    col_block_index = 3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (1.5 * rep_tens[v_orb, SOMO1, SOMO1, SOMO2] + rep_tens[v_orb, SOMO2, o_orb1, o_orb1] - 0.5 * rep_tens[v_orb, SOMO2, SOMO2, SOMO2])
            else:
                xcish[row, col] = (1 / np.sqrt(2)) * (rep_tens[v_orb, SOMO2, o_orb1, o_orb2])
            xcish[col,row] = xcish[row,col]
    #66 <HS2|H|HL3>
    col_block_index = 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                xcish[row, col] = 0.5 * rep_tens[v_orb, SOMO2, SOMO2, SOMO2] + 0.5 * rep_tens[v_orb, SOMO1, SOMO1, SOMO2] - rep_tens[v_orb, SOMO2, o_orb1, o_orb1]
            else:
                xcish[row, col] = - rep_tens[v_orb, SOMO2, o_orb1, o_orb2]
            xcish[col,row] = xcish[row,col]
    
            
    row_block_index = 2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 4
    #67 <SL1|H|SL1>
    col_block_index =  2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(row, col_block_index + nvirt):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[SOMO1] - rep_tens[v_orb1, v_orb1, SOMO1, SOMO1] + 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] \
                             - 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] - 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb1] + 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb1]
            else:    
                xcish[row, col] =  0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] - 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb2] - rep_tens[v_orb1, v_orb2, SOMO1, SOMO1]
            xcish[col, row] = xcish[row,col]
    #68 <SL1|H|SL2>
    col_block_index = 2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = rep_tens[v_orb1, v_orb1, SOMO1, SOMO2] - rep_tens[v_orb1, SOMO1, SOMO2, v_orb1] - 0.5 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO2] - 0.5 * rep_tens[SOMO1, SOMO2, SOMO2, SOMO2]
            else:    
                xcish[row, col] = rep_tens[v_orb1, v_orb2, SOMO1, SOMO2] - rep_tens[v_orb1, SOMO2, SOMO1, v_orb2]
            xcish[col, row] = xcish[row,col]
    #69 <SL1|H|HL1>
    col_block_index = 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[o_orb, v_orb1, v_orb1, SOMO1] + 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO1] + 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO1] \
                                  - rep_tens[o_orb, SOMO1, v_orb1, v_orb1])
            else:
                xcish[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[v_orb1, SOMO1, v_orb2, o_orb] - rep_tens[o_orb, SOMO1, v_orb1, v_orb2])
            xcish[col,row] = xcish[row,col]
    #70 <SL1|H|HL2>
    col_block_index = 3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (rep_tens[o_orb, SOMO1, v_orb1, v_orb1] + 1.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO1] - 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO1])
            else:
                xcish[row, col] = - (1 / np.sqrt(2)) * (rep_tens[o_orb, SOMO1, v_orb1, v_orb2])
            xcish[col,row] = xcish[row,col]
    #71 <SL1|H|HL3>
    col_block_index = 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = rep_tens[o_orb, SOMO1, v_orb1, v_orb1] - 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO1] - 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO1]
            else:
                xcish[row, col] = rep_tens[o_orb, SOMO1, v_orb1, v_orb2]
            xcish[col,row] = xcish[row,col]
    
    
    row_block_index = 2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 4
    #72 <SL2|H|SL2>
    col_block_index = 2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(row, col_block_index + nvirt):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[SOMO2] - rep_tens[v_orb1, v_orb1, SOMO2, SOMO2] + 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] \
                             - 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] - 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb1] + 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb1]
            else:    
                xcish[row, col] =  0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] - rep_tens[v_orb1, v_orb2, SOMO2, SOMO2] - 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb2]
            xcish[col, row] = xcish[row,col]
    #73 <SL2|H|HL1>
    col_block_index = 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (rep_tens[o_orb, SOMO2, v_orb1, v_orb1] - 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO2] - 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO2] \
                                  - 2 * rep_tens[o_orb, v_orb1, v_orb1, SOMO2])
            else:
                xcish[row, col] = (1 / np.sqrt(2)) * (rep_tens[o_orb, SOMO2, v_orb1, v_orb2] - 2 * rep_tens[v_orb1, SOMO2, v_orb2, o_orb])
            xcish[col,row] = xcish[row,col]
    #74 <SL2|H|HL2>
    col_block_index = 3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (rep_tens[o_orb, SOMO2, v_orb1, v_orb1] + 1.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO2] - 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO2])
            else:
                xcish[row, col] = - (1 / np.sqrt(2)) * (rep_tens[o_orb, SOMO2, v_orb1, v_orb2])
            xcish[col,row] = xcish[row,col]
    #75 <SL2|H|HL3>
    col_block_index = 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO2] + 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO2] - rep_tens[o_orb, SOMO2, v_orb1, v_orb1]
            else:
                xcish[row, col] = - rep_tens[o_orb, SOMO2, v_orb1, v_orb2]
            xcish[col,row] = xcish[row,col]  
    
    
    row_block_index = 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    #76 <HL1|H|HL1>
    col_block_index = 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + (ndocc * nvirt)):
        o_orb1 = (row - row_block_index) // nvirt
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                xcish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb1] - 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) - 0.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1] \
                                 + 2 * rep_tens[o_orb1, v_orb1, v_orb1, o_orb1]
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                xcish[row, col] =  2 * rep_tens[o_orb1, v_orb1, v_orb1, o_orb2] - rep_tens[o_orb1, o_orb2, v_orb1, v_orb1]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                xcish[row, col] = rep_tens[v_orb1, o_orb1, o_orb1, v_orb2] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb2]
            else:
                xcish[row, col] = 2 * rep_tens[o_orb1, v_orb1, o_orb2, v_orb2] - rep_tens[o_orb1, o_orb2, v_orb1, v_orb2]
            xcish[col, row] = xcish[row,col]
    #77 <HL1|H|HL2>
    col_block_index = 3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + (ndocc * nvirt)):
        o_orb1 = (row - row_block_index) // nvirt
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                xcish[row, col] = 0.5 * (rep_tens[o_orb1,SOMO2,SOMO2,o_orb1] - rep_tens[o_orb1,SOMO1,SOMO1,o_orb1] + rep_tens[v_orb1,SOMO1,SOMO1,v_orb1] - rep_tens[v_orb1,SOMO2,SOMO2,v_orb1])
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                xcish[row, col] =  0.5 * (rep_tens[o_orb1, SOMO2, SOMO2, o_orb2] - rep_tens[o_orb1, SOMO1, SOMO1, o_orb2])
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                xcish[row, col] =  0.5 * (rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] - rep_tens[v_orb1, SOMO2, SOMO2, v_orb2])
            xcish[col, row] = xcish[row,col]
    #78 <HL1|H|HL3>
    col_block_index = 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + (ndocc * nvirt)):
        o_orb1 = (row - row_block_index) // nvirt
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (rep_tens[v_orb1,SOMO1,SOMO1,v_orb1] + rep_tens[v_orb1,SOMO2,SOMO2,v_orb1] - rep_tens[o_orb1,SOMO1,SOMO1,o_orb1] - rep_tens[o_orb1,SOMO2,SOMO2,o_orb1])
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                xcish[row, col] =  - (1 / np.sqrt(2)) * (rep_tens[o_orb1, SOMO1, SOMO1, o_orb2] + rep_tens[o_orb1, SOMO2, SOMO2, o_orb2])
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                xcish[row, col] =  (1 / np.sqrt(2)) * (rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] + rep_tens[v_orb1, SOMO2, SOMO2, v_orb2])
            xcish[col, row] = xcish[row,col]
    
    
    row_block_index = 3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    #79 <HL2|H|HL2>
    col_block_index = 3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + (ndocc * nvirt)):
        o_orb1 = (row - row_block_index) // nvirt
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                xcish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb1] - 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) + 1.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1]
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                xcish[row, col] =  - rep_tens[o_orb1, o_orb2, v_orb1, v_orb1]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                xcish[row, col] = - rep_tens[v_orb1, o_orb1, o_orb1, v_orb2] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb2]
            else:
                xcish[row, col] = - rep_tens[o_orb1, o_orb2, v_orb1, v_orb2]
            xcish[col, row] = xcish[row,col]
    #80 <HL2|H|HL3>
    col_block_index = 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + (ndocc * nvirt)):
        o_orb1 = (row - row_block_index) // nvirt
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (rep_tens[o_orb1,SOMO1,SOMO1,o_orb1] - rep_tens[o_orb1,SOMO2,SOMO2,o_orb1] + rep_tens[v_orb1,SOMO1,SOMO1,v_orb1] - rep_tens[v_orb1,SOMO2,SOMO2,v_orb1])
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                xcish[row, col] =  (1 / np.sqrt(2)) * (rep_tens[o_orb1, SOMO1, SOMO1, o_orb2] - rep_tens[o_orb1, SOMO2, SOMO2, o_orb2])
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                xcish[row, col] =  (1 / np.sqrt(2)) * (rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] - rep_tens[v_orb1, SOMO2, SOMO2, v_orb2])
            xcish[col, row] = xcish[row,col]
    
    row_block_index = 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    #81 <HL3|H|HL3>
    col_block_index = 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + (ndocc * nvirt)):
        o_orb1 = (row - row_block_index) // nvirt
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                xcish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb1] - 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) - 0.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1] \
                                  + 0.5 * (rep_tens[o_orb1, SOMO1, SOMO1, o_orb1] + rep_tens[o_orb1, SOMO2, SOMO2, o_orb1] + rep_tens[v_orb1, SOMO1, SOMO1, v_orb1] + rep_tens[v_orb1, SOMO2, SOMO2, v_orb1])
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                xcish[row, col] =  0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb2] + 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb2] - rep_tens[o_orb1, o_orb2, v_orb1, v_orb1]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                xcish[row, col] = 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] + 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb2] - rep_tens[v_orb1, v_orb2, o_orb1, o_orb1]
            else:
                xcish[row, col] = - rep_tens[o_orb1, o_orb2, v_orb1, v_orb2]
            xcish[col, row] = xcish[row,col]
    
            
    ################# QUINTET STATE ##################
    
    row_block_index = 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    #82 <5Q|H|5Q>
    col_block_index = 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + (ndocc * nvirt)):
        o_orb1 = (row - row_block_index) // nvirt # Increase o_orb after every ndocc rows
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1) # Increase v_orb for every column and reset after ndocc rows
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                xcish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb1] - 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) - 0.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1] \
                                 - 0.5 * (rep_tens[o_orb1, SOMO1, SOMO1, o_orb1] + rep_tens[o_orb1, SOMO2, SOMO2, o_orb1] + rep_tens[v_orb1, SOMO1, SOMO1, v_orb1] + rep_tens[v_orb1, SOMO2, SOMO2, v_orb1])
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                xcish[row, col] =  - rep_tens[o_orb1, o_orb2, v_orb1, v_orb1] - 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb2] - 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb2]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                xcish[row, col] = - rep_tens[v_orb1, v_orb2, o_orb1, o_orb1] - 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] - 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb2]
            else:
                xcish[row, col] = - rep_tens[o_orb1, o_orb2, v_orb1, v_orb2]
            xcish[col, row] = xcish[row,col]
    
    
    return xcish



def hetero_xcisd_ham_rot(ndocc, norbs, energy0, orb_energies, rep_tens):
    '''
    Form the CIS Hamiltonian matrix in the rotated CSF basis. Matrix elements on off-diagonals are typically 2e integrals, found in the working doc.
    Note that the basis is given in the working doc, i.e the ordering of CSFs. We have singlets then triplets, making the Hamiltonian block diagonal.
    This function adds in double excitations from the virtual orbitals to the SOMOs.
    
    Args:
        ndocc (int): Number of doubly occupied orbitals.
        energy0 (float): Ground state HF energy.
        orb_energies (array): HF orbital energies.
        rep_tens (array): 4D tensor giving two-electron repulsion integrals in the MO basis.
    Returns:
    '''
    SOMO1 = ndocc # Index of SOMO1
    SOMO2 = ndocc + 1 # Index of SOMO2
    nvirt = norbs - ndocc - 2 # Number of virtual orbitals
    ndoc3 = int((ndocc ** 2 - ndocc) / 2) # Number of doubly excited occupied to core triplet CSFs
    ndoc1 = int((ndocc ** 2 + ndocc) / 2) # Number of doubly excited occupied to core singlet CSFs
    ndcv3 = int((nvirt ** 2 - nvirt) / 2) # Number of doubly excited occupied to core triplet CSFs
    ndcv1 = int((nvirt ** 2 + nvirt) / 2) # Number of doubly excited occupied to core singlet CSFs
    nstates = nvirt ** 2 + ndocc ** 2 + 6 * (ndocc * nvirt) + 4 * ndocc + 4 * nvirt + 4  # nvirt ** 2 doubles (SOMO to LUMO), ndocc ** 2 doubles (HOMO to SOMO), 6 * ndocc * nvirt doubles (HOMO to LUMO), 4 * ndocc singles (HOMO to SOMO), 4 * nvirt singles (SOMO to LUMO)
                                                                                         # and 4 reference configurations (OS GSs and Zwitterions)
    
    xcish = np.zeros((nstates,nstates))
    
    ################# SINGLET BLOCK ######################
    #1 <OS1|H|OS1>
    xcish[0,0] = energy0 - 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) + (1.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1])
    #2 <OS1|H|ZW->
    xcish[0,1] = (rep_tens[SOMO1,SOMO2,SOMO1,SOMO1] - rep_tens[SOMO1,SOMO2,SOMO2,SOMO2])
    xcish[1,0] = xcish[0,1]
    #3 <OS1|H|ZW+>
    xcish[0,2] = 0
    xcish[2,0] = xcish[0,2]
    #4 <OS1|H|HS1> 
    block_index = 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        xcish[0,col] = 1.5 * rep_tens[o_orb,SOMO2,SOMO2,SOMO1] - 0.5 * rep_tens[o_orb,SOMO1,SOMO1,SOMO1]
        xcish[col,0] = xcish[0,col]
    #5 <OS1|H|HS2> 
    block_index = ndocc + 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        xcish[0,col] = 0.5 * rep_tens[o_orb,SOMO2,SOMO2,SOMO2] - 1.5 * rep_tens[o_orb,SOMO1,SOMO1,SOMO2]
        xcish[col,0] = xcish[0,col]
    #6 <OS1|H|SL1>
    block_index = 2 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        xcish[0,col] = 1.5 * rep_tens[v_orb,SOMO2,SOMO2,SOMO1] - 0.5 * rep_tens[v_orb,SOMO1,SOMO1,SOMO1]
        xcish[col,0] = xcish[0,col]
    #7 <OS1|H|SL2>
    block_index = nvirt + 2 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        xcish[0,col] = 0.5 * rep_tens[v_orb,SOMO2,SOMO2,SOMO2] - 1.5 * rep_tens[v_orb,SOMO1,SOMO1,SOMO2]
        xcish[col,0] = xcish[0,col]
    #8 <OS1|H|HL1> = 0
    #9 <OS1|H|HL2>
    block_index =  (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for col in range(block_index, block_index + (ndocc * nvirt)):
        o_orb = (col - block_index) // nvirt # Increase o_orb after every ndocc cols
        v_orb = (col - block_index) % nvirt + (SOMO2 + 1) # Increase v_orb then reset after ndocc cols
        xcish[0,col] = np.sqrt(1.5) * (rep_tens[o_orb, SOMO2, SOMO2, v_orb] - rep_tens[o_orb, SOMO1, SOMO1, v_orb])
        xcish[col,0] = xcish[0,col]
    #10 <OS1|H|1^HSD>
    block_index = 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    o_orb1 = 0
    o_orb2 = 0
    for col in range(block_index, block_index + ndoc1):
        if o_orb1 == o_orb2:
            xcish[0,col] = - np.sqrt(2) * rep_tens[o_orb1, SOMO1, SOMO2, o_orb1]
        else:
            xcish[0,col] = - rep_tens[o_orb1, SOMO1, SOMO2, o_orb2] - rep_tens[o_orb1, SOMO2, SOMO1, o_orb2]
        xcish[col,0] = xcish[0,col]
        o_orb2 += 1
        if o_orb2 == ndocc:
            o_orb1 += 1
            o_orb2 = o_orb1
    #11 <OS1|H|1^SLD> # ONLY INCLUDING EXCITATIONS TO THE SAME VIRTUAL ORBITAL FOR NOW
    block_index = ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    v_orb1 = SOMO2 + 1
    v_orb2 = SOMO2 + 1
    for col in range(block_index, block_index + ndcv1):
        if v_orb1 == v_orb2:
            xcish[0,col] = np.sqrt(2) * rep_tens[v_orb1, SOMO1, SOMO2, v_orb1]
        else:
            xcish[0,col] = rep_tens[v_orb1, SOMO1, SOMO2, v_orb2] + rep_tens[v_orb1, SOMO2, SOMO1, v_orb2]
        xcish[col,0] = xcish[0,col]
        v_orb2 += 1
        if v_orb2 == norbs:
            v_orb1 += 1
            v_orb2 = v_orb1


    
    #10 <ZW-|H|ZW->
    xcish[1,1] = energy0 + 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) - 0.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1] - rep_tens[SOMO1, SOMO1, SOMO2, SOMO2]
    #11 <ZW-|H|ZW+>
    xcish[1,2] = orb_energies[SOMO1] - orb_energies[SOMO2]
    xcish[2,1] = xcish[1,2]
    #12 <ZW-|H|HS1>
    block_index = 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        xcish[1,col] = rep_tens[o_orb,SOMO2,SOMO1,SOMO1] + 0.5 * rep_tens[o_orb,SOMO1,SOMO1,SOMO2] - 0.5 * rep_tens[o_orb,SOMO2,SOMO2,SOMO2]
        xcish[col,1] = xcish[1,col]
    #13 <ZW-|H|HS2>
    block_index = ndocc + 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        xcish[1,col] = rep_tens[o_orb,SOMO1,SOMO2,SOMO2] + 0.5 * rep_tens[o_orb,SOMO2,SOMO2,SOMO1] - 0.5 * rep_tens[o_orb,SOMO1,SOMO1,SOMO1]
        xcish[col, 1] = xcish[1,col]
    #14 <ZW-|H|SL1>
    block_index = 2 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        xcish[1,col] = rep_tens[v_orb,SOMO2,SOMO1,SOMO1] + 0.5 * rep_tens[v_orb,SOMO1,SOMO1,SOMO2] - 0.5 * rep_tens[v_orb,SOMO2,SOMO2,SOMO2]
        xcish[col,1] = xcish[1,col]
    #15 <ZW-|H|SL2>
    block_index = nvirt + 2 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        xcish[1,col] = rep_tens[v_orb,SOMO1,SOMO2,SOMO2] + 0.5 * rep_tens[v_orb,SOMO2,SOMO2,SOMO1] - 0.5 * rep_tens[v_orb,SOMO1,SOMO1,SOMO1]
        xcish[col,1] = xcish[1,col]
    #16 <ZW-|H|HL1>
    block_index = 2 * nvirt + 2 * ndocc + 3
    for col in range(block_index, block_index + (ndocc * nvirt)):
        o_orb = (col - block_index) // nvirt
        v_orb = (col - block_index) % nvirt + (SOMO2 + 1)
        xcish[1,col] = (1 / np.sqrt(2)) * (rep_tens[o_orb, SOMO1, SOMO2, v_orb] - rep_tens[o_orb, SOMO2, SOMO1, v_orb])
        xcish[col,1] = xcish[1,col]
    #17 <ZW-|H|HL2>
    block_index = (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for col in range(block_index, block_index + (ndocc * nvirt)):
        o_orb = (col - block_index) // nvirt
        v_orb = (col - block_index) % nvirt + (SOMO2 + 1)
        xcish[1,col] = np.sqrt(1.5) * (rep_tens[o_orb, SOMO1, SOMO2, v_orb] + rep_tens[o_orb, SOMO2, SOMO1, v_orb])
        xcish[col,1] = xcish[1,col]
    #18 <ZW-|H|1^HSD>
    block_index = 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    o_orb1 = 0
    o_orb2 = 0
    for col in range(block_index, block_index + ndoc1):
        if o_orb1 == o_orb2:
            xcish[1,col] = (1 / np.sqrt(2)) * (rep_tens[o_orb1, SOMO2, SOMO2, o_orb1] - rep_tens[o_orb1, SOMO1, SOMO1, o_orb1])
        else:
            xcish[1,col] = rep_tens[o_orb1, SOMO2, SOMO2, o_orb2] - rep_tens[o_orb1, SOMO1, SOMO1, o_orb2]
        xcish[col,1] = xcish[1,col]
        o_orb2 += 1
        if o_orb2 == ndocc:
            o_orb1 += 1
            o_orb2 = o_orb1
    
    #11 <ZW-|H|1^SLD> # ONLY INCLUDING EXCITATIONS TO THE SAME VIRTUAL ORBITAL FOR NOW
    block_index = ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    v_orb1 = SOMO2 + 1
    v_orb2 = SOMO2 + 1
    for col in range(block_index, block_index + ndcv1):
        if v_orb1 == v_orb2:
            xcish[1,col] = (1 / np.sqrt(2)) * (rep_tens[v_orb1, SOMO1, SOMO1, v_orb1] - rep_tens[v_orb1, SOMO2, SOMO2, v_orb1])
        else:
            xcish[1,col] = rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] - rep_tens[v_orb1, SOMO2, SOMO2, v_orb2]
        xcish[col,1] = xcish[1,col]
        v_orb2 += 1
        if v_orb2 == norbs:
            v_orb1 += 1
            v_orb2 = v_orb1
        
    #18 <ZW0+|H|ZW0+>
    xcish[2,2] = energy0 + 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) + 1.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1] - rep_tens[SOMO1, SOMO1, SOMO2, SOMO2]
    #19 <ZW0+|H|HS1>
    block_index = 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        xcish[2,col] = rep_tens[o_orb,SOMO2,SOMO1,SOMO1] - 1.5 * rep_tens[o_orb,SOMO1,SOMO1,SOMO2] - 0.5 * rep_tens[o_orb,SOMO2,SOMO2,SOMO2]
        xcish[col,2] = xcish[2,col]
    #20 <ZW0+|H|HS2>
    block_index = ndocc + 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        xcish[2,col] = 1.5 * rep_tens[o_orb,SOMO2,SOMO2,SOMO1] + 0.5 * rep_tens[o_orb,SOMO1,SOMO1,SOMO1] - rep_tens[o_orb,SOMO1,SOMO2,SOMO2]
        xcish[col,2] = xcish[2,col]
    #21 <ZW0+|H|SL1>
    block_index = 2 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        xcish[2,col] = 1.5 * rep_tens[v_orb,SOMO1,SOMO1,SOMO2] + 0.5 * rep_tens[v_orb,SOMO2,SOMO2,SOMO2] - rep_tens[v_orb,SOMO2,SOMO1,SOMO1]
        xcish[col,2] = xcish[2,col]
    #22 <ZW0+|H|SL2>
    block_index = nvirt + 2 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        xcish[2,col] = rep_tens[v_orb,SOMO1,SOMO2,SOMO2] - 1.5 * rep_tens[v_orb,SOMO2,SOMO2,SOMO1] - 0.5 * rep_tens[v_orb,SOMO1,SOMO1,SOMO1]
        xcish[col,2] = xcish[2,col]
    #23 <ZW0+|H|HL1>
    block_index = 2 * nvirt + 2 * ndocc + 3
    for col in range(block_index, block_index + (ndocc * nvirt)):
        o_orb = (col - block_index) // nvirt
        v_orb = (col - block_index) % nvirt + (SOMO2 + 1)
        xcish[2,col] = (1 / np.sqrt(2)) * (4 * rep_tens[o_orb, v_orb, SOMO1, SOMO2] - rep_tens[o_orb, SOMO1, SOMO2, v_orb] - rep_tens[o_orb, SOMO2, SOMO1, v_orb])
        xcish[col,2] = xcish[2,col]
    #24 <ZW0+|H|HL2>
    block_index = (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for col in range(block_index, block_index + (ndocc * nvirt)):
        o_orb = (col - block_index) // nvirt
        v_orb = (col - block_index) % nvirt + (SOMO2 + 1)
        xcish[2,col] = np.sqrt(1.5) * (rep_tens[o_orb, SOMO2, SOMO1, v_orb] - rep_tens[o_orb, SOMO1, SOMO2, v_orb])
        xcish[col,2] = xcish[2,col]
    #25 <ZW0+|H|1^HSD>
    block_index = 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    o_orb1 = 0
    o_orb2 = 0
    for col in range(block_index, block_index + ndoc1):
        if o_orb1 == o_orb2:
            xcish[2,col] = (1 / np.sqrt(2)) * (rep_tens[o_orb1, SOMO1, SOMO1, o_orb1] + rep_tens[o_orb1, SOMO2, SOMO2, o_orb1])
        else:
            xcish[2,col] = rep_tens[o_orb1, SOMO1, SOMO1, o_orb2] + rep_tens[o_orb1, SOMO2, SOMO2, o_orb2]
        xcish[col,2] = xcish[2,col]
        o_orb2 += 1
        if o_orb2 == ndocc:
            o_orb1 += 1
            o_orb2 = o_orb1
    #11 <ZW0+|H|1^SLD> 
    block_index = ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    v_orb1 = SOMO2 + 1
    v_orb2 = SOMO2 + 1
    for col in range(block_index, block_index + ndcv1):
        if v_orb1 == v_orb2:
            xcish[2,col] = (1 / np.sqrt(2)) * (rep_tens[v_orb1, SOMO2, SOMO2, v_orb1] + rep_tens[v_orb1, SOMO1, SOMO1, v_orb1])
        else:
            xcish[2,col] = rep_tens[v_orb1, SOMO2, SOMO2, v_orb2] + rep_tens[v_orb1, SOMO1, SOMO1, v_orb2]
        xcish[col,2] = xcish[2,col]
        v_orb2 += 1
        if v_orb2 == norbs:
            v_orb1 += 1
            v_orb2 = v_orb1
    
    
    row_block_index = 3
    #25 <HS1|H|HS1>
    col_block_index = 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(row, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                xcish[row, col] = energy0 + orb_energies[SOMO1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, SOMO1, SOMO1] + 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] \
                             - 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] + 1.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb1] + 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb1]
            else:    
                xcish[row, col] = 0.5 * rep_tens[o_orb2, SOMO1, SOMO1, o_orb1] + 1.5 * rep_tens[o_orb2, SOMO2, SOMO2, o_orb1] - rep_tens[o_orb2, o_orb1, SOMO1, SOMO1]
            xcish[col, row] = xcish[row,col]
    #26 <HS1|H|HS2>
    col_block_index = ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                xcish[row, col] = 0.5 * rep_tens[SOMO1, SOMO2, SOMO1, SOMO1] + 0.5 * rep_tens[SOMO1, SOMO2, SOMO2, SOMO2] - rep_tens[SOMO1, o_orb1, o_orb1, SOMO2] - rep_tens[o_orb1, o_orb1, SOMO1, SOMO2]
            else:    
                xcish[row, col] = - rep_tens[o_orb1, o_orb2, SOMO1, SOMO2] - rep_tens[o_orb2, SOMO1, SOMO2, o_orb1]
            xcish[col, row] = xcish[row, col]
    #27 <HS1|H|SL1>
    col_block_index = 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb = col - col_block_index + (SOMO2 + 1)
            xcish[row, col] = - rep_tens[o_orb, SOMO1, SOMO1, v_orb] 
            xcish[col, row] = xcish[row,col]
    #28 <HS1|H|SL2>
    col_block_index = nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb = col - col_block_index + (SOMO2 + 1)
            xcish[row, col] = rep_tens[o_orb, SOMO1, SOMO2, v_orb] - 2 * rep_tens[o_orb, SOMO2, SOMO1, v_orb]
            xcish[col, row] = xcish[row,col]
    #29 <HS1|H|HL1>
    col_block_index = 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (rep_tens[o_orb1, o_orb1, SOMO1, v_orb] + 1.5 * rep_tens[v_orb, SOMO2, SOMO2, SOMO1] - 0.5 * rep_tens[v_orb, SOMO1, SOMO1, SOMO1] \
                                   - 2 * rep_tens[v_orb, o_orb1, o_orb1, SOMO1])
            else:
                xcish[row, col] = (1 / np.sqrt(2)) * (rep_tens[v_orb, SOMO1, o_orb1, o_orb2] - 2 * rep_tens[v_orb, o_orb2, o_orb1, SOMO1])
            xcish[col,row] = xcish[row,col]
    #30 <HS1|H|HL2>
    col_block_index = (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                xcish[row, col] = np.sqrt(1.5) * (0.5 * rep_tens[v_orb, SOMO1, SOMO1, SOMO1] + 0.5 * rep_tens[v_orb, SOMO2, SOMO2, SOMO1] - rep_tens[v_orb, SOMO1, o_orb1, o_orb1])
            else:
                xcish[row, col] = - np.sqrt(1.5) * (rep_tens[v_orb, SOMO1, o_orb1, o_orb2])
            xcish[col,row] = xcish[row,col]
    #31 <HS1|H|1^HSD>
    col_block_index = 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        o_orb2 = 0
        o_orb3 = 0
        for col in range(col_block_index, col_block_index + ndoc1):
            if o_orb1 == o_orb2 and o_orb1 == o_orb3:
                xcish[row, col] = np.sqrt(2) * (rep_tens[o_orb1, SOMO2, SOMO1, SOMO1] - rep_tens[o_orb1, SOMO2, o_orb1, o_orb1] + 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, SOMO2] - 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, SOMO2])
            elif o_orb1 != o_orb2 and o_orb2 == o_orb3:
                xcish[row, col] = - np.sqrt(2) * rep_tens[o_orb2, SOMO2, o_orb1, o_orb2]
            elif o_orb1 == o_orb2 and o_orb1 != o_orb3:
                xcish[row, col] = rep_tens[o_orb3, SOMO2, SOMO1, SOMO1] - rep_tens[o_orb3, SOMO2, o_orb1, o_orb1] - rep_tens[o_orb3, o_orb1, o_orb1, SOMO2] + 0.5 * rep_tens[o_orb3, SOMO2, SOMO2, SOMO2] - 0.5 * rep_tens[o_orb3, SOMO1, SOMO1, SOMO2]
            elif o_orb1 == o_orb3 and o_orb1 != o_orb2:
                xcish[row, col] = rep_tens[o_orb2, SOMO2, SOMO1, SOMO1] - rep_tens[o_orb2, SOMO2, o_orb1, o_orb1] - rep_tens[o_orb2, o_orb1, o_orb1, SOMO2] + 0.5 * rep_tens[o_orb2, SOMO2, SOMO2, SOMO2] - 0.5 * rep_tens[o_orb2, SOMO1, SOMO1, SOMO2]
            else:
                xcish[row, col] = - rep_tens[o_orb2, SOMO2, o_orb1, o_orb3] - rep_tens[o_orb3, SOMO2, o_orb1, o_orb2]
            xcish[col,row] = xcish[row,col]
            o_orb3 += 1
            if o_orb3 == ndocc:
                o_orb2 += 1
                o_orb3 = o_orb2
    #<HS1|H|1^SLD> = 0
    
    row_block_index = ndocc + 3
    #31 <HS2|H|HS2>
    col_block_index = ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(row, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                xcish[row, col] = energy0 + orb_energies[SOMO2] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, SOMO2, SOMO2] + 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] \
                             - 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] + 1.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb1] + 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb1]
            else:    
                xcish[row, col] = 1.5 * rep_tens[o_orb2, SOMO1, SOMO1, o_orb1] + 0.5 * rep_tens[o_orb2, SOMO2, SOMO2, o_orb1] - rep_tens[o_orb2, o_orb1, SOMO2, SOMO2]
            xcish[col, row] = xcish[row,col]
    #32 <HS2|H|SL1>
    col_block_index = 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb = col - col_block_index + (SOMO2 + 1)
            xcish[row, col] = rep_tens[o_orb, SOMO2, SOMO1, v_orb] - 2 * rep_tens[o_orb, SOMO1, SOMO2, v_orb]
            xcish[col, row] = xcish[row, col]
    #33 <HS2|H|SL2>
    col_block_index = nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb = col - col_block_index + (SOMO2 + 1)
            xcish[row, col] = - rep_tens[o_orb, SOMO2, SOMO2, v_orb]
            xcish[col, row] = xcish[row,col]
    #34 <HS2|H|HL1>
    col_block_index = 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (2*rep_tens[v_orb,o_orb1,o_orb1,SOMO2] - rep_tens[o_orb1,o_orb1,SOMO2,v_orb] - 1.5*rep_tens[v_orb, SOMO1, SOMO1, SOMO2] + 0.5*rep_tens[v_orb,SOMO2,SOMO2,SOMO2])
            else:
                xcish[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[v_orb, o_orb2, o_orb1, SOMO2] - rep_tens[v_orb, SOMO2, o_orb1, o_orb2])
            xcish[col,row] = xcish[row,col]
    #35 <HS2|H|HL2>
    col_block_index = (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                xcish[row, col] = np.sqrt(1.5) * (0.5 * rep_tens[v_orb, SOMO2, SOMO2, SOMO2] + 0.5 * rep_tens[v_orb, SOMO1, SOMO1, SOMO2] - rep_tens[v_orb, SOMO2, o_orb1, o_orb1])
            else:
                xcish[row, col] = - np.sqrt(1.5) * (rep_tens[v_orb, SOMO2, o_orb1, o_orb2])
            xcish[col,row] = xcish[row,col]
    #36 <HS2|H|1^HSD>
    col_block_index = 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        o_orb2 = 0
        o_orb3 = 0
        for col in range(col_block_index, col_block_index + ndoc1):
            if o_orb1 == o_orb2 and o_orb1 == o_orb3:
                xcish[row, col] = - np.sqrt(2) * (rep_tens[o_orb1, SOMO1, SOMO2, SOMO2] - rep_tens[o_orb1, SOMO1, o_orb1, o_orb1] + 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, SOMO1] - 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, SOMO1])
            elif o_orb1 != o_orb2 and o_orb2 == o_orb3:
                xcish[row, col] = np.sqrt(2) * rep_tens[o_orb2, SOMO1, o_orb1, o_orb2]
            elif o_orb1 == o_orb2 and o_orb1 != o_orb3:
                xcish[row, col] = - rep_tens[o_orb3, SOMO1, SOMO2, SOMO2] + rep_tens[o_orb3, SOMO1, o_orb1, o_orb1] + rep_tens[o_orb3, o_orb1, o_orb1, SOMO1] + 0.5 * rep_tens[o_orb3, SOMO1, SOMO1, SOMO1] - 0.5 * rep_tens[o_orb3, SOMO2, SOMO2, SOMO1]
            elif o_orb1 == o_orb3 and o_orb1 != o_orb2:
                xcish[row, col] = - rep_tens[o_orb2, SOMO1, SOMO2, SOMO2] + rep_tens[o_orb2, SOMO1, o_orb1, o_orb1] + rep_tens[o_orb2, o_orb1, o_orb1, SOMO1] + 0.5 * rep_tens[o_orb2, SOMO1, SOMO1, SOMO1] - 0.5 * rep_tens[o_orb2, SOMO2, SOMO2, SOMO1]
            else:
                xcish[row, col] = rep_tens[o_orb2, SOMO1, o_orb1, o_orb3] + rep_tens[o_orb3, SOMO1, o_orb1, o_orb2]
            xcish[col,row] = xcish[row,col]
            o_orb3 += 1
            if o_orb3 == ndocc:
                o_orb2 += 1
                o_orb3 = o_orb2
    #<HS2|H|1^SLD> = 0
    
    
    row_block_index = 2 * ndocc + 3
    #36 <SL1|H|SL1>
    col_block_index = 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(row, col_block_index + nvirt):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[SOMO1] - rep_tens[v_orb1, v_orb1, SOMO1, SOMO1] + 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] \
                             - 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] + 1.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb1] + 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb1]
            else:    
                xcish[row, col] = 1.5 * rep_tens[v_orb2, SOMO2, SOMO2, v_orb1] + 0.5 * rep_tens[v_orb2, SOMO1, SOMO1, v_orb1] -  rep_tens[v_orb2, v_orb1, SOMO2, SOMO2]
            xcish[col, row] = xcish[row,col]
    #37 <SL1|H|SL2>
    col_block_index = nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = 0.5 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO2] + 0.5 * rep_tens[SOMO1, SOMO2, SOMO2, SOMO2] - rep_tens[v_orb1, v_orb1, SOMO1, SOMO2] - rep_tens[v_orb1, SOMO1, SOMO2, v_orb1]
            else:    
                xcish[row, col] = - rep_tens[v_orb1, v_orb2, SOMO1, SOMO2] - rep_tens[v_orb1, SOMO2, SOMO1, v_orb2]
            xcish[col, row] = xcish[row,col]
    #38 <SL1|H|HL1>
    col_block_index = 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[o_orb, v_orb1, v_orb1, SOMO1] + 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO1] - rep_tens[o_orb, SOMO1, v_orb1, v_orb1] - 1.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO1])
            else:
                xcish[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[v_orb1, SOMO1, v_orb2, o_orb] - rep_tens[o_orb, SOMO1, v_orb1, v_orb2])
            xcish[col,row] = xcish[row,col]
    #39 <SL1|H|HL2>
    col_block_index = (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = np.sqrt(1.5) * (0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO1] + 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO1] - rep_tens[o_orb, SOMO1, v_orb1, v_orb1])
            else:
                xcish[row, col] = - np.sqrt(1.5) * (rep_tens[o_orb, SOMO1, v_orb1, v_orb2])
            xcish[col,row] = xcish[row,col]
    #<SL1|H|1^HSD> = 0
    #<SL1|H|1^SLD>
    col_block_index = ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        v_orb2 = SOMO2 + 1
        v_orb3 = SOMO2 + 1
        for col in range(col_block_index, col_block_index + ndcv1):
            if v_orb2 == v_orb3:
                if v_orb1 == v_orb2:
                    xcish[row, col] = np.sqrt(2) * (rep_tens[v_orb1, v_orb1, v_orb1, SOMO2] - rep_tens[v_orb1, SOMO2, SOMO1, SOMO1] + 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, SOMO2] - 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, SOMO2])
                else:
                    xcish[row, col] = np.sqrt(2) * rep_tens[v_orb2, SOMO2, v_orb2, v_orb1] 
            else:
                if v_orb1 == v_orb2:
                    xcish[row, col] = rep_tens[v_orb3, SOMO2, v_orb1, v_orb1] + rep_tens[SOMO2, v_orb1, v_orb1, v_orb3] - rep_tens[v_orb3, SOMO2, SOMO1, SOMO1] + 0.5 * rep_tens[v_orb3, SOMO1, SOMO1, SOMO2] - 0.5 * rep_tens[v_orb3, SOMO2, SOMO2, SOMO2]
                elif v_orb1 == v_orb3:
                    xcish[row, col] = rep_tens[v_orb2, SOMO2, v_orb1, v_orb1] + rep_tens[SOMO2, v_orb1, v_orb1, v_orb2] - rep_tens[v_orb2, SOMO2, SOMO1, SOMO1] + 0.5 * rep_tens[v_orb2, SOMO1, SOMO1, SOMO2] - 0.5 * rep_tens[v_orb2, SOMO2, SOMO2, SOMO2]
                else:
                    xcish[row, col] = rep_tens[v_orb2, SOMO2, v_orb1, v_orb3] + rep_tens[v_orb3, SOMO2, v_orb1, v_orb2]
            xcish[col,row] = xcish[row,col]
            v_orb3 += 1
            if v_orb3 == norbs:
                v_orb2 += 1
                v_orb3 = v_orb2
    
    
    
    row_block_index = nvirt + 2 * ndocc + 3
    #40 <SL2|H|SL2>
    col_block_index = nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(row, col_block_index + nvirt):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[SOMO2] - rep_tens[v_orb1, v_orb1, SOMO2, SOMO2] + 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] \
                             - 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] + 1.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb1] + 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb1]
            else:    
                xcish[row, col] = 1.5 * rep_tens[v_orb2, SOMO2, SOMO2, v_orb1] + 0.5 * rep_tens[v_orb2, SOMO1, SOMO1, v_orb1] -  rep_tens[v_orb2, v_orb1, SOMO2, SOMO2]
            xcish[col, row] = xcish[row,col]
    #41 <SL2|H|HL1>
    col_block_index = 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (rep_tens[o_orb,SOMO2,v_orb1,v_orb1] + 1.5*rep_tens[o_orb,SOMO1,SOMO1,SOMO2] - 2*rep_tens[o_orb,v_orb1,v_orb1,SOMO2] - 0.5*rep_tens[o_orb,SOMO2,SOMO2,SOMO2])
            else:
                xcish[row, col] = (1 / np.sqrt(2)) * (rep_tens[o_orb, SOMO2, v_orb1, v_orb2] - 2 * rep_tens[v_orb1, SOMO2, v_orb2, o_orb])
            xcish[col,row] = xcish[row,col]
    #42 <SL2|H|HL2>
    col_block_index = (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = np.sqrt(1.5) * (0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO2] + 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO2] - rep_tens[o_orb, SOMO2, v_orb1, v_orb1])
            else:
                xcish[row, col] = - np.sqrt(1.5) * (rep_tens[o_orb, SOMO2, v_orb1, v_orb2])
            xcish[col,row] = xcish[row,col]
    #<SL2|H|1^HSD> = 0
    #<SL2|H|1^SLD>
    col_block_index = ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        v_orb2 = SOMO2 + 1
        v_orb3 = SOMO2 + 1
        for col in range(col_block_index, col_block_index + ndcv1):
            if v_orb2 == v_orb3:
                if v_orb1 == v_orb2:
                    xcish[row, col] = np.sqrt(2) * (rep_tens[v_orb1, SOMO1, SOMO2, SOMO2] - rep_tens[v_orb1, v_orb1, v_orb1, SOMO1] + 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, SOMO1] - 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, SOMO1])
                else:
                    xcish[row, col] = - np.sqrt(2) * rep_tens[v_orb2, SOMO1, v_orb2, v_orb1] 
            else:
                if v_orb1 == v_orb2:
                    xcish[row, col] = rep_tens[v_orb3, SOMO1, SOMO2, SOMO2] - rep_tens[v_orb3, SOMO1, v_orb1, v_orb1] - rep_tens[SOMO1, v_orb1, v_orb1, v_orb3] + 0.5 * rep_tens[v_orb3, SOMO1, SOMO1, SOMO1] - 0.5 * rep_tens[v_orb3, SOMO2, SOMO2, SOMO1]
                elif v_orb1 == v_orb3:
                    xcish[row, col] = rep_tens[v_orb2, SOMO1, SOMO2, SOMO2] - rep_tens[v_orb2, SOMO1, v_orb1, v_orb1] - rep_tens[SOMO1, v_orb1, v_orb1, v_orb2] + 0.5 * rep_tens[v_orb2, SOMO1, SOMO1, SOMO1] - 0.5 * rep_tens[v_orb2, SOMO2, SOMO2, SOMO1]
                else:
                    xcish[row, col] = - rep_tens[v_orb2, SOMO1, v_orb1, v_orb3] - rep_tens[v_orb3, SOMO1, v_orb1, v_orb2]
            xcish[col,row] = xcish[row,col]
            v_orb3 += 1
            if v_orb3 == norbs:
                v_orb2 += 1
                v_orb3 = v_orb2
    
    
    row_block_index = 2 * nvirt + 2 * ndocc + 3
    #43 <HL1|H|HL1>
    col_block_index = 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + (ndocc * nvirt)):
        o_orb1 = (row - row_block_index) // nvirt
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                xcish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb1] - 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) + 1.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1] \
                                 + 2 * rep_tens[o_orb1, v_orb1, v_orb1, o_orb1]
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                xcish[row, col] =  2 * rep_tens[o_orb1, v_orb1, v_orb1, o_orb2] - rep_tens[o_orb1, o_orb2, v_orb1, v_orb1]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                xcish[row, col] = rep_tens[v_orb1, o_orb1, o_orb1, v_orb2] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb2]
            else:
                xcish[row, col] = 2 * rep_tens[o_orb1, v_orb1, o_orb2, v_orb2] - rep_tens[o_orb1, o_orb2, v_orb1, v_orb2]
            xcish[col, row] = xcish[row,col]
    #44 <HL1|H|HL2>
    col_block_index = (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + (ndocc * nvirt)):
        o_orb1 = (row - row_block_index) // nvirt
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                xcish[row, col] = (np.sqrt(3) / 2) * (rep_tens[o_orb1,SOMO1,SOMO1,o_orb1] - rep_tens[o_orb1,SOMO2,SOMO2,o_orb1] + rep_tens[v_orb1,SOMO2,SOMO2,v_orb1] - rep_tens[v_orb1,SOMO1,SOMO1,v_orb1])
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                xcish[row, col] =  (np.sqrt(3) / 2) * (rep_tens[o_orb1, SOMO1, SOMO1, o_orb2] - rep_tens[o_orb1, SOMO2, SOMO2, o_orb2])
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                xcish[row, col] =  (np.sqrt(3) / 2) * (rep_tens[v_orb1, SOMO2, SOMO2, v_orb2] - rep_tens[v_orb1, SOMO1, SOMO1, v_orb2])
            xcish[col, row] = xcish[row,col]
    #36 <HL1|H|1^HSD>
    col_block_index = 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + (ndocc * nvirt)):
        o_orb1 = (row - row_block_index) // nvirt
        o_orb2 = 0
        o_orb3 = 0
        v_orb = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + ndoc1):
            if o_orb1 == o_orb2 and o_orb1 == o_orb3:
                xcish[row, col] = - rep_tens[o_orb1, SOMO2, SOMO1, v_orb] - rep_tens[o_orb1, SOMO1, SOMO2, v_orb]
            elif o_orb1 != o_orb2 and o_orb2 == o_orb3:
                xcish[row, col] = 0
            elif o_orb1 == o_orb2 and o_orb1 != o_orb3:
                xcish[row, col] = - (np.sqrt(2) / 2) * (rep_tens[o_orb3, SOMO1, SOMO2, v_orb] + rep_tens[o_orb3, SOMO2, SOMO1, v_orb])
            elif o_orb1 == o_orb3 and o_orb1 != o_orb2:
                xcish[row, col] = - (np.sqrt(2) / 2) * (rep_tens[o_orb2, SOMO1, SOMO2, v_orb] + rep_tens[o_orb2, SOMO2, SOMO1, v_orb])
            else:
                xcish[row, col] = 0
            xcish[col,row] = xcish[row,col]
            o_orb3 += 1
            if o_orb3 == ndocc:
                o_orb2 += 1
                o_orb3 = o_orb2
    #<HL1|H|1^SLD>
    col_block_index = ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + (nvirt * ndocc)):
        o_orb = (row - row_block_index) // nvirt
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        v_orb2 = SOMO2 + 1
        v_orb3 = SOMO2 + 1
        for col in range(col_block_index, col_block_index + ndcv1):
            if v_orb2 == v_orb3:
                if v_orb1 == v_orb2:
                    xcish[row, col] = - rep_tens[o_orb, SOMO2, SOMO1, v_orb1] - rep_tens[o_orb, SOMO1, SOMO2, v_orb1]
            else:
                if v_orb1 == v_orb2:
                    xcish[row, col] = - (np.sqrt(2) / 2) * (rep_tens[o_orb, SOMO1, SOMO2, v_orb3] + rep_tens[o_orb, SOMO2, SOMO1, v_orb3])
                elif v_orb1 == v_orb3:
                    xcish[row, col] = - (np.sqrt(2) / 2) * (rep_tens[o_orb, SOMO1, SOMO2, v_orb2] + rep_tens[o_orb, SOMO2, SOMO1, v_orb2])
            xcish[col,row] = xcish[row,col]
            v_orb3 += 1
            if v_orb3 == norbs:
                v_orb2 += 1
                v_orb3 = v_orb2
    
    
    row_block_index = (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    #45 <HL2|H|HL2>
    col_block_index = (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + (ndocc * nvirt)):
        o_orb1 = (row - row_block_index) // nvirt
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                xcish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb1] - 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) - 0.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1] \
                                 + rep_tens[o_orb1, SOMO1, SOMO1, o_orb1] + rep_tens[o_orb1, SOMO2, SOMO2, o_orb1] + rep_tens[v_orb1, SOMO1, SOMO1, v_orb1] + rep_tens[v_orb1, SOMO2, SOMO2, v_orb1]
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                xcish[row, col] = rep_tens[o_orb1, SOMO1, SOMO1, o_orb2] + rep_tens[o_orb1, SOMO2, SOMO2, o_orb2] - rep_tens[o_orb1, o_orb2, v_orb1, v_orb1]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                xcish[row, col] = rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] + rep_tens[v_orb1, SOMO2, SOMO2, v_orb2] - rep_tens[v_orb1, v_orb2, o_orb1, o_orb1]
            else:
                xcish[row, col] = - rep_tens[o_orb1, o_orb2, v_orb1, v_orb2]
            xcish[col, row] = xcish[row,col]
    #36 <HL2|H|1^HSD>
    col_block_index = 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + (ndocc * nvirt)):
        o_orb1 = (row - row_block_index) // nvirt
        o_orb2 = 0
        o_orb3 = 0
        v_orb = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + ndoc1):
            if o_orb1 == o_orb2 and o_orb1 == o_orb3:
                xcish[row, col] = np.sqrt(3) * (rep_tens[o_orb1, SOMO2, SOMO1, v_orb] - rep_tens[o_orb1, SOMO1, SOMO2, v_orb])
            elif o_orb1 != o_orb2 and o_orb2 == o_orb3:
                xcish[row, col] = 0
            elif o_orb1 == o_orb2 and o_orb1 != o_orb3:
                xcish[row, col] = (np.sqrt(1.5)) * (rep_tens[o_orb3, SOMO2, SOMO1, v_orb] - rep_tens[o_orb3, SOMO1, SOMO2, v_orb])
            elif o_orb1 == o_orb3 and o_orb1 != o_orb2:
                xcish[row, col] = (np.sqrt(1.5)) * (rep_tens[o_orb2, SOMO2, SOMO1, v_orb] - rep_tens[o_orb2, SOMO1, SOMO2, v_orb])
            else:
                xcish[row, col] = 0
            xcish[col,row] = xcish[row,col]
            o_orb3 += 1
            if o_orb3 == ndocc:
                o_orb2 += 1
                o_orb3 = o_orb2
    #<HL2|H|1^SLD>
    col_block_index = ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + (nvirt * ndocc)):
        o_orb = (row - row_block_index) // nvirt
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        v_orb2 = SOMO2 + 1
        v_orb3 = SOMO2 + 1
        for col in range(col_block_index, col_block_index + ndcv1):
            if v_orb2 == v_orb3:
                if v_orb1 == v_orb2:
                    xcish[row, col] = np.sqrt(3) * (rep_tens[o_orb, SOMO2, SOMO1, v_orb1] - rep_tens[o_orb, SOMO1, SOMO2, v_orb1])
            else:
                if v_orb1 == v_orb2:
                    xcish[row, col] = np.sqrt(1.5) * (rep_tens[o_orb, SOMO2, SOMO1, v_orb3] - rep_tens[o_orb, SOMO1, SOMO2, v_orb3])
                elif v_orb1 == v_orb3:
                    xcish[row, col] = np.sqrt(1.5) * (rep_tens[o_orb, SOMO2, SOMO1, v_orb2] - rep_tens[o_orb, SOMO1, SOMO2, v_orb2])
            xcish[col,row] = xcish[row,col]
            v_orb3 += 1
            if v_orb3 == norbs:
                v_orb2 += 1
                v_orb3 = v_orb2
    
    
    #37 <1^HSD|H|1^HSD>
    row_block_index = 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    col_block_index = 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    o_orb1 = 0
    o_orb2 = 0
    for row in range(row_block_index, row_block_index + ndoc1):
        o_orb3 = 0
        o_orb4 = 0
        for col in range(col_block_index, col_block_index + ndoc1):
            if o_orb1 == o_orb2 and o_orb3 == o_orb4:
                if o_orb1 == o_orb3:
                    xcish[row, col] = energy0 - 2 * orb_energies[o_orb1] + orb_energies[SOMO1] + orb_energies[SOMO2] + rep_tens[o_orb1, o_orb1, o_orb1, o_orb1] - 2 * rep_tens[o_orb1,o_orb1,SOMO1,SOMO1] - 2 * rep_tens[o_orb1,o_orb1,SOMO2,SOMO2] \
                      + rep_tens[o_orb1, SOMO1, SOMO1, o_orb1] + rep_tens[o_orb1, SOMO2, SOMO2, o_orb1] + rep_tens[SOMO1,SOMO1,SOMO2,SOMO2] - 0.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1] + 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2])
                elif o_orb1 != o_orb3:
                    xcish[row, col] = rep_tens[o_orb1,o_orb3,o_orb3,o_orb1]
            elif o_orb1 == o_orb2 and o_orb3 != o_orb4:
                if o_orb1 == o_orb3:
                    xcish[row, col] = np.sqrt(2) * (rep_tens[o_orb1, o_orb4, o_orb1, o_orb1] - rep_tens[o_orb1, o_orb4, SOMO1, SOMO1] - rep_tens[o_orb1, o_orb4, SOMO2, SOMO2] + 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb4] + 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb4])
                elif o_orb1 == o_orb4:
                    xcish[row, col] = np.sqrt(2) * (rep_tens[o_orb1, o_orb3, o_orb1, o_orb1] - rep_tens[o_orb1, o_orb3, SOMO1, SOMO1] - rep_tens[o_orb1, o_orb3, SOMO2, SOMO2] + 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb3] + 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb3])
                elif o_orb3 != o_orb1 and o_orb4 != o_orb1:
                    xcish[row, col] = np.sqrt(2) * rep_tens[o_orb1, o_orb3, o_orb4, o_orb1]
            elif o_orb3 == o_orb4 and o_orb1 != o_orb2:
                if o_orb3 == o_orb1:
                    xcish[row, col] = np.sqrt(2) * (rep_tens[o_orb3, o_orb2, o_orb3, o_orb3] - rep_tens[o_orb3, o_orb2, SOMO1, SOMO1] - rep_tens[o_orb3, o_orb2, SOMO2, SOMO2] + 0.5 * rep_tens[o_orb3, SOMO1, SOMO1, o_orb2] + 0.5 * rep_tens[o_orb3, SOMO2, SOMO2, o_orb2])
                elif o_orb3 == o_orb2:
                    xcish[row, col] = np.sqrt(2) * (rep_tens[o_orb3, o_orb1, o_orb3, o_orb3] - rep_tens[o_orb3, o_orb1, SOMO1, SOMO1] - rep_tens[o_orb3, o_orb1, SOMO2, SOMO2] + 0.5 * rep_tens[o_orb3, SOMO1, SOMO1, o_orb1] + 0.5 * rep_tens[o_orb3, SOMO2, SOMO2, o_orb1])
                elif o_orb1 != o_orb3 and o_orb2 != o_orb4:
                    xcish[row, col] = np.sqrt(2) * rep_tens[o_orb3, o_orb1, o_orb2, o_orb3]
            elif o_orb1 != o_orb2 and o_orb3 != o_orb4:
                if o_orb1 == o_orb3 and o_orb2 == o_orb4:
                    xcish[row, col] = energy0 - orb_energies[o_orb1] - orb_energies[o_orb2] + orb_energies[SOMO1] + orb_energies[SOMO2] + rep_tens[o_orb1, o_orb1, o_orb2, o_orb2] - rep_tens[o_orb1, o_orb1, SOMO1, SOMO1] - rep_tens[o_orb1, o_orb1, SOMO2, SOMO2] \
                      - rep_tens[o_orb2, o_orb2, SOMO1, SOMO1] - rep_tens[o_orb2, o_orb2, SOMO2, SOMO2] + 0.5 * (rep_tens[o_orb1, SOMO1, SOMO1, o_orb1] + rep_tens[o_orb1, SOMO2, SOMO2, o_orb1] + rep_tens[o_orb2, SOMO1, SOMO1, o_orb2] + rep_tens[o_orb2, SOMO2, SOMO2, o_orb2]) \
                      + 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) + rep_tens[SOMO1,SOMO1,SOMO2,SOMO2] - 0.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1] + rep_tens[o_orb1, o_orb2, o_orb2, o_orb1]
                elif o_orb1 == o_orb3 and o_orb2 != o_orb4:
                    xcish[row,col] = rep_tens[o_orb2, o_orb4, o_orb1, o_orb1] + rep_tens[o_orb2, o_orb1, o_orb1, o_orb4] - rep_tens[o_orb2, o_orb4, SOMO1, SOMO1] - rep_tens[o_orb2, o_orb4, SOMO2, SOMO2]+ 0.5 * rep_tens[o_orb2, SOMO1, SOMO1, o_orb4] + 0.5 * rep_tens[o_orb2, SOMO2, SOMO2, o_orb4]
                elif o_orb2 == o_orb4 and o_orb1 != o_orb3:
                    xcish[row,col] = rep_tens[o_orb1, o_orb3, o_orb2, o_orb2] + rep_tens[o_orb1, o_orb2, o_orb2, o_orb3] - rep_tens[o_orb1, o_orb3, SOMO1, SOMO1] - rep_tens[o_orb1, o_orb3, SOMO2, SOMO2]+ 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb3] + 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb3]
                elif o_orb1 != o_orb3 and o_orb2 != o_orb4:
                    xcish[row,col] = rep_tens[o_orb1, o_orb3, o_orb2, o_orb4] + rep_tens[o_orb1, o_orb4, o_orb2, o_orb3]
            xcish[col,row] = xcish[row,col]
            o_orb4 += 1
            if o_orb4 == ndocc:
                o_orb3 += 1
                o_orb4 = o_orb3
        o_orb2 += 1
        if o_orb2 == ndocc:
            o_orb1 += 1
            o_orb2 = o_orb1
    #37 <1^HSD|H|1^SLD> = 0
    
    # <1^SLD|H|1^SLD> 
    row_block_index = ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    col_block_index = ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    v_orb1 = SOMO2 + 1
    v_orb2 = SOMO2 + 1
    for row in range(row_block_index, row_block_index + ndcv1):
        v_orb3 = SOMO2 + 1
        v_orb4 = SOMO2 + 1
        for col in range(col_block_index, col_block_index + ndcv1):
            if v_orb1 == v_orb2 and v_orb3 == v_orb4:
                if v_orb1 == v_orb3:
                    xcish[row, col] = energy0 + 2 * orb_energies[v_orb1] - orb_energies[SOMO1] - orb_energies[SOMO2] + rep_tens[v_orb1, v_orb1, v_orb1, v_orb1] - 2 * rep_tens[v_orb1,v_orb1,SOMO1,SOMO1] - 2 * rep_tens[v_orb1,v_orb1,SOMO2,SOMO2] \
                        + rep_tens[v_orb1, SOMO1, SOMO1, v_orb1] + rep_tens[v_orb1, SOMO2, SOMO2, v_orb1] + rep_tens[SOMO1,SOMO1,SOMO2,SOMO2] - 0.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1] + 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2])
                else:
                    xcish[row, col] = rep_tens[v_orb1,v_orb3,v_orb3,v_orb1]
            elif v_orb1 == v_orb2 and v_orb3 != v_orb4:
                if v_orb1 == v_orb3:
                    xcish[row, col] = np.sqrt(2) * (rep_tens[v_orb1, v_orb4, v_orb1, v_orb1] - rep_tens[v_orb1, v_orb4, SOMO1, SOMO1] - rep_tens[v_orb1, v_orb4, SOMO2, SOMO2] + 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb4] + 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb4])
                elif v_orb1 == v_orb4:
                    xcish[row, col] = np.sqrt(2) * (rep_tens[v_orb1, v_orb3, v_orb1, v_orb1] - rep_tens[v_orb1, v_orb3, SOMO1, SOMO1] - rep_tens[v_orb1, v_orb3, SOMO2, SOMO2] + 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb3] + 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb3])
                else:
                    xcish[row,col] = np.sqrt(2) * rep_tens[v_orb1,v_orb3,v_orb4,v_orb1]
            elif v_orb1 != v_orb2 and v_orb3 == v_orb4:
                if v_orb3 == v_orb1:
                    xcish[row, col] = np.sqrt(2) * (rep_tens[v_orb3, v_orb2, v_orb3, v_orb3] - rep_tens[v_orb3, v_orb2, SOMO1, SOMO1] - rep_tens[v_orb3, v_orb2, SOMO2, SOMO2] + 0.5 * rep_tens[v_orb3, SOMO1, SOMO1, v_orb2] + 0.5 * rep_tens[v_orb3, SOMO2, SOMO2, v_orb2])
                elif v_orb3 == v_orb2:
                    xcish[row, col] = np.sqrt(2) * (rep_tens[v_orb3, v_orb1, v_orb3, v_orb3] - rep_tens[v_orb3, v_orb1, SOMO1, SOMO1] - rep_tens[v_orb3, v_orb1, SOMO2, SOMO2] + 0.5 * rep_tens[v_orb3, SOMO1, SOMO1, v_orb1] + 0.5 * rep_tens[v_orb3, SOMO2, SOMO2, v_orb1])
                else:
                    xcish[row,col] = np.sqrt(2) * rep_tens[v_orb3,v_orb1,v_orb2,v_orb3]
            else:
                if v_orb1 == v_orb3 and v_orb2 == v_orb4:
                    xcish[row, col] = energy0 + orb_energies[v_orb1] + orb_energies[v_orb2] - orb_energies[SOMO1] - orb_energies[SOMO2] + rep_tens[v_orb1, v_orb1, v_orb2, v_orb2] - rep_tens[v_orb1, v_orb1, SOMO1, SOMO1] - rep_tens[v_orb1, v_orb1, SOMO2, SOMO2] \
                        - rep_tens[v_orb2, v_orb2, SOMO1, SOMO1] - rep_tens[v_orb2, v_orb2, SOMO2, SOMO2] + 0.5 * (rep_tens[v_orb1, SOMO1, SOMO1, v_orb1] + rep_tens[v_orb1, SOMO2, SOMO2, v_orb1] + rep_tens[v_orb2, SOMO1, SOMO1, v_orb2] + rep_tens[v_orb2, SOMO2, SOMO2, v_orb2]) \
                        + 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) + rep_tens[SOMO1,SOMO1,SOMO2,SOMO2] - 0.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1] + rep_tens[v_orb1, v_orb2, v_orb2, v_orb1]
                elif v_orb1 == v_orb3 and v_orb2 != v_orb4:
                    xcish[row, col] = rep_tens[v_orb2, v_orb4, v_orb1, v_orb1] + rep_tens[v_orb2, v_orb1, v_orb1, v_orb4] - rep_tens[v_orb2, v_orb4, SOMO1, SOMO1] - rep_tens[v_orb2, v_orb4, SOMO2, SOMO2]+ 0.5 * rep_tens[v_orb2, SOMO1, SOMO1, v_orb4] + 0.5 * rep_tens[v_orb2, SOMO2, SOMO2, v_orb4]
                elif v_orb1 != v_orb3 and v_orb2 == v_orb4:
                    xcish[row, col] = rep_tens[v_orb1, v_orb3, v_orb2, v_orb2] + rep_tens[v_orb1, v_orb2, v_orb2, v_orb3] - rep_tens[v_orb1, v_orb3, SOMO1, SOMO1] - rep_tens[v_orb1, v_orb3, SOMO2, SOMO2]+ 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb3] + 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb3]
                else:
                    xcish[row, col] = rep_tens[v_orb1, v_orb3, v_orb2, v_orb4] + rep_tens[v_orb1, v_orb4, v_orb2, v_orb3]
            xcish[col,row] = xcish[row,col]
            v_orb4 += 1
            if v_orb4 == norbs:
                v_orb3 += 1
                v_orb4 = v_orb3
        v_orb2 += 1
        if v_orb2 == norbs:
            v_orb1 += 1
            v_orb2 = v_orb1
    
    ################# TRIPLET BLOCK ######################
    
    row_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    #46 <OS3|H|OS3>
    xcish[row_index, row_index] = energy0 - 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) - (0.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1])
    #47 <OS3|H|HS1>
    col_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 4
    for col in range(col_index, col_index + ndocc):
        o_orb = col - col_index
        xcish[row_index, col] = 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO1] + 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO1]
        xcish[col, row_index] = xcish[row_index,col]
    #48 <OS3|H|HS2>
    col_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 3 * ndocc + 4
    for col in range(col_index, col_index + ndocc):
        o_orb = col - col_index
        xcish[row_index, col] = 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO2] + 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO2]
        xcish[col, row_index] = xcish[row_index,col]
    #49 <OS3|H|SL1>
    col_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 4
    for col in range(col_index, col_index + nvirt):
        v_orb = col - col_index + (SOMO2 + 1)
        xcish[row_index, col] = - 0.5 * rep_tens[v_orb, SOMO1, SOMO1, SOMO1] - 0.5 * rep_tens[v_orb, SOMO2, SOMO2, SOMO1]
        xcish[col, row_index] = xcish[row_index,col]
    #50 <OS3|H|SL2>
    col_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 4
    for col in range(col_index, col_index + nvirt):
        v_orb = col - col_index + (SOMO2 + 1)
        xcish[row_index, col] = 0.5 * rep_tens[v_orb, SOMO1, SOMO1, SOMO2] + 0.5 * rep_tens[v_orb, SOMO2, SOMO2, SOMO2]
        xcish[col, row_index] = xcish[row_index, col]
    #51 <OS3|H|HL1> = 0
    #52 <OS3|H|HL2>
    col_index =  ndcv1 + ndoc1 + 3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for col in range(col_index, col_index + (ndocc * nvirt)):
        o_orb = (col - col_index) // nvirt # Increase o_orb after every ndocc cols
        v_orb = (col - col_index) % nvirt + (SOMO2 + 1) # Increase v_orb then reset after ndocc cols
        xcish[row_index,col] = (1 / np.sqrt(2)) * (rep_tens[o_orb, SOMO1, SOMO1, v_orb] - rep_tens[o_orb, SOMO2, SOMO2, v_orb])
        xcish[col,row_index] = xcish[row_index,col]
    #53 <OS3|H|HL3>
    col_index = ndcv1 + ndoc1 + 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for col in range(col_index, col_index + (ndocc * nvirt)):
        o_orb = (col - col_index) // nvirt # Increase o_orb after every ndocc cols
        v_orb = (col - col_index) % nvirt + (SOMO2 + 1) # Increase v_orb then reset after ndocc cols
        xcish[row_index,col] = rep_tens[o_orb, SOMO1, SOMO1, v_orb] + rep_tens[o_orb, SOMO2, SOMO2, v_orb]
        xcish[col,row_index] = xcish[row_index,col]
    #54 <OS3|H|3^HSD>
    col_index = ndcv1 + ndoc1 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    o_orb1 = 0
    o_orb2 = 1
    for col in range(col_index, col_index + ndoc3):
        xcish[row_index,col] = rep_tens[o_orb1, SOMO1, SOMO2, o_orb2] - rep_tens[o_orb1, SOMO2, SOMO1, o_orb2]
        xcish[col,row_index] = xcish[row_index,col]
        o_orb2 += 1
        if o_orb2 >= ndocc:
            o_orb1 += 1
            o_orb2 = o_orb1 + 1
    # <OS3|H|3^SLD>
    col_index = ndcv1 + ndocc ** 2 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    v_orb1 = SOMO2 + 1
    v_orb2 = SOMO2 + 2
    for col in range(col_index, col_index + ndcv3):
        xcish[row_index,col] = rep_tens[v_orb1, SOMO1, SOMO2, v_orb2] - rep_tens[v_orb1, SOMO2, SOMO1, v_orb2]
        xcish[col,row_index] = xcish[row_index,col]
        v_orb2 += 1
        if v_orb2 >= norbs:
            v_orb1 += 1
            v_orb2 = v_orb1 + 1

    row_block_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 4
    #54 <HS1|H|HS1>
    col_block_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(row, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                xcish[row, col] = energy0 + orb_energies[SOMO1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, SOMO1, SOMO1] + 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] \
                             - 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] - 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb1] + 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb1]
            else:    
                xcish[row, col] = 0.5 * rep_tens[o_orb2, SOMO1, SOMO1, o_orb1] - 0.5 * rep_tens[o_orb2, SOMO2, SOMO2, o_orb1] - rep_tens[o_orb2, o_orb1, SOMO1, SOMO1]
            xcish[col, row] = xcish[row,col]
    #55 <HS1|H|HS2>
    col_block_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 3 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                xcish[row, col] = rep_tens[SOMO1, o_orb1, o_orb1, SOMO2] - rep_tens[o_orb1, o_orb1, SOMO1, SOMO2] + 0.5 * rep_tens[SOMO2, SOMO1, SOMO1, SOMO1] \
                             + 0.5 * rep_tens[SOMO1, SOMO2, SOMO2, SOMO2]                                                               
            else:    
                xcish[row, col] = rep_tens[o_orb2, SOMO1, SOMO2, o_orb1] - rep_tens[o_orb2, o_orb1, SOMO1, SOMO2]
            xcish[col, row] = xcish[row,col]
    #56 <HS1|H|SL1>
    col_block_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb = col - col_block_index + (SOMO2 + 1)
            xcish[row, col] = rep_tens[o_orb, SOMO1, SOMO1, v_orb]
            xcish[col, row] = xcish[row,col]
    #57 <HS1|H|SL2>
    col_block_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb = col - col_block_index + (SOMO2 + 1)
            xcish[row, col] = - rep_tens[o_orb, SOMO1, SOMO2, v_orb]
            xcish[col, row] = xcish[row,col]
    #58 <HS1|H|HL1>
    col_block_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[v_orb, o_orb1, o_orb1, SOMO1] + 0.5 * rep_tens[v_orb, SOMO2, SOMO2, SOMO1] + 0.5 * rep_tens[v_orb, SOMO1, SOMO1, SOMO1] \
                                   - rep_tens[o_orb1, o_orb1, SOMO1, v_orb])
            else:
                xcish[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[v_orb, o_orb2, o_orb1, SOMO1] - rep_tens[v_orb, SOMO1, o_orb1, o_orb2])
            xcish[col,row] = xcish[row,col]
    #59 <HS1|H|HL2>
    col_block_index = ndcv1 + ndoc1 + 3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (0.5 * rep_tens[v_orb, SOMO1, SOMO1, SOMO1] - 1.5 * rep_tens[v_orb, SOMO2, SOMO2, SOMO1] - rep_tens[v_orb, SOMO1, o_orb1, o_orb1])
            else:
                xcish[row, col] = - (1 / np.sqrt(2)) * (rep_tens[v_orb, SOMO1, o_orb1, o_orb2])
            xcish[col,row] = xcish[row,col]
    #60 <HS1|H|HL3>
    col_block_index = ndcv1 + ndoc1 + 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                xcish[row, col] = 0.5 * rep_tens[v_orb, SOMO1, SOMO1, SOMO1] + 0.5 * rep_tens[v_orb, SOMO2, SOMO2, SOMO1] - rep_tens[v_orb, SOMO1, o_orb1, o_orb1]
            else:
                xcish[row, col] = - rep_tens[v_orb, SOMO1, o_orb1, o_orb2]
            xcish[col,row] = xcish[row,col]
    #61 <HS1|H|3^HSD>
    col_block_index = ndcv1 + ndoc1 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        o_orb2 = 0
        o_orb3 = 1
        for col in range(col_block_index, col_block_index + ndoc3):
            if o_orb1 == o_orb2:
                xcish[row, col] = rep_tens[o_orb3, o_orb1, o_orb1, SOMO2] + rep_tens[o_orb3, SOMO1, SOMO1, SOMO2] - rep_tens[o_orb3, SOMO2, o_orb1, o_orb1] + 0.5 * rep_tens[o_orb3, SOMO2, SOMO2, SOMO2] - 0.5 * rep_tens[o_orb3, SOMO1, SOMO1, SOMO2]
            elif o_orb1 == o_orb3:
                xcish[row, col] = - (rep_tens[o_orb2, o_orb1, o_orb1, SOMO2] + rep_tens[o_orb2, SOMO1, SOMO1, SOMO2] - rep_tens[o_orb2, SOMO2, o_orb1, o_orb1] + 0.5 * rep_tens[o_orb2, SOMO2, SOMO2, SOMO2] - 0.5 * rep_tens[o_orb2, SOMO1, SOMO1, SOMO2])
            else:
                xcish[row, col] = rep_tens[o_orb2, SOMO2, o_orb1, o_orb3] - rep_tens[o_orb3, SOMO2, o_orb1, o_orb2]
            xcish[col,row] = xcish[row,col]
            o_orb3 += 1
            if o_orb3 >= ndocc:
                o_orb2 += 1
                o_orb3 = o_orb2 + 1
    #61 <HS1|H|3^SLD> = 0
    
    row_block_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 3 * ndocc + 4
    #61 <HS2|H|HS2>
    col_block_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 3 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(row, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                xcish[row, col] = energy0 + orb_energies[SOMO2] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, SOMO2, SOMO2] + 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] \
                             - 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] - 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb1] + 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb1]
            else:
                xcish[row, col] = rep_tens[o_orb1, o_orb2, SOMO2, SOMO2] - 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb2] - 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb2]
            xcish[col, row] = xcish[row,col]
    #62 <HS2|H|SL1>
    col_block_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb = col - col_block_index + (SOMO2 + 1)
            xcish[row, col] = rep_tens[o_orb, SOMO2, SOMO1, v_orb] 
            xcish[col, row] = xcish[row,col]
    #63 <HS2|H|SL2>
    col_block_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb = col - col_block_index + (SOMO2 + 1)
            xcish[row, col] = - rep_tens[o_orb, SOMO2, SOMO2, v_orb]
            xcish[col, row] = xcish[row,col]
    #64 <HS2|H|HL1>
    col_block_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[v_orb, o_orb1, o_orb1, SOMO2] + 0.5 * rep_tens[v_orb, SOMO1, SOMO1, SOMO2] + 0.5 * rep_tens[v_orb, SOMO2, SOMO2, SOMO2] \
                                   - rep_tens[o_orb1, o_orb1, SOMO2, v_orb])
            else:
                xcish[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[v_orb, o_orb2, o_orb1, SOMO2] - rep_tens[v_orb, SOMO2, o_orb1, o_orb2])
            xcish[col,row] = xcish[row,col]
    #65 <HS2|H|HL2>
    col_block_index = ndcv1 + ndoc1 + 3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (1.5 * rep_tens[v_orb, SOMO1, SOMO1, SOMO2] + rep_tens[v_orb, SOMO2, o_orb1, o_orb1] - 0.5 * rep_tens[v_orb, SOMO2, SOMO2, SOMO2])
            else:
                xcish[row, col] = (1 / np.sqrt(2)) * (rep_tens[v_orb, SOMO2, o_orb1, o_orb2])
            xcish[col,row] = xcish[row,col]
    #66 <HS2|H|HL3>
    col_block_index = ndcv1 + ndoc1 + 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                xcish[row, col] = 0.5 * rep_tens[v_orb, SOMO2, SOMO2, SOMO2] + 0.5 * rep_tens[v_orb, SOMO1, SOMO1, SOMO2] - rep_tens[v_orb, SOMO2, o_orb1, o_orb1]
            else:
                xcish[row, col] = - rep_tens[v_orb, SOMO2, o_orb1, o_orb2]
            xcish[col,row] = xcish[row,col]
    # <HS2|H|3^HSD>
    col_block_index = ndcv1 + ndoc1 + 5 * (ndocc * nvirt) + 4 * ndocc + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        o_orb2 = 0
        o_orb3 = 1
        for col in range(col_block_index, col_block_index + ndoc3):
            if o_orb1 == o_orb2:
                xcish[row, col] = - (rep_tens[o_orb3, o_orb1, o_orb1, SOMO1] + rep_tens[o_orb3, SOMO2, SOMO2, SOMO1] - rep_tens[o_orb3, SOMO1, o_orb1, o_orb1] + 0.5 * rep_tens[o_orb3, SOMO1, SOMO1, SOMO1] - 0.5 * rep_tens[o_orb3, SOMO2, SOMO2, SOMO1])
            elif o_orb1 == o_orb3:
                xcish[row, col] = rep_tens[o_orb2, o_orb1, o_orb1, SOMO1] + rep_tens[o_orb2, SOMO2, SOMO2, SOMO1] - rep_tens[o_orb2, SOMO1, o_orb1, o_orb1] + 0.5 * rep_tens[o_orb2, SOMO1, SOMO1, SOMO1] - 0.5 * rep_tens[o_orb2, SOMO2, SOMO2, SOMO1]
            else:
                xcish[row, col] = rep_tens[o_orb3, SOMO1, o_orb1, o_orb2] - rep_tens[o_orb2, SOMO1, o_orb1, o_orb3]
            xcish[col,row] = xcish[row,col]
            o_orb3 += 1
            if o_orb3 >= ndocc:
                o_orb2 += 1
                o_orb3 = o_orb2 + 1
    # <HS2|H|3^SLD> = 0
    
    
    row_block_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 4
    #67 <SL1|H|SL1>
    col_block_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(row, col_block_index + nvirt):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[SOMO1] - rep_tens[v_orb1, v_orb1, SOMO1, SOMO1] + 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] \
                             - 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] - 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb1] + 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb1]
            else:    
                xcish[row, col] =  0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] - 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb2] - rep_tens[v_orb1, v_orb2, SOMO1, SOMO1]
            xcish[col, row] = xcish[row,col]
    #68 <SL1|H|SL2>
    col_block_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = rep_tens[v_orb1, v_orb1, SOMO1, SOMO2] - rep_tens[v_orb1, SOMO1, SOMO2, v_orb1] - 0.5 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO2] - 0.5 * rep_tens[SOMO1, SOMO2, SOMO2, SOMO2]
            else:    
                xcish[row, col] = rep_tens[v_orb1, v_orb2, SOMO1, SOMO2] - rep_tens[v_orb1, SOMO2, SOMO1, v_orb2]
            xcish[col, row] = xcish[row,col]
    #69 <SL1|H|HL1>
    col_block_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[o_orb, v_orb1, v_orb1, SOMO1] + 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO1] + 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO1] \
                                  - rep_tens[o_orb, SOMO1, v_orb1, v_orb1])
            else:
                xcish[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[v_orb1, SOMO1, v_orb2, o_orb] - rep_tens[o_orb, SOMO1, v_orb1, v_orb2])
            xcish[col,row] = xcish[row,col]
    #70 <SL1|H|HL2>
    col_block_index = ndcv1 + ndoc1 + 3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (rep_tens[o_orb, SOMO1, v_orb1, v_orb1] + 1.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO1] - 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO1])
            else:
                xcish[row, col] = - (1 / np.sqrt(2)) * (rep_tens[o_orb, SOMO1, v_orb1, v_orb2])
            xcish[col,row] = xcish[row,col]
    #71 <SL1|H|HL3>
    col_block_index = ndcv1 + ndoc1 + 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = rep_tens[o_orb, SOMO1, v_orb1, v_orb1] - 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO1] - 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO1]
            else:
                xcish[row, col] = rep_tens[o_orb, SOMO1, v_orb1, v_orb2]
            xcish[col,row] = xcish[row,col]
    # <SL1|H|3^HSD> = 0
    # <SL1|H|3^SLD>
    col_block_index = ndcv1 + ndocc ** 2 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        v_orb2 = SOMO2 + 1
        v_orb3 = SOMO2 + 2
        for col in range(col_block_index, col_block_index + ndcv3):
            if v_orb1 == v_orb2:
                xcish[row, col] = - (rep_tens[v_orb3, SOMO2, SOMO1, SOMO1] - rep_tens[v_orb3, SOMO2, v_orb1, v_orb1] + rep_tens[SOMO2, v_orb1, v_orb1, v_orb3] + 0.5 * rep_tens[v_orb3, SOMO2, SOMO2, SOMO2] - 0.5 * rep_tens[v_orb3, SOMO1, SOMO1, SOMO2])
            elif v_orb1 == v_orb3:
                xcish[row, col] = - (rep_tens[v_orb2, SOMO2, SOMO1, SOMO1] - rep_tens[v_orb2, SOMO2, v_orb1, v_orb1] + rep_tens[SOMO2, v_orb1, v_orb1, v_orb2] + 0.5 * rep_tens[v_orb2, SOMO2, SOMO2, SOMO2] - 0.5 * rep_tens[v_orb2, SOMO1, SOMO1, SOMO2])
            else:
                xcish[row, col] = rep_tens[v_orb3, SOMO2, v_orb1, v_orb2] - rep_tens[v_orb2, SOMO2, v_orb1, v_orb3]
            xcish[col,row] = xcish[row,col]
            v_orb3 += 1
            if v_orb3 >= norbs:
                v_orb2 += 1
                v_orb3 = v_orb2 + 1
    
    
    row_block_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 4
    #72 <SL2|H|SL2>
    col_block_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(row, col_block_index + nvirt):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[SOMO2] - rep_tens[v_orb1, v_orb1, SOMO2, SOMO2] + 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] \
                             - 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] - 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb1] + 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb1]
            else:    
                xcish[row, col] =  0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] - rep_tens[v_orb1, v_orb2, SOMO2, SOMO2] - 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb2]
            xcish[col, row] = xcish[row,col]
    #73 <SL2|H|HL1>
    col_block_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (rep_tens[o_orb, SOMO2, v_orb1, v_orb1] - 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO2] - 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO2] \
                                  - 2 * rep_tens[o_orb, v_orb1, v_orb1, SOMO2])
            else:
                xcish[row, col] = (1 / np.sqrt(2)) * (rep_tens[o_orb, SOMO2, v_orb1, v_orb2] - 2 * rep_tens[v_orb1, SOMO2, v_orb2, o_orb])
            xcish[col,row] = xcish[row,col]
    #74 <SL2|H|HL2>
    col_block_index = ndcv1 + ndoc1 + 3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (rep_tens[o_orb, SOMO2, v_orb1, v_orb1] + 1.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO2] - 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO2])
            else:
                xcish[row, col] = - (1 / np.sqrt(2)) * (rep_tens[o_orb, SOMO2, v_orb1, v_orb2])
            xcish[col,row] = xcish[row,col]
    #75 <SL2|H|HL3>
    col_block_index = ndcv1 + ndoc1 + 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO2] + 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO2] - rep_tens[o_orb, SOMO2, v_orb1, v_orb1]
            else:
                xcish[row, col] = - rep_tens[o_orb, SOMO2, v_orb1, v_orb2]
            xcish[col,row] = xcish[row,col]    
    # <SL2|H|3^HSD> = 0
    # <SL2|H|3^SLD>
    col_block_index = ndcv1 + ndocc ** 2 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        v_orb2 = SOMO2 + 1
        v_orb3 = SOMO2 + 2
        for col in range(col_block_index, col_block_index + ndcv3):
            if v_orb1 == v_orb2:
                xcish[row, col] = rep_tens[v_orb3, SOMO1, SOMO2, SOMO2] - rep_tens[v_orb3, SOMO1, v_orb1, v_orb1] + rep_tens[SOMO1, v_orb1, v_orb1, v_orb3] + 0.5 * rep_tens[v_orb3, SOMO1, SOMO1, SOMO1] - 0.5 * rep_tens[v_orb3, SOMO2, SOMO2, SOMO1]
            elif v_orb1 == v_orb3:
                xcish[row, col] = rep_tens[v_orb2, SOMO1, SOMO2, SOMO2] - rep_tens[v_orb2, SOMO1, v_orb1, v_orb1] + rep_tens[SOMO1, v_orb1, v_orb1, v_orb2] + 0.5 * rep_tens[v_orb2, SOMO1, SOMO1, SOMO1] - 0.5 * rep_tens[v_orb2, SOMO2, SOMO2, SOMO1]
            else:
                xcish[row, col] = rep_tens[v_orb2, SOMO1, v_orb1, v_orb3] - rep_tens[v_orb3, SOMO1, v_orb1, v_orb2]
            xcish[col,row] = xcish[row,col]
            v_orb3 += 1
            if v_orb3 >= norbs:
                v_orb2 += 1
                v_orb3 = v_orb2 + 1
    
    
    row_block_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    #76 <HL1|H|HL1>
    col_block_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + (ndocc * nvirt)):
        o_orb1 = (row - row_block_index) // nvirt
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                xcish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb1] - 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) - 0.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1] \
                                 + 2 * rep_tens[o_orb1, v_orb1, v_orb1, o_orb1]
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                xcish[row, col] =  2 * rep_tens[o_orb1, v_orb1, v_orb1, o_orb2] - rep_tens[o_orb1, o_orb2, v_orb1, v_orb1]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                xcish[row, col] = rep_tens[v_orb1, o_orb1, o_orb1, v_orb2] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb2]
            else:
                xcish[row, col] = 2 * rep_tens[o_orb1, v_orb1, o_orb2, v_orb2] - rep_tens[o_orb1, o_orb2, v_orb1, v_orb2]
            xcish[col, row] = xcish[row,col]
    #77 <HL1|H|HL2>
    col_block_index = ndcv1 + ndoc1 + 3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + (ndocc * nvirt)):
        o_orb1 = (row - row_block_index) // nvirt
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                xcish[row, col] = 0.5 * (rep_tens[o_orb1,SOMO2,SOMO2,o_orb1] - rep_tens[o_orb1,SOMO1,SOMO1,o_orb1] + rep_tens[v_orb1,SOMO1,SOMO1,v_orb1] - rep_tens[v_orb1,SOMO2,SOMO2,v_orb1])
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                xcish[row, col] =  0.5 * (rep_tens[o_orb1, SOMO2, SOMO2, o_orb2] - rep_tens[o_orb1, SOMO1, SOMO1, o_orb2])
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                xcish[row, col] =  0.5 * (rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] - rep_tens[v_orb1, SOMO2, SOMO2, v_orb2])
            xcish[col, row] = xcish[row,col]
    #78 <HL1|H|HL3>
    col_block_index = ndcv1 + ndoc1 + 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + (ndocc * nvirt)):
        o_orb1 = (row - row_block_index) // nvirt
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (rep_tens[v_orb1,SOMO1,SOMO1,v_orb1] + rep_tens[v_orb1,SOMO2,SOMO2,v_orb1] - rep_tens[o_orb1,SOMO1,SOMO1,o_orb1] - rep_tens[o_orb1,SOMO2,SOMO2,o_orb1])
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                xcish[row, col] =  - (1 / np.sqrt(2)) * (rep_tens[o_orb1, SOMO1, SOMO1, o_orb2] + rep_tens[o_orb1, SOMO2, SOMO2, o_orb2])
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                xcish[row, col] =  (1 / np.sqrt(2)) * (rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] + rep_tens[v_orb1, SOMO2, SOMO2, v_orb2])
            xcish[col, row] = xcish[row,col]
    # <HL1|H|3^HSD>
    col_block_index = ndcv1 + ndoc1 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + (ndocc * nvirt)):
        o_orb1 = (row - row_block_index) // nvirt
        o_orb2 = 0
        o_orb3 = 1
        v_orb = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + ndoc3):
            if o_orb1 == o_orb2:
                xcish[row, col] = (np.sqrt(2) / 2) * (rep_tens[o_orb3, SOMO2, SOMO1, v_orb] - rep_tens[o_orb3, SOMO1, SOMO2, v_orb])
            elif o_orb1 == o_orb3:
                xcish[row, col] = - (np.sqrt(2) / 2) * (rep_tens[o_orb2, SOMO2, SOMO1, v_orb] - rep_tens[o_orb2, SOMO1, SOMO2, v_orb])
            else:
                xcish[row, col] = 0
            xcish[col,row] = xcish[row,col]
            o_orb3 += 1
            if o_orb3 >= ndocc:
                o_orb2 += 1
                o_orb3 = o_orb2 + 1
    # <HL1|H|3^HSD>
    col_block_index = ndcv1 + ndocc ** 2 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + (nvirt * ndocc)):
        o_orb = (row - row_block_index) // nvirt
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        v_orb2 = SOMO2 + 1
        v_orb3 = SOMO2 + 2
        for col in range(col_block_index, col_block_index + ndcv3):
            if v_orb1 == v_orb2:
                xcish[row, col] = (np.sqrt(2) / 2) * (rep_tens[o_orb, SOMO2, SOMO1, v_orb3] - rep_tens[o_orb, SOMO1, SOMO2, v_orb3])
            elif v_orb1 == v_orb3:
                xcish[row, col] = (np.sqrt(2) / 2) * (rep_tens[o_orb, SOMO1, SOMO2, v_orb2] - rep_tens[o_orb, SOMO2, SOMO1, v_orb2])
            xcish[col,row] = xcish[row,col]
            v_orb3 += 1
            if v_orb3 >= norbs:
                v_orb2 += 1
                v_orb3 = v_orb2 + 1
    
    row_block_index = ndcv1 + ndoc1 + 3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    #79 <HL2|H|HL2>
    col_block_index = ndcv1 + ndoc1 + 3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + (ndocc * nvirt)):
        o_orb1 = (row - row_block_index) // nvirt
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                xcish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb1] - 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) + 1.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1]
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                xcish[row, col] =  - rep_tens[o_orb1, o_orb2, v_orb1, v_orb1]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                xcish[row, col] = - rep_tens[v_orb1, o_orb1, o_orb1, v_orb2] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb2]
            else:
                xcish[row, col] = - rep_tens[o_orb1, o_orb2, v_orb1, v_orb2]
            xcish[col, row] = xcish[row,col]
    #80 <HL2|H|HL3>
    col_block_index = ndcv1 + ndoc1 + 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + (ndocc * nvirt)):
        o_orb1 = (row - row_block_index) // nvirt
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (rep_tens[o_orb1,SOMO1,SOMO1,o_orb1] - rep_tens[o_orb1,SOMO2,SOMO2,o_orb1] + rep_tens[v_orb1,SOMO1,SOMO1,v_orb1] - rep_tens[v_orb1,SOMO2,SOMO2,v_orb1])
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                xcish[row, col] =  (1 / np.sqrt(2)) * (rep_tens[o_orb1, SOMO1, SOMO1, o_orb2] - rep_tens[o_orb1, SOMO2, SOMO2, o_orb2])
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                xcish[row, col] =  (1 / np.sqrt(2)) * (rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] - rep_tens[v_orb1, SOMO2, SOMO2, v_orb2])
            xcish[col, row] = xcish[row,col]
    # <HL2|H|3^HSD>
    col_block_index = ndcv1 + ndoc1 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + (ndocc * nvirt)):
        o_orb1 = (row - row_block_index) // nvirt
        o_orb2 = 0
        o_orb3 = 1
        v_orb = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + ndoc3):
            if o_orb1 == o_orb2:
                xcish[row, col] = - (np.sqrt(2) / 2) * (rep_tens[o_orb3, SOMO2, SOMO1, v_orb] + rep_tens[o_orb3, SOMO1, SOMO2, v_orb])
            elif o_orb1 == o_orb3:
                xcish[row, col] = (np.sqrt(2) / 2) * (rep_tens[o_orb2, SOMO2, SOMO1, v_orb] + rep_tens[o_orb2, SOMO1, SOMO2, v_orb])
            else:
                xcish[row, col] = 0
            xcish[col,row] = xcish[row,col]
            o_orb3 += 1
            if o_orb3 >= ndocc:
                o_orb2 += 1
                o_orb3 = o_orb2 + 1
    # <HL2|H|3^SLD>
    col_block_index = ndcv1 + ndocc ** 2 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + (nvirt * ndocc)):
        o_orb = (row - row_block_index) // nvirt
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        v_orb2 = SOMO2 + 1
        v_orb3 = SOMO2 + 2
        for col in range(col_block_index, col_block_index + ndcv3):
            if v_orb1 == v_orb2:
                xcish[row, col] = (np.sqrt(2) / 2) * (rep_tens[o_orb, SOMO2, SOMO1, v_orb3] + rep_tens[o_orb, SOMO1, SOMO2, v_orb3])
            elif v_orb1 == v_orb3:
                xcish[row, col] = - (np.sqrt(2) / 2) * (rep_tens[o_orb, SOMO2, SOMO1, v_orb2] + rep_tens[o_orb, SOMO1, SOMO2, v_orb2])
            xcish[col,row] = xcish[row,col]
            v_orb3 += 1
            if v_orb3 >= norbs:
                v_orb2 += 1
                v_orb3 = v_orb2 + 1
    
    row_block_index = ndcv1 + ndoc1 + 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    #81 <HL3|H|HL3>
    col_block_index = ndcv1 + ndoc1 + 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + (ndocc * nvirt)):
        o_orb1 = (row - row_block_index) // nvirt
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                xcish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb1] - 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) - 0.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1] \
                                  + 0.5 * (rep_tens[o_orb1, SOMO1, SOMO1, o_orb1] + rep_tens[o_orb1, SOMO2, SOMO2, o_orb1] + rep_tens[v_orb1, SOMO1, SOMO1, v_orb1] + rep_tens[v_orb1, SOMO2, SOMO2, v_orb1])
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                xcish[row, col] =  0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb2] + 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb2] - rep_tens[o_orb1, o_orb2, v_orb1, v_orb1]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                xcish[row, col] = 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] + 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb2] - rep_tens[v_orb1, v_orb2, o_orb1, o_orb1]
            else:
                xcish[row, col] = - rep_tens[o_orb1, o_orb2, v_orb1, v_orb2]
            xcish[col, row] = xcish[row,col]
    # <HL3|H|3^HSD>
    col_block_index = ndcv1 + ndoc1 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + (ndocc * nvirt)):
        o_orb1 = (row - row_block_index) // nvirt
        o_orb2 = 0
        o_orb3 = 1
        v_orb = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + ndoc3):
            if o_orb1 == o_orb2:
                xcish[row, col] = (np.sqrt(2) / 2) * (rep_tens[o_orb3, SOMO2, SOMO1, v_orb] - rep_tens[o_orb3, SOMO1, SOMO2, v_orb])
            elif o_orb1 == o_orb3:
                xcish[row, col] = - (np.sqrt(2) / 2) * (rep_tens[o_orb2, SOMO2, SOMO1, v_orb] - rep_tens[o_orb2, SOMO1, SOMO2, v_orb])
            else:
                xcish[row, col] = 0
            xcish[col,row] = xcish[row,col]
            o_orb3 += 1
            if o_orb3 >= ndocc:
                o_orb2 += 1
                o_orb3 = o_orb2 + 1
    # <HL3|H|3^SLD>
    col_block_index = ndcv1 + ndocc ** 2 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + (nvirt * ndocc)):
        o_orb = (row - row_block_index) // nvirt
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        v_orb2 = SOMO2 + 1
        v_orb3 = SOMO2 + 2
        for col in range(col_block_index, col_block_index + ndcv3):
            if v_orb1 == v_orb2:
                xcish[row, col] = rep_tens[o_orb, SOMO1, SOMO2, v_orb3] - rep_tens[o_orb, SOMO2, SOMO1, v_orb3]
            elif v_orb1 == v_orb3:
                xcish[row, col] = rep_tens[o_orb, SOMO2, SOMO1, v_orb2] - rep_tens[o_orb, SOMO1, SOMO2, v_orb2]
            xcish[col,row] = xcish[row,col]
            v_orb3 += 1
            if v_orb3 >= norbs:
                v_orb2 += 1
                v_orb3 = v_orb2 + 1
    
    
    # <3^HSD|H|3^HSD>
    row_block_index = ndcv1 + ndoc1 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    col_block_index = ndcv1 + ndoc1 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    o_orb1 = 0
    o_orb2 = 1
    for row in range(row_block_index, row_block_index + ndoc3):
        o_orb3 = 0
        o_orb4 = 1
        for col in range(col_block_index, col_block_index + ndoc3):
            if o_orb1 == o_orb3 and o_orb2 == o_orb4:
                xcish[row, col] = energy0 - orb_energies[o_orb1] - orb_energies[o_orb2] + orb_energies[SOMO1] + orb_energies[SOMO2] + rep_tens[o_orb1, o_orb1, o_orb2, o_orb2] - rep_tens[o_orb1, o_orb1, SOMO1, SOMO1] - rep_tens[o_orb1, o_orb1, SOMO2, SOMO2] \
                    - rep_tens[o_orb2, o_orb2, SOMO1, SOMO1] - rep_tens[o_orb2, o_orb2, SOMO2, SOMO2] + 0.5 * (rep_tens[o_orb1, SOMO1, SOMO1, o_orb1] + rep_tens[o_orb1, SOMO2, SOMO2, o_orb1] + rep_tens[o_orb2, SOMO1, SOMO1, o_orb2] + rep_tens[o_orb2, SOMO2, SOMO2, o_orb2]) \
                    + 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) + rep_tens[SOMO1,SOMO1,SOMO2,SOMO2] - 0.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1] - rep_tens[o_orb1, o_orb2, o_orb2, o_orb1]
            elif o_orb1 == o_orb3 and o_orb2 != o_orb4:
                xcish[row,col] = rep_tens[o_orb2, o_orb1, o_orb1, o_orb4] - rep_tens[o_orb2, o_orb4, o_orb1, o_orb1] + rep_tens[o_orb2, o_orb4, SOMO1, SOMO1] + rep_tens[o_orb2, o_orb4, SOMO2, SOMO2] - 0.5 * rep_tens[o_orb2, SOMO1, SOMO1, o_orb4] - 0.5 * rep_tens[o_orb2, SOMO2, SOMO2, o_orb4]
            elif o_orb2 == o_orb4 and o_orb1 != o_orb3:
                xcish[row,col] = -(rep_tens[o_orb1, o_orb2, o_orb2, o_orb3] - rep_tens[o_orb1, o_orb3, o_orb2, o_orb2] + rep_tens[o_orb1, o_orb3, SOMO1, SOMO1] + rep_tens[o_orb1, o_orb3, SOMO2, SOMO2] - 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb3] - 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb3])
            else:
                xcish[row,col] = rep_tens[o_orb1, o_orb3, o_orb2, o_orb4] - rep_tens[o_orb1, o_orb4, o_orb2, o_orb3]
            xcish[col,row] = xcish[row,col]
            o_orb4 += 1
            if o_orb4 >= ndocc:
                o_orb3 += 1
                o_orb4 = o_orb3 + 1
        o_orb2 += 1
        if o_orb2 >= ndocc:
            o_orb1 += 1
            o_orb2 = o_orb1 + 1
    # <3^HSD|H|3^SLD> = 0
    
    # <3^SLD|H|3^SLD> 
    row_block_index = ndcv1 + ndocc ** 2 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    col_block_index = ndcv1 + ndocc ** 2 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    v_orb1 = SOMO2 + 1
    v_orb2 = SOMO2 + 2
    for row in range(row_block_index, row_block_index + ndcv3):
        v_orb3 = SOMO2 + 1
        v_orb4 = SOMO2 + 2
        for col in range(col_block_index, col_block_index + ndcv3):
            if v_orb1 == v_orb3 and v_orb2 == v_orb4:
                xcish[row, col] = energy0 + orb_energies[v_orb1] + orb_energies[v_orb2] - orb_energies[SOMO1] - orb_energies[SOMO2] + rep_tens[v_orb1, v_orb1, v_orb2, v_orb2] - rep_tens[v_orb1, v_orb1, SOMO1, SOMO1] - rep_tens[v_orb1, v_orb1, SOMO2, SOMO2] \
                    - rep_tens[v_orb2, v_orb2, SOMO1, SOMO1] - rep_tens[v_orb2, v_orb2, SOMO2, SOMO2] + 0.5 * (rep_tens[v_orb1, SOMO1, SOMO1, v_orb1] + rep_tens[v_orb1, SOMO2, SOMO2, v_orb1] + rep_tens[v_orb2, SOMO1, SOMO1, v_orb2] + rep_tens[v_orb2, SOMO2, SOMO2, v_orb2]) \
                    + 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) + rep_tens[SOMO1,SOMO1,SOMO2,SOMO2] - 0.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1] - rep_tens[v_orb1, v_orb2, v_orb2, v_orb1]
            elif v_orb1 == v_orb3 and v_orb2 != v_orb4:
                xcish[row, col] = rep_tens[v_orb2, v_orb4, v_orb1, v_orb1] - rep_tens[v_orb2, v_orb1, v_orb1, v_orb4] - rep_tens[v_orb2, v_orb4, SOMO1, SOMO1] - rep_tens[v_orb2, v_orb4, SOMO2, SOMO2] + 0.5 * rep_tens[v_orb2, SOMO1, SOMO1, v_orb4] + 0.5 * rep_tens[v_orb2, SOMO2, SOMO2, v_orb4]
            elif v_orb1 != v_orb3 and v_orb2 == v_orb4:
                xcish[row, col] = rep_tens[v_orb1, v_orb3, v_orb2, v_orb2] - rep_tens[v_orb1, v_orb2, v_orb2, v_orb3] - rep_tens[v_orb1, v_orb3, SOMO1, SOMO1] - rep_tens[v_orb1, v_orb3, SOMO2, SOMO2] + 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb3] + 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb3]
            else:
                xcish[row, col] = rep_tens[v_orb1, v_orb3, v_orb2, v_orb4] - rep_tens[v_orb1, v_orb4, v_orb2, v_orb3]
            xcish[col,row] = xcish[row,col]
            v_orb4 += 1
            if v_orb4 >= norbs:
                v_orb3 += 1
                v_orb4 = v_orb3 + 1
        v_orb2 += 1
        if v_orb2 >= norbs:
            v_orb1 += 1
            v_orb2 = v_orb1 + 1
    
            
    ################# QUINTET STATE ##################
    
    row_block_index = nvirt ** 2 + ndocc ** 2 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    #82 <5Q|H|5Q>
    col_block_index = nvirt ** 2 + ndocc ** 2 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + (ndocc * nvirt)):
        o_orb1 = (row - row_block_index) // nvirt # Increase o_orb after every ndocc rows
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1) # Increase v_orb for every column and reset after ndocc rows
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                xcish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb1] - 0.25 * (rep_tens[SOMO1,SOMO1,SOMO1,SOMO1] + rep_tens[SOMO2,SOMO2,SOMO2,SOMO2]) - 0.5 * rep_tens[SOMO1,SOMO2,SOMO2,SOMO1] \
                                 - 0.5 * (rep_tens[o_orb1, SOMO1, SOMO1, o_orb1] + rep_tens[o_orb1, SOMO2, SOMO2, o_orb1] + rep_tens[v_orb1, SOMO1, SOMO1, v_orb1] + rep_tens[v_orb1, SOMO2, SOMO2, v_orb1])
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                xcish[row, col] =  - rep_tens[o_orb1, o_orb2, v_orb1, v_orb1] - 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb2] - 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb2]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                xcish[row, col] = - rep_tens[v_orb1, v_orb2, o_orb1, o_orb1] - 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] - 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb2]
            else:
                xcish[row, col] = - rep_tens[o_orb1, o_orb2, v_orb1, v_orb2]
            xcish[col, row] = xcish[row,col]
    
    
    return xcish






def dipole_xcis(coords,atoms,norbs,hf_orbs,ndocc,nstates):
    '''
    Routine to calculate the one electron dipole moment matrix (x, y and z) in the basis of orbitals, 
    and then the dipole moment matrix in the basis of csfs.
    
    '''
    natoms = coords.shape[0]
    print("Calculating dipole moments ...\n")
    dip1el = cartesian_operators(coords,hf_orbs)[0]
    SOMO1 = ndocc
    SOMO2 = ndocc + 1
    nvirt = norbs - ndocc - 2

   # print("x norm= %f"%linalg.norm(dip1el[:,:,0] - dip1el[:,:,0].T))  # checking symmetric
   # print("y norm= %f"%linalg.norm(dip1el[:,:,1] - dip1el[:,:,1].T))
   # print("z norm= %f"%linalg.norm(dip1el[:,:,2] - dip1el[:,:,2].T))
   # print(" ")
    dipoles = np.zeros((nstates,nstates,3)) 

    #1 <OS1|mu|OS1>
    for o in range(ndocc):
        dipoles[0,0,:] -= 2*dip1el[o,o,:]
    dipoles[0,0,:] -= (dip1el[SOMO1,SOMO1,:] + dip1el[SOMO2,SOMO2,:]) #Adding contribution from SOMOs
    #2 <OS1|mu|ZW-> 
    dipoles[0,1,:] = 0
    dipoles[1,0,:] = dipoles[0,1,:] 
    #3 <OS1|mu|ZW+> 
    dipoles[0,2,:] = - 2 * dip1el[SOMO1,SOMO2,:]
    dipoles[2,0,:] = dipoles[0,2,:] 
    #4 <OS1|mu|HS1> 
    block_index = 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        dipoles[0,col,:] = dip1el[o_orb, SOMO1, :]
        dipoles[col,0,:] = dipoles[0,col,:]
    #5 <OS1|mu|HS2> 
    block_index = ndocc + 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        dipoles[0,col,:] = - dip1el[o_orb, SOMO2, :]
        dipoles[col,0,:] = dipoles[0,col,:]
    #6 <OS1|mu|SL1> 
    block_index = 2 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        dipoles[0,col,:] = -dip1el[v_orb, SOMO1, :]
        dipoles[col,0,:] = dipoles[0,col,:]
    #7 <OS1|mu|SL2> 
    block_index = nvirt + 2 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        dipoles[0,col,:] = dip1el[v_orb, SOMO2, :]
        dipoles[col,0,:] = dipoles[0,col,:]
    #8 <OS1|mu|HL1>
    block_index = 2 * nvirt + 2 * ndocc + 3
    for col in range(block_index, block_index + (ndocc * nvirt)):
        o_orb = (col - block_index) // nvirt # Increase o_orb after every ndocc cols
        v_orb = (col - block_index) % nvirt + (SOMO2 + 1) # Increase v_orb then reset after nvirt cols
        dipoles[0,col,:] = (-2 ** 0.5) * dip1el[o_orb, v_orb, :]
        dipoles[col,0,:] = dipoles[0,col,:]
    #9 <OS1|mu|HL2>=0
    
    
    #10 <ZW-|mu|ZW->
    for o in range(ndocc):
        dipoles[1,1,:] -= 2*dip1el[o,o,:]
    dipoles[1,1,:] -= (dip1el[SOMO1,SOMO1,:] + dip1el[SOMO2,SOMO2,:]) #Adding contribution from SOMOs
    #11 <ZW-|mu|ZW+>
    dipoles[1,2,:] = - (dip1el[SOMO1,SOMO1,:] - dip1el[SOMO2,SOMO2,:])
    dipoles[2,1,:] = dipoles[1,2,:]
    #12 <ZW-|mu|HS1>
    block_index = 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        dipoles[1,col,:] = - dip1el[o_orb, SOMO2, :]
        dipoles[col,1,:] = dipoles[1,col,:]
    #13 <ZW-|mu|HS2>
    block_index = ndocc + 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        dipoles[1,col,:] = - dip1el[o_orb, SOMO1, :]
        dipoles[col,1,:] = dipoles[1,col,:]
    #14 <ZW-|mu|SL1> 
    block_index = 2 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        dipoles[1,col,:] = dip1el[v_orb, SOMO1, :]
        dipoles[col,1,:] = dipoles[1,col,:]
    #15 <ZW-|mu|SL2>
    block_index = 3 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        dipoles[1,col,:] = dip1el[v_orb, SOMO2, :]
        dipoles[col,1,:] = dipoles[1,col,:]
    #16 <ZW-|mu|HL1> = 0
    #17 <ZW-|mu|HL2> = 0
    
    
    #18 <ZW+|mu|ZW+>
    for o in range(ndocc):
        dipoles[2,2,:] -= 2*dip1el[o,o,:]
    dipoles[2,2,:] -= (dip1el[SOMO1,SOMO1,:] + dip1el[SOMO2,SOMO2,:]) #Adding contribution from SOMOs
    #19 <ZW+|mu|HS1>
    block_index = 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        dipoles[2,col,:] = - dip1el[o_orb, SOMO2, :]
        dipoles[col,2,:] = dipoles[2,col,:]
    #20 <ZW+|mu|HS2>
    block_index = ndocc + 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        dipoles[2,col,:] = dip1el[o_orb, SOMO1, :]
        dipoles[col,2,:] = dipoles[2,col,:]
    #21 <ZW+|mu|SL1>
    block_index = 2 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        dipoles[2,col,:] = dip1el[v_orb, SOMO1, :]
        dipoles[col,2,:] = dipoles[2,col,:]
    #22 <ZW+|mu|SL2>
    block_index = nvirt + 2 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        dipoles[2,col,:] = - dip1el[v_orb, SOMO2, :]
        dipoles[col,2,:] = dipoles[2,col,:]
    #23 <ZW+|mu|HL1> = 0
    #24 <ZW+|mu|HL2> = 0
    
    
    row_block_index = 3
    #25 <HS1|mu|HS1>
    col_block_index = 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(row, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                for o in range(ndocc + 1): #Include contribution from 2e in SOMO1
                    dipoles[row,col,:] -= 2 * dip1el[o, o, :]
                dipoles[row,col,:] += dip1el[o_orb1, o_orb1, :] # Remove contribution from 1e in HOMO i
                dipoles[row,col,:] -= dip1el[SOMO2, SOMO2, :] # Add contribution from 1e in SOMO2
            else:    
                dipoles[row,col,:] = dip1el[o_orb1, o_orb2, :] 
            dipoles[col,row,:] = dipoles[row,col,:] 
    #26 <HS1|mu|HS2>
    col_block_index = ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                dipoles[row,col,:] = -dip1el[SOMO1, SOMO2, :] #Only diagonal elements are non-zero
                dipoles[col,row,:] = dipoles[row,col, :]
    #27 <HS1|mu|SL1> = 0
    #28 <HS1|mu|SL2> = 0
    #29 <HS1|mu|HL1>
    col_block_index = 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                dipoles[row, col, :] = (1 / np.sqrt(2)) * dip1el[SOMO1, v_orb, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #30 <HS1|mu|HL2>
    col_block_index = (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                dipoles[row, col, :] = - np.sqrt(1.5) * dip1el[SOMO1, v_orb, :]
                dipoles[col,row,:] = dipoles[row,col,:]


    row_block_index = ndocc + 3
    #31 <HS2|mu|HS2>
    col_block_index = ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(row, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                for o in range(ndocc):
                    dipoles[row,col,:] -= 2 * dip1el[o, o, :]
                dipoles[row,col,:] += dip1el[o_orb1, o_orb1, :] # Remove contribution from 1e in HOMO i
                dipoles[row,col,:] -= 2 * dip1el[SOMO2, SOMO2, :] # Add contribution from 2e in SOMO2
                dipoles[row,col,:] -= dip1el[SOMO1, SOMO1, :] # Add contribution from 1e in SOMO1
            else:    
                dipoles[row,col,:] = dip1el[o_orb1, o_orb2, :]
            dipoles[col,row,:] = dipoles[row,col,:]
    #32 <HS2|mu|SL1> = 0
    #33 <HS2|mu|SL2> = 0
    #34 <HS2|mu|HL1>
    col_block_index = 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                dipoles[row, col, :] = - (1 / np.sqrt(2)) * dip1el[SOMO2, v_orb, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #35 <HS2|mu|HL2>
    col_block_index = (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                dipoles[row, col, :] = - np.sqrt(1.5) * dip1el[SOMO2, v_orb, :]
                dipoles[col,row,:] = dipoles[row,col,:]

    
    row_block_index = 2 * ndocc + 3
    #36 <SL1|mu|SL1>
    col_block_index = 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(row, col_block_index + nvirt):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                for o in range(ndocc):
                    dipoles[row,col,:] -= 2 * dip1el[o, o, :]
                dipoles[row,col,:] -= dip1el[v_orb1, v_orb1, :] # Add contribution from 1e in LUMO j
                dipoles[row,col,:] -= dip1el[SOMO2, SOMO2, :] # Add contribution from 1e in SOMO2
            else:    
                dipoles[row,col,:] = -dip1el[v_orb1, v_orb2, :]
            dipoles[col,row,:] = dipoles[row,col,:]
    #37 <SL1|mu|SL2>
    col_block_index = nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                dipoles[row,col,:] = dip1el[SOMO2,SOMO1,:]
                dipoles[col,row,:] = dipoles[row,col,:]
    #38 <SL1|mu|HL1>
    col_block_index = 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                dipoles[row, col, :] = (1 / np.sqrt(2)) * dip1el[o_orb, SOMO1, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #39 <SL1|mu|HL2>
    col_block_index = (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                dipoles[row, col, :] = np.sqrt(1.5) * dip1el[o_orb, SOMO1, :]
                dipoles[col,row,:] = dipoles[row,col,:]


    row_block_index = nvirt + 2 * ndocc + 3
    #40 <SL2|mu|SL2>
    col_block_index = nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(row, col_block_index + nvirt):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                for o in range(ndocc):
                    dipoles[row,col,:] -= 2 * dip1el[o, o, :]
                dipoles[row,col,:] -= dip1el[v_orb1, v_orb1, :] # Add contribution from 1e in LUMO j
                dipoles[row,col,:] -= dip1el[SOMO1, SOMO1, :] # Add contribution from 1e in SOMO1
            else:    
                dipoles[row,col,:] = -dip1el[v_orb1, v_orb2, :]
            dipoles[col,row,:] = dipoles[row,col,:]
    #41 <SL2|mu|HL1>
    col_block_index = 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                dipoles[row, col, :] = - (1 / np.sqrt(2)) * dip1el[o_orb, SOMO2, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #42 <SL2|mu|HL2>
    col_block_index = (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                dipoles[row, col, :] = np.sqrt(1.5) * dip1el[o_orb, SOMO2, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    
    
    row_block_index = 2 * nvirt + 2 * ndocc + 3
    #43 <HL1|mu|HL1>
    col_block_index = 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + (ndocc * nvirt)):
        o_orb1 = (row - row_block_index) // nvirt
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                for o in range(ndocc):
                    dipoles[row,col,:] -= 2 * dip1el[o,o,:]
                dipoles[row,col,:] += dip1el[o_orb1, o_orb1, :] # Remove contribution from 1e in HOMO i
                dipoles[row,col,:] -= dip1el[SOMO1, SOMO1, :] # Add contribution from 1e in SOMO1
                dipoles[row,col,:] -= dip1el[SOMO2, SOMO2, :] # Add contribution from 1e in SOMO2
                dipoles[row,col,:] -= dip1el[v_orb1, v_orb1, :] # Add contribution from 1e in LUMO j
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                dipoles[row, col, :] = dip1el[o_orb1, o_orb2, :]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                dipoles[row, col, :] = -dip1el[v_orb1, v_orb2, :]
            dipoles[col,row,:] = dipoles[row,col,:]
    #44 <HL1|mu|HL2> = 0


    row_block_index = (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    #43 <HL2|mu|HL2>
    col_block_index = (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + (ndocc * nvirt)):
        o_orb1 = (row - row_block_index) // nvirt
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                for o in range(ndocc):
                    dipoles[row,col,:] -= 2 * dip1el[o,o,:]
                dipoles[row,col,:] += dip1el[o_orb1, o_orb1, :] # Remove contribution from 1e in HOMO i
                dipoles[row,col,:] -= dip1el[SOMO1, SOMO1, :] # Add contribution from 1e in SOMO1
                dipoles[row,col,:] -= dip1el[SOMO2, SOMO2, :] # Add contribution from 1e in SOMO2
                dipoles[row,col,:] -= dip1el[v_orb1, v_orb1, :] # Add contribution from 1e in LUMO j
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                dipoles[row, col, :] = dip1el[o_orb1, o_orb2, :]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                dipoles[row, col, :] = -dip1el[v_orb1, v_orb2, :]
            dipoles[col,row,:] = dipoles[row,col,:]
    
    
    

################# TRIPLET BLOCK ######################

    row_index = 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    #46 <OS3|mu|OS3>
    for o in range(ndocc):
        dipoles[row_index,row_index,:] -= 2*dip1el[o,o,:]
    dipoles[row_index,row_index,:] -= (dip1el[SOMO1,SOMO1,:] + dip1el[SOMO2,SOMO2,:])
    #47 <OS3|mu|HS1>
    col_index = 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 4
    for col in range(col_index, col_index + ndocc):
        o_orb = col - col_index
        dipoles[row_index,col,:] = -dip1el[o_orb,SOMO1,:]
        dipoles[col,row_index,:] = dipoles[row_index,col,:]
    #48 <OS3|mu|HS2>
    col_index = 2 * (ndocc * nvirt) + 2 * nvirt + 3 * ndocc + 4
    for col in range(col_index, col_index + ndocc):
        o_orb = col - col_index
        dipoles[row_index,col,:] = -dip1el[o_orb,SOMO2,:]
        dipoles[col,row_index,:] = dipoles[row_index,col,:]
    #49 <OS3|mu|SL1>
    col_index = 2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 4
    for col in range(col_index, col_index + nvirt):
        v_orb = col - col_index + (SOMO2 + 1)
        dipoles[row_index,col,:] = -dip1el[v_orb,SOMO1,:]
        dipoles[col,row_index,:] = dipoles[row_index,col,:]
    #50 <OS3|mu|SL2>
    col_index = 2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 4
    for col in range(col_index, col_index + nvirt):
        v_orb = col - col_index + (SOMO2 + 1)
        dipoles[row_index,col,:] = dip1el[v_orb,SOMO2,:]
        dipoles[col,row_index,:] = dipoles[row_index,col,:]
    #51 <OS3|mu|HL1>
    col_index = 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for col in range(col_index, col_index + (ndocc * nvirt)):
        o_orb = (col - col_index) // nvirt # Increase o_orb after every nvirt cols
        v_orb = (col - col_index) % nvirt + (SOMO2 + 1) # Increase v_orb then reset after nvirt cols
        dipoles[row_index,col,:] = (-2 ** 0.5) * dip1el[o_orb, v_orb, :]
        dipoles[col,row_index,:] = dipoles[row_index,col,:]
    #52 <OS3|mu|HL2> = 0
    #53 <OS3|mu|HL3> = 0
    

    row_block_index = 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 4
    #54 <HS1|H|HS1>
    col_block_index = 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(row, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                for o in range(ndocc + 1): #Include contribution from 2e in SOMO1
                    dipoles[row,col,:] -= 2 * dip1el[o, o, :]
                dipoles[row,col,:] += dip1el[o_orb1, o_orb1, :] # Remove contribution from 1e in HOMO i
                dipoles[row,col,:] -= dip1el[SOMO2, SOMO2, :] # Add contribution from 1e in SOMO2
            else:    
                dipoles[row,col,:] = dip1el[o_orb1, o_orb2, :] 
            dipoles[col,row,:] = dipoles[row,col,:] 
    #55 <HS1|mu|HS2>
    col_block_index = 2 * (ndocc * nvirt) + 2 * nvirt + 3 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                dipoles[row,col,:] = -dip1el[SOMO1, SOMO2, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #56 <HS1|mu|SL1> = 0
    #57 <HS1|mu|SL2> = 0
    #58 <HS1|mu|HL1>
    col_block_index = 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                dipoles[row, col, :] = - (1 / np.sqrt(2)) * dip1el[SOMO1, v_orb, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #59 <HS1|mu|HL2>
    col_block_index = 3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                dipoles[row, col, :] = - (1 / np.sqrt(2)) * dip1el[SOMO1, v_orb, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #60 <HS1|mu|HL3>
    col_block_index = 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                dipoles[row, col, :] = - dip1el[SOMO1, v_orb, :]
                dipoles[col,row,:] = dipoles[row,col,:]


    row_block_index = 2 * (ndocc * nvirt) + 2 * nvirt + 3 * ndocc + 4
    #61 <HS2|mu|HS2>
    col_block_index = 2 * (ndocc * nvirt) + 2 * nvirt + 3 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(row, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                for o in range(ndocc):
                    dipoles[row,col,:] -= 2 * dip1el[o, o, :]
                dipoles[row,col,:] += dip1el[o_orb1, o_orb1, :] # Remove contribution from 1e in HOMO i
                dipoles[row,col,:] -= 2 * dip1el[SOMO2, SOMO2, :] # Add contribution from 2e in SOMO2
                dipoles[row,col,:] -= dip1el[SOMO1, SOMO1, :] # Add contribution from 1e in SOMO1
            else:    
                dipoles[row,col,:] = dip1el[o_orb1, o_orb2, :]
            dipoles[col,row,:] = dipoles[row,col,:]
    #62 <HS2|mu|SL1> = 0
    #63 <HS2|mu|SL2> = 0
    #64 <HS2|mu|HL1>
    col_block_index = 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                dipoles[row, col, :] = - (1 / np.sqrt(2)) * dip1el[SOMO2, v_orb, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #65 <HS2|mu|HL2>
    col_block_index = 3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                dipoles[row, col, :] = (1 / np.sqrt(2)) * dip1el[SOMO2, v_orb, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #65 <HS2|mu|HL3>
    col_block_index = 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                dipoles[row, col, :] = - dip1el[SOMO2, v_orb, :]
                dipoles[col,row,:] = dipoles[row,col,:]


    row_block_index = 2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 4
    #66 <SL1|mu|SL1>
    col_block_index = 2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(row, col_block_index + nvirt):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                for o in range(ndocc):
                    dipoles[row,col,:] -= 2 * dip1el[o, o, :]
                dipoles[row,col,:] -= dip1el[v_orb1, v_orb1, :] # Add contribution from 1e in LUMO j
                dipoles[row,col,:] -= dip1el[SOMO2, SOMO2, :] # Add contribution from 1e in SOMO2
            else:    
                dipoles[row,col,:] = -dip1el[v_orb1, v_orb2, :]
            dipoles[col,row,:] = dipoles[row,col,:]
    #67 <SL1|mu|SL2>
    col_block_index = 2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                dipoles[row,col,:] = -dip1el[SOMO1,SOMO2,:]
                dipoles[col,row,:] = dipoles[row,col,:]
    #68 <SL1|mu|HL1>
    col_block_index = 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                dipoles[row, col, :] = (1 / np.sqrt(2)) * dip1el[o_orb, SOMO1, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #69 <SL1|mu|HL2>
    col_block_index = 3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                dipoles[row, col, :] = - (1 / np.sqrt(2)) * dip1el[o_orb, SOMO1, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #70 <SL1|mu|HL3>
    col_block_index = 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                dipoles[row, col, :] = - dip1el[o_orb, SOMO1, :]
                dipoles[col,row,:] = dipoles[row,col,:]
 
    
    row_block_index = 2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 4
    #71 <SL2|H|SL2>
    col_block_index = 2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(row, col_block_index + nvirt):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                for o in range(ndocc):
                    dipoles[row,col,:] -= 2 * dip1el[o, o, :]
                dipoles[row,col,:] -= dip1el[v_orb1, v_orb1, :] # Add contribution from 1e in LUMO j
                dipoles[row,col,:] -= dip1el[SOMO1, SOMO1, :] # Add contribution from 1e in SOMO1
            else:    
                dipoles[row,col,:] = -dip1el[v_orb1, v_orb2, :]
            dipoles[col,row,:] = dipoles[row,col,:]
    #72 <SL2|mu|HL1>
    col_block_index = 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                dipoles[row, col, :] = - (1 / np.sqrt(2)) * dip1el[o_orb, SOMO2, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #73 <SL2|mu|HL2>
    col_block_index = 3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                dipoles[row, col, :] = - (1 / np.sqrt(2)) * dip1el[o_orb, SOMO2, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #74 <SL2|mu|HL3>
    col_block_index = 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                dipoles[row, col, :] = dip1el[o_orb, SOMO2, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    
    
    row_block_index = 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    #75 <HL1|mu|HL1>
    col_block_index = 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + (ndocc * nvirt)):
        o_orb1 = (row - row_block_index) // nvirt
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                for o in range(ndocc):
                    dipoles[row,col,:] -= 2 * dip1el[o,o,:]
                dipoles[row,col,:] += dip1el[o_orb1, o_orb1, :] # Remove contribution from 1e in HOMO i
                dipoles[row,col,:] -= dip1el[SOMO1, SOMO1, :] # Add contribution from 1e in SOMO1
                dipoles[row,col,:] -= dip1el[SOMO2, SOMO2, :] # Add contribution from 1e in SOMO2
                dipoles[row,col,:] -= dip1el[v_orb1, v_orb1, :] # Add contribution from 1e in LUMO j
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                dipoles[row, col, :] = dip1el[o_orb1, o_orb2, :]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                dipoles[row, col, :] = -dip1el[v_orb1, v_orb2, :]
            dipoles[col,row,:] = dipoles[row,col,:]
    #76 <HL1|mu|HL2> = 0
    #77 <HL1|mu|HL3> = 0
    
    
    row_block_index = 3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    #78 <HL2|mu|HL2>
    col_block_index = 3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + (ndocc * nvirt)):
        o_orb1 = (row - row_block_index) // nvirt
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                for o in range(ndocc):
                    dipoles[row,col,:] -= 2 * dip1el[o,o,:]
                dipoles[row,col,:] += dip1el[o_orb1, o_orb1, :] # Remove contribution from 1e in HOMO i
                dipoles[row,col,:] -= dip1el[SOMO1, SOMO1, :] # Add contribution from 1e in SOMO1
                dipoles[row,col,:] -= dip1el[SOMO2, SOMO2, :] # Add contribution from 1e in SOMO2
                dipoles[row,col,:] -= dip1el[v_orb1, v_orb1, :] # Add contribution from 1e in LUMO j
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                dipoles[row, col, :] = dip1el[o_orb1, o_orb2, :]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                dipoles[row, col, :] = -dip1el[v_orb1, v_orb2, :]
            dipoles[col,row,:] = dipoles[row,col,:]
    #79 <HL2|mu|HL3> = 0


    row_block_index = 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    #80 <HL3|mu|HL3>
    col_block_index = 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + (ndocc * nvirt)):
        o_orb1 = (row - row_block_index) // nvirt
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                for o in range(ndocc):
                    dipoles[row,col,:] -= 2 * dip1el[o,o,:]
                dipoles[row,col,:] += dip1el[o_orb1, o_orb1, :] # Remove contribution from 1e in HOMO i
                dipoles[row,col,:] -= dip1el[SOMO1, SOMO1, :] # Add contribution from 1e in SOMO1
                dipoles[row,col,:] -= dip1el[SOMO2, SOMO2, :] # Add contribution from 1e in SOMO2
                dipoles[row,col,:] -= dip1el[v_orb1, v_orb1, :] # Add contribution from 1e in LUMO j
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                dipoles[row, col, :] = dip1el[o_orb1, o_orb2, :]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                dipoles[row, col, :] = -dip1el[v_orb1, v_orb2, :]
            dipoles[col,row,:] = dipoles[row,col,:]
    
    
    row_block_index = 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    #81 <DQ|mu|DQ>
    col_block_index = 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + (ndocc * nvirt)):
        o_orb1 = (row - row_block_index) // nvirt
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                for o in range(ndocc):
                    dipoles[row,col,:] -= 2 * dip1el[o,o,:]
                dipoles[row,col,:] += dip1el[o_orb1, o_orb1, :] # Remove contribution from 1e in HOMO i
                dipoles[row,col,:] -= dip1el[SOMO1, SOMO1, :] # Add contribution from 1e in SOMO1
                dipoles[row,col,:] -= dip1el[SOMO2, SOMO2, :] # Add contribution from 1e in SOMO2
                dipoles[row,col,:] -= dip1el[v_orb1, v_orb1, :] # Add contribution from 1e in LUMO j
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                dipoles[row, col, :] = dip1el[o_orb1, o_orb2, :]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                dipoles[row, col, :] = -dip1el[v_orb1, v_orb2, :]
            dipoles[col,row,:] = dipoles[row,col,:]
        
    #print("%10.5f"%linalg.norm(dipoles[:,:,0] - dipoles[:,:,0].T))  # checking symmetric
    #print("%10.5f"%linalg.norm(dipoles[:,:,1] - dipoles[:,:,1].T))
    #print("%10.5f"%linalg.norm(dipoles[:,:,2] - dipoles[:,:,2].T))   

    OS1_perm_dip=dipoles[0,0,:]
    ZWminus_perm_dip=dipoles[1,1,:]
    ZWplus_perm_dip=dipoles[2,2,:]
    OS3_perm_dip=dipoles[2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3,2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3,:]
    for dipole in [OS1_perm_dip, ZWminus_perm_dip, ZWplus_perm_dip, OS3_perm_dip]:
        for i in range(natoms):
            atom_z=0
            if atoms[i][0] in ['C','c','n1','N1']:
                atom_z=1
            elif atoms[i][0] in ['Cl','cl','CL','N2','n2']:
                atom_z=2   
            # for x in range(3):
            #     perm_dip[x]+=atom_z*coords[i,x]*tobohr
            dipole[:]+=atom_z*coords[i,:]*tobohr
    print(f"Permanent dipole moment of reference states:\n \
            Open-Shell Singlet (|OS1>) = {OS1_perm_dip[0]:.3f} x {OS1_perm_dip[1]:.3f} y {OS1_perm_dip[2]:.3f} z\n \
            Zwitterion - (|ZW->) = {ZWminus_perm_dip[0]:.3f} x {ZWminus_perm_dip[1]:.3f} y {ZWminus_perm_dip[2]:.3f} z\n \
            Zwitterion + (|ZW+>) = {ZWplus_perm_dip[0]:.3f} x {ZWplus_perm_dip[1]:.3f} y {ZWplus_perm_dip[2]:.3f} z\n \
            Open-Shell Triplet (|OS3>) = {OS3_perm_dip[0]:.3f} x {OS3_perm_dip[1]:.3f} y {OS3_perm_dip[2]:.3f} z\n")
    return dipoles

def dipole_xcisd(coords,atoms,norbs,hf_orbs,ndocc,nstates):
    '''
    Routine to calculate the one electron dipole moment matrix (x, y and z) in the basis of orbitals, 
    and then the dipole moment matrix in the basis of csfs.
    
    '''
    natoms = coords.shape[0]
    print("Calculating dipole moments ...\n")
    dip1el = cartesian_operators(coords,hf_orbs)[0]
    SOMO1 = ndocc # Index of SOMO1
    SOMO2 = ndocc + 1 # Index of SOMO2
    nvirt = norbs - ndocc - 2 # Number of virtual orbitals
    ndoc3 = int((ndocc ** 2 - ndocc) / 2) # Number of doubly excited occupied to core triplet CSFs
    ndoc1 = int((ndocc ** 2 + ndocc) / 2) # Number of doubly excited occupied to core singlet CSFs
    ndcv3 = int((nvirt ** 2 - nvirt) / 2) # Number of doubly excited occupied to core triplet CSFs
    ndcv1 = int((nvirt ** 2 + nvirt) / 2) # Number of doubly excited occupied to core singlet CSFs
    nstates = nvirt ** 2 + ndocc ** 2 + 6 * (ndocc * nvirt) + 4 * ndocc + 4 * nvirt + 4  # nvirt ** 2 doubles (SOMO to LUMO), ndocc ** 2 doubles (HOMO to SOMO), 6 * ndocc * nvirt doubles (HOMO to LUMO), 4 * ndocc singles (HOMO to SOMO), 4 * nvirt singles (SOMO to LUMO)
                                                                                         # and 4 reference configurations (OS GSs and Zwitterions)

   # print("x norm= %f"%linalg.norm(dip1el[:,:,0] - dip1el[:,:,0].T))  # checking symmetric
   # print("y norm= %f"%linalg.norm(dip1el[:,:,1] - dip1el[:,:,1].T))
   # print("z norm= %f"%linalg.norm(dip1el[:,:,2] - dip1el[:,:,2].T))
   # print(" ")
    dipoles = np.zeros((nstates,nstates,3)) 

    #1 <OS1|mu|OS1>
    for o in range(ndocc):
        dipoles[0,0,:] -= 2*dip1el[o,o,:]
    dipoles[0,0,:] -= (dip1el[SOMO1,SOMO1,:] + dip1el[SOMO2,SOMO2,:]) #Adding contribution from SOMOs
    #2 <OS1|mu|ZW-> 
    dipoles[0,1,:] = 0
    dipoles[1,0,:] = dipoles[0,1,:] 
    #3 <OS1|mu|ZW+> 
    dipoles[0,2,:] = - 2 * dip1el[SOMO1,SOMO2,:]
    dipoles[2,0,:] = dipoles[0,2,:] 
    #4 <OS1|mu|HS1> 
    block_index = 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        dipoles[0,col,:] = dip1el[o_orb, SOMO1, :]
        dipoles[col,0,:] = dipoles[0,col,:]
    #5 <OS1|mu|HS2> 
    block_index = ndocc + 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        dipoles[0,col,:] = - dip1el[o_orb, SOMO2, :]
        dipoles[col,0,:] = dipoles[0,col,:]
    #6 <OS1|mu|SL1> 
    block_index = 2 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        dipoles[0,col,:] = - dip1el[v_orb, SOMO1, :]
        dipoles[col,0,:] = dipoles[0,col,:]
    #7 <OS1|mu|SL2> 
    block_index = nvirt + 2 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        dipoles[0,col,:] = dip1el[v_orb, SOMO2, :]
        dipoles[col,0,:] = dipoles[0,col,:]
    #8 <OS1|mu|HL1>
    block_index = 2 * nvirt + 2 * ndocc + 3
    for col in range(block_index, block_index + (ndocc * nvirt)):
        o_orb = (col - block_index) // nvirt # Increase o_orb after every ndocc cols
        v_orb = (col - block_index) % nvirt + (SOMO2 + 1) # Increase v_orb then reset after nvirt cols
        dipoles[0,col,:] = (-2 ** 0.5) * dip1el[o_orb, v_orb, :]
        dipoles[col,0,:] = dipoles[0,col,:]
    #9 <OS1|mu|HL2>=0
    #9 <OS1|mu|1^HSD>=0
    # <OS1|mu|1^SLD>=0
    
    
    #10 <ZW-|mu|ZW->
    for o in range(ndocc):
        dipoles[1,1,:] -= 2*dip1el[o,o,:]
    dipoles[1,1,:] -= (dip1el[SOMO1,SOMO1,:] + dip1el[SOMO2,SOMO2,:]) #Adding contribution from SOMOs
    #11 <ZW-|mu|ZW+>
    dipoles[1,2,:] = - (dip1el[SOMO1,SOMO1,:] - dip1el[SOMO2,SOMO2,:])
    dipoles[2,1,:] = dipoles[1,2,:]
    #12 <ZW-|mu|HS1>
    block_index = 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        dipoles[1,col,:] = - dip1el[o_orb, SOMO2, :]
        dipoles[col,1,:] = dipoles[1,col,:]
    #13 <ZW-|mu|HS2>
    block_index = ndocc + 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        dipoles[1,col,:] = - dip1el[o_orb, SOMO1, :]
        dipoles[col,1,:] = dipoles[1,col,:]
    #14 <ZW-|mu|SL1> 
    block_index = 2 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        dipoles[1,col,:] = dip1el[v_orb, SOMO1, :]
        dipoles[col,1,:] = dipoles[1,col,:]
    #15 <ZW-|mu|SL2>
    block_index = 3 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        dipoles[1,col,:] = dip1el[v_orb, SOMO2, :]
        dipoles[col,1,:] = dipoles[1,col,:]
    #16 <ZW-|mu|HL1> = 0
    #17 <ZW-|mu|HL2> = 0
    #17 <ZW-|mu|1^HSD>=0
    # <ZW-|mu|1^SLD>=0
    
    
    #18 <ZW+|mu|ZW+>
    for o in range(ndocc):
        dipoles[2,2,:] -= 2*dip1el[o,o,:]
    dipoles[2,2,:] -= (dip1el[SOMO1,SOMO1,:] + dip1el[SOMO2,SOMO2,:]) #Adding contribution from SOMOs
    #19 <ZW+|mu|HS1>
    block_index = 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        dipoles[2,col,:] = - dip1el[o_orb, SOMO2, :]
        dipoles[col,2,:] = dipoles[2,col,:]
    #20 <ZW+|mu|HS2>
    block_index = ndocc + 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        dipoles[2,col,:] = dip1el[o_orb, SOMO1, :]
        dipoles[col,2,:] = dipoles[2,col,:]
    #21 <ZW+|mu|SL1>
    block_index = 2 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        dipoles[2,col,:] = dip1el[v_orb, SOMO1, :]
        dipoles[col,2,:] = dipoles[2,col,:]
    #22 <ZW+|mu|SL2>
    block_index = nvirt + 2 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        dipoles[2,col,:] = - dip1el[v_orb, SOMO2, :]
        dipoles[col,2,:] = dipoles[2,col,:]
    #23 <ZW+|mu|HL1> = 0
    #24 <ZW+|mu|HL2> = 0
    #24 <ZW+|mu|1^HSD>=0
    # <ZW+|mu|1^SLD>=0
    
    
    row_block_index = 3
    #25 <HS1|mu|HS1>
    col_block_index = 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(row, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                for o in range(ndocc + 1): #Include contribution from 2e in SOMO1
                    dipoles[row,col,:] -= 2 * dip1el[o, o, :]
                dipoles[row,col,:] += dip1el[o_orb1, o_orb1, :] # Remove contribution from 1e in HOMO i
                dipoles[row,col,:] -= dip1el[SOMO2, SOMO2, :] # Add contribution from 1e in SOMO2
            else:    
                dipoles[row,col,:] = dip1el[o_orb1, o_orb2, :] 
            dipoles[col,row,:] = dipoles[row,col,:] 
    #26 <HS1|mu|HS2>
    col_block_index = ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc):
             o_orb2 = col - col_block_index
             if o_orb1 == o_orb2:
                 dipoles[row,col,:] = -dip1el[SOMO1, SOMO2, :] #Only diagonal elements are non-zero
                 dipoles[col,row,:] = dipoles[row,col, :]
    #27 <HS1|mu|SL1> = 0
    #28 <HS1|mu|SL2> = 0
    #29 <HS1|mu|HL1>
    col_block_index = 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                dipoles[row, col, :] = (1 / np.sqrt(2)) * dip1el[SOMO1, v_orb, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #30 <HS1|mu|HL2>
    col_block_index = (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                dipoles[row, col, :] = - np.sqrt(1.5) * dip1el[SOMO1, v_orb, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #30 <HS1|mu|1^HSD>
    col_block_index = 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        o_orb2 = 0
        o_orb3 = 0
        for col in range(col_block_index, col_block_index + ndoc1):
            if o_orb1 == o_orb2:
                if o_orb2 == o_orb3:
                    dipoles[row, col, :] = - np.sqrt(2) * dip1el[o_orb1, SOMO2, :]
                else:
                    dipoles[row, col, :] = - dip1el[o_orb3, SOMO2, :]
            elif o_orb1 == o_orb3:
                if o_orb2 == o_orb3:
                    dipoles[row, col, :] = - np.sqrt(2) * dip1el[o_orb1, SOMO2, :]
                else:
                    dipoles[row, col, :] = - dip1el[o_orb2, SOMO2, :]
            dipoles[col,row,:] = dipoles[row,col,:]
            o_orb3 += 1
            if o_orb3 == ndocc:
                o_orb2 += 1
                o_orb3 = o_orb2
    # <HS1|mu|1^SLD> = 0

    row_block_index = ndocc + 3
    #31 <HS2|mu|HS2>
    col_block_index = ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(row, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                for o in range(ndocc):
                    dipoles[row,col,:] -= 2 * dip1el[o, o, :]
                dipoles[row,col,:] += dip1el[o_orb1, o_orb1, :] # Remove contribution from 1e in HOMO i
                dipoles[row,col,:] -= 2 * dip1el[SOMO2, SOMO2, :] # Add contribution from 2e in SOMO2
                dipoles[row,col,:] -= dip1el[SOMO1, SOMO1, :] # Add contribution from 1e in SOMO1
            else:    
                dipoles[row,col,:] = dip1el[o_orb1, o_orb2, :]
            dipoles[col,row,:] = dipoles[row,col,:]
    #32 <HS2|mu|SL1> = 0
    #33 <HS2|mu|SL2> = 0
    #34 <HS2|mu|HL1>
    col_block_index = 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                dipoles[row, col, :] = - (1 / np.sqrt(2)) * dip1el[SOMO2, v_orb, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #35 <HS2|mu|HL2>
    col_block_index = (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                dipoles[row, col, :] = - np.sqrt(1.5) * dip1el[SOMO2, v_orb, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #35 <HS2|mu|1^HSD>
    col_block_index = 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        o_orb2 = 0
        o_orb3 = 0
        for col in range(col_block_index, col_block_index + ndoc1):
            if o_orb1 == o_orb2:
                if o_orb2 == o_orb3:
                    dipoles[row, col, :] = np.sqrt(2) * dip1el[o_orb1, SOMO1, :]
                else:
                    dipoles[row, col, :] = dip1el[o_orb3, SOMO1, :]
            elif o_orb1 == o_orb3:
                if o_orb2 == o_orb3:
                    dipoles[row, col, :] = np.sqrt(2) * dip1el[o_orb1, SOMO1, :]
                else:
                    dipoles[row, col, :] = dip1el[o_orb2, SOMO1, :]
            dipoles[col,row,:] = dipoles[row,col,:]
            o_orb3 += 1
            if o_orb3 == ndocc:
                o_orb2 += 1
                o_orb3 = o_orb2
    # <HS2|mu|1^SLD> = 0


    row_block_index = 2 * ndocc + 3
    #36 <SL1|mu|SL1>
    col_block_index = 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(row, col_block_index + nvirt):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                for o in range(ndocc):
                    dipoles[row,col,:] -= 2 * dip1el[o, o, :]
                dipoles[row,col,:] -= dip1el[v_orb1, v_orb1, :] # Add contribution from 1e in LUMO j
                dipoles[row,col,:] -= dip1el[SOMO2, SOMO2, :] # Add contribution from 1e in SOMO2
            else:    
                dipoles[row,col,:] = -dip1el[v_orb1, v_orb2, :]
            dipoles[col,row,:] = dipoles[row,col,:]
    #37 <SL1|mu|SL2>
    col_block_index = nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                dipoles[row,col,:] = dip1el[SOMO1,SOMO2,:]                 
                dipoles[col,row,:] = dipoles[row,col,:]
    #38 <SL1|mu|HL1>
    col_block_index = 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                dipoles[row, col, :] = (1 / np.sqrt(2)) * dip1el[o_orb, SOMO1, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #39 <SL1|mu|HL2>
    col_block_index = (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                dipoles[row, col, :] = np.sqrt(1.5) * dip1el[o_orb, SOMO1, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #39 <SL1|mu|HSD> = 0
    # <SL1|mu|1^SLD> 
    col_block_index = ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        v_orb2 = SOMO2 + 1
        v_orb3 = SOMO2 + 1
        for col in range(col_block_index, col_block_index + ndcv1):
            if v_orb2 == v_orb3:
                if v_orb1 == v_orb2:
                    dipoles[row,col,:] = - np.sqrt(2) * dip1el[v_orb1,SOMO2,:]                 
            else:
                if v_orb1 == v_orb2:
                    dipoles[row,col,:] = - dip1el[v_orb3,SOMO2,:] 
                elif v_orb1 == v_orb3:
                    dipoles[row,col,:] = - dip1el[v_orb2,SOMO2,:]
            dipoles[col,row,:] = dipoles[row,col,:]


    row_block_index = nvirt + 2 * ndocc + 3
    #40 <SL2|mu|SL2>
    col_block_index = nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(row, col_block_index + nvirt):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                for o in range(ndocc):
                    dipoles[row,col,:] -= 2 * dip1el[o, o, :]
                dipoles[row,col,:] -= dip1el[v_orb1, v_orb1, :] # Add contribution from 1e in LUMO j
                dipoles[row,col,:] -= dip1el[SOMO1, SOMO1, :] # Add contribution from 1e in SOMO1
            else:    
                dipoles[row,col,:] = -dip1el[v_orb1, v_orb2, :]
            dipoles[col,row,:] = dipoles[row,col,:]
    #41 <SL2|mu|HL1>
    col_block_index = 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                dipoles[row, col, :] = - (1 / np.sqrt(2)) * dip1el[o_orb, SOMO2, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #42 <SL2|mu|HL2>
    col_block_index = (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                dipoles[row, col, :] = np.sqrt(1.5) * dip1el[o_orb, SOMO2, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #42 <SL2|mu|HSD> = 0
    # <SL2|mu|SLD> 
    col_block_index = ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        v_orb2 = SOMO2 + 1
        v_orb3 = SOMO2 + 1
        for col in range(col_block_index, col_block_index + ndcv1):
            if v_orb2 == v_orb3:
                if v_orb1 == v_orb2:
                    dipoles[row,col,:] = np.sqrt(2) * dip1el[v_orb1,SOMO1,:]                 
            else:
                if v_orb1 == v_orb2:
                    dipoles[row,col,:] = dip1el[v_orb3,SOMO1,:] 
                elif v_orb1 == v_orb3:
                    dipoles[row,col,:] = dip1el[v_orb2,SOMO1,:]
            dipoles[col,row,:] = dipoles[row,col,:]
    
    
    row_block_index = 2 * nvirt + 2 * ndocc + 3
    #43 <HL1|mu|HL1>
    col_block_index = 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + (ndocc * nvirt)):
        o_orb1 = (row - row_block_index) // nvirt
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                for o in range(ndocc):
                    dipoles[row,col,:] -= 2 * dip1el[o,o,:]
                dipoles[row,col,:] += dip1el[o_orb1, o_orb1, :] # Remove contribution from 1e in HOMO i
                dipoles[row,col,:] -= dip1el[SOMO1, SOMO1, :] # Add contribution from 1e in SOMO1
                dipoles[row,col,:] -= dip1el[SOMO2, SOMO2, :] # Add contribution from 1e in SOMO2
                dipoles[row,col,:] -= dip1el[v_orb1, v_orb1, :] # Add contribution from 1e in LUMO j
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                dipoles[row, col, :] = dip1el[o_orb1, o_orb2, :]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                dipoles[row, col, :] = -dip1el[v_orb1, v_orb2, :]
            dipoles[col,row,:] = dipoles[row,col,:]
    #44 <HL1|mu|HL2> = 0
    #45 <HL1|mu|HSD> = 0
    # <HL1|mu|SLD> = 0


    row_block_index = (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    #43 <HL2|mu|HL2>
    col_block_index = (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + (ndocc * nvirt)):
        o_orb1 = (row - row_block_index) // nvirt
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                for o in range(ndocc):
                    dipoles[row,col,:] -= 2 * dip1el[o,o,:]
                dipoles[row,col,:] += dip1el[o_orb1, o_orb1, :] # Remove contribution from 1e in HOMO i
                dipoles[row,col,:] -= dip1el[SOMO1, SOMO1, :] # Add contribution from 1e in SOMO1
                dipoles[row,col,:] -= dip1el[SOMO2, SOMO2, :] # Add contribution from 1e in SOMO2
                dipoles[row,col,:] -= dip1el[v_orb1, v_orb1, :] # Add contribution from 1e in LUMO j
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                dipoles[row, col, :] = dip1el[o_orb1, o_orb2, :]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                dipoles[row, col, :] = -dip1el[v_orb1, v_orb2, :]
            dipoles[col,row,:] = dipoles[row,col,:]
    # <HL1|mu|HLD> = 0
    # <HL1|mu|SLD> = 0
    
    row_block_index = 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    #44 <1^HSD|mu|1^HSD>
    col_block_index = 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    o_orb1 = 0
    o_orb2 = 0
    for row in range(row_block_index, row_block_index + ndoc1):
        o_orb3 = 0
        o_orb4 = 0
        for col in range(col_block_index, col_block_index + ndoc1):
            if o_orb1 == o_orb2 and o_orb3 == o_orb4:
                if o_orb1 == o_orb3:
                    for o in range(ndocc):
                        dipoles[row, col, :] -= 2 * dip1el[o, o, :]
                    dipoles[row, col, :] += 2 * dip1el[o_orb1, o_orb1, :]
                    dipoles[row, col, :] -= 2 * dip1el[SOMO1, SOMO1, :]
                    dipoles[row, col, :] -= 2 * dip1el[SOMO2, SOMO2, :]
            elif o_orb1 == o_orb2 and o_orb3 != o_orb4:
                if o_orb1 == o_orb3:
                    dipoles[row, col, :] = np.sqrt(2) * dip1el[o_orb1, o_orb4, :]
                elif o_orb1 == o_orb4:
                    dipoles[row, col, :] = np.sqrt(2) * dip1el[o_orb1, o_orb3, :]
            elif o_orb3 == o_orb4 and o_orb1 != o_orb2:
                if o_orb3 == o_orb1:
                    dipoles[row, col, :] = np.sqrt(2) * dip1el[o_orb3, o_orb2, :]
                elif o_orb3 == o_orb2:
                    dipoles[row, col, :] = np.sqrt(2) * dip1el[o_orb3, o_orb1, :]
            else:
                if o_orb1 == o_orb3 and o_orb2 == o_orb4:
                    for o in range(ndocc):
                        dipoles[row, col, :] -= 2 * dip1el[o, o, :]
                        dipoles[row, col, :] += dip1el[o_orb1, o_orb1, :]
                        dipoles[row, col, :] += dip1el[o_orb2, o_orb2, :]
                        dipoles[row, col, :] -= 2 * dip1el[SOMO1, SOMO1, :]
                        dipoles[row, col, :] -= 2 * dip1el[SOMO2, SOMO2, :]
                elif o_orb1 == o_orb3 and o_orb2 != o_orb4:
                    dipoles[row,col,:] = dip1el[o_orb2,o_orb4,:]
                elif o_orb2 == o_orb4 and o_orb1 != o_orb3:
                    dipoles[row,col,:] = dip1el[o_orb1,o_orb3,:]
            dipoles[col,row,:] = dipoles[row,col,:]
            o_orb4 += 1
            if o_orb4 == ndocc:
                o_orb3 += 1
                o_orb4 = o_orb3
        o_orb2 += 1
        if o_orb2 == ndocc:
            o_orb1 += 1
            o_orb2 = o_orb1
    # <HSD|mu|SLD> = 0
    
    # <SLD|mu|SLD>
    row_block_index = ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    col_block_index = ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    v_orb1 = SOMO2 + 1
    v_orb2 = SOMO2 + 1
    for row in range(row_block_index, row_block_index + ndcv1):
        v_orb3 = SOMO2 + 1
        v_orb4 = SOMO2 + 1
        for col in range(col_block_index, col_block_index + ndcv1):
            if v_orb1 == v_orb2 and v_orb3 == v_orb4:
                if v_orb1 == v_orb3:
                    for o in range(ndocc):
                        dipoles[row, col, :] -= 2 * dip1el[o, o, :]
                    dipoles[row,col,:] -= 2 * dip1el[v_orb1,v_orb1,:]                 
            elif v_orb1 == v_orb2 and v_orb3 != v_orb4:
                if v_orb1 == v_orb3:
                    dipoles[row, col, :] = - np.sqrt(2) * dip1el[v_orb1, v_orb4, :]
                elif v_orb1 == v_orb4:
                    dipoles[row, col, :] = - np.sqrt(2) * dip1el[v_orb1, v_orb3, :]
            elif v_orb1 != v_orb2 and v_orb3 == v_orb4:
                if v_orb3 == v_orb1:
                    dipoles[row, col, :] = - np.sqrt(2) * dip1el[v_orb3, v_orb2, :]
                elif v_orb3 == v_orb2:
                    dipoles[row, col, :] = - np.sqrt(2) * dip1el[v_orb3, v_orb1, :]
            else:
                if v_orb1 == v_orb3 and v_orb2 == v_orb4:
                    for o in range(ndocc):
                        dipoles[row, col, :] -= 2 * dip1el[o, o, :]
                    dipoles[row,col,:] -= dip1el[v_orb1,v_orb1,:] 
                    dipoles[row,col,:] -= dip1el[v_orb2,v_orb2,:] 
                elif v_orb1 == v_orb3 and v_orb2 != v_orb4:
                    dipoles[row,col,:] = -dip1el[v_orb2,v_orb4,:]
                elif v_orb1 != v_orb3 and v_orb2 == v_orb4:
                    dipoles[row,col,:] = -dip1el[v_orb1,v_orb3,:]
            
            dipoles[col,row,:] = dipoles[row,col,:]
    

################# TRIPLET BLOCK ######################

    row_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    #46 <OS3|mu|OS3>
    for o in range(ndocc):
        dipoles[row_index,row_index,:] -= 2*dip1el[o,o,:]
    dipoles[row_index,row_index,:] -= (dip1el[SOMO1,SOMO1,:] + dip1el[SOMO2,SOMO2,:])
    #47 <OS3|mu|HS1>
    col_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 4
    for col in range(col_index, col_index + ndocc):
        o_orb = col - col_index
        dipoles[row_index,col,:] = -dip1el[o_orb,SOMO1,:]
        dipoles[col,row_index,:] = dipoles[row_index,col,:]
    #48 <OS3|mu|HS2>
    col_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 3 * ndocc + 4
    for col in range(col_index, col_index + ndocc):
        o_orb = col - col_index
        dipoles[row_index,col,:] = -dip1el[o_orb,SOMO2,:]
        dipoles[col,row_index,:] = dipoles[row_index,col,:]
    #49 <OS3|mu|SL1>
    col_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 4
    for col in range(col_index, col_index + nvirt):
        v_orb = col - col_index + (SOMO2 + 1)
        dipoles[row_index,col,:] = -dip1el[v_orb,SOMO1,:]
        dipoles[col,row_index,:] = dipoles[row_index,col,:]
    #50 <OS3|mu|SL2>
    col_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 4
    for col in range(col_index, col_index + nvirt):
        v_orb = col - col_index + (SOMO2 + 1)
        dipoles[row_index,col,:] = dip1el[v_orb,SOMO2,:]
        dipoles[col,row_index,:] = dipoles[row_index,col,:]
    #51 <OS3|mu|HL1>
    col_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for col in range(col_index, col_index + (ndocc * nvirt)):
        o_orb = (col - col_index) // nvirt # Increase o_orb after every nvirt cols
        v_orb = (col - col_index) % nvirt + (SOMO2 + 1) # Increase v_orb then reset after nvirt cols
        dipoles[row_index,col,:] = (-2 ** 0.5) * dip1el[o_orb, v_orb, :]
        dipoles[col,row_index,:] = dipoles[row_index,col,:]
    #52 <OS3|mu|HL2> = 0
    #53 <OS3|mu|HL3> = 0
    #54 <OS3|mu|3^HSD> = 0
    # <OS3|mu|SLD> = 0
    

    row_block_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 4
    #54 <HS1|H|HS1>
    col_block_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(row, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                for o in range(ndocc + 1): #Include contribution from 2e in SOMO1
                    dipoles[row,col,:] -= 2 * dip1el[o, o, :]
                dipoles[row,col,:] += dip1el[o_orb1, o_orb1, :] # Remove contribution from 1e in HOMO i
                dipoles[row,col,:] -= dip1el[SOMO2, SOMO2, :] # Add contribution from 1e in SOMO2
            else:    
                dipoles[row,col,:] = dip1el[o_orb1, o_orb2, :] 
            dipoles[col,row,:] = dipoles[row,col,:] 
    #55 <HS1|mu|HS2>
    col_block_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 3 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                dipoles[row,col,:] = -dip1el[SOMO1, SOMO2, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #56 <HS1|mu|SL1> = 0
    #57 <HS1|mu|SL2> = 0
    #58 <HS1|mu|HL1>
    col_block_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                dipoles[row, col, :] = - (1 / np.sqrt(2)) * dip1el[SOMO1, v_orb, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #59 <HS1|mu|HL2>
    col_block_index = ndcv1 + ndoc1 + 3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                dipoles[row, col, :] = - (1 / np.sqrt(2)) * dip1el[SOMO1, v_orb, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #60 <HS1|mu|HL3>
    col_block_index = ndcv1 + ndoc1 + 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                dipoles[row, col, :] = - dip1el[SOMO1, v_orb, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #60 <HS1|mu|3^HSD>
    col_block_index = ndcv1 + ndoc1 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        o_orb2 = 0
        o_orb3 = 1
        for col in range(col_block_index, col_block_index + ndoc3):
            if o_orb1 == o_orb2:
                dipoles[row, col, :] = - dip1el[o_orb3, SOMO2, :]
            elif o_orb1 == o_orb3:
                dipoles[row, col, :] = dip1el[o_orb2, SOMO2, :]
            dipoles[col,row,:] = dipoles[row,col,:]
            o_orb3 += 1
            if o_orb3 == ndocc:
                o_orb2 += 1
                o_orb3 = o_orb2 + 1
    # <HS1|mu|SLD> = 0

    row_block_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 3 * ndocc + 4
    #61 <HS2|mu|HS2>
    col_block_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 3 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(row, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                for o in range(ndocc):
                    dipoles[row,col,:] -= 2 * dip1el[o, o, :]
                dipoles[row,col,:] += dip1el[o_orb1, o_orb1, :] # Remove contribution from 1e in HOMO i
                dipoles[row,col,:] -= 2 * dip1el[SOMO2, SOMO2, :] # Add contribution from 2e in SOMO2
                dipoles[row,col,:] -= dip1el[SOMO1, SOMO1, :] # Add contribution from 1e in SOMO1
            else:    
                dipoles[row,col,:] = dip1el[o_orb1, o_orb2, :]
            dipoles[col,row,:] = dipoles[row,col,:]
    #62 <HS2|mu|SL1> = 0
    #63 <HS2|mu|SL2> = 0
    #64 <HS2|mu|HL1>
    col_block_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                dipoles[row, col, :] = - (1 / np.sqrt(2)) * dip1el[SOMO2, v_orb, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #65 <HS2|mu|HL2>
    col_block_index = ndcv1 + ndoc1 + 3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                dipoles[row, col, :] = (1 / np.sqrt(2)) * dip1el[SOMO2, v_orb, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #65 <HS2|mu|HL3>
    col_block_index = ndcv1 + ndoc1 + 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                dipoles[row, col, :] = - dip1el[SOMO2, v_orb, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #65 <HS2|mu|3^HSD>
    col_block_index = ndcv1 + ndoc1 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        o_orb2 = 0
        o_orb3 = 1
        for col in range(col_block_index, col_block_index + ndoc3):
            if o_orb1 == o_orb2:
                dipoles[row, col, :] = dip1el[o_orb3, SOMO1, :]
            elif o_orb1 == o_orb3:
                dipoles[row, col, :] = - dip1el[o_orb2, SOMO1, :]
            dipoles[col,row,:] = dipoles[row,col,:]
            o_orb3 += 1
            if o_orb3 == ndocc:
                o_orb2 += 1
                o_orb3 = o_orb2 + 1
    # <HS2|mu|SLD> = 0

    row_block_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 4
    #66 <SL1|mu|SL1>
    col_block_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(row, col_block_index + nvirt):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                for o in range(ndocc):
                    dipoles[row,col,:] -= 2 * dip1el[o, o, :]
                dipoles[row,col,:] -= dip1el[v_orb1, v_orb1, :] # Add contribution from 1e in LUMO j
                dipoles[row,col,:] -= dip1el[SOMO2, SOMO2, :] # Add contribution from 1e in SOMO2
            else:    
                dipoles[row,col,:] = -dip1el[v_orb1, v_orb2, :]
            dipoles[col,row,:] = dipoles[row,col,:]
    #67 <SL1|mu|SL2>
    col_block_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                dipoles[row,col,:] = -dip1el[SOMO1,SOMO2,:]
                dipoles[col,row,:] = dipoles[row,col,:]
    #68 <SL1|mu|HL1>
    col_block_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                dipoles[row, col, :] = (1 / np.sqrt(2)) * dip1el[o_orb, SOMO1, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #69 <SL1|mu|HL2>
    col_block_index = ndcv1 + ndoc1 + 3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                dipoles[row, col, :] = - (1 / np.sqrt(2)) * dip1el[o_orb, SOMO1, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #70 <SL1|mu|HL3>
    col_block_index = ndcv1 + ndoc1 + 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                dipoles[row, col, :] = - dip1el[o_orb, SOMO1, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #70 <SL1|mu|3^HSD> = 0
    # <SL1|mu|SLD>
    col_block_index = ndcv1 + ndocc ** 2 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = (row - row_block_index) + (SOMO2 + 1)
        v_orb2 = SOMO2 + 1
        v_orb3 = SOMO2 + 2
        for col in range(col_block_index, col_block_index + ndcv3):
            if v_orb1 == v_orb2:
                dipoles[row, col, :] = - dip1el[v_orb3, SOMO2, :]
            elif v_orb1 == v_orb3:
                dipoles[row, col, :] = dip1el[v_orb2, SOMO2, :]
            dipoles[col,row,:] = dipoles[row,col,:]
            v_orb3 += 1
            if v_orb3 >= norbs:
                v_orb2 += 1
                v_orb3 = v_orb2 + 1
    
    
    
    row_block_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 4
    #71 <SL2|H|SL2>
    col_block_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(row, col_block_index + nvirt):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                for o in range(ndocc):
                    dipoles[row,col,:] -= 2 * dip1el[o, o, :]
                dipoles[row,col,:] -= dip1el[v_orb1, v_orb1, :] # Add contribution from 1e in LUMO j
                dipoles[row,col,:] -= dip1el[SOMO1, SOMO1, :] # Add contribution from 1e in SOMO1
            else:    
                dipoles[row,col,:] = -dip1el[v_orb1, v_orb2, :]
            dipoles[col,row,:] = dipoles[row,col,:]
    #72 <SL2|mu|HL1>
    col_block_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                dipoles[row, col, :] = - (1 / np.sqrt(2)) * dip1el[o_orb, SOMO2, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #73 <SL2|mu|HL2>
    col_block_index = ndcv1 + ndoc1 + 3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                dipoles[row, col, :] = - (1 / np.sqrt(2)) * dip1el[o_orb, SOMO2, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #74 <SL2|mu|HL3>
    col_block_index = ndcv1 + ndoc1 + 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                dipoles[row, col, :] = dip1el[o_orb, SOMO2, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #74 <SL2|mu|3^HSD> = 0
    # <SL2|mu|SLD>
    col_block_index = ndcv1 + ndocc ** 2 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = (row - row_block_index) + (SOMO2 + 1)
        v_orb2 = SOMO2 + 1
        v_orb3 = SOMO2 + 2
        for col in range(col_block_index, col_block_index + ndcv3):
            if v_orb1 == v_orb2:
                dipoles[row, col, :] = dip1el[v_orb3, SOMO1, :]
            elif v_orb1 == v_orb3:
                dipoles[row, col, :] = - dip1el[v_orb2, SOMO1, :]
            dipoles[col,row,:] = dipoles[row,col,:]
            v_orb3 += 1
            if v_orb3 >= norbs:
                v_orb2 += 1
                v_orb3 = v_orb2 + 1
    
    
    row_block_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    #75 <HL1|mu|HL1>
    col_block_index = ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + (ndocc * nvirt)):
        o_orb1 = (row - row_block_index) // nvirt
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                for o in range(ndocc):
                    dipoles[row,col,:] -= 2 * dip1el[o,o,:]
                dipoles[row,col,:] += dip1el[o_orb1, o_orb1, :] # Remove contribution from 1e in HOMO i
                dipoles[row,col,:] -= dip1el[SOMO1, SOMO1, :] # Add contribution from 1e in SOMO1
                dipoles[row,col,:] -= dip1el[SOMO2, SOMO2, :] # Add contribution from 1e in SOMO2
                dipoles[row,col,:] -= dip1el[v_orb1, v_orb1, :] # Add contribution from 1e in LUMO j
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                dipoles[row, col, :] = dip1el[o_orb1, o_orb2, :]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                dipoles[row, col, :] = -dip1el[v_orb1, v_orb2, :]
            dipoles[col,row,:] = dipoles[row,col,:]
    #76 <HL1|mu|HL2> = 0
    #77 <HL1|mu|HL3> = 0
    #78 <HL1|mu|3^HSD> = 0
    # <HL1|mu|SLD> = 0
    
    
    row_block_index = ndcv1 + ndoc1 + 3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    #78 <HL2|mu|HL2>
    col_block_index = ndcv1 + ndoc1 + 3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + (ndocc * nvirt)):
        o_orb1 = (row - row_block_index) // nvirt
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                for o in range(ndocc):
                    dipoles[row,col,:] -= 2 * dip1el[o,o,:]
                dipoles[row,col,:] += dip1el[o_orb1, o_orb1, :] # Remove contribution from 1e in HOMO i
                dipoles[row,col,:] -= dip1el[SOMO1, SOMO1, :] # Add contribution from 1e in SOMO1
                dipoles[row,col,:] -= dip1el[SOMO2, SOMO2, :] # Add contribution from 1e in SOMO2
                dipoles[row,col,:] -= dip1el[v_orb1, v_orb1, :] # Add contribution from 1e in LUMO j
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                dipoles[row, col, :] = dip1el[o_orb1, o_orb2, :]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                dipoles[row, col, :] = -dip1el[v_orb1, v_orb2, :]
            dipoles[col,row,:] = dipoles[row,col,:]
    #79 <HL2|mu|HL3> = 0
    #80 <HL2|mu|3^HSD> = 0
    # <HL2|mu|SLD> = 0


    row_block_index = ndcv1 + ndoc1 + 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    #80 <HL3|mu|HL3>
    col_block_index = ndcv1 + ndoc1 + 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + (ndocc * nvirt)):
        o_orb1 = (row - row_block_index) // nvirt
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                for o in range(ndocc):
                    dipoles[row,col,:] -= 2 * dip1el[o,o,:]
                dipoles[row,col,:] += dip1el[o_orb1, o_orb1, :] # Remove contribution from 1e in HOMO i
                dipoles[row,col,:] -= dip1el[SOMO1, SOMO1, :] # Add contribution from 1e in SOMO1
                dipoles[row,col,:] -= dip1el[SOMO2, SOMO2, :] # Add contribution from 1e in SOMO2
                dipoles[row,col,:] -= dip1el[v_orb1, v_orb1, :] # Add contribution from 1e in LUMO j
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                dipoles[row, col, :] = dip1el[o_orb1, o_orb2, :]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                dipoles[row, col, :] = -dip1el[v_orb1, v_orb2, :]
            dipoles[col,row,:] = dipoles[row,col,:]
    #80 <HL3|H|3^HSD> = 0
    # <HL3|H|SLD> = 0
    
    
    row_block_index = ndcv1 + ndoc1 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    #81 <3^HSD|mu|3^HSD>
    col_block_index = ndcv1 + ndoc1 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    o_orb1 = 0
    o_orb2 = 1
    for row in range(row_block_index, row_block_index + ndoc3):
        o_orb3 = 0
        o_orb4 = 1
        for col in range(col_block_index, col_block_index + ndoc3):
            if o_orb1 == o_orb3 and o_orb2 == o_orb4:
                for o in range(ndocc):
                    dipoles[row, col, :] -= 2 * dip1el[o, o, :]
                dipoles[row, col, :] += dip1el[o_orb1, o_orb1, :]
                dipoles[row, col, :] += dip1el[o_orb2, o_orb2, :]
                dipoles[row, col, :] -= 2 * dip1el[SOMO1, SOMO1, :]
                dipoles[row, col, :] -= 2 * dip1el[SOMO2, SOMO2, :]
            elif o_orb1 == o_orb3 and o_orb2 != o_orb4:
                dipoles[row,col,:] = - dip1el[o_orb2,o_orb4,:]
            elif o_orb2 == o_orb4 and o_orb1 != o_orb3:
                dipoles[row,col,:] = dip1el[o_orb1,o_orb3,:]
            dipoles[col,row,:] = dipoles[row,col,:]
            o_orb4 += 1
            if o_orb4 == ndocc:
                o_orb3 += 1
                o_orb4 = o_orb3 + 1
        o_orb2 += 1
        if o_orb2 == ndocc:
            o_orb1 += 1
            o_orb2 = o_orb1 + 1
    # <3^HSD|mu|SLD> = 0
    
    # <SLD|mu|SLD>
    row_block_index = ndcv1 + ndocc ** 2 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    #81 <3^HSD|mu|3^HSD>
    col_block_index = ndcv1 + ndocc ** 2 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    v_orb1 = SOMO2 + 1
    v_orb2 = SOMO2 + 2
    for row in range(row_block_index, row_block_index + ndcv3):
        v_orb3 = SOMO2 + 1
        v_orb4 = SOMO2 + 2
        for col in range(col_block_index, col_block_index + ndcv3):
            if v_orb1 == v_orb3 and v_orb2 == v_orb4:
                for o in range(ndocc):
                    dipoles[row, col, :] -= 2 * dip1el[o, o, :]
                dipoles[row, col, :] -= dip1el[v_orb1, v_orb1, :]
                dipoles[row, col, :] -= dip1el[v_orb2, v_orb2, :]
            elif v_orb1 == v_orb3 and v_orb2 != v_orb4:
                dipoles[row,col,:] = - dip1el[v_orb2,v_orb4,:]
            elif v_orb2 == v_orb4 and v_orb1 != v_orb3:
                dipoles[row,col,:] = dip1el[v_orb1,v_orb3,:]
            dipoles[col,row,:] = dipoles[row,col,:]
            v_orb4 += 1
            if v_orb4 == norbs:
                v_orb3 += 1
                v_orb4 = v_orb3 + 1
        v_orb2 += 1
        if v_orb2 == norbs:
            v_orb1 += 1
            v_orb2 = v_orb1 + 1
    
    
    row_block_index = nvirt ** 2 + ndocc ** 2 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    #81 <DQ|mu|DQ>
    col_block_index = nvirt ** 2 + ndocc ** 2 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + (ndocc * nvirt)):
        o_orb1 = (row - row_block_index) // nvirt
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // nvirt
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                for o in range(ndocc):
                    dipoles[row,col,:] -= 2 * dip1el[o,o,:]
                dipoles[row,col,:] += dip1el[o_orb1, o_orb1, :] # Remove contribution from 1e in HOMO i
                dipoles[row,col,:] -= dip1el[SOMO1, SOMO1, :] # Add contribution from 1e in SOMO1
                dipoles[row,col,:] -= dip1el[SOMO2, SOMO2, :] # Add contribution from 1e in SOMO2
                dipoles[row,col,:] -= dip1el[v_orb1, v_orb1, :] # Add contribution from 1e in LUMO j
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                dipoles[row, col, :] = dip1el[o_orb1, o_orb2, :]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                dipoles[row, col, :] = -dip1el[v_orb1, v_orb2, :]
            dipoles[col,row,:] = dipoles[row,col,:]
        
    #print("%10.5f"%linalg.norm(dipoles[:,:,0] - dipoles[:,:,0].T))  # checking symmetric
    #print("%10.5f"%linalg.norm(dipoles[:,:,1] - dipoles[:,:,1].T))
    #print("%10.5f"%linalg.norm(dipoles[:,:,2] - dipoles[:,:,2].T))   

    OS1_perm_dip=dipoles[0,0,:]
    ZWminus_perm_dip=dipoles[1,1,:]
    ZWplus_perm_dip=dipoles[2,2,:]
    OS3_perm_dip=dipoles[ndcv1 + ndoc1 + 2*(ndocc * nvirt) + 2*nvirt + 2*ndocc +3, ndcv1 + ndoc1 + 2*(ndocc * nvirt) + 2*nvirt + 2*ndocc +3,:]
    for dipole in [OS1_perm_dip, ZWminus_perm_dip, ZWplus_perm_dip, OS3_perm_dip]:
        for i in range(natoms):
            atom_z=0
            if atoms[i][0] in ['C','c','n1','N1']:
                atom_z=1
            elif atoms[i][0] in ['Cl','cl','CL','N2','n2']:
                atom_z=2   
            # for x in range(3):
            #     perm_dip[x]+=atom_z*coords[i,x]*tobohr
            dipole[:]+=atom_z*coords[i,:]*tobohr
    print(f"Permanent dipole moment of reference states:\n \
            Open-Shell Singlet (|OS1>) = {OS1_perm_dip[0]:.3f} x {OS1_perm_dip[1]:.3f} y {OS1_perm_dip[2]:.3f} z\n \
            Zwitterion - (|ZW->) = {ZWminus_perm_dip[0]:.3f} x {ZWminus_perm_dip[1]:.3f} y {ZWminus_perm_dip[2]:.3f} z\n \
            Zwitterion + (|ZW+>) = {ZWplus_perm_dip[0]:.3f} x {ZWplus_perm_dip[1]:.3f} y {ZWplus_perm_dip[2]:.3f} z\n \
            Open-Shell Triplet (|OS3>) = {OS3_perm_dip[0]:.3f} x {OS3_perm_dip[1]:.3f} y {OS3_perm_dip[2]:.3f} z\n")
    return dipoles


def print_ci_info(out_file, ci_energies, ci_coeffs, ndocc, norbs, tdms, rng, cutoff_energy, ci_type, csf_tol=0.01):
    print("Energy of the lowest CI state:", ci_energies[0])
    osc_array1 = np.zeros_like(ci_energies)
    osc_array3 = np.zeros_like(ci_energies)
    s2_array = np.zeros_like(ci_energies)
    nvirt = norbs - ndocc - 2
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
                    
            if ci_type == 'XCIS':
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
                elif j > (2 * nvirt + 2 * ndocc + 2) and j <= ((ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 2):
                    o_orb = ndocc - ((j - (2 * nvirt + 2 * ndocc + 3)) // nvirt)
                    v_orb = ((j - (2 * nvirt + 2 * ndocc + 3)) % nvirt) + 1
                    str = f"|1S^CV({o_orb}->{v_orb}')>" 
                    # S^2 = 0
            # Singlet Core to Virtual 2 (|1T^CV>)
                elif j > ((ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 2) and j <= (2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 2):
                    o_orb = ndocc - ((j - ((ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3)) // nvirt)
                    v_orb = ((j - ((ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3)) % nvirt) + 1
                    str = f"|1T^CV({o_orb}->{v_orb}')>" 
                    # S^2 = 0
            ########### TRIPLET CSFs ###########
            # Triplet ground state (|OS3>)
                elif j == (2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3): 
                    str = "|3^OS>"
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet core to SOMO 0 (|3^CS>)
                elif j > (2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3) and j <= (2 * (ndocc * nvirt) + 2 * nvirt + 3 * ndocc + 3):
                    iorb = (2 * (ndocc * nvirt) + 2 * nvirt + 3 * ndocc + 4) - j
                    str = f"|3^CS({iorb}->0)>" 
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1) 
            # Triplet core to SOMO 0' (|3^CS>)
                elif j > (2 * (ndocc * nvirt) + 2 * nvirt + 3 * ndocc + 3) and j <= (2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 3):
                    iorb = (2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 4) - j
                    str = f"|3^CS({iorb}->0')>" 
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet SOMO 0 to virtual (|3^SV>)
                elif j > (2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 3) and j <= (2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 3):
                    iorb = j - (2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 3)
                    str = f"|3^SV(0->{iorb}')>"
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet SOMO 0' to virtual (|3^SV>)
                elif j > (2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 3) and j <= (2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3):
                    iorb = j - (2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 3)
                    str = f"|3^SV(0'->{iorb}')>"
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet Core to Virtual 1 (|3T^CV>)
                elif j > (2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3) and j <= (3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|3T^CV({o_orb}->{v_orb}')>" 
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet Core to Virtual 2 (|3S^CV>)
                elif j > (3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3) and j <= (4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|3S^CV({o_orb}->{v_orb}')>" 
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet Core to Virtual 3 (|3X^CV>)
                elif j > (4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3) and j <= (5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|3X^CV({o_orb}->{v_orb}')>" 
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Quintet Core to Virtual (|5^CV>)
                elif j > (5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|5^CV({o_orb}->{v_orb}')>" 
                    spin += 6 * ci_coeffs[j,i]**2 # (S=2)
                    
                if np.absolute(ci_coeffs[j,i]) > csf_tol:
                    print("%s %10.5f" %(str, ci_coeffs[j,i]))
                    out_file.write("%s %10.5f \n" %(str, ci_coeffs[j,i]))

            elif ci_type == 'XCISD':
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
                elif j > (2 * nvirt + 2 * ndocc + 2) and j <= ((ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 2):
                    o_orb = ndocc - ((j - (2 * nvirt + 2 * ndocc + 3)) // nvirt)
                    v_orb = ((j - (2 * nvirt + 2 * ndocc + 3)) % nvirt) + 1
                    str = f"|1S^CV({o_orb}->{v_orb}')>" 
                    # S^2 = 0
            # Singlet Core to Virtual 2 (|1T^CV>)
                elif j > ((ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 2) and j <= (2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 2):
                    o_orb = ndocc - ((j - ((ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3)) // nvirt)
                    v_orb = ((j - ((ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3)) % nvirt) + 1
                    str = f"|1T^CV({o_orb}->{v_orb}')>" 
                    # S^2 = 0
            # Singlet Double Core to SOMO (|1^CSD>)
                elif j > (2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 2) and j <= (ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 2):
                    block_start = 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
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
                elif j > (ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 2) and j <= (ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 2):
                    block_start = ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
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
                elif j == (ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3): 
                    str = "|3^OS>"
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet Homo to SOMO 1 (|3^HS1>)
                elif j > (ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3) and j <= (ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 3 * ndocc + 3):
                    iorb = (ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 3 * ndocc + 4) - j
                    str = f"|3^CS({iorb}->0)>" 
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1) 
            # Triplet Homo to SOMO 2 (|3^HS2>)
                elif j > (ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 3 * ndocc + 3) and j <= (ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 3):
                    iorb = (ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 4) - j
                    str = f"|3^CS({iorb}->0')>" 
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet SOMO to LUMO 1 (|3^SL1>)
                elif j > (ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 3) and j <= (ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 3):
                    iorb = j - (ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 3)
                    str = f"|3^SV(0->{iorb}')>"
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet SOMO to LUMO 2 (|3^SL2>)
                elif j > (ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 3) and j <= (ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3):
                    iorb = j - (ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 3)
                    str = f"|3^SV(0'->{iorb}')>"
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet HOMO to LUMO 1 (|1^HL1>)
                elif j > (ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3) and j <= (ndcv1 + ndoc1 + 3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|3T^CV({o_orb}->{v_orb}')>" 
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet HOMO to LUMO 2 (|3^HL2>)
                elif j > (ndcv1 + ndoc1 + 3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3) and j <= (ndcv1 + ndoc1 + 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (ndcv1 + ndoc1 + 3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (ndcv1 + ndoc1 + 3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|3S^CV({o_orb}->{v_orb}')>" 
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet HOMO to LUMO 3 (|3^HL2>)
                elif j > (ndcv1 + ndoc1 + 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3) and j <= (ndcv1 + ndoc1 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (ndcv1 + ndoc1 + 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (ndcv1 + ndoc1 + 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|3X^CV({o_orb}->{v_orb}')>" 
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet double Core to SOMO    
                elif j > (ndcv1 + ndoc1 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3) and j <= (ndcv1 + ndocc ** 2 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3):
                    block_start = ndcv1 + ndoc1 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
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
                elif j > (ndcv1 + ndocc ** 2 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3) and j <= (nvirt ** 2 + ndocc ** 2 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3):
                    block_start = ndcv1 + ndocc ** 2 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
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
            # Quintet HOMO to LUMO (|5^HL>)
                elif j > (nvirt ** 2 + ndocc ** 2 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (nvirt ** 2 + ndocc ** 2 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (nvirt ** 2 + ndocc ** 2 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
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


def print_csf_info(ham_rot, norbs, ndocc, ci_type= 'XCIS'):
    # Print energies of CSFs
        nvirt = norbs - ndocc - 2
        ndoc1 = int((ndocc ** 2 + ndocc) / 2)
        ndcv1 = int((nvirt ** 2 + nvirt) / 2)
        for j in range(ham_rot.shape[1]):
                
            if ci_type == 'XCIS':
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
                elif j > (2 * nvirt + 2 * ndocc + 2) and j <= ((ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 2):
                    o_orb = ndocc - ((j - (2 * nvirt + 2 * ndocc + 3)) // nvirt)
                    v_orb = ((j - (2 * nvirt + 2 * ndocc + 3)) % nvirt) + 1
                    str = f"|1S^CV({o_orb}->{v_orb}')>"
                elif j > ((ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 2) and j <= (2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 2):
                    o_orb = ndocc - ((j - ((ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3)) // nvirt)
                    v_orb = ((j - ((ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3)) % nvirt) + 1
                    str = f"|1T^CV({o_orb}->{v_orb}')>"
                elif j == (2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3): 
                    str = "|3^OS>"
                elif j > (2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3) and j <= (2 * (ndocc * nvirt) + 2 * nvirt + 3 * ndocc + 3):
                    iorb = (2 * (ndocc * nvirt) + 2 * nvirt + 3 * ndocc + 4) - j
                    str = f"|3^CS({iorb}->0)>"
                elif j > (2 * (ndocc * nvirt) + 2 * nvirt + 3 * ndocc + 3) and j <= (2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 3):
                    iorb = (2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 4) - j
                    str = f"|3^CS({iorb}->0')>" 
                elif j > (2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 3) and j <= (2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 3):
                    iorb = j - (2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 3)
                    str = f"|3^SV(0->{iorb}')>"
                elif j > (2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 3) and j <= (2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3):
                    iorb = j - (2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 3)
                    str = f"|3^SV(0'->{iorb}')>"
                elif j > (2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3) and j <= (3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|3T^CV({o_orb}->{v_orb}')>"
                elif j > (3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3) and j <= (4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|3S^CV({o_orb}->{v_orb}')>"
                elif j > (4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3) and j <= (5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|3X^CV({o_orb}->{v_orb}')>"
                elif j > (5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|5^CV({o_orb}->{v_orb}')>"
                print(f"Energy of CSF {str}:", np.diag(ham_rot)[j])
            
            if ci_type == 'XCISD':
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
                elif j > (2 * nvirt + 2 * ndocc + 2) and j <= ((ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 2):
                    o_orb = ndocc - ((j - (2 * nvirt + 2 * ndocc + 3)) // nvirt)
                    v_orb = ((j - (2 * nvirt + 2 * ndocc + 3)) % nvirt) + 1
                    str = f"|1S^CV({o_orb}->{v_orb}')>"
                elif j > ((ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 2) and j <= (2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 2):
                    o_orb = ndocc - ((j - ((ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3)) // nvirt)
                    v_orb = ((j - ((ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3)) % nvirt) + 1
                    str = f"|1T^CV({o_orb}->{v_orb}')>"
                elif j > (2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 2) and j <= (ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 2):
                    block_start = 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
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
                elif j > (ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 2) and j <= (ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 2):
                    block_start = ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
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
                ########### TRIPLET CSFs ###########
                elif j == (ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3): 
                    str = "|3^OS>"
                elif j > (ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3) and j <= (ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 3 * ndocc + 3):
                    iorb = (ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 3 * ndocc + 4) - j
                    str = f"|3^CS({iorb}->0)>" 
                elif j > (ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 3 * ndocc + 3) and j <= (ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 3):
                    iorb = (ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 4) - j
                    str = f"|3^CS({iorb}->0')>" 
                elif j > (ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 3) and j <= (ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 3):
                    iorb = j - (ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 3)
                    str = f"|3^SV(0->{iorb}')>"
                elif j > (ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 3) and j <= (ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3):
                    iorb = j - (ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 3)
                    str = f"|3^SV(0'->{iorb}')>"
                elif j > (ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3) and j <= (ndcv1 + ndoc1 + 3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (ndcv1 + ndoc1 + 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|3T^CV({o_orb}->{v_orb}')>" 
                elif j > (ndcv1 + ndoc1 + 3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3) and j <= (ndcv1 + ndoc1 + 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (ndcv1 + ndoc1 + 3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (ndcv1 + ndoc1 + 3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|3S^CV({o_orb}->{v_orb}')>" 
                elif j > (ndcv1 + ndoc1 + 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3) and j <= (ndcv1 + ndoc1 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (ndcv1 + ndoc1 + 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (ndcv1 + ndoc1 + 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|3X^CV({o_orb}->{v_orb}')>" 
                elif j > (ndcv1 + ndoc1 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3) and j <= (ndcv1 + ndocc ** 2 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3):
                    block_start = ndcv1 + ndoc1 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
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
                elif j > (ndcv1 + ndocc ** 2 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3) and j <= (nvirt ** 2 + ndocc ** 2 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3):
                    block_start = ndcv1 + ndocc ** 2 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
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
            # QUINTET
                elif j > (nvirt ** 2 + ndocc ** 2 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (nvirt ** 2 + ndocc ** 2 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (nvirt ** 2 + ndocc ** 2 + 5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|5^CV({o_orb}->{v_orb}')>"
            
                print(f"Energy of CSF {str}:", np.diag(ham_rot)[j])
   

def diagonalise_xcis(ham_rot, ndocc, norbs, rng, nstates, out, ci_type='XCIS'):
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

    # ------------------------------------------------------------------
    # 1. Compute block boundaries
    # ------------------------------------------------------------------
    nvirt = norbs - ndocc - 2
    ndoc1 = int((ndocc ** 2 + ndocc) / 2)
    ndcv1 = int((nvirt ** 2 + nvirt) / 2)
    ndoc3 = ndocc ** 2 - ndoc1
    ndcv3 = nvirt ** 2 - ndcv1
    
    if ci_type == 'XCIS':    
        n_singlet = 2 * (ndocc * nvirt) + 2 * ndocc + 2 * nvirt + 3
        n_triplet = 3 * (ndocc * nvirt) + 2 * ndocc + 2 * nvirt + 1
    else:
        n_singlet = ndoc1 + ndcv1 + 2 * (ndocc * nvirt) + 2 * ndocc + 2 * nvirt + 3
        n_triplet = ndoc3 + ndcv3 + 3 * (ndocc * nvirt) + 2 * ndocc + 2 * nvirt + 1
    n_quintet = ndocc * nvirt

    assert n_singlet + n_triplet + n_quintet == nstates, (
        f"Block sizes do not sum to nstates: "
        f"{n_singlet} + {n_triplet} + {n_quintet} = "
        f"{n_singlet + n_triplet + n_quintet} != {nstates}"
    )

    s_start, s_end = 0,                    n_singlet
    t_start, t_end = n_singlet,            n_singlet + n_triplet
    q_start, q_end = n_singlet + n_triplet, nstates

    msg = (
        f"Block-diagonalising XCIS Hamiltonian ...\n"
        f"  Singlet block : rows {s_start}–{s_end-1}  ({n_singlet} states)\n"
        f"  Triplet block : rows {t_start}–{t_end-1}  ({n_triplet} states)\n"
        f"  Quintet block : rows {q_start}–{q_end-1}  ({n_quintet} states)\n"
    )
    print(msg)
    out.write(msg)

    # Slice the three diagonal blocks
    H_s = ham_rot[s_start:s_end, s_start:s_end]
    H_t = ham_rot[t_start:t_end, t_start:t_end]
    H_q = ham_rot[q_start:q_end, q_start:q_end]

    # ------------------------------------------------------------------
    # 2. Diagonalise each block
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
        e_q, v_q = sp.eigsh(H_q, k=k_q, which="SA")

    else:
        # Dense path — full diagonalisation of each block
        msg = "Using dense solver (eigh) on each block ...\n"
        print(msg)
        out.write(msg)

        e_s, v_s = linalg.eigh(H_s)
        e_t, v_t = linalg.eigh(H_t)
        e_q, v_q = linalg.eigh(H_q)

    # ------------------------------------------------------------------
    # 3. Embed block eigenvectors into the full CSF basis
    #    Each column of ci_coeffs_block is a state vector of length nstates,
    #    with zeros outside the relevant block.
    # ------------------------------------------------------------------
    def embed(v, start, total):
        """Pad eigenvector matrix v into the full basis of size `total`."""
        n_basis, n_vecs = v.shape
        full = np.zeros((total, n_vecs))
        full[start:start + n_basis, :] = v
        return full          # shape: (nstates, n_vecs)

    V_s = embed(v_s, s_start, nstates)   # (nstates, k_s or n_singlet)
    V_t = embed(v_t, t_start, nstates)   # (nstates, k_t or n_triplet)
    V_q = embed(v_q, q_start, nstates)   # (nstates, k_q or n_quintet)

    # ------------------------------------------------------------------
    # 4. Concatenate all eigenvalues/vectors and sort by energy
    # ------------------------------------------------------------------
    all_energies = np.concatenate([e_s, e_t, e_q])
    all_coeffs   = np.concatenate([V_s, V_t, V_q], axis=1)  # (nstates, total_vecs)

    sort_idx = np.argsort(all_energies)
    all_energies = all_energies[sort_idx]
    all_coeffs   = all_coeffs[:, sort_idx]

    # ------------------------------------------------------------------
    # 5. Trim to rng states if using the sparse path
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



def ci_rot(ndocc,norbs,coords,atoms,energy0,repulsion,orb_energies,hf_orbs, file, ci_type = "XCIS"):
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
        if ci_type == 'XCIS':
            ham_rot = hetero_xcis_ham_rot(ndocc, norbs, energy0, orb_energies, rep_tens)
        elif ci_type == 'XCISD':
            ham_rot = hetero_xcisd_ham_rot(ndocc, norbs, energy0, orb_energies, rep_tens)
            
        print('Dimensions of CI matrix:', ham_rot.shape)
        print("Checking that the Hamiltonian is symmetric (a value of zero means matrix is symmetric) ... ")
        print("Frobenius norm of matrix - matrix transpose = %f.\n" %(linalg.norm(ham_rot-ham_rot.T)))

        out.write("Checking that the Hamiltonian is symmetric (a value of zero means matrix is symmetric) ... \n")
        out.write("Frobenius norm of matrix - matrix transpose = %f.\n" %(linalg.norm(ham_rot-ham_rot.T)))
        
        # Print energies of CSFs
        #print_csf_info(ham_rot, norbs, ndocc, ci_type=ci_type)
        
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
        
        #ci_energies, ci_coeffs = diagonalise_xcis(ham_rot, ndocc, norbs, rng, nstates, out, ci_type=ci_type)
        
        
        # Diagonalize CIS Hamiltonianfor first rng excited states
        if rng < nstates:
            print("Diagonalizing Hamiltonian using the sparse matrix method ...\n")
            out.write("Diagonalizing Hamiltonian using the sparse matrix method ...\n")

            ci_energies, ci_coeffs = sp.eigsh(ham_rot,k=rng,which="SA")
        elif rng == nstates:
            print("Diagonalizing Hamiltonian using the dense matrix method ...\n")
            out.write("Diagonalizing Hamiltonian using the dense matrix method ...\n")
            ci_energies, ci_coeffs = linalg.eigh(ham_rot)
        

        # Calculate transition dipole moment matrix
        if ci_type == 'XCIS':
            dip_array = dipole_xcis(coords,atoms,norbs,hf_orbs,ndocc,nstates)
        elif ci_type == 'XCISD':
            dip_array = dipole_xcisd(coords,atoms,norbs,hf_orbs,ndocc,nstates)
        
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
        strngs, osc_arrays, s2_array = print_ci_info(out, ci_energies, ci_coeffs, ndocc, norbs, tdms, rng, cutoff_energy, ci_type=ci_type, csf_tol=0.05)
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
    strngs, ci_energies_array, osc_arrays, s2_array = ci_rot(ndocc, natoms, coord, atoms_array, energy0, two_body, orb_energy, hf_orbs, file, ci_type = 'XCISD')
    return strngs, ci_energies_array, osc_arrays, s2_array  #return gnuplot data for plotting spectrum




