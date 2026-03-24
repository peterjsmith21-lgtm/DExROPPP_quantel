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
    for line in f: # Read through lines of file
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
   

def conec(ncarb, dist_array):
    '''
    Group atoms in alternant hydrocarbons into starred and unstarred lists.
    
    Args:
        - ncarb (int): Number of carbon atoms in the molecule.
        - dist_array (ndarray): 2D Array of interatomic distances in Angstrom.
    Returns:
        - star (list): List of indices of starred atoms.
        - unst (list): List of indices of unstarred atoms.
    '''
    star = []
    unst = []
    star.append(0)
    satom = [0]
    for n in range(ncarb):
        if len(star)+len(unst) == ncarb:
            break
        uatom = []
        for i in satom:
            for j in range(i+1,ncarb):
                if dist_array[i,j] < cutoff and j not in unst:
                    uatom.append(j)
                    unst.append(j)
        satom = []            
        for i in uatom:
            for j in range(i+1,ncarb):
                if dist_array[i,j] < cutoff and j not in star:
                    satom.append(j)
                    star.append(j)
    if len(star) < len(unst):
        print('Swapping starred and unstarred atoms ...')
        array = star
        star = unst
        unst = array
    print(' ')               
    print('Starred atoms: ' +str(star))
    print('Un-starred atoms: ' +str(unst)+'\n')
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
  


def orb_sign(orbs,orb_energies,nelec,dist_array,alt):
    '''
    Adjusts orbital phases to satisfy alternant hydrocarbon symmetry.Ensures that starred atoms retain their sign 
    across a pair, while unstarred atoms undergo a phase inversion in the antibonding orbital.

    Args:
        orbs (ndarray): Matrix of orbital coefficients (rows=atoms, cols=orbitals).
        orb_energies (ndarray): Array of orbital energies.
        nelec (int): Total number of electrons in the system.
        dist_array (ndarray): Matrix of inter-atomic distances.
        alt (bool): Alternacy status flag.

    Returns:
        orbs (ndarray): The orbital coefficient matrix with standardized phases.
    '''
    if alt==True:
        print('\nGrouping orbitals according to alternacy symmetry...')
        ncarb = orbs.shape[0]
        somo_energy = orb_energies[int((nelec-1)/2)]
        for i in range(orb_energies.shape[0]):
            orb_energies[i] = orb_energies[i] - somo_energy
        orb_list,alt = order_orbs(ncarb,orbs,orb_energies,alt)
    if alt==True:
        star,unst = conec(ncarb,dist_array)
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

def compute_j00(orbs,repulsion,ndocc):
    """Calculates the Coulomb self-repulsion integral (J00) for both of the SOMOs.

    Args:
        orbs (ndarray): Matrix of orbital coefficients (rows=atoms, cols=orbitals).
        repulsion (ndarray): Matrix of inter-atomic electron repulsion integrals.
        ndocc (int): The number of doubly-occupied orbitals. Gives the appropriate index for the first SOMO.

    Returns:
        J00 (float): The calculated Coulomb repulsion term for both of the SOMOs.
    """
    J00 = 0
    for l in range(orbs.shape[0]): # atom l
        for m in range(orbs.shape[0]): # atom m
            J00 += (orbs[l,ndocc]**2 * orbs[m,ndocc]**2 + orbs[l,ndocc+1]**2 * orbs[m,ndocc+1]**2) * repulsion[l,m]
    return J00

def compute_k00(orbs,repulsion,ndocc):
    """Calculates the Exchange interaction between the SOMOs (K00').

    Args:
        orbs (ndarray): Matrix of orbital coefficients (rows=atoms, cols=orbitals).
        repulsion (ndarray): Matrix of inter-atomic electron repulsion integrals.
        ndocc (int): The number of doubly-occupied orbitals. Gives the appropriate index for the first SOMO.

    Returns:
        K00 (float): The calculated Exchange interaction term for both of the SOMOs.
    """
    K00 = 0
    for l in range(orbs.shape[0]): # atom l
        for m in range(orbs.shape[0]): # atom m
            K00 += (orbs[l,ndocc] * orbs[l,ndocc+1]) * (orbs[m,ndocc] * orbs[m,ndocc+1]) * repulsion[l,m]
    return K00


def energy(hopping,repulsion,fock_mat,density,orbs,ndocc):
    """Calculates the total SCF energy of a fictituous system that is close to the energy of the open-shell singlet ground state.

    Returns:
        float: The total calculated SCF energy of the system from PPP theory.
    """
    return 0.5 * (np.dot(density.flatten(), hopping.flatten()) + np.dot(density.flatten(), fock_mat.flatten()))

def cartesian_operators(coords,hf_orbs):
    natoms = coords.shape[0]
    norbs = hf_orbs.shape[1]
    dip1el = np.zeros((norbs,norbs,3))
    for i in range(norbs):
        for j in range(i,norbs):
            for u in range(natoms):
                dip1el[i,j,:] += hf_orbs[u,i] * coords[u,:] * hf_orbs[u,j] * tobohr
                dip1el[j,i,:] = dip1el[i,j,:]
    
    x_operator = dip1el[:, :, 0]
    y_operator = dip1el[:, :, 1]
    z_operator = dip1el[:, :, 2]
    full_cartesian_operator = dip1el
    
    return full_cartesian_operator, x_operator, y_operator, z_operator

class DIIS:
    def __init__(self, max_iter=100):
        self.max_iter = max_iter
        self.fock_list = []
        self.error_list = []
    
    def get_extrapolated_fock(self, F, D):
        error = F @ D - D @ F
        
        self.fock_list.append(F)
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

#Main HF function
def main_scf(file, params, maxcycles=1000, d_tol=5e-15):
    '''
    main Hartree-Fock function to perform SCF calculation for a radical molecule using the ExROPPP method.
    
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
    guess_evals,evecs = np.linalg.eigh(hopping)
    guess_dens = density(evecs,ndocc)
    #iterate until convergence 
    energy1=0
    diis = DIIS()
    use_diis = False
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
        if iter == 1000 and conv_crit > 0.01:
            print('--DM Struggling to converge after 40 iterations, implementing DIIS solver--')
            use_diis = True
        if use_diis:
            fock_mat = diis.get_extrapolated_fock(fock_mat, guess_dens)
        evals, orbs = np.linalg.eigh(fock_mat)
        dens = density(orbs,ndocc)
        energy2 = energy(hopping, repulsion, fock_mat, dens, orbs, ndocc)
        conv_crit = np.absolute(guess_dens-dens).max()
        print(iter, energy2, conv_crit, energy2 - energy1)
        if conv_crit < d_tol:
            break
        if energy2 > energy1:
            print('\nEnergy rises!')
        energy1 = energy2
        if not use_diis:
            #guess_dens = (0.05 * dens) + (0.95 * guess_dens)
            guess_dens = dens
        else:
            guess_dens = dens

    
    SOMO1 = ndocc
    SOMO2 = ndocc + 1
    
    #assert abs(evals[SOMO1] - evals[SOMO2]) < 1e-12, "SOMOs are not degenerate!"
    
    
    print('\nEnforcing Spatial Symmetry in x for denerate SOMOs\n')
    x_operator = cartesian_operators(coord,orbs)[1]
    SOMOs_in_x_basis = x_operator[np.ix_([SOMO1, SOMO2], [SOMO1, SOMO2])]
    _, x_rotation = np.linalg.eigh(SOMOs_in_x_basis)
    SOMOs_x_rot = np.dot(orbs[:, [SOMO1, SOMO2]], x_rotation)
    orbs[:, [SOMO1, SOMO2]] = SOMOs_x_rot
    '''
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



def cis_ham_rot(ndocc, energy0, orb_energies, j00, k00, rep_tens):
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
    nstates = 8 * ndocc + 4 # Ignoring HOMO to LUMO (double) excitations for the time being
    cish = np.zeros((nstates,nstates))
    
    ################# SINGLET BLOCK ######################
    #1 <OS1|H|OS1>
    cish[0,0] = energy0 - (0.25 * j00 ) + (1.5 * k00)
    #2 <OS1|H|ZW0>
    cish[0,1] = ((2 ** 0.5) / 2) * (rep_tens[SOMO1,SOMO2,SOMO1,SOMO1] - rep_tens[SOMO1,SOMO2,SOMO2,SOMO2])
    cish[1,0] = cish[0,1]
    #3 <OS1|H|ZW0'>
    cish[0,2] = ((2 ** 0.5) / 2) * (rep_tens[SOMO1,SOMO2,SOMO1,SOMO1] - rep_tens[SOMO1,SOMO2,SOMO2,SOMO2])
    cish[2,0] = cish[0,2]
    #4 <OS1|H|HS1> 
    block_index = 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        cish[0,col] = 1.5 * rep_tens[o_orb,SOMO2,SOMO2,SOMO1] - 0.5 * rep_tens[o_orb,SOMO1,SOMO1,SOMO1]
        cish[col,0] = cish[0,col]
    #5 <OS1|H|HS2> 
    block_index = ndocc + 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        cish[0,col] = 0.5 * rep_tens[o_orb,SOMO2,SOMO2,SOMO2] - 1.5 * rep_tens[o_orb,SOMO1,SOMO1,SOMO2]
        cish[col,0] = cish[0,col]
    #6 <OS1|H|SL1>
    block_index = 2 * ndocc + 3
    for col in range(block_index, block_index + ndocc):
        v_orb = col - block_index + (SOMO2 + 1)
        cish[0,col] = 0.5 * rep_tens[v_orb,SOMO2,SOMO2,SOMO2] - 1.5 * rep_tens[v_orb,SOMO1,SOMO1,SOMO2]
        cish[col,0] = cish[0,col]
    #7 <OS1|H|SL2>
    block_index = 3 * ndocc + 3
    for col in range(block_index, block_index + ndocc):
        v_orb = col - block_index + (SOMO2 + 1)
        cish[0,col] = 1.5 * rep_tens[v_orb,SOMO2,SOMO2,SOMO1] - 0.5 * rep_tens[v_orb,SOMO1,SOMO1,SOMO1]
        cish[col,0] = cish[0,col]
    #8 - 13 are triplet states so have 0 interaction.
    
    # 14 <ZW0|H|ZW0>
    cish[1,1] = energy0 + orb_energies[SOMO1] - orb_energies[SOMO2] + 0.25 * j00 + 0.5 * k00 - rep_tens[SOMO1, SOMO1, SOMO2, SOMO2] + 0.5 * k00
    #15 <ZW0|H|ZW0'>
    cish[1,2] = rep_tens[SOMO1,SOMO2,SOMO2,SOMO1]
    cish[2,1] = cish[1,2]
    #16 <ZW0|H|HS1>
    block_index = 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        cish[1,col] = (2 ** 0.5) * (rep_tens[o_orb,SOMO2,SOMO1,SOMO1] - 0.5 * rep_tens[o_orb,SOMO1,SOMO1,SOMO2] - 0.5 * rep_tens[o_orb,SOMO2,SOMO2,SOMO2])
        cish[col,1] = cish[1,col]
    #17 <ZW0|H|HS2>
    block_index = ndocc + 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        cish[1,col] = (-2 ** 0.5) * rep_tens[o_orb,SOMO2,SOMO2,SOMO1]
        cish[col, 1] = cish[1,col]
    #18 <ZW0|H|SL1>
    block_index = 2 * ndocc + 3
    for col in range(block_index, block_index + ndocc):
        v_orb = col - block_index + (SOMO2 + 1)
        cish[1,col] = (2 ** 0.5) * (rep_tens[v_orb,SOMO1,SOMO2,SOMO2] - 0.5 * rep_tens[v_orb,SOMO1,SOMO1,SOMO1] - 0.5 * rep_tens[v_orb,SOMO2,SOMO2,SOMO1])
        cish[col,1] = cish[1,col]
    #19 <ZW0|H|SL2>
    block_index = 3*ndocc + 3
    for col in range(block_index, block_index + ndocc):
        v_orb = col - block_index + (SOMO2 + 1)
        cish[1,col] = (2 ** 0.5) * rep_tens[v_orb,SOMO1,SOMO1,SOMO2]
        cish[col,1] = cish[1,col]
    #20 - 25 are triplet states so have 0 interaction.
    
    #26 <ZW0'|H|ZW0'>
    cish[2,2] = energy0 + orb_energies[SOMO2] - orb_energies[SOMO1] + 0.25 * j00 + 0.5 * k00 - rep_tens[SOMO1, SOMO1, SOMO2, SOMO2] + 0.5 * k00
    #27 <ZW0'|H|HS1>
    block_index = 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        cish[2,col] = (-2 ** 0.5) * rep_tens[o_orb,SOMO1,SOMO1,SOMO2]
        cish[col,2] = cish[2,col]
    #28 <ZW0'|H|HS2>
    block_index = ndocc + 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        cish[2,col] = (2 ** 0.5) * (rep_tens[o_orb,SOMO1,SOMO2,SOMO2] - 0.5 * rep_tens[o_orb,SOMO2,SOMO2,SOMO1] - 0.5 * rep_tens[o_orb,SOMO1,SOMO1,SOMO1]) # CHECK SIGN
        cish[col,2] = cish[2,col]
    #29 <ZW0'|H|SL1>
    block_index = 2 * ndocc + 3
    for col in range(block_index, block_index + ndocc):
        v_orb = col - block_index + (SOMO2 + 1)
        cish[2,col] = (-2 ** 0.5) * rep_tens[v_orb,SOMO2,SOMO2,SOMO1]
        cish[col,2] = cish[2,col]
    #30 <ZW0'|H|SL2>
    block_index = 3 * ndocc + 3
    for col in range(block_index, block_index + ndocc):
        v_orb = col - block_index + (SOMO2 + 1)
        cish[2,col] = (2 ** 0.5) * (0.5 * rep_tens[v_orb,SOMO2,SOMO2,SOMO2] + 0.5 * rep_tens[v_orb,SOMO1,SOMO1,SOMO2] - rep_tens[v_orb,SOMO2,SOMO1,SOMO1]) # CHECK SIGN
        cish[col,2] = cish[2,col]
    #31 - 36 are triplet states so have 0 interaction.
    
    row_block_index = 3
    #37 <HS1|H|HS1>
    col_block_index = 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(row, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                cish[row, col] = energy0 + orb_energies[SOMO1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, SOMO1, SOMO1] + 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] \
                             + 1.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb1] + 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb1]
            else:    
                cish[row, col] = 0.5 * (rep_tens[o_orb2, SOMO1, SOMO1, o_orb1] + 0.5 * rep_tens[o_orb2, SOMO2, SOMO2, o_orb1]) - rep_tens[o_orb2, o_orb1, SOMO1, SOMO1]
            cish[col, row] = cish[row,col]
    #38 <HS1|H|HS2>
    col_block_index = ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                cish[row, col] = rep_tens[SOMO1, o_orb1, o_orb1, SOMO2] - rep_tens[o_orb1, o_orb1, SOMO1, SOMO2] + 0.5 * rep_tens[SOMO1, SOMO2, SOMO1, SOMO1] \
                             + 0.5 * rep_tens[SOMO1, SOMO2, SOMO2, SOMO2]
            else:    
                cish[row, col] = - rep_tens[o_orb1, o_orb2, SOMO1, SOMO2] - rep_tens[o_orb1, SOMO1, SOMO2, o_orb2]
            cish[col, row] = cish[row, col]
    #39 <HS1|H|SL1>
    col_block_index = 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc):
            v_orb = col - col_block_index + (SOMO2 + 1)
            cish[row, col] = rep_tens[o_orb, SOMO1, SOMO2, v_orb] - 2 * rep_tens[o_orb, SOMO2, SOMO1, v_orb]
            cish[col, row] = cish[row,col]
    #40 <HS1|H|SL2>
    col_block_index = 3*ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc):
            v_orb = col - col_block_index + (SOMO2 + 1)
            cish[row, col] = - rep_tens[o_orb, SOMO1, SOMO1, v_orb]
            cish[col, row] = cish[row,col]
    # 41 - 46 are triplets so have no interaction
    
    row_block_index = ndocc + 3
    #47 <HS2|H|HS2>
    col_block_index = ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(row, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                cish[row, col] = energy0 + orb_energies[SOMO2] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, SOMO2, SOMO2] + 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] \
                             + 1.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb1] + 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb1]
            else:    
                cish[row, col] = 1.5 * (rep_tens[o_orb2, SOMO1, SOMO1, o_orb1] + rep_tens[o_orb2, o_orb1, SOMO2, SOMO2]) - 0.5 * rep_tens[o_orb2, SOMO2, SOMO2, o_orb1]
            cish[col, row] = cish[row,col]
    #48 <HS2|H|SL1>
    col_block_index = 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc):
            v_orb = col - col_block_index + (SOMO2 + 1)
            cish[row, col] = - rep_tens[o_orb, SOMO2, SOMO2, v_orb]
            cish[col, row] = cish[row, col]
    #49 <HS2|H|SL2>
    col_block_index = 3 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc):
            v_orb = col - col_block_index + (SOMO2 + 1)
            cish[row, col] = rep_tens[o_orb, SOMO2, SOMO1, v_orb] - 2 * rep_tens[o_orb, SOMO1, SOMO2, v_orb]
            cish[col, row] = cish[row,col]
    #50 - 55 are triplets so no interaction
    
    row_block_index = 2 * ndocc + 3
    #56 <SL1|H|SL1>
    col_block_index = 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(row, col_block_index + ndocc):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                cish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[SOMO2] - rep_tens[v_orb1, v_orb1, SOMO2, SOMO2] + 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] \
                             - 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] + 1.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb1] + 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb1]
            else:    
                cish[row, col] = 1.5 * rep_tens[v_orb2, SOMO2, SOMO2, v_orb1] + 0.5 * rep_tens[v_orb2, SOMO1, SOMO1, v_orb1] -  rep_tens[v_orb2, v_orb1, SOMO2, SOMO2]
            cish[col, row] = cish[row,col]
    #57 <SL1|H|SL2>
    col_block_index = 3 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + ndocc):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                cish[row, col] = 0.5 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO2] + 0.5 * rep_tens[SOMO1, SOMO2, SOMO2, SOMO2] - rep_tens[v_orb1, v_orb1, SOMO1, SOMO2] - rep_tens[v_orb1, SOMO1, SOMO2, v_orb1]
            else:    
                cish[row, col] = - rep_tens[v_orb1, v_orb2, SOMO1, SOMO2] - rep_tens[v_orb1, SOMO2, SOMO1, v_orb2]
            cish[col, row] = cish[row,col]
    
    #58 - 63 are all triplets so have no interaction
    
    row_block_index = 3 * ndocc + 3
    #64 <SL2|H|SL2>
    col_block_index = 3 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(row, col_block_index + ndocc):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                cish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[SOMO1] - rep_tens[v_orb1, v_orb1, SOMO1, SOMO1] + 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] \
                             - 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2]+ 1.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb1] + 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb1]
            else:    
                cish[row, col] = 1.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb2] - 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] - rep_tens[v_orb1, v_orb2, SOMO1, SOMO1] - rep_tens[v_orb1, v_orb2, SOMO2, SOMO2]
            cish[col, row] = cish[row,col]
    
    #65 - 70 are all triplets so have no interaction
    
    ################# TRIPLET BLOCK ######################
    
    row_index = 4 * ndocc + 3
    #71 <OS3|H|OS3>
    cish[row_index, row_index] = energy0 - (0.25 * j00 ) - (0.5 * k00)
    #72 <OS3|H|HS1>
    col_index = 4 * ndocc + 4
    for col in range(col_index, col_index + ndocc):
        o_orb = col - col_index
        cish[row_index, col] = 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO1] + 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO1]
        cish[col, row_index] = cish[row_index,col]
    #73 <OS3|H|HS2>
    col_index = 5 * ndocc + 4
    for col in range(col_index, col_index + ndocc):
        o_orb = col - col_index
        cish[row_index, col] = 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO2] + 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO2]
        cish[col, row_index] = cish[row_index,col]
    #73 <OS3|H|SL1>
    col_index = 6 * ndocc + 4
    for col in range(col_index, col_index + ndocc):
        v_orb = col - col_index + (SOMO2 + 1)
        cish[row_index, col] = 0.5 * rep_tens[v_orb, SOMO1, SOMO2, SOMO1] - 0.5 * rep_tens[v_orb, SOMO2, SOMO2, SOMO2]
        cish[col, row_index] = cish[row_index,col]
    #73 <OS3|H|SL2>
    col_index = 7 * ndocc + 4
    for col in range(col_index, col_index + ndocc):
        v_orb = col - col_index + (SOMO2 + 1)
        cish[row_index, col] = 0.5 * rep_tens[v_orb, SOMO1, SOMO1, SOMO1] - 0.5 * rep_tens[v_orb, SOMO2, SOMO2, SOMO1]
        cish[col, row_index] = cish[row_index, col]
    
    row_block_index = 4 * ndocc + 4
    #74 <HS1|H|HS1>
    col_block_index = 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(row, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                cish[row, col] = energy0 + orb_energies[SOMO1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, SOMO1, SOMO1] + 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] \
                             - 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb1] + 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb1]
            else:    
                cish[row, col] = 0.5 * (rep_tens[o_orb2, SOMO1, SOMO1, o_orb1] - rep_tens[o_orb2, o_orb1, SOMO1, SOMO1]) - 1.5 * rep_tens[o_orb2, SOMO2, SOMO2, o_orb1]
            cish[col, row] = cish[row,col]
    #75 <HS1|H|HS2>
    col_block_index = 5 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                cish[row, col] = rep_tens[SOMO1, o_orb1, o_orb1, SOMO2] - rep_tens[o_orb1, o_orb1, SOMO1, SOMO2] + 0.5 * rep_tens[SOMO2, SOMO1, SOMO1, SOMO1] \
                             + 0.5 * rep_tens[SOMO1, SOMO2, SOMO2, SOMO2]                                                               #CHECK RESULT, SAME AS SINGLET?
            else:    
                cish[row, col] = rep_tens[o_orb2, o_orb1, SOMO1, SOMO2] - rep_tens[o_orb2, SOMO1, SOMO2, o_orb1]
            cish[col, row] = cish[row,col]
    #76 <HS1|H|SL1>
    col_block_index = 6 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc):
            v_orb = col - col_block_index + (SOMO2 + 1)
            cish[row, col] = - rep_tens[o_orb, SOMO1, SOMO2, v_orb] # CHECK SIGN
            cish[col, row] = cish[row,col]
    #77 <HS1|H|SL2>
    col_block_index = 7 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc):
            v_orb = col - col_block_index + (SOMO2 + 1)
            cish[row, col] = rep_tens[o_orb, SOMO1, SOMO1, v_orb] # CHECK SIGN
            cish[col, row] = cish[row,col]

    
    row_block_index = 5 * ndocc + 4
    #78 <HS2|H|HS2>
    col_block_index = 5 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(row, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:

                cish[row, col] = energy0 + orb_energies[SOMO2] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, SOMO2, SOMO2] + 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] \
                             - 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb1] + 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb1]
            else:
                cish[row, col] = rep_tens[o_orb1, o_orb2, SOMO2, SOMO2] - 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb2] - 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb2]
            cish[col, row] = cish[row,col]
    #79 <HS2|H|SL1>
    col_block_index = 6 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc):
            v_orb = col - col_block_index + (SOMO2 + 1)
            cish[row, col] = - rep_tens[o_orb, SOMO2, SOMO2, v_orb]
            cish[col, row] = cish[row,col]
    #80 <HS2|H|SL2>
    col_block_index = 7 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc):
            v_orb = col - col_block_index + (SOMO2 + 1)
            cish[row, col] = rep_tens[o_orb, SOMO2, SOMO1, v_orb]
            cish[col, row] = cish[row,col]
            
    row_block_index = 6 * ndocc + 4
    #81 <SL1|H|SL1>
    col_block_index =  6 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(row, col_block_index + ndocc):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                cish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[SOMO2] - rep_tens[v_orb1, v_orb1, SOMO2, SOMO2] + 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] \
                             - 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] - 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb1] + 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb1]
            else:    
                cish[row, col] =  0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] - rep_tens[v_orb1, v_orb2, SOMO2, SOMO2] - 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb2]
            cish[col, row] = cish[row,col]
    #82 <SL1|H|SL2>
    col_block_index = 7 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + ndocc):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                cish[row, col] = rep_tens[v_orb1, v_orb1, SOMO1, SOMO2] - rep_tens[v_orb1, SOMO1, SOMO2, v_orb1] - 0.5 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO2] - 0.5 * rep_tens[SOMO1, SOMO2, SOMO2, SOMO2]
            else:    
                cish[row, col] = rep_tens[v_orb1, v_orb2, SOMO1, SOMO2] - rep_tens[v_orb1, SOMO2, SOMO1, v_orb2] # CHECK SIGN
            cish[col, row] = cish[row,col]
    
    row_block_index = 7 * ndocc + 4
    #83 <SL2|H|SL2>
    col_block_index =  7 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(row, col_block_index + ndocc):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                cish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[SOMO1] - rep_tens[v_orb1, v_orb1, SOMO1, SOMO1] + 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] \
                             - 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] - 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb1] + 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb1]
            else:    
                cish[row, col] =  0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] - 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb2] - rep_tens[v_orb1, v_orb2, SOMO1, SOMO1] \
                                  - rep_tens[v_orb1, v_orb2, SOMO2, SOMO2]
            cish[col, row] = cish[row,col]
    
    return cish


def xcis_ham_rot(ndocc, energy0, orb_energies, j00, k00, rep_tens):
    '''
    Form the XCIS Hamiltonian matrix in the rotated CSF basis. Matrix elements on off-diagonals are typically 2e integrals, found in the working doc.
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
    nstates = 6 * ndocc ** 2 + 8 * ndocc + 4  # 6 * ndocc^2 doubles (HOMO to LUMO), 8 * ndocc singles (HOMO to SOMO and SOMO to LUMO), 4 reference configurations (OS GSs and Zwitterions)
    xcish = np.zeros((nstates,nstates))
    
    ################# SINGLET BLOCK ######################
    #1 <OS1|H|OS1>
    xcish[0,0] = energy0 - (0.25 * j00 ) + (1.5 * k00)
    #2 <OS1|H|ZW0>
    xcish[0,1] = ((2 ** 0.5) / 2) * (rep_tens[SOMO1,SOMO2,SOMO1,SOMO1] - rep_tens[SOMO1,SOMO2,SOMO2,SOMO2])
    xcish[1,0] = xcish[0,1]
    #3 <OS1|H|ZW0'>
    xcish[0,2] = ((2 ** 0.5) / 2) * (rep_tens[SOMO1,SOMO2,SOMO1,SOMO1] - rep_tens[SOMO1,SOMO2,SOMO2,SOMO2])
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
    for col in range(block_index, block_index + ndocc):
        v_orb = col - block_index + (SOMO2 + 1)
        xcish[0,col] = 0.5 * rep_tens[v_orb,SOMO2,SOMO2,SOMO2] - 1.5 * rep_tens[v_orb,SOMO1,SOMO1,SOMO2]
        xcish[col,0] = xcish[0,col]
    #7 <OS1|H|SL2>
    block_index = 3 * ndocc + 3
    for col in range(block_index, block_index + ndocc):
        v_orb = col - block_index + (SOMO2 + 1)
        xcish[0,col] = 1.5 * rep_tens[v_orb,SOMO2,SOMO2,SOMO1] - 0.5 * rep_tens[v_orb,SOMO1,SOMO1,SOMO1]
        xcish[col,0] = xcish[0,col]
    #8 <OS1|H|HL1> = 0
    #9 <OS1|H|HL2>
    block_index =  ndocc ** 2 + 4 * ndocc + 3
    for col in range(block_index, block_index + ndocc ** 2):
        o_orb = (col - block_index) // ndocc # Increase o_orb after every ndocc cols
        v_orb = (col - block_index) % ndocc + (SOMO2 + 1) # Increase v_orb then reset after ndocc cols
        xcish[0,col] = np.sqrt(1.5) * (rep_tens[o_orb, SOMO2, SOMO2, v_orb] - rep_tens[o_orb, SOMO1, SOMO1, v_orb])
        xcish[col,0] = xcish[0,col]

    
    #10 <ZW0|H|ZW0>
    xcish[1,1] = energy0 + orb_energies[SOMO1] - orb_energies[SOMO2] + 0.25 * j00 + 0.5 * k00 - rep_tens[SOMO1, SOMO1, SOMO2, SOMO2] + 0.5 * k00
    #11 <ZW0|H|ZW0'>
    xcish[1,2] = rep_tens[SOMO1,SOMO2,SOMO2,SOMO1]
    xcish[2,1] = xcish[1,2]
    #12 <ZW0|H|HS1>
    block_index = 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        xcish[1,col] = (2 ** 0.5) * (rep_tens[o_orb,SOMO2,SOMO1,SOMO1] - 0.5 * rep_tens[o_orb,SOMO1,SOMO1,SOMO2] - 0.5 * rep_tens[o_orb,SOMO2,SOMO2,SOMO2])
        xcish[col,1] = xcish[1,col]
    #13 <ZW0|H|HS2>
    block_index = ndocc + 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        xcish[1,col] = (-2 ** 0.5) * rep_tens[o_orb,SOMO2,SOMO2,SOMO1]
        xcish[col, 1] = xcish[1,col]
    #14 <ZW0|H|SL1>
    block_index = 2 * ndocc + 3
    for col in range(block_index, block_index + ndocc):
        v_orb = col - block_index + (SOMO2 + 1)
        xcish[1,col] = (2 ** 0.5) * (rep_tens[v_orb,SOMO1,SOMO2,SOMO2] - 0.5 * rep_tens[v_orb,SOMO1,SOMO1,SOMO1] - 0.5 * rep_tens[v_orb,SOMO2,SOMO2,SOMO1])
        xcish[col,1] = xcish[1,col]
    #15 <ZW0|H|SL2>
    block_index = 3 * ndocc + 3
    for col in range(block_index, block_index + ndocc):
        v_orb = col - block_index + (SOMO2 + 1)
        xcish[1,col] = (2 ** 0.5) * rep_tens[v_orb,SOMO1,SOMO1,SOMO2]
        xcish[col,1] = xcish[1,col]
    #16 <ZW0|H|HL1>
    block_index = 4 * ndocc + 3
    for col in range(block_index, block_index + ndocc ** 2):
        o_orb = (col - block_index) // ndocc
        v_orb = (col - block_index) % ndocc + (SOMO2 + 1)
        xcish[1,col] = 2 * rep_tens[o_orb, v_orb, SOMO1, SOMO2] - rep_tens[o_orb, SOMO2, SOMO1, v_orb]
        xcish[col,1] = xcish[1,col]
    #17 <ZW0|H|HL2>
    block_index = ndocc ** 2 + 4 * ndocc + 3
    for col in range(block_index, block_index + ndocc ** 2):
        o_orb = (col - block_index) // ndocc
        v_orb = (col - block_index) % ndocc + (SOMO2 + 1)
        xcish[1,col] = np.sqrt(3) * rep_tens[o_orb, SOMO2, SOMO1, v_orb]
        xcish[col,1] = xcish[1,col]
        
        
    #18 <ZW0'|H|ZW0'>
    xcish[2,2] = energy0 + orb_energies[SOMO2] - orb_energies[SOMO1] + 0.25 * j00 + 0.5 * k00 - rep_tens[SOMO1, SOMO1, SOMO2, SOMO2] + 0.5 * k00
    #19 <ZW0'|H|HS1>
    block_index = 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        xcish[2,col] = (-2 ** 0.5) * rep_tens[o_orb,SOMO1,SOMO1,SOMO2]
        xcish[col,2] = xcish[2,col]
    #20 <ZW0'|H|HS2>
    block_index = ndocc + 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        xcish[2,col] = (2 ** 0.5) * (rep_tens[o_orb,SOMO1,SOMO2,SOMO2] - 0.5 * rep_tens[o_orb,SOMO2,SOMO2,SOMO1] - 0.5 * rep_tens[o_orb,SOMO1,SOMO1,SOMO1]) # CHECK SIGN
        xcish[col,2] = xcish[2,col]
    #21 <ZW0'|H|SL1>
    block_index = 2 * ndocc + 3
    for col in range(block_index, block_index + ndocc):
        v_orb = col - block_index + (SOMO2 + 1)
        xcish[2,col] = (-2 ** 0.5) * rep_tens[v_orb,SOMO2,SOMO2,SOMO1]
        xcish[col,2] = xcish[2,col]
    #22 <ZW0'|H|SL2>
    block_index = 3 * ndocc + 3
    for col in range(block_index, block_index + ndocc):
        v_orb = col - block_index + (SOMO2 + 1)
        xcish[2,col] = (2 ** 0.5) * (0.5 * rep_tens[v_orb,SOMO2,SOMO2,SOMO2] + 0.5 * rep_tens[v_orb,SOMO1,SOMO1,SOMO2] - rep_tens[v_orb,SOMO2,SOMO1,SOMO1]) # CHECK SIGN
        xcish[col,2] = xcish[2,col]
    #23 <ZW0'|H|HL1>
    block_index = 4 * ndocc + 3
    for col in range(block_index, block_index + ndocc ** 2):
        o_orb = (col - block_index) // ndocc
        v_orb = (col - block_index) % ndocc + (SOMO2 + 1)
        xcish[1,col] = 2 * rep_tens[o_orb, v_orb, SOMO1, SOMO2] - rep_tens[o_orb, SOMO1, SOMO2, v_orb]
        xcish[col,1] = xcish[1,col]
    #24 <ZW0'|H|HL2>
    block_index = ndocc ** 2 + 4 * ndocc + 3
    for col in range(block_index, block_index + ndocc ** 2):
        o_orb = (col - block_index) // ndocc
        v_orb = (col - block_index) % ndocc + (SOMO2 + 1)
        xcish[1,col] = - np.sqrt(3) * rep_tens[o_orb, SOMO1, SOMO2, v_orb]
        xcish[col,1] = xcish[1,col]
    
    
    row_block_index = 3
    #25 <HS1|H|HS1>
    col_block_index = 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(row, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                xcish[row, col] = energy0 + orb_energies[SOMO1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, SOMO1, SOMO1] + 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] \
                             + 1.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb1] + 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb1]
            else:    
                xcish[row, col] = 0.5 * (rep_tens[o_orb2, SOMO1, SOMO1, o_orb1] + 0.5 * rep_tens[o_orb2, SOMO2, SOMO2, o_orb1]) - rep_tens[o_orb2, o_orb1, SOMO1, SOMO1]
            xcish[col, row] = xcish[row,col]
    #26 <HS1|H|HS2>
    col_block_index = ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                xcish[row, col] = rep_tens[SOMO1, o_orb1, o_orb1, SOMO2] - rep_tens[o_orb1, o_orb1, SOMO1, SOMO2] + 0.5 * rep_tens[SOMO1, SOMO2, SOMO1, SOMO1] \
                             + 0.5 * rep_tens[SOMO1, SOMO2, SOMO2, SOMO2]
            else:    
                xcish[row, col] = - rep_tens[o_orb1, o_orb2, SOMO1, SOMO2] - rep_tens[o_orb1, SOMO1, SOMO2, o_orb2]
            xcish[col, row] = xcish[row, col]
    #27 <HS1|H|SL1>
    col_block_index = 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc):
            v_orb = col - col_block_index + (SOMO2 + 1)
            xcish[row, col] = rep_tens[o_orb, SOMO1, SOMO2, v_orb] - 2 * rep_tens[o_orb, SOMO2, SOMO1, v_orb]
            xcish[col, row] = xcish[row,col]
    #28 <HS1|H|SL2>
    col_block_index = 3 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc):
            v_orb = col - col_block_index + (SOMO2 + 1)
            xcish[row, col] = - rep_tens[o_orb, SOMO1, SOMO1, v_orb]
            xcish[col, row] = xcish[row,col]
    #29 <HS1|H|HL1>
    col_block_index = 4 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc ** 2):
            o_orb2 = (col - col_block_index) // ndocc
            v_orb = (col - col_block_index) % ndocc + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (rep_tens[o_orb1, o_orb1, SOMO1, v_orb] + 1.5 * rep_tens[v_orb, SOMO2, SOMO2, SOMO1] - 0.5 * rep_tens[v_orb, SOMO1, SOMO1, SOMO1] \
                                   - 2 * rep_tens[v_orb, o_orb1, o_orb1, SOMO1])
            else:
                xcish[row, col] = (1 / np.sqrt(2)) * (rep_tens[v_orb, SOMO1, o_orb1, o_orb2] - 2 * rep_tens[v_orb, o_orb2, o_orb1, SOMO1])
            xcish[col,row] = xcish[row,col]
    #30 <HS1|H|HL2>
    col_block_index = ndocc ** 2 + 4 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc ** 2):
            o_orb2 = (col - col_block_index) // ndocc
            v_orb = (col - col_block_index) % ndocc + (SOMO2 + 1)
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
                             + 1.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb1] + 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb1]
            else:    
                xcish[row, col] = 1.5 * (rep_tens[o_orb2, SOMO1, SOMO1, o_orb1] + rep_tens[o_orb2, o_orb1, SOMO2, SOMO2]) - 0.5 * rep_tens[o_orb2, SOMO2, SOMO2, o_orb1]
            xcish[col, row] = xcish[row,col]
    #32 <HS2|H|SL1>
    col_block_index = 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc):
            v_orb = col - col_block_index + (SOMO2 + 1)
            xcish[row, col] = - rep_tens[o_orb, SOMO2, SOMO2, v_orb]
            xcish[col, row] = xcish[row, col]
    #33 <HS2|H|SL2>
    col_block_index = 3 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc):
            v_orb = col - col_block_index + (SOMO2 + 1)
            xcish[row, col] = rep_tens[o_orb, SOMO2, SOMO1, v_orb] - 2 * rep_tens[o_orb, SOMO1, SOMO2, v_orb]
            xcish[col, row] = xcish[row,col]
    #34 <HS2|H|HL1>
    col_block_index = 4 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc ** 2):
            o_orb2 = (col - col_block_index) // ndocc
            v_orb = (col - col_block_index) % ndocc + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (2*rep_tens[v_orb,o_orb1,o_orb1,SOMO2] - rep_tens[o_orb1,o_orb1,SOMO2,v_orb] - 1.5*rep_tens[v_orb, SOMO1, SOMO1, SOMO2] + 0.5*rep_tens[v_orb,SOMO2,SOMO2,SOMO2])
            else:
                xcish[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[v_orb, o_orb2, o_orb1, SOMO2] - rep_tens[v_orb, SOMO2, o_orb1, o_orb2])
            xcish[col,row] = xcish[row,col]
    #35 <HS2|H|HL2>
    col_block_index = ndocc ** 2 + 4 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc ** 2):
            o_orb2 = (col - col_block_index) // ndocc
            v_orb = (col - col_block_index) % ndocc + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                xcish[row, col] = np.sqrt(1.5) * (0.5 * rep_tens[v_orb, SOMO2, SOMO2, SOMO2] + 0.5 * rep_tens[v_orb, SOMO1, SOMO1, SOMO2] - rep_tens[v_orb, SOMO2, o_orb1, o_orb1])
            else:
                xcish[row, col] = - np.sqrt(1.5) * (rep_tens[v_orb, SOMO2, o_orb1, o_orb2])
            xcish[col,row] = xcish[row,col]

    
    row_block_index = 2 * ndocc + 3
    #36 <SL1|H|SL1>
    col_block_index = 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(row, col_block_index + ndocc):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[SOMO2] - rep_tens[v_orb1, v_orb1, SOMO2, SOMO2] + 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] \
                             - 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] + 1.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb1] + 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb1]
            else:    
                xcish[row, col] = 1.5 * rep_tens[v_orb2, SOMO2, SOMO2, v_orb1] + 0.5 * rep_tens[v_orb2, SOMO1, SOMO1, v_orb1] -  rep_tens[v_orb2, v_orb1, SOMO2, SOMO2]
            xcish[col, row] = xcish[row,col]
    #37 <SL1|H|SL2>
    col_block_index = 3 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + ndocc):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = 0.5 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO2] + 0.5 * rep_tens[SOMO1, SOMO2, SOMO2, SOMO2] - rep_tens[v_orb1, v_orb1, SOMO1, SOMO2] - rep_tens[v_orb1, SOMO1, SOMO2, v_orb1]
            else:    
                xcish[row, col] = - rep_tens[v_orb1, v_orb2, SOMO1, SOMO2] - rep_tens[v_orb1, SOMO2, SOMO1, v_orb2]
            xcish[col, row] = xcish[row,col]
    #38 <SL1|H|HL1>
    col_block_index = 4 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + ndocc ** 2):
            o_orb = (col - col_block_index) // ndocc
            v_orb2 = (col - col_block_index) % ndocc + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (rep_tens[o_orb,SOMO2,v_orb1,v_orb1] + 1.5*rep_tens[o_orb,SOMO1,SOMO1,SOMO2] - 2*rep_tens[o_orb,v_orb1,v_orb1,SOMO2] - 0.5*rep_tens[o_orb,SOMO2,SOMO2,SOMO2])
            else:
                xcish[row, col] = (1 / np.sqrt(2)) * (rep_tens[o_orb, SOMO2, v_orb1, v_orb2] - 2 * rep_tens[v_orb1, SOMO2, v_orb2, o_orb])
            xcish[col,row] = xcish[row,col]
    #39 <SL1|H|HL2>
    col_block_index = ndocc ** 2 + 4 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + ndocc ** 2):
            o_orb = (col - col_block_index) // ndocc
            v_orb2 = (col - col_block_index) % ndocc + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = np.sqrt(1.5) * (0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO2] + 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO2] - rep_tens[o_orb, SOMO2, v_orb1, v_orb1])
            else:
                xcish[row, col] = - np.sqrt(1.5) * (rep_tens[o_orb, SOMO2, v_orb1, v_orb2])
            xcish[col,row] = xcish[row,col]
    
    
    row_block_index = 3 * ndocc + 3
    #40 <SL2|H|SL2>
    col_block_index = 3 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(row, col_block_index + ndocc):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[SOMO1] - rep_tens[v_orb1, v_orb1, SOMO1, SOMO1] + 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] \
                             - 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2]+ 1.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb1] + 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb1]
            else:    
                xcish[row, col] = 1.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb2] - 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] - rep_tens[v_orb1, v_orb2, SOMO1, SOMO1] - rep_tens[v_orb1, v_orb2, SOMO2, SOMO2]
            xcish[col, row] = xcish[row,col]
    #41 <SL2|H|HL1>
    col_block_index = 4 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + ndocc ** 2):
            o_orb = (col - col_block_index) // ndocc
            v_orb2 = (col - col_block_index) % ndocc + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[o_orb, v_orb1, v_orb1, SOMO1] + 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO1] - rep_tens[o_orb, SOMO1, v_orb1, v_orb1] - 1.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO1])
            else:
                xcish[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[v_orb1, SOMO1, v_orb2, o_orb] - rep_tens[o_orb, SOMO1, v_orb1, v_orb2])
            xcish[col,row] = xcish[row,col]
    #42 <SL2|H|HL2>
    col_block_index = ndocc ** 2 + 4 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + ndocc ** 2):
            o_orb = (col - col_block_index) // ndocc
            v_orb2 = (col - col_block_index) % ndocc + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = np.sqrt(1.5) * (0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO1] + 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO1] - rep_tens[o_orb, SOMO1, v_orb1, v_orb1])
            else:
                xcish[row, col] = - np.sqrt(1.5) * (rep_tens[o_orb, SOMO1, v_orb1, v_orb2])
            xcish[col,row] = xcish[row,col]
    
    
    row_block_index = 4 * ndocc + 3
    #43 <HL1|H|HL1>
    col_block_index = 4 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc ** 2):
        o_orb1 = (row - row_block_index) // ndocc
        v_orb1 = (row - row_block_index) % ndocc + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + ndocc ** 2):
            o_orb2 = (col - col_block_index) // ndocc
            v_orb2 = (col - col_block_index) % ndocc + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                xcish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb1] - 0.25 * j00 + 1.5 * k00 \
                                 + 2 * rep_tens[o_orb1, v_orb1, v_orb1, o_orb1]
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                xcish[row, col] =  2 * rep_tens[o_orb1, v_orb1, v_orb1, o_orb2] - rep_tens[o_orb1, o_orb2, v_orb1, v_orb1]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                xcish[row, col] = rep_tens[v_orb1, o_orb1, o_orb1, v_orb2] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb2]
            else:
                xcish[row, col] = 2 * rep_tens[o_orb1, v_orb1, o_orb2, v_orb2] - rep_tens[o_orb1, o_orb2, v_orb1, v_orb2]
            xcish[col, row] = xcish[row,col]
    #44 <HL1|H|HL2>
    col_block_index = ndocc ** 2 + 4 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc ** 2):
        o_orb1 = (row - row_block_index) // ndocc
        v_orb1 = (row - row_block_index) % ndocc + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + ndocc ** 2):
            o_orb2 = (col - col_block_index) // ndocc
            v_orb2 = (col - col_block_index) % ndocc + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                xcish[row, col] = (np.sqrt(3) / 2) * (rep_tens[o_orb1,SOMO1,SOMO1,o_orb1] - rep_tens[o_orb1,SOMO2,SOMO2,o_orb1] + rep_tens[v_orb1,SOMO2,SOMO2,v_orb1] - rep_tens[v_orb1,SOMO1,SOMO1,v_orb1])
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                xcish[row, col] =  (np.sqrt(3) / 2) * (rep_tens[o_orb1, SOMO1, SOMO1, o_orb2] - rep_tens[o_orb1, SOMO2, SOMO2, o_orb2])
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                xcish[row, col] =  (np.sqrt(3) / 2) * (rep_tens[v_orb1, SOMO2, SOMO2, v_orb2] - rep_tens[v_orb1, SOMO1, SOMO1, v_orb2])
            xcish[col, row] = xcish[row,col]
    
    row_block_index = ndocc ** 2 + 4 * ndocc + 3
    #45 <HL2|H|HL2>
    col_block_index = ndocc ** 2 + 4 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc ** 2):
        o_orb1 = (row - row_block_index) // ndocc
        v_orb1 = (row - row_block_index) % ndocc + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + ndocc ** 2):
            o_orb2 = (col - col_block_index) // ndocc
            v_orb2 = (col - col_block_index) % ndocc + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                xcish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb1] - 0.25 * j00 - 0.5 * k00 \
                                 + rep_tens[o_orb1, SOMO1, SOMO1, o_orb1] + rep_tens[o_orb1, SOMO2, SOMO2, o_orb1] + rep_tens[v_orb1, SOMO1, SOMO1, v_orb1] + rep_tens[v_orb2, SOMO2, SOMO2, v_orb2]
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                xcish[row, col] = rep_tens[o_orb1, SOMO1, SOMO1, o_orb2] + rep_tens[o_orb1, SOMO2, SOMO2, o_orb2] - rep_tens[o_orb1, o_orb2, v_orb1, v_orb1]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                xcish[row, col] = rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] + rep_tens[v_orb1, SOMO2, SOMO2, v_orb2] - rep_tens[v_orb1, v_orb2, o_orb1, o_orb1]
            else:
                xcish[row, col] = - rep_tens[o_orb1, o_orb2, v_orb1, v_orb2]
            xcish[col, row] = xcish[row,col]
    
    
    ################# TRIPLET BLOCK ######################
    
    row_index = 2 * ndocc ** 2 + 4 * ndocc + 3
    #46 <OS3|H|OS3>
    xcish[row_index, row_index] = energy0 - (0.25 * j00 ) - (0.5 * k00)
    #47 <OS3|H|HS1>
    col_index = 2 * ndocc ** 2 + 4 * ndocc + 4
    for col in range(col_index, col_index + ndocc):
        o_orb = col - col_index
        xcish[row_index, col] = 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO1] + 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO1]
        xcish[col, row_index] = xcish[row_index,col]
    #48 <OS3|H|HS2>
    col_index = 2 * ndocc ** 2 + 5 * ndocc + 4
    for col in range(col_index, col_index + ndocc):
        o_orb = col - col_index
        xcish[row_index, col] = 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO2] + 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO2]
        xcish[col, row_index] = xcish[row_index,col]
    #49 <OS3|H|SL1>
    col_index = 2 * ndocc ** 2 + 6 * ndocc + 4
    for col in range(col_index, col_index + ndocc):
        v_orb = col - col_index + (SOMO2 + 1)
        xcish[row_index, col] = 0.5 * rep_tens[v_orb, SOMO1, SOMO2, SOMO1] - 0.5 * rep_tens[v_orb, SOMO2, SOMO2, SOMO2]
        xcish[col, row_index] = xcish[row_index,col]
    #50 <OS3|H|SL2>
    col_index = 2 * ndocc ** 2 + 7 * ndocc + 4
    for col in range(col_index, col_index + ndocc):
        v_orb = col - col_index + (SOMO2 + 1)
        xcish[row_index, col] = 0.5 * rep_tens[v_orb, SOMO1, SOMO1, SOMO1] - 0.5 * rep_tens[v_orb, SOMO2, SOMO2, SOMO1]
        xcish[col, row_index] = xcish[row_index, col]
    #51 <OS3|H|HL1> = 0
    #52 <OS3|H|HL2>
    col_index =  3 * ndocc ** 2 + 8 * ndocc + 4
    for col in range(block_index, block_index + ndocc ** 2):
        o_orb = (col - block_index) // ndocc # Increase o_orb after every ndocc cols
        v_orb = (col - block_index) % ndocc + (SOMO2 + 1) # Increase v_orb then reset after ndocc cols
        xcish[0,col] = (1 / np.sqrt(2)) * (rep_tens[o_orb, SOMO1, SOMO1, v_orb] - rep_tens[o_orb, SOMO2, SOMO2, v_orb])
        xcish[col,0] = xcish[0,col]
    #53 <OS3|H|HL3>
    col_index =  4 * ndocc ** 2 + 8 * ndocc + 4
    for col in range(block_index, block_index + ndocc ** 2):
        o_orb = (col - block_index) // ndocc # Increase o_orb after every ndocc cols
        v_orb = (col - block_index) % ndocc + (SOMO2 + 1) # Increase v_orb then reset after ndocc cols
        xcish[0,col] = rep_tens[o_orb, SOMO1, SOMO1, v_orb] + rep_tens[o_orb, SOMO2, SOMO2, v_orb]
        xcish[col,0] = xcish[0,col]
    
    row_block_index = 2 * ndocc ** 2 + 4 * ndocc + 4
    #54 <HS1|H|HS1>
    col_block_index = 2 * ndocc ** 2 + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(row, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                xcish[row, col] = energy0 + orb_energies[SOMO1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, SOMO1, SOMO1] + 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] \
                             - 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb1] + 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb1]
            else:    
                xcish[row, col] = 0.5 * (rep_tens[o_orb2, SOMO1, SOMO1, o_orb1] - rep_tens[o_orb2, o_orb1, SOMO1, SOMO1]) - 1.5 * rep_tens[o_orb2, SOMO2, SOMO2, o_orb1]
            xcish[col, row] = xcish[row,col]
    #55 <HS1|H|HS2>
    col_block_index = 2 * ndocc ** 2 + 5 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                xcish[row, col] = rep_tens[SOMO1, o_orb1, o_orb1, SOMO2] - rep_tens[o_orb1, o_orb1, SOMO1, SOMO2] + 0.5 * rep_tens[SOMO2, SOMO1, SOMO1, SOMO1] \
                             + 0.5 * rep_tens[SOMO1, SOMO2, SOMO2, SOMO2]                                                               #CHECK RESULT, SAME AS SINGLET?
            else:    
                xcish[row, col] = rep_tens[o_orb2, o_orb1, SOMO1, SOMO2] - rep_tens[o_orb2, SOMO1, SOMO2, o_orb1]
            xcish[col, row] = xcish[row,col]
    #56 <HS1|H|SL1>
    col_block_index = 2 * ndocc ** 2 + 6 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc):
            v_orb = col - col_block_index + (SOMO2 + 1)
            xcish[row, col] = - rep_tens[o_orb, SOMO1, SOMO2, v_orb] # CHECK SIGN
            xcish[col, row] = xcish[row,col]
    #57 <HS1|H|SL2>
    col_block_index = 2 * ndocc ** 2 + 7 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc):
            v_orb = col - col_block_index + (SOMO2 + 1)
            xcish[row, col] = rep_tens[o_orb, SOMO1, SOMO1, v_orb] # CHECK SIGN
            xcish[col, row] = xcish[row,col]
    #58 <HS1|H|HL1>
    col_block_index = 2 * ndocc ** 2 + 8 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc ** 2):
            o_orb2 = (col - col_block_index) // ndocc
            v_orb = (col - col_block_index) % ndocc + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[v_orb, o_orb1, o_orb1, SOMO1] + 0.5 * rep_tens[v_orb, SOMO2, SOMO2, SOMO1] + 0.5 * rep_tens[v_orb, SOMO1, SOMO1, SOMO1] \
                                   - rep_tens[o_orb1, o_orb1, SOMO1, v_orb])
            else:
                xcish[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[v_orb, o_orb2, o_orb1, SOMO1] - rep_tens[v_orb, SOMO1, o_orb1, o_orb2])
            xcish[col,row] = xcish[row,col]
    #59 <HS1|H|HL2>
    col_block_index = 3 * ndocc ** 2 + 8 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc ** 2):
            o_orb2 = (col - col_block_index) // ndocc
            v_orb = (col - col_block_index) % ndocc + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (0.5 * rep_tens[v_orb, SOMO1, SOMO1, SOMO1] - 1.5 * rep_tens[v_orb, SOMO2, SOMO2, SOMO1] - rep_tens[v_orb, SOMO1, o_orb1, o_orb1])
            else:
                xcish[row, col] = - (1 / np.sqrt(2)) * (rep_tens[v_orb, SOMO1, o_orb1, o_orb2])
            xcish[col,row] = xcish[row,col]
    #60 <HS1|H|HL3>
    col_block_index = 4 * ndocc ** 2 + 8 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc ** 2):
            o_orb2 = (col - col_block_index) // ndocc
            v_orb = (col - col_block_index) % ndocc + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                xcish[row, col] = 0.5 * rep_tens[v_orb, SOMO1, SOMO1, SOMO1] + 0.5 * rep_tens[v_orb, SOMO2, SOMO2, SOMO1] - rep_tens[v_orb, SOMO1, o_orb1, o_orb1]
            else:
                xcish[row, col] = - rep_tens[v_orb, SOMO1, o_orb1, o_orb2]
            xcish[col,row] = xcish[row,col]

    
    row_block_index = 2 * ndocc ** 2 + 5 * ndocc + 4
    #61 <HS2|H|HS2>
    col_block_index = 2 * ndocc ** 2 + 5 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(row, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:

                xcish[row, col] = energy0 + orb_energies[SOMO2] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, SOMO2, SOMO2] + 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] \
                             - 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb1] + 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb1]
            else:
                xcish[row, col] = rep_tens[o_orb1, o_orb2, SOMO2, SOMO2] - 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb2] - 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb2]
            xcish[col, row] = xcish[row,col]
    #62 <HS2|H|SL1>
    col_block_index = 2 * ndocc ** 2 + 6 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc):
            v_orb = col - col_block_index + (SOMO2 + 1)
            xcish[row, col] = - rep_tens[o_orb, SOMO2, SOMO2, v_orb]
            xcish[col, row] = xcish[row,col]
    #63 <HS2|H|SL2>
    col_block_index = 2 * ndocc ** 2 + 7 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc):
            v_orb = col - col_block_index + (SOMO2 + 1)
            xcish[row, col] = rep_tens[o_orb, SOMO2, SOMO1, v_orb]
            xcish[col, row] = xcish[row,col]
    #64 <HS2|H|HL1>
    col_block_index = 2 * ndocc ** 2 + 8 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc ** 2):
            o_orb2 = (col - col_block_index) // ndocc
            v_orb = (col - col_block_index) % ndocc + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[v_orb, o_orb1, o_orb1, SOMO2] + 0.5 * rep_tens[v_orb, SOMO1, SOMO1, SOMO2] + 0.5 * rep_tens[v_orb, SOMO2, SOMO2, SOMO2] \
                                   - rep_tens[o_orb1, o_orb1, SOMO2, v_orb])
            else:
                xcish[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[v_orb, o_orb2, o_orb1, SOMO2] - rep_tens[v_orb, SOMO2, o_orb1, o_orb2])
            xcish[col,row] = xcish[row,col]
    #65 <HS2|H|HL2>
    col_block_index = 3 * ndocc ** 2 + 8 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc ** 2):
            o_orb2 = (col - col_block_index) // ndocc
            v_orb = (col - col_block_index) % ndocc + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (1.5 * rep_tens[v_orb, SOMO1, SOMO1, SOMO2] + rep_tens[v_orb, SOMO2, o_orb1, o_orb1] - 0.5 * rep_tens[v_orb, SOMO2, SOMO2, SOMO2])
            else:
                xcish[row, col] = (1 / np.sqrt(2)) * (rep_tens[v_orb, SOMO2, o_orb1, o_orb2])
            xcish[col,row] = xcish[row,col]
    #66 <HS2|H|HL3>
    col_block_index = 4 * ndocc ** 2 + 8 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc ** 2):
            o_orb2 = (col - col_block_index) // ndocc
            v_orb = (col - col_block_index) % ndocc + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                xcish[row, col] = 0.5 * rep_tens[v_orb, SOMO2, SOMO2, SOMO2] + 0.5 * rep_tens[v_orb, SOMO1, SOMO1, SOMO2] - rep_tens[v_orb, SOMO2, o_orb1, o_orb1]
            else:
                xcish[row, col] = - rep_tens[v_orb, SOMO2, o_orb1, o_orb2]
            xcish[col,row] = xcish[row,col]
    
            
    row_block_index = 2 * ndocc ** 2 + 6 * ndocc + 4
    #67 <SL1|H|SL1>
    col_block_index =  2 * ndocc ** 2 + 6 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(row, col_block_index + ndocc):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[SOMO2] - rep_tens[v_orb1, v_orb1, SOMO2, SOMO2] + 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] \
                             - 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] - 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb1] + 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb1]
            else:    
                xcish[row, col] =  0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] - rep_tens[v_orb1, v_orb2, SOMO2, SOMO2] - 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb2]
            xcish[col, row] = xcish[row,col]
    #68 <SL1|H|SL2>
    col_block_index = 2 * ndocc ** 2 + 7 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + ndocc):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = rep_tens[v_orb1, v_orb1, SOMO1, SOMO2] - rep_tens[v_orb1, SOMO1, SOMO2, v_orb1] - 0.5 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO2] - 0.5 * rep_tens[SOMO1, SOMO2, SOMO2, SOMO2]
            else:    
                xcish[row, col] = rep_tens[v_orb1, v_orb2, SOMO1, SOMO2] - rep_tens[v_orb1, SOMO2, SOMO1, v_orb2]
            xcish[col, row] = xcish[row,col]
    #69 <SL1|H|HL1>
    col_block_index = 2 * ndocc ** 2 + 8 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + ndocc ** 2):
            o_orb = (col - col_block_index) // ndocc
            v_orb2 = (col - col_block_index) % ndocc + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (rep_tens[o_orb, SOMO2, v_orb1, v_orb1] - 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO2] - 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO2] \
                                  - 2 * rep_tens[o_orb, v_orb1, v_orb1, SOMO2])
            else:
                xcish[row, col] = (1 / np.sqrt(2)) * (rep_tens[o_orb, SOMO2, v_orb1, v_orb2] - 2 * rep_tens[v_orb1, SOMO2, v_orb2, o_orb])
            xcish[col,row] = xcish[row,col]
    #70 <SL1|H|HL2>
    col_block_index = 3 * ndocc ** 2 + 8 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + ndocc ** 2):
            o_orb = (col - col_block_index) // ndocc
            v_orb2 = (col - col_block_index) % ndocc + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (rep_tens[o_orb, SOMO2, v_orb1, v_orb1] + 1.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO2] - 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO2])
            else:
                xcish[row, col] = - (1 / np.sqrt(2)) * (rep_tens[o_orb, SOMO2, v_orb1, v_orb2])
            xcish[col,row] = xcish[row,col]
    #71 <SL1|H|HL3>
    col_block_index = 4 * ndocc ** 2 + 8 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + ndocc ** 2):
            o_orb = (col - col_block_index) // ndocc
            v_orb2 = (col - col_block_index) % ndocc + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO2] + 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO2] - rep_tens[o_orb, SOMO2, v_orb1, v_orb1]
            else:
                xcish[row, col] = - rep_tens[o_orb, SOMO2, v_orb1, v_orb2]
            xcish[col,row] = xcish[row,col]
    
    
    row_block_index = 2 * ndocc ** 2 + 7 * ndocc + 4
    #72 <SL2|H|SL2>
    col_block_index = 2 * ndocc ** 2 + 7 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(row, col_block_index + ndocc):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[SOMO1] - rep_tens[v_orb1, v_orb1, SOMO1, SOMO1] + 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] \
                             - 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] - 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb1] + 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb1]
            else:    
                xcish[row, col] =  0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] - 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb2] - rep_tens[v_orb1, v_orb2, SOMO1, SOMO1] \
                                  - rep_tens[v_orb1, v_orb2, SOMO2, SOMO2]
            xcish[col, row] = xcish[row,col]
    #73 <SL2|H|HL1>
    col_block_index = 2 * ndocc ** 2 + 8 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + ndocc ** 2):
            o_orb = (col - col_block_index) // ndocc
            v_orb2 = (col - col_block_index) % ndocc + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[o_orb, v_orb1, v_orb1, SOMO1] + 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO1] + 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO1] \
                                  - rep_tens[o_orb, SOMO1, v_orb1, v_orb1])
            else:
                xcish[row, col] = (1 / np.sqrt(2)) * (2 * rep_tens[v_orb1, SOMO1, v_orb2, o_orb] - rep_tens[o_orb, SOMO1, v_orb1, v_orb2])
            xcish[col,row] = xcish[row,col]
    #74 <SL2|H|HL2>
    col_block_index = 3 * ndocc ** 2 + 8 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + ndocc ** 2):
            o_orb = (col - col_block_index) // ndocc
            v_orb2 = (col - col_block_index) % ndocc + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (rep_tens[o_orb, SOMO1, v_orb1, v_orb1] + 1.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO1] - 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO1])
            else:
                xcish[row, col] = - (1 / np.sqrt(2)) * (rep_tens[o_orb, SOMO1, v_orb1, v_orb2])
            xcish[col,row] = xcish[row,col]
    #75 <SL2|H|HL3>
    col_block_index = 4 * ndocc ** 2 + 8 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + ndocc ** 2):
            o_orb = (col - col_block_index) // ndocc
            v_orb2 = (col - col_block_index) % ndocc + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = rep_tens[o_orb, SOMO1, v_orb1, v_orb1] - 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO1] - 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO1]
            else:
                xcish[row, col] = rep_tens[o_orb, SOMO1, v_orb1, v_orb2]
            xcish[col,row] = xcish[row,col]
    
    
    row_block_index = 2 * ndocc ** 2 + 8 * ndocc + 4
    #76 <HL1|H|HL1>
    col_block_index = 2 * ndocc ** 2 + 8 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc ** 2):
        o_orb1 = (row - row_block_index) // ndocc
        v_orb1 = (row - row_block_index) % ndocc + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + ndocc ** 2):
            o_orb2 = (col - col_block_index) // ndocc
            v_orb2 = (col - col_block_index) % ndocc + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                xcish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb1] - 0.25 * j00 - 0.5 * k00 \
                                 + 2 * rep_tens[o_orb1, v_orb1, v_orb1, o_orb1]
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                xcish[row, col] =  2 * rep_tens[o_orb1, v_orb1, v_orb1, o_orb2] - rep_tens[o_orb1, o_orb2, v_orb1, v_orb1]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                xcish[row, col] = rep_tens[v_orb1, o_orb1, o_orb1, v_orb2] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb2]
            else:
                xcish[row, col] = 2 * rep_tens[o_orb1, v_orb1, o_orb2, v_orb2] - rep_tens[o_orb1, o_orb2, v_orb1, v_orb2]
            xcish[col, row] = xcish[row,col]
    #77 <HL1|H|HL2>
    col_block_index = 3 * ndocc ** 2 + 8 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc ** 2):
        o_orb1 = (row - row_block_index) // ndocc
        v_orb1 = (row - row_block_index) % ndocc + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + ndocc ** 2):
            o_orb2 = (col - col_block_index) // ndocc
            v_orb2 = (col - col_block_index) % ndocc + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                xcish[row, col] = 0.5 * (rep_tens[o_orb1,SOMO2,SOMO2,o_orb1] - rep_tens[o_orb1,SOMO1,SOMO1,o_orb1] + rep_tens[v_orb1,SOMO1,SOMO1,v_orb1] - rep_tens[v_orb1,SOMO2,SOMO2,v_orb1])
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                xcish[row, col] =  0.5 * (rep_tens[o_orb1, SOMO2, SOMO2, o_orb2] - rep_tens[o_orb1, SOMO1, SOMO1, o_orb2])
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                xcish[row, col] =  0.5 * (rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] - rep_tens[v_orb1, SOMO2, SOMO2, v_orb2])
            xcish[col, row] = xcish[row,col]
    #78 <HL1|H|HL3>
    col_block_index = 4 * ndocc ** 2 + 8 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc ** 2):
        o_orb1 = (row - row_block_index) // ndocc
        v_orb1 = (row - row_block_index) % ndocc + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + ndocc ** 2):
            o_orb2 = (col - col_block_index) // ndocc
            v_orb2 = (col - col_block_index) % ndocc + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (rep_tens[v_orb1,SOMO1,SOMO1,v_orb1] + rep_tens[v_orb1,SOMO2,SOMO2,v_orb1] - rep_tens[o_orb1,SOMO1,SOMO1,o_orb1] - rep_tens[o_orb1,SOMO2,SOMO2,o_orb1])
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                xcish[row, col] =  - (1 / np.sqrt(2)) * (rep_tens[o_orb1, SOMO1, SOMO1, o_orb2] + rep_tens[o_orb1, SOMO2, SOMO2, o_orb2])
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                xcish[row, col] =  (1 / np.sqrt(2)) * (rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] + rep_tens[v_orb1, SOMO2, SOMO2, v_orb2])
            xcish[col, row] = xcish[row,col]
    
    
    row_block_index = 3 * ndocc ** 2 + 8 * ndocc + 4
    #79 <HL2|H|HL2>
    col_block_index = 3 * ndocc ** 2 + 8 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc ** 2):
        o_orb1 = (row - row_block_index) // ndocc
        v_orb1 = (row - row_block_index) % ndocc + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + ndocc ** 2):
            o_orb2 = (col - col_block_index) // ndocc
            v_orb2 = (col - col_block_index) % ndocc + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                xcish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb1] - 0.25 * j00 + 1.5 * k00
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                xcish[row, col] =  - rep_tens[o_orb1, o_orb2, v_orb1, v_orb1]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                xcish[row, col] = - rep_tens[v_orb1, o_orb1, o_orb1, v_orb2] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb2]
            else:
                xcish[row, col] = - rep_tens[o_orb1, o_orb2, v_orb1, v_orb2]
            xcish[col, row] = xcish[row,col]
    #80 <HL2|H|HL3>
    col_block_index = 4 * ndocc ** 2 + 8 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc ** 2):
        o_orb1 = (row - row_block_index) // ndocc
        v_orb1 = (row - row_block_index) % ndocc + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + ndocc ** 2):
            o_orb2 = (col - col_block_index) // ndocc
            v_orb2 = (col - col_block_index) % ndocc + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                xcish[row, col] = (1 / np.sqrt(2)) * (rep_tens[o_orb1,SOMO1,SOMO1,o_orb1] - rep_tens[o_orb1,SOMO2,SOMO2,o_orb1] + rep_tens[v_orb1,SOMO1,SOMO1,v_orb1] - rep_tens[v_orb1,SOMO2,SOMO2,v_orb1])
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                xcish[row, col] =  (1 / np.sqrt(2)) * (rep_tens[o_orb1, SOMO1, SOMO1, o_orb2] - rep_tens[o_orb1, SOMO2, SOMO2, o_orb2])
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                xcish[row, col] =  (1 / np.sqrt(2)) * (rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] - rep_tens[v_orb1, SOMO2, SOMO2, v_orb2])
            xcish[col, row] = xcish[row,col]
    
    row_block_index = 4 * ndocc ** 2 + 8 * ndocc + 4
    #81 <HL2|H|HL2>
    col_block_index = 4 * ndocc ** 2 + 8 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc ** 2):
        o_orb1 = (row - row_block_index) // ndocc
        v_orb1 = (row - row_block_index) % ndocc + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + ndocc ** 2):
            o_orb2 = (col - col_block_index) // ndocc
            v_orb2 = (col - col_block_index) % ndocc + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                xcish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb1] - 0.25 * j00 - 0.5 * k00 \
                                  + 0.5 * (rep_tens[o_orb1, SOMO1, SOMO1, o_orb1] + rep_tens[o_orb1, SOMO2, SOMO2, o_orb1] + rep_tens[v_orb1, SOMO1, SOMO1, v_orb1] + rep_tens[v_orb1, SOMO2, SOMO2, v_orb1])
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                xcish[row, col] =  0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb2] + 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb2] - rep_tens[o_orb1, o_orb2, v_orb1, v_orb1]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                xcish[row, col] = 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] + 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb2] - rep_tens[v_orb1, v_orb2, o_orb1, o_orb1]
            else:
                xcish[row, col] = - rep_tens[o_orb1, o_orb2, v_orb1, v_orb2]
            xcish[col, row] = xcish[row,col]
    
            
    ################# QUINTET STATE ##################
    
    row_block_index = 5 * ndocc ** 2 + 8 * ndocc + 4
    #82 <5Q|H|5Q>
    col_block_index =  5 * ndocc ** 2 + 8 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc ** 2):
        o_orb1 = (row - row_block_index) // ndocc # Increase o_orb after every ndocc rows
        v_orb1 = (row - row_block_index) % ndocc + (SOMO2 + 1) # Increase v_orb for every column and reset after ndocc rows
        for col in range(col_block_index, col_block_index + ndocc ** 2):
            o_orb2 = (col - col_block_index) // ndocc
            v_orb2 = (col - col_block_index) % ndocc + (SOMO2 + 1)
            if o_orb1 == o_orb2 and v_orb1 == v_orb2:
                xcish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb1] - 0.25 * j00 - 0.5 * k00 \
                                 - 0.5 * (rep_tens[o_orb1, SOMO1, SOMO1, o_orb1] + rep_tens[o_orb1, SOMO2, SOMO2, o_orb1] + rep_tens[v_orb1, SOMO1, SOMO1, v_orb1] + rep_tens[v_orb1, SOMO2, SOMO2, v_orb1])
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                xcish[row, col] =  - rep_tens[o_orb1, o_orb2, v_orb1, v_orb1] - 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb2] - 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb2]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                xcish[row, col] = - rep_tens[v_orb1, v_orb2, o_orb1, o_orb1] - 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] - 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb2]
            else:
                xcish[row, col] = - rep_tens[o_orb1, o_orb2, v_orb1, v_orb2]
            xcish[col, row] = xcish[row,col]
    
    
    return xcish


    
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



def dipole_cis(coords,atoms,norbs,hf_orbs,ndocc,nstates):
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
    dipoles[0,0,:] -= (dip1el[SOMO1,SOMO1,:] + dip1el[SOMO2,SOMO2,:])
    #2 <OS1|mu|ZW0> 
    dipoles[0,1,:] = (-2 ** 0.5) * dip1el[SOMO1,SOMO2,:]
    dipoles[1,0,:] = dipoles[0,1,:] 
    #3 <OS1|mu|ZW0'> 
    dipoles[0,2,:] = (-2 ** 0.5) * dip1el[SOMO1,SOMO2,:]
    dipoles[2,0,:] = dipoles[0,1,:] 
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
        dipoles[0,col,:] = dip1el[v_orb, SOMO2, :]
        dipoles[col,0,:] = dipoles[0,col,:]
    #7 <OS1|mu|SL2> 
    block_index = nvirt + 2 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        dipoles[0,col,:] = -dip1el[v_orb, SOMO1, :]
        dipoles[col,0,:] = dipoles[0,col,:]
    
    #8 <ZW0|mu|ZW0>
    for o in range(ndocc + 1):
        dipoles[1,1,:] -= 2*dip1el[o,o,:]
    #9 <ZW0|mu|ZW0'> = 0
    #10 <ZW0|mu|HS1>
    block_index = 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        dipoles[1,col,:] = (-2 ** 0.5) * dip1el[o_orb, SOMO2, :]
        dipoles[col,1,:] = dipoles[1,col,:]
    #11 <ZW0|mu|HS2> = 0
    #12 <ZW0|mu|SL1> 
    block_index = 2 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        dipoles[1,col,:] = (2 ** 0.5) * dip1el[v_orb, SOMO1, :]
        dipoles[col,1,:] = dipoles[1,col,:]
    #13 <ZW0|mu|SL2> = 0 
    
    #14 <ZW0'|mu|ZW0'>
    for o in range(ndocc):
        dipoles[2,2,:] -= 2*dip1el[o,o,:]
    dipoles[2,2,:] -= 2 * dip1el[SOMO2,SOMO2,:]
    #15 <ZW0'|mu|HS1> = 0
    #16 <ZW0'|mu|HS2>
    block_index = ndocc + 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        dipoles[2,col,:] = (2 ** 0.5) * dip1el[o_orb, SOMO1, :]
        dipoles[col,2,:] = dipoles[2,col,:]
    #17 <ZW0'|mu|SL1> = 0
    #18 <ZW0'|mu|SL2>
    block_index = nvirt + 2 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        dipoles[2,col,:] = (- 2 ** 0.5) * dip1el[v_orb, SOMO2, :]
        dipoles[col,2,:] = dipoles[2,col,:]
    
    row_block_index = 3
    #19 <HS1|mu|HS1>
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
    #20 <HS1|mu|HS2>
    col_block_index = ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        dipoles[row,row,:] = -dip1el[SOMO1, SOMO2, :] #Only diagonal elements are non-zero
    #21 <HS1|mu|SL1> = 0
    #22 <HS1|mu|SL2> = 0

    row_block_index = ndocc + 3
    #23 <HS2|mu|HS2>
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
    #24 <HS2|mu|SL1> = 0
    #25 <HS2|mu|SL2> = 0

    row_block_index = 2 * ndocc + 3
    #26 <SL1|mu|SL1>
    col_block_index = 2 * ndocc + 3
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
    #27 <SL1|mu|SL2>
    col_block_index = nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        dipoles[row,row,:] = dip1el[SOMO1,SOMO2,:]

    row_block_index = nvirt + 2 * ndocc + 3
    #28 <SL2|mu|SL2>
    col_block_index = nvirt + 2 * ndocc + 3
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

################# TRIPLET BLOCK ######################

    row_index = 2 * nvirt + 2 * ndocc + 3
    #29 <OS3|mu|OS3>
    for o in range(ndocc):
        dipoles[row_index,row_index,:] -= 2*dip1el[o,o,:]
    dipoles[row_index,row_index,:] -= (dip1el[SOMO1,SOMO1,:] + dip1el[SOMO2,SOMO2,:])
    #30 <OS3|mu|HS1>
    col_index = 2 * nvirt + 2 * ndocc + 4
    for col in range(col_index, col_index + ndocc):
        o_orb = col - col_index
        dipoles[row_index,col,:] = -dip1el[o_orb,SOMO1,:]
        dipoles[col,row_index,:] = dipoles[row_index,col,:]
    #31 <OS3|mu|HS2>
    col_index = 2 * nvirt + 3 * ndocc + 4
    for col in range(col_index, col_index + ndocc):
        o_orb = col - col_index
        dipoles[row_index,col,:] = -dip1el[o_orb,SOMO2,:]
        dipoles[col,row_index,:] = dipoles[row_index,col,:]
    #32 <OS3|mu|SL1>
    col_index = 2 * nvirt + 4 * ndocc + 4
    for col in range(col_index, col_index + nvirt):
        v_orb = col - col_index + (SOMO2 + 1)
        dipoles[row_index,col,:] = dip1el[v_orb,SOMO2,:]
        dipoles[col,row_index,:] = dipoles[row_index,col,:]
    #33 <OS3|mu|SL2>
    col_index = 3 * nvirt + 4 * ndocc + 4
    for col in range(col_index, col_index + nvirt):
        v_orb = col - col_index + (SOMO2 + 1)
        dipoles[row_index,col,:] = -dip1el[v_orb,SOMO1,:]
        dipoles[col,row_index,:] = dipoles[row_index,col,:]
        
    row_block_index = 2 * nvirt + 2 * ndocc + 4
    #34 <HS1|H|HS1>
    col_block_index = 2 * nvirt + 2 * ndocc + 4
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
    #35 <HS1|mu|HS2>
    col_block_index = 2 * nvirt + 3 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        dipoles[row,row,:] = -dip1el[SOMO1, SOMO2, :] #Only diagonal elements are non-zero
    #36 <HS1|mu|SL1> = 0
    #37 <HS1|mu|SL2> = 0

    row_block_index = 2 * nvirt + 3 * ndocc + 4
    #38 <HS2|mu|HS2>
    col_block_index = 2 * nvirt + 3 * ndocc + 4
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
    #39 <HS2|mu|SL1> = 0
    #40 <HS2|mu|SL2> = 0

    row_block_index = 2 * nvirt + 4 * ndocc + 4
    #41 <SL1|mu|SL1>
    col_block_index = 2 * nvirt + 4 * ndocc + 4
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
    #42 <SL1|mu|SL2>
    col_block_index = 3 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        dipoles[row,row,:] = -dip1el[SOMO1,SOMO2,:]
    
    row_block_index = 3 * nvirt + 4 * ndocc + 4
    #43 <SL2|H|SL2>
    col_block_index = 3 * nvirt + 4 * ndocc + 4
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
        
    #print("%10.5f"%linalg.norm(dipoles[:,:,0] - dipoles[:,:,0].T))  # checking symmetric
    #print("%10.5f"%linalg.norm(dipoles[:,:,1] - dipoles[:,:,1].T))
    #print("%10.5f"%linalg.norm(dipoles[:,:,2] - dipoles[:,:,2].T))   

    OS1_perm_dip=dipoles[0,0,:]
    ZW0_perm_dip=dipoles[1,1,:]
    ZW0p_perm_dip=dipoles[2,2,:]
    OS3_perm_dip=dipoles[2 * nvirt + 2 * ndocc + 3, 2 * nvirt + 2 * ndocc + 3, :]
    for dipole in [OS1_perm_dip, ZW0_perm_dip, ZW0p_perm_dip, OS3_perm_dip]:
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
            Zwitterion 0 (|ZW0>) = {ZW0_perm_dip[0]:.3f} x {ZW0_perm_dip[1]:.3f} y {ZW0_perm_dip[2]:.3f} z\n \
            Zwitterion 0' (|ZW0'>) = {ZW0p_perm_dip[0]:.3f} x {ZW0p_perm_dip[1]:.3f} y {ZW0p_perm_dip[2]:.3f} z\n \
            Open-Shell Triplet (|OS3>) = {OS3_perm_dip[0]:.3f} x {OS3_perm_dip[1]:.3f} y {OS3_perm_dip[2]:.3f} z\n")
    return dipoles

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
    #2 <OS1|mu|ZW0> 
    dipoles[0,1,:] = (-2 ** 0.5) * dip1el[SOMO1,SOMO2,:]
    dipoles[1,0,:] = dipoles[0,1,:] 
    #3 <OS1|mu|ZW0'> 
    dipoles[0,2,:] = (-2 ** 0.5) * dip1el[SOMO1,SOMO2,:]
    dipoles[2,0,:] = dipoles[0,1,:] 
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
        dipoles[0,col,:] = dip1el[v_orb, SOMO2, :]
        dipoles[col,0,:] = dipoles[0,col,:]
    #7 <OS1|mu|SL2> 
    block_index = nvirt + 2 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        dipoles[0,col,:] = -dip1el[v_orb, SOMO1, :]
        dipoles[col,0,:] = dipoles[0,col,:]
    #8 <OS1|mu|HL1>
    block_index = 2 * nvirt + 2 * ndocc + 3
    for col in range(block_index, block_index + (ndocc * nvirt)):
        o_orb = (col - block_index) // ndocc # Increase o_orb after every ndocc cols
        v_orb = (col - block_index) % nvirt + (SOMO2 + 1) # Increase v_orb then reset after nvirt cols
        dipoles[0,col,:] = (-2 ** 0.5) * dip1el[o_orb, v_orb, :]
        dipoles[col,0,:] = dipoles[0,col,:]
    #9 <OS1|mu|HL2>=0
    
    
    #10 <ZW0|mu|ZW0>
    for o in range(ndocc + 1):
        dipoles[1,1,:] -= 2*dip1el[o,o,:]
    #11 <ZW0|mu|ZW0'> = 0
    #12 <ZW0|mu|HS1>
    block_index = 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        dipoles[1,col,:] = (-2 ** 0.5) * dip1el[o_orb, SOMO2, :]
        dipoles[col,1,:] = dipoles[1,col,:]
    #13 <ZW0|mu|HS2> = 0
    #14 <ZW0|mu|SL1> 
    block_index = 2 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        dipoles[1,col,:] = (2 ** 0.5) * dip1el[v_orb, SOMO1, :]
        dipoles[col,1,:] = dipoles[1,col,:]
    #15 <ZW0|mu|SL2> = 0
    #16 <ZW0|mu|HL1> = 0
    #17 <ZW0|mu|HL2> = 0
    
    
    #18 <ZW0'|mu|ZW0'>
    for o in range(ndocc):
        dipoles[2,2,:] -= 2*dip1el[o,o,:]
    dipoles[2,2,:] -= 2 * dip1el[SOMO2,SOMO2,:]
    #19 <ZW0'|mu|HS1> = 0
    #20 <ZW0'|mu|HS2>
    block_index = ndocc + 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        dipoles[2,col,:] = (2 ** 0.5) * dip1el[o_orb, SOMO1, :]
        dipoles[col,2,:] = dipoles[2,col,:]
    #21 <ZW0'|mu|SL1> = 0
    #22 <ZW0'|mu|SL2>
    block_index = nvirt + 2 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        dipoles[2,col,:] = (- 2 ** 0.5) * dip1el[v_orb, SOMO2, :]
        dipoles[col,2,:] = dipoles[2,col,:]
    #23 <ZW0'|mu|HL1> = 0
    #24 <ZW0'|mu|HL2> = 0
    
    
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
        dipoles[row,row,:] = -dip1el[SOMO1, SOMO2, :] #Only diagonal elements are non-zero
    #27 <HS1|mu|SL1> = 0
    #28 <HS1|mu|SL2> = 0
    #29 <HS1|mu|HL1>
    col_block_index = 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // ndocc
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                dipoles[row, col, :] = (1 / np.sqrt(2)) * dip1el[SOMO1, v_orb, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #30 <HS1|mu|HL2>...
    col_block_index = (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // ndocc
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
            o_orb2 = (col - col_block_index) // ndocc
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                dipoles[row, col, :] = - (1 / np.sqrt(2)) * dip1el[SOMO2, v_orb, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #35 <HS2|mu|HL2>
    col_block_index = (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // ndocc
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
                dipoles[row,col,:] -= dip1el[SOMO1, SOMO1, :] # Add contribution from 1e in SOMO1
            else:    
                dipoles[row,col,:] = -dip1el[v_orb1, v_orb2, :]
            dipoles[col,row,:] = dipoles[row,col,:]
    #37 <SL1|mu|SL2>
    col_block_index = nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        dipoles[row,row,:] = dip1el[SOMO1,SOMO2,:]
    #38 <SL1|mu|HL1>
    col_block_index = 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // ndocc
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                dipoles[row, col, :] = - (1 / np.sqrt(2)) * dip1el[o_orb, SOMO2, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #39 <SL1|mu|HL2>
    col_block_index = (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // ndocc
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                dipoles[row, col, :] = np.sqrt(1.5) * dip1el[o_orb, SOMO2, :]
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
                dipoles[row,col,:] -= dip1el[SOMO2, SOMO2, :] # Add contribution from 1e in SOMO2
            else:    
                dipoles[row,col,:] = -dip1el[v_orb1, v_orb2, :]
            dipoles[col,row,:] = dipoles[row,col,:]
    #41 <SL2|mu|HL1>
    col_block_index = 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // ndocc
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                dipoles[row, col, :] = (1 / np.sqrt(2)) * dip1el[o_orb, SOMO1, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #42 <SL2|mu|HL2>
    col_block_index = (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // ndocc
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                dipoles[row, col, :] = np.sqrt(1.5) * dip1el[o_orb, SOMO1, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    
    
    row_block_index = 2 * nvirt + 2 * ndocc + 3
    #43 <HL1|mu|HL1>
    col_block_index = 2 * nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + (ndocc * nvirt)):
        o_orb1 = (row - row_block_index) // ndocc
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // ndocc
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
        o_orb1 = (row - row_block_index) // ndocc
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // ndocc
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
        dipoles[row_index,col,:] = dip1el[v_orb,SOMO2,:]
        dipoles[col,row_index,:] = dipoles[row_index,col,:]
    #50 <OS3|mu|SL2>
    col_index = 2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 4
    for col in range(col_index, col_index + nvirt):
        v_orb = col - col_index + (SOMO2 + 1)
        dipoles[row_index,col,:] = -dip1el[v_orb,SOMO1,:]
        dipoles[col,row_index,:] = dipoles[row_index,col,:]
    #51 <OS3|mu|HL1>
    col_index = 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for col in range(block_index, block_index + (ndocc * nvirt)):
        o_orb = (col - block_index) // ndocc # Increase o_orb after every ndocc cols
        v_orb = (col - block_index) % nvirt + (SOMO2 + 1) # Increase v_orb then reset after nvirt cols
        dipoles[row_index,col,:] = (-2 ** 0.5) * dip1el[o_orb, v_orb, :]
        dipoles[col,row_index,:] = dipoles[0,col,:]
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
        dipoles[row,row,:] = -dip1el[SOMO1, SOMO2, :] #Only diagonal elements are non-zero
    #56 <HS1|mu|SL1> = 0
    #57 <HS1|mu|SL2> = 0
    #58 <HS1|mu|HL1>
    col_index = 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // ndocc
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                dipoles[row, col, :] = - (1 / np.sqrt(2)) * dip1el[SOMO1, v_orb, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #59 <HS1|mu|HL2>
    col_index = 3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // ndocc
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                dipoles[row, col, :] = - (1 / np.sqrt(2)) * dip1el[SOMO1, v_orb, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #60 <HS1|mu|HL3>
    col_index = 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // ndocc
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
    col_index = 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // ndocc
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                dipoles[row, col, :] = - (1 / np.sqrt(2)) * dip1el[SOMO2, v_orb, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #65 <HS2|mu|HL2>
    col_index = 3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // ndocc
            v_orb = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if o_orb1 == o_orb2:
                dipoles[row, col, :] = (1 / np.sqrt(2)) * dip1el[SOMO2, v_orb, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #65 <HS2|mu|HL3>
    col_index = 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // ndocc
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
                dipoles[row,col,:] -= dip1el[SOMO1, SOMO1, :] # Add contribution from 1e in SOMO1
            else:    
                dipoles[row,col,:] = -dip1el[v_orb1, v_orb2, :]
            dipoles[col,row,:] = dipoles[row,col,:]
    #67 <SL1|mu|SL2>
    col_block_index = 2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        dipoles[row,row,:] = -dip1el[SOMO1,SOMO2,:]
    #68 <SL1|mu|HL1>
    col_block_index = 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // ndocc
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                dipoles[row, col, :] = - (1 / np.sqrt(2)) * dip1el[o_orb, SOMO2, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #69 <SL1|mu|HL2>
    col_block_index = 3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // ndocc
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                dipoles[row, col, :] = - (1 / np.sqrt(2)) * dip1el[o_orb, SOMO2, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #70 <SL1|mu|HL3>
    col_block_index = 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // ndocc
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                dipoles[row, col, :] = dip1el[o_orb, SOMO2, :]
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
                dipoles[row,col,:] -= dip1el[SOMO2, SOMO2, :] # Add contribution from 1e in SOMO2
            else:    
                dipoles[row,col,:] = -dip1el[v_orb1, v_orb2, :]
            dipoles[col,row,:] = dipoles[row,col,:]
    #72 <SL2|mu|HL1>
    col_block_index = 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // ndocc
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                dipoles[row, col, :] = (1 / np.sqrt(2)) * dip1el[o_orb, SOMO1, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #73 <SL2|mu|HL2>
    col_block_index = 3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // ndocc
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                dipoles[row, col, :] = - (1 / np.sqrt(2)) * dip1el[o_orb, SOMO1, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    #74 <SL2|mu|HL3>
    col_block_index = 4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb = (col - col_block_index) // ndocc
            v_orb2 = (col - col_block_index) % nvirt + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                dipoles[row, col, :] = - dip1el[o_orb, SOMO1, :]
                dipoles[col,row,:] = dipoles[row,col,:]
    
    
    row_block_index = 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    #75 <HL1|mu|HL1>
    col_block_index = 2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + (ndocc * nvirt)):
        o_orb1 = (row - row_block_index) // ndocc
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // ndocc
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
        o_orb1 = (row - row_block_index) // ndocc
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // ndocc
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
        o_orb1 = (row - row_block_index) // ndocc
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // ndocc
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
        o_orb1 = (row - row_block_index) // ndocc
        v_orb1 = (row - row_block_index) % nvirt + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + (ndocc * nvirt)):
            o_orb2 = (col - col_block_index) // ndocc
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
    ZW0_perm_dip=dipoles[1,1,:]
    ZW0p_perm_dip=dipoles[2,2,:]
    OS3_perm_dip=dipoles[4 * ndocc + 3,4 * ndocc + 3,:]
    for dipole in [OS1_perm_dip, ZW0_perm_dip, ZW0p_perm_dip, OS3_perm_dip]:
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
            Zwitterion 0 (|ZW0>) = {ZW0_perm_dip[0]:.3f} x {ZW0_perm_dip[1]:.3f} y {ZW0_perm_dip[2]:.3f} z\n \
            Zwitterion 0' (|ZW0'>) = {ZW0p_perm_dip[0]:.3f} x {ZW0p_perm_dip[1]:.3f} y {ZW0p_perm_dip[2]:.3f} z\n \
            Open-Shell Triplet (|OS3>) = {OS3_perm_dip[0]:.3f} x {OS3_perm_dip[1]:.3f} y {OS3_perm_dip[2]:.3f} z\n")
    return dipoles


def print_ci_info(out_file, ci_energies, ci_coeffs, ndocc, norbs, state0_tdms, rng, cutoff_energy, ci_type, csf_tol=0.05):
    print("Energy of the lowest CI state:", ci_energies[0])
    osc_array = np.zeros_like(ci_energies)
    s2_array = np.zeros_like(ci_energies)
    strng = ""
    nvirt = norbs - ndocc - 2
    for i in range(rng): # Loop over CIS states
        if ci_energies[i] - ci_energies[0] > cutoff_energy:
            break
        print("\nState %s %04.3f eV " % (i, ci_energies[i] - ci_energies[0]))
        print("Excitation    CI Coef")
        out_file.write("State %s %04.3f eV \n" % (i, ci_energies[i] - ci_energies[0]))
        out_file.write("Excitation    CI Coef\n")
        spin = 0 # initialise total spin
        for j in range (ci_coeffs.shape[0]): # Loop over configurations in each CIS state
            if ci_type == 'CIS':
                ########### SINGLET CSFS ############   
            # Open shell singlet ground state (|OS1>)
                if j == 0: 
                    str = "|1^OS>"
                    # S^2 = 0
            # Zwitterion 0 (|ZW0>)    
                elif j == 1:
                    str = "|1^ZW0>"
                    # S^2 = 0
            # Zwitterion 0' (|ZW0'>)   
                elif j == 2:
                    str = "|1^ZW0'>"
                    # S^2 = 0
            # Singlet Homo to SOMO 1 (|1^HS1>)
                elif j > 2 and j <= ndocc + 2:
                    iorb = ndocc + 3 - j
                    str = f"|1^HS1_{iorb}>" 
                    # S^2 = 0 
            # Singlet Homo to SOMO 2 (|1^HS2>)
                elif j > ndocc + 2 and j <= (2 * ndocc + 2):
                    iorb = 2 * ndocc + 3 - j
                    str = f"|1^HS2_{iorb}>" 
                    # S^2 = 0
            # Singlet SOMO to LUMO 1 (|1^SL1>)
                elif j > (2 * ndocc + 2) and j <= (nvirt + 2 * ndocc + 2):
                    iorb = j - (2 * ndocc + 2)
                    str = f"|1^SL1_{iorb}'>"
                    # S^2 = 0
            # Singlet SOMO to LUMO 2 (|1^SL2>)
                elif j > (nvirt + 2 * ndocc + 2) and j <= (2 * nvirt + 2 * ndocc + 2):
                    iorb = j - (nvirt + 2 * ndocc + 2)
                    str = f"|1^SL2_{iorb}'>"
                    # S^2 = 0
                ########### TRIPLET CSFs ###########
            # Triplet ground state (|OS3>)
                elif j == (2 * nvirt + 2 * ndocc + 3): 
                    str = "|3^OS>"
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet Homo to SOMO 1 (|3^HS1>)
                elif j > (2 * nvirt + 2 * ndocc + 3) and j <= (2 * nvirt + 3 * ndocc + 3):
                    iorb = 2 * nvirt + 3 * ndocc + 4 - j
                    str = f"|3^HS1_{iorb}>" 
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1) 
            # Triplet Homo to SOMO 2 (|3^HS2>)
                elif j > (2 * nvirt + 3 * ndocc + 3) and j <= (2 * nvirt + 4 * ndocc + 3):
                    iorb = 2 * nvirt + 4 * ndocc + 4 - j
                    str = f"|3^HS2_{iorb}>" 
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet SOMO to LUMO 1 (|3^SL1>)
                elif j > (2 * nvirt + 4 * ndocc + 3) and j <= (3 * nvirt + 4 * ndocc + 3):
                    iorb = j - (2 * nvirt + 4 * ndocc + 3)
                    str = f"|3^SL1_{iorb}'>"
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet SOMO to LUMO 2 (|3^SL2>)
                elif j > (3 * nvirt + 4 * ndocc + 3) and j <= (4 * nvirt + 4 * ndocc + 3):
                    iorb = j - (3 * nvirt + 4 * ndocc + 3)
                    str = f"|3^SL2_{iorb}'>"
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
                if np.absolute(ci_coeffs[j,i]) > csf_tol:
                    print("%s %10.5f" %(str, ci_coeffs[j,i]))
                    out_file.write("%s %10.5f \n" %(str, ci_coeffs[j,i]))
                    
            else:
            ########## SINGLET CSFS ##########   
            # Open shell singlet ground state (|OS1>)
                if j == 0: 
                    str = "|1^OS>"
                    # S^2 = 0
            # Zwitterion 0 (|ZW0>)    
                elif j == 1:
                    str = "|1^ZW0>"
                    # S^2 = 0
            # Zwitterion 0' (|ZW0'>)   
                elif j == 2:
                    str = "|1^ZW0'>"
                    # S^2 = 0
            # Singlet Homo to SOMO 1 (|1^HS1>)
                elif j > 2 and j <= ndocc + 2:
                    iorb = ndocc + 3 - j
                    str = f"|1^HS1_{iorb}>" 
                    # S^2 = 0 
            # Singlet Homo to SOMO 2 (|1^HS2>)
                elif j > ndocc + 2 and j <= (2 * ndocc + 2):
                    iorb = 2 * ndocc + 3 - j
                    str = f"|1^HS2_{iorb}>" 
                    # S^2 = 0
            # Singlet SOMO to LUMO 1 (|1^SL1>)
                elif j > (2 * ndocc + 2) and j <= (nvirt + 2 * ndocc + 2):
                    iorb = j - (2 * ndocc + 2)
                    str = f"|1^SL1_{iorb}'>"
                    # S^2 = 0
            # Singlet SOMO to LUMO 2 (|1^SL2>)
                elif j > (nvirt + 2 * ndocc + 2) and j <= (2 * nvirt + 2 * ndocc + 2):
                    iorb = j - (nvirt + 2 * ndocc + 2)
                    str = f"|1^SL2_{iorb}'>"
                    # S^2 = 0
            # Singlet HOMO to LUMO 1 (|1^HL1>)
                elif j > (2 * nvirt + 2 * ndocc + 2) and j <= ((ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 2):
                    o_orb = ndocc - ((j - (2 * nvirt + 2 * ndocc + 3)) // nvirt)
                    v_orb = ((j - (2 * nvirt + 2 * ndocc + 3)) % nvirt) + 1
                    str = f"|1^HL1_{o_orb}{v_orb}'>" 
                    # S^2 = 0
            # Singlet HOMO to LUMO 2 (|1^HL2>)
                elif j > ((ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 2) and j <= (2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 2):
                    o_orb = ndocc - ((j - ((ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3)) // nvirt)
                    v_orb = ((j - ((ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3)) % nvirt) + 1
                    str = f"|1^HL2_{o_orb}{v_orb}'>" 
                    # S^2 = 0
            ########### TRIPLET CSFs ###########
            # Triplet ground state (|OS3>)
                elif j == (2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3): 
                    str = "|3^OS>"
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet Homo to SOMO 1 (|3^HS1>)
                elif j > (2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3) and j <= (2 * (ndocc * nvirt) + 2 * nvirt + 3 * ndocc + 3):
                    iorb = (2 * (ndocc * nvirt) + 2 * nvirt + 3 * ndocc + 4) - j
                    str = f"|3^HS1_{iorb}>" 
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1) 
            # Triplet Homo to SOMO 2 (|3^HS2>)
                elif j > (2 * (ndocc * nvirt) + 2 * nvirt + 3 * ndocc + 3) and j <= (2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 3):
                    iorb = (2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 4) - j
                    str = f"|3^HS2_{iorb}>" 
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet SOMO to LUMO 1 (|3^SL1>)
                elif j > (2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 3) and j <= (2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 3):
                    iorb = j - (2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 3)
                    str = f"|3^SL1_{iorb}'>"
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet SOMO to LUMO 2 (|3^SL2>)
                elif j > (2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 3) and j <= (2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3):
                    iorb = j - (2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 3)
                    str = f"|3^SL2_{iorb}'>"
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet HOMO to LUMO 1 (|1^HL1>)
                elif j > (2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3) and j <= (3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|3^HL1_{o_orb}{v_orb}'>" 
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet HOMO to LUMO 2 (|3^HL2>)
                elif j > (3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3) and j <= (3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|3^HL2_{o_orb}{v_orb}'>" 
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Triplet HOMO to LUMO 3 (|3^HL2>)
                elif j > (4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3) and j <= (5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|3^HL3_{o_orb}{v_orb}'>" 
                    spin += 2 * ci_coeffs[j,i]**2 # (S=1)
            # Quintet HOMO to LUMO (|5^HL>)
                elif j > (5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|5^HL_{o_orb}{v_orb}'>" 
                    spin += 6 * ci_coeffs[j,i]**2 # (S=2)
                if np.absolute(ci_coeffs[j,i]) > csf_tol:
                    print("%s %10.5f" %(str, ci_coeffs[j,i]))
                    out_file.write("%s %10.5f \n" %(str, ci_coeffs[j,i]))
        
        osc = 2.0/3.0 * ((ci_energies[i] - ci_energies[0]) / toev) * (state0_tdms[i,0]**2 + state0_tdms[i,1]**2 + state0_tdms[i,2]**2)  # Calculating Oscillator Strength
        osc_array[i] = osc
        s2_array[i] = spin
        print("TDMs with State 0")
        print("TDMX:%04.3f   TDMY:%04.3f   TDMZ:%04.3f   Oscillator Strength:%04.5f   <S**2>: %04.3f" % (state0_tdms[i,0], state0_tdms[i,1], state0_tdms[i,2], osc, spin))
        print("--------------------------------------------------------------------\n")
        out_file.write("TDMs with State 0")
        out_file.write("TDMX:%04.3f   TDMY:%04.3f   TDMZ:%04.3f   Oscillator Strength:%04.5f   <S**2>: %04.3f" % (state0_tdms[i,0], state0_tdms[i,1], state0_tdms[i,2], osc, spin))
        out_file.write("--------------------------------------------------------------------\n")
        strng = strng + broaden(20.0,osc,ci_energies[i]-ci_energies[0]) 
        strng = strng + broaden(FWHM,osc,ci_energies[i]-ci_energies[0])
    
    return strng, osc_array, s2_array


def print_csf_info(ham_rot, norbs, ndocc, ci_type= 'CIS'):
    # Print energies of CSFs
        nvirt = norbs - ndocc - 2
        for j in range(ham_rot.shape[1]):
            if ci_type == 'CIS':
                if j == 0: 
                    str = "|1^OS>"
                elif j == 1:
                    str = "|1^ZW0>"
                elif j == 2:
                    str = "|1^ZW0'>"
                elif j > 2 and j <= ndocc + 2:
                    iorb = ndocc + 3 - j
                    str = f"|1^HS1_{iorb}>"
                elif j > ndocc + 2 and j <= (2 * ndocc + 2):
                    iorb = 2 * ndocc + 3 - j
                    str = f"|1^HS2_{iorb}>" 
                elif j > (2 * ndocc + 2) and j <= (nvirt + 2 * ndocc + 2):
                    iorb = j - (2 * ndocc + 2)
                    str = f"|1^SL1_{iorb}'>"
                elif j > (nvirt + 2 * ndocc + 2) and j <= (2 * nvirt + 2 * ndocc + 2):
                    iorb = j - (nvirt + 2 * ndocc + 2)
                    str = f"|1^SL2_{iorb}'>"
                elif j == (2 * nvirt + 2 * ndocc + 3): 
                    str = "|3^OS>"
                elif j > (2 * nvirt + 2 * ndocc + 3) and j <= (2 * nvirt + 3 * ndocc + 3):
                    iorb = 2 * nvirt + 3 * ndocc + 4 - j
                    str = f"|3^HS1_{iorb}>" 
                elif j > (2 * nvirt + 3 * ndocc + 3) and j <= (2 * nvirt + 4 * ndocc + 3):
                    iorb = 2 * nvirt + 4 * ndocc + 4 - j
                    str = f"|3^HS2_{iorb}>" 
                elif j > (2 * nvirt + 4 * ndocc + 3) and j <= (3 * nvirt + 4 * ndocc + 3):
                    iorb = j - (2 * nvirt + 4 * ndocc + 3)
                    str = f"|3^SL1_{iorb}'>"
                elif j > (3 * nvirt + 4 * ndocc + 3) and j <= (4 * nvirt + 4 * ndocc + 3):
                    iorb = j - (3 * nvirt + 4 * ndocc + 3)
                    str = f"|3^SL2_{iorb}'>"
                print(f"Energy of CSF {str}:", np.diag(ham_rot)[j])
            else:
                if j == 0: 
                    str = "|1^OS>" 
                elif j == 1:
                    str = "|1^ZW0>"
                elif j == 2:
                    str = "|1^ZW0'>"
                elif j > 2 and j <= ndocc + 2:
                    iorb = ndocc + 3 - j
                    str = f"|1^HS1_{iorb}>"
                elif j > ndocc + 2 and j <= (2 * ndocc + 2):
                    iorb = 2 * ndocc + 3 - j
                    str = f"|1^HS2_{iorb}>"
                elif j > (2 * ndocc + 2) and j <= (nvirt + 2 * ndocc + 2):
                    iorb = j - (nvirt + 2 * ndocc + 2)
                    str = f"|1^SL1_{iorb}'>"
                elif j > (nvirt + 2 * ndocc + 2) and j <= (2 * nvirt + 2 * ndocc + 2):
                    iorb = j - (nvirt + 2 * ndocc + 2)
                    str = f"|1^SL2_{iorb}'>"
                elif j > (2 * nvirt + 2 * ndocc + 2) and j <= ((ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 2):
                    o_orb = ndocc - ((j - (2 * nvirt + 2 * ndocc + 3)) // nvirt)
                    v_orb = ((j - (2 * nvirt + 2 * ndocc + 3)) % nvirt) + 1
                    str = f"|1^HL1_{o_orb}{v_orb}'>"
                elif j > ((ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 2) and j <= (2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 2):
                    o_orb = ndocc - ((j - ((ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3)) // nvirt)
                    v_orb = ((j - ((ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3)) % nvirt) + 1
                    str = f"|1^HL2_{o_orb}{v_orb}'>"
                elif j == (2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3): 
                    str = "|3^OS>"
                elif j > (2 * (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3) and j <= (2 * (ndocc * nvirt) + 2 * nvirt + 3 * ndocc + 3):
                    iorb = (2 * (ndocc * nvirt) + 2 * nvirt + 3 * ndocc + 4) - j
                    str = f"|3^HS1_{iorb}>" 
                elif j > (2 * (ndocc * nvirt) + 2 * nvirt + 3 * ndocc + 3) and j <= (2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 3):
                    iorb = (2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 4) - j
                    str = f"|3^HS2_{iorb}>" 
                elif j > (2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 3) and j <= (2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 3):
                    iorb = j - (2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 3)
                    str = f"|3^SL1_{iorb}'>"
                elif j > (2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 3) and j <= (2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3):
                    iorb = j - (2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 3)
                    str = f"|3^SL2_{iorb}'>"
                elif j > (2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3) and j <= (3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (2 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|3^HL1_{o_orb}{v_orb}'>"
                elif j > (3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3) and j <= (3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|3^HL2_{o_orb}{v_orb}'>"
                elif j > (4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3) and j <= (5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (4 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|3^HL3_{o_orb}{v_orb}'>"
                elif j > (5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 3):
                    o_orb = ndocc - ((j - (5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) // nvirt)
                    v_orb = ((j - (5 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4)) % nvirt) + 1
                    str = f"|5^HL_{o_orb}{v_orb}'>"
                print(f"Energy of CSF {str}:", np.diag(ham_rot)[j])
   


def ci_rot(ndocc,norbs,coords,atoms,energy0,repulsion,orb_energies,hf_orbs, file, ci_type = "CIS"):
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
        print('Coulomb Matrix in MO basis, J_ij = (ii|jj):\n', (np.einsum('iijj->ij', rep_tens) * (0.533371 / 4.77853591)))
        print('Coulomb Matrix in MO basis, K_ij = (ij|ji):\n', (np.einsum('ijij->ij', rep_tens) * (0.533371 / 4.77853591)))
        # Get exchange and Coulomb terms for SOMOs
        j00 = compute_j00(hf_orbs,repulsion,ndocc)
        k00 = compute_k00(hf_orbs,repulsion,ndocc)
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
        if ci_type == 'CIS':
            ham_rot = hetero_cis_ham_rot(ndocc, norbs, energy0, orb_energies, j00, k00, rep_tens)
        else:
            ham_rot = hetero_xcis_ham_rot(ndocc, norbs, energy0, orb_energies, j00, k00, rep_tens)
        print("Checking that the Hamiltonian is symmetric (a value of zero means matrix is symmetric) ... ")
        print("Frobenius norm of matrix - matrix transpose = %f.\n" %(linalg.norm(ham_rot-ham_rot.T)))

        out.write("Checking that the Hamiltonian is symmetric (a value of zero means matrix is symmetric) ... \n")
        out.write("Frobenius norm of matrix - matrix transpose = %f.\n" %(linalg.norm(ham_rot-ham_rot.T)))
        
        # Print energies of CSFs
        #print_csf_info(ham_rot, norbs, ndocc, ci_type= 'CIS')
        
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
        if ci_type == 'CIS':
            dip_array = dipole_cis(coords,atoms,norbs,hf_orbs,ndocc,nstates)
        else:
            dip_array = dipole_xcis(coords,atoms,norbs,hf_orbs,ndocc,nstates)
        dip_couplings = np.einsum("ijx,jk",dip_array,ci_coeffs)
        state0_tdms = np.einsum("j,jix",ci_coeffs[:,0].T, dip_couplings)
        
        # Print information about CI states
        strng, osc_array, s2_array = print_ci_info(out, ci_energies, ci_coeffs, ndocc, norbs, state0_tdms, rng, cutoff_energy, ci_type, csf_tol=0.1)
        strng = strng[1:]    
    return strng, ci_energies - ci_energies[0], osc_array, s2_array



def hetero_cis_ham_rot(ndocc, norbs, energy0, orb_energies, j00, k00, rep_tens):
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
    nstates = 4 * ndocc + 4 * nvirt + 4
    cish = np.zeros((nstates,nstates))
    
    ################# SINGLET BLOCK ######################
    #1 <OS1|H|OS1>
    cish[0,0] = energy0 - (0.25 * j00 ) + (1.5 * k00)
    #2 <OS1|H|ZW0>
    cish[0,1] = ((2 ** 0.5) / 2) * (rep_tens[SOMO1,SOMO2,SOMO1,SOMO1] - rep_tens[SOMO1,SOMO2,SOMO2,SOMO2])
    cish[1,0] = cish[0,1]
    #3 <OS1|H|ZW0'>
    cish[0,2] = ((2 ** 0.5) / 2) * (rep_tens[SOMO1,SOMO2,SOMO1,SOMO1] - rep_tens[SOMO1,SOMO2,SOMO2,SOMO2])
    cish[2,0] = cish[0,2]
    #4 <OS1|H|HS1> 
    block_index = 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        cish[0,col] = 1.5 * rep_tens[o_orb,SOMO2,SOMO2,SOMO1] - 0.5 * rep_tens[o_orb,SOMO1,SOMO1,SOMO1]
        cish[col,0] = cish[0,col]
    #5 <OS1|H|HS2> 
    block_index = ndocc + 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        cish[0,col] = 0.5 * rep_tens[o_orb,SOMO2,SOMO2,SOMO2] - 1.5 * rep_tens[o_orb,SOMO1,SOMO1,SOMO2]
        cish[col,0] = cish[0,col]
    #6 <OS1|H|SL1>
    block_index = 2 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        cish[0,col] = 0.5 * rep_tens[v_orb,SOMO2,SOMO2,SOMO2] - 1.5 * rep_tens[v_orb,SOMO1,SOMO1,SOMO2]
        cish[col,0] = cish[0,col]
    #7 <OS1|H|SL2>
    block_index = nvirt + 2 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        cish[0,col] = 1.5 * rep_tens[v_orb,SOMO2,SOMO2,SOMO1] - 0.5 * rep_tens[v_orb,SOMO1,SOMO1,SOMO1]
        cish[col,0] = cish[0,col]
    #8 - 13 are triplet states so have 0 interaction.
    
    # 14 <ZW0|H|ZW0>
    cish[1,1] = energy0 + orb_energies[SOMO1] - orb_energies[SOMO2] + 0.25 * j00 + 0.5 * k00 - rep_tens[SOMO1, SOMO1, SOMO2, SOMO2] + 0.5 * k00
    #15 <ZW0|H|ZW0'>
    cish[1,2] = rep_tens[SOMO1,SOMO2,SOMO2,SOMO1]
    cish[2,1] = cish[1,2]
    #16 <ZW0|H|HS1>
    block_index = 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        cish[1,col] = (2 ** 0.5) * (rep_tens[o_orb,SOMO2,SOMO1,SOMO1] - 0.5 * rep_tens[o_orb,SOMO1,SOMO1,SOMO2] - 0.5 * rep_tens[o_orb,SOMO2,SOMO2,SOMO2])
        cish[col,1] = cish[1,col]
    #17 <ZW0|H|HS2>
    block_index = ndocc + 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        cish[1,col] = (-2 ** 0.5) * rep_tens[o_orb,SOMO2,SOMO2,SOMO1]
        cish[col, 1] = cish[1,col]
    #18 <ZW0|H|SL1>
    block_index = 2 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        cish[1,col] = (2 ** 0.5) * (rep_tens[v_orb,SOMO1,SOMO2,SOMO2] - 0.5 * rep_tens[v_orb,SOMO1,SOMO1,SOMO1] - 0.5 * rep_tens[v_orb,SOMO2,SOMO2,SOMO1])
        cish[col,1] = cish[1,col]
    #19 <ZW0|H|SL2>
    block_index = nvirt + 2 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        cish[1,col] = (2 ** 0.5) * rep_tens[v_orb,SOMO1,SOMO1,SOMO2]
        cish[col,1] = cish[1,col]
    #20 - 25 are triplet states so have 0 interaction.
    
    #26 <ZW0'|H|ZW0'>
    cish[2,2] = energy0 + orb_energies[SOMO2] - orb_energies[SOMO1] + 0.25 * j00 + 0.5 * k00 - rep_tens[SOMO1, SOMO1, SOMO2, SOMO2] + 0.5 * k00
    #27 <ZW0'|H|HS1>
    block_index = 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        cish[2,col] = (-2 ** 0.5) * rep_tens[o_orb,SOMO1,SOMO1,SOMO2]
        cish[col,2] = cish[2,col]
    #28 <ZW0'|H|HS2>
    block_index = ndocc + 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        cish[2,col] = (2 ** 0.5) * (rep_tens[o_orb,SOMO1,SOMO2,SOMO2] - 0.5 * rep_tens[o_orb,SOMO2,SOMO2,SOMO1] - 0.5 * rep_tens[o_orb,SOMO1,SOMO1,SOMO1]) # CHECK SIGN
        cish[col,2] = cish[2,col]
    #29 <ZW0'|H|SL1>
    block_index = 2 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        cish[2,col] = (-2 ** 0.5) * rep_tens[v_orb,SOMO2,SOMO2,SOMO1]
        cish[col,2] = cish[2,col]
    #30 <ZW0'|H|SL2>
    block_index = nvirt + 2 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        cish[2,col] = (2 ** 0.5) * (0.5 * rep_tens[v_orb,SOMO2,SOMO2,SOMO2] + 0.5 * rep_tens[v_orb,SOMO1,SOMO1,SOMO2] - rep_tens[v_orb,SOMO2,SOMO1,SOMO1]) # CHECK SIGN
        cish[col,2] = cish[2,col]
    #31 - 36 are triplet states so have 0 interaction.
    
    row_block_index = 3
    #37 <HS1|H|HS1>
    col_block_index = 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(row, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                cish[row, col] = energy0 + orb_energies[SOMO1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, SOMO1, SOMO1] + 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] \
                             + 1.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb1] + 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb1]
            else:    
                cish[row, col] = 0.5 * (rep_tens[o_orb2, SOMO1, SOMO1, o_orb1] + 0.5 * rep_tens[o_orb2, SOMO2, SOMO2, o_orb1]) - rep_tens[o_orb2, o_orb1, SOMO1, SOMO1]
            cish[col, row] = cish[row,col]
    #38 <HS1|H|HS2>
    col_block_index = ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                cish[row, col] = rep_tens[SOMO1, o_orb1, o_orb1, SOMO2] - rep_tens[o_orb1, o_orb1, SOMO1, SOMO2] + 0.5 * rep_tens[SOMO1, SOMO2, SOMO1, SOMO1] \
                             + 0.5 * rep_tens[SOMO1, SOMO2, SOMO2, SOMO2]
            else:    
                cish[row, col] = - rep_tens[o_orb1, o_orb2, SOMO1, SOMO2] - rep_tens[o_orb1, SOMO1, SOMO2, o_orb2]
            cish[col, row] = cish[row, col]
    #39 <HS1|H|SL1>
    col_block_index = 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb = col - col_block_index + (SOMO2 + 1)
            cish[row, col] = rep_tens[o_orb, SOMO1, SOMO2, v_orb] - 2 * rep_tens[o_orb, SOMO2, SOMO1, v_orb]
            cish[col, row] = cish[row,col]
    #40 <HS1|H|SL2>
    col_block_index = nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb = col - col_block_index + (SOMO2 + 1)
            cish[row, col] = - rep_tens[o_orb, SOMO1, SOMO1, v_orb]
            cish[col, row] = cish[row,col]
    # 41 - 46 are triplets so have no interaction
    
    row_block_index = ndocc + 3
    #47 <HS2|H|HS2>
    col_block_index = ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(row, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                cish[row, col] = energy0 + orb_energies[SOMO2] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, SOMO2, SOMO2] + 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] \
                             + 1.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb1] + 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb1]
            else:    
                cish[row, col] = 1.5 * (rep_tens[o_orb2, SOMO1, SOMO1, o_orb1] + rep_tens[o_orb2, o_orb1, SOMO2, SOMO2]) - 0.5 * rep_tens[o_orb2, SOMO2, SOMO2, o_orb1]
            cish[col, row] = cish[row,col]
    #48 <HS2|H|SL1>
    col_block_index = 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb = col - col_block_index + (SOMO2 + 1)
            cish[row, col] = - rep_tens[o_orb, SOMO2, SOMO2, v_orb]
            cish[col, row] = cish[row, col]
    #49 <HS2|H|SL2>
    col_block_index = nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb = col - col_block_index + (SOMO2 + 1)
            cish[row, col] = rep_tens[o_orb, SOMO2, SOMO1, v_orb] - 2 * rep_tens[o_orb, SOMO1, SOMO2, v_orb]
            cish[col, row] = cish[row,col]
    #50 - 55 are triplets so no interaction
    
    row_block_index = 2 * ndocc + 3
    #56 <SL1|H|SL1>
    col_block_index = 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(row, col_block_index + nvirt):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                cish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[SOMO2] - rep_tens[v_orb1, v_orb1, SOMO2, SOMO2] + 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] \
                             - 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] + 1.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb1] + 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb1]
            else:    
                cish[row, col] = 1.5 * rep_tens[v_orb2, SOMO2, SOMO2, v_orb1] + 0.5 * rep_tens[v_orb2, SOMO1, SOMO1, v_orb1] -  rep_tens[v_orb2, v_orb1, SOMO2, SOMO2]
            cish[col, row] = cish[row,col]
    #57 <SL1|H|SL2>
    col_block_index = nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                cish[row, col] = 0.5 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO2] + 0.5 * rep_tens[SOMO1, SOMO2, SOMO2, SOMO2] - rep_tens[v_orb1, v_orb1, SOMO1, SOMO2] - rep_tens[v_orb1, SOMO1, SOMO2, v_orb1]
            else:    
                cish[row, col] = - rep_tens[v_orb1, v_orb2, SOMO1, SOMO2] - rep_tens[v_orb1, SOMO2, SOMO1, v_orb2]
            cish[col, row] = cish[row,col]
    
    #58 - 63 are all triplets so have no interaction
    
    row_block_index = nvirt + 2 * ndocc + 3
    #64 <SL2|H|SL2>
    col_block_index = nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(row, col_block_index + nvirt):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                cish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[SOMO1] - rep_tens[v_orb1, v_orb1, SOMO1, SOMO1] + 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] \
                             - 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2]+ 1.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb1] + 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb1]
            else:    
                cish[row, col] = 1.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb2] - 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] - rep_tens[v_orb1, v_orb2, SOMO1, SOMO1] - rep_tens[v_orb1, v_orb2, SOMO2, SOMO2]
            cish[col, row] = cish[row,col]
    
    #65 - 70 are all triplets so have no interaction
    
    ################# TRIPLET BLOCK ######################
    
    row_index = 2 * nvirt + 2 * ndocc + 3
    #71 <OS3|H|OS3>
    cish[row_index, row_index] = energy0 - (0.25 * j00 ) - (0.5 * k00)
    #72 <OS3|H|HS1>
    col_index = 2 * nvirt + 2 * ndocc + 4
    for col in range(col_index, col_index + ndocc):
        o_orb = col - col_index
        cish[row_index, col] = 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO1] + 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO1]
        cish[col, row_index] = cish[row_index,col]
    #73 <OS3|H|HS2>
    col_index = 2 * nvirt + 3 * ndocc + 4
    for col in range(col_index, col_index + ndocc):
        o_orb = col - col_index
        cish[row_index, col] = 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO2] + 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO2]
        cish[col, row_index] = cish[row_index,col]
    #73 <OS3|H|SL1>
    col_index = 2 * nvirt + 4 * ndocc + 4
    for col in range(col_index, col_index + nvirt):
        v_orb = col - col_index + (SOMO2 + 1)
        cish[row_index, col] = 0.5 * rep_tens[v_orb, SOMO1, SOMO2, SOMO1] - 0.5 * rep_tens[v_orb, SOMO2, SOMO2, SOMO2]
        cish[col, row_index] = cish[row_index,col]
    #73 <OS3|H|SL2>
    col_index = 3 * nvirt + 4 * ndocc + 4
    for col in range(col_index, col_index + nvirt):
        v_orb = col - col_index + (SOMO2 + 1)
        cish[row_index, col] = 0.5 * rep_tens[v_orb, SOMO1, SOMO1, SOMO1] - 0.5 * rep_tens[v_orb, SOMO2, SOMO2, SOMO1]
        cish[col, row_index] = cish[row_index, col]
    
    row_block_index = 2 * nvirt + 2 * ndocc + 4
    #74 <HS1|H|HS1>
    col_block_index = 2 * nvirt + 2 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(row, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                cish[row, col] = energy0 + orb_energies[SOMO1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, SOMO1, SOMO1] + 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] \
                             - 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb1] + 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb1]
            else:    
                cish[row, col] = 0.5 * (rep_tens[o_orb2, SOMO1, SOMO1, o_orb1] - rep_tens[o_orb2, o_orb1, SOMO1, SOMO1]) - 1.5 * rep_tens[o_orb2, SOMO2, SOMO2, o_orb1]
            cish[col, row] = cish[row,col]
    #75 <HS1|H|HS2>
    col_block_index = 2 * nvirt + 3 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                cish[row, col] = rep_tens[SOMO1, o_orb1, o_orb1, SOMO2] - rep_tens[o_orb1, o_orb1, SOMO1, SOMO2] + 0.5 * rep_tens[SOMO2, SOMO1, SOMO1, SOMO1] \
                             + 0.5 * rep_tens[SOMO1, SOMO2, SOMO2, SOMO2]                                                               #CHECK RESULT, SAME AS SINGLET?
            else:    
                cish[row, col] = rep_tens[o_orb2, o_orb1, SOMO1, SOMO2] - rep_tens[o_orb2, SOMO1, SOMO2, o_orb1]
            cish[col, row] = cish[row,col]
    #76 <HS1|H|SL1>
    col_block_index = 2 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb = col - col_block_index + (SOMO2 + 1)
            cish[row, col] = - rep_tens[o_orb, SOMO1, SOMO2, v_orb]
            cish[col, row] = cish[row,col]
    #77 <HS1|H|SL2>
    col_block_index = 3 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb = col - col_block_index + (SOMO2 + 1)
            cish[row, col] = rep_tens[o_orb, SOMO1, SOMO1, v_orb]
            cish[col, row] = cish[row,col]

    
    row_block_index = 2 * nvirt + 3 * ndocc + 4
    #78 <HS2|H|HS2>
    col_block_index = 2 * nvirt + 3 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(row, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:

                cish[row, col] = energy0 + orb_energies[SOMO2] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, SOMO2, SOMO2] + 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] \
                             - 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb1] + 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb1]
            else:
                cish[row, col] = rep_tens[o_orb1, o_orb2, SOMO2, SOMO2] - 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb2] - 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb2]
            cish[col, row] = cish[row,col]
    #79 <HS2|H|SL1>
    col_block_index = 2 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb = col - col_block_index + (SOMO2 + 1)
            cish[row, col] = - rep_tens[o_orb, SOMO2, SOMO2, v_orb]
            cish[col, row] = cish[row,col]
    #80 <HS2|H|SL2>
    col_block_index = 3 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb = col - col_block_index + (SOMO2 + 1)
            cish[row, col] = rep_tens[o_orb, SOMO2, SOMO1, v_orb]
            cish[col, row] = cish[row,col]
            
    row_block_index = 2 * nvirt + 4 * ndocc + 4
    #81 <SL1|H|SL1>
    col_block_index =  2 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(row, col_block_index + nvirt):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                cish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[SOMO2] - rep_tens[v_orb1, v_orb1, SOMO2, SOMO2] + 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] \
                             - 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] - 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb1] + 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb1]
            else:    
                cish[row, col] =  0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] - rep_tens[v_orb1, v_orb2, SOMO2, SOMO2] - 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb2]
            cish[col, row] = cish[row,col]
    #82 <SL1|H|SL2>
    col_block_index = 3 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                cish[row, col] = rep_tens[v_orb1, v_orb1, SOMO1, SOMO2] - rep_tens[v_orb1, SOMO1, SOMO2, v_orb1] - 0.5 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO2] - 0.5 * rep_tens[SOMO1, SOMO2, SOMO2, SOMO2]
            else:    
                cish[row, col] = rep_tens[v_orb1, v_orb2, SOMO1, SOMO2] - rep_tens[v_orb1, SOMO2, SOMO1, v_orb2] # CHECK SIGN
            cish[col, row] = cish[row,col]
    
    row_block_index =  3 * nvirt + 4 * ndocc + 4
    #83 <SL2|H|SL2>
    col_block_index =   3 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(row, col_block_index + nvirt):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                cish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[SOMO1] - rep_tens[v_orb1, v_orb1, SOMO1, SOMO1] + 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] \
                             - 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] - 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb1] + 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb1]
            else:    
                cish[row, col] =  0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] - 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb2] - rep_tens[v_orb1, v_orb2, SOMO1, SOMO1] \
                                  - rep_tens[v_orb1, v_orb2, SOMO2, SOMO2]
            cish[col, row] = cish[row,col]
    
    return cish


def hetero_xcis_ham_rot(ndocc, norbs, energy0, orb_energies, j00, k00, rep_tens):
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
    xcish[0,0] = energy0 - (0.25 * j00 ) + (1.5 * k00)
    #2 <OS1|H|ZW0>
    xcish[0,1] = ((2 ** 0.5) / 2) * (rep_tens[SOMO1,SOMO2,SOMO1,SOMO1] - rep_tens[SOMO1,SOMO2,SOMO2,SOMO2])
    xcish[1,0] = xcish[0,1]
    #3 <OS1|H|ZW0'>
    xcish[0,2] = ((2 ** 0.5) / 2) * (rep_tens[SOMO1,SOMO2,SOMO1,SOMO1] - rep_tens[SOMO1,SOMO2,SOMO2,SOMO2])
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
        xcish[0,col] = 0.5 * rep_tens[v_orb,SOMO2,SOMO2,SOMO2] - 1.5 * rep_tens[v_orb,SOMO1,SOMO1,SOMO2]
        xcish[col,0] = xcish[0,col]
    #7 <OS1|H|SL2>
    block_index = nvirt + 2 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        xcish[0,col] = 1.5 * rep_tens[v_orb,SOMO2,SOMO2,SOMO1] - 0.5 * rep_tens[v_orb,SOMO1,SOMO1,SOMO1]
        xcish[col,0] = xcish[0,col]
    #8 <OS1|H|HL1> = 0
    #9 <OS1|H|HL2>
    block_index =  (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for col in range(block_index, block_index + (ndocc * nvirt)):
        o_orb = (col - block_index) // nvirt # Increase o_orb after every ndocc cols
        v_orb = (col - block_index) % nvirt + (SOMO2 + 1) # Increase v_orb then reset after ndocc cols
        xcish[0,col] = np.sqrt(1.5) * (rep_tens[o_orb, SOMO2, SOMO2, v_orb] - rep_tens[o_orb, SOMO1, SOMO1, v_orb])
        xcish[col,0] = xcish[0,col]

    
    #10 <ZW0|H|ZW0>
    xcish[1,1] = energy0 + orb_energies[SOMO1] - orb_energies[SOMO2] + 0.25 * j00 + 0.5 * k00 - rep_tens[SOMO1, SOMO1, SOMO2, SOMO2] + 0.5 * k00
    #11 <ZW0|H|ZW0'>
    xcish[1,2] = rep_tens[SOMO1,SOMO2,SOMO2,SOMO1]
    xcish[2,1] = xcish[1,2]
    #12 <ZW0|H|HS1>
    block_index = 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        xcish[1,col] = (2 ** 0.5) * (rep_tens[o_orb,SOMO2,SOMO1,SOMO1] - 0.5 * rep_tens[o_orb,SOMO1,SOMO1,SOMO2] - 0.5 * rep_tens[o_orb,SOMO2,SOMO2,SOMO2])
        xcish[col,1] = xcish[1,col]
    #13 <ZW0|H|HS2>
    block_index = ndocc + 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        xcish[1,col] = (-2 ** 0.5) * rep_tens[o_orb,SOMO2,SOMO2,SOMO1]
        xcish[col, 1] = xcish[1,col]
    #14 <ZW0|H|SL1>
    block_index = 2 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        xcish[1,col] = (2 ** 0.5) * (rep_tens[v_orb,SOMO1,SOMO2,SOMO2] - 0.5 * rep_tens[v_orb,SOMO1,SOMO1,SOMO1] - 0.5 * rep_tens[v_orb,SOMO2,SOMO2,SOMO1])
        xcish[col,1] = xcish[1,col]
    #15 <ZW0|H|SL2>
    block_index = nvirt + 2 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        xcish[1,col] = (2 ** 0.5) * rep_tens[v_orb,SOMO1,SOMO1,SOMO2]
        xcish[col,1] = xcish[1,col]
    #16 <ZW0|H|HL1>
    block_index = 2 * nvirt + 2 * ndocc + 3
    for col in range(block_index, block_index + (ndocc * nvirt)):
        o_orb = (col - block_index) // nvirt
        v_orb = (col - block_index) % nvirt + (SOMO2 + 1)
        xcish[1,col] = 2 * rep_tens[o_orb, v_orb, SOMO1, SOMO2] - rep_tens[o_orb, SOMO2, SOMO1, v_orb]
        xcish[col,1] = xcish[1,col]
    #17 <ZW0|H|HL2>
    block_index = (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for col in range(block_index, block_index + (ndocc * nvirt)):
        o_orb = (col - block_index) // nvirt
        v_orb = (col - block_index) % nvirt + (SOMO2 + 1)
        xcish[1,col] = np.sqrt(3) * rep_tens[o_orb, SOMO2, SOMO1, v_orb]
        xcish[col,1] = xcish[1,col]
        
        
    #18 <ZW0'|H|ZW0'>
    xcish[2,2] = energy0 + orb_energies[SOMO2] - orb_energies[SOMO1] + 0.25 * j00 + 0.5 * k00 - rep_tens[SOMO1, SOMO1, SOMO2, SOMO2] + 0.5 * k00
    #19 <ZW0'|H|HS1>
    block_index = 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        xcish[2,col] = (-2 ** 0.5) * rep_tens[o_orb,SOMO1,SOMO1,SOMO2]
        xcish[col,2] = xcish[2,col]
    #20 <ZW0'|H|HS2>
    block_index = ndocc + 3
    for col in range(block_index, block_index + ndocc):
        o_orb = col - block_index
        xcish[2,col] = (2 ** 0.5) * (rep_tens[o_orb,SOMO1,SOMO2,SOMO2] - 0.5 * rep_tens[o_orb,SOMO2,SOMO2,SOMO1] - 0.5 * rep_tens[o_orb,SOMO1,SOMO1,SOMO1]) # CHECK SIGN
        xcish[col,2] = xcish[2,col]
    #21 <ZW0'|H|SL1>
    block_index = 2 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        xcish[2,col] = (-2 ** 0.5) * rep_tens[v_orb,SOMO2,SOMO2,SOMO1]
        xcish[col,2] = xcish[2,col]
    #22 <ZW0'|H|SL2>
    block_index = nvirt + 2 * ndocc + 3
    for col in range(block_index, block_index + nvirt):
        v_orb = col - block_index + (SOMO2 + 1)
        xcish[2,col] = (2 ** 0.5) * (0.5 * rep_tens[v_orb,SOMO2,SOMO2,SOMO2] + 0.5 * rep_tens[v_orb,SOMO1,SOMO1,SOMO2] - rep_tens[v_orb,SOMO2,SOMO1,SOMO1]) # CHECK SIGN
        xcish[col,2] = xcish[2,col]
    #23 <ZW0'|H|HL1>
    block_index = 2 * nvirt + 2 * ndocc + 3
    for col in range(block_index, block_index + (ndocc * nvirt)):
        o_orb = (col - block_index) // nvirt
        v_orb = (col - block_index) % nvirt + (SOMO2 + 1)
        xcish[1,col] = 2 * rep_tens[o_orb, v_orb, SOMO1, SOMO2] - rep_tens[o_orb, SOMO1, SOMO2, v_orb]
        xcish[col,1] = xcish[1,col]
    #24 <ZW0'|H|HL2>
    block_index = (ndocc * nvirt) + 2 * nvirt + 2 * ndocc + 3
    for col in range(block_index, block_index + (ndocc * nvirt)):
        o_orb = (col - block_index) // nvirt
        v_orb = (col - block_index) % nvirt + (SOMO2 + 1)
        xcish[1,col] = - np.sqrt(3) * rep_tens[o_orb, SOMO1, SOMO2, v_orb]
        xcish[col,1] = xcish[1,col]
    
    
    row_block_index = 3
    #25 <HS1|H|HS1>
    col_block_index = 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(row, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                xcish[row, col] = energy0 + orb_energies[SOMO1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, SOMO1, SOMO1] + 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] \
                             + 1.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb1] + 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb1]
            else:    
                xcish[row, col] = 0.5 * (rep_tens[o_orb2, SOMO1, SOMO1, o_orb1] + 0.5 * rep_tens[o_orb2, SOMO2, SOMO2, o_orb1]) - rep_tens[o_orb2, o_orb1, SOMO1, SOMO1]
            xcish[col, row] = xcish[row,col]
    #26 <HS1|H|HS2>
    col_block_index = ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                xcish[row, col] = rep_tens[SOMO1, o_orb1, o_orb1, SOMO2] - rep_tens[o_orb1, o_orb1, SOMO1, SOMO2] + 0.5 * rep_tens[SOMO1, SOMO2, SOMO1, SOMO1] \
                             + 0.5 * rep_tens[SOMO1, SOMO2, SOMO2, SOMO2]
            else:    
                xcish[row, col] = - rep_tens[o_orb1, o_orb2, SOMO1, SOMO2] - rep_tens[o_orb1, SOMO1, SOMO2, o_orb2]
            xcish[col, row] = xcish[row, col]
    #27 <HS1|H|SL1>
    col_block_index = 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb = col - col_block_index + (SOMO2 + 1)
            xcish[row, col] = rep_tens[o_orb, SOMO1, SOMO2, v_orb] - 2 * rep_tens[o_orb, SOMO2, SOMO1, v_orb]
            xcish[col, row] = xcish[row,col]
    #28 <HS1|H|SL2>
    col_block_index = nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb = col - col_block_index + (SOMO2 + 1)
            xcish[row, col] = - rep_tens[o_orb, SOMO1, SOMO1, v_orb]
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
                             + 1.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb1] + 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb1]
            else:    
                xcish[row, col] = 1.5 * (rep_tens[o_orb2, SOMO1, SOMO1, o_orb1] + rep_tens[o_orb2, o_orb1, SOMO2, SOMO2]) - 0.5 * rep_tens[o_orb2, SOMO2, SOMO2, o_orb1]
            xcish[col, row] = xcish[row,col]
    #32 <HS2|H|SL1>
    col_block_index = 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb = col - col_block_index + (SOMO2 + 1)
            xcish[row, col] = - rep_tens[o_orb, SOMO2, SOMO2, v_orb]
            xcish[col, row] = xcish[row, col]
    #33 <HS2|H|SL2>
    col_block_index = nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb = col - col_block_index + (SOMO2 + 1)
            xcish[row, col] = rep_tens[o_orb, SOMO2, SOMO1, v_orb] - 2 * rep_tens[o_orb, SOMO1, SOMO2, v_orb]
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
                xcish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[SOMO2] - rep_tens[v_orb1, v_orb1, SOMO2, SOMO2] + 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] \
                             - 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] + 1.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb1] + 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb1]
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
                xcish[row, col] = (1 / np.sqrt(2)) * (rep_tens[o_orb,SOMO2,v_orb1,v_orb1] + 1.5*rep_tens[o_orb,SOMO1,SOMO1,SOMO2] - 2*rep_tens[o_orb,v_orb1,v_orb1,SOMO2] - 0.5*rep_tens[o_orb,SOMO2,SOMO2,SOMO2])
            else:
                xcish[row, col] = (1 / np.sqrt(2)) * (rep_tens[o_orb, SOMO2, v_orb1, v_orb2] - 2 * rep_tens[v_orb1, SOMO2, v_orb2, o_orb])
            xcish[col,row] = xcish[row,col]
    #39 <SL1|H|HL2>
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
    
    
    row_block_index = nvirt + 2 * ndocc + 3
    #40 <SL2|H|SL2>
    col_block_index = nvirt + 2 * ndocc + 3
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(row, col_block_index + nvirt):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[SOMO1] - rep_tens[v_orb1, v_orb1, SOMO1, SOMO1] + 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] \
                             - 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2]+ 1.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb1] + 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb1]
            else:    
                xcish[row, col] = 1.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb2] - 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] - rep_tens[v_orb1, v_orb2, SOMO1, SOMO1] - rep_tens[v_orb1, v_orb2, SOMO2, SOMO2]
            xcish[col, row] = xcish[row,col]
    #41 <SL2|H|HL1>
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
    #42 <SL2|H|HL2>
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
                xcish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb1] - 0.25 * j00 + 1.5 * k00 \
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
                xcish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb1] - 0.25 * j00 - 0.5 * k00 \
                                 + rep_tens[o_orb1, SOMO1, SOMO1, o_orb1] + rep_tens[o_orb1, SOMO2, SOMO2, o_orb1] + rep_tens[v_orb1, SOMO1, SOMO1, v_orb1] + rep_tens[v_orb2, SOMO2, SOMO2, v_orb2]
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
    xcish[row_index, row_index] = energy0 - (0.25 * j00 ) - (0.5 * k00)
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
        xcish[row_index, col] = 0.5 * rep_tens[v_orb, SOMO1, SOMO2, SOMO1] - 0.5 * rep_tens[v_orb, SOMO2, SOMO2, SOMO2]
        xcish[col, row_index] = xcish[row_index,col]
    #50 <OS3|H|SL2>
    col_index = 2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 4
    for col in range(col_index, col_index + nvirt):
        v_orb = col - col_index + (SOMO2 + 1)
        xcish[row_index, col] = 0.5 * rep_tens[v_orb, SOMO1, SOMO1, SOMO1] - 0.5 * rep_tens[v_orb, SOMO2, SOMO2, SOMO1]
        xcish[col, row_index] = xcish[row_index, col]
    #51 <OS3|H|HL1> = 0
    #52 <OS3|H|HL2>
    col_index =  3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for col in range(block_index, block_index + (ndocc * nvirt)):
        o_orb = (col - block_index) // nvirt # Increase o_orb after every ndocc cols
        v_orb = (col - block_index) % nvirt + (SOMO2 + 1) # Increase v_orb then reset after ndocc cols
        xcish[0,col] = (1 / np.sqrt(2)) * (rep_tens[o_orb, SOMO1, SOMO1, v_orb] - rep_tens[o_orb, SOMO2, SOMO2, v_orb])
        xcish[col,0] = xcish[0,col]
    #53 <OS3|H|HL3>
    col_index =  3 * (ndocc * nvirt) + 4 * nvirt + 4 * ndocc + 4
    for col in range(block_index, block_index + (ndocc * nvirt)):
        o_orb = (col - block_index) // nvirt # Increase o_orb after every ndocc cols
        v_orb = (col - block_index) % nvirt + (SOMO2 + 1) # Increase v_orb then reset after ndocc cols
        xcish[0,col] = rep_tens[o_orb, SOMO1, SOMO1, v_orb] + rep_tens[o_orb, SOMO2, SOMO2, v_orb]
        xcish[col,0] = xcish[0,col]
    
    row_block_index = 2 * (ndocc * nvirt) + 2 * ndocc + 2 * ndocc + 4
    #54 <HS1|H|HS1>
    col_block_index = 2 * (ndocc * nvirt) + 2 * ndocc + 2 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(row, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                xcish[row, col] = energy0 + orb_energies[SOMO1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, SOMO1, SOMO1] + 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] \
                             - 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb1] + 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb1]
            else:    
                xcish[row, col] = 0.5 * (rep_tens[o_orb2, SOMO1, SOMO1, o_orb1] - rep_tens[o_orb2, o_orb1, SOMO1, SOMO1]) - 1.5 * rep_tens[o_orb2, SOMO2, SOMO2, o_orb1]
            xcish[col, row] = xcish[row,col]
    #55 <HS1|H|HS2>
    col_block_index = 2 * (ndocc * nvirt) + 2 * ndocc + 3 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb1 = row - row_block_index
        for col in range(col_block_index, col_block_index + ndocc):
            o_orb2 = col - col_block_index
            if o_orb1 == o_orb2:
                xcish[row, col] = rep_tens[SOMO1, o_orb1, o_orb1, SOMO2] - rep_tens[o_orb1, o_orb1, SOMO1, SOMO2] + 0.5 * rep_tens[SOMO2, SOMO1, SOMO1, SOMO1] \
                             + 0.5 * rep_tens[SOMO1, SOMO2, SOMO2, SOMO2]                                                               #CHECK RESULT, SAME AS SINGLET?
            else:    
                xcish[row, col] = rep_tens[o_orb2, o_orb1, SOMO1, SOMO2] - rep_tens[o_orb2, SOMO1, SOMO2, o_orb1]
            xcish[col, row] = xcish[row,col]
    #56 <HS1|H|SL1>
    col_block_index = 2 * (ndocc * nvirt) + 2 * ndocc + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb = col - col_block_index + (SOMO2 + 1)
            xcish[row, col] = - rep_tens[o_orb, SOMO1, SOMO2, v_orb]
            xcish[col, row] = xcish[row,col]
    #57 <HS1|H|SL2>
    col_block_index = 2 * (ndocc * nvirt) + 3 * ndocc + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb = col - col_block_index + (SOMO2 + 1)
            xcish[row, col] = rep_tens[o_orb, SOMO1, SOMO1, v_orb]
            xcish[col, row] = xcish[row,col]
    #58 <HS1|H|HL1>
    col_block_index = 2 * (ndocc * nvirt) + 4 * ndocc + 4 * ndocc + 4
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
    col_block_index = 3 * (ndocc * nvirt) + 4 * ndocc + 4 * ndocc + 4
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
    col_block_index = 4 * (ndocc * nvirt) + 4 * ndocc + 4 * ndocc + 4
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
                             - 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb1] + 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb1]
            else:
                xcish[row, col] = rep_tens[o_orb1, o_orb2, SOMO2, SOMO2] - 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb2] - 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb2]
            xcish[col, row] = xcish[row,col]
    #62 <HS2|H|SL1>
    col_block_index = 2 * (ndocc * nvirt) + 2 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb = col - col_block_index + (SOMO2 + 1)
            xcish[row, col] = - rep_tens[o_orb, SOMO2, SOMO2, v_orb]
            xcish[col, row] = xcish[row,col]
    #63 <HS2|H|SL2>
    col_block_index = 2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + ndocc):
        o_orb = row - row_block_index
        for col in range(col_block_index, col_block_index + nvirt):
            v_orb = col - col_block_index + (SOMO2 + 1)
            xcish[row, col] = rep_tens[o_orb, SOMO2, SOMO1, v_orb]
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
                xcish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[SOMO2] - rep_tens[v_orb1, v_orb1, SOMO2, SOMO2] + 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] \
                             - 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] - 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb1] + 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb1]
            else:    
                xcish[row, col] =  0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] - rep_tens[v_orb1, v_orb2, SOMO2, SOMO2] - 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb2]
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
                xcish[row, col] = (1 / np.sqrt(2)) * (rep_tens[o_orb, SOMO2, v_orb1, v_orb1] - 0.5 * rep_tens[o_orb, SOMO1, SOMO1, SOMO2] - 0.5 * rep_tens[o_orb, SOMO2, SOMO2, SOMO2] \
                                  - 2 * rep_tens[o_orb, v_orb1, v_orb1, SOMO2])
            else:
                xcish[row, col] = (1 / np.sqrt(2)) * (rep_tens[o_orb, SOMO2, v_orb1, v_orb2] - 2 * rep_tens[v_orb1, SOMO2, v_orb2, o_orb])
            xcish[col,row] = xcish[row,col]
    #70 <SL1|H|HL2>
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
    #71 <SL1|H|HL3>
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
    
    
    row_block_index = 2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 4
    #72 <SL2|H|SL2>
    col_block_index = 2 * (ndocc * nvirt) + 3 * nvirt + 4 * ndocc + 4
    for row in range(row_block_index, row_block_index + nvirt):
        v_orb1 = row - row_block_index + (SOMO2 + 1)
        for col in range(row, col_block_index + nvirt):
            v_orb2 = col - col_block_index + (SOMO2 + 1)
            if v_orb1 == v_orb2:
                xcish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[SOMO1] - rep_tens[v_orb1, v_orb1, SOMO1, SOMO1] + 0.25 * rep_tens[SOMO1, SOMO1, SOMO1, SOMO1] \
                             - 0.25 * rep_tens[SOMO2, SOMO2, SOMO2, SOMO2] - 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb1] + 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb1]
            else:    
                xcish[row, col] =  0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] - 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb2] - rep_tens[v_orb1, v_orb2, SOMO1, SOMO1] \
                                  - rep_tens[v_orb1, v_orb2, SOMO2, SOMO2]
            xcish[col, row] = xcish[row,col]
    #73 <SL2|H|HL1>
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
    #74 <SL2|H|HL2>
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
    #75 <SL2|H|HL3>
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
                xcish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb1] - 0.25 * j00 - 0.5 * k00 \
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
                xcish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb1] - 0.25 * j00 + 1.5 * k00
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
                xcish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb1] - 0.25 * j00 - 0.5 * k00 \
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
                xcish[row, col] = energy0 + orb_energies[v_orb1] - orb_energies[o_orb1] - rep_tens[o_orb1, o_orb1, v_orb1, v_orb1] - 0.25 * j00 - 0.5 * k00 \
                                 - 0.5 * (rep_tens[o_orb1, SOMO1, SOMO1, o_orb1] + rep_tens[o_orb1, SOMO2, SOMO2, o_orb1] + rep_tens[v_orb1, SOMO1, SOMO1, v_orb1] + rep_tens[v_orb1, SOMO2, SOMO2, v_orb1])
            elif v_orb1 == v_orb2 and o_orb1 != o_orb2:    
                xcish[row, col] =  - rep_tens[o_orb1, o_orb2, v_orb1, v_orb1] - 0.5 * rep_tens[o_orb1, SOMO1, SOMO1, o_orb2] - 0.5 * rep_tens[o_orb1, SOMO2, SOMO2, o_orb2]
            elif o_orb1 == o_orb2 and v_orb1 != v_orb2:
                xcish[row, col] = - rep_tens[v_orb1, v_orb2, o_orb1, o_orb1] - 0.5 * rep_tens[v_orb1, SOMO1, SOMO1, v_orb2] - 0.5 * rep_tens[v_orb1, SOMO2, SOMO2, v_orb2]
            else:
                xcish[row, col] = - rep_tens[o_orb1, o_orb2, v_orb1, v_orb2]
            xcish[col, row] = xcish[row,col]
    
    
    return xcish

def hetero_xcis_rot(ndocc,norbs,coords,atoms,energy0,repulsion,orb_energies,hf_orbs, file):
    with open(f'Excited_States/{file}_excitedstates.xyz','w') as out:
        print("")
        print("------------------------")
        print("Starting ExROPPP calculation for monoradical heterocycle in rotated basis")
        print("------------------------\n")

        out.write("")
        out.write("------------------------")
        out.write("Starting ExROPPP calculation for monoradical heterocycle in rotated basis")
        out.write("------------------------\n")
        # Transform 2-el ingrls into mo basis
        rep_tens = transform(repulsion,hf_orbs)
        # Construct CIS Hamiltonian
        het_ham_rot = hetero_cis_ham_rot(ndocc,norbs,energy0,orb_energies,rep_tens)
        #print("Checking that the Hamiltonian is symmetric (a value of zero means matrix is symmetric) ... ")
        #print("Frobenius norm of matrix - matrix transpose = %f.\n" %(linalg.norm(het_ham_rot-het_ham_rot.T)))
        #print(linalg.norm(het_ham_rot-het_ham_rot.T))
        nunocc = norbs-ndocc-1
        nstates = 3*ndocc*nunocc +ndocc+nunocc +1
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
        # Diagonalize CIS Hamiltonianfor first rng excited states
        if rng<nstates:
            print("Diagonalizing Hamiltonian using the sparse matrix method ...\n")
            out.write("Diagonalizing Hamiltonian using the sparse matrix method ...\n")

            cis_energies,cis_coeffs=sp.eigsh(het_ham_rot,k=rng,which="SA")
        elif rng==nstates:
            print("Diagonalizing Hamiltonian using the dense matrix method ...\n")
            out.write("Diagonalizing Hamiltonian using the dense matrix method ...\n")

            cis_energies,cis_coeffs=linalg.eigh(het_ham_rot)
        dip_array = dipole_cis(coords,atoms,norbs,hf_orbs,ndocc,nstates)
        aku=np.einsum("ijx,jk",dip_array,cis_coeffs)
        mu0u=np.einsum("j,jix",cis_coeffs[:,0].T,aku)
        osc_array=np.zeros_like(cis_energies)
        s2_array=np.zeros_like(cis_energies)
        print("Ground state energy relative to E(|0>): %04.3f eV"%(cis_energies[0]-energy0))
        out.write("Ground state energy relative to E(|0>): %04.3f eV\n"%(cis_energies[0]-energy0))

        rt = 2.**.5
        strng = ""
        eqn=0
        for i in range(rng): # Loop over CIS states
            if cis_energies[i]-cis_energies[0] > cutoff_energy:
                break
            print("State %s %04.3f eV \n" % (i,cis_energies[i]-cis_energies[0])) #print("State %s %04.3f eV \n" % (i,energy-cis_energies[0]))
            print("Excitation    CI Coef    CI C*rt(2)")
            out.write("State %s %04.3f eV \n" % (i,cis_energies[i]-cis_energies[0])) #print("State %s %04.3f eV \n" % (i,energy-cis_energies[0]))
            out.write("Excitation    CI Coef    CI C*rt(2)\n")
            spin = 0 # initialise total spin
            for j in range (cis_coeffs.shape[0]): # Loop over configurations in each CIS state   
            # if configuration is the ground determinant
                if j == 0: 
                    if np.absolute(cis_coeffs[j,i]) > 1e-2:
                        print('|0>           %10.5f'  %(cis_coeffs[j,i]))
                        out.write('|0>           %10.5f\n'  %(cis_coeffs[j,i]))

                    spin += 0.75*cis_coeffs[j,i]**2 # (S=0.5)
                    continue
            # if configuration is |ibar->0bar>     
                elif j>0 and j<=ndocc:
                    iorb = j-1
                    str1 = str(ndocc-iorb) + "bar" #str(iorb) + "bar" #
                    str2 = "0bar" #"3bar"#
                    spin += 0.75*cis_coeffs[j,i]**2 # (S=0.5)
        # if configuration is |0->j'> 
                elif j>ndocc and j<=ndocc+nunocc:
                    jorb = j 
                    str1 = "0" 
                    str2 = str(jorb-ndocc)+"'" #str(jorb) #
                    spin += 0.75*cis_coeffs[j,i]**2 # (S=0.5)
            # if configuration is |Qi->j'>
                elif j>ndocc+nunocc and j<=ndocc+nunocc + ndocc*nunocc:
                    iorb = int(np.floor((j-ndocc-nunocc-1)/nunocc))
                    jorb = (j-ndocc-nunocc-1)-iorb*nunocc+ndocc +1
                    str1 = "Q " + str(ndocc-iorb)
                    str2 = str(jorb-ndocc)+"'" 
                    spin += 3.75*cis_coeffs[j,i]**2 # (S=1.5) 
            # if configuration is |D(S)i->j'> (bright doublet state)
                elif j>ndocc+nunocc + ndocc*nunocc and j<=ndocc+nunocc + 2*ndocc*nunocc:
                    iorb = int(np.floor((j-ndocc-nunocc-ndocc*nunocc-1)/nunocc))
                    jorb = (j-ndocc-nunocc-ndocc*nunocc-1)-iorb*nunocc+ndocc +1
                    str1 = "D(S) " +str(ndocc-iorb)
                    str2 = str(jorb-ndocc)+"'"
                    spin += 0.75*cis_coeffs[j,i]**2 # (S=0.5)
            #if configuration is |D(T)i->j'> (dark doublet state)
                elif j>ndocc+nunocc + 2*ndocc*nunocc:
                    iorb = int(np.floor((j-ndocc-nunocc-2*ndocc*nunocc-1)/nunocc))
                    jorb = (j-ndocc-nunocc-2*ndocc*nunocc-1)-iorb*nunocc+ndocc +1
                    str1 = "D(T) "+str(ndocc-iorb)
                    str2 = str(jorb-ndocc)+"'"
                    spin += 0.75*cis_coeffs[j,i]**2 # (S=0.5)
                if np.absolute(cis_coeffs[j,i]) > 1e-1:
                    print("%s->%s %10.5f %10.5f " \
                    %(str1,str2,cis_coeffs[j,i],cis_coeffs[j,i]*rt))
                    out.write("%s->%s %10.5f %10.5f\n" \
                    %(str1,str2,cis_coeffs[j,i],cis_coeffs[j,i]*rt))
            if i==0:
                print("\n<S**2>: %04.3f" %spin)
                print("--------------------------------------------------------------------\n")
                out.write("<S**2>: %04.3f\n" %spin)
                out.write("--------------------------------------------------------------------\n")
                continue
            osc = 2.0/3.0*((cis_energies[i]-cis_energies[0])/toev)*(mu0u[i,0]**2+mu0u[i,1]**2+mu0u[i,2]**2) 
            print("")
            print("TDMX:%04.3f   TDMY:%04.3f   TDMZ:%04.3f   Oscillator Strength:%05.5f   <S**2>: %04.3f" % (mu0u[i,0], mu0u[i,1], mu0u[i,2], osc, spin))
            print("--------------------------------------------------------------------\n")
            out.write("")
            out.write("TDMX:%04.3f   TDMY:%04.3f   TDMZ:%04.3f   Oscillator Strength:%05.5f   <S**2>: %04.3f\n" % (mu0u[i,0], mu0u[i,1], mu0u[i,2], osc, spin))
            out.write("--------------------------------------------------------------------\n")
            strng = strng + broaden(FWHM,osc,cis_energies[i]-cis_energies[0])
            #eqn+=broaden_as_fn(FWHM,osc,cis_energies[i]-cis_energies[0])
            osc_array[i]=osc
            s2_array[i]=spin
        strng = strng[1:]   
    return strng, cis_energies-cis_energies[0],osc_array,s2_array



def rad_calc(file,params):
    filename = os.path.basename(file)
    coord,atoms_array,coord_w_h,dist_array,nelec,ndocc,n_list,natoms_c,natoms_n,natoms_cl,energy0,one_body,two_body,orb_energy,hf_orbs,fock_mat=main_scf(file,params)
    com,coord = re_center(coord,atoms_array,coord_w_h)
    hf_orbs = orb_sign(hf_orbs,orb_energy,nelec,dist_array,alt)
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
    strng,ci_energies_array,osc_array,s2_array = ci_rot(ndocc, natoms, coord, atoms_array, energy0, two_body, orb_energy, hf_orbs, file, ci_type = 'XCIS')
    return strng, ci_energies_array, osc_array, s2_array  #return gnuplot data for plotting spectrum




