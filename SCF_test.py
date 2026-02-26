import numpy as np
import scipy.linalg as linalg
from ExROPPP_settings_opt import *



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
    prefactor_matrix[[ndocc, ndocc + 1], [ndocc, ndocc + 1]] = 1 # SOMOs are not multiplied by 1 rather than 2.
    density = orbs[:,:ndocc] @ (prefactor_matrix @ orbs[:,:ndocc].T) # P = C_occ *  Prefactor * C_occ^T  
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
    """Calculates the Coulomb self-repulsion integral (J00) for the SOMO.

    Args:
        orbs (ndarray): Matrix of orbital coefficients (rows=atoms, cols=orbitals).
        repulsion (ndarray): Matrix of inter-atomic electron repulsion integrals.
        ndocc (int): The number of doubly-occupied orbitals. Gives the appropriate index for the SOMO.

    Returns:
        J00 (float): The calculated Coulomb repulsion term for the SOMO.
    """
    J00 = 0
    for l in range(orbs.shape[0]): # atom l
        for m in range(orbs.shape[0]): # atom m
            J00 += orbs[l,ndocc]**2 * orbs[m,ndocc]**2 * repulsion[l,m]
    return J00

#Function to calculate open-shell SCF energy
def energy(hopping,repulsion,fock_mat,density,orbs,ndocc):
    """Calculates the total open-shell SCF energy.

    Returns:
        float: The total calculated SCF energy of the system.
    """
    J00 = compute_j00(orbs,repulsion,ndocc)
    return 0.5 * (np.dot(density.flatten(), hopping.flatten()) + np.dot(density.flatten(), fock_mat.flatten())) - 0.25 * J00

#Main HF function
def main_scf(file, params, maxcycles=500, d_tol=1e-7):
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
    guess_dens = density(evecs,natoms,ndocc)
#iterate until convergence 
    energy1=0
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
        fock_mat = fock(repulsion,hopping,guess_dens,natoms_c,natoms_n,natoms,n_list)
        evals, orbs = np.linalg.eigh(fock_mat)
        dens = density(orbs,natoms,ndocc)
        energy2 = energy(hopping,repulsion,fock_mat,dens,orbs,ndocc)
        conv_crit = np.absolute(guess_dens-dens).max()
        print(iter, energy2, conv_crit, energy2 - energy1)
        if conv_crit < d_tol:
            return coord,atoms_array,coord_w_h,dist_array,nelec,ndocc,n_list,natoms_c,natoms_n,natoms_cl,energy2,hopping,repulsion,evals,orbs,fock_mat
        if energy2 > energy1:
            print('\nEnergy rises!')
        energy1 = energy2
        guess_dens = dens

    