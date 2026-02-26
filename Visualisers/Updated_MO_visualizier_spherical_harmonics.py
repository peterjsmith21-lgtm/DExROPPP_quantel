import numpy as np
import os
import re
import matplotlib.pyplot as plt
from itertools import combinations
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.transform import Rotation as R
import argparse

# Get the coolwarm colormap
coolwarm = plt.get_cmap('coolwarm')
# Create a new colormap with only the cool part (0.0 to 0.5)
cool_cmap = LinearSegmentedColormap.from_list('cool', coolwarm(np.linspace(0, 0.3, 256)))
warm_cmap = LinearSegmentedColormap.from_list('warm', coolwarm(np.linspace(0.7, 1, 256)))
grey_cmap = LinearSegmentedColormap.from_list('warm', coolwarm(np.linspace(0.5, 1, 256)))

class MolecularOrbitals:
    """
    A class to represent molecular orbitals extracted from an output file.

    Attributes
    ----------
    output_file_path : str
        Path to the output file containing molecular orbital data.
    coordinates : dict
        Dictionary containing atomic coordinates.
    eigenvectors : list
        List of eigenvectors extracted from the output file.
    basis : list
        List of basis coordinates for non-zero eigenvectors.
    non_zero_eigenvectors : list
        List of non-zero eigenvectors.
    mos : dict
        Dictionary containing molecular orbitals.
    num : int
        Number of molecular orbitals.

    Methods
    -------
    mos_num() -> int:
        Returns the number of molecular orbitals.
    extract_coordinate() -> dict:
        Extracts and returns a dictionary of atomic coordinates from the output file.
    extract_eigenvectors() -> list:
        Extracts and returns a list of eigenvectors from the output file.
    extract_basis() -> tuple[list, list]:
        Extracts and returns basis coordinates and non-zero eigenvectors.
    mo_dict() -> dict:
        Creates and returns a dictionary of molecular orbitals.
    """

    def __init__(self, output_file_path, compound_name):
        """
        Constructs all the necessary attributes for the MolecularOrbitals object.

        Parameters
        ----------
        output_file_path : str
            Path to the output file containing molecular orbital data.
        """
        self.output_file_path = output_file_path
        self.coordinates = self.extract_coordinate()
        self.eigenvectors = self.extract_eigenvectors()
        self.basis, self.non_zero_eigenvectors = self.extract_basis()
        self.mos = self.mo_dict()
        self.num = self.mos_num()
        self.name = compound_name
        self.somo_index = self.calculate_somo_index()
    
    def calculate_somo_index(self) -> int:
        """
        Calculates and returns the SOMO index based on valence electrons in the output file.

        Returns
        -------
        int
            SOMO index calculated as the total number of valence electrons (made even by adding 1 if odd) divided by 2.
        """
        with open(self.output_file_path, 'r') as read_mos:
            all_data_str = read_mos.read()

        # Initialize counters for different atom types
        valence_electron_count = 0

        # Patterns to match atom types and coordinates
        coord_pattern_C = re.compile(r'C\s+6\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)')
        coord_pattern_N1 = re.compile(r'N1\s+7\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)')
        coord_pattern_N2 = re.compile(r'N2\s+7\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)')
        coord_pattern_Cl = re.compile(r'Cl\s+17\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)')

        # Count valence electrons based on atom matches
        valence_electron_count += len(coord_pattern_C.findall(all_data_str)) * 1  # Carbon contributes 1 valence electron
        valence_electron_count += len(coord_pattern_N2.findall(all_data_str)) * 2  # N2 contributes 2 valence electrons
        valence_electron_count += len(coord_pattern_N1.findall(all_data_str)) * 1  # N1 contributes 1 valence electron
        valence_electron_count += len(coord_pattern_Cl.findall(all_data_str)) * 2  # Chlorine contributes 2 valence electrons

        # Add 1 to make the total even if it's odd, then divide by 2 to calculate SOMO
        if valence_electron_count % 2 != 0:
            valence_electron_count += 1

        somo_index = valence_electron_count // 2
        return somo_index


    def mos_num(self) -> int:
        """
        Returns the number of molecular orbitals.

        Returns
        -------
        int
            Number of molecular orbitals.
        """
        return len(self.mos)
    
    def extract_coordinate(self) -> dict:
        """
        Extracts and returns a dictionary of atomic coordinates from the output file.

        Returns
        -------
        dict
            Dictionary where each key (element) has value (coordinate) in a list.
        """
        with open(self.output_file_path, 'r') as read_mos:
            all_data_str = read_mos.read()

        counter = 1
        coordinates = {}

        coord_pattern_C = re.compile(r'C\s+6\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)')
        coord_pattern_N = re.compile(r'N\s+7\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)')
        coord_pattern_N1 = re.compile(r'N1\s+7\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)')        
        coord_pattern_N2 = re.compile(r'N2\s+7\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)')
        coord_pattern_Cl = re.compile(r'Cl\s+17\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)')

        for match in coord_pattern_C.finditer(all_data_str):
            coordinates[f'C{counter}'] = [float(match.group(1)), 
                                          float(match.group(2)), 
                                          float(match.group(3))]
            counter += 1

        for match in coord_pattern_N.finditer(all_data_str):
            coordinates[f'N{counter}'] = [float(match.group(1)), 
                                          float(match.group(2)), 
                                          float(match.group(3))]
            counter += 1

        for match in coord_pattern_N2.finditer(all_data_str):
            coordinates[f'N{counter}'] = [float(match.group(1)), 
                                          float(match.group(2)), 
                                          float(match.group(3))]
            counter += 1
        
        for match in coord_pattern_N1.finditer(all_data_str):
            coordinates[f'N{counter}'] = [float(match.group(1)), 
                                          float(match.group(2)), 
                                          float(match.group(3))]
            counter += 1

        for match in coord_pattern_Cl.finditer(all_data_str):
            coordinates[f'Cl{counter}'] = [float(match.group(1)), 
                                           float(match.group(2)), 
                                           float(match.group(3))]
            counter += 1
        
        return coordinates

    def extract_eigenvectors(self) -> list:
        """
        Extracts and returns all the eigenvectors from the output file.

        Returns
        -------
        list
            List of eigenvectors.
        """
        eigenvectors = []
        collecting = False
        with open(self.output_file_path, 'r') as f2:
            all_data_lines = f2.readlines()
        line_pattern = re.compile(r'\s*(\d+)\s+([A-Z][a-zA-Z]?\s*\d+)\s+([SXYZ])\s+([-.\d]+)\s*')
        for line in all_data_lines:
            if "EIGENVECTORS" in line:
                collecting = True
                coefficients = []
            elif "...... END OF ROHF CALCULATION ......" in line:
                if collecting:
                    eigenvectors.append(coefficients)
                    collecting = False
            elif collecting:
                match = line_pattern.match(line)
                if match:
                    atom_info = re.sub(r'\s+', '', match.group(2))  # Remove any extra spaces
                    formatted_basis = [atom_info, match.group(3), float(match.group(4))]
                    coefficients.append(formatted_basis)
        
        return eigenvectors

    def extract_basis(self) -> tuple[list, list]:
        """
        Extracts and returns basis coordinates and non-zero eigenvectors.

        Returns
        -------
        tuple
            Two lists: non-zero eigenvectors and basis coordinates.
        """
        basis = []
        non_zero_eigenvectors = []
        counter = 0
        for i in range(len(self.eigenvectors)):
            basis.append([self.coordinates.get(eigenvector[0].strip()) for eigenvector in self.eigenvectors[i] if eigenvector[-1] != 0.0])
            
            non_zero_eigenvectors.append([eigenvector for eigenvector in self.eigenvectors[i] if eigenvector[-1] != 0.0])
            counter += 1
        # print(basis)
        return basis, non_zero_eigenvectors

    def mo_dict(self) -> dict:
        """
        Creates and returns a dictionary of molecular orbitals.

        Returns
        -------
        dict
            Dictionary of molecular orbitals, each composed of an array of non-zero n_atoms * 4 (x,y,z,coef).
        """
        mos = {}
        counter = 1        
        # print(self.non_zero_eigenvectors)
        for bas, vec in zip(self.basis, self.non_zero_eigenvectors):
            for_plot = []
            for i in range(len(bas)):
                basis_coef = float(np.array(vec)[i,-1])
                
                
                coord = np.array(bas)[i].astype(np.float64)
                combined = np.append(coord, basis_coef)
                for_plot.append(combined)
            mos[f'MO{counter}'] = np.array(for_plot)
            counter += 1     
        return mos

class MODrawer(MolecularOrbitals):
    """
    A class to draw and visualize molecular orbitals.

    Methods
    -------
    set_axes_equal(ax):
        Sets equal scaling for 3D plot axes.
    mo_drawer(key, orbital_scale=2, viewing_angle=[45, -90], figsize=(14, 10), transparency=0.6, bondthickness=2, savefig=False, re_orientate=None):
        Draws the molecular orbital specified by the key.
    """
    def __init__(self, output_file_path, compound_name):
        
        super().__init__(output_file_path, compound_name)
        self.output_file_path = output_file_path
        self.coordinates = self.extract_coordinate()
        self.eigenvectors = self.extract_eigenvectors()
        self.basis, self.non_zero_eigenvectors = self.extract_basis()
        self.mos = self.mo_dict()
        self.num = self.mos_num()
        self.name = compound_name
        self.somo_index = self.calculate_somo_index()
        
    def set_axes_equal(self, ax):
        """
        Sets equal scaling for 3D plot axes.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.Axes3DSubplot
            3D axes to set equal scaling.
        """
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        plot_radius = 0.3 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
    @staticmethod
    def axis_angle_to_quaternion(axis, angle):
        # axis = axis / np.linalg.norm(axis)  # Normalize the axis
        sin_half_angle = np.sin(angle / 2.0)
        cos_half_angle = np.cos(angle / 2.0)
        return np.array([cos_half_angle, axis[0] * sin_half_angle, axis[1] * sin_half_angle, axis[2] * sin_half_angle])

    @staticmethod
    def quaternion_conjugate(quaternion):
        q = quaternion.copy()
        q[1:] = -q[1:]
        return q

    @staticmethod
    def quaternion_multiply(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    @staticmethod
    def rotate_mine(x, y, z, axis, angle):
        q = MODrawer.axis_angle_to_quaternion(axis, angle)
        q_conj = MODrawer.quaternion_conjugate(q)
        
        rotated_points = []
        for point in zip(x.flatten(), y.flatten(), z.flatten()):
            p = np.array([0, point[0], point[1], point[2]])
            rotated_p = MODrawer.quaternion_multiply(MODrawer.quaternion_multiply(q, p), q_conj)
            rotated_points.append(rotated_p[1:])
        
        rotated_points = np.array(rotated_points)
        return rotated_points[:, 0].reshape(x.shape), rotated_points[:, 1].reshape(y.shape), rotated_points[:, 2].reshape(z.shape)

    def mo_drawer(self, key, orbital_scale=2, viewing_angle=[-90, 0,0], figsize=(14, 10), transparency=0.6, bondthickness = 2, savefig = False, re_orientate =  True):
        """
        Draws the molecular orbital specified by the key.

        Parameters
        ----------
        key : str
            Key for the molecular orbital to be drawn.
        orbital_scale : float, optional
            Scale of the orbital (default is 2).
        viewing_angle : list, optional
            Viewing angles for elevation and azimuth (default is [45, -90]).
        figsize : tuple, optional
            Size of the figure (default is (18, 14)).
        transparency : float, optional
            Transparency of the orbital surface (default is 0.6).
        re_orientate : tuple or list, optional
            Axis and angle for reorientation (default is None).
        """
        phi = np.linspace(0, 2 * np.pi, 100)
        theta = np.linspace(0, np.pi, 100)
        phi, theta = np.meshgrid(phi, theta)
        plt.style.use('seaborn-v0_8-white')
        # plt.style.use('classic')
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')
        if key == 'SOMO':
            key = f'MO{self.somo_index}'
        elif key == 'HOMO':
            key = f'MO{self.somo_index-1}'
        elif key == 'LUMO':
            key = f'MO{self.somo_index+1}'
        basis = self.mos.get(key)
        atom_centres = np.array(basis[:, :-1])
        atom_coefs = np.array(basis[:, -1])
        normals = []
        def sph_harm_2pz(theta):
            return np.cos(theta)
        
        def rotate(x,y,z,axis, angle):
            rotation = R.from_rotvec(angle * np.array(axis))  # axis-angle to quaternion internally
            # Flatten the input arrays (x, y, z) to form a list of 3D points
            points = np.column_stack((x.flatten(), y.flatten(), z.flatten()))

            # Apply the rotation to all points
            rotated_points = rotation.apply(points)

            # Reshape the rotated points back to the original shape
            x_rot = rotated_points[:, 0].reshape(x.shape)
            y_rot = rotated_points[:, 1].reshape(y.shape)
            z_rot = rotated_points[:, 2].reshape(z.shape)

            return x_rot, y_rot, z_rot
        
        for centre, atom_coef in zip(atom_centres, atom_coefs):

            
           
            # og spherical coordinates
            r = orbital_scale * np.abs(atom_coef) * sph_harm_2pz(theta)
            r2 = -orbital_scale * np.abs(atom_coef) * sph_harm_2pz(theta)

            if re_orientate is None:
                x = r * np.sin(theta) * np.cos(phi) + centre[0]
                y = r * np.sin(theta) * np.sin(phi) + centre[1]
                z = r * np.cos(theta) + centre[2]

                x2 = r2 * np.sin(theta) * np.cos(phi) + centre[0]
                y2 = r2 * np.sin(theta) * np.sin(phi) + centre[1]
                z2 = r2 * np.cos(theta) + centre[2]

                if atom_coef >= 0:
                    ax.plot_surface(x, y, z, cmap=warm_cmap)
                    ax.plot_surface(x2, y2, z2, cmap=cool_cmap)
                else:
                    ax.plot_surface(x, y, z, cmap=warm_cmap)
                    ax.plot_surface(x2, y2, z2, cmap=cool_cmap)
            else:
                
                # Find two nearest neighbors for the current atom `centre`
                distances = np.linalg.norm(atom_centres - centre, axis=1)
                sorted_indices = np.argsort(distances)  # Sort distances, smallest first
                nearest_neighbors = sorted_indices[1:3]
                # Compute vectors from the current atom to its two nearest neighbors
                vec1 = atom_centres[nearest_neighbors[0]] - centre
                vec2 = atom_centres[nearest_neighbors[1]] - centre
                # compute the cross product
                normal_vec = np.cross(vec1, vec2)
                normal_vec /= np.linalg.norm(normal_vec)

                
                global_z = np.array([0.01,0.01 ,1])
                global_z = global_z/np.linalg.norm(global_z)

                cos_theta = np.dot(normal_vec, global_z + centre) / (np.linalg.norm(normal_vec) * np.linalg.norm(global_z+centre))
                angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                angle_deg = np.degrees(angle_rad)

                if angle_deg > 90:  
                    normal_vec =normal_vec* -1
                if np.dot(normal_vec, global_z) < 0:
                    normal_vec *= -1
                normals.append(normal_vec)

        ref_normal = normals[0]
        for i, vec in enumerate(normals):
            if np.dot(ref_normal, vec) <0:
                normals[i] = -1*vec
        for centre, atom_coef, n_vec in zip(atom_centres, atom_coefs, normals):
            global_z = np.array([0.0001,0.00001 ,1])
            global_z = global_z/np.linalg.norm(global_z)
            axis_of_rotation = np.cross(global_z,n_vec)
            # If the vectors are not collinear, normalize the axis
            if np.linalg.norm(axis_of_rotation) > 1e-6:
                axis_of_rotation /= np.linalg.norm(axis_of_rotation)
            # plt.plot([centre[0],centre[0]+n_vec[0]*4],[centre[1],centre[1]+n_vec[1]*4],[centre[2],centre[2]+n_vec[2]*4], color = 'red')

            # Find the angle between the normal vector and the z-axis
            angle_of_rotation = np.arccos(np.clip(np.dot(n_vec, global_z), -1.0, 1.0))
            r = orbital_scale * np.abs(atom_coef) * sph_harm_2pz(theta)
            r2 = -orbital_scale * np.abs(atom_coef) * sph_harm_2pz(theta)
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)

            x2 = r2 * np.sin(theta) * np.cos(phi)
            y2 = r2 * np.sin(theta) * np.sin(phi)
            z2 = r2 * np.cos(theta)
            
            x_rot, y_rot, z_rot = rotate(x, y, z, axis_of_rotation, angle_of_rotation)
            x2_rot, y2_rot, z2_rot = rotate(x2, y2, z2, axis_of_rotation, angle_of_rotation)

            x_rot += centre[0]
            y_rot += centre[1]
            z_rot += centre[2]

            x2_rot += centre[0]
            y2_rot += centre[1]
            z2_rot += centre[2]
            if atom_coef >= 0:
                ax.plot_surface(x_rot, y_rot, z_rot, cmap=warm_cmap, alpha = transparency)
                ax.plot_surface(x2_rot, y2_rot, z2_rot, cmap=cool_cmap, alpha = transparency)
            elif atom_coef < 0:
                ax.plot_surface(x_rot, y_rot, z_rot, cmap=cool_cmap, alpha = transparency)
                ax.plot_surface(x2_rot, y2_rot, z2_rot, cmap=warm_cmap, alpha = transparency)
  

        # Creating bonds
        list_coords = np.asarray(list(self.coordinates.values()))
        dist_pair = np.asarray(list(combinations(list_coords, 2)))

        # Creating a list of atom types
        list_atoms = np.asarray(list(self.coordinates.keys()))
        atom_pair = np.asarray(list(combinations(list_atoms, 2)))

        # Tolerance using C-Cl bond in Bohr of 1.77 Angstroms
        bool_array = np.linalg.norm(dist_pair[:,1] - dist_pair[:,0], axis=1) < 3.35
        
        atom_array = atom_pair[bool_array]
        bond_array = dist_pair[bool_array]

        for bond, atoms in zip(bond_array, atom_array):
        
            # Bonds are in pair of coordinatesm so zip pairs the respeictve coord together which are use to plot
            x_coords, y_coords, z_coords = zip(*bond)
            #atom info
            atom1, atom2 = atoms
               # Calculate the midpoint coordinates
            mid_point_x = (x_coords[0] + x_coords[1]) / 2
            mid_point_y = (y_coords[0] + y_coords[1]) / 2
            mid_point_z = (z_coords[0] + z_coords[1]) / 2

            # Separate conditions for each bond type

            # C-Cl bond (from black to green)
            if atom1.startswith('C') and atom2.startswith('Cl'):
                # print('C-Cl bond detected')
                # First half of the bond
                ax.plot([x_coords[0], mid_point_x], [y_coords[0], mid_point_y], [z_coords[0], mid_point_z], color='black', linewidth=bondthickness)
                # Second half of the bond
                ax.plot([mid_point_x, x_coords[1]], [mid_point_y, y_coords[1]], [mid_point_z, z_coords[1]], color='green', linewidth=bondthickness)

            # Cl-C bond (from green to black)
            elif atom1.startswith('Cl') and atom2.startswith('C'):
                # print('Cl-C bond detected')
                # First half of the bond
                ax.plot([x_coords[0], mid_point_x], [y_coords[0], mid_point_y], [z_coords[0], mid_point_z], color='green', linewidth=bondthickness)
                # Second half of the bond
                ax.plot([mid_point_x, x_coords[1]], [mid_point_y, y_coords[1]], [mid_point_z, z_coords[1]], color='black', linewidth=bondthickness)

            # C-N bond (from black to blue)
            elif atom1.startswith('C') and atom2.startswith('N'):
                # print('C-N bond detected')
                # First half of the bond
                ax.plot([x_coords[0], mid_point_x], [y_coords[0], mid_point_y], [z_coords[0], mid_point_z], color='black', linewidth=bondthickness)
                # Second half of the bond
                ax.plot([mid_point_x, x_coords[1]], [mid_point_y, y_coords[1]], [mid_point_z, z_coords[1]], color='blue', linewidth=bondthickness)

            # N-C bond (from blue to black)
            elif atom1.startswith('N') and atom2.startswith('C'):
                # print('N-C bond detected')
                # First half of the bond
                ax.plot([x_coords[0], mid_point_x], [y_coords[0], mid_point_y], [z_coords[0], mid_point_z], color='blue', linewidth=bondthickness)
                # Second half of the bond
                ax.plot([mid_point_x, x_coords[1]], [mid_point_y, y_coords[1]], [mid_point_z, z_coords[1]], color='black', linewidth=bondthickness)

            # C-C bond (black)
            elif atom1.startswith('C') and atom2.startswith('C'):
                # print('C-C bond detected')
                ax.plot(x_coords, y_coords, z_coords, color='black', linewidth=bondthickness)

            # Cl-Cl bond (green)
            elif atom1.startswith('Cl') and atom2.startswith('Cl'):
                # print('Cl-Cl bond detected')
                ax.plot(x_coords, y_coords, z_coords, color='green', linewidth=bondthickness)

            # N-N bond (blue)
            elif atom1.startswith('N') and atom2.startswith('N'):
                # print('N-N bond detected')
                ax.plot(x_coords, y_coords, z_coords, color='blue', linewidth=bondthickness)

            # Default case if none of the above apply
            else:
                # print('Default bond detected')
                ax.plot(x_coords, y_coords, z_coords, color='black', linewidth=bondthickness)

        self.set_axes_equal(ax)
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.xaxis.line.set_color('w')
        ax.yaxis.line.set_color('w')
        ax.zaxis.line.set_color('w')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=viewing_angle[0], azim=viewing_angle[1], roll = viewing_angle[2])
        # ax.text2D(0.5, 0.95, f'{self.name} ExROPPP {key} diagram', transform=ax.transAxes, ha='center')

        # Adding the inset with the orientation
        # inset_ax = fig.add_axes([0.8, 0.8, 0.15, 0.15], projection='3d')
        # self._add_orientation_inset(inset_ax, viewing_angle)

        if savefig:
            plt.savefig(f'{savefig}.pdf', bbox_inches='tight')
        else:
            plt.show()
    def _add_orientation_inset(self, ax, viewing_angle):
        """
        Adds an orientation inset to the plot.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.Axes3DSubplot
            3D axes for the inset.
        viewing_angle : list
            Viewing angles for elevation and azimuth.
        """
        x, y, z = [0, 1], [0, 1], [0, 1]
        
        ax.quiver(0, 0, 0, 1, 0, 0, color='r', arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 1, 0, color='g', arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 0, 1, color='b', arrow_length_ratio=0.1)

        ax.text(1, 0, 0, 'X', color='r')
        ax.text(0, 1, 0, 'Y', color='g')
        ax.text(0, 0, 1, 'Z', color='b')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)

        ax.view_init(elev=viewing_angle[0], azim=viewing_angle[-1])

if __name__ == "__main__":
    # Get the absolute path to the folder containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Set up argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('MO_input', type=str, help='Python module containing MO settings')
    args = parser.parse_args()
    
    # Dynamically import the module specified on the command line
    MO_input = __import__(args.MO_input)

    # Retrieve parameters from MO_input.py
    Molecule = MO_input.Molecule  
    MO_index = MO_input.MO_index 
    scale = MO_input.scale
    viewing_angle = MO_input.viewing_angle
    figsize = MO_input.figsize
    savefig = MO_input.savefig

    # Build the path to Converged_orbitals/<Molecule>.out based on the script directory
    out_file = os.path.join(script_dir, '..', 'Converged_orbitals', f'{Molecule}.out')

    if not os.path.exists(out_file):
        raise FileNotFoundError(f'Could not find output file: {out_file}')

    # Now create your MolecularOrbitals instance
    # (Assuming MODrawer is defined in the same script or imported)
    MolClass = MODrawer(out_file, 'ttm')

    print(f"The SOMO is MO{MolClass.somo_index}")

    # Finally, draw the MO
    MolClass.mo_drawer(
        MO_index,
        orbital_scale=scale,
        viewing_angle=viewing_angle,
        transparency=0.4,
        figsize=figsize,
        re_orientate=True,
        savefig=savefig
    )
