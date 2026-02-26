import numpy as np
import os
import re
import matplotlib.pyplot as plt
from itertools import combinations
from matplotlib.animation import FuncAnimation, writers
import argparse


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
        Calculates and returns the SOMO indices based on valence electrons in the output file.

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

    def extract_eigenvectors(self):# -> list:
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

    def extract_basis(self):# -> tuple[list, list]:
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

    def mo_dict(self):# -> dict:
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
    mo_drawer(key, orbital_scale=2, viewing_angle=[45, -90], figsize=(14, 10), transparency=0.6, bondthickness=2, savefig=False):
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

    def mo_drawer(self, key, orbital_scale=1, viewing_angle=[-90, 0,0], figsize=(18, 14), bondthickness=2, transparency=0.6, savefig=False):
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
            Size of the figure (default is (10, 8)).
        """
        phi = np.linspace(0, 2 * np.pi, 100)
        theta = np.linspace(0, np.pi, 100)
        phi, theta = np.meshgrid(phi, theta)

        #plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')
        if key == 'SOMO' or str(key) == 'SOMO':
            key = f'MO{self.somo_index}'
        elif key == 'HOMO' or str(key) == 'HOMO':
            key = f'MO{self.somo_index-1}'
        elif key == 'LUMO' or str(key) == 'LUMO':
            key = f'MO{self.somo_index+1}'        
        basis = self.mos.get(key)
        atom_centres = np.array(basis[:, :-1])
        atom_coefs = np.array(basis[:, -1])

        for centre, atom_coef in zip(atom_centres, atom_coefs):
            r = orbital_scale * np.abs(atom_coef) 

            x = r * np.sin(theta) * np.cos(phi) + centre[0]
            y = r * np.sin(theta) * np.sin(phi) + centre[1]
            z = r * np.cos(theta) + centre[2]

            # Plotting the positive lobe
            if atom_coef >= 0:
                ax.plot_surface(x, y, z, color='r', alpha=transparency)
                # ax.plot_surface(x2, y2, z2, color='b', alpha=transparency)  # Plotting the negative lobe
            else:
                ax.plot_surface(x, y, z, color='b', alpha=transparency)
                # ax.plot_surface(x2, y2, z2, color='r', alpha=transparency) # Plotting the negative lobe
        
        # Creating bonds
        list_coords = np.asarray(list(self.coordinates.values()))
        dist_pair = np.asarray(list(combinations(list_coords, 2)))

        # Tolerance using C-Cl bond in Bohr of 1.77 Angstroms
        bool_array = np.linalg.norm(dist_pair[:,1] - dist_pair[:,0], axis=1) < 3.35
        bond_array = dist_pair[bool_array]
        
        for bond in bond_array:
            # Bonds are in pair of coordinatesm so zip pairs the respeictve coord together which are use to plot
            x_coords, y_coords, z_coords = zip(*bond)
            
            ax.plot(x_coords, y_coords, z_coords, color='black', linewidth=bondthickness)

        # Drawing small dots or circles based on atom type
        for atom, coord in self.coordinates.items():
            if atom.startswith('Cl') or atom.startswith('CL'):
                ax.scatter(coord[0], coord[1], coord[2], color='green', s=50, edgecolor='green')  # Small green circle
            elif atom.startswith('C'):
                ax.scatter(coord[0], coord[1], coord[2], color='black', s=50)  # Small black dot
            elif atom.startswith('N'):
                ax.scatter(coord[0], coord[1], coord[2], color='blue', s=50)  # Small blue dot

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

        # ax.set_xlabel('x (Bohr)')
        # ax.set_ylabel('y (Bohr)')
        # ax.set_zlabel('z (Bohr)')
        ax.view_init(elev=viewing_angle[0], azim=viewing_angle[1], roll = viewing_angle[2])
        # ax.set_title(f'{self.name} ExROPPP {key} diagram')
        # ax.text2D(0.5, 0.95, f'{self.name} ExROPPP {key} diagram', transform=ax.transAxes, ha='center')


        # Adding the inset with the orientation
        # inset_ax = fig.add_axes([0.8, 0.8, 0.15, 0.15], projection='3d')
        # self._add_orientation_inset(inset_ax, viewing_angle)



        if savefig:
            plt.savefig(f'{savefig}.pdf', dpi=1000, bbox_inches='tight')
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
    MolClass = MODrawer(out_file, "ttm")

    print(f"The SOMO is MO{MolClass.somo_index}")

    # Finally, draw the MO
    MolClass.mo_drawer(
        MO_index,
        orbital_scale=scale,
        viewing_angle=viewing_angle,
        transparency=0.4,
        figsize=figsize,
        savefig=savefig
    )



