import argparse
import sys
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('geometry', type=str, help='file containing geometry')
parser.add_argument('--rotation_matrix', type=str, default=None,
                    help='file containing rotation matrix (numpy .npy or text)')
parser.add_argument('--converged_orbs', type=str, default=None,
                    help='file containing converged orbitals (numpy .npy or text)')
args = parser.parse_args()
optimized_geometry = args.geometry

lit_params = [[-28.08,   1.66,  8,    1.328,  0.],
              [ -2.96, -23.53,  1.66, 12.34,  1.115],
              [-17.56, -22.16,  1.66, 16.76,  1.115],
              [-12.65, -27.1,   1.66,  8,     1.987]]

opt_params = [[-22.71707507,   1.70561621,  8.42083845,  1.17315691,  0.],
              [ -3.486745,   -25.23133814,  1.76801716, 12.80518166,  1.20074375],
              [-17.68133786, -24.73720244,  1.43363853, 17.97984271,  1.11179102],
              [-10.33567426, -26.02193733,  1.45186057,  9.64299129,  2.25331612]]

if args.rotation_matrix is not None:
    rotation_matrix = np.load(args.rotation_matrix)
else:
    rotation_matrix = None

if args.converged_orbs is not None:
    converged_orbs = np.load(args.converged_orbs)
else:
    converged_orbs = None


class FCIDUMPWritten(Exception):
    """Raised once the FCIDUMP message has been printed."""
    pass


class StopAfterFCIDUMP:
    """Wraps stdout and aborts the run right after the FCIDUMP message."""
    def __init__(self, stream):
        self.stream = stream

    def write(self, s):
        self.stream.write(s)
        if 'FCIDUMP file in AO basis written' in s:
            self.stream.write('\n')
            self.stream.flush()
            raise FCIDUMPWritten

    def flush(self):
        self.stream.flush()


if __name__ == '__main__':
    from Diradical_ExROPPP import rad_calc

    sys.stdout = StopAfterFCIDUMP(sys.__stdout__)
    try:
        rad_calc(file=optimized_geometry, params=opt_params,
                 rotation_matrix=rotation_matrix, converged_orbs=converged_orbs)
    except FCIDUMPWritten:
        pass
    finally:
        sys.stdout = sys.__stdout__