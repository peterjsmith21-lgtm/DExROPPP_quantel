import argparse
parser = argparse.ArgumentParser()
parser.add_argument('geometry', type = str, help = 'file containing geometry')
args = parser.parse_args()
optimized_geometry = args.geometry


params = [[-22.71707507,   1.70561621 ,  8.42083845  , 1.17315691 ,  0.        ],
 [ -3.486745 ,  -25.23133814 ,  1.76801716,  12.80518166  , 1.20074375],
 [-17.68133786, -24.73720244 ,  1.43363853 , 17.97984271 ,  1.11179102],
 [-10.33567426, -26.02193733 ,  1.45186057 ,  9.64299129 ,  2.25331612]]

if __name__ == '__main__':
    from Diradical_ExROPPP import rad_calc
    from ExROPPP import write_gnu as gnu_Exroppp

    # For doing individual ExROPPP calculations on one monoradical
    strng,ci_energies_array, osc_array, s2_array  = rad_calc(file=optimized_geometry, params = params)
    gnu_Exroppp(strng, optimized_geometry)
