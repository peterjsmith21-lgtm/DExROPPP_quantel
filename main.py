import argparse
parser = argparse.ArgumentParser()
parser.add_argument('geometry', type = str, help = 'file containing geometry')
args = parser.parse_args()
optimized_geometry = args.geometry


lit_params = [[-28.08,   1.66 ,  8  , 1.328 ,  0.        ],
 [ -2.96 ,  -23.53 ,  1.66,  12.34  , 1.115],
 [-17.56, -22.16 ,  1.66 , 16.76 ,  1.115],
 [-12.65, -27.1 ,  1.66 ,  8 , 1.987]]


opt_params = [[-22.71707507,   1.70561621 ,  8.42083845  , 1.17315691 ,  0.        ],
 [ -3.486745 ,  -25.23133814 ,  1.76801716,  12.80518166  , 1.20074375],
 [-17.68133786, -24.73720244 ,  1.43363853 , 17.97984271 ,  1.11179102],
 [-10.33567426, -26.02193733 ,  1.45186057 ,  9.64299129 ,  2.25331612]]


if __name__ == '__main__':
    from Diradical_ExROPPP import rad_calc
    from Diradical_ExROPPP import write_gnu as gnu_Exroppp

    # For doing individual ExROPPP calculations on one monoradical
    strngs,ci_energies_array, osc_arrays, s2_array  = rad_calc(file=optimized_geometry, params = opt_params)
    #Get spectrum plot for triplet
    filename = optimized_geometry + 'Triplet_Ref'
    gnu_Exroppp(strngs[0], filename)
    #Get spectrum plot for singlet
    filename = optimized_geometry + 'Singlet_Ref'
    gnu_Exroppp(strngs[1], filename)
