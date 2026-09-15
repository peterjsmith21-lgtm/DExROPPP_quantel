from quantel.wfn.csf import CSF
from quantel.ints.fcidump_integrals import FCIDUMP
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('FCIDUMP', type = str, help = 'FCIDUMP file containing PPP integrals')
args = parser.parse_args()
fcidump = args.FCIDUMP

fcidump_ints = FCIDUMP(fcidump)
fcidump_ints.print()
nmo = fcidump_ints.nmo()


if __name__ == "__main__":
    
    print("===============================================")
    print(f" Testing CSF optimisation method")
    print("===============================================")

    # Bug is in the CSF class, not the fcidump_ints format.
    # Initialise CSF object for an high spin triplet state
    wfn = CSF(fcidump_ints, '++')
    wfn.get_orbital_guess(method="core", localise=False)
    

    # Setup optimiser
    for guess in ("gwh", "core"):
        print("\n************************************************")
        print(f" Testing '{guess}' initial guess method")
        print("************************************************")
        from quantel.opt.lbfgs import LBFGS
        wfn.get_orbital_guess(method=guess, localise=False)
        LBFGS().run(wfn)
        
        # Test canonicalisation 
        wfn.canonicalize()
        # Test Hessian index
        wfn.get_davidson_hessian_index(approx_hess=False)
        wfn.print(verbose=5)
        
        print(wfn.mo_coeff)
        outfile = f"{fcidump}_{guess}_mo_coeff.npy"
        np.save(outfile, wfn.mo_coeff)
        print(f" Saved MO coefficients to {outfile}")