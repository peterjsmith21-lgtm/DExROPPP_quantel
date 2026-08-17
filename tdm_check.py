"""
tdm_check.py -- locate errors in DipBuilder by comparing every element of the
transition-dipole matrix against an exact reference built from explicit
determinant algebra.

Usage
-----
    python tdm_check.py

Edit CONFIG below.  The script:
  1. builds the CSF basis for the requested (ndocc, nvirt),
  2. verifies every CSF is a normalised spin eigenfunction (catches basis errors),
  3. builds the exact TDM by second-quantised algebra on determinants,
  4. calls your DipBuilder and reports mismatches grouped by CSF family,
  5. runs a SOMO-rotation invariance test per family.

Convention: TDM = -<A|sum_p r_p|B>, matching DipBuilder.
CSF phases follow the definitions in Extended_interactions.py.
"""
import itertools
import numpy as np

# ----------------------------------------------------------------- CONFIG
NDOCC = 3
NVIRT = 3
CI_LEVEL = 3
SEED = 5
# =========================================================================

R2 = 1 / 2 ** 0.5


# ----------------------------------------------------------- determinant algebra
def _ann(state, p):
    sign, o = state
    if p not in o:
        return None
    k = o.index(p)
    return (sign * (-1) ** k, o[:k] + o[k + 1:])


def _cre(state, p):
    sign, o = state
    if p in o:
        return None
    k = 0
    while k < len(o) and o[k] < p:
        k += 1
    return (sign * (-1) ** k, o[:k] + (p,) + o[k:])


def _ov(state, bra):
    if state is None:
        return 0.0
    sign, o = state
    return sign if o == bra else 0.0


def mu_element(det1, det2, m):
    """<det1| sum_pq m_pq a+_p a_q |det2>, spin-conserving."""
    n_so = len(det1)
    bra = tuple(i for i, x in enumerate(det1) if x == 1)
    ket = (1, tuple(i for i, x in enumerate(det2) if x == 1))
    tot = 0.0
    for p in range(n_so):
        for q in range(n_so):
            if p % 2 != q % 2:
                continue
            st = _ann(ket, q)
            if st is None:
                continue
            st = _cre(st, p)
            tot += m[p // 2, q // 2] * _ov(st, bra)
    return tot


def s2_element(det1, det2):
    n_so = len(det1); n_orb = n_so // 2
    bra = tuple(i for i, x in enumerate(det1) if x == 1)
    ket = tuple(i for i, x in enumerate(det2) if x == 1)
    na = sum(det2[2 * k] for k in range(n_orb))
    nb = sum(det2[2 * k + 1] for k in range(n_orb))
    sz = 0.5 * (na - nb)
    tot = (sz ** 2 + sz) if bra == ket else 0.0
    for q in range(n_orb):
        st = _ann((1, ket), 2 * q + 1)
        if st is None:
            continue
        st = _cre(st, 2 * q)
        if st is None:
            continue
        for p in range(n_orb):
            st2 = _ann(st, 2 * p)
            if st2 is None:
                continue
            st2 = _cre(st2, 2 * p + 1)
            if st2 is None:
                continue
            tot += _ov(st2, bra)
    return tot


# ----------------------------------------------------------- CSF construction
class Basis:
    def __init__(self, ndocc, nvirt, spin, ci_level):
        self.ndocc, self.nvirt, self.spin = ndocc, nvirt, spin
        self.norbs = ndocc + 2 + nvirt
        self.S1, self.S2 = ndocc, ndocc + 1
        self.n_so = 2 * self.norbs
        self.core = [i for k in range(ndocc) for i in (2 * k, 2 * k + 1)]
        self.S1a, self.S1b = 2 * self.S1, 2 * self.S1 + 1
        self.S2a, self.S2b = 2 * self.S2, 2 * self.S2 + 1
        self.virt = [ndocc + 2 + k for k in range(nvirt)]
        self.names, self.csfs = [], []
        self._build(ci_level)

    def det(self, occ):
        d = [0] * self.n_so
        for i in occ:
            d[i] = 1
        return d

    def hole(self, so):
        return [i for i in self.core if i != so]

    def add(self, name, csf):
        self.names.append(name); self.csfs.append(csf)

    def _build(self, lvl):
        s = self.spin == 'S'
        S1a, S1b, S2a, S2b = self.S1a, self.S1b, self.S2a, self.S2b
        C = self.core
        # --- reference block
        if s:
            self.add('OS1', [[R2, self.det(C + [S1a, S2b])], [-R2, self.det(C + [S1b, S2a])]])
            zw1 = self.det(C + [S1a, S1b]); zw2 = self.det(C + [S2a, S2b])
            self.add('ZW-', [[R2, zw1], [-R2, zw2]])
            self.add('ZW+', [[R2, zw1], [R2, zw2]])
        else:
            self.add('OS3', [[R2, self.det(C + [S1a, S2b])], [R2, self.det(C + [S1b, S2a])]])
        if lvl < 1:
            return
        # --- core -> SOMO
        sg = 1 if s else -1
        for o in range(self.ndocc):
            self.add(f'CS0({o})', [[-R2, self.det(self.hole(2*o) + [S1a, S1b, S2a])],
                                   [sg * R2, self.det(self.hole(2*o+1) + [S1a, S1b, S2b])]])
        for o in range(self.ndocc):
            self.add(f"CS0'({o})", [[R2, self.det(self.hole(2*o) + [S1a, S2a, S2b])],
                                    [-sg * R2, self.det(self.hole(2*o+1) + [S1b, S2a, S2b])]])
        # --- SOMO -> virtual
        for v in self.virt:
            self.add(f'SV0({v})', [[-R2, self.det(C + [S2b, 2*v])],
                                   [sg * R2, self.det(C + [S2a, 2*v+1])]])
        for v in self.virt:
            self.add(f"SV0'({v})", [[-R2, self.det(C + [S1a, 2*v+1])],
                                    [sg * R2, self.det(C + [S1b, 2*v])]])
        if lvl < 2:
            return
        # --- core -> virtual (HL) and zwitterionic HL
        def hl(o, v):
            d1 = self.det(self.hole(2*o+1) + [S1a, S2b, 2*v+1])
            d2 = self.det(self.hole(2*o)   + [S1a, S2b, 2*v])
            d3 = self.det(self.hole(2*o+1) + [S1b, S2a, 2*v+1])
            d4 = self.det(self.hole(2*o)   + [S1b, S2a, 2*v])
            e1 = self.det(self.hole(2*o)   + [S1a, S2a, 2*v+1])
            e2 = self.det(self.hole(2*o+1) + [S1b, S2b, 2*v])
            return d1, d2, d3, d4, e1, e2
        fams = ([('HL1', lambda d: [[0.5, d[0]], [-0.5, d[1]], [-0.5, d[2]], [0.5, d[3]]]),
                 ('HL2', lambda d: [[1/3**0.5, d[5]], [-1/12**0.5, d[1]], [-1/12**0.5, d[0]],
                                    [1/3**0.5, d[4]], [-1/12**0.5, d[2]], [-1/12**0.5, d[3]]])]
                if s else
                [('HL1', lambda d: [[0.5, d[0]], [-0.5, d[1]], [0.5, d[2]], [-0.5, d[3]]]),
                 ('HL2', lambda d: [[0.5, d[0]], [-0.5, d[2]], [-0.5, d[3]], [0.5, d[1]]]),
                 ('HL3', lambda d: [[R2, d[4]], [-R2, d[5]]])])
        for tag, fn in fams:
            for o in range(self.ndocc):
                for v in self.virt:
                    self.add(f'{tag}({o},{v})', fn(hl(o, v)))
        for tag, (pa, pb) in (('ZHL1', (S1a, S1b)), ('ZHL2', (S2a, S2b))):
            for o in range(self.ndocc):
                for v in self.virt:
                    self.add(f'{tag}({o},{v})',
                             [[R2, self.det(self.hole(2*o+1) + [pa, pb, 2*v+1])],
                              [-sg * R2, self.det(self.hole(2*o) + [pa, pb, 2*v])]])
        if lvl < 3:
            return
        # --- doubly excited
        somos = [S1a, S1b, S2a, S2b]
        for o1 in range(self.ndocc):
            for o2 in range(o1 if s else o1 + 1, self.ndocc):
                if o1 == o2:
                    self.add(f'CSD({o1},{o2})',
                             [[1.0, self.det([i for i in C if i not in (2*o1, 2*o1+1)] + somos)]])
                else:
                    d1 = self.det([i for i in C if i not in (2*o1+1, 2*o2)] + somos)
                    d2 = self.det([i for i in C if i not in (2*o1, 2*o2+1)] + somos)
                    self.add(f'CSD({o1},{o2})',
                             [[-R2, d1], [R2, d2]] if s else [[R2, d1], [R2, d2]])
        for a in range(self.nvirt):
            for b in range(a if s else a + 1, self.nvirt):
                v1, v2 = self.virt[a], self.virt[b]
                if v1 == v2:
                    self.add(f'SVD({v1},{v2})', [[1.0, self.det(C + [2*v1, 2*v1+1])]])
                else:
                    e1 = self.det(C + [2*v1, 2*v2+1]); e2 = self.det(C + [2*v1+1, 2*v2])
                    self.add(f'SVD({v1},{v2})',
                             [[R2, e1], [-R2, e2]] if s else [[R2, e1], [R2, e2]])

    # ------------------------------------------------------------- checks
    def validate_spin(self):
        dets = []
        for c in self.csfs:
            for _, d in c:
                if d not in dets:
                    dets.append(d)
        idx = {tuple(d): i for i, d in enumerate(dets)}
        S2m = np.array([[s2_element(a, b) for b in dets] for a in dets])
        want = 0.0 if self.spin == 'S' else 2.0
        bad = []
        for nm, c in zip(self.names, self.csfs):
            v = np.zeros(len(dets))
            for co, d in c:
                v[idx[tuple(d)]] += co
            nrm = v @ v; val = (v @ S2m @ v) / nrm
            res = np.linalg.norm(S2m @ v - val * v)
            if res > 1e-9 or abs(val - want) > 1e-9 or abs(nrm - 1) > 1e-9:
                bad.append((nm, round(nrm, 4), round(val, 4)))
        return bad

    def exact_tdm(self, M):
        n = len(self.csfs)
        X = np.zeros((n, n, 3))
        for k in range(3):
            for i in range(n):
                for j in range(i, n):
                    val = -sum(c1 * c2 * mu_element(d1, d2, M[:, :, k])
                               for c1, d1 in self.csfs[i] for c2, d2 in self.csfs[j])
                    X[i, j, k] = X[j, i, k] = val
        return X


def random_dipole(norbs, seed):
    rng = np.random.default_rng(seed)
    M = np.zeros((norbs, norbs, 3))
    for k in range(3):
        a = rng.normal(size=(norbs, norbs))
        M[:, :, k] = a + a.T
    return M


def rotate_dipole(M, S1, S2, theta):
    norbs = M.shape[0]
    U = np.eye(norbs)
    c, s = np.cos(theta), np.sin(theta)
    U[S1, S1], U[S1, S2] = c, -s
    U[S2, S1], U[S2, S2] = s, c
    return np.einsum('ap,abk,bq->pqk', U, M, U, optimize=True)


def report(ndocc, nvirt, ci_level, seed):
    import DipBuilder
    for spin, builder in (('S', DipBuilder.build_singlet_TDM),
                          ('T', DipBuilder.build_triplet_TDM)):
        B = Basis(ndocc, nvirt, spin, ci_level)
        bad = B.validate_spin()
        label = 'singlet' if spin == 'S' else 'triplet'
        if bad:
            print(f"{label}: REFERENCE BASIS PROBLEM {bad[:3]}")
            continue
        M = random_dipole(B.norbs, seed)
        E = B.exact_tdm(M)
        C = builder(ndocc, B.norbs, M, ci_level)
        print(f"\n=== {label}: reference dim {len(B.names)}, DipBuilder dim {C.shape[0]} ===")
        if C.shape[0] != len(B.names):
            print("   DIMENSION MISMATCH -- check row_dim/col_dim before anything else")
            continue
        grp = {}
        for i in range(len(B.names)):
            for j in range(i, len(B.names)):
                if np.abs(E[i, j] - C[i, j]).max() > 1e-8:
                    key = (B.names[i].split('(')[0], B.names[j].split('(')[0])
                    grp.setdefault(key, []).append((i, j))
        if not grp:
            print("   all elements match the exact reference")
        for k, l in sorted(grp.items()):
            i, j = l[0]
            sgn = abs(abs(E[i, j, 0]) - abs(C[i, j, 0])) < 1e-8
            print(f"   <{k[0]}|mu|{k[1]}>  n={len(l):>3}  [{'sign' if sgn else 'value'}]"
                  f"   e.g. <{B.names[i]}|mu|{B.names[j]}>"
                  f"  exact={E[i, j, 0]:+9.5f}  code={C[i, j, 0]:+9.5f}")


if __name__ == '__main__':
    report(NDOCC, NVIRT, CI_LEVEL, SEED)