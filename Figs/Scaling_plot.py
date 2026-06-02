import numpy as np
import matplotlib.pyplot as plt
from math import comb


def rasci_hphp_equal(m_tot):
    """RASCI(h,p,hp) det count for 3 equal RAS subspaces, m_tot total orbitals.
       RAS1 doubly occupied, RAS2 half-filled, RAS3 empty."""
    m   = m_tot // 3
    k   = m // 2
    ref       = comb(m, k) ** 2
    h_p_sect  = 4 * m * comb(m, k) * comb(m, k + 1)   # h + p (4 spin variants)
    hp_sect   = 2 * m**2 * ref                        # hp (2 spin variants)
    return ref + h_p_sect + hp_sect


# Active space sizes (n electrons in n orbitals)
n_tot = list(range(6, 48, 6))
n_docc = [(n - 2) // 2 for n in n_tot] # N total = N_occ * 2 + 2
# N electrons = N total = N_docc * 2 + 2
n_dets_casscf = [comb(n, n//2) * comb(n, n//2) for n in n_tot]
n_dets_xcisd = [20 * (n ** 4) + 12 * (n ** 3) + 14 * (n **2) + 8*(n) + 4 for n in n_docc]
n_dets_xcis_d = [8 * (n **2) + 8*(n) + 4 for n in n_docc]
n_dets_xcis = [6 * (n**2) + 8*(n) + 4 for n in n_docc]
n_dets_rasci = [rasci_hphp_equal(n) for n in n_tot]

n_dets_xcisd[0] = 220

N_ref = np.linspace(min(n_tot), max(n_tot), 100)

# Anchor each curve at the smallest N so they start near the data, then scale up
N0 = n_tot[0]

# Anchor values chosen to place each curve in a useful part of the plot.
# Tweak these constants to slide the reference lines up/down.
refs = [
    (r'$\sim N^2$',  10  * (N_ref / N0) ** 2,  'gray'),
    (r'$\sim N^4$',  10  * (N_ref / N0) ** 4,  'gray'),
    (r'$\sim N^6$',  10  * (N_ref / N0) ** 6,  'gray'),
    (r'$\sim e^N$',  10  * np.exp(N_ref - N0), 'gray'),
]

scaling_labels = [
    (n_dets_casscf,  r'$\sim 16^{N}/N$',     'darkred'),
    (n_dets_rasci,   r'$\sim N^{3}\,2.52^{N}$', 'darkorange'),
    (n_dets_xcisd,   r'$\sim N^{4}$',         'darkblue'),
    (n_dets_xcis_d,  r'$\sim N^{2}$',         'darkgreen'),
]


fig, ax = plt.subplots(figsize=(7, 5))

ax.semilogy(n_tot, n_dets_casscf, 'o-', color='darkred', linewidth=2,
            markersize=7, markerfacecolor='white', markeredgewidth=2,
            label='CASSCF (N, N)')

ax.semilogy(n_tot, n_dets_rasci, 'd-', color='darkorange', linewidth=2,
            markersize=7, markerfacecolor='white', markeredgewidth=2,
            label=r'RASCI $(\frac{N}{3},\frac{N}{3},\frac{N}{3})$ (h,p,hp)')

ax.semilogy(n_tot, n_dets_xcisd, 's-', color='darkblue', linewidth=2,
            markersize=7, markerfacecolor='white', markeredgewidth=2,
            label='XCISD')

ax.semilogy(n_tot, n_dets_xcis_d, 's-', color='darkgreen', linewidth=2,
            markersize=7, markerfacecolor='white', markeredgewidth=2,
            label='XCIS-D')

ax.semilogy(n_tot, n_dets_xcis, '^-', color='darkgreen', linewidth=2,
            markersize=7, markerfacecolor='white', markeredgewidth=2,
            label='XCIS')
'''
for series, label, color in scaling_labels:
    ax.annotate(label,
                xy=(n_tot[-1], series[-1]),
                xytext=(5, 0), textcoords='offset points',
                color=color, fontsize=10, va='center')
'''

ax.set_xlabel(r'Total number of $\pi$-orbitals (N)', fontsize=11)
ax.set_ylabel('Dimension of CI Matrix', fontsize=11)
ax.set_xlim(min(n_tot) - .5, max(n_tot) + .5)
ax.set_xticks(n_tot)
ax.set_xticklabels([f'{n}' for n in n_tot])
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('Scaling_plot.png', dpi=300, bbox_inches='tight')
