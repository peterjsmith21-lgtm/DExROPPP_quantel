import matplotlib.pyplot as plt
import matplotlib.colors as mc
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
experimental = [
    2.049256198, 2.213928571, 2.167482517, 1.967936508,
    1.898621746, 2.182746479, 2.283241252, 1.579363057,
    1.853213752, 1.847690015, 1.934165367, 1.875642965,
    1.839465875, 1.470699881,
]

DEXROPPP = [
    (1.904454685, 0.0), (2.63787234,  0.0), (2.484569138, 0.0),
    (1.958609795, 2.0), (2.464811133, 2.0), (2.499596774, 0.0),
    (2.626694915, 0.0), (1.388353863, 2.0), (1.907384615, 2.0),
    (1.919195046, 2.0), (2.250090744, 2.0), (2.326078799, 2.0),
    (2.35256167,  2.0), (1.585421995, 2.0),
]

B3LYP = [
    (1.44330617,  1.15), (2.29168207,  1.11), (1.693715847, 1.09),
    (1.631315789, 1.09), (3.572910663, 1.88), (1.655273698, 1.02),
    (1.63994709,  1.03), (1.295506792, 0.42), (1.967936508, 1.30),
    (1.996457327, 1.24), (1.804657933, 1.07), (1.831314623, 1.07),
    (1.842199108, 1.00), (1.589487179, 0.65),
]

# ── Colour mapping ─────────────────────────────────────────────────────────────
# S² = 0  → pure singlet → blue
# S² = 2  → pure triplet → red
# S² ≈ 1  → mixed        → white
def truncate_cmap(cmap, lo=0.15, hi=0.85):
    return mc.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{lo:.2f},{hi:.2f})',
        cmap(np.linspace(lo, hi, 256)),
    )

cmap = truncate_cmap(plt.cm.RdBu_r, lo=0.15, hi=0.85)
s2_norm = mc.Normalize(vmin=0, vmax=2)


# ── Helper: axis limits with equal padding ────────────────────────────────────
def axis_limits(x_vals, y_vals, pad=0.08):
    lo = min(min(x_vals), min(y_vals)) - pad
    hi = max(max(x_vals), max(y_vals)) + pad
    return lo, hi


# ── Helper: RMSE ──────────────────────────────────────────────────────────────
def rmse(predicted, observed):
    return np.sqrt(np.mean((np.array(predicted) - np.array(observed)) ** 2))


# ── Plot function ─────────────────────────────────────────────────────────────
def make_scatter(ax, exp_vals, calc_data, method_label):
    energies = [d[0] for d in calc_data]
    s2_vals  = [d[1] for d in calc_data]

    lo, hi = axis_limits(exp_vals, energies)

    # y = x reference line
    ref = np.linspace(1.2, 3.7, 200)
    ax.plot(ref, ref, 'k--', linewidth=1, alpha=0.4, label='y = x', zorder=1)

    # Scatter with S²-based colouring
    sc = ax.scatter(
        exp_vals, energies,
        c=s2_vals, cmap=cmap, norm=s2_norm,
        s=70, edgecolors='black', linewidths=0.5, zorder=5,
    )

    # RMSE annotation
    r = rmse(energies, exp_vals)
    msd = np.mean(np.array(energies) - np.array(exp_vals))
    ax.text(
        0.96, 0.04,
        f'RMSE = {r:.3f} eV\nMSD  = {msd:.3f} eV\n— y = x',
        transform=ax.transAxes,
        va='bottom', ha='right', fontsize=11,
        linespacing=1.5,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8),
    )

    #ax.set_xlim(lo, hi)
    #ax.set_ylim(lo, hi)
    ax.set_xlim(1.2, 3.7)
    ax.set_ylim(1.2, 3.7)
    ax.tick_params(axis='both', labelsize=11)
    ax.set_aspect('equal')
    ax.set_xlabel('Exp. Transition Energy (eV)', fontsize=11)
    #ax.set_ylabel(f'{method_label} Transition Energy (eV)', fontsize=11)

    return sc


# ── Build figure ──────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-paper')
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sc1 = make_scatter(axes[0], experimental, DEXROPPP, 'D-ExROPPP')
sc2 = make_scatter(axes[1], experimental, B3LYP,  'SF-TD-DFT/B3LYP')

# Shared colorbar on the right
cbar = fig.colorbar(sc2, ax=axes.tolist(), shrink=0.75, pad=0.02)
cbar.set_label(r'$\langle S^2 \rangle$', fontsize=11, rotation=0, labelpad=12)
cbar.set_ticks([0, 0.5, 1.0, 1.5, 2.0])

plt.savefig('absorption_scatter.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()