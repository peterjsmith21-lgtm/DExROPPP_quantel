import matplotlib.pyplot as plt
import matplotlib.colors as mc
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
experimental_peak2 = [
    2.883255814,
    2.966028708,
    2.876566125,
    2.588308977,
    2.99468599,
    2.811337868,
    2.837070938,
    2.761247216,
    2.701089325,
    2.660515021,
]

DEXROPPP_peak2 = [
    (3.22025974, 2.0),
    (3.332795699, 2.0),
    (2.695217391, 2.0),
    (2.798645598, 2.0),
    (3.350810811, 2.0),
    (2.786067416, 2.0),
    (3.115075377, 2.0),
    (2.804977376, 2.0),
    (2.761247216, 2.0),
    (2.749002217, 2.0),

]

experimental_peak3 = [
    3.32386059,
    3.359891599,
    3.245549738,
    3.341778976,
    3.297340426,
    3.341778976,
    3.341778976,
    3.359891599,
    3.306133333,
    3.195360825,
    3.271240106,
    3.254068241,
    3.17084399,
]

DEXROPPP_peak3 = [
    (3.874375, 0.0),
    (3.55243553, 1.0),
    (3.583236994, 1.0),
    (3.700895522, 1.0),
    (3.614577259, 1.0),
    (3.625146199, 1.0),
    (3.604069767, 2.0),
    (3.350810811, 1.0),
    (3.254068241, 1.0),
    (3.689880952, 2.0),
    (3.635777126, 2.0),
    (3.604069767, 2.0),
    (3.463128492, 1.0),
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
bounds = [0, 1, 2]
norm = mc.BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5], ncolors=cmap.N)


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
    ref = np.linspace(lo, hi, 200)
    ax.plot(ref, ref, 'k--', linewidth=1, alpha=0.4, label='y = x', zorder=1)

    # Scatter with S²-based colouring
    sc = ax.scatter(
        exp_vals, energies,
        c=s2_vals, cmap=cmap, norm=norm,
        s=70, edgecolors='black', linewidths=0.5, zorder=5,
    )

    # RMSE annotation
    r = rmse(energies, exp_vals)
    mae = np.mean(np.abs(np.array(energies) - np.array(exp_vals)))
    ax.text(
        0.96, 0.04,
        f'RMSE = {r:.3f} eV\nMAE  = {mae:.3f} eV\n— y = x',
        transform=ax.transAxes,
        va='bottom', ha='right', fontsize=11,
        linespacing=1.5,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8),
    )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.tick_params(axis='both', labelsize=11)
    ax.set_aspect('equal')
    ax.set_xlabel('Exp. Transition Energy (eV)', fontsize=11)
    ax.set_ylabel(f'{method_label} Transition Energy (eV)', fontsize=11)

    return sc


# ── Build figure ──────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-paper')
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sc1 = make_scatter(axes[0], experimental_peak2, DEXROPPP_peak2, 'D-ExROPPP')
sc2 = make_scatter(axes[1], experimental_peak3, DEXROPPP_peak3, 'D-ExROPPP')

# Shared colorbar on the right
cbar = fig.colorbar(sc2, ax=axes.tolist(), shrink=0.75, pad=0.02,
                    boundaries=[-0.5, 0.5, 1.5, 2.5], ticks=[0, 1, 2])
cbar.set_label(r'$\langle S^2 \rangle$', fontsize=11, rotation=0, labelpad=12)

plt.savefig('Peaks_scatter.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()