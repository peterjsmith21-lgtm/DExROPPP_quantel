import matplotlib.pyplot as plt
import matplotlib.colors as mc
import numpy as np
import matplotlib.patches as mpatches
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde

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
def truncate_cmap(cmap, lo=0.15, hi=0.85):
    return mc.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{lo:.2f},{hi:.2f})',
        cmap(np.linspace(lo, hi, 256)),
    )

cmap = truncate_cmap(plt.cm.RdBu_r, lo=0.15, hi=0.85)
norm = mc.BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5], ncolors=cmap.N)

# ── Helper: axis limits ───────────────────────────────────────────────────────
def axis_limits(*val_lists, pad=0.08):
    all_vals = [v for lst in val_lists for v in lst]
    return min(all_vals) - pad, max(all_vals) + pad

# ── Helper: RMSE / MAE ────────────────────────────────────────────────────────
def rmse(predicted, observed):
    return np.sqrt(np.mean((np.array(predicted) - np.array(observed)) ** 2))

def mae(predicted, observed):
    return np.mean(np.abs(np.array(predicted) - np.array(observed)))

# ── Helper: convex hull shading ───────────────────────────────────────────────
def draw_hull(ax, x_vals, y_vals, color, alpha=0.12, pad=0.04, label=None):
    points = np.column_stack([x_vals, y_vals])
    if len(points) < 3:
        return  # ConvexHull needs at least 3 points
    hull = ConvexHull(points)
    hull_pts = points[hull.vertices]
    centroid = hull_pts.mean(axis=0)
    # Expand hull outward from centroid for a small padding margin
    directions = hull_pts - centroid
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    expanded = hull_pts + pad * directions / norms
    polygon = mpatches.Polygon(
        expanded, closed=True,
        facecolor=color, alpha=alpha,
        edgecolor=color, linewidth=1.5, linestyle='--',
        label=label, zorder=2,
    )
    ax.add_patch(polygon)
    

def draw_smooth_hull(ax, x_vals, y_vals, color, alpha=0.15, label=None):
    # Fit KDE to the points
    points = np.vstack([x_vals, y_vals])
    kde = gaussian_kde(points, bw_method=0.6)  # increase bw_method for smoother/wider
    
    # Evaluate KDE on a fine grid
    x_grid = np.linspace(min(x_vals) - 0.1, max(x_vals) + 0.1, 200)
    y_grid = np.linspace(min(y_vals) - 0.1, max(y_vals) + 0.1, 200)
    xx, yy = np.meshgrid(x_grid, y_grid)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    
    # Draw filled contour at a low density threshold
    ax.contourf(xx, yy, zz, levels=[zz.max() * 0.05, zz.max()],
                colors=[color], alpha=alpha, zorder=2)
    ax.contour(xx, yy, zz, levels=[zz.max() * 0.05],
               colors=[color], linewidths=0, linestyles='--',
               alpha=0.6, zorder=2)
    
    # Dummy patch for legend
    if label:
        return mpatches.Patch(facecolor=color, alpha=alpha, label=label)    


# ── Build combined figure ─────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-paper')
fig, ax = plt.subplots(figsize=(7, 6))

# Unpack energies and S² for each group
e2 = [d[0] for d in DEXROPPP_peak2]
s2_peak2 = [d[1] for d in DEXROPPP_peak2]

e3 = [d[0] for d in DEXROPPP_peak3]
s2_peak3 = [d[1] for d in DEXROPPP_peak3]

# Axis limits across all data
lo, hi = axis_limits(experimental_peak2, e2, experimental_peak3, e3)
ref = np.linspace(lo, hi, 200)
ax.plot(ref, ref, 'k--', linewidth=1, alpha=0.4, zorder=1)

# Convex hull shading
'''
draw_hull(ax, experimental_peak2, e2, color='royalblue',  label='Peak 2 group')
draw_hull(ax, experimental_peak3, e3, color='darkorange', label='Peak 3 group')
'''

patch2 = draw_smooth_hull(ax, experimental_peak2, e2, color='royalblue',  label='Peak 2')
patch3 = draw_smooth_hull(ax, experimental_peak3, e3, color='darkorange', label='Peak 3')


# Scatter — different markers per group, shared S² colourmap
sc2 = ax.scatter(experimental_peak2, e2, c=s2_peak2, cmap=cmap, norm=norm,
                 s=70, marker='o', edgecolors='black', linewidths=0.5, zorder=5)
sc3 = ax.scatter(experimental_peak3, e3, c=s2_peak3, cmap=cmap, norm=norm,
                 s=70, marker='^', edgecolors='black', linewidths=0.5, zorder=5)

# Metrics annotation
r2  = rmse(e2, experimental_peak2);  mae2 = mae(e2, experimental_peak2)
r3  = rmse(e3, experimental_peak3);  mae3 = mae(e3, experimental_peak3)
ax.text(
    0.96, 0.04,
    (f'Peak 2 — RMSE={r2:.3f} eV  MAE={mae2:.3f} eV\n'
     f'Peak 3 — RMSE={r3:.3f} eV  MAE={mae3:.3f} eV\n'
     f'— y = x'),
    transform=ax.transAxes,
    va='bottom', ha='right', fontsize=9, linespacing=1.5,
    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8),
)

# Legend for groups (hull patches) + marker shapes
legend_handles = [
    patch2,
    patch3,
    mpatches.Patch(facecolor='royalblue',  alpha=0.4, label='Peak 2'),
    mpatches.Patch(facecolor='darkorange', alpha=0.4, label='Peak 3'),
]
ax.legend(handles=legend_handles, fontsize=10, loc='upper left')

ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_aspect('equal')
ax.tick_params(axis='both', labelsize=11)
ax.set_xlabel('Exp. Transition Energy (eV)', fontsize=11)
ax.set_ylabel('D-ExROPPP Transition Energy (eV)', fontsize=11)

# Colorbar
cbar = fig.colorbar(sc3, ax=ax, shrink=0.75, pad=0.02,
                    boundaries=[-0.5, 0.5, 1.5, 2.5], ticks=[0, 1, 2])
cbar.set_label(r'$\langle S^2 \rangle$', fontsize=11, rotation=0, labelpad=12)

plt.tight_layout()
plt.savefig('Peaks_combined_scatter.png', dpi=300, bbox_inches='tight')
plt.show()