import matplotlib.pyplot as plt
import numpy as np

DEXROPPP = [-0.083, -0.087, -0.028, -0.073, -0.085,
               0.011, 0.015, -0.087, -0.087, -0.039, 0.024, -0.050]
experimental = [ -0.065, -0.014, -0.0005, 0, 0, 0, 0,
                -0.044, -0.031, -0.002, 0.0007, -0.0015]

est_b3lyp = [
    -0.0068,    # TTM-Ph-TTM (para)    (Singlet)
    -0.07,      # DTA                  (Singlet)
     0.025,     # 5-7-ICz-TTM2         (Triplet)
    -0.0046,    # 5-8-ICz-TTM2         (Singlet)
    -0.00478,   # 5-11-ICz-TTM2        (Singlet)
     0.000647,  # pseudo-o-PCP-TTM2    (Triplet)
    -0.000982,  # pseudo-p-PCP-TTM2    (Singlet)
    -0.013,     # PyBTM-Ph-PyBTM       (Singlet)
    -0.0016,    # PyBTM-Hex2-PyBTM     (Singlet)
    -3.34,      # PCz-(PyBTM')2        (Singlet)
     0.00633,   # THDBA-(PyBTM')2      (Triplet)
    -0.009,     # TPA(Me)-(PyBTM'')2   (Singlet)
]

est_cam = [
    -0.0028,    # TTM-Ph-TTM (para)    (Singlet)
    -0.0283,    # DTA                  (Singlet)
     0.0198,    # 5-7-ICz-TTM2         (Triplet)
     0.0027,    # 5-8-ICz-TTM2         (Triplet)
    -0.00143,   # 5-11-ICz-TTM2        (Singlet)
     0.000118,  # pseudo-o-PCP-TTM2    (Triplet)
    -0.035,     # pseudo-p-PCP-TTM2    (Singlet)
    -0.0052,    # PyBTM-Ph-PyBTM       (Singlet)
    -0.0006,    # PyBTM-Hex2-PyBTM     (Singlet)
    -3.68,      # PCz-(PyBTM')2        (Singlet)
     0.00333,   # THDBA-(PyBTM')2      (Triplet)
    -0.0017,    # TPA(Me)-(PyBTM'')2   (Singlet)
]

est_pbe0 = [
    -0.00582,   # TTM-Ph-TTM (para)    (Singlet)
    -0.0594,    # DTA                  (Singlet)
     0.0272,    # 5-7-ICz-TTM2         (Triplet)
    -0.004432,  # 5-8-ICz-TTM2         (Singlet)
    -0.0035,    # 5-11-ICz-TTM2        (Singlet)
     0.000585,  # pseudo-o-PCP-TTM2    (Triplet)
    -0.033013,  # pseudo-p-PCP-TTM2    (Singlet)
    -0.0109,    # PyBTM-Ph-PyBTM       (Singlet)
    -0.0013,    # PyBTM-Hex2-PyBTM     (Singlet)
    -3.42,      # PCz-(PyBTM')2        (Singlet)
     0.00698,   # THDBA-(PyBTM')2      (Triplet)
    -0.00696,   # TPA(Me)-(PyBTM'')2   (Singlet)
]

functionals = [
    (est_b3lyp, 'B3LYP',     'steelblue', 'x'),
    (est_cam,   'CAM-B3LYP', 'firebrick',  '.'),
    (est_pbe0,  'PBE0',      'seagreen',   '*'),
]



# Filter out N/A
pairs = [(t, e) for t, e in zip(DEXROPPP, experimental) if e is not None]
t_vals, e_vals = zip(*pairs)
r2 = np.corrcoef(e_vals, t_vals)[0, 1] ** 2
rmse_dex = np.sqrt(np.mean((np.array(t_vals) - np.array(e_vals)) ** 2))
mse_signed = np.mean(np.array(t_vals) - np.array(e_vals))
print(f"D-ExROPPP RMSE: {rmse_dex:.4f} eV")
print(f"D-ExROPPP MSE (signed): {mse_signed:.4f} eV")


plt.style.use('seaborn-v0_8-paper')
fig, ax = plt.subplots()


x_min, x_max = -0.1, 0.04
y_min, y_max = -0.1, 0.04

ax.fill_between([x_min, 0], [y_min, y_min], [0, 0],         color='steelblue',    alpha=0.05)
ax.fill_between([0, x_max], [0, 0],         [y_max, y_max], color='firebrick',     alpha=0.05)
ax.fill_between([0, x_max], [y_min, y_min], [0, 0],         color='mediumpurple',  alpha=0.05)
ax.fill_between([x_min, 0], [0, 0],         [y_max, y_max], color='goldenrod',     alpha=0.05)


# Line of best fit — swap polyfit order
m, b = np.polyfit(e_vals, t_vals, 1)
x_fit = np.linspace(x_min, x_max, 200)

ax.axhline(0, color='gray', linewidth=0.8)
ax.axvline(0, color='gray', linewidth=0.8)


ax.plot(x_fit, x_fit, 'k--', linewidth=1, alpha=0.5, label='Exact agreement')

# DEXROPPP scatter — swap x, y
ax.scatter(e_vals, t_vals, marker='+', color='black', s=80,
           linewidths=1.5, zorder=5, label=f'D-ExROPPP (R²={r2:.2f})')

# Functional scatter — swap x, y
for est_list, label, color, marker in functionals:
    pairs_f = [(e, t) for e, t in zip(e_vals, est_list) if e is not None]
    e_f, t_f = zip(*pairs_f)
    r2_f = np.corrcoef(e_f, t_f)[0, 1] ** 2
    ax.scatter(e_f, t_f, marker=marker, color=color, s=40,
               linewidths=1.5, zorder=5, label=f'{label} (R²={r2_f:.2f})')
    
   
for calc_vals, label, color, marker in functionals:
    pairs_f = [(c, e) for c, e in zip(calc_vals, experimental) if e is not None]
    c_arr, e_arr_f = np.array([p[0] for p in pairs_f]), np.array([p[1] for p in pairs_f])
    mse_signed = np.mean(np.array(c_arr) - np.array(e_arr_f))
    mae_f  = np.mean(np.abs(c_arr - e_arr_f))
    rmse_f = np.sqrt(np.mean((c_arr - e_arr_f) ** 2))
    r2_f   = np.corrcoef(e_arr_f, c_arr)[0, 1] ** 2
    
    print(f"{label:<12}  MSE (signed)={mse_signed:.4f} eV  RMSE={rmse_f:.4f} eV  r²={r2_f:.3f}")


# Swap axis labels
ax.set_xlabel("Experimental $\Delta E_{ST}$ (eV)")
ax.set_ylabel("Calculated $\Delta E_{ST}$ (eV)")

# Swap the two "wrong" quadrant labels (top-left and bottom-right swap meaning)
ax.text(-0.025, -0.07, 'Singlet,\npredicted Singlet', ha='center', va='center', fontsize=10, fontstyle='italic')
ax.text( 0.02,  0.03,  'Triplet,\npredicted Triplet', ha='center', va='center', fontsize=10, fontstyle='italic')
ax.text( 0.02, -0.07,  'Triplet,\npredicted Singlet', ha='center', va='center', fontsize=10, fontstyle='italic')  # swapped
ax.text(-0.025,  0.03, 'Singlet,\npredicted Triplet', ha='center', va='center', fontsize=10, fontstyle='italic')  # swapped



ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig("EST_Scatter.png", dpi=400, transparent=True)
plt.show()
