
import matplotlib.pyplot as plt
import numpy as np


# (experimental, exroppp, tddft_bs, tddft_t)
# None = Absent or N/A

peak1 = [
    (595, 651, 653, 547),   # TTM-TTM
    (557, 471, 617, 522),   # TTM-Ph-TTM (para)
    (572, 499, 497, 495),   # TTM-Ph-TTM (meta)
    (676, 591, 542, 532),   # DTA
    (647, 516, 641, 621),   # 5-7-ICzTTM2
    (662, 541, 655, 639),   # 5-8-ICzTTM2
    (669, 512, 669, 648),   # 5-11-ICzTTM2
    (595, 602, 576, 588),   # pseudo-o-PCP-TTM2
    (594, 616, 590, 588),   # pseudo-p-PCP-TTM2
    (569, 487, 614, 539),   # PyBTM-Ph-PyBTM
    (549, 482, 499, 509),   # PyBTM-Hex-PyBTM
    (665, 497, 602, 663),   # PCz-(PyBTM)2
]

peak2 = [
    (426, 450, 485, 452),   # TTM-TTM
    (430, 385, 405, 423),   # TTM-Ph-TTM (para)
    (402, 372, None, 404),  # TTM-Ph-TTM (meta)
    (466, 451, 418, None),  # DTA
    (449, 442, 441, 441),   # 5-7-ICzTTM2
    (459, 452, 441, 442),   # 5-8-ICzTTM2
    (439, 445, 448, 452),   # pseudo-o-PCP-TTM2
    (439, 399, None, 452),  # pseudo-p-PCP-TTM2
    (414, 382, 410, 425),   # PyBTM-Ph-PyBTM
    (479, 443, 430, 435),   # PCz-(PyBTM)2
    (433, 460, None, None), # THDBA-(PyBTM)2
]

peak3 = [
    (373, 320, 414, 374),   # TTM-TTM
    (369, 344, 373, 374),   # TTM-Ph-TTM (para)
    (378, 346, None, 374),  # TTM-Ph-TTM (meta)
    (391, 358, 375, 402),   # DTA
    (388, 331, 376, 376),   # 5-7-ICzTTM2
    (379, 332, 389, 363),   # 5-8-ICzTTM2
    (381, 344, 375, 375),   # 5-11-ICzTTM2
    (371, 360, 391, 374),   # pseudo-o-PCP-TTM2
    (374, 369, 401, 375),   # pseudo-p-PCP-TTM2
    (371, 343, 391, 356),   # PyBTM-Ph-PyBTM
    (371, 349, 371, 381),   # PyBTM-Hex-PyBTM
    (376, 343, None, None), # PCz-(PyBTM)2
    (371, 340, None, None), # THDBA-(PyBTM)2
]



methods = ['D-ExROPPP', 'TD-DFT - Broken Symmetry', 'TD-DFT - Triplet']
markers  = ['.',         '+',           'x'          ]
colors   = ['steelblue', 'firebrick',   'goldenrod'  ]
sizes  = [100,           100,             80            ]
col_idx  = [1,           2,             3            ]  # indices into each data row

def extract(data):
    return [(row[0], row[1], row[2], row[3]) for row in data]

def make_scatter(ax, data, title):
    all_exp, all_calc = [], []
    handles = []

    for method, marker, color, size, ci in zip(methods, markers, colors, sizes, col_idx):
        exp_vals, calc_vals = [], []
        for row in data:
            exp = row[0]
            calc = row[ci]
            if exp is not None and calc is not None:
                exp_vals.append(exp)
                calc_vals.append(calc)
                all_exp.extend([exp])
                all_calc.extend([calc])

        sc = ax.scatter(exp_vals, calc_vals,
                        marker=marker, color=color, s=size, linewidths=1.5,
                        label=method, zorder=5)
        handles.append(sc)

    if all_exp:
        x_min = min(all_exp) - 10
        x_max = max(all_exp) + 10
        y_min = min(all_calc) - 10
        y_max = max(all_calc) + 10

        # y = x line clipped to whichever range is smaller
        line_min = max(x_min, y_min)
        line_max = min(x_max, y_max)
        yx, = ax.plot([line_min, line_max], [line_min, line_max],
                      'k--', linewidth=0.8, label='$y = x$')
        handles.append(yx)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel('Experimental Peak Position / nm', fontsize=12)
    ax.set_ylabel('Computed Peak Position / nm', fontsize=12)

    return handles


fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, data, title in zip(axes, [peak3, peak2, peak1], ['Peak 3', 'Peak 2', 'Peak 1']):
    handles = make_scatter(ax, extract(data), title)

fig.legend(handles=handles,
           labels=methods + ['$y = x$'],
           loc='lower center',
           ncol=4,
           fontsize=15,
           bbox_to_anchor=(0.5, -0.08))

plt.tight_layout()
plt.savefig('all_peaks_scatter.png', dpi=200, bbox_inches='tight', transparent=True)
plt.show()