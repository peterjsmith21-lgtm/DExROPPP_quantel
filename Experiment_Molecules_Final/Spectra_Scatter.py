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
sizes  = [80,           80,             60            ]
col_idx  = [1,           2,             3            ]  # indices into each data row


def make_scatter(data, title, filename):
    fig, ax = plt.subplots(figsize=(6, 6))

    all_vals = []

    for method, marker, color, size, ci in zip(methods, markers, colors, sizes, col_idx):
        exp_vals, calc_vals = [], []
        for row in data:
            exp = row[0]        # experimental is first numeric column
            calc = row[ci]
            if exp is not None and calc is not None:
                exp_vals.append(exp)
                calc_vals.append(calc)
                all_vals.extend([exp, calc])

        ax.scatter(exp_vals, calc_vals,
                   marker=marker, color=color, s=size, linewidths=1.5,
                   label=method, zorder=5)


    # y = x line
    if all_vals:
        exp_min = min(exp_vals) - 20
        exp_max = max(exp_vals) + 20
        lim_min = min(all_vals) - 10
        lim_max = max(all_vals) + 10
        ax.plot([exp_min, exp_max], [lim_min, lim_max],
                'k--', linewidth=0.8, label='$y = x$')
        ax.set_xlim(exp_min, exp_max)
        ax.set_ylim(lim_min, lim_max)

    ax.set_xlabel('Experimental Peak Position / nm', fontsize=11)
    ax.set_ylabel('Computed Peak Position / nm', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()
    print(f"Saved {filename}")

def extract(data):
    return [(row[0], row[1], row[2], row[3]) for row in data]


def rmse(data, col):
    errors = []
    for row in data:
        exp, calc = row[0], row[col]
        if exp is not None and calc is not None:
            errors.append((calc - exp) ** 2)
    return np.sqrt(np.mean(errors))

make_scatter(extract(peak1), 'Peak 1', 'peak1_scatter.png')
make_scatter(extract(peak2), 'Peak 2', 'peak2_scatter.png')
make_scatter(extract(peak3), 'Peak 3', 'peak3_scatter.png')


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



print(f"{'Method':<15} {'Peak 1':>10} {'Peak 2':>10} {'Peak 3':>10}")
print("-" * 48)
for method, col in zip(methods, [1, 2, 3]):
    r1 = rmse(peak1, col)
    r2 = rmse(peak2, col)
    r3 = rmse(peak3, col)
    print(f"{method:<15} {r1:>10.1f} {r2:>10.1f} {r3:>10.1f}")


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
    (466, 451, 418, 0),  # DTA
    (449, 442, 441, 441),   # 5-7-ICzTTM2
    (459, 452, 441, 442),   # 5-8-ICzTTM2
    (439, 445, 448, 452),   # pseudo-o-PCP-TTM2
    (439, 399, 0, 452),  # pseudo-p-PCP-TTM2
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
    (376, 343, 0, 0), # PCz-(PyBTM)2
    (371, 340, None, None), # THDBA-(PyBTM)2
]

print(f"{'Penalised Method':<15} {'Peak 1':>10} {'Peak 2':>10} {'Peak 3':>10}")
print("-" * 48)
for method, col in zip(methods, [1, 2, 3]):
    r1 = rmse(peak1, col)
    r2 = rmse(peak2, col)
    r3 = rmse(peak3, col)
    print(f"{method:<15} {r1:>10.1f} {r2:>10.1f} {r3:>10.1f}")