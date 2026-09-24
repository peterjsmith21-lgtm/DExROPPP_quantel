#!/usr/bin/env python3
"""
mo_bubble.py  –  Hückel-style MO bubble diagram from LCAO 2pz coefficients.

Usage
-----
    python mo_bubble.py <xyz_file> <mo_coeff_file> <mo_number>
                        [-o output.png] [--show-coeffs]

Arguments
---------
    xyz_file       Standard .xyz file (Angström coordinates)
    mo_coeff_file  MO coefficients .txt file (see format below)
    mo_number      1-indexed MO number to plot

Options
-------
    -o, --output FILE    Output filename  [default: <mol>_MO<n>.png]
    --show-coeffs        Annotate each atom with its coefficient value
    --scale FLOAT        Multiply all bubble radii by this factor (default 1.0)

Coefficient file format
-----------------------
    Energy line is optional.  Both forms are accepted:

    orbital number 1 energy -6.462          # with energy
    [-0.42 -0.37 ...]

    orbital number 1                        # without energy
    [-0.42 -0.37 ...]

Notes
-----
  * π-active elements (C, N, Cl) are included in the LCAO basis; all other
    atoms (H, etc.) are drawn as small grey markers.
  * Coefficients must be listed in the same order as π-active atoms appear
    in the .xyz file.
  * Bubble radius is proportional to |coefficient|; blue = positive,
    red = negative, open grey circle = nodal (|c| < 1 % of max).
  * Heteroatom bubbles carry a coloured edge: dark blue for N, dark green
    for Cl, so element identity is visible without text labels.
"""

import argparse
import sys
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless – safe on any system
import matplotlib.pyplot as plt


# ── Thresholds & appearance ───────────────────────────────────────────────────

# Elements that participate in the π LCAO basis
PI_ELEMENTS = {'C', 'N', 'Cl'}          # set  – for membership tests & bond lookup

# The Hückel code groups basis atoms by element rather than interleaving them
# in xyz-file order.  Coefficient vector layout: all C (xyz order), then all N
# (xyz order), then all Cl (xyz order).  This list must match that convention.
PI_ELEMENTS_ORDER = ['C', 'N', 'Cl']    # list – defines coefficient grouping

# Per-pair bond detection thresholds between π-active atoms (Å).
# C–Cl aromatic bonds are ~1.73 Å so need a wider window than C–C.
PI_BOND_THRESH = {
    frozenset(['C',  'C' ]): 1.65,
    frozenset(['C',  'N' ]): 1.55,
    frozenset(['C',  'Cl']): 1.85,
    frozenset(['N',  'N' ]): 1.55,
    frozenset(['N',  'Cl']): 1.85,
    frozenset(['Cl', 'Cl']): 2.20,
}
PI_BOND_DEFAULT = 1.65  # fallback for any unrecognised element pair

BOND_CH    = 1.15   # Å  –  π-atom → H bond threshold (C–H ~1.09, N–H ~1.01)
MAX_RADIUS = 0.6    # Å  –  bubble radius for a coefficient of exactly 1.0.
                    # Absolute scale: radius = |c| × MAX_RADIUS, so bubbles from
                    # different orbitals and different files are directly comparable.
NODE_R     = 0.08   # Å  –  radius of a near-zero (nodal) marker
NODE_FRAC  = 0.01   # coefficient < NODE_FRAC × max  →  treated as node

# ── Skeleton line widths (points) – change these to adjust bond thickness ─────
BOND_LW_CC = 1.5    # π–π bonds  (C–C, C–N, C–Cl, …)
BOND_LW_CH = 1.0    # C–H / N–H bonds

POS_COLOR  = '#4878D0'   # blue  –  positive lobe
NEG_COLOR  = '#EE6666'   # red   –  negative lobe
NODE_COLOR = '#888888'   # grey  –  nodal atom
BOND_COLOR = '#1a1a1a'
CH_COLOR   = '#bbbbbb'
H_COLOR    = '#aaaaaa'

# Bubble edge colour by element – signals atom identity without text labels
ATOM_EDGE_COLOR = {
    'C':  '#1a1a1a',   # near-black
    'N':  '#1a3a8a',   # dark blue
    'Cl': '#1a6b2a',   # dark green
}




# ══ Parsers ═══════════════════════════════════════════════════════════════════

def parse_xyz(path: str):
    """Return (mol_name, atoms, coords).

    atoms  : list[str]  element symbols, length N
    coords : ndarray (N, 3)  Cartesian coordinates in Å
    """
    with open(path) as fh:
        lines = fh.readlines()
    n_atoms  = int(lines[0].strip())
    mol_name = lines[1].strip()
    atoms, coords = [], []
    for line in lines[2 : 2 + n_atoms]:
        parts = line.split()
        atoms.append(parts[0])
        coords.append([float(v) for v in parts[1:4]])
    return mol_name, atoms, np.array(coords)


def parse_mo_file(path: str):
    """Return list of orbital dicts: {number, energy, coeffs}.

    Handles coefficients on one line or split across multiple lines,
    and energies in plain decimal or scientific notation.
    """
    with open(path) as fh:
        text = fh.read()

    # Each orbital block: header line (energy is optional) + bracketed array.
    # Matches both "orbital number 1 energy -6.46\n[...]"
    # and          "orbital number 1\n[...]"
    pattern = re.compile(
        r'orbital number\s+(\d+)'                    # MO number (group 1)
        r'(?:\s+energy\s+([\-+\d.eE]+))?'           # energy – optional (group 2)
        r'\s*\n'
        r'(\[[\s\S]*?\])',                            # coefficient array (group 3)
        re.MULTILINE
    )
    orbitals = []
    for m in pattern.finditer(text):
        number = int(m.group(1))
        energy = float(m.group(2)) if m.group(2) is not None else float('nan')
        raw    = m.group(3).strip()[1:-1]   # strip outer [ ]
        coeffs = np.array([float(x) for x in raw.split()])
        orbitals.append({'number': number, 'energy': energy, 'coeffs': coeffs})

    if not orbitals:
        sys.exit("Error: no orbitals found – check coefficient file format.")
    return orbitals


# ══ Geometry helpers ══════════════════════════════════════════════════════════

def inplane_axes(coords: np.ndarray):
    """Return the two axis indices that span the molecular plane.

    The third axis (lowest variance) is discarded – that is the
    out-of-plane direction for a flat molecule.
    """
    variance  = np.var(coords, axis=0)
    flat_axis = int(np.argmin(variance))
    return [i for i in range(3) if i != flat_axis]


def to_2d(coords: np.ndarray, axes):
    """Project (N,3) → (N,2) using the given axis pair."""
    return coords[:, axes]


# ══ Plotting ══════════════════════════════════════════════════════════════════

def plot_mo_bubble(
    mol_name    : str,
    atoms       : list,
    coords      : np.ndarray,
    orbitals    : list,
    mo_num      : int,
    out_path    : str   = None,
    show_coeffs : bool  = False,
    mo_type     : str   = None,
    scale       : float = 1.0,
):
    # ── Locate the requested orbital ─────────────────────────────────────────
    mo = next((o for o in orbitals if o['number'] == mo_num), None)
    if mo is None:
        sys.exit(f"Error: MO {mo_num} not found in coefficient file.")
    coeffs = mo['coeffs']

    # ── π-active atoms, grouped by element to match coefficient ordering ─────
    # The Hückel code outputs coefficients as: all C (xyz order) → all N → all Cl.
    # Using simple "a in PI_ELEMENTS" would interleave elements in xyz order,
    # misaligning coefficients for molecules containing heteroatoms.
    pi_idx = []
    for elem in PI_ELEMENTS_ORDER:
        pi_idx.extend([i for i, a in enumerate(atoms) if a == elem])
    if len(coeffs) != len(pi_idx):
        sys.exit(
            f"Error: {len(coeffs)} coefficient(s) but {len(pi_idx)} π-active "
            f"atom(s) ({', '.join(PI_ELEMENTS)}) in the .xyz file – these must match."
        )
    pi_coords = coords[pi_idx]

    h_idx    = [i for i, a in enumerate(atoms) if a not in PI_ELEMENTS]
    h_coords = coords[h_idx]

    # ── Project to 2D ────────────────────────────────────────────────────────
    axes  = inplane_axes(coords)
    c2d   = to_2d(pi_coords, axes)
    h2d   = to_2d(h_coords,  axes)
    all2d = to_2d(coords,    axes)

    # ── Bubble radius scale ──────────────────────────────────────────────────
    # Absolute base scale: radius = |c| × MAX_RADIUS × scale.
    # MAX_RADIUS is the radius for a coefficient of 1.0; scale shifts all
    # bubbles uniformly so diagrams from different systems can be compared.
    max_c     = max(np.max(np.abs(coeffs)), 1e-8)   # used for node threshold
    r_scale   = MAX_RADIUS * scale                   # absolute, then user-scaled
    threshold = NODE_FRAC * max_c

    # ── Figure – no frame, no ticks, no grid ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.axis('off')

    # ── C–H bonds ────────────────────────────────────────────────────────────
    for cxyz, hxyz in [(coords[ci], coords[hi])
                       for ci in pi_idx for hi in h_idx
                       if np.linalg.norm(coords[ci] - coords[hi]) < BOND_CH]:
        ax.plot(
            [cxyz[axes[0]], hxyz[axes[0]]],
            [cxyz[axes[1]], hxyz[axes[1]]],
            '-', color=CH_COLOR, lw=BOND_LW_CH, zorder=1
        )

    # ── π–π bonds (C–C, C–N, C–Cl, …) with per-pair distance threshold ───────
    for i in range(len(pi_idx)):
        for j in range(i + 1, len(pi_idx)):
            ei, ej   = atoms[pi_idx[i]], atoms[pi_idx[j]]
            thresh   = PI_BOND_THRESH.get(frozenset([ei, ej]), PI_BOND_DEFAULT)
            if np.linalg.norm(pi_coords[i] - pi_coords[j]) < thresh:
                ax.plot(
                    [c2d[i, 0], c2d[j, 0]],
                    [c2d[i, 1], c2d[j, 1]],
                    '-', color=BOND_COLOR, lw=BOND_LW_CC, zorder=2,
                    solid_capstyle='round'
                )

    # ── H atom markers ───────────────────────────────────────────────────────
    if len(h2d):
        ax.scatter(h2d[:, 0], h2d[:, 1],
                   s=22, color=H_COLOR, zorder=3, linewidths=0)

    # ── Bubbles ──────────────────────────────────────────────────────────────
    centroid = c2d.mean(axis=0)

    for i, (c, pos) in enumerate(zip(coeffs, c2d)):
        is_node = abs(c) <= threshold
        radius  = NODE_R if is_node else abs(c) * r_scale

        element    = atoms[pi_idx[i]]
        edge_color = ATOM_EDGE_COLOR.get(element, '#1a1a1a')

        if is_node:
            circ = plt.Circle(
                pos, NODE_R,
                facecolor='white', edgecolor=NODE_COLOR,
                linewidth=1.4, zorder=4
            )
        else:
            color = POS_COLOR if c > 0 else NEG_COLOR
            circ  = plt.Circle(
                pos, radius,
                facecolor=color, edgecolor='none',
                alpha=0.80, zorder=4
            )
        ax.add_patch(circ)

        # ── Optional coefficient label ────────────────────────────────────
        if show_coeffs:
            # Radially offset from the bubble centre, away from the centroid
            direction = pos - centroid
            norm = np.linalg.norm(direction)
            direction = direction / norm if norm > 1e-6 else np.array([0.0, 1.0])

            offset = radius + 0.22
            lx = pos[0] + direction[0] * offset
            ly = pos[1] + direction[1] * offset

            coeff_str = f'{c:+.3f}' if not is_node else '≈0'
            ax.text(lx, ly, coeff_str,
                    ha='center', va='center',
                    fontsize=7.5, color='#222222', zorder=5,
                    bbox=dict(boxstyle='round,pad=0.15',
                              fc='white', ec='none', alpha=0.75))

    # ── Axis limits with enough room for bubbles (and labels if shown) ────────
    pad = MAX_RADIUS + (0.7 if show_coeffs else 0.3)
    ax.set_xlim(all2d[:, 0].min() - pad, all2d[:, 0].max() + pad)
    ax.set_ylim(all2d[:, 1].min() - pad, all2d[:, 1].max() + pad)

    # ── Save ─────────────────────────────────────────────────────────────────
    if out_path is None:
        safe_name = mol_name.replace(' ', '_').replace('/', '-')
        type_part = f'_{mo_type}' if mo_type else ''
        out_path  = f'{safe_name}{type_part}_MO{mo_num}.png'
    plt.savefig(out_path, dpi=450, bbox_inches='tight', pad_inches=0.05, transparent=True)
    print(f'Saved → {out_path}')
    return out_path


# ══ Entry point ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Hückel-style MO bubble diagram from LCAO 2pz coefficients.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('xyz_file',        help='.xyz geometry file')
    parser.add_argument('mo_coeff_file',   help='MO coefficients .txt file')
    parser.add_argument('mo_number', type=int, help='MO to plot (1-indexed)')
    parser.add_argument('-o', '--output',  default=None,
                        help='Output filename [default: <mol>_MO<n>.png]')
    parser.add_argument('--show-coeffs',   action='store_true',
                        help='Annotate each atom with its coefficient value')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Scale all bubble radii by this factor (default 1.0)')

    args = parser.parse_args()

    # Derive MO type from coefficient filename: "Feff_orbs.txt" -> "Feff"
    import os
    coeff_basename = os.path.basename(args.mo_coeff_file)
    if coeff_basename.endswith('_orbs.txt'):
        mo_type = coeff_basename[: -len('_orbs.txt')]
    else:
        mo_type = coeff_basename.rsplit('.', 1)[0]   # fallback: strip extension

    mol_name, atoms, coords = parse_xyz(args.xyz_file)
    orbitals = parse_mo_file(args.mo_coeff_file)
    plot_mo_bubble(
        mol_name, atoms, coords, orbitals,
        args.mo_number, args.output, args.show_coeffs, mo_type, args.scale,
    )
