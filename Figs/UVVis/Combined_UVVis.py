#!/usr/bin/env python3
"""
uv_vis_plot.py
==============
Plot a UV-Vis spectrum by overlaying:
  - Experimental data  : two-column .txt  (wavelength nm | intensity)
  - Calculated spectrum: sum of Lorentzians expressed in gnuplot format

The gnuplot Lorentzian form is:
    amp * 1/(1 + ((centre - x) / (0.5 * gamma * centre * x))**2)

This is mathematically equivalent to a standard Lorentzian in wavenumber
space with HWHM = gamma/2 (units: inverse-nm = 1e7 cm⁻¹), mapped
analytically into wavelength space so peak positions are exact.

Usage
-----
    python uv_vis_plot.py <experimental.txt> <script.gp>

The gnuplot script must end with a plot line of the form:
    p <lorentzian_sum_expression>
where 'p' is the gnuplot shorthand for 'plot'.
All other lines in the gnuplot file are ignored.
Adjust WL_MIN / WL_MAX and the display options in SETTINGS below.
"""

import re
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
#  SETTINGS  –  edit these
#  (the two input files are passed on the command line, not set here)
# ══════════════════════════════════════════════════════════════════════════════

# Wavelength range and resolution for evaluating the calculated spectrum
WL_MIN  = 200    # nm
WL_MAX  = 350    # nm
WL_NPTS = 2000

# ── Display options ───────────────────────────────────────────────────────────
# NORMALISE = True  → both curves scaled to [0, 1] on a single y-axis
# NORMALISE = False → twin y-axes (left = experiment, right = calculation)
NORMALISE = True

# Draw vertical stick lines at each transition wavelength
SHOW_STICKS = False

# Save the figure (set to None to skip saving)
SAVE_TO = "uvvis_spectrum.png"

# Axis labels (change if your units differ)
XLABEL     = "Wavelength (nm)"
EXP_LABEL  = "Experiment"
CALC_LABEL = "Calculation"
EXP_YLABEL  = "Absorbance"           # used on left axis when NORMALISE = False
CALC_YLABEL = "Oscillator strength"  # used on right axis when NORMALISE = False


# ══════════════════════════════════════════════════════════════════════════════
#  GNUPLOT FILE READER
# ══════════════════════════════════════════════════════════════════════════════

def read_gnuplot_file(path: Path) -> str:
    """
    Read a gnuplot script and return the Lorentzian sum string.

    The function scans the file bottom-up for the first line that starts
    with 'p ' (the gnuplot shorthand for 'plot') and returns everything
    after that two-character prefix.  Backslash line-continuations that
    gnuplot uses to wrap long plot commands across multiple lines are
    joined into a single string before returning.
    """
    lines = path.read_text().splitlines()

    # Walk backwards to find the plot line
    plot_line_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].lstrip().startswith("p "):
            plot_line_idx = i
            break

    if plot_line_idx is None:
        raise ValueError(
            f"No line starting with 'p ' found in '{path}'.\n"
            "The gnuplot script must contain a plot command of the form:\n"
            "    p <lorentzian_expression>"
        )

    # Collect the plot line and any continuation lines (ending with '\')
    collected = []
    for line in lines[plot_line_idx:]:
        stripped = line.rstrip()
        if stripped.endswith("\\"):
            collected.append(stripped[:-1])   # drop the backslash
        else:
            collected.append(stripped)
            break                              # last continuation line reached

    full_line = "".join(collected).lstrip()

    # Strip the leading 'p ' (or 'plot ') command keyword
    for prefix in ("plot ", "p "):
        if full_line.startswith(prefix):
            return full_line[len(prefix):]

    return full_line   # fallback (shouldn't be reached)


# ══════════════════════════════════════════════════════════════════════════════
#  GNUPLOT PARSER
# ══════════════════════════════════════════════════════════════════════════════

# One floating-point number, optionally signed, optionally in sci-notation
_NUM = r'[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?'

# Pattern for one Lorentzian term
_TERM = re.compile(
    rf'({_NUM})'              # group 1 – amplitude
    r'\*1/\(1\+\(\('
    rf'({_NUM})'              # group 2 – centre (numerator)
    r'-x\)/\(0\.5\*'
    rf'({_NUM})'              # group 3 – gamma
    r'\*'
    rf'({_NUM})'              # group 4 – centre (denominator, must equal grp 2)
    r'\*x\)\)\*\*2\)'
)


def parse_gnuplot(gp_string: str) -> list:
    """
    Extract (amplitude, centre_nm, gamma) from a gnuplot Lorentzian sum.

    Returns only terms where:
      - amplitude  != 0
      - centre is a finite positive real number

    Tokens like '0inf' that fail to parse as a plain float are silently
    skipped (their amplitude is always 0 in well-formed gnuplot output).
    """
    # Collapse whitespace and backslash line-continuations into one long string
    s = re.sub(r'[\s\\]+', '', gp_string)

    terms = []
    for m in _TERM.finditer(s):
        amp_str    = m.group(1)
        centre_str = m.group(2)
        gamma_str  = m.group(3)

        # Skip terms whose centre token contains non-numeric characters
        # (e.g. '0inf')
        if not re.fullmatch(_NUM, centre_str):
            print(f"  ⚠  Ignoring term: centre token '{centre_str}' "
                  f"is not a plain number.")
            continue

        amp    = float(amp_str)
        centre = float(centre_str)
        gamma  = float(gamma_str)

        if amp == 0.0:
            continue
        if not np.isfinite(centre) or centre <= 0.0:
            print(f"  ⚠  Ignoring term: centre = {centre:.4g} "
                  f"(must be finite and positive).")
            continue

        terms.append((amp, centre, gamma))

    return terms


# ══════════════════════════════════════════════════════════════════════════════
#  SPECTRUM EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════

def lorentzian_sum(wl: np.ndarray, terms: list) -> np.ndarray:
    """
    Evaluate the sum of Lorentzians at wavelengths `wl` (nm).

    The form
        amp / (1 + ((centre - x) / (0.5 * gamma * centre * x))**2)
    simplifies – after substituting wavenumbers ν = 1/λ – to a standard
    Lorentzian in ν-space with HWHM = gamma/2 (nm⁻¹ = 1e7 cm⁻¹).
    """
    y = np.zeros_like(wl, dtype=float)
    for amp, centre, gamma in terms:
        hwhm = 0.5 * gamma * abs(centre) * wl   # wavelength-dependent HWHM
        y   += amp / (1.0 + ((centre - wl) / hwhm) ** 2)
    return y


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_experimental(path: Path):
    """
    Load two-column experimental data from a text file.

    Handles whitespace-separated, comma-separated, and mixed formats
    (including trailing commas) by splitting each data line on any
    combination of commas and whitespace and taking the first two tokens.
    Lines that are blank or start with a comment character are skipped.
    """
    comment_chars = ('#', '!', '%', '@')
    wl, intensity = [], []

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(comment_chars):
            continue
        tokens = [t for t in re.split(r'[,\s]+', line) if t]
        if len(tokens) < 2:
            continue
        try:
            wl.append(float(tokens[0]))
            intensity.append(float(tokens[1]))
        except ValueError:
            continue   # skip header/label lines that can't be converted

    if not wl:
        raise ValueError(f"No numeric data could be read from '{path}'.")

    return np.array(wl), np.array(intensity)


def normalise(y: np.ndarray) -> np.ndarray:
    lo, hi = y.min(), y.max()
    if hi == lo:
        return np.zeros_like(y)
    return (y - lo) / (hi - lo)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():

    # ── Command-line arguments ────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Plot experimental and calculated UV-Vis spectra."
    )
    parser.add_argument(
        "experimental",
        metavar="experimental.txt",
        help="Two-column whitespace-separated file: wavelength (nm) | intensity",
    )
    parser.add_argument(
        "gnuplot_script",
        metavar="script.gp",
        help="Gnuplot script whose final 'p ...' line contains the Lorentzian sum",
    )
    args = parser.parse_args()

    # ── Load experimental data ────────────────────────────────────────────────
    p = Path(args.experimental)
    if not p.exists():
        sys.exit(f"Error: experimental file '{p}' not found.")

    exp_wl, exp_int = load_experimental(p)
    print(f"Loaded {len(exp_wl):,} experimental points from '{p.name}'")

    # ── Read gnuplot file and parse the Lorentzian sum ───────────────────────
    gp_path = Path(args.gnuplot_script)
    if not gp_path.exists():
        sys.exit(f"Error: gnuplot script '{gp_path}' not found.")

    print(f"Reading gnuplot script from '{gp_path.name}' …")
    gp_string = read_gnuplot_file(gp_path)

    print("Parsing gnuplot string …")
    terms = parse_gnuplot(gp_string)

    if not terms:
        print("  ⚠  No non-zero Lorentzian terms found – "
              "the calculated curve will be flat.\n"
              "     Check that GNUPLOT_STRING is set correctly.")
    else:
        print(f"  Found {len(terms)} active Lorentzian term(s):")
        for i, (a, c, g) in enumerate(terms, 1):
            fwhm_approx = g * c ** 2   # FWHM in nm at the peak (λ ≈ centre)
            print(f"    [{i}]  amp={a:+.4f}   centre={c:.3f} nm"
                  f"   gamma={g:.4e} nm⁻¹   (FWHM≈{fwhm_approx:.1f} nm @ peak)")

    x_calc = np.linspace(WL_MIN, WL_MAX, WL_NPTS)
    y_calc = lorentzian_sum(x_calc, terms)

    # ── Build figure ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    COLOR_EXP  = "#2c7bb6"   # steel blue
    COLOR_CALC = "#d7191c"   # red-orange
    COLOR_STCK = "#fdae61"   # pale orange for sticks

    if NORMALISE:
        # ── Single y-axis, both curves on [0, 1] ─────────────────────────────
        y_exp_plot  = normalise(exp_int)
        y_calc_plot = normalise(y_calc) if y_calc.max() != 0 else y_calc

        # Sticks (draw first so they sit behind the broadened curve)
        if SHOW_STICKS and terms:
            amps = np.array([a for a, c, g in terms])
            cens = np.array([c for a, c, g in terms])
            amps_n = amps / np.abs(amps).max()   # normalise stick heights
            ax.vlines(cens, 0, amps_n,
                      color=COLOR_STCK, lw=1.5, alpha=0.8,
                      label="Transitions (sticks)")

        ax.plot(x_calc,  y_calc_plot, lw=2.0, color=COLOR_CALC,
                label=CALC_LABEL, zorder=3)
        ax.plot(exp_wl,  y_exp_plot,  lw=1.5, color=COLOR_EXP,
                label=EXP_LABEL, zorder=4)

        ax.set_ylabel("Normalised intensity", fontsize=12)
        ax.legend(fontsize=11, framealpha=0.9)

    else:
        # ── Twin y-axes for different absolute scales ─────────────────────────
        ax2 = ax.twinx()

        # Sticks on ax2 (calculated side)
        if SHOW_STICKS and terms:
            amps = np.array([a for a, c, g in terms])
            cens = np.array([c for a, c, g in terms])
            ax2.vlines(cens, 0, amps,
                       color=COLOR_STCK, lw=1.5, alpha=0.8,
                       label="Transitions (sticks)")

        ax.plot(exp_wl,  exp_int,  lw=1.5, color=COLOR_EXP,
                label=EXP_LABEL)
        ax2.plot(x_calc, y_calc,   lw=2.0, color=COLOR_CALC,
                 ls="--", label=CALC_LABEL)

        ax.set_ylabel(EXP_YLABEL,  color=COLOR_EXP,  fontsize=12)
        ax2.set_ylabel(CALC_YLABEL, color=COLOR_CALC, fontsize=12)
        ax.tick_params(axis="y", colors=COLOR_EXP)
        ax2.tick_params(axis="y", colors=COLOR_CALC)

        # Merge legends from both axes
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=11, framealpha=0.9)

    ax.set_xlabel(XLABEL, fontsize=12)
    ax.set_xlim(WL_MIN, WL_MAX)
    ax.set_ylim(bottom=0)
    #ax.grid(True, alpha=0.25, ls=":")

    plt.tight_layout()

    if SAVE_TO:
        plt.savefig(SAVE_TO, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved to '{SAVE_TO}'")

    plt.show()


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()