#!/usr/bin/env python3
"""
Stage-1 "Benchmark" of the paper's Section 5: validate the nonlinear-Maxwell /
Heisenberg-Euler solver against the analytic vacuum-birefringence indices.

Produces two figures:
  fig_benchmark_indices  - measured Yee indices vs the exact constitutive index
                           and the leading-order Adler targets, over a range of
                           backgrounds eps = xi B0^2.
  fig_benchmark_coeffs   - the per-eps coefficients (n-1)/eps converging to the
                           Adler {4, 7, 3} as eps -> 0, with the 7:4 ratio.

These are the numeric-vs-analytic comparison plots the referee asked for, at the
reduced-dimensional (1D) benchmark level.
"""

import numpy as np
import _style as S
import matplotlib.pyplot as plt
import yee_solver as ys


def main():
    S.apply_style()

    # ---- sweep over background strength -------------------------------------
    eps_grid = np.array([2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2])
    n_perp_meas = np.empty_like(eps_grid)
    n_para_meas = np.empty_like(eps_grid)
    for i, e in enumerate(eps_grid):
        B0 = np.sqrt(e)
        n_perp_meas[i] = ys.measure_index("perp", 1.0, B0)
        n_para_meas[i] = ys.measure_index("para", 1.0, B0)

    # dense analytic curves
    e_dense = np.linspace(eps_grid.min() * 0.5, eps_grid.max() * 1.05, 400)
    npe = np.array([ys.exact_indices(1.0, np.sqrt(e))[0] for e in e_dense])
    npa = np.array([ys.exact_indices(1.0, np.sqrt(e))[1] for e in e_dense])

    # ---- Figure 1: indices vs eps ------------------------------------------
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    ax.plot(e_dense, npe - 1, color=S.C_PERP, label=r"exact $n_\perp-1$")
    ax.plot(e_dense, npa - 1, color=S.C_PARA, label=r"exact $n_\parallel-1$")
    ax.plot(e_dense, 4 * e_dense, "--", color=S.C_PERP, lw=1.0, alpha=0.7,
            label=r"Adler LO $4\,\xi B_0^2$")
    ax.plot(e_dense, 7 * e_dense, "--", color=S.C_PARA, lw=1.0, alpha=0.7,
            label=r"Adler LO $7\,\xi B_0^2$")
    ax.plot(eps_grid, n_perp_meas - 1, "o", color=S.C_PERP, ms=6,
            mec="white", mew=0.8, label="Yee solver (perp)", zorder=5)
    ax.plot(eps_grid, n_para_meas - 1, "s", color=S.C_PARA, ms=6,
            mec="white", mew=0.8, label="Yee solver (para)", zorder=5)
    ax.set_xlabel(r"background $\varepsilon = \xi B_0^2$")
    ax.set_ylabel(r"refractive-index shift  $n-1$")
    ax.set_title("Stage-1 benchmark: vacuum birefringence indices")
    ax.legend(ncol=2, loc="upper left")
    S.save(fig, "fig_benchmark_indices")
    plt.close(fig)

    # ---- Figure 2: coefficient convergence + ratio --------------------------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.4, 4.0))

    cp = (n_perp_meas - 1) / eps_grid
    ca = (n_para_meas - 1) / eps_grid
    cd = (n_para_meas - n_perp_meas) / eps_grid

    axL.axhline(4, color=S.C_PERP, ls=":", lw=1)
    axL.axhline(7, color=S.C_PARA, ls=":", lw=1)
    axL.axhline(3, color=S.C_DELTA, ls=":", lw=1)
    axL.plot(eps_grid, cp, "o-", color=S.C_PERP, ms=5, label=r"$(n_\perp-1)/\varepsilon$")
    axL.plot(eps_grid, ca, "s-", color=S.C_PARA, ms=5, label=r"$(n_\parallel-1)/\varepsilon$")
    axL.plot(eps_grid, cd, "^-", color=S.C_DELTA, ms=5, label=r"$\Delta n/\varepsilon$")
    axL.set_xscale("log")
    axL.set_xlabel(r"$\varepsilon = \xi B_0^2$")
    axL.set_ylabel(r"recovered coefficient")
    axL.set_title(r"convergence to Adler $\{4,7,3\}$")
    axL.legend(loc="center left")
    axL.text(eps_grid[0], 4.0, "  4", va="bottom", color=S.C_PERP, fontsize=9)
    axL.text(eps_grid[0], 7.0, "  7", va="bottom", color=S.C_PARA, fontsize=9)
    axL.text(eps_grid[0], 3.0, "  3", va="bottom", color=S.C_DELTA, fontsize=9)

    ratio = ca / cp
    axR.axhline(7 / 4, color=S.C_NEUTRAL, ls=":", lw=1, label="Adler 7:4 = 1.75")
    axR.plot(eps_grid, ratio, "D-", color=S.C_ACCENT, ms=5,
             label=r"$n_\parallel\!-\!1 : n_\perp\!-\!1$")
    axR.set_xscale("log")
    axR.set_xlabel(r"$\varepsilon = \xi B_0^2$")
    axR.set_ylabel("birefringence ratio")
    axR.set_title("the 7:4 ratio")
    axR.set_ylim(1.55, 1.80)
    axR.legend(loc="lower left")

    S.save(fig, "fig_benchmark_coeffs")
    plt.close(fig)

    # ---- numeric summary to stdout -----------------------------------------
    fit = ys.coefficient_sweep([2e-4, 5e-4, 1e-3, 2e-3])
    print("\nStage-1 benchmark summary")
    print(f"  exact-index match at eps=1e-2:  perp/para within "
          f"{ys.run_birefringence(B0=0.1)['n_perp_exact_err']*100:.3f}%")
    print(f"  extrapolated Adler coefficients: "
          f"perp={fit['coeff_perp']:.3f}  para={fit['coeff_para']:.3f}  "
          f"delta={fit['coeff_delta']:.3f}  (target 4 / 7 / 3)")


if __name__ == "__main__":
    main()
